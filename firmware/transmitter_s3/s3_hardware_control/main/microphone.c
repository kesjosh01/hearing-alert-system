#include "microphone.h"

#include "driver/i2s_std.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

enum { DMA_FRAMES = 160, DMA_BUFFERS = 8, READY_DEPTH = 2, HOP_COUNT = READY_DEPTH + 2 };
_Static_assert(HA_HOP_SAMPLES % DMA_FRAMES == 0, "DMA blocks must fill each hop exactly");

static i2s_chan_handle_t rx;
static QueueHandle_t available, ready;
static uint32_t overflows; // ISR increments, capture reads atomically.
static const char *TAG = "ha_mic";

static bool IRAM_ATTR on_overflow(i2s_chan_handle_t channel, i2s_event_data_t *event, void *ctx) {
    __atomic_fetch_add(&overflows, 1, __ATOMIC_RELAXED);
    return false;
}

ha_audio_hop_t *ha_mic_take(void) {
    ha_audio_hop_t *hop;
    xQueueReceive(ready, &hop, portMAX_DELAY);
    return hop;
}

void ha_mic_release(ha_audio_hop_t *hop) {
    xQueueSend(available, &hop, portMAX_DELAY);
}

static void capture(void *arg) {
    ha_audio_hop_t *hop;
    uint32_t seq = 0, seen_overflows = 0;
    int32_t raw[DMA_FRAMES];
    size_t filled = 0;
    unsigned silent_hops = 0;
    uint16_t sample_bits = 0;
    bool gap = true;
    unsigned discard = 30; // Allow 300 ms of clocks for INMP441 power-up.
    xQueueReceive(available, &hop, portMAX_DELAY);
    for (;;) {
        size_t bytes = 0;
        esp_err_t err = i2s_channel_read(rx, raw, sizeof(raw), &bytes, 1000);
        uint32_t count = __atomic_load_n(&overflows, __ATOMIC_RELAXED);
        if (err != ESP_OK || bytes != sizeof(raw) || count != seen_overflows) {
            ESP_LOGW(TAG, "audio gap: read=%s bytes=%u dma_overflows=%lu",
                     esp_err_to_name(err), (unsigned)bytes, (unsigned long)count);
            seen_overflows = count;
            silent_hops = 0;
            sample_bits = 0;
            filled = 0;
            gap = true;
            discard = DMA_BUFFERS + 1; // Flush all buffered data spanning the loss.
            if (err != ESP_OK) vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }
        if (discard) {
            --discard;
            continue;
        }
        for (unsigned i = 0; i < DMA_FRAMES; ++i) {
            int16_t sample = (int16_t)(raw[i] >> 16);
            hop->samples[filled++] = sample;
            sample_bits |= (uint16_t)sample;
        }
        if (filled != HA_HOP_SAMPLES) continue;

        silent_hops = sample_bits ? 0 : silent_hops + 1;
        if (silent_hops % 60 == 4)
            ESP_LOGW(TAG, "digital silence: check INMP441 power, SD and L/R slot selection");
        sample_bits = 0;
        hop->end_ms = esp_timer_get_time() / 1000;
        hop->gap = gap;
        hop->seq = seq++;
        if (xQueueSend(ready, &hop, 0) != pdTRUE) {
            ha_audio_hop_t *old;
            if (xQueueReceive(ready, &old, 0) == pdTRUE) ha_mic_release(old);
            // Only this task adds ready hops. A missing seq resets the detector.
            xQueueSend(ready, &hop, portMAX_DELAY);
        }
        xQueueReceive(available, &hop, portMAX_DELAY);
        filled = 0;
        gap = false;
    }
}

esp_err_t ha_mic_start(void) {
    if (rx) return ESP_ERR_INVALID_STATE;
    esp_err_t err = ESP_ERR_NO_MEM;
    bool enabled = false;
    ha_audio_hop_t *pool = heap_caps_calloc(HOP_COUNT, sizeof(*pool), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    available = xQueueCreate(HOP_COUNT, sizeof(ha_audio_hop_t *));
    ready = xQueueCreate(READY_DEPTH, sizeof(ha_audio_hop_t *));
    if (!pool || !ready || !available) goto fail;
    for (size_t i = 0; i < HOP_COUNT; ++i) ha_mic_release(&pool[i]);

    i2s_chan_config_t channel = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    channel.dma_frame_num = DMA_FRAMES;
    channel.dma_desc_num = DMA_BUFFERS;
    err = i2s_new_channel(&channel, NULL, &rx);
    if (err != ESP_OK) goto fail;
    i2s_std_config_t cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {.mclk = I2S_GPIO_UNUSED, .bclk = GPIO_NUM_4, .ws = GPIO_NUM_5,
                     .dout = I2S_GPIO_UNUSED, .din = GPIO_NUM_6},
    };
#if CONFIG_HA_MIC_RIGHT_SLOT
    cfg.slot_cfg.slot_mask = I2S_STD_SLOT_RIGHT;
#else
    cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
#endif
    err = i2s_channel_init_std_mode(rx, &cfg);
    if (err != ESP_OK) goto fail;
    i2s_event_callbacks_t callbacks = {.on_recv_q_ovf = on_overflow};
    err = i2s_channel_register_event_callback(rx, &callbacks, NULL);
    if (err != ESP_OK) goto fail;
    err = i2s_channel_enable(rx);
    if (err != ESP_OK) goto fail;
    enabled = true;
    if (xTaskCreatePinnedToCore(capture, "ha_mic", 4096, NULL, 8, NULL, 1) != pdPASS) {
        err = ESP_ERR_NO_MEM;
        goto fail;
    }
    ESP_LOGI(TAG, "INMP441 16 kHz, BCLK=4 WS=5 DIN=6 slot=%s",
             cfg.slot_cfg.slot_mask == I2S_STD_SLOT_LEFT ? "left (L/R low)" : "right (L/R high)");
    return ESP_OK;
fail:
    if (enabled) i2s_channel_disable(rx);
    if (rx) i2s_del_channel(rx);
    if (ready) vQueueDelete(ready);
    if (available) vQueueDelete(available);
    heap_caps_free(pool);
    rx = NULL;
    ready = NULL;
    available = NULL;
    return err;
}
