#include "ha_haptic.h"

#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

static const char *TAG = "ha_haptic";

#define CMD_STOP (-1)

static int s_gpio = -1;
static bool s_active_high = true;
static QueueHandle_t s_cmd;
static uint32_t s_max_play_ms;

static void motor(bool on) {
    if (s_gpio >= 0) gpio_set_level(s_gpio, on == s_active_high ? 1 : 0);
}

esp_err_t ha_haptic_init(int gpio, bool active_high) {
    s_gpio = gpio;
    s_active_high = active_high;
    if (gpio < 0) return ESP_ERR_NOT_SUPPORTED;
    // Set the "off" level before switching the pin to output: no kick at boot.
    gpio_set_level(gpio, active_high ? 0 : 1);
    const gpio_config_t io = {.pin_bit_mask = 1ULL << gpio, .mode = GPIO_MODE_OUTPUT};
    esp_err_t err = gpio_config(&io);
    if (err != ESP_OK) s_gpio = -1;
    motor(false);
    return err;
}

bool ha_haptic_ok(void) { return s_gpio >= 0 && s_cmd; }

void ha_haptic_play(ha_pattern_id_t id) {
    int cmd = (int)id;
    if (s_cmd) xQueueOverwrite(s_cmd, &cmd); // newest command wins
}

void ha_haptic_stop(void) {
    int cmd = CMD_STOP;
    if (s_cmd) xQueueOverwrite(s_cmd, &cmd);
}

static int64_t now_ms(void) { return esp_timer_get_time() / 1000; }

static void haptic_task(void *arg) {
    (void)arg;
    const ha_vib_pattern_t *pat = NULL;
    int step = 0;
    int64_t step_end = 0, play_start = 0;
    for (;;) {
        TickType_t wait = portMAX_DELAY; // idle: block, the CPU may light-sleep
        if (pat) {
            int64_t left = step_end - now_ms();
            wait = left > 0 ? pdMS_TO_TICKS(left) : 0;
        }
        int cmd;
        if (xQueueReceive(s_cmd, &cmd, wait) == pdTRUE) {
            const ha_vib_pattern_t *p = cmd >= 0 ? ha_vib_pattern((ha_pattern_id_t)cmd) : NULL;
            pat = (p && p->n) ? p : NULL;
            if (!pat) {
                motor(false);
                continue;
            }
            step = 0;
            play_start = now_ms();
            motor(pat->steps[0].duty > 0);
            step_end = play_start + pat->steps[0].ms;
            continue;
        }
        // Current step finished.
        int64_t now = now_ms();
        if (s_max_play_ms && now - play_start >= (int64_t)s_max_play_ms) {
            ESP_LOGW(TAG, "pattern stopped by safety limit (%u ms)", (unsigned)s_max_play_ms);
            pat = NULL;
            motor(false);
            continue;
        }
        if (++step >= pat->n) {
            if (!pat->repeat) {
                pat = NULL;
                motor(false);
                continue;
            }
            step = 0;
        }
        motor(pat->steps[step].duty > 0);
        step_end = now + pat->steps[step].ms;
    }
}

esp_err_t ha_haptic_start(uint32_t max_play_ms) {
    if (s_gpio < 0) return ESP_ERR_INVALID_STATE;
    if (s_cmd) return ESP_OK;
    s_max_play_ms = max_play_ms;
    s_cmd = xQueueCreate(1, sizeof(int));
    if (!s_cmd) return ESP_ERR_NO_MEM;
    if (xTaskCreate(haptic_task, "ha_haptic", 2048, NULL, 4, NULL) != pdPASS) {
        vQueueDelete(s_cmd);
        s_cmd = NULL;
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "motor GPIO%d active_%s, safety limit %u ms", s_gpio,
             s_active_high ? "high" : "low", (unsigned)max_play_ms);
    return ESP_OK;
}
