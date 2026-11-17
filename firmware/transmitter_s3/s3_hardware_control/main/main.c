
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/i2s_std.h"
#include "driver/ledc.h"

#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

#include "dtw_classifier.h"

// ============================================================
// Audio configuration
// ============================================================

#define SAMPLE_RATE       16000
#define WINDOW_SAMPLES    16000
#define HOP_SAMPLES       3200
#define CHUNK_SAMPLES     1024

// INMP441: LILYGO T7_S3 V1.1
#define I2S_BCLK_GPIO     GPIO_NUM_4
#define I2S_WS_GPIO       GPIO_NUM_5
#define I2S_DIN_GPIO      GPIO_NUM_6

// ============================================================
// RGB LED configuration
// Common Cathode Arduino RGB LED
// ============================================================

#define LED_R_GPIO        GPIO_NUM_7
#define LED_G_GPIO        GPIO_NUM_9
#define LED_B_GPIO        GPIO_NUM_10

#define LEDC_MODE         LEDC_LOW_SPEED_MODE
#define LEDC_TIMER        LEDC_TIMER_0
#define LEDC_FREQUENCY    5000
#define LEDC_RESOLUTION   LEDC_TIMER_8_BIT

#define LEDC_R_CHANNEL    LEDC_CHANNEL_0
#define LEDC_G_CHANNEL    LEDC_CHANNEL_1
#define LEDC_B_CHANNEL    LEDC_CHANNEL_2

#define LED_BRIGHTNESS    64

// ============================================================
// Classification voting configuration
// ============================================================

#define VOTE_COUNT        3
#define REQUIRED_VOTES    2

static const char *TAG = "DTW";

// ============================================================
// Global variables
// ============================================================

static i2s_chan_handle_t rx_handle = NULL;

static int32_t raw_buffer[CHUNK_SAMPLES];

static int16_t *pcm_buffer = NULL;

// Classification history
static sound_class_t class_history[VOTE_COUNT] = {
    SOUND_UNKNOWN,
    SOUND_UNKNOWN,
    SOUND_UNKNOWN
};

static int history_index = 0;
static int history_count = 0;

// Current confirmed LED state
static sound_class_t confirmed_class = SOUND_UNKNOWN;

// ============================================================
// RGB LED initialization
// ============================================================

static void rgb_led_init(void)
{
    ledc_timer_config_t timer_config = {
        .speed_mode = LEDC_MODE,
        .duty_resolution = LEDC_RESOLUTION,
        .timer_num = LEDC_TIMER,
        .freq_hz = LEDC_FREQUENCY,
        .clk_cfg = LEDC_AUTO_CLK
    };

    ESP_ERROR_CHECK(
        ledc_timer_config(&timer_config)
    );

    ledc_channel_config_t channels[] = {
        {
            .gpio_num = LED_R_GPIO,
            .speed_mode = LEDC_MODE,
            .channel = LEDC_R_CHANNEL,
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = LEDC_TIMER,
            .duty = 0,
            .hpoint = 0
        },
        {
            .gpio_num = LED_G_GPIO,
            .speed_mode = LEDC_MODE,
            .channel = LEDC_G_CHANNEL,
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = LEDC_TIMER,
            .duty = 0,
            .hpoint = 0
        },
        {
            .gpio_num = LED_B_GPIO,
            .speed_mode = LEDC_MODE,
            .channel = LEDC_B_CHANNEL,
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = LEDC_TIMER,
            .duty = 0,
            .hpoint = 0
        }
    };

    for (int i = 0; i < 3; i++) {
        ESP_ERROR_CHECK(
            ledc_channel_config(&channels[i])
        );
    }

    ESP_LOGI(TAG, "RGB LED initialized");
}

// ============================================================
// Set RGB LED color
// Common Cathode: HIGH duty = ON
// ============================================================

static void rgb_led_set_color(
    uint8_t red,
    uint8_t green,
    uint8_t blue
)
{
    ESP_ERROR_CHECK(
        ledc_set_duty(
            LEDC_MODE,
            LEDC_R_CHANNEL,
            red
        )
    );

    ESP_ERROR_CHECK(
        ledc_update_duty(
            LEDC_MODE,
            LEDC_R_CHANNEL
        )
    );

    ESP_ERROR_CHECK(
        ledc_set_duty(
            LEDC_MODE,
            LEDC_G_CHANNEL,
            green
        )
    );

    ESP_ERROR_CHECK(
        ledc_update_duty(
            LEDC_MODE,
            LEDC_G_CHANNEL
        )
    );

    ESP_ERROR_CHECK(
        ledc_set_duty(
            LEDC_MODE,
            LEDC_B_CHANNEL,
            blue
        )
    );

    ESP_ERROR_CHECK(
        ledc_update_duty(
            LEDC_MODE,
            LEDC_B_CHANNEL
        )
    );
}

// ============================================================
// Update LED according to classification voting
// ============================================================

static void update_led(const dtw_result_t *result)
{
    // UNKNOWN: reset history and show GREEN
    if (result->sound_class == SOUND_UNKNOWN) {

        history_index = 0;
        history_count = 0;

        for (int i = 0; i < VOTE_COUNT; i++) {
            class_history[i] = SOUND_UNKNOWN;
        }

        confirmed_class = SOUND_UNKNOWN;

        rgb_led_set_color(
            0,
            LED_BRIGHTNESS,
            0
        );

        return;
    }

    // Add new classification to history
    class_history[history_index] = result->sound_class;

    history_index = (history_index + 1) % VOTE_COUNT;

    if (history_count < VOTE_COUNT) {
        history_count++;
    }

    // Count recent classifications
    int fire_votes = 0;
    int doorbell_votes = 0;

    for (int i = 0; i < history_count; i++) {

        if (class_history[i] == SOUND_FIRE_ALARM) {
            fire_votes++;
        }
        else if (class_history[i] == SOUND_DOORBELL) {
            doorbell_votes++;
        }
    }

    // FIRE ALARM: at least 2 votes
    if (fire_votes >= REQUIRED_VOTES) {

        if (confirmed_class != SOUND_FIRE_ALARM) {

            confirmed_class = SOUND_FIRE_ALARM;

            rgb_led_set_color(
                LED_BRIGHTNESS,
                0,
                0
            );

            ESP_LOGI(TAG, "LED: RED - FIRE ALARM");
        }
    }

    // DOORBELL: at least 2 votes
    else if (doorbell_votes >= REQUIRED_VOTES) {

        if (confirmed_class != SOUND_DOORBELL) {

            confirmed_class = SOUND_DOORBELL;

            rgb_led_set_color(
                0,
                0,
                LED_BRIGHTNESS
            );

            ESP_LOGI(TAG, "LED: BLUE - DOORBELL");
        }
    }

    // No majority: keep previous LED color
}

// ============================================================
// Microphone initialization
// ============================================================

static void microphone_init(void)
{
    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(
            I2S_NUM_0,
            I2S_ROLE_MASTER
        );

    chan_cfg.dma_desc_num = 6;
    chan_cfg.dma_frame_num = 256;

    ESP_ERROR_CHECK(
        i2s_new_channel(
            &chan_cfg,
            NULL,
            &rx_handle
        )
    );

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(
            SAMPLE_RATE
        ),

        .slot_cfg =
            I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                I2S_DATA_BIT_WIDTH_32BIT,
                I2S_SLOT_MODE_MONO
            ),

        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCLK_GPIO,
            .ws = I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din = I2S_DIN_GPIO,

            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false
            }
        }
    };

    // INMP441 L/R = GND
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(
        i2s_channel_init_std_mode(
            rx_handle,
            &std_cfg
        )
    );

    ESP_ERROR_CHECK(
        i2s_channel_enable(rx_handle)
    );

    ESP_LOGI(TAG, "INMP441 initialized");
}

// ============================================================
// Read PCM samples from microphone
// ============================================================

static void read_audio_samples(
    int16_t *destination,
    size_t sample_count
)
{
    size_t collected = 0;

    while (collected < sample_count)
    {
        size_t bytes_read = 0;

        size_t remaining = sample_count - collected;

        size_t request_samples =
            remaining > CHUNK_SAMPLES
            ? CHUNK_SAMPLES
            : remaining;

        ESP_ERROR_CHECK(
            i2s_channel_read(
                rx_handle,
                raw_buffer,
                request_samples * sizeof(int32_t),
                &bytes_read,
                portMAX_DELAY
            )
        );

        size_t samples_read =
            bytes_read / sizeof(int32_t);

        if (samples_read == 0) {
            continue;
        }

        if (samples_read > request_samples) {
            samples_read = request_samples;
        }

        for (size_t i = 0; i < samples_read; i++)
        {
            destination[collected + i] =
                (int16_t)(raw_buffer[i] >> 16);
        }

        collected += samples_read;
    }
}

// ============================================================
// Print classification result
// ============================================================

static void print_result(const dtw_result_t *result)
{
    switch (result->sound_class)
    {
        case SOUND_FIRE_ALARM:

            ESP_LOGI(
                TAG,
                "FIRE=%.2f, DOORBELL=%.2f -> FIRE ALARM",
                result->fire_distance,
                result->doorbell_distance
            );

            break;

        case SOUND_DOORBELL:

            ESP_LOGI(
                TAG,
                "FIRE=%.2f, DOORBELL=%.2f -> DOORBELL",
                result->fire_distance,
                result->doorbell_distance
            );

            break;

        default:

            ESP_LOGI(TAG, "UNKNOWN");

            break;
    }
}

// ============================================================
// Main application
// ============================================================

void app_main(void)
{
    ESP_LOGI(TAG, "Starting real-time sound classification");

    // Initialize RGB LED
    rgb_led_init();

    // Initial state: UNKNOWN = GREEN
    rgb_led_set_color(
        0,
        LED_BRIGHTNESS,
        0
    );

    // Allocate 1-second PCM buffer
    pcm_buffer = heap_caps_malloc(
        WINDOW_SAMPLES * sizeof(int16_t),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );

    if (pcm_buffer == NULL) {
        pcm_buffer = heap_caps_malloc(
            WINDOW_SAMPLES * sizeof(int16_t),
            MALLOC_CAP_8BIT
        );
    }

    if (pcm_buffer == NULL) {
        ESP_LOGE(TAG, "Audio buffer allocation failed");
        return;
    }

    // Initialize microphone
    microphone_init();

    ESP_LOGI(TAG, "Waiting for audio...");

    // Initial 1-second recording
    read_audio_samples(
        pcm_buffer,
        WINDOW_SAMPLES
    );

    ESP_LOGI(TAG, "Initial audio window ready");

    // Continuous classification
    while (1)
    {
        // Classify current 1-second window
        dtw_result_t result = dtw_classify(pcm_buffer);

        // Print classification
        print_result(&result);

        // Update LED using majority voting
        update_led(&result);

        // Keep newest 0.8 seconds
        memmove(
            pcm_buffer,
            pcm_buffer + HOP_SAMPLES,
            (WINDOW_SAMPLES - HOP_SAMPLES) * sizeof(int16_t)
        );

        // Read 0.2 seconds of new audio
        read_audio_samples(
            pcm_buffer + (WINDOW_SAMPLES - HOP_SAMPLES),
            HOP_SAMPLES
        );
    }
}
