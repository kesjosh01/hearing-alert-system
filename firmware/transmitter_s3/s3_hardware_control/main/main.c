
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_err.h"
#include "driver/ledc.h"

// -----------------------------
// 기본 설정
// -----------------------------

#define SAMPLE_RATE     16000
#define BUFFER_SIZE     256

// 12회마다 로그 출력 (약 0.192초)
#define LOG_INTERVAL    12

// INMP441 연결 GPIO
#define I2S_BCLK_GPIO   4
#define I2S_WS_GPIO     5
#define I2S_DIN_GPIO    6

// 소리 크기 정의
#define RMS_THRESHOLD_NORMAL  30000.0f
#define RMS_THRESHOLD_LOUD   500000.0f

// RGB LED GPIO
#define LED_R_GPIO  7
#define LED_G_GPIO  9
#define LED_B_GPIO  10

// PWM 설정
#define LED_PWM_FREQ  5000
#define LED_PWM_RES   LEDC_TIMER_8_BIT

static const char *TAG = "MIC_TEST";

static i2s_chan_handle_t rx_handle;

// -----------------------------
// I2S 마이크 초기화
// -----------------------------

static void microphone_init(void)
{
    // I2S RX 채널 생성
    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(
            I2S_NUM_0,
            I2S_ROLE_MASTER
        );

    ESP_ERROR_CHECK(
        i2s_new_channel(
            &chan_cfg,
            NULL,
            &rx_handle
        )
    );

    // I2S 표준 모드 설정
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),

        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT,
            I2S_SLOT_MODE_MONO
        ),

        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCLK_GPIO,
            .ws   = I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S_DIN_GPIO,

            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false
            }
        }
    };

    // INMP441의 L/R 핀이 GND에 연결되어 있으므로
    // 왼쪽 채널 데이터를 수신
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

    ESP_LOGI(TAG, "INMP441 initialization complete");
}

// -----------------------------
// RMS 계산
// -----------------------------

static float calculate_rms(
    int32_t *samples,
    int count
)
{
    double sum = 0.0;
    double mean = 0.0;

    // DC 성분 계산
    for (int i = 0; i < count; i++) {
        mean += (double)(samples[i] >> 8);
    }

    mean /= count;

    // DC 성분 제거 후 제곱합 계산
    for (int i = 0; i < count; i++) {
        double value = (double)(samples[i] >> 8) - mean;
        sum += value * value;
    }

    return (float)sqrt(sum / count);
}

// RGB LED 초기화

static void led_init(void)
{
    ledc_timer_config_t timer_cfg = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .timer_num = LEDC_TIMER_0,
        .duty_resolution = LED_PWM_RES,
        .freq_hz = LED_PWM_FREQ,
        .clk_cfg = LEDC_AUTO_CLK
    };

    ESP_ERROR_CHECK(ledc_timer_config(&timer_cfg));

    ledc_channel_config_t channels[] = {
        {
            .gpio_num = LED_R_GPIO,
            .speed_mode = LEDC_LOW_SPEED_MODE,
            .channel = LEDC_CHANNEL_0,
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = LEDC_TIMER_0,
            .duty = 0,
            .hpoint = 0
        },
        {
            .gpio_num = LED_G_GPIO,
            .speed_mode = LEDC_LOW_SPEED_MODE,
            .channel = LEDC_CHANNEL_1,
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = LEDC_TIMER_0,
            .duty = 0,
            .hpoint = 0
        },
        {
            .gpio_num = LED_B_GPIO,
            .speed_mode = LEDC_LOW_SPEED_MODE,
            .channel = LEDC_CHANNEL_2,
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = LEDC_TIMER_0,
            .duty = 0,
            .hpoint = 0
        }
    };

    for (int i = 0; i < 3; i++)
    {
        ESP_ERROR_CHECK(ledc_channel_config(&channels[i]));
    }

    ESP_LOGI(TAG, "RGB LED initialization complete");
}

// RGB LED 색상 설정

static void set_rgb_color(uint32_t red, uint32_t green, uint32_t blue)
{
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, red);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);

    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1, green);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1);

    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_2, blue);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_2);
}

// -----------------------------
// 메인 프로그램
// -----------------------------

void app_main(void)
{
    microphone_init();

    led_init();

    // 시작할 때 초록색
    set_rgb_color(0, 255, 0);

    int32_t samples[BUFFER_SIZE];

    size_t bytes_read = 0;

    float rms_sum = 0.0f;
    int rms_count = 0;

    ESP_LOGI(TAG, "Microphone sound monitoring started");

    while (1)
    {
        // I2S 데이터 읽기
        esp_err_t ret = i2s_channel_read(
            rx_handle,
            samples,
            sizeof(samples),
            &bytes_read,
            portMAX_DELAY
        );

        if (ret == ESP_OK && bytes_read > 0)
        {
            int sample_count = bytes_read / sizeof(int32_t);

            if (sample_count > 0)
            {
                // RMS 계산
                float rms = calculate_rms(
                    samples,
                    sample_count
                );

                // 여러 번의 RMS 결과 누적
                rms_sum += rms;
                rms_count++;

                // 약 0.2초마다 소리 크기 구분
                if (rms_count >= LOG_INTERVAL)
                {
                    float average_rms = rms_sum / rms_count;
                    
                    if (average_rms < RMS_THRESHOLD_NORMAL)
                    {
                        ESP_LOGI(TAG, "Sound level: Normal (RMS: %.2f)", average_rms);

                        // 파란색
                        set_rgb_color(0, 0, 255);
                    }
                    else if (average_rms < RMS_THRESHOLD_LOUD)
                    {
                        ESP_LOGW(TAG, "Sound level: Loud (RMS: %.2f)", average_rms);

                        // 초록색
                        set_rgb_color(0, 255, 0);
                    }
                    else
                    {
                        ESP_LOGE(TAG, "Sound level: Very Loud! (RMS: %.2f)", average_rms);
                        // 빨간색
                        set_rgb_color(255, 0, 0);
                    }
                    // 누적값 초기화
                    rms_sum = 0.0f;
                    rms_count = 0;
                }
            }
        }
        else if (ret != ESP_OK)
        {
            ESP_LOGE(TAG, "I2S read failed: %s", esp_err_to_name(ret));
        }
    }
}
