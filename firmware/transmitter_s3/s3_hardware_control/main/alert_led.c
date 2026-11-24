#include "alert_led.h"

#include "driver/gpio.h"
#include "led_strip.h"
#include "sdkconfig.h"

#ifndef CONFIG_HA_LED_GPIO
#define CONFIG_HA_LED_GPIO 10
#endif
#ifndef CONFIG_HA_LED_BRIGHTNESS
#define CONFIG_HA_LED_BRIGHTNESS 32
#endif

static led_strip_handle_t strip;
static ha_led_state_t current = (ha_led_state_t)-1;

esp_err_t ha_led_init(void) {
    const led_strip_config_t strip_config = {
        .strip_gpio_num = CONFIG_HA_LED_GPIO,
        .max_leds = 1,
        .led_pixel_format = LED_PIXEL_FORMAT_GRB,
        .led_model = LED_MODEL_WS2812,
        .flags.invert_out = false,
    };
    const led_strip_rmt_config_t rmt_config = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10000000,
        .flags.with_dma = false,
    };
    esp_err_t err = led_strip_new_rmt_device(&strip_config, &rmt_config, &strip);
    if (err != ESP_OK) return err;
    return ha_led_set_state(HA_LED_IDLE);
}

esp_err_t ha_led_set_state(ha_led_state_t state) {
    if (!strip) return ESP_ERR_INVALID_STATE;
    if (state == current) return ESP_OK;

    uint8_t red = 0, green = 0, blue = 0;
    switch (state) {
        case HA_LED_FIRE: red = CONFIG_HA_LED_BRIGHTNESS; break;
        case HA_LED_DOORBELL: blue = CONFIG_HA_LED_BRIGHTNESS; break;
        case HA_LED_IDLE: green = CONFIG_HA_LED_BRIGHTNESS; break;
        default: return ESP_ERR_INVALID_ARG;
    }
    esp_err_t err = led_strip_set_pixel(strip, 0, red, green, blue);
    if (err != ESP_OK) return err;
    err = led_strip_refresh(strip);
    if (err == ESP_OK) current = state;
    return err;
}
