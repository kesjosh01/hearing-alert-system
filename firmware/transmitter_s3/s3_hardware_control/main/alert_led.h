#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HA_LED_IDLE = 0,
    HA_LED_FIRE,
    HA_LED_DOORBELL,
} ha_led_state_t;

esp_err_t ha_led_init(void);
esp_err_t ha_led_set_state(ha_led_state_t state);

#ifdef __cplusplus
}
#endif
