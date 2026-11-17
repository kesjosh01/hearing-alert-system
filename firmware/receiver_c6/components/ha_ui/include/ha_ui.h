/*
 * C6 receiver UI: turns received alerts into screen + vibration and sends the
 * ACK once the alert is on screen (ACK-on-display, 2026-11-17).
 *
 * Boot order (implementation_design 4-5):
 *   ha_ui_early_init()   motor pin off before anything else
 *   ha_receiver_start()  radio first (association takes time)
 *   ha_ui_start()        display, motor, button, self test, then the UI task
 *                        attaches as the receiver's consumer
 */
#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t ha_ui_early_init(void);
esp_err_t ha_ui_start(void);

#ifdef __cplusplus
}
#endif
