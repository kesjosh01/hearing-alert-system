/*
 * Product C6 (XIAO ESP32-C6) application entry.
 *
 * Order: motor pin off -> radio (association takes time) -> UI (display,
 * motor, button, then the UI task attaches as the alert consumer). EVENTs are
 * ACKed only after their screen has been drawn (CONFIG_HA_RX_ACK_ON_DISPLAY).
 */
#include "esp_log.h"
#include "ha_receiver.h"
#include "ha_ui.h"

void app_main(void) {
    ha_ui_early_init();
    ESP_ERROR_CHECK(ha_receiver_start());
    if (ha_ui_start() != ESP_OK)
        ESP_LOGE("app", "UI not running: EVENTs will not be ACKed (see ha_ui log)");
}
