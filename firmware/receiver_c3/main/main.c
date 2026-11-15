/*
 * Product C3 application entry. Starts the receiver and returns.
 *
 * No display/vibration consumer exists in this repository yet. Until the UI
 * team calls ha_receiver_attach_consumer() and drains ha_receiver_wait_alert(),
 * received EVENTs are deliberately NOT ACKed (the S3 keeps them queued and
 * retransmits). This is the agreed acceptance contract, not a fault.
 */
#include "esp_log.h"
#include "ha_receiver.h"

void app_main(void) {
    ESP_ERROR_CHECK(ha_receiver_start());
    if (!ha_receiver_consumer_attached())
        ESP_LOGW("app", "UI consumer: NOT attached (display/vibration not integrated); "
                        "EVENTs will not be ACKed");
}
