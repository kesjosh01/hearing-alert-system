/*
 * Product S3 application entry. Starts the SoftAP/UDP sender and returns.
 *
 * No event source is attached here: audio input, inference and the final
 * detection policy belong to the ML integration, which will call
 * ha_publish_event() / ha_publish_events() (ha_sender.h). Without that input
 * the board boots, serves the AP and waits; it never fabricates events.
 */
#include "esp_log.h"
#include "ha_sender.h"

void app_main(void) {
    ESP_ERROR_CHECK(ha_sender_start());
    ESP_LOGI("app", "S3 ready. Event source: none attached (ML integration pending); "
                    "waiting for ha_publish_event()");
}
