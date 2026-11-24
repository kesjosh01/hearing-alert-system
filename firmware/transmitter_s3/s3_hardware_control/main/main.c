#include "alert_led.h"
#include "esp_log.h"
#include "ha_sender.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

void ha_detector_task(void *arg);

void app_main(void) {
    ESP_ERROR_CHECK(ha_led_init());
    ESP_ERROR_CHECK(ha_sender_start());
    if (xTaskCreatePinnedToCore(ha_detector_task, "ha_detect", 16384, NULL, 5, NULL, 1) != pdPASS)
        ESP_LOGE("ha_detect", "cannot start detector task");
}
