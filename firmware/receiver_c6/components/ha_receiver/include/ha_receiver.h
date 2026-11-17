/*
 * C6 receiver (XIAO ESP32-C6): joins the S3 SoftAP, validates EVENTs, suppresses duplicates
 * and hands accepted alerts to the UI (display / vibration) consumer.
 *
 * Acceptance contract (2026-11-15, ACK timing changed 2026-11-17):
 *  - An EVENT is accepted only if a UI consumer is attached
 *    (ha_receiver_attach_consumer()) and the alert queue has room. Without a
 *    consumer, or with a full queue, nothing is ACKed or recorded; the S3
 *    keeps retransmitting the same event.
 *  - CONFIG_HA_RX_ACK_ON_DISPLAY (default y): the ACK is sent only when the UI
 *    calls ha_receiver_alert_displayed() after the alert's screen was drawn.
 *    ACK then means "the alert's frame was transferred to the LCD without
 *    error" (with HA_DISP_READBACK also "the LCD answered, display on").
 *    Without a read line an unplugged LCD or a dead backlight is NOT
 *    detected. It never means the user noticed it. Retransmissions before that stay unanswered (no re-alert).
 *  - With the option off, ACK means "accepted into the UI queue" (old rule).
 *
 * Typical UI task:
 *     ha_receiver_attach_consumer();
 *     for (;;) {
 *         ha_alert_t a;
 *         if (ha_receiver_wait_alert(&a, portMAX_DELAY) && draw_screen(a.event_id))
 *             ha_receiver_alert_displayed(&a);   // ACK goes out here
 *     }
 * Blocking in ha_receiver_wait_alert() lets the C6 auto light-sleep. Long
 * vibration patterns run in the UI task and never delay reception or ACKs.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "ha_events.h"
#include "ha_rx.h"
#include "ha_time.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t event_id;        /* canonical ID (ha_events.h) */
    uint32_t s3_session;      /* identity of this alert, for UI de-dup/logs */
    uint32_t seq;
    int64_t received_mono_ms; /* esp_timer based, not wall-clock */
} ha_alert_t;

/* Start STA, power management, UDP receive and HELLO/report tasks. */
esp_err_t ha_receiver_start(void);

/* UI declares it will drain alerts. Until called, EVENTs are not ACKed. */
esp_err_t ha_receiver_attach_consumer(void);
void ha_receiver_detach_consumer(void);
bool ha_receiver_consumer_attached(void);

/* Pop the next accepted alert (FIFO). Returns false on timeout. */
bool ha_receiver_wait_alert(ha_alert_t *out, TickType_t wait);

/* UI: the screen for this alert has been drawn -> send its ACK now.
 * ESP_ERR_NOT_FOUND if it is not waiting (already ACKed / aged out);
 * ESP_ERR_NOT_SUPPORTED when CONFIG_HA_RX_ACK_ON_DISPLAY is off (ACK was
 * already sent on acceptance). Safe to call from any task. */
esp_err_t ha_receiver_alert_displayed(const ha_alert_t *alert);

/* Display watchdog input: reception time (esp_timer ms) of the oldest alert
 * that was accepted but is still not on screen. False if none (or when
 * CONFIG_HA_RX_ACK_ON_DISPLAY is off). */
bool ha_receiver_oldest_undisplayed(int64_t *received_mono_ms);

/* Optional wall clock (TIME_SYNC extension). HA_CLOCK_UNSYNCED until the S3
 * sends a valid time; outputs untouched in that case. */
ha_clock_state_t ha_receiver_get_time(int64_t *utc_ms, uint32_t *uncertainty_ms);

void ha_receiver_get_stats(ha_rx_stats_t *out);

#ifdef __cplusplus
}
#endif
