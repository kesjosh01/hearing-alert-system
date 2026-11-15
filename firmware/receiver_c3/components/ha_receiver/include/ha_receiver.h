/*
 * C3 receiver: joins the S3 SoftAP, validates EVENTs, suppresses duplicates
 * and hands accepted alerts to the UI (display / vibration) consumer.
 *
 * Acceptance contract (decided 2026-11-15):
 *  - An EVENT is ACKed only after it was placed in the alert queue AND a UI
 *    consumer is attached (ha_receiver_attach_consumer()). Without a consumer,
 *    or with a full queue, nothing is ACKed or recorded; the S3 keeps
 *    retransmitting the same event.
 *  - ACK means "C3 software accepted it". It does NOT mean the LED/screen/motor
 *    finished or that the user noticed.
 *
 * Typical UI task:
 *     ha_receiver_attach_consumer();
 *     for (;;) {
 *         ha_alert_t a;
 *         if (ha_receiver_wait_alert(&a, portMAX_DELAY)) play_pattern(a.event_id);
 *     }
 * Blocking in ha_receiver_wait_alert() lets the C3 auto light-sleep. Long
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

/* Optional wall clock (TIME_SYNC extension). HA_CLOCK_UNSYNCED until the S3
 * sends a valid time; outputs untouched in that case. */
ha_clock_state_t ha_receiver_get_time(int64_t *utc_ms, uint32_t *uncertainty_ms);

void ha_receiver_get_stats(ha_rx_stats_t *out);

#ifdef __cplusplus
}
#endif
