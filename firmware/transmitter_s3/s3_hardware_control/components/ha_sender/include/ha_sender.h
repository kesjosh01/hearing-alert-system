/*
 * S3 event sender: the ONLY entry point the ML / control layer needs.
 *
 *   #include "ha_sender.h"
 *   ha_publish_event(HA_EVENT_FIRE_ALARM, NULL);
 *
 * Contract (see firmware/transmitter_s3/README.md):
 *  - Pass a final, canonical event ID (ha_events.h), never a model class index.
 *    Background/"other" is not an event. Rising-edge / re-alert / re-arm policy
 *    is the caller's responsibility: every successful call is a NEW alert.
 *  - Never blocks on the network or waits for an ACK. Safe from any task (not
 *    from an ISR). Returns once the event is queued (or rejected).
 *  - Retransmissions are internal and keep the same (session, seq, event_id).
 *  - Two sounds decided together: use ha_publish_events(); they are queued
 *    FIRE_ALARM then DOORBELL, each with its own seq, sent stop-and-wait.
 *  - Queue is RAM only (lost on S3 reboot). See README for saturation / TTL.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"
#include "ha_events.h"
#include "ha_tx.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HA_PUBLISH_OK = 0,
    HA_PUBLISH_OK_EVICTED_OLDEST, /* queued; an older queued event was dropped (status cb: EVICTED) */
    HA_PUBLISH_ERR_INVALID_ID,    /* 0, TEST (product build) or unregistered ID */
    HA_PUBLISH_ERR_QUEUE_FULL,    /* rejected: full and nothing evictable under the policy */
    HA_PUBLISH_ERR_NOT_STARTED,   /* ha_sender_start() not called / failed */
    HA_PUBLISH_ERR_BUSY,          /* internal lock not obtained within a few ms */
} ha_publish_result_t;

/* Opaque handle for ha_cancel_event() and status reports. 0 = none. */
typedef uint32_t ha_event_token_t;

/* Start SoftAP + UDP task. Initialises NVS, netif and the default event loop. */
esp_err_t ha_sender_start(void);

ha_publish_result_t ha_publish_event(uint32_t event_id, ha_event_token_t *token_out);

/* Publish several events decided at the same time. results/tokens may be NULL.
 * Returns the number accepted. */
size_t ha_publish_events(const uint32_t *event_ids, size_t count, ha_publish_result_t *results,
                         ha_event_token_t *tokens);

/* Remove a queued or in-flight event (e.g. the detector retracts it). If it was
 * already transmitted the C3 may still alert. */
esp_err_t ha_cancel_event(ha_event_token_t token);

/* Outcome of every accepted event: ACKED / EXPIRED / EVICTED / CANCELED.
 * Called from the sender task; must return quickly and must not block. */
typedef void (*ha_sender_status_cb_t)(const ha_tx_report_t *report, void *ctx);
void ha_sender_set_status_callback(ha_sender_status_cb_t cb, void *ctx);

/* Hook for the S3's own light/indicator. Called in the publisher's context for
 * every valid product event, whether or not the radio queue accepted it, and
 * never waits for the C3 ACK. Must not block. No LED driver is provided here. */
typedef void (*ha_local_alert_hook_t)(uint32_t event_id, void *ctx);
void ha_sender_set_local_alert_hook(ha_local_alert_hook_t hook, void *ctx);

/* Optional wall-clock source for the TIME_SYNC extension (CONFIG_HA_TIME_SYNC_TX).
 * Return true only with a real UTC time (e.g. after SNTP); never a build time.
 * Called from the sender task; must not block. No SNTP client is included. */
typedef bool (*ha_time_provider_t)(void *ctx, int64_t *utc_ms, uint32_t *accuracy_ms);
void ha_sender_set_time_provider(ha_time_provider_t provider, void *ctx);

typedef struct {
    ha_tx_stats_t tx;
    uint32_t queued;
    bool peer_registered, peer_associated, peer_stale;
    uint32_t s3_session;
    uint32_t reports_dropped;
} ha_sender_info_t;

esp_err_t ha_sender_get_info(ha_sender_info_t *out);

const char *ha_publish_result_name(ha_publish_result_t r);

#ifdef __cplusplus
}
#endif
