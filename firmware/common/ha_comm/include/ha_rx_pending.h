/*
 * Receiver "accepted, waiting for display" table (ACK-on-display contract,
 * decided 2026-11-17).
 *
 * With ACK-on-display the receiver accepts an EVENT into the UI queue (ha_rx
 * records it for duplicate suppression) but sends NO ACK yet. The UI calls
 * back once the alert is on screen; only then is the ACK sent. Meanwhile the
 * S3 keeps retransmitting the same (session, seq): ha_rx classifies those as
 * duplicates, and this table tells the glue "not displayed yet -> stay quiet".
 *
 * An entry is dropped (when room is needed) once it is older than its event's
 * TTL: the S3 counts the TTL from publish, which is before our reception, so
 * it has already given up on that event. Unknown IDs use max_age_ms.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "ha_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HA_RX_PENDING_LEN
#define HA_RX_PENDING_LEN (HA_RX_ALERT_QUEUE_LEN + 4)
#endif

typedef struct {
    bool used;
    uint32_t session, seq, event_id;
    int64_t since_ms;
} ha_rx_pending_entry_t;

typedef struct {
    ha_rx_pending_entry_t e[HA_RX_PENDING_LEN];
    int64_t max_age_ms;
    uint32_t aged_out; /* entries dropped without display (diagnostic) */
} ha_rx_pending_t;

/* max_age_ms <= 0 selects the longest registered event TTL. */
void ha_rx_pending_init(ha_rx_pending_t *p, int64_t max_age_ms);

/* True if add() would succeed now (ages out stale entries first). */
bool ha_rx_pending_has_room(ha_rx_pending_t *p, int64_t now_ms);

/* Remember an accepted, not yet displayed event. False if full. */
bool ha_rx_pending_add(ha_rx_pending_t *p, uint32_t session, uint32_t seq, uint32_t event_id,
                       int64_t now_ms);

bool ha_rx_pending_contains(const ha_rx_pending_t *p, uint32_t session, uint32_t seq);

/* Displayed: remove the entry. False if it is not pending (already ACKed,
 * aged out, or never accepted) or if event_id does not match. */
bool ha_rx_pending_take(ha_rx_pending_t *p, uint32_t session, uint32_t seq, uint32_t event_id);

int ha_rx_pending_count(const ha_rx_pending_t *p);

/* Oldest entry still waiting (not aged out at now_ms). False if none. Used by
 * the display watchdog: an alert that is not on screen after a few seconds
 * means the display path is stuck. */
bool ha_rx_pending_oldest(const ha_rx_pending_t *p, int64_t now_ms, int64_t *since_ms);

#ifdef __cplusplus
}
#endif
