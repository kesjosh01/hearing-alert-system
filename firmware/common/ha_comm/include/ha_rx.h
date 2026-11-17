/*
 * C3 receiver core: EVENT validation, duplicate suppression and the
 * "accept, then record, then ACK" contract.
 *
 * Duplicate state is RAM only:
 *   - per S3 session (a few most-recent sessions, LRU): the highest accepted seq
 *   - a short history of accepted (session, seq, event_id) for conflict checks
 * Keeping more than one session lets a delayed packet from a previous S3 boot
 * be recognised as a duplicate instead of triggering a second alert. A C3
 * reboot clears everything, so an event that was accepted but whose ACK was
 * lost may be accepted again after the reboot (at-least-once, not exactly-once).
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "ha_config.h"
#include "ha_proto.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool used;
    uint32_t session;
    uint32_t max_seq;
    uint32_t last_use;
} ha_rx_session_t;

typedef struct {
    bool used;
    uint32_t session, seq, event_id;
} ha_rx_hist_t;

typedef struct {
    uint32_t accepted, duplicates, stale, conflicts, rejected_invalid, accept_failed;
} ha_rx_stats_t;

typedef struct {
    bool allow_test;
    uint32_t clock;
    ha_rx_session_t sessions[HA_RX_SESSION_SLOTS];
    ha_rx_hist_t hist[HA_RX_HISTORY_LEN];
    uint32_t hist_next;
    ha_rx_stats_t stats;
} ha_rx_t;

typedef enum {
    HA_RX_ACK_NEW = 0,       /* accepted by the consumer; ACK sent */
    HA_RX_ACK_DUPLICATE,     /* already accepted; ACK again, no new alert */
    HA_RX_ACK_STALE,         /* older than an accepted seq of that session; ACK, no alert */
    HA_RX_DROP_INVALID,      /* seq 0 / not EVENT / unregistered or disallowed id */
    HA_RX_DROP_CONFLICT,     /* known (session, seq) with a different event_id */
    HA_RX_DROP_NOT_ACCEPTED, /* consumer refused (full / not attached): no ACK, no record */
} ha_rx_outcome_t;

/* Consumer hand-off. Must be non-blocking; return true only if the alert was
 * really queued for the UI/output layer. */
typedef bool (*ha_rx_accept_fn)(void *ctx, const ha_v1_frame_t *event);

void ha_rx_init(ha_rx_t *rx, bool allow_test);

/* Process one decoded EVENT that already passed source IP/port checks.
 * On HA_RX_ACK_* the caller sends ACK with the frame's own three fields. */
ha_rx_outcome_t ha_rx_handle_event(ha_rx_t *rx, const ha_v1_frame_t *f, ha_rx_accept_fn accept,
                                   void *ctx);

static inline bool ha_rx_should_ack(ha_rx_outcome_t o) {
    return o == HA_RX_ACK_NEW || o == HA_RX_ACK_DUPLICATE || o == HA_RX_ACK_STALE;
}

const char *ha_rx_outcome_name(ha_rx_outcome_t o);

#ifdef __cplusplus
}
#endif
