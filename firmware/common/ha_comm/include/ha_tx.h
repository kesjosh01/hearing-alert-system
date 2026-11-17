/*
 * S3 sender core: bounded event queue + stop-and-wait retransmission + ACK
 * matching + single-peer registration state.
 *
 * Portable and single-threaded: the ESP-IDF glue (ha_sender component) owns
 * the lock, the socket and the monotonic clock and calls these functions.
 * All times are monotonic milliseconds (esp_timer based in firmware). UTC time
 * is never used here, so time synchronisation cannot move retry deadlines.
 *
 * Sequence numbers are assigned when an entry becomes the in-flight event, not
 * when it is published. Transmission order therefore always equals seq order,
 * which keeps the C3 "seq <= max is a duplicate" rule valid even if entries are
 * evicted, cancelled, expired or (optionally) reordered fire-first.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ha_proto.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HA_TX_QUEUE_MAX 32u

typedef enum {
    HA_PUB_OK = 0,
    HA_PUB_OK_EVICTED,       /* accepted; an older queued event was evicted (reported) */
    HA_PUB_ERR_INVALID_ID,   /* reserved, unregistered or TEST without diag option */
    HA_PUB_ERR_QUEUE_FULL,   /* rejected: queue full and nothing evictable */
} ha_pub_status_t;

typedef enum {
    HA_TXR_ACKED = 0,  /* C3 software accepted it (not "user noticed") */
    HA_TXR_EXPIRED,    /* TTL elapsed before ACK */
    HA_TXR_EVICTED,    /* removed by saturation policy */
    HA_TXR_CANCELED,   /* removed by ha_tx_cancel() */
} ha_tx_outcome_t;

typedef struct {
    ha_tx_outcome_t outcome;
    uint32_t token;
    uint32_t event_id;
    uint32_t session;        /* S3 session used for transmission (0 if never sent) */
    uint32_t seq;            /* 0 if it never became in-flight */
    uint32_t attempts_total; /* datagrams sent for this event */
    int64_t age_ms;          /* publish -> outcome */
    int64_t ack_after_last_tx_ms; /* ACKED only: last transmit -> ACK (upper bound of RTT) */
} ha_tx_report_t;

typedef void (*ha_tx_report_fn)(void *ctx, const ha_tx_report_t *report);

typedef struct {
    uint32_t capacity;           /* 1..HA_TX_QUEUE_MAX */
    uint32_t ack_wait_ms;        /* wait after each attempt */
    uint32_t attempts_per_burst; /* attempts before backing off */
    uint32_t backoff_min_ms;     /* pause after a failed burst, doubles ... */
    uint32_t backoff_max_ms;     /* ... up to this */
    uint32_t peer_stale_ms;      /* no HELLO for this long -> stale (still retried) */
    uint16_t peer_port;          /* required C3 source UDP port */
    bool allow_test;             /* diagnostic: accept TEST(1) */
    bool fire_first;             /* optional: next in-flight = oldest protected entry */
    uint32_t (*rand32)(void *ctx); /* new S3 session on seq exhaustion */
    void *rand_ctx;
    ha_tx_report_fn report;      /* outcome reports; must not block */
    void *report_ctx;
} ha_tx_config_t;

typedef struct {
    uint32_t token;
    uint32_t event_id;
    uint32_t seq;
    int64_t enq_ms;
    int64_t deadline_ms; /* INT64_MAX = no expiry */
    uint32_t attempts_total;
} ha_tx_entry_t;

typedef struct {
    bool registered;   /* got a valid HELLO */
    bool associated;   /* Wi-Fi station currently joined */
    bool stale;        /* registered but no HELLO for peer_stale_ms */
    uint32_t ip;       /* as received (network order, opaque here) */
    uint16_t port;     /* as received (network order, opaque here) */
    uint32_t c3_session;
    int64_t last_seen_ms;
} ha_tx_peer_t;

typedef struct {
    uint32_t published, evicted, rejected_full, rejected_invalid;
    uint32_t acked, expired, canceled, datagrams, bursts_failed;
    uint32_t acks_ignored;
} ha_tx_stats_t;

typedef struct {
    ha_tx_config_t cfg;
    uint32_t session;
    uint32_t seq; /* last assigned */
    uint32_t next_token;
    ha_tx_entry_t q[HA_TX_QUEUE_MAX]; /* FIFO; q[0] is in-flight when inflight */
    uint32_t count;
    bool inflight;
    uint32_t burst_attempts;
    uint32_t backoff_ms;
    int64_t next_send_ms;
    int64_t last_tx_ms;
    ha_tx_peer_t peer;
    ha_tx_stats_t stats;
} ha_tx_t;

/* Fills cfg with ha_config.h defaults (report/rand left NULL). */
void ha_tx_default_config(ha_tx_config_t *cfg);

void ha_tx_init(ha_tx_t *tx, const ha_tx_config_t *cfg, uint32_t session);

/* Queue one final event. Never blocks, never waits for ACK. */
ha_pub_status_t ha_tx_publish(ha_tx_t *tx, uint32_t event_id, int64_t now_ms, uint32_t *token);

/* Queue several events decided together: sorted to canonical batch order
 * (FIRE_ALARM, DOORBELL) and published individually. Returns accepted count. */
size_t ha_tx_publish_batch(ha_tx_t *tx, const uint32_t *ids, size_t n, int64_t now_ms,
                           uint32_t *tokens, ha_pub_status_t *results);

/* Remove a queued/in-flight event. If it was already transmitted, C3 may have
 * accepted it anyway. Returns false if the token is unknown. */
bool ha_tx_cancel(ha_tx_t *tx, uint32_t token, int64_t now_ms);

typedef struct {
    bool send;               /* out[] holds a 16-byte EVENT to send to the peer */
    bool new_burst;          /* send is the first attempt of a retry burst after backoff */
    bool peer_went_stale;    /* transition to stale happened in this call */
    uint32_t seq, event_id, burst_attempt;
} ha_tx_action_t;

/* Advance timers: expiry, staleness, in-flight selection, (re)transmission. */
ha_tx_action_t ha_tx_poll(ha_tx_t *tx, int64_t now_ms, uint8_t out[HA_V1_LEN]);

/* Milliseconds until ha_tx_poll() has something to do (0 = now). */
int64_t ha_tx_ms_until_due(const ha_tx_t *tx, int64_t now_ms);

typedef enum {
    HA_HELLO_INVALID = 0,
    HA_HELLO_REFRESH,       /* same peer, same C3 session */
    HA_HELLO_NEW_PEER,
    HA_HELLO_ADDR_CHANGED,  /* new IP/port (e.g. DHCP renew) */
    HA_HELLO_C3_REBOOTED,   /* same address, new C3 session */
} ha_hello_result_t;

/* Register/refresh the peer. Never deletes queued events and never changes the
 * S3 session. A (re)registration restarts the current burst immediately. */
ha_hello_result_t ha_tx_on_hello(ha_tx_t *tx, const ha_v1_frame_t *f, uint32_t ip,
                                 uint16_t port, int64_t now_ms);

/* Wi-Fi association edges from the AP driver (single peer: any station). */
void ha_tx_on_station_joined(ha_tx_t *tx, int64_t now_ms);
void ha_tx_on_station_left(ha_tx_t *tx, int64_t now_ms);

typedef enum {
    HA_ACK_MATCHED = 0,
    HA_ACK_IGNORED_NO_PEER,
    HA_ACK_IGNORED_ADDR,       /* not from the registered C3 IP/port */
    HA_ACK_IGNORED_NOT_INFLIGHT,
    HA_ACK_IGNORED_MISMATCH,   /* session, seq or event_id differs */
} ha_ack_result_t;

/* Only an ACK whose session, seq AND event_id equal the in-flight event, sent
 * from the registered address/port, removes it. */
ha_ack_result_t ha_tx_on_ack(ha_tx_t *tx, const ha_v1_frame_t *f, uint32_t ip, uint16_t port,
                             int64_t now_ms);

/* True if a diagnostic packet (PM_STATS, POWER_STATE) comes from the peer. */
bool ha_tx_from_peer(const ha_tx_t *tx, const ha_v1_frame_t *f, uint32_t ip, uint16_t port);

/* Can EVENTs be transmitted right now? */
bool ha_tx_link_up(const ha_tx_t *tx);

/* No in-flight and nothing queued: lower-priority traffic (TIME_SYNC) may go. */
bool ha_tx_idle(const ha_tx_t *tx);

const char *ha_pub_status_name(ha_pub_status_t s);
const char *ha_tx_outcome_name(ha_tx_outcome_t o);
const char *ha_ack_result_name(ha_ack_result_t r);
const char *ha_hello_result_name(ha_hello_result_t r);

#ifdef __cplusplus
}
#endif
