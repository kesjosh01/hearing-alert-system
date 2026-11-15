/*
 * Host unit tests for firmware/common/ha_comm (the code linked into both
 * firmware images). Run: bash firmware/common/ha_comm/test/host/run_tests.sh
 *
 * These tests exercise protocol bytes, queue/ACK/duplicate logic and the time
 * extension with simulated clocks and a simulated lossy link. They are NOT a
 * substitute for on-board radio, sleep or current measurements.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ha_config.h"
#include "ha_events.h"
#include "ha_proto.h"
#include "ha_rx.h"
#include "ha_time.h"
#include "ha_tx.h"

static int g_fail, g_checks;
#define CHECK(c)                                                                 \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(c)) {                                                              \
            ++g_fail;                                                            \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);         \
        }                                                                        \
    } while (0)
#define CHECK_EQ(a, b) CHECK((long long)(a) == (long long)(b))

static int hex_to_bytes(const char *hex, uint8_t *out) {
    int n = 0;
    while (*hex) {
        if (*hex == ' ') { ++hex; continue; }
        unsigned v;
        sscanf(hex, "%2x", &v);
        out[n++] = (uint8_t)v;
        hex += 2;
    }
    return n;
}

/* ---------------------------------------------------------------- vectors */
/* From docs/communication/README.md section 6 (session 0x12345678). */
static const char *V_FIRE_EVENT = "48 41 01 02 12 34 56 78 00 00 00 01 00 00 01 00";
static const char *V_FIRE_ACK = "48 41 01 03 12 34 56 78 00 00 00 01 00 00 01 00";
static const char *V_DOOR_EVENT = "48 41 01 02 12 34 56 78 00 00 00 02 00 00 01 01";
static const char *V_DOOR_ACK = "48 41 01 03 12 34 56 78 00 00 00 02 00 00 01 01";

static void test_doc_vectors(void) {
    struct { const char *hex; uint8_t type; uint32_t seq, id; } v[] = {
        {V_FIRE_EVENT, HA_MSG_EVENT, 1, HA_EVENT_FIRE_ALARM},
        {V_FIRE_ACK, HA_MSG_ACK, 1, HA_EVENT_FIRE_ALARM},
        {V_DOOR_EVENT, HA_MSG_EVENT, 2, HA_EVENT_DOORBELL},
        {V_DOOR_ACK, HA_MSG_ACK, 2, HA_EVENT_DOORBELL},
    };
    for (size_t i = 0; i < 4; ++i) {
        uint8_t want[32], got[HA_V1_LEN];
        CHECK_EQ(hex_to_bytes(v[i].hex, want), 16);
        ha_v1_frame_t f = {v[i].type, 0x12345678u, v[i].seq, v[i].id};
        CHECK_EQ(ha_v1_encode(&f, got), 16);
        CHECK(memcmp(want, got, 16) == 0);
        ha_v1_frame_t d;
        CHECK_EQ(ha_v1_decode(want, 16, &d), HA_DEC_OK);
        CHECK(d.type == v[i].type && d.session == 305419896u && d.seq == v[i].seq &&
              d.value == v[i].id);
    }
}

static void test_error_vectors(void) {
    uint8_t b[32];
    ha_v1_frame_t d;
    hex_to_bytes(V_FIRE_EVENT, b);
    CHECK_EQ(ha_v1_decode(b, 15, &d), HA_DEC_ERR_LENGTH);
    b[16] = 0x00;
    CHECK_EQ(ha_v1_decode(b, 17, &d), HA_DEC_ERR_LENGTH); // never truncated to 16
    CHECK_EQ(ha_v1_decode(b, 0, &d), HA_DEC_ERR_LENGTH);

    hex_to_bytes(V_FIRE_EVENT, b); b[0] = 0x00;
    CHECK_EQ(ha_v1_decode(b, 16, &d), HA_DEC_ERR_MAGIC);
    hex_to_bytes(V_FIRE_EVENT, b); b[2] = 0x02;
    CHECK_EQ(ha_v1_decode(b, 16, &d), HA_DEC_ERR_VERSION);
    hex_to_bytes(V_FIRE_EVENT, b); b[3] = 0xFF;
    CHECK_EQ(ha_v1_decode(b, 16, &d), HA_DEC_ERR_TYPE);
    hex_to_bytes(V_FIRE_EVENT, b); b[3] = 0x00;
    CHECK_EQ(ha_v1_decode(b, 16, &d), HA_DEC_ERR_TYPE);

    // seq = 0
    hex_to_bytes(V_FIRE_EVENT, b); memset(b + 8, 0, 4);
    CHECK_EQ(ha_v1_decode(b, 16, &d), HA_DEC_OK);
    CHECK(!ha_v1_event_fields_valid(&d));

    // id 258 (unregistered), 0 (reserved), 1 (TEST)
    CHECK(!ha_event_allowed(0x102, false));
    CHECK(!ha_event_allowed(0x102, true));
    CHECK(!ha_event_allowed(HA_EVENT_RESERVED, true));
    CHECK(!ha_event_allowed(HA_EVENT_TEST, false));
    CHECK(ha_event_allowed(HA_EVENT_TEST, true));
    CHECK(ha_event_allowed(HA_EVENT_FIRE_ALARM, false));
    CHECK(ha_event_allowed(HA_EVENT_DOORBELL, false));

    // HELLO field rules
    ha_v1_frame_t h = {HA_MSG_HELLO, 7, 0, 0};
    CHECK(ha_v1_hello_fields_valid(&h));
    h.seq = 1;
    CHECK(!ha_v1_hello_fields_valid(&h));
}

static void test_time_sync_codec(void) {
    ha_time_sync_t m = {0, 0x12345678u, 1, 1763164800123LL, 250};
    uint8_t b[64];
    CHECK_EQ(ha_time_sync_encode(&m, b), 28);
    const char *want =
        /* utc 1763164800123 = 0x0000019A84CFC47B (cross-checked with Python
         * int.to_bytes(8, "big")), accuracy 250 = 0xFA */
        "48 41 02 10 00 1C 00 00 12 34 56 78 00 00 00 01 00 00 01 9A 84 CF C4 7B 00 00 00 FA";
    uint8_t w[64];
    CHECK_EQ(hex_to_bytes(want, w), 28);
    CHECK(memcmp(b, w, 28) == 0);
    ha_time_sync_t d;
    CHECK_EQ(ha_time_sync_decode(b, 28, &d), HA_DEC_OK);
    CHECK(d.utc_ms == m.utc_ms && d.accuracy_ms == 250 && d.seq == 1 && d.session == m.session);
    CHECK_EQ(ha_frame_version(b, 28), 2);

    ha_v1_frame_t v1;
    CHECK_EQ(ha_v1_decode(b, 28, &v1), HA_DEC_ERR_LENGTH);  // v1 receiver rejects
    CHECK_EQ(ha_v1_decode(b, 16, &v1), HA_DEC_ERR_VERSION); // even if truncated
    CHECK_EQ(ha_time_sync_decode(b, 27, &d), HA_DEC_ERR_LENGTH);
    CHECK_EQ(ha_time_sync_decode(b, 29, &d), HA_DEC_ERR_LENGTH);
    uint8_t bad[64];
    memcpy(bad, b, 28); bad[5] = 27;
    CHECK_EQ(ha_time_sync_decode(bad, 28, &d), HA_DEC_ERR_FIELD);
    memcpy(bad, b, 28); bad[7] = 1;
    CHECK_EQ(ha_time_sync_decode(bad, 28, &d), HA_DEC_ERR_FIELD);
    memcpy(bad, b, 28); memset(bad + 12, 0, 4);
    CHECK_EQ(ha_time_sync_decode(bad, 28, &d), HA_DEC_ERR_FIELD);
    ha_time_sync_t early = m;
    early.utc_ms = 1000; // epoch-near: "S3 clock not set"
    ha_time_sync_encode(&early, bad);
    CHECK_EQ(ha_time_sync_decode(bad, 28, &d), HA_DEC_ERR_FIELD);
    memcpy(bad, b, 28); bad[3] = HA_MSG_EVENT;
    CHECK_EQ(ha_time_sync_decode(bad, 28, &d), HA_DEC_ERR_TYPE);
}

/* ------------------------------------------------------------ TX helpers */
#define PEER_IP 0x0204A8C0u /* opaque */
#define PEER_PORT 0x7210u   /* opaque */
#define S3_SESSION 0x12345678u

typedef struct {
    ha_tx_report_t r[64];
    int n;
} reports_t;

static void collect(void *ctx, const ha_tx_report_t *r) {
    reports_t *rs = ctx;
    if (rs->n < 64) rs->r[rs->n++] = *r;
}

static uint32_t fixed_rand(void *ctx) { (void)ctx; return 0xCAFEF00Du; }

static void tx_setup(ha_tx_t *tx, reports_t *rs) {
    ha_tx_config_t cfg;
    ha_tx_default_config(&cfg);
    cfg.peer_port = PEER_PORT;
    cfg.report = collect;
    cfg.report_ctx = rs;
    cfg.rand32 = fixed_rand;
    memset(rs, 0, sizeof(*rs));
    ha_tx_init(tx, &cfg, S3_SESSION);
}

static ha_hello_result_t hello(ha_tx_t *tx, uint32_t c3_session, uint32_t ip, int64_t now) {
    ha_v1_frame_t h = {HA_MSG_HELLO, c3_session, 0, 0};
    return ha_tx_on_hello(tx, &h, ip, PEER_PORT, now);
}

static ha_ack_result_t ack(ha_tx_t *tx, uint32_t session, uint32_t seq, uint32_t id, int64_t now) {
    ha_v1_frame_t a = {HA_MSG_ACK, session, seq, id};
    return ha_tx_on_ack(tx, &a, PEER_IP, PEER_PORT, now);
}

static void test_tx_basic_and_ack_matching(void) {
    ha_tx_t tx; reports_t rs; tx_setup(&tx, &rs);
    uint8_t out[16];
    uint32_t tok;
    CHECK_EQ(ha_tx_publish(&tx, 0, 0, &tok), HA_PUB_ERR_INVALID_ID);
    CHECK_EQ(ha_tx_publish(&tx, HA_EVENT_TEST, 0, &tok), HA_PUB_ERR_INVALID_ID);
    CHECK_EQ(ha_tx_publish(&tx, 0x102, 0, &tok), HA_PUB_ERR_INVALID_ID);
    CHECK_EQ(tx.count, 0);

    // Offline publish: queued, nothing sent, no seq consumed.
    CHECK_EQ(ha_tx_publish(&tx, HA_EVENT_FIRE_ALARM, 0, &tok), HA_PUB_OK);
    CHECK(tok != 0);
    ha_tx_action_t a = ha_tx_poll(&tx, 10, out);
    CHECK(!a.send);
    CHECK_EQ(tx.seq, 0);

    CHECK_EQ(hello(&tx, 77, PEER_IP, 100), HA_HELLO_NEW_PEER);
    a = ha_tx_poll(&tx, 100, out);
    CHECK(a.send && a.seq == 1);
    uint8_t want[32]; hex_to_bytes(V_FIRE_EVENT, want);
    CHECK(memcmp(out, want, 16) == 0); // exact doc vector on the wire

    // Wrong ACKs never clear the in-flight event.
    CHECK_EQ(ack(&tx, S3_SESSION + 1, 1, HA_EVENT_FIRE_ALARM, 150), HA_ACK_IGNORED_MISMATCH);
    CHECK_EQ(ack(&tx, S3_SESSION, 2, HA_EVENT_FIRE_ALARM, 150), HA_ACK_IGNORED_MISMATCH);
    CHECK_EQ(ack(&tx, S3_SESSION, 1, HA_EVENT_DOORBELL, 150), HA_ACK_IGNORED_MISMATCH);
    ha_v1_frame_t good = {HA_MSG_ACK, S3_SESSION, 1, HA_EVENT_FIRE_ALARM};
    CHECK_EQ(ha_tx_on_ack(&tx, &good, PEER_IP + 1, PEER_PORT, 150), HA_ACK_IGNORED_ADDR);
    CHECK_EQ(ha_tx_on_ack(&tx, &good, PEER_IP, PEER_PORT + 1, 150), HA_ACK_IGNORED_ADDR);
    CHECK_EQ(tx.count, 1);
    CHECK(tx.inflight);

    CHECK_EQ(ack(&tx, S3_SESSION, 1, HA_EVENT_FIRE_ALARM, 160), HA_ACK_MATCHED);
    CHECK_EQ(tx.count, 0);
    CHECK_EQ(rs.n, 1);
    CHECK(rs.r[0].outcome == HA_TXR_ACKED && rs.r[0].seq == 1 && rs.r[0].token == tok);
    CHECK_EQ(rs.r[0].ack_after_last_tx_ms, 60);
    // Late duplicate ACK after completion is ignored.
    CHECK_EQ(ack(&tx, S3_SESSION, 1, HA_EVENT_FIRE_ALARM, 170), HA_ACK_IGNORED_NOT_INFLIGHT);
}

static void test_tx_retransmit_and_backoff(void) {
    ha_tx_t tx; reports_t rs; tx_setup(&tx, &rs);
    uint8_t out[16], first[16];
    hello(&tx, 1, PEER_IP, 0);
    ha_tx_publish(&tx, HA_EVENT_DOORBELL, 0, NULL);
    ha_tx_action_t a = ha_tx_poll(&tx, 0, first);
    CHECK(a.send && a.burst_attempt == 1);
    // Not before ACK wait.
    a = ha_tx_poll(&tx, HA_TX_ACK_WAIT_MS - 1, out);
    CHECK(!a.send);
    int64_t t = HA_TX_ACK_WAIT_MS;
    for (uint32_t k = 2; k <= HA_TX_ATTEMPTS_PER_BURST; ++k) {
        a = ha_tx_poll(&tx, t, out);
        CHECK(a.send && a.burst_attempt == k);
        CHECK(memcmp(out, first, 16) == 0); // same session, seq, id
        t += HA_TX_ACK_WAIT_MS;
    }
    // After the last attempt: ACK wait + backoff (2 s) before the next burst.
    a = ha_tx_poll(&tx, t + HA_TX_BACKOFF_MIN_MS - 1, out);
    CHECK(!a.send);
    a = ha_tx_poll(&tx, t + HA_TX_BACKOFF_MIN_MS, out);
    CHECK(a.send && a.new_burst && a.burst_attempt == 1);
    CHECK(memcmp(out, first, 16) == 0);
    CHECK_EQ(tx.stats.bursts_failed, 1);
    CHECK_EQ(tx.backoff_ms, HA_TX_BACKOFF_MIN_MS * 2);
    // Backoff caps at max.
    for (int i = 0; i < 20; ++i) {
        int64_t due = ha_tx_ms_until_due(&tx, t);
        t += due;
        ha_tx_poll(&tx, t, out);
    }
    CHECK(tx.backoff_ms == HA_TX_BACKOFF_MAX_MS);
    // ACK after many attempts still matches the same identity.
    CHECK_EQ(ack(&tx, S3_SESSION, 1, HA_EVENT_DOORBELL, t + 10), HA_ACK_MATCHED);
    CHECK(rs.r[0].attempts_total > HA_TX_ATTEMPTS_PER_BURST);
}

static void test_tx_simultaneous_batch_and_queue_during_ack(void) {
    ha_tx_t tx; reports_t rs; tx_setup(&tx, &rs);
    uint8_t out[16];
    hello(&tx, 1, PEER_IP, 0);
    uint32_t ids[2] = {HA_EVENT_DOORBELL, HA_EVENT_FIRE_ALARM}; // given in "wrong" order
    uint32_t toks[2]; ha_pub_status_t st[2];
    CHECK_EQ(ha_tx_publish_batch(&tx, ids, 2, 0, toks, st), 2);
    CHECK(st[0] == HA_PUB_OK && st[1] == HA_PUB_OK);
    CHECK(toks[1] < toks[0]); // FIRE queued first
    ha_tx_action_t a = ha_tx_poll(&tx, 0, out);
    CHECK(a.send && a.event_id == HA_EVENT_FIRE_ALARM && a.seq == 1);
    // A new detection while waiting for the first ACK: queued, in-flight untouched.
    CHECK_EQ(ha_tx_publish(&tx, HA_EVENT_FIRE_ALARM, 50, NULL), HA_PUB_OK);
    CHECK(tx.q[0].seq == 1 && tx.q[0].event_id == HA_EVENT_FIRE_ALARM);
    CHECK_EQ(tx.count, 3);
    // Stop-and-wait: DOORBELL is not sent before FIRE's ACK.
    a = ha_tx_poll(&tx, 60, out);
    CHECK(!a.send);
    CHECK_EQ(ack(&tx, S3_SESSION, 1, HA_EVENT_FIRE_ALARM, 70), HA_ACK_MATCHED);
    a = ha_tx_poll(&tx, 70, out);
    CHECK(a.send && a.event_id == HA_EVENT_DOORBELL && a.seq == 2);
    uint8_t want[32]; hex_to_bytes(V_DOOR_EVENT, want);
    CHECK(memcmp(out, want, 16) == 0);
    CHECK_EQ(ack(&tx, S3_SESSION, 2, HA_EVENT_DOORBELL, 80), HA_ACK_MATCHED);
    a = ha_tx_poll(&tx, 80, out);
    CHECK(a.send && a.event_id == HA_EVENT_FIRE_ALARM && a.seq == 3);
}

static void test_tx_saturation_policy(void) {
    ha_tx_t tx; reports_t rs; tx_setup(&tx, &rs);
    uint8_t out[16];
    hello(&tx, 1, PEER_IP, 0);
    const uint32_t cap = tx.cfg.capacity;
    CHECK_EQ(cap, 8);

    // Case 1: all doorbells -> new doorbell evicts the oldest non-in-flight.
    uint32_t tok[16];
    for (uint32_t i = 0; i < cap; ++i)
        CHECK_EQ(ha_tx_publish(&tx, HA_EVENT_DOORBELL, i, &tok[i]), HA_PUB_OK);
    ha_tx_poll(&tx, 10, out); // tok[0] in flight
    CHECK_EQ(ha_tx_publish(&tx, HA_EVENT_DOORBELL, 20, &tok[8]), HA_PUB_OK_EVICTED);
    CHECK(rs.n == 1 && rs.r[0].outcome == HA_TXR_EVICTED && rs.r[0].token == tok[1]);
    CHECK_EQ(tx.q[0].token, tok[0]); // in-flight never evicted
    CHECK_EQ(tx.count, cap);

    // Case 2: new FIRE evicts the oldest DOORBELL (not a FIRE).
    tx_setup(&tx, &rs); hello(&tx, 1, PEER_IP, 0);
    ha_tx_publish(&tx, HA_EVENT_FIRE_ALARM, 0, &tok[0]);
    ha_tx_poll(&tx, 0, out);
    ha_tx_publish(&tx, HA_EVENT_FIRE_ALARM, 1, &tok[1]);
    ha_tx_publish(&tx, HA_EVENT_DOORBELL, 2, &tok[2]);
    for (uint32_t i = 3; i < cap; ++i) ha_tx_publish(&tx, HA_EVENT_FIRE_ALARM, i, &tok[i]);
    CHECK_EQ(ha_tx_publish(&tx, HA_EVENT_FIRE_ALARM, 20, &tok[8]), HA_PUB_OK_EVICTED);
    CHECK(rs.r[0].token == tok[2] && rs.r[0].event_id == HA_EVENT_DOORBELL);

    // Case 3: queue full of FIRE -> DOORBELL rejected, FIRE evicts oldest FIRE.
    CHECK_EQ(ha_tx_publish(&tx, HA_EVENT_DOORBELL, 21, NULL), HA_PUB_ERR_QUEUE_FULL);
    CHECK_EQ(tx.stats.rejected_full, 1);
    CHECK_EQ(ha_tx_publish(&tx, HA_EVENT_FIRE_ALARM, 22, NULL), HA_PUB_OK_EVICTED);
    CHECK(rs.r[1].token == tok[1]); // oldest non-in-flight fire
    CHECK_EQ(tx.q[0].token, tok[0]);

    // Case 4: capacity 1 with in-flight -> nothing evictable.
    ha_tx_config_t c = tx.cfg; c.capacity = 1;
    ha_tx_init(&tx, &c, S3_SESSION); hello(&tx, 1, PEER_IP, 0);
    ha_tx_publish(&tx, HA_EVENT_DOORBELL, 0, NULL);
    ha_tx_poll(&tx, 0, out);
    CHECK_EQ(ha_tx_publish(&tx, HA_EVENT_FIRE_ALARM, 1, NULL), HA_PUB_ERR_QUEUE_FULL);
}

static void test_tx_disconnect_reregister_and_c3_reboot(void) {
    ha_tx_t tx; reports_t rs; tx_setup(&tx, &rs);
    uint8_t out[16];
    hello(&tx, 0xAAAA, PEER_IP, 0);
    ha_tx_publish(&tx, HA_EVENT_FIRE_ALARM, 0, NULL);
    CHECK(ha_tx_poll(&tx, 0, out).send);
    ha_tx_on_station_left(&tx, 100);
    // Long disconnection: no transmissions, queue keeps accepting.
    for (int64_t t = 100; t < 30000; t += 500) CHECK(!ha_tx_poll(&tx, t, out).send);
    CHECK_EQ(ha_tx_publish(&tx, HA_EVENT_DOORBELL, 1000, NULL), HA_PUB_OK);
    // C3 rejoins with a new IP and a new boot session (C3 rebooted).
    CHECK_EQ(hello(&tx, 0xBBBB, PEER_IP + 0x01000000u, 30000), HA_HELLO_ADDR_CHANGED);
    ha_tx_action_t a = ha_tx_poll(&tx, 30000, out);
    CHECK(a.send && a.seq == 1 && a.event_id == HA_EVENT_FIRE_ALARM); // same identity
    CHECK_EQ(tx.session, S3_SESSION); // S3 session unchanged
    CHECK_EQ(tx.count, 2);
    // ACK from the old address is ignored now.
    CHECK_EQ(ack(&tx, S3_SESSION, 1, HA_EVENT_FIRE_ALARM, 30010), HA_ACK_IGNORED_ADDR);

    // Same address, new session -> C3_REBOOTED, burst restarted immediately.
    tx_setup(&tx, &rs);
    hello(&tx, 1, PEER_IP, 0);
    ha_tx_publish(&tx, HA_EVENT_DOORBELL, 0, NULL);
    for (int64_t t = 0; t <= 2800; t += 700) ha_tx_poll(&tx, t, out); // burst exhausted
    CHECK(!ha_tx_poll(&tx, 3000, out).send); // in backoff
    CHECK_EQ(hello(&tx, 2, PEER_IP, 3000), HA_HELLO_C3_REBOOTED);
    a = ha_tx_poll(&tx, 3000, out);
    CHECK(a.send && a.seq == 1 && a.burst_attempt == 1);
}

static void test_tx_stale_peer_keeps_retrying(void) {
    ha_tx_t tx; reports_t rs; tx_setup(&tx, &rs);
    uint8_t out[16];
    hello(&tx, 1, PEER_IP, 0);
    ha_tx_publish(&tx, HA_EVENT_FIRE_ALARM, 0, NULL);
    bool went_stale = false;
    int sends_after_stale = 0;
    for (int64_t t = 0; t < 120000; t += 100) {
        ha_tx_action_t a = ha_tx_poll(&tx, t, out);
        if (a.peer_went_stale) { went_stale = true; CHECK(t > HA_PEER_STALE_MS); }
        if (went_stale && a.send) ++sends_after_stale;
    }
    CHECK(went_stale);
    CHECK(sends_after_stale > 0);
    CHECK_EQ(tx.count, 1); // fire TTL is 10 min: still queued after 2 min
    // HELLO after stale -> immediate retry.
    CHECK_EQ(hello(&tx, 1, PEER_IP, 120000), HA_HELLO_REFRESH);
    CHECK(ha_tx_poll(&tx, 120000, out).send);
}

static void test_tx_ttl_and_cancel(void) {
    ha_tx_t tx; reports_t rs; tx_setup(&tx, &rs);
    uint8_t out[16];
    uint32_t door, fire, fire2;
    ha_tx_publish(&tx, HA_EVENT_DOORBELL, 0, &door);
    ha_tx_publish(&tx, HA_EVENT_FIRE_ALARM, 0, &fire);
    ha_tx_poll(&tx, HA_TTL_DOORBELL_MS - 1, out);
    CHECK_EQ(tx.count, 2);
    ha_tx_poll(&tx, HA_TTL_DOORBELL_MS, out);
    CHECK_EQ(tx.count, 1);
    CHECK(rs.r[0].outcome == HA_TXR_EXPIRED && rs.r[0].token == door && rs.r[0].seq == 0);
    ha_tx_poll(&tx, HA_TTL_FIRE_ALARM_MS, out);
    CHECK_EQ(tx.count, 0);
    CHECK(rs.r[1].outcome == HA_TXR_EXPIRED && rs.r[1].token == fire);

    // In-flight expiry: next event gets the next seq (C3 dedup stays valid).
    tx_setup(&tx, &rs);
    hello(&tx, 1, PEER_IP, 0);
    ha_tx_publish(&tx, HA_EVENT_DOORBELL, 0, &door);
    ha_tx_poll(&tx, 0, out);
    CHECK_EQ(tx.q[0].seq, 1);
    for (int64_t t = 0; t < HA_TTL_DOORBELL_MS; t += 1000) {
        hello(&tx, 1, PEER_IP, t);
        ha_tx_poll(&tx, t, out);
    }
    ha_tx_publish(&tx, HA_EVENT_FIRE_ALARM, HA_TTL_DOORBELL_MS - 1, &fire);
    ha_tx_action_t a = ha_tx_poll(&tx, HA_TTL_DOORBELL_MS, out);
    CHECK(rs.n >= 1 && rs.r[0].outcome == HA_TXR_EXPIRED && rs.r[0].seq == 1);
    CHECK(a.send && a.seq == 2 && a.event_id == HA_EVENT_FIRE_ALARM);

    // Cancel queued and in-flight.
    ha_tx_publish(&tx, HA_EVENT_DOORBELL, HA_TTL_DOORBELL_MS, &fire2);
    CHECK(ha_tx_cancel(&tx, fire2, HA_TTL_DOORBELL_MS));
    CHECK(ha_tx_cancel(&tx, fire, HA_TTL_DOORBELL_MS));
    CHECK(!ha_tx_cancel(&tx, fire, HA_TTL_DOORBELL_MS));
    CHECK(!tx.inflight && tx.count == 0);
}

static void test_tx_seq_exhaustion(void) {
    ha_tx_t tx; reports_t rs; tx_setup(&tx, &rs);
    uint8_t out[16];
    hello(&tx, 1, PEER_IP, 0);
    tx.seq = UINT32_MAX - 1;
    ha_tx_publish(&tx, HA_EVENT_DOORBELL, 0, NULL);
    ha_tx_publish(&tx, HA_EVENT_DOORBELL, 0, NULL);
    ha_tx_action_t a = ha_tx_poll(&tx, 0, out);
    CHECK(a.seq == UINT32_MAX && tx.session == S3_SESSION);
    ack(&tx, S3_SESSION, UINT32_MAX, HA_EVENT_DOORBELL, 1);
    a = ha_tx_poll(&tx, 1, out);
    CHECK(a.send && a.seq == 1 && tx.session == 0xCAFEF00Du);
    ha_v1_frame_t f; ha_v1_decode(out, 16, &f);
    CHECK(f.session == 0xCAFEF00Du && f.seq == 1);
}

static void test_tx_fire_first_option(void) {
    ha_tx_t tx; reports_t rs; tx_setup(&tx, &rs);
    uint8_t out[16];
    ha_tx_config_t c = tx.cfg; c.fire_first = true;
    ha_tx_init(&tx, &c, S3_SESSION);
    hello(&tx, 1, PEER_IP, 0);
    ha_tx_publish(&tx, HA_EVENT_DOORBELL, 0, NULL);
    ha_tx_poll(&tx, 0, out); // doorbell already in flight: not preempted
    ha_tx_publish(&tx, HA_EVENT_DOORBELL, 1, NULL);
    ha_tx_publish(&tx, HA_EVENT_FIRE_ALARM, 2, NULL);
    CHECK_EQ(ack(&tx, S3_SESSION, 1, HA_EVENT_DOORBELL, 3), HA_ACK_MATCHED);
    ha_tx_action_t a = ha_tx_poll(&tx, 3, out);
    CHECK(a.event_id == HA_EVENT_FIRE_ALARM && a.seq == 2); // seq follows send order
    ack(&tx, S3_SESSION, 2, HA_EVENT_FIRE_ALARM, 4);
    a = ha_tx_poll(&tx, 4, out);
    CHECK(a.event_id == HA_EVENT_DOORBELL && a.seq == 3);
}

/* ------------------------------------------------------------- RX tests */
typedef struct {
    bool ok;
    int calls;
    uint32_t last_id;
} consumer_t;

static bool accept_fn(void *ctx, const ha_v1_frame_t *f) {
    consumer_t *c = ctx;
    ++c->calls;
    if (!c->ok) return false;
    c->last_id = f->value;
    return true;
}

static void test_rx_dedup_and_accept_contract(void) {
    ha_rx_t rx; ha_rx_init(&rx, false);
    consumer_t c = {true, 0, 0};
    ha_v1_frame_t fire = {HA_MSG_EVENT, S3_SESSION, 1, HA_EVENT_FIRE_ALARM};

    // Consumer not ready -> no ACK, no record; retransmission accepted later.
    c.ok = false;
    CHECK_EQ(ha_rx_handle_event(&rx, &fire, accept_fn, &c), HA_RX_DROP_NOT_ACCEPTED);
    CHECK_EQ(ha_rx_handle_event(&rx, &fire, NULL, NULL), HA_RX_DROP_NOT_ACCEPTED);
    CHECK(!ha_rx_should_ack(HA_RX_DROP_NOT_ACCEPTED));
    c.ok = true;
    CHECK_EQ(ha_rx_handle_event(&rx, &fire, accept_fn, &c), HA_RX_ACK_NEW);
    CHECK_EQ(c.last_id, HA_EVENT_FIRE_ALARM);
    int calls = c.calls;
    // Lost ACK: same frame again -> ACK again, consumer NOT called.
    CHECK_EQ(ha_rx_handle_event(&rx, &fire, accept_fn, &c), HA_RX_ACK_DUPLICATE);
    CHECK_EQ(c.calls, calls);
    // Same session+seq with another id -> conflict, no ACK.
    ha_v1_frame_t conflict = fire; conflict.value = HA_EVENT_DOORBELL;
    CHECK_EQ(ha_rx_handle_event(&rx, &conflict, accept_fn, &c), HA_RX_DROP_CONFLICT);
    CHECK(!ha_rx_should_ack(HA_RX_DROP_CONFLICT));
    CHECK_EQ(c.calls, calls);

    // Invalid: seq 0, reserved, TEST, unregistered, wrong type.
    ha_v1_frame_t bad = fire; bad.seq = 0;
    CHECK_EQ(ha_rx_handle_event(&rx, &bad, accept_fn, &c), HA_RX_DROP_INVALID);
    bad = fire; bad.seq = 9; bad.value = 0;
    CHECK_EQ(ha_rx_handle_event(&rx, &bad, accept_fn, &c), HA_RX_DROP_INVALID);
    bad.value = HA_EVENT_TEST;
    CHECK_EQ(ha_rx_handle_event(&rx, &bad, accept_fn, &c), HA_RX_DROP_INVALID);
    bad.value = 0x102;
    CHECK_EQ(ha_rx_handle_event(&rx, &bad, accept_fn, &c), HA_RX_DROP_INVALID);
    bad = fire; bad.seq = 9; bad.type = HA_MSG_ACK;
    CHECK_EQ(ha_rx_handle_event(&rx, &bad, accept_fn, &c), HA_RX_DROP_INVALID);
    CHECK_EQ(c.calls, calls);
    // An invalid frame did not change state: seq 9 is still acceptable.
    ha_v1_frame_t d9 = {HA_MSG_EVENT, S3_SESSION, 9, HA_EVENT_DOORBELL};
    CHECK_EQ(ha_rx_handle_event(&rx, &d9, accept_fn, &c), HA_RX_ACK_NEW);

    // Older seq in the same session after a newer one -> stale ACK, no alert.
    calls = c.calls;
    ha_v1_frame_t old = {HA_MSG_EVENT, S3_SESSION, 5, HA_EVENT_DOORBELL};
    CHECK_EQ(ha_rx_handle_event(&rx, &old, accept_fn, &c), HA_RX_ACK_STALE);
    CHECK_EQ(c.calls, calls);

    // TEST accepted only in diagnostic mode.
    ha_rx_t diag; ha_rx_init(&diag, true);
    ha_v1_frame_t t = {HA_MSG_EVENT, 1, 1, HA_EVENT_TEST};
    CHECK_EQ(ha_rx_handle_event(&diag, &t, accept_fn, &c), HA_RX_ACK_NEW);
}

static void test_rx_session_changes(void) {
    ha_rx_t rx; ha_rx_init(&rx, false);
    consumer_t c = {true, 0, 0};
    ha_v1_frame_t a5 = {HA_MSG_EVENT, 0xA, 5, HA_EVENT_FIRE_ALARM};
    CHECK_EQ(ha_rx_handle_event(&rx, &a5, accept_fn, &c), HA_RX_ACK_NEW);
    // S3 rebooted: new session starting at seq 1 is accepted.
    ha_v1_frame_t b1 = {HA_MSG_EVENT, 0xB, 1, HA_EVENT_DOORBELL};
    CHECK_EQ(ha_rx_handle_event(&rx, &b1, accept_fn, &c), HA_RX_ACK_NEW);
    // Delayed copy from the previous S3 session: duplicate, not a new alert.
    int calls = c.calls;
    CHECK_EQ(ha_rx_handle_event(&rx, &a5, accept_fn, &c), HA_RX_ACK_DUPLICATE);
    ha_v1_frame_t a4 = {HA_MSG_EVENT, 0xA, 4, HA_EVENT_FIRE_ALARM};
    CHECK_EQ(ha_rx_handle_event(&rx, &a4, accept_fn, &c), HA_RX_ACK_STALE);
    CHECK_EQ(c.calls, calls);

    // C3 reboot (fresh RAM state): the same event is accepted again.
    // Documented limitation: at-least-once across C3 reboots.
    ha_rx_t rebooted; ha_rx_init(&rebooted, false);
    CHECK_EQ(ha_rx_handle_event(&rebooted, &b1, accept_fn, &c), HA_RX_ACK_NEW);
}

/* --------------------------------------------- end-to-end over a lossy link */
static void test_end_to_end_ack_loss(void) {
    ha_tx_t tx; reports_t rs; tx_setup(&tx, &rs);
    ha_rx_t rx; ha_rx_init(&rx, false);
    consumer_t c = {true, 0, 0};
    uint32_t accepted_ids[16]; int n_acc = 0;
    hello(&tx, 1, PEER_IP, 0);
    uint32_t ids[2] = {HA_EVENT_FIRE_ALARM, HA_EVENT_DOORBELL};
    ha_tx_publish_batch(&tx, ids, 2, 0, NULL, NULL);
    int ack_drops = 3; // drop the first three ACKs
    int c_calls_before = 0;
    for (int64_t t = 0; t < 60000 && tx.count; t += 10) {
        uint8_t out[16];
        ha_tx_action_t a = ha_tx_poll(&tx, t, out);
        if (!a.send) continue;
        ha_v1_frame_t f; CHECK_EQ(ha_v1_decode(out, 16, &f), HA_DEC_OK);
        c_calls_before = c.calls;
        ha_rx_outcome_t o = ha_rx_handle_event(&rx, &f, accept_fn, &c);
        if (o == HA_RX_ACK_NEW) accepted_ids[n_acc++] = f.value;
        (void)c_calls_before;
        if (!ha_rx_should_ack(o)) continue;
        if (ack_drops > 0) { --ack_drops; continue; }
        ha_v1_frame_t ackf = {HA_MSG_ACK, f.session, f.seq, f.value};
        uint8_t wire[16]; ha_v1_encode(&ackf, wire);
        ha_v1_frame_t back; ha_v1_decode(wire, 16, &back);
        CHECK_EQ(ha_tx_on_ack(&tx, &back, PEER_IP, PEER_PORT, t + 5), HA_ACK_MATCHED);
    }
    CHECK_EQ(tx.count, 0);
    CHECK_EQ(n_acc, 2); // each event delivered to the consumer exactly once
    CHECK(accepted_ids[0] == HA_EVENT_FIRE_ALARM && accepted_ids[1] == HA_EVENT_DOORBELL);
    CHECK_EQ(rx.stats.duplicates, 3);
    CHECK_EQ(tx.stats.acked, 2);
}

/* ------------------------------------------------------------ time tests */
static void test_clock(void) {
    ha_clock_config_t cfg; ha_clock_default_config(&cfg);
    ha_clock_t c; ha_clock_init(&c, &cfg);
    int64_t utc = 0; uint32_t unc = 0;
    CHECK_EQ(ha_clock_now(&c, 5000, &utc, &unc), HA_CLOCK_UNSYNCED);
    CHECK_EQ(utc, 0); // never invents a time

    ha_time_sync_t m = {0, 0xA, 1, 1763164800000LL, 200};
    CHECK_EQ(ha_clock_apply(&c, &m, 10000), HA_TIME_APPLIED);
    CHECK_EQ(ha_clock_now(&c, 10000, &utc, &unc), HA_CLOCK_SYNCED);
    CHECK_EQ(utc, 1763164800000LL);
    CHECK_EQ(unc, 200 + HA_TIME_DELIVERY_BOUND_MS);
    // One hour later: monotonic elapsed added, uncertainty grows with drift.
    CHECK_EQ(ha_clock_now(&c, 10000 + 3600000, &utc, &unc), HA_CLOCK_SYNCED);
    CHECK_EQ(utc, 1763164800000LL + 3600000);
    CHECK_EQ(unc, 200 + HA_TIME_DELIVERY_BOUND_MS + 3600000LL * HA_TIME_DRIFT_PPM / 1000000);
    // Delayed/duplicate message of the same session is ignored.
    ha_time_sync_t old = m; old.utc_ms -= 5000;
    CHECK_EQ(ha_clock_apply(&c, &old, 20000), HA_TIME_IGNORED_OLD);
    // New S3 session accepted.
    ha_time_sync_t s2 = {0, 0xB, 1, 1763168400000LL, 50};
    CHECK_EQ(ha_clock_apply(&c, &s2, 20000), HA_TIME_APPLIED);
    // Stale after the configured period.
    CHECK_EQ(ha_clock_now(&c, 20000 + HA_TIME_STALE_AFTER_MS + 1, &utc, &unc), HA_CLOCK_STALE);
}

static void test_time_sync_does_not_touch_tx_timers(void) {
    // Two identical senders; one gets clock updates interleaved. Schedules
    // must be byte-identical because ha_tx only uses the monotonic clock.
    ha_tx_t a, b; reports_t ra, rb;
    tx_setup(&a, &ra); tx_setup(&b, &rb);
    ha_clock_config_t cfg; ha_clock_default_config(&cfg);
    ha_clock_t clk; ha_clock_init(&clk, &cfg);
    hello(&a, 1, PEER_IP, 0); hello(&b, 1, PEER_IP, 0);
    ha_tx_publish(&a, HA_EVENT_FIRE_ALARM, 0, NULL);
    ha_tx_publish(&b, HA_EVENT_FIRE_ALARM, 0, NULL);
    uint32_t tseq = 0;
    for (int64_t t = 0; t < 20000; t += 50) {
        if (t % 1000 == 0) {
            ha_time_sync_t m = {0, 0xA, ++tseq, 1763164800000LL + (int64_t)tseq * 7777, 10};
            ha_clock_apply(&clk, &m, t); // big UTC jumps
        }
        uint8_t oa[16], ob[16];
        ha_tx_action_t xa = ha_tx_poll(&a, t, oa), xb = ha_tx_poll(&b, t, ob);
        CHECK(xa.send == xb.send);
        if (xa.send) CHECK(memcmp(oa, ob, 16) == 0);
    }
    CHECK(a.next_send_ms == b.next_send_ms && a.q[0].deadline_ms == b.q[0].deadline_ms);
}

static void test_registry(void) {
    CHECK_EQ(ha_event_count(), 3);
    CHECK(ha_event_batch_rank(HA_EVENT_FIRE_ALARM) < ha_event_batch_rank(HA_EVENT_DOORBELL));
    // IDs unique, never 0.
    for (size_t i = 0; i < ha_event_count(); ++i) {
        CHECK(ha_event_at(i)->id != 0);
        for (size_t j = i + 1; j < ha_event_count(); ++j)
            CHECK(ha_event_at(i)->id != ha_event_at(j)->id);
    }
    CHECK_EQ(ha_event_find(HA_EVENT_FIRE_ALARM)->ttl_ms, 600000);
    CHECK_EQ(ha_event_find(HA_EVENT_DOORBELL)->ttl_ms, 120000);
    CHECK(ha_event_find(HA_EVENT_FIRE_ALARM)->evict_protected);
    CHECK(!ha_event_find(HA_EVENT_TEST)->product);
    CHECK_EQ(HA_NOMINAL_WAKE_US, 307200);
}

int main(void) {
    test_doc_vectors();
    test_error_vectors();
    test_time_sync_codec();
    test_registry();
    test_tx_basic_and_ack_matching();
    test_tx_retransmit_and_backoff();
    test_tx_simultaneous_batch_and_queue_during_ack();
    test_tx_saturation_policy();
    test_tx_disconnect_reregister_and_c3_reboot();
    test_tx_stale_peer_keeps_retrying();
    test_tx_ttl_and_cancel();
    test_tx_seq_exhaustion();
    test_tx_fire_first_option();
    test_rx_dedup_and_accept_contract();
    test_rx_session_changes();
    test_end_to_end_ack_loss();
    test_clock();
    test_time_sync_does_not_touch_tx_timers();
    printf("%s: %d checks, %d failed\n", g_fail ? "HA_COMM_HOST_TESTS_FAIL" : "HA_COMM_HOST_TESTS_PASS",
           g_checks, g_fail);
    return g_fail ? 1 : 0;
}
