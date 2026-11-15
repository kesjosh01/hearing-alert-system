#include "ha_tx.h"

#include <string.h>

#include "ha_config.h"
#include "ha_events.h"

#define NO_DEADLINE INT64_MAX

void ha_tx_default_config(ha_tx_config_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->capacity = HA_TX_QUEUE_CAPACITY;
    cfg->ack_wait_ms = HA_TX_ACK_WAIT_MS;
    cfg->attempts_per_burst = HA_TX_ATTEMPTS_PER_BURST;
    cfg->backoff_min_ms = HA_TX_BACKOFF_MIN_MS;
    cfg->backoff_max_ms = HA_TX_BACKOFF_MAX_MS;
    cfg->peer_stale_ms = HA_PEER_STALE_MS;
    cfg->allow_test = HA_ALLOW_TEST_EVENT;
}

void ha_tx_init(ha_tx_t *tx, const ha_tx_config_t *cfg, uint32_t session) {
    memset(tx, 0, sizeof(*tx));
    tx->cfg = *cfg;
    if (tx->cfg.capacity == 0) tx->cfg.capacity = 1;
    if (tx->cfg.capacity > HA_TX_QUEUE_MAX) tx->cfg.capacity = HA_TX_QUEUE_MAX;
    if (tx->cfg.attempts_per_burst == 0) tx->cfg.attempts_per_burst = 1;
    if (tx->cfg.backoff_max_ms < tx->cfg.backoff_min_ms)
        tx->cfg.backoff_max_ms = tx->cfg.backoff_min_ms;
    tx->session = session;
    tx->next_token = 1;
    tx->backoff_ms = tx->cfg.backoff_min_ms;
}

static void report(ha_tx_t *tx, ha_tx_outcome_t outcome, const ha_tx_entry_t *e,
                   int64_t now_ms, int64_t ack_after_tx) {
    if (!tx->cfg.report) return;
    ha_tx_report_t r = {
        .outcome = outcome,
        .token = e->token,
        .event_id = e->event_id,
        .session = e->seq ? tx->session : 0,
        .seq = e->seq,
        .attempts_total = e->attempts_total,
        .age_ms = now_ms - e->enq_ms,
        .ack_after_last_tx_ms = ack_after_tx,
    };
    tx->cfg.report(tx->cfg.report_ctx, &r);
}

/* Remove index i, keeping FIFO order. Clears in-flight if i == 0. */
static void remove_at(ha_tx_t *tx, uint32_t i) {
    if (i == 0 && tx->inflight) tx->inflight = false;
    for (uint32_t k = i + 1; k < tx->count; ++k) tx->q[k - 1] = tx->q[k];
    --tx->count;
    memset(&tx->q[tx->count], 0, sizeof(tx->q[0]));
}

static bool is_protected(uint32_t id) {
    const ha_event_def_t *def = ha_event_find(id);
    return def && def->evict_protected;
}

/* Saturation policy (user decision 2026-11-15): evict the oldest queued event,
 * but protected events (FIRE_ALARM) are only evicted by another protected
 * event. The in-flight event is never evicted. Returns index or -1. */
static int pick_victim(const ha_tx_t *tx, uint32_t new_id) {
    uint32_t first = tx->inflight ? 1u : 0u;
    for (uint32_t i = first; i < tx->count; ++i)
        if (!is_protected(tx->q[i].event_id)) return (int)i;
    if (is_protected(new_id) && first < tx->count) return (int)first;
    return -1;
}

ha_pub_status_t ha_tx_publish(ha_tx_t *tx, uint32_t event_id, int64_t now_ms, uint32_t *token) {
    if (token) *token = 0;
    if (!ha_event_allowed(event_id, tx->cfg.allow_test)) {
        ++tx->stats.rejected_invalid;
        return HA_PUB_ERR_INVALID_ID;
    }
    ha_pub_status_t status = HA_PUB_OK;
    if (tx->count >= tx->cfg.capacity) {
        int victim = pick_victim(tx, event_id);
        if (victim < 0) {
            ++tx->stats.rejected_full;
            return HA_PUB_ERR_QUEUE_FULL;
        }
        ha_tx_entry_t gone = tx->q[victim];
        remove_at(tx, (uint32_t)victim);
        ++tx->stats.evicted;
        report(tx, HA_TXR_EVICTED, &gone, now_ms, -1);
        status = HA_PUB_OK_EVICTED;
    }
    const ha_event_def_t *def = ha_event_find(event_id);
    ha_tx_entry_t *e = &tx->q[tx->count++];
    memset(e, 0, sizeof(*e));
    e->token = tx->next_token++;
    if (tx->next_token == 0) tx->next_token = 1;
    e->event_id = event_id;
    e->enq_ms = now_ms;
    e->deadline_ms = def->ttl_ms ? now_ms + (int64_t)def->ttl_ms : NO_DEADLINE;
    ++tx->stats.published;
    if (token) *token = e->token;
    return status;
}

size_t ha_tx_publish_batch(ha_tx_t *tx, const uint32_t *ids, size_t n, int64_t now_ms,
                           uint32_t *tokens, ha_pub_status_t *results) {
    /* Stable sort of indices by canonical rank; n is tiny (a few sounds). */
    size_t order[HA_TX_QUEUE_MAX];
    if (n > HA_TX_QUEUE_MAX) n = HA_TX_QUEUE_MAX;
    for (size_t i = 0; i < n; ++i) order[i] = i;
    for (size_t i = 1; i < n; ++i) {
        size_t cur = order[i], j = i;
        while (j > 0 && ha_event_batch_rank(ids[order[j - 1]]) > ha_event_batch_rank(ids[cur])) {
            order[j] = order[j - 1];
            --j;
        }
        order[j] = cur;
    }
    size_t accepted = 0;
    for (size_t k = 0; k < n; ++k) {
        size_t i = order[k];
        uint32_t tok = 0;
        ha_pub_status_t st = ha_tx_publish(tx, ids[i], now_ms, &tok);
        if (tokens) tokens[i] = tok;
        if (results) results[i] = st;
        if (st == HA_PUB_OK || st == HA_PUB_OK_EVICTED) ++accepted;
    }
    return accepted;
}

bool ha_tx_cancel(ha_tx_t *tx, uint32_t token, int64_t now_ms) {
    for (uint32_t i = 0; i < tx->count; ++i) {
        if (tx->q[i].token != token) continue;
        ha_tx_entry_t gone = tx->q[i];
        remove_at(tx, i);
        ++tx->stats.canceled;
        report(tx, HA_TXR_CANCELED, &gone, now_ms, -1);
        return true;
    }
    return false;
}

bool ha_tx_link_up(const ha_tx_t *tx) {
    return tx->peer.registered && tx->peer.associated;
}

bool ha_tx_idle(const ha_tx_t *tx) { return !tx->inflight && tx->count == 0; }

static void expire(ha_tx_t *tx, int64_t now_ms) {
    for (uint32_t i = 0; i < tx->count;) {
        if (tx->q[i].deadline_ms <= now_ms) {
            ha_tx_entry_t gone = tx->q[i];
            remove_at(tx, i);
            ++tx->stats.expired;
            report(tx, HA_TXR_EXPIRED, &gone, now_ms, -1);
        } else {
            ++i;
        }
    }
}

/* Pick the next in-flight entry and assign the next seq. */
static void start_next(ha_tx_t *tx, int64_t now_ms) {
    if (tx->cfg.fire_first) {
        for (uint32_t i = 1; i < tx->count; ++i) {
            if (!is_protected(tx->q[i].event_id) || is_protected(tx->q[0].event_id)) continue;
            ha_tx_entry_t pick = tx->q[i];
            for (uint32_t k = i; k > 0; --k) tx->q[k] = tx->q[k - 1];
            tx->q[0] = pick;
            break;
        }
    }
    if (tx->seq == UINT32_MAX) {
        // Sequence space exhausted: start a new S3 session at seq 1 so C3
        // never sees a wrapped (smaller) seq inside the same session.
        uint32_t s = tx->cfg.rand32 ? tx->cfg.rand32(tx->cfg.rand_ctx) : tx->session + 1u;
        if (s == tx->session) s ^= 0x5A5A5A5Au;
        tx->session = s;
        tx->seq = 0;
    }
    tx->q[0].seq = ++tx->seq;
    tx->inflight = true;
    tx->burst_attempts = 0;
    tx->backoff_ms = tx->cfg.backoff_min_ms;
    tx->next_send_ms = now_ms;
}

ha_tx_action_t ha_tx_poll(ha_tx_t *tx, int64_t now_ms, uint8_t out[HA_V1_LEN]) {
    ha_tx_action_t a = {0};
    expire(tx, now_ms);

    if (tx->peer.registered && !tx->peer.stale &&
        now_ms - tx->peer.last_seen_ms > (int64_t)tx->cfg.peer_stale_ms) {
        tx->peer.stale = true; // keep peer + queue; keep retrying with backoff
        a.peer_went_stale = true;
    }

    bool up = ha_tx_link_up(tx);
    // Do not consume a seq while nobody can receive it.
    if (!tx->inflight && tx->count > 0 && up) start_next(tx, now_ms);
    if (!tx->inflight || !up || now_ms < tx->next_send_ms) return a;

    if (tx->burst_attempts >= tx->cfg.attempts_per_burst) {
        tx->burst_attempts = 0; // previous burst got no ACK; backoff elapsed
        a.new_burst = true;
        ++tx->stats.bursts_failed;
    }
    ha_tx_entry_t *e = &tx->q[0];
    ha_v1_frame_t f = {HA_MSG_EVENT, tx->session, e->seq, e->event_id};
    ha_v1_encode(&f, out);
    ++tx->burst_attempts;
    ++e->attempts_total;
    ++tx->stats.datagrams;
    tx->last_tx_ms = now_ms;
    if (tx->burst_attempts < tx->cfg.attempts_per_burst) {
        tx->next_send_ms = now_ms + tx->cfg.ack_wait_ms;
    } else {
        // Last attempt of the burst: wait for its ACK, then back off.
        tx->next_send_ms = now_ms + tx->cfg.ack_wait_ms + tx->backoff_ms;
        uint32_t next = tx->backoff_ms * 2u;
        tx->backoff_ms = next > tx->cfg.backoff_max_ms ? tx->cfg.backoff_max_ms : next;
    }
    a.send = true;
    a.seq = e->seq;
    a.event_id = e->event_id;
    a.burst_attempt = tx->burst_attempts;
    return a;
}

int64_t ha_tx_ms_until_due(const ha_tx_t *tx, int64_t now_ms) {
    int64_t due = INT64_MAX;
    for (uint32_t i = 0; i < tx->count; ++i)
        if (tx->q[i].deadline_ms < due) due = tx->q[i].deadline_ms;
    if (ha_tx_link_up(tx)) {
        if (!tx->inflight && tx->count > 0) due = now_ms;
        else if (tx->inflight && tx->next_send_ms < due) due = tx->next_send_ms;
    }
    if (tx->peer.registered && !tx->peer.stale) {
        int64_t stale_at = tx->peer.last_seen_ms + (int64_t)tx->cfg.peer_stale_ms + 1;
        if (stale_at < due) due = stale_at;
    }
    if (due == INT64_MAX) return INT64_MAX;
    return due > now_ms ? due - now_ms : 0;
}

static void restart_burst(ha_tx_t *tx, int64_t now_ms) {
    if (!tx->inflight) return;
    tx->burst_attempts = 0;
    tx->backoff_ms = tx->cfg.backoff_min_ms;
    tx->next_send_ms = now_ms;
}

ha_hello_result_t ha_tx_on_hello(ha_tx_t *tx, const ha_v1_frame_t *f, uint32_t ip,
                                 uint16_t port, int64_t now_ms) {
    if (!ha_v1_hello_fields_valid(f) || port != tx->cfg.peer_port) return HA_HELLO_INVALID;
    ha_tx_peer_t *p = &tx->peer;
    ha_hello_result_t r;
    if (!p->registered) r = HA_HELLO_NEW_PEER;
    else if (p->ip != ip || p->port != port) r = HA_HELLO_ADDR_CHANGED;
    else if (p->c3_session != f->session) r = HA_HELLO_C3_REBOOTED;
    else r = HA_HELLO_REFRESH;
    bool was_down = !p->associated || p->stale;
    p->registered = true;
    p->associated = true; // a HELLO proves association
    p->stale = false;
    p->ip = ip;
    p->port = port;
    p->c3_session = f->session;
    p->last_seen_ms = now_ms;
    // Keep the queue and the S3 session; just retry the in-flight event now.
    if (r != HA_HELLO_REFRESH || was_down) restart_burst(tx, now_ms);
    return r;
}

void ha_tx_on_station_joined(ha_tx_t *tx, int64_t now_ms) {
    (void)now_ms;
    // Wait for HELLO: the DHCP address may have changed.
    (void)tx;
}

void ha_tx_on_station_left(ha_tx_t *tx, int64_t now_ms) {
    (void)now_ms;
    tx->peer.associated = false; // pause EVENT transmission; keep queue + seq
}

ha_ack_result_t ha_tx_on_ack(ha_tx_t *tx, const ha_v1_frame_t *f, uint32_t ip, uint16_t port,
                             int64_t now_ms) {
    ha_ack_result_t r;
    if (!tx->peer.registered) r = HA_ACK_IGNORED_NO_PEER;
    else if (ip != tx->peer.ip || port != tx->peer.port) r = HA_ACK_IGNORED_ADDR;
    else if (!tx->inflight) r = HA_ACK_IGNORED_NOT_INFLIGHT;
    else if (f->type != HA_MSG_ACK || f->session != tx->session || f->seq != tx->q[0].seq ||
             f->value != tx->q[0].event_id)
        r = HA_ACK_IGNORED_MISMATCH;
    else r = HA_ACK_MATCHED;
    if (r != HA_ACK_MATCHED) {
        ++tx->stats.acks_ignored;
        return r;
    }
    ha_tx_entry_t done = tx->q[0];
    remove_at(tx, 0);
    tx->peer.last_seen_ms = now_ms;
    tx->peer.stale = false;
    ++tx->stats.acked;
    report(tx, HA_TXR_ACKED, &done, now_ms, now_ms - tx->last_tx_ms);
    return HA_ACK_MATCHED;
}

bool ha_tx_from_peer(const ha_tx_t *tx, const ha_v1_frame_t *f, uint32_t ip, uint16_t port) {
    return tx->peer.registered && ip == tx->peer.ip && port == tx->peer.port &&
           f->session == tx->peer.c3_session;
}

const char *ha_pub_status_name(ha_pub_status_t s) {
    switch (s) {
    case HA_PUB_OK: return "ok";
    case HA_PUB_OK_EVICTED: return "ok_evicted_oldest";
    case HA_PUB_ERR_INVALID_ID: return "invalid_id";
    case HA_PUB_ERR_QUEUE_FULL: return "queue_full";
    }
    return "?";
}

const char *ha_tx_outcome_name(ha_tx_outcome_t o) {
    switch (o) {
    case HA_TXR_ACKED: return "ACKED";
    case HA_TXR_EXPIRED: return "EXPIRED";
    case HA_TXR_EVICTED: return "EVICTED";
    case HA_TXR_CANCELED: return "CANCELED";
    }
    return "?";
}

const char *ha_ack_result_name(ha_ack_result_t r) {
    switch (r) {
    case HA_ACK_MATCHED: return "matched";
    case HA_ACK_IGNORED_NO_PEER: return "no_peer";
    case HA_ACK_IGNORED_ADDR: return "wrong_addr";
    case HA_ACK_IGNORED_NOT_INFLIGHT: return "not_inflight";
    case HA_ACK_IGNORED_MISMATCH: return "field_mismatch";
    }
    return "?";
}

const char *ha_hello_result_name(ha_hello_result_t r) {
    switch (r) {
    case HA_HELLO_INVALID: return "invalid";
    case HA_HELLO_REFRESH: return "refresh";
    case HA_HELLO_NEW_PEER: return "new_peer";
    case HA_HELLO_ADDR_CHANGED: return "addr_changed";
    case HA_HELLO_C3_REBOOTED: return "c3_rebooted";
    }
    return "?";
}
