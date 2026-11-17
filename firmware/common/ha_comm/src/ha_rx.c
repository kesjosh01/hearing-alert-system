#include "ha_rx.h"

#include <string.h>

#include "ha_events.h"

void ha_rx_init(ha_rx_t *rx, bool allow_test) {
    memset(rx, 0, sizeof(*rx));
    rx->allow_test = allow_test;
}

static ha_rx_session_t *find_session(ha_rx_t *rx, uint32_t session) {
    for (int i = 0; i < HA_RX_SESSION_SLOTS; ++i)
        if (rx->sessions[i].used && rx->sessions[i].session == session) return &rx->sessions[i];
    return NULL;
}

static const ha_rx_hist_t *find_hist(const ha_rx_t *rx, uint32_t session, uint32_t seq) {
    for (int i = 0; i < HA_RX_HISTORY_LEN; ++i) {
        const ha_rx_hist_t *h = &rx->hist[i];
        if (h->used && h->session == session && h->seq == seq) return h;
    }
    return NULL;
}

/* Record a successfully accepted event (called only after consumer success). */
static void commit(ha_rx_t *rx, const ha_v1_frame_t *f) {
    ha_rx_session_t *s = find_session(rx, f->session);
    if (!s) {
        // Replace an empty slot, else the least recently used session.
        s = &rx->sessions[0];
        for (int i = 0; i < HA_RX_SESSION_SLOTS; ++i) {
            if (!rx->sessions[i].used) { s = &rx->sessions[i]; break; }
            if (rx->sessions[i].last_use < s->last_use) s = &rx->sessions[i];
        }
        s->used = true;
        s->session = f->session;
        s->max_seq = 0;
    }
    if (f->seq > s->max_seq) s->max_seq = f->seq;
    s->last_use = ++rx->clock;
    ha_rx_hist_t *h = &rx->hist[rx->hist_next];
    rx->hist_next = (rx->hist_next + 1) % HA_RX_HISTORY_LEN;
    h->used = true;
    h->session = f->session;
    h->seq = f->seq;
    h->event_id = f->value;
}

ha_rx_outcome_t ha_rx_handle_event(ha_rx_t *rx, const ha_v1_frame_t *f, ha_rx_accept_fn accept,
                                   void *ctx) {
    if (!ha_v1_event_fields_valid(f) || !ha_event_allowed(f->value, rx->allow_test)) {
        ++rx->stats.rejected_invalid; // no alert, no ACK, no state change
        return HA_RX_DROP_INVALID;
    }
    const ha_rx_hist_t *h = find_hist(rx, f->session, f->seq);
    if (h) {
        if (h->event_id != f->value) {
            ++rx->stats.conflicts; // same (session, seq) but a different sound
            return HA_RX_DROP_CONFLICT;
        }
        ++rx->stats.duplicates; // lost ACK: answer again, do not re-alert
        return HA_RX_ACK_DUPLICATE;
    }
    ha_rx_session_t *s = find_session(rx, f->session);
    if (s && f->seq <= s->max_seq) {
        // Older than something already accepted in that session, and outside
        // the conflict history. The stop-and-wait sender never moves backwards,
        // so this is a delayed retransmission: ACK it, never alert.
        s->last_use = ++rx->clock;
        ++rx->stats.stale;
        return HA_RX_ACK_STALE;
    }
    if (!accept || !accept(ctx, f)) {
        ++rx->stats.accept_failed; // S3 will retransmit; we may accept later
        return HA_RX_DROP_NOT_ACCEPTED;
    }
    commit(rx, f);
    ++rx->stats.accepted;
    return HA_RX_ACK_NEW;
}

const char *ha_rx_outcome_name(ha_rx_outcome_t o) {
    switch (o) {
    case HA_RX_ACK_NEW: return "new";
    case HA_RX_ACK_DUPLICATE: return "duplicate";
    case HA_RX_ACK_STALE: return "stale";
    case HA_RX_DROP_INVALID: return "invalid";
    case HA_RX_DROP_CONFLICT: return "conflict";
    case HA_RX_DROP_NOT_ACCEPTED: return "not_accepted";
    }
    return "?";
}
