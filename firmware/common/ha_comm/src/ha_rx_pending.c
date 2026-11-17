#include "ha_rx_pending.h"

#include <string.h>

#include "ha_events.h"

void ha_rx_pending_init(ha_rx_pending_t *p, int64_t max_age_ms) {
    memset(p, 0, sizeof(*p));
    if (max_age_ms <= 0) {
        uint32_t longest = 0;
        for (size_t i = 0; i < ha_event_count(); ++i) {
            const ha_event_def_t *d = ha_event_at(i);
            if (d->ttl_ms > longest) longest = d->ttl_ms;
        }
        max_age_ms = longest ? (int64_t)longest : (int64_t)10 * 60 * 1000;
    }
    p->max_age_ms = max_age_ms;
}

static int64_t max_age(const ha_rx_pending_t *p, uint32_t event_id) {
    const ha_event_def_t *d = ha_event_find(event_id);
    return (d && d->ttl_ms) ? (int64_t)d->ttl_ms : p->max_age_ms;
}

static bool expired(const ha_rx_pending_t *p, const ha_rx_pending_entry_t *e, int64_t now_ms) {
    return now_ms - e->since_ms > max_age(p, e->event_id);
}

static void age_out(ha_rx_pending_t *p, int64_t now_ms) {
    for (int i = 0; i < HA_RX_PENDING_LEN; ++i) {
        if (p->e[i].used && expired(p, &p->e[i], now_ms)) {
            p->e[i].used = false;
            ++p->aged_out;
        }
    }
}

static int free_slot(const ha_rx_pending_t *p) {
    for (int i = 0; i < HA_RX_PENDING_LEN; ++i)
        if (!p->e[i].used) return i;
    return -1;
}

bool ha_rx_pending_has_room(ha_rx_pending_t *p, int64_t now_ms) {
    if (free_slot(p) >= 0) return true;
    age_out(p, now_ms);
    return free_slot(p) >= 0;
}

bool ha_rx_pending_add(ha_rx_pending_t *p, uint32_t session, uint32_t seq, uint32_t event_id,
                       int64_t now_ms) {
    if (!ha_rx_pending_has_room(p, now_ms)) return false;
    ha_rx_pending_entry_t *e = &p->e[free_slot(p)];
    e->used = true;
    e->session = session;
    e->seq = seq;
    e->event_id = event_id;
    e->since_ms = now_ms;
    return true;
}

static int find(const ha_rx_pending_t *p, uint32_t session, uint32_t seq) {
    for (int i = 0; i < HA_RX_PENDING_LEN; ++i)
        if (p->e[i].used && p->e[i].session == session && p->e[i].seq == seq) return i;
    return -1;
}

bool ha_rx_pending_contains(const ha_rx_pending_t *p, uint32_t session, uint32_t seq) {
    return find(p, session, seq) >= 0;
}

bool ha_rx_pending_take(ha_rx_pending_t *p, uint32_t session, uint32_t seq, uint32_t event_id) {
    int i = find(p, session, seq);
    if (i < 0 || p->e[i].event_id != event_id) return false;
    p->e[i].used = false;
    return true;
}

int ha_rx_pending_count(const ha_rx_pending_t *p) {
    int n = 0;
    for (int i = 0; i < HA_RX_PENDING_LEN; ++i) n += p->e[i].used;
    return n;
}

bool ha_rx_pending_oldest(const ha_rx_pending_t *p, int64_t now_ms, int64_t *since_ms) {
    bool any = false;
    for (int i = 0; i < HA_RX_PENDING_LEN; ++i) {
        const ha_rx_pending_entry_t *e = &p->e[i];
        if (!e->used || expired(p, e, now_ms)) continue;
        if (!any || e->since_ms < *since_ms) *since_ms = e->since_ms;
        any = true;
    }
    return any;
}
