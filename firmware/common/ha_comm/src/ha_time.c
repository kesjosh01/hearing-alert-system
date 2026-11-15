#include "ha_time.h"

#include <string.h>

#include "ha_config.h"

void ha_clock_default_config(ha_clock_config_t *cfg) {
    cfg->delivery_bound_ms = HA_TIME_DELIVERY_BOUND_MS;
    cfg->drift_ppm = HA_TIME_DRIFT_PPM;
    cfg->stale_after_ms = HA_TIME_STALE_AFTER_MS;
}

void ha_clock_init(ha_clock_t *c, const ha_clock_config_t *cfg) {
    memset(c, 0, sizeof(*c));
    c->cfg = *cfg;
}

ha_time_apply_t ha_clock_apply(ha_clock_t *c, const ha_time_sync_t *msg, int64_t mono_now_ms) {
    if (c->have && msg->session == c->session && msg->seq <= c->seq) return HA_TIME_IGNORED_OLD;
    c->have = true;
    c->session = msg->session;
    c->seq = msg->seq;
    // utc_ms was sampled before transmission; the true time at reception is
    // later by an unknown delivery delay, which goes into the uncertainty.
    c->utc_at_rx_ms = msg->utc_ms;
    c->mono_at_rx_ms = mono_now_ms;
    uint64_t base = (uint64_t)msg->accuracy_ms + c->cfg.delivery_bound_ms;
    c->base_uncertainty_ms = base > UINT32_MAX ? UINT32_MAX : (uint32_t)base;
    return HA_TIME_APPLIED;
}

ha_clock_state_t ha_clock_now(const ha_clock_t *c, int64_t mono_now_ms, int64_t *utc_ms,
                              uint32_t *uncertainty_ms) {
    if (!c->have) return HA_CLOCK_UNSYNCED;
    int64_t elapsed = mono_now_ms - c->mono_at_rx_ms;
    if (elapsed < 0) elapsed = 0;
    if (utc_ms) *utc_ms = c->utc_at_rx_ms + elapsed;
    if (uncertainty_ms) {
        uint64_t drift = (uint64_t)elapsed * c->cfg.drift_ppm / 1000000u;
        uint64_t total = (uint64_t)c->base_uncertainty_ms + drift;
        *uncertainty_ms = total > UINT32_MAX ? UINT32_MAX : (uint32_t)total;
    }
    return elapsed > c->cfg.stale_after_ms ? HA_CLOCK_STALE : HA_CLOCK_SYNCED;
}
