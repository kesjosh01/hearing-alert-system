#include "ha_recovery.h"

#include <string.h>

#define NONE_SINCE INT64_MIN

void ha_rec_init(ha_rec_t *r, const ha_rec_cfg_t *cfg, uint8_t reboots_so_far) {
    memset(r, 0, sizeof(*r));
    r->cfg = *cfg;
    r->reboots = reboots_so_far;
    r->asked_for = NONE_SINCE;
    r->degraded_for = NONE_SINCE;
}

ha_rec_action_t ha_rec_step(ha_rec_t *r, bool has_waiting, int64_t since, int64_t now,
                            int64_t *next_check_ms) {
    *next_check_ms = INT64_MAX;
    if (!has_waiting) return HA_REC_NONE;
    int64_t t1 = since + r->cfg.recover_after_ms, t2 = since + r->cfg.reboot_after_ms;
    if (now < t1) {
        *next_check_ms = t1;
        return HA_REC_NONE;
    }
    if (now < t2) {
        *next_check_ms = t2;
        if (r->asked_for == since) return HA_REC_NONE;
        r->asked_for = since;
        return HA_REC_ASK_UI;
    }
    if (r->reboots < r->cfg.max_reboots) {
        ++r->reboots;
        return HA_REC_REBOOT;
    }
    // Budget exhausted: report once per stuck alert, then wait for a new alert.
    if (r->degraded_for == since) return HA_REC_NONE;
    r->degraded_for = since;
    return HA_REC_DEGRADED;
}

void ha_rec_displayed_ok(ha_rec_t *r) {
    r->reboots = 0;
    r->degraded_for = NONE_SINCE;
}

const char *ha_rec_action_name(ha_rec_action_t a) {
    switch (a) {
    case HA_REC_NONE: return "none";
    case HA_REC_ASK_UI: return "reinit_display";
    case HA_REC_REBOOT: return "reboot";
    case HA_REC_DEGRADED: return "degraded";
    }
    return "?";
}
