/*
 * C3 wall-clock state fed by v2 TIME_SYNC messages (optional extension).
 *
 * The C3 never invents a time: before the first valid TIME_SYNC the state is
 * UNSYNCED (no compile-time or epoch default is shown as "now"). After a sync
 * the estimate is  utc_at_rx + (mono_now - mono_at_rx)  and the uncertainty is
 *   S3 accuracy + delivery bound (C3 sleep / AP buffering) + drift * elapsed.
 * A message is a delayed copy if its seq is not newer within the same S3
 * session. A new S3 session (S3 reboot) is accepted.
 *
 * Alert timers never read this state; they use the monotonic clock only.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "ha_proto.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HA_CLOCK_UNSYNCED = 0,
    HA_CLOCK_SYNCED,
    HA_CLOCK_STALE, /* last sync older than stale_after_ms; estimate still given */
} ha_clock_state_t;

typedef struct {
    uint32_t delivery_bound_ms;
    uint32_t drift_ppm;
    int64_t stale_after_ms;
} ha_clock_config_t;

typedef struct {
    ha_clock_config_t cfg;
    bool have;
    uint32_t session, seq;
    int64_t utc_at_rx_ms;
    int64_t mono_at_rx_ms;
    uint32_t base_uncertainty_ms;
} ha_clock_t;

typedef enum {
    HA_TIME_APPLIED = 0,
    HA_TIME_IGNORED_OLD, /* same session, seq not newer: delayed/duplicate */
} ha_time_apply_t;

void ha_clock_default_config(ha_clock_config_t *cfg);
void ha_clock_init(ha_clock_t *c, const ha_clock_config_t *cfg);

/* msg must come from ha_time_sync_decode() (range/flags already validated). */
ha_time_apply_t ha_clock_apply(ha_clock_t *c, const ha_time_sync_t *msg, int64_t mono_now_ms);

/* Current estimate. Returns UNSYNCED (outputs untouched) if never synced. */
ha_clock_state_t ha_clock_now(const ha_clock_t *c, int64_t mono_now_ms, int64_t *utc_ms,
                              uint32_t *uncertainty_ms);

#ifdef __cplusplus
}
#endif
