/*
 * Display watchdog policy (pure C): what to do when an accepted alert is not
 * on screen (= not ACKed) after a while.
 *
 *   age < recover_after_ms              wait (a normal draw takes ~50 ms)
 *   recover_after_ms <= age < reboot_after_ms
 *                                       ASK_UI once: re-initialise the LCD and
 *                                       redraw (UI task)
 *   age >= reboot_after_ms              REBOOT the C6 while the reboot budget
 *                                       lasts (the S3 still holds the event and
 *                                       resends it after the reboot); then
 *                                       DEGRADED once: vibration only, no more
 *                                       reboots, the UI keeps retrying the LCD
 *
 * The reboot budget counts consecutive watchdog reboots (kept across the
 * reboot by the glue) and is refilled when an alert is displayed again, so a
 * dead LCD cannot cause an endless reboot loop.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HA_REC_NONE = 0,
    HA_REC_ASK_UI,   /* re-init display + redraw */
    HA_REC_REBOOT,   /* esp_restart() (budget already decremented) */
    HA_REC_DEGRADED, /* budget exhausted: report once, keep running */
} ha_rec_action_t;

typedef struct {
    uint32_t recover_after_ms;
    uint32_t reboot_after_ms; /* > recover_after_ms */
    uint8_t max_reboots;      /* consecutive watchdog reboots allowed */
} ha_rec_cfg_t;

typedef struct {
    ha_rec_cfg_t cfg;
    uint8_t reboots;          /* consecutive watchdog reboots so far (persisted) */
    int64_t asked_for;        /* since_ms of the alert ASK_UI was issued for */
    int64_t degraded_for;     /* since_ms of the alert DEGRADED was reported for */
} ha_rec_t;

void ha_rec_init(ha_rec_t *r, const ha_rec_cfg_t *cfg, uint8_t reboots_so_far);

/* has_waiting/oldest_since: oldest accepted-but-not-displayed alert.
 * *next_check_ms: when to call again (INT64_MAX = only on a new alert). */
ha_rec_action_t ha_rec_step(ha_rec_t *r, bool has_waiting, int64_t oldest_since, int64_t now_ms,
                            int64_t *next_check_ms);

/* An alert reached the screen: the display works, refill the reboot budget. */
void ha_rec_displayed_ok(ha_rec_t *r);

const char *ha_rec_action_name(ha_rec_action_t a);

#ifdef __cplusplus
}
#endif
