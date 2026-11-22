/*
 * What the C6 shows and plays for each received alert (pure C state machine).
 *
 * Inputs: alert (event id), button, time. Output: a command for the display
 * and the vibration motor, plus "shown" = the alert is visible once the screen
 * command has been drawn (the UI then sends the ACK).
 *
 * Priority rules (implementation_design 4-2):
 *   idle          + fire      -> FIRE screen, fire vibration
 *   idle          + doorbell  -> DOORBELL screen, doorbell vibration
 *   doorbell      + fire      -> doorbell stops at once, FIRE (+ bell badge)
 *   fire          + doorbell  -> fire keeps going, bell badge added
 *   same kind again (S3 re-alert) -> pattern restarts, hold time extended
 *   button during an alert    -> back to the watch face, vibration off
 *   hold time over            -> back to the watch face, vibration off
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "ha_screens.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- vibration patterns ---- */
typedef struct {
    uint8_t duty; /* 0..255; the current driver is on/off (duty > 0 = on) */
    uint16_t ms;
} ha_vib_step_t;

typedef struct {
    const ha_vib_step_t *steps;
    uint8_t n;
    bool repeat; /* loop until stopped (or the driver's safety limit) */
} ha_vib_pattern_t;

typedef enum {
    HA_PAT_DOORBELL = 0, /* five short pulses, once */
    HA_PAT_FIRE,         /* urgent, continuously repeated */
    HA_PAT_SELFTEST,     /* one short pulse at boot */
    HA_PAT_COUNT
} ha_pattern_id_t;

const ha_vib_pattern_t *ha_vib_pattern(ha_pattern_id_t id);

/* ---- presenter ---- */
typedef enum { HA_UI_IDLE = 0, HA_UI_DOORBELL, HA_UI_FIRE } ha_ui_state_t;

#define HA_UI_KEEP (-1) /* screen/haptic: no change */
#define HA_HAP_STOP (-2)

typedef struct {
    int screen; /* HA_UI_KEEP or ha_screen_t */
    int haptic; /* HA_UI_KEEP, HA_HAP_STOP or ha_pattern_id_t */
    bool shown; /* the alert is on screen once `screen` is drawn -> ACK it */
} ha_ui_cmd_t;

typedef struct {
    uint32_t fire_hold_ms;     /* 0 = until the button */
    uint32_t doorbell_hold_ms; /* 0 = until the button */
} ha_presenter_cfg_t;

typedef struct {
    ha_presenter_cfg_t cfg;
    ha_ui_state_t state;
    bool doorbell_too; /* a doorbell arrived during (or just before) the fire */
    int64_t until_ms;  /* INT64_MAX = no timeout */
} ha_presenter_t;

void ha_presenter_init(ha_presenter_t *p, const ha_presenter_cfg_t *cfg);
ha_ui_cmd_t ha_presenter_on_alert(ha_presenter_t *p, uint32_t event_id, int64_t now_ms);
ha_ui_cmd_t ha_presenter_on_button(ha_presenter_t *p, int64_t now_ms);
ha_ui_cmd_t ha_presenter_on_tick(ha_presenter_t *p, int64_t now_ms);
/* Next time on_tick() must run (INT64_MAX = none). */
int64_t ha_presenter_deadline(const ha_presenter_t *p);
ha_screen_t ha_presenter_screen(const ha_presenter_t *p);

/* ---- watch face text ---- */
/* utc_ms -> "HH:MM" and "M월 D일 요일" at the given UTC offset (minutes).
 * Pure arithmetic (no TZ database); Korea has no DST. */
void ha_watch_format(int64_t utc_ms, int utc_offset_min, char time_out[8], char date_out[32]);
/* Text shown while no valid time exists. */
void ha_watch_format_unknown(char time_out[8], char date_out[32]);
/* Milliseconds until the displayed minute changes. */
int64_t ha_watch_ms_to_next_minute(int64_t utc_ms, int utc_offset_min);

/* DEMO ONLY: __DATE__ ("Nov 17 2026") + __TIME__ ("14:02:33"), taken as local
 * time at utc_offset_min, -> UTC ms. False if the strings do not parse. */
bool ha_watch_build_time_utc(const char *date, const char *time, int utc_offset_min,
                             int64_t *utc_ms);

/* Battery voltage [mV] -> 0..100 % (linear between empty and full). */
int ha_battery_pct(int mv, int empty_mv, int full_mv);

#ifdef __cplusplus
}
#endif
