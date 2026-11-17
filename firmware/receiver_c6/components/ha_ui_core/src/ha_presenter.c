#include "ha_presenter.h"

#include <stdio.h>
#include <string.h>

#include "ha_events.h"

/* ---------------------------------------------------------- patterns */

// Demo VIB_DOORBELL: {255,200},{0,150},{255,200},{0,2500}
static const ha_vib_step_t k_doorbell[] = {{255, 200}, {0, 150}, {255, 200}, {0, 2500}};
// Demo VIB_EMERGENCY: {255,500},{0,200}
static const ha_vib_step_t k_fire[] = {{255, 500}, {0, 200}};
static const ha_vib_step_t k_selftest[] = {{255, 150}};

static const ha_vib_pattern_t k_patterns[HA_PAT_COUNT] = {
    [HA_PAT_DOORBELL] = {k_doorbell, 4, true},
    [HA_PAT_FIRE] = {k_fire, 2, true},
    [HA_PAT_SELFTEST] = {k_selftest, 1, false},
};

const ha_vib_pattern_t *ha_vib_pattern(ha_pattern_id_t id) {
    return (id >= 0 && id < HA_PAT_COUNT) ? &k_patterns[id] : NULL;
}

/* --------------------------------------------------------- presenter */

static int64_t hold_until(int64_t now, uint32_t hold_ms) {
    return hold_ms ? now + (int64_t)hold_ms : INT64_MAX;
}

static ha_screen_t fire_screen(const ha_presenter_t *p) {
    return p->doorbell_too ? HA_SCR_FIRE_DOORBELL : HA_SCR_FIRE;
}

void ha_presenter_init(ha_presenter_t *p, const ha_presenter_cfg_t *cfg) {
    memset(p, 0, sizeof(*p));
    p->cfg = *cfg;
    p->state = HA_UI_IDLE;
    p->until_ms = INT64_MAX;
}

static ha_ui_cmd_t to_idle(ha_presenter_t *p) {
    p->state = HA_UI_IDLE;
    p->doorbell_too = false;
    p->until_ms = INT64_MAX;
    return (ha_ui_cmd_t){HA_SCR_WATCH, HA_HAP_STOP, false};
}

ha_ui_cmd_t ha_presenter_on_alert(ha_presenter_t *p, uint32_t event_id, int64_t now) {
    switch (event_id) {
    case HA_EVENT_FIRE_ALARM:
        // A doorbell that was on screen is cut off but stays visible as a badge.
        if (p->state == HA_UI_DOORBELL) p->doorbell_too = true;
        p->state = HA_UI_FIRE;
        p->until_ms = hold_until(now, p->cfg.fire_hold_ms);
        return (ha_ui_cmd_t){fire_screen(p), HA_PAT_FIRE, true};
    case HA_EVENT_DOORBELL:
        if (p->state == HA_UI_FIRE) {
            // Fire keeps priority (vibration and hold time unchanged).
            p->doorbell_too = true;
            return (ha_ui_cmd_t){HA_SCR_FIRE_DOORBELL, HA_UI_KEEP, true};
        }
        p->state = HA_UI_DOORBELL;
        p->until_ms = hold_until(now, p->cfg.doorbell_hold_ms);
        return (ha_ui_cmd_t){HA_SCR_DOORBELL, HA_PAT_DOORBELL, true};
    default:
        return (ha_ui_cmd_t){HA_UI_KEEP, HA_UI_KEEP, false}; // no screen for it
    }
}

ha_ui_cmd_t ha_presenter_on_button(ha_presenter_t *p, int64_t now) {
    (void)now;
    if (p->state == HA_UI_IDLE) return (ha_ui_cmd_t){HA_UI_KEEP, HA_UI_KEEP, false};
    return to_idle(p);
}

ha_ui_cmd_t ha_presenter_on_tick(ha_presenter_t *p, int64_t now) {
    if (p->state != HA_UI_IDLE && now >= p->until_ms) return to_idle(p);
    return (ha_ui_cmd_t){HA_UI_KEEP, HA_UI_KEEP, false};
}

int64_t ha_presenter_deadline(const ha_presenter_t *p) {
    return p->state == HA_UI_IDLE ? INT64_MAX : p->until_ms;
}

ha_screen_t ha_presenter_screen(const ha_presenter_t *p) {
    switch (p->state) {
    case HA_UI_FIRE: return fire_screen(p);
    case HA_UI_DOORBELL: return HA_SCR_DOORBELL;
    default: return HA_SCR_WATCH;
    }
}

/* -------------------------------------------------------- watch text */

static int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

/* Days since 1970-01-01 -> civil date (H. Hinnant's algorithm). */
static void civil_from_days(int64_t z, int *y, int *m, int *d) {
    z += 719468;
    int64_t era = floor_div(z, 146097);
    int64_t doe = z - era * 146097;
    int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    int64_t mp = (5 * doy + 2) / 153;
    *d = (int)(doy - (153 * mp + 2) / 5 + 1);
    *m = (int)(mp < 10 ? mp + 3 : mp - 9);
    *y = (int)(yoe + era * 400 + (*m <= 2));
}

void ha_watch_format(int64_t utc_ms, int utc_offset_min, char time_out[8], char date_out[32]) {
    static const char *const k_wday[] = {"일", "월", "화", "수", "목", "금", "토"};
    int64_t local_s = floor_div(utc_ms, 1000) + (int64_t)utc_offset_min * 60;
    int64_t days = floor_div(local_s, 86400);
    int64_t sod = local_s - days * 86400;
    int y, m, d;
    civil_from_days(days, &y, &m, &d);
    int wday = (int)(((days % 7) + 7 + 4) % 7); // 1970-01-01 was a Thursday
    snprintf(time_out, 8, "%02d:%02d", (int)(sod / 3600), (int)((sod / 60) % 60));
    snprintf(date_out, 32, "%d월 %d일 %s", m, d, k_wday[wday]);
}

void ha_watch_format_unknown(char time_out[8], char date_out[32]) {
    snprintf(time_out, 8, "--:--");
    snprintf(date_out, 32, "시간 없음");
}

int64_t ha_watch_ms_to_next_minute(int64_t utc_ms, int utc_offset_min) {
    int64_t local_ms = utc_ms + (int64_t)utc_offset_min * 60000;
    int64_t into = local_ms - floor_div(local_ms, 60000) * 60000;
    return 60000 - into;
}

/* Civil date -> days since 1970-01-01 (H. Hinnant's algorithm). */
static int64_t days_from_civil(int y, int m, int d) {
    y -= m <= 2;
    int64_t era = floor_div(y, 400);
    int64_t yoe = y - era * 400;
    int64_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

bool ha_watch_build_time_utc(const char *date, const char *time, int utc_offset_min,
                             int64_t *utc_ms) {
    static const char k_mon[] = "JanFebMarAprMayJunJulAugSepOctNovDec";
    char mon[4] = {0};
    int d, y, hh, mm, ss;
    if (sscanf(date, "%3s %d %d", mon, &d, &y) != 3 || sscanf(time, "%d:%d:%d", &hh, &mm, &ss) != 3)
        return false;
    const char *hit = strstr(k_mon, mon);
    if (!hit || (hit - k_mon) % 3) return false;
    int m = (int)((hit - k_mon) / 3) + 1;
    int64_t local_s = days_from_civil(y, m, d) * 86400 + hh * 3600 + mm * 60 + ss;
    *utc_ms = (local_s - (int64_t)utc_offset_min * 60) * 1000;
    return true;
}

int ha_battery_pct(int mv, int empty_mv, int full_mv) {
    if (full_mv <= empty_mv) return -1;
    if (mv <= empty_mv) return 0;
    if (mv >= full_mv) return 100;
    return (int)(((long)(mv - empty_mv) * 100 + (full_mv - empty_mv) / 2) / (full_mv - empty_mv));
}
