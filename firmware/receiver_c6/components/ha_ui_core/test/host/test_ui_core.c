/*
 * Host tests for ha_ui_core (presenter rules, watch text, rasteriser).
 * Run: bash firmware/receiver_c6/components/ha_ui_core/test/host/run_tests.sh [preview_dir]
 * With preview_dir, every screen is also written as a PPM image for review.
 * Not a substitute for checking the real LCD/motor.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ha_events.h"
#include "ha_gfx.h"
#include "ha_presenter.h"
#include "ha_recovery.h"
#include "ha_screens.h"

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
#define CHECK_STR(a, b) CHECK(strcmp((a), (b)) == 0)

static const ha_presenter_cfg_t k_cfg = {60000, 15000};

static void test_presenter_basic(void) {
    ha_presenter_t p;
    ha_presenter_init(&p, &k_cfg);
    CHECK_EQ(ha_presenter_screen(&p), HA_SCR_WATCH);
    CHECK_EQ(ha_presenter_deadline(&p), INT64_MAX);

    ha_ui_cmd_t c = ha_presenter_on_alert(&p, HA_EVENT_DOORBELL, 1000);
    CHECK_EQ(c.screen, HA_SCR_DOORBELL);
    CHECK_EQ(c.haptic, HA_PAT_DOORBELL);
    CHECK(c.shown);
    CHECK_EQ(ha_presenter_deadline(&p), 16000);
    // Not yet expired, then expired -> watch face and motor off.
    c = ha_presenter_on_tick(&p, 15999);
    CHECK_EQ(c.screen, HA_UI_KEEP);
    c = ha_presenter_on_tick(&p, 16000);
    CHECK_EQ(c.screen, HA_SCR_WATCH);
    CHECK_EQ(c.haptic, HA_HAP_STOP);
    CHECK(!c.shown);
    CHECK_EQ(p.state, HA_UI_IDLE);

    // Fire, then the button.
    c = ha_presenter_on_alert(&p, HA_EVENT_FIRE_ALARM, 20000);
    CHECK_EQ(c.screen, HA_SCR_FIRE);
    CHECK_EQ(c.haptic, HA_PAT_FIRE);
    CHECK_EQ(ha_presenter_deadline(&p), 80000);
    c = ha_presenter_on_button(&p, 21000);
    CHECK_EQ(c.screen, HA_SCR_WATCH);
    CHECK_EQ(c.haptic, HA_HAP_STOP);
    // Button while idle: nothing to change (UI may wake the backlight).
    c = ha_presenter_on_button(&p, 22000);
    CHECK_EQ(c.screen, HA_UI_KEEP);
    CHECK_EQ(c.haptic, HA_UI_KEEP);
}

static void test_presenter_priority(void) {
    ha_presenter_t p;
    ha_presenter_init(&p, &k_cfg);
    // Doorbell showing, fire arrives: doorbell cut off, fire + bell badge.
    ha_presenter_on_alert(&p, HA_EVENT_DOORBELL, 0);
    ha_ui_cmd_t c = ha_presenter_on_alert(&p, HA_EVENT_FIRE_ALARM, 500);
    CHECK_EQ(c.screen, HA_SCR_FIRE_DOORBELL);
    CHECK_EQ(c.haptic, HA_PAT_FIRE);
    CHECK(c.shown);
    CHECK_EQ(ha_presenter_deadline(&p), 60500);

    // Fire showing, doorbell arrives: fire keeps vibration and hold time.
    ha_presenter_init(&p, &k_cfg);
    ha_presenter_on_alert(&p, HA_EVENT_FIRE_ALARM, 0);
    c = ha_presenter_on_alert(&p, HA_EVENT_DOORBELL, 30000);
    CHECK_EQ(c.screen, HA_SCR_FIRE_DOORBELL);
    CHECK_EQ(c.haptic, HA_UI_KEEP);
    CHECK(c.shown);
    CHECK_EQ(ha_presenter_deadline(&p), 60000);
    CHECK_EQ(ha_presenter_screen(&p), HA_SCR_FIRE_DOORBELL);

    // Same kind again (S3 re-alert): restart pattern, extend hold.
    c = ha_presenter_on_alert(&p, HA_EVENT_FIRE_ALARM, 50000);
    CHECK_EQ(c.screen, HA_SCR_FIRE_DOORBELL); // badge kept until the fire ends
    CHECK_EQ(c.haptic, HA_PAT_FIRE);
    CHECK_EQ(ha_presenter_deadline(&p), 110000);
    c = ha_presenter_on_tick(&p, 110000);
    CHECK_EQ(c.screen, HA_SCR_WATCH);
    CHECK(!p.doorbell_too);

    ha_presenter_init(&p, &k_cfg);
    ha_presenter_on_alert(&p, HA_EVENT_DOORBELL, 0);
    c = ha_presenter_on_alert(&p, HA_EVENT_DOORBELL, 10000);
    CHECK_EQ(c.screen, HA_SCR_DOORBELL);
    CHECK_EQ(c.haptic, HA_PAT_DOORBELL);
    CHECK_EQ(ha_presenter_deadline(&p), 25000);

    // Unknown / TEST id: no screen, not "shown", state untouched.
    c = ha_presenter_on_alert(&p, HA_EVENT_TEST, 11000);
    CHECK_EQ(c.screen, HA_UI_KEEP);
    CHECK(!c.shown);
    CHECK_EQ(p.state, HA_UI_DOORBELL);

    // Hold 0 = until the button.
    ha_presenter_cfg_t forever = {0, 0};
    ha_presenter_init(&p, &forever);
    ha_presenter_on_alert(&p, HA_EVENT_FIRE_ALARM, 0);
    CHECK_EQ(ha_presenter_deadline(&p), INT64_MAX);
    CHECK_EQ(ha_presenter_on_tick(&p, 1000000000).screen, HA_UI_KEEP);
}

static void test_patterns(void) {
    const ha_vib_pattern_t *d = ha_vib_pattern(HA_PAT_DOORBELL);
    CHECK(d && !d->repeat && d->n == 9);
    for (int i = 0; i < d->n; ++i) {
        CHECK_EQ(d->steps[i].duty, i % 2 ? 0 : 255);
        CHECK_EQ(d->steps[i].ms, i % 2 ? 220 : 180);
    }
    const ha_vib_pattern_t *f = ha_vib_pattern(HA_PAT_FIRE);
    CHECK(f && f->repeat && f->steps[0].ms == 350 && f->steps[1].ms == 100);
    CHECK(!ha_vib_pattern(HA_PAT_SELFTEST)->repeat);
    CHECK(ha_vib_pattern(HA_PAT_COUNT) == NULL);
}

static void test_watch_text(void) {
    char t[8], d[32];
    // 2026-11-17 05:04:59 UTC = 14:04 KST, Tuesday.
    int64_t ms = 1794891899000LL;
    ha_watch_format(ms, 540, t, d);
    CHECK_STR(t, "14:04");
    CHECK_STR(d, "11월 17일 화");
    CHECK_EQ(ha_watch_ms_to_next_minute(ms, 540), 1000);
    // UTC 15:00 on Dec 31 -> KST 00:00 on Jan 1 (year rollover), Thursday 2026-01-01.
    ha_watch_format(1767193200000LL, 540, t, d);
    CHECK_STR(t, "00:00");
    CHECK_STR(d, "1월 1일 목");
    CHECK_EQ(ha_watch_ms_to_next_minute(1767193200000LL, 540), 60000);
    // Leap day 2028-02-29 12:00 KST (Tuesday).
    ha_watch_format(1835406000000LL, 540, t, d);
    CHECK_STR(t, "12:00");
    CHECK_STR(d, "2월 29일 화");
    ha_watch_format(0, 0, t, d);
    CHECK_STR(t, "00:00");
    CHECK_STR(d, "1월 1일 목");
    int64_t bt = 0;
    CHECK(ha_watch_build_time_utc("Nov 17 2026", "14:04:59", 540, &bt));
    CHECK_EQ(bt, ms);
    CHECK(ha_watch_build_time_utc("Feb 29 2028", "12:00:00", 540, &bt));
    CHECK_EQ(bt, 1835406000000LL);
    CHECK(!ha_watch_build_time_utc("Foo 1 2026", "00:00:00", 540, &bt));
    CHECK(!ha_watch_build_time_utc("Nov 17 2026", "xx", 540, &bt));
    ha_watch_format_unknown(t, d);
    CHECK_STR(t, "--:--");
    CHECK_STR(d, "시간 없음");
}

static void test_battery(void) {
    CHECK_EQ(ha_battery_pct(3300, 3300, 4200), 0);
    CHECK_EQ(ha_battery_pct(3000, 3300, 4200), 0);
    CHECK_EQ(ha_battery_pct(4200, 3300, 4200), 100);
    CHECK_EQ(ha_battery_pct(4500, 3300, 4200), 100);
    CHECK_EQ(ha_battery_pct(3750, 3300, 4200), 50);
    CHECK_EQ(ha_battery_pct(3900, 4200, 3300), -1);
}

static void test_font_and_text(void) {
    // Every character the screens print exists in the generated fonts.
    const char *s_texts[] = {"0123456789%-", "월일화수목금토", "시간 없음"};
    for (size_t i = 0; i < sizeof(s_texts) / sizeof(s_texts[0]); ++i) {
        const char *p = s_texts[i];
        int x1, y1, w, h;
        ha_gfx_text_bounds(&ha_font_small, p, &x1, &y1, &w, &h);
        CHECK(w > 0 && h > 0);
    }
    CHECK(ha_font_find(&ha_font_medium, 0xCD08) != NULL); // 초
    CHECK(ha_font_find(&ha_font_large, 0xAE34) != NULL); // 긴
    CHECK(ha_font_find(&ha_font_clock, ':') != NULL);
    CHECK(ha_font_find(&ha_font_clock, 'A') == NULL);
    int x1, y1, w, h;
    ha_gfx_text_bounds(&ha_font_clock, "88:88", &x1, &y1, &w, &h);
    CHECK(w > 140 && w < 200); // demo clock is ~180 px wide
    CHECK(h > 38 && h < 56);   // demo clock is 42 px tall
    ha_gfx_text_bounds(&ha_font_small, "12월 31일 수", &x1, &y1, &w, &h);
    CHECK(w < 220); // fits the circle at the date row
}

/* -------------------------------------------------- raster + preview */

static uint16_t g_full[HA_LCD_W * HA_LCD_H];

static void render_full(const ha_scene_t *s, bool be) {
    ha_canvas_t c = {g_full, 0, 0, HA_LCD_W, HA_LCD_H, be};
    ha_scene_draw(s, &c);
}

/* Band-by-band (as on the device) must equal one full-frame draw. */
static bool bands_match(const ha_scene_t *s, int band_h) {
    static uint16_t band[HA_LCD_W * 32];
    render_full(s, false);
    for (int y0 = 0; y0 < HA_LCD_H; y0 += band_h) {
        int h = (y0 + band_h > HA_LCD_H) ? HA_LCD_H - y0 : band_h;
        ha_canvas_t c = {band, 0, y0, HA_LCD_W, h, true};
        ha_scene_draw(s, &c);
        for (int y = y0; y < y0 + h; ++y)
            for (int x = 0; x < HA_LCD_W; ++x)
                if (ha_gfx_get(&c, x, y) != g_full[y * HA_LCD_W + x]) return false;
    }
    // A partial (dirty-rect) canvas must also match.
    ha_rect_t r = ha_scene_watch_text_rect();
    static uint16_t part[HA_LCD_W * HA_LCD_H];
    ha_canvas_t pc = {part, r.x, r.y, r.w, r.h, true};
    ha_scene_draw(s, &pc);
    for (int y = r.y; y < r.y + r.h; ++y)
        for (int x = r.x; x < r.x + r.w; ++x)
            if (ha_gfx_get(&pc, x, y) != g_full[y * HA_LCD_W + x]) return false;
    return true;
}

static void write_ppm(const char *dir, const char *name) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.ppm", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    fprintf(f, "P6\n%d %d\n255\n", HA_LCD_W, HA_LCD_H);
    for (int y = 0; y < HA_LCD_H; ++y)
        for (int x = 0; x < HA_LCD_W; ++x) {
            uint16_t v = g_full[y * HA_LCD_W + x];
            // The panel is round: show what is outside the glass as grey.
            int dx = x - 120, dy = y - 120;
            unsigned char rgb[3] = {(unsigned char)((v >> 11) * 255 / 31),
                                    (unsigned char)(((v >> 5) & 63) * 255 / 63),
                                    (unsigned char)((v & 31) * 255 / 31)};
            if (dx * dx + dy * dy > 120 * 120) rgb[0] = rgb[1] = rgb[2] = 90;
            fwrite(rgb, 1, 3, f);
        }
    fclose(f);
}

static void test_scenes(const char *preview_dir) {
    struct { const char *name; ha_scene_t s; } k[] = {
        {"watch", {HA_SCR_WATCH, false, "14:04", "11월 17일 화", 70}},
        {"watch_no_time", {HA_SCR_WATCH, false, "--:--", "시간 없음", -1}},
        {"doorbell_a", {HA_SCR_DOORBELL, false, "", "", 0}},
        {"doorbell_b", {HA_SCR_DOORBELL, true, "", "", 0}},
        {"fire_a", {HA_SCR_FIRE, false, "", "", 0}},
        {"fire_b", {HA_SCR_FIRE, true, "", "", 0}},
        {"fire_doorbell_a", {HA_SCR_FIRE_DOORBELL, false, "", "", 0}},
        {"fire_doorbell_b", {HA_SCR_FIRE_DOORBELL, true, "", "", 0}},
    };
    for (size_t i = 0; i < sizeof(k) / sizeof(k[0]); ++i) {
        CHECK(bands_match(&k[i].s, 20));
        CHECK(bands_match(&k[i].s, 7));
        render_full(&k[i].s, false);
        if (preview_dir) write_ppm(preview_dir, k[i].name);
    }
    // Spot checks against the demo design.
    render_full(&k[0].s, false);
    CHECK_EQ(g_full[(120 - 112) * HA_LCD_W + 120], HA_GREEN); // 12 o'clock dot is green
    CHECK_EQ(g_full[120 * HA_LCD_W + 120 - 112], HA_DGRAY);   // 9 o'clock: beyond 70 %
    CHECK_EQ(g_full[5 * HA_LCD_W + 5], HA_BLACK);
    render_full(&k[2].s, false);
    CHECK_EQ(g_full[0], HA_NAVY);
    CHECK_EQ(g_full[90 * HA_LCD_W + 120], HA_YELLOW); // bell dome
    render_full(&k[3].s, false);
    CHECK_EQ(g_full[(120 - 115) * HA_LCD_W + 120], HA_WHITE); // rim on
    render_full(&k[4].s, false);
    CHECK_EQ(g_full[0], HA_RED);
    CHECK_EQ(g_full[100 * HA_LCD_W + 120], HA_RED); // "!" bar in background colour
    render_full(&k[5].s, false);
    CHECK_EQ(g_full[0], HA_BLACK);
    CHECK_EQ(g_full[125 * HA_LCD_W + 90], HA_ORANGE);
    render_full(&k[6].s, false);
    CHECK_EQ(g_full[40 * HA_LCD_W + 186], HA_WHITE); // badge
}

static void test_recovery(void) {
    const ha_rec_cfg_t cfg = {3000, 6000, 3};
    ha_rec_t r;
    int64_t next;
    ha_rec_init(&r, &cfg, 0);
    // Nothing waiting: nothing to do, no timer.
    CHECK_EQ(ha_rec_step(&r, false, 0, 100, &next), HA_REC_NONE);
    CHECK_EQ(next, INT64_MAX);
    // Alert accepted at t=1000, normal draw: no action, check again at 4000.
    CHECK_EQ(ha_rec_step(&r, true, 1000, 1050, &next), HA_REC_NONE);
    CHECK_EQ(next, 4000);
    // Stuck for 3 s: ask the UI once.
    CHECK_EQ(ha_rec_step(&r, true, 1000, 4000, &next), HA_REC_ASK_UI);
    CHECK_EQ(next, 7000);
    CHECK_EQ(ha_rec_step(&r, true, 1000, 5000, &next), HA_REC_NONE);
    // Still stuck at 6 s: reboot (budget 3).
    CHECK_EQ(ha_rec_step(&r, true, 1000, 7000, &next), HA_REC_REBOOT);
    CHECK_EQ(r.reboots, 1);

    // After 3 consecutive watchdog reboots (counter kept across reboots)...
    ha_rec_init(&r, &cfg, 3);
    CHECK_EQ(ha_rec_step(&r, true, 0, 3000, &next), HA_REC_ASK_UI); // still try the LCD
    CHECK_EQ(ha_rec_step(&r, true, 0, 6000, &next), HA_REC_DEGRADED);
    CHECK_EQ(next, INT64_MAX);
    CHECK_EQ(ha_rec_step(&r, true, 0, 9000, &next), HA_REC_NONE); // reported once only
    CHECK_EQ(r.reboots, 3);
    // A newer stuck alert is reported again (no reboot).
    CHECK_EQ(ha_rec_step(&r, true, 20000, 26000, &next), HA_REC_DEGRADED);

    // Display works again: budget refilled, next stuck alert may reboot.
    ha_rec_displayed_ok(&r);
    CHECK_EQ(r.reboots, 0);
    CHECK_EQ(ha_rec_step(&r, true, 30000, 36000, &next), HA_REC_REBOOT);
    CHECK(strcmp(ha_rec_action_name(HA_REC_REBOOT), "reboot") == 0);
}

int main(int argc, char **argv) {
    test_presenter_basic();
    test_presenter_priority();
    test_patterns();
    test_watch_text();
    test_battery();
    test_recovery();
    test_font_and_text();
    test_scenes(argc > 1 ? argv[1] : NULL);
    printf("%s: %d checks, %d failed\n", g_fail ? "HA_UI_HOST_TESTS_FAIL" : "HA_UI_HOST_TESTS_PASS",
           g_checks, g_fail);
    return g_fail ? 1 : 0;
}
