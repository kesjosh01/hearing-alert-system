#include "ha_screens.h"

#include <math.h>
#include <stdio.h>

#define CX 120
#define CY 120

/* Battery ring: demo drawArcDots(112, from, to, 3, colour): a dot every 1.5
 * degrees from 12 o'clock, clockwise; 0..360 inclusive = 241 dots. */
#define ARC_R 112
#define ARC_DOTS 241
#define ARC_DOT_R 3

static int16_t s_arc_x[ARC_DOTS], s_arc_y[ARC_DOTS];
static bool s_arc_ready;

static void arc_init(void) {
    if (s_arc_ready) return;
    for (int i = 0; i < ARC_DOTS; ++i) {
        double rad = (i * 1.5) * 3.14159265358979323846 / 180.0;
        // Arduino casts the float sum to int (truncation toward zero).
        s_arc_x[i] = (int16_t)(CX + ARC_R * sin(rad));
        s_arc_y[i] = (int16_t)(CY - ARC_R * cos(rad));
    }
    s_arc_ready = true;
}

static void arc_dots(ha_canvas_t *c, int n, uint16_t color) {
    for (int i = 0; i < n && i < ARC_DOTS; ++i)
        ha_gfx_circle(c, s_arc_x[i], s_arc_y[i], ARC_DOT_R, color);
}

/* Bell (demo bellIcon), scaled by num/den around (x, y). */
static void bell_icon(ha_canvas_t *c, int x, int y, uint16_t col, int num, int den) {
#define S(v) ((v) * num / den)
    ha_gfx_circle(c, x, y - S(28), S(6), col);                           // knob
    ha_gfx_circle(c, x, y, S(28), col);                                  // dome
    ha_gfx_rect(c, x - S(28), y, S(57), S(25), col);                     // body
    ha_gfx_round_rect(c, x - S(38), y + S(22), S(77), S(10), S(5), col); // rim
    ha_gfx_circle(c, x, y + S(40), S(7), col);                           // clapper
#undef S
}

/* Warning sign (demo warnIcon). */
static void warn_icon(ha_canvas_t *c, int x, int y, uint16_t col, uint16_t mark) {
    ha_gfx_triangle(c, x, y - 45, x - 52, y + 42, x + 52, y + 42, col);
    ha_gfx_rect(c, x - 6, y - 15, 12, 35, mark); // "!" bar
    ha_gfx_circle(c, x, y + 30, 7, mark);        // "!" dot
}

static void draw_watch(const ha_scene_t *s, ha_canvas_t *c) {
    arc_init();
    ha_gfx_fill(c, HA_BLACK);
    arc_dots(c, ARC_DOTS, HA_DGRAY); // track
    char pct[8];
    if (s->battery_pct >= 0) {
        int p = s->battery_pct > 100 ? 100 : s->battery_pct;
        // Same dot count as the demo loop: a = 0, 1.5, ... <= 360 * p / 100.
        arc_dots(c, (int)((360.0 * p / 100.0) / 1.5) + 1, HA_GREEN);
        snprintf(pct, sizeof(pct), "%d%%", p);
        ha_gfx_text_center(c, &ha_font_small, pct, CX, 198, HA_GREEN);
    } else {
        ha_gfx_text_center(c, &ha_font_small, "--%", CX, 198, HA_DGRAY);
    }
    ha_gfx_text_center(c, &ha_font_small, s->date, CX, 88, HA_WHITE);
    ha_gfx_text_center(c, &ha_font_clock, s->time, CX, 138, HA_WHITE);
}

static void draw_doorbell(const ha_scene_t *s, ha_canvas_t *c) {
    ha_gfx_fill(c, HA_NAVY);
    bell_icon(c, CX, 90, HA_YELLOW, 1, 1);
    ha_gfx_text_center(c, &ha_font_medium, "초인종", CX, 172, HA_WHITE);
    ha_gfx_annulus(c, CX, CY, 112, 119, s->blink ? HA_WHITE : HA_NAVY); // rim blink
}

static void draw_fire(const ha_scene_t *s, ha_canvas_t *c, bool doorbell_badge) {
    // Demo starts an alert with the "on" (red) frame; blink=false is that frame.
    bool on = !s->blink;
    uint16_t bg = on ? HA_RED : HA_BLACK;
    ha_gfx_fill(c, bg);
    warn_icon(c, CX, 88, on ? HA_YELLOW : HA_ORANGE, bg);
    ha_gfx_text_center(c, &ha_font_large, "긴급", CX, 172, HA_WHITE);
    if (doorbell_badge) {
        // Steady white badge with a navy bell (upper right, inside the circle).
        ha_gfx_circle(c, 186, 60, 24, HA_WHITE);
        bell_icon(c, 186, 58, HA_NAVY, 2, 5);
    }
}

void ha_scene_draw(const ha_scene_t *s, ha_canvas_t *c) {
    switch (s->screen) {
    case HA_SCR_DOORBELL: draw_doorbell(s, c); break;
    case HA_SCR_FIRE: draw_fire(s, c, false); break;
    case HA_SCR_FIRE_DOORBELL: draw_fire(s, c, true); break;
    case HA_SCR_WATCH:
    default: draw_watch(s, c); break;
    }
}

ha_rect_t ha_scene_watch_text_rect(void) {
    // Date (cy 88) and clock (cy 138); kept inside the battery ring (r >= 109).
    return (ha_rect_t){22, 66, 196, 100};
}

const char *ha_screen_name(ha_screen_t s) {
    switch (s) {
    case HA_SCR_WATCH: return "WATCH";
    case HA_SCR_DOORBELL: return "DOORBELL";
    case HA_SCR_FIRE: return "FIRE";
    case HA_SCR_FIRE_DOORBELL: return "FIRE+DOORBELL";
    default: return "?";
    }
}
