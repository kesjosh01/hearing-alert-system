/*
 * The receiver's screens, drawn exactly like the 2026-11 breadboard demo
 * (receiver_demo_final.ino): same 240x240 coordinates, colours and icons.
 *
 *   WATCH         date + weekday, HH:MM, battery ring and %      (idle)
 *   DOORBELL      navy background, yellow bell, "초인종", blinking white rim
 *   FIRE          red/black full-screen blink, warning sign, "긴급"
 *   FIRE_DOORBELL FIRE + a white bell badge: a doorbell arrived during a fire
 *
 * ha_scene_draw() is clip-safe: call it for each band/dirty rectangle.
 */
#pragma once

#include <stdbool.h>

#include "ha_gfx.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HA_SCR_WATCH = 0,
    HA_SCR_DOORBELL,
    HA_SCR_FIRE,
    HA_SCR_FIRE_DOORBELL,
    HA_SCR_COUNT
} ha_screen_t;

typedef struct {
    ha_screen_t screen;
    bool blink;          /* animation phase, toggled every HA_UI_BLINK_MS */
    char time[8];        /* "HH:MM" or "--:--" */
    char date[32];       /* "11월 17일 화" or "시간 없음" */
    int battery_pct;     /* 0..100, or -1 = not measured */
} ha_scene_t;

#define HA_UI_BLINK_MS 400 /* demo: 400 ms animation step */

void ha_scene_draw(const ha_scene_t *s, ha_canvas_t *c);

/* Area that changes when only the watch's time/date text changes. */
ha_rect_t ha_scene_watch_text_rect(void);

const char *ha_screen_name(ha_screen_t s);

#ifdef __cplusplus
}
#endif
