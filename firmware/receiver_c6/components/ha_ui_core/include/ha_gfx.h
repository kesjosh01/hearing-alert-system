/*
 * Tiny clipped RGB565 rasteriser for the 240x240 round LCD.
 *
 * Screens are drawn into a "canvas" that covers only part of the panel (a
 * band of rows, or a dirty rectangle). Every primitive clips to the canvas, so
 * a whole screen can be produced band by band with ~10 KB of RAM instead of a
 * 115 KB frame buffer: the scene function is simply called once per band.
 * Pure C (no ESP-IDF), unit-tested and previewed on the host.
 *
 * Shapes follow the demo sketch (Adafruit-GFX style integer geometry), so the
 * same coordinates produce the same picture.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "ha_font.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HA_LCD_W 240
#define HA_LCD_H 240

/* RGB565 colours used by the demo design. */
#define HA_BLACK 0x0000u
#define HA_WHITE 0xFFFFu
#define HA_RED 0xF800u
#define HA_GREEN 0x07E0u
#define HA_YELLOW 0xFFE0u
#define HA_ORANGE 0xFD20u
#define HA_DGRAY 0x2104u
#define HA_NAVY 0x000Fu

typedef struct {
    uint16_t *px;      /* w*h pixels, row-major */
    int x0, y0, w, h;  /* panel area covered by px */
    bool big_endian;   /* store byte-swapped (SPI panel order) */
} ha_canvas_t;

typedef struct { int x, y, w, h; } ha_rect_t;

void ha_gfx_fill(ha_canvas_t *c, uint16_t color);
void ha_gfx_rect(ha_canvas_t *c, int x, int y, int w, int h, uint16_t color);
void ha_gfx_circle(ha_canvas_t *c, int cx, int cy, int r, uint16_t color);
/* Ring of all pixels between radii r_in..r_out (inclusive, like drawCircle loop). */
void ha_gfx_annulus(ha_canvas_t *c, int cx, int cy, int r_in, int r_out, uint16_t color);
void ha_gfx_round_rect(ha_canvas_t *c, int x, int y, int w, int h, int r, uint16_t color);
void ha_gfx_triangle(ha_canvas_t *c, int x0, int y0, int x1, int y1, int x2, int y2,
                     uint16_t color);

/* Ink bounds of a UTF-8 string drawn with the pen at (0, 0) on the baseline. */
void ha_gfx_text_bounds(const ha_font_t *f, const char *utf8, int *x1, int *y1, int *w, int *h);
/* Draw with the pen origin on the baseline; anti-aliased over existing pixels. */
void ha_gfx_text(ha_canvas_t *c, const ha_font_t *f, const char *utf8, int x, int y,
                 uint16_t color);
/* Centre the ink box of the string at (cx, cy) (demo centerText()). */
void ha_gfx_text_center(ha_canvas_t *c, const ha_font_t *f, const char *utf8, int cx, int cy,
                        uint16_t color);

/* Pixel read-back (host tests / preview); returns native RGB565. */
uint16_t ha_gfx_get(const ha_canvas_t *c, int x, int y);

#ifdef __cplusplus
}
#endif
