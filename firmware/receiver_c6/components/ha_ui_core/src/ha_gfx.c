#include "ha_gfx.h"

#include <stddef.h>

static inline uint16_t swap16(uint16_t v) { return (uint16_t)((v << 8) | (v >> 8)); }

static inline uint16_t store(const ha_canvas_t *c, uint16_t color) {
    return c->big_endian ? swap16(color) : color;
}

uint16_t ha_gfx_get(const ha_canvas_t *c, int x, int y) {
    int lx = x - c->x0, ly = y - c->y0;
    if (lx < 0 || ly < 0 || lx >= c->w || ly >= c->h) return 0;
    uint16_t v = c->px[ly * c->w + lx];
    return c->big_endian ? swap16(v) : v;
}

/* Horizontal span [xa, xb] on row y, clipped to the canvas. */
static void hspan(ha_canvas_t *c, int y, int xa, int xb, uint16_t v) {
    if (y < c->y0 || y >= c->y0 + c->h) return;
    if (xa < c->x0) xa = c->x0;
    if (xb > c->x0 + c->w - 1) xb = c->x0 + c->w - 1;
    if (xa > xb) return;
    uint16_t *p = c->px + (y - c->y0) * c->w + (xa - c->x0);
    for (int x = xa; x <= xb; ++x) *p++ = v;
}

/* Rows of the canvas that intersect [ya, yb]; false if none. */
static bool row_range(const ha_canvas_t *c, int *ya, int *yb) {
    if (*ya < c->y0) *ya = c->y0;
    if (*yb > c->y0 + c->h - 1) *yb = c->y0 + c->h - 1;
    return *ya <= *yb;
}

void ha_gfx_fill(ha_canvas_t *c, uint16_t color) {
    uint16_t v = store(c, color);
    for (int i = 0; i < c->w * c->h; ++i) c->px[i] = v;
}

void ha_gfx_rect(ha_canvas_t *c, int x, int y, int w, int h, uint16_t color) {
    if (w <= 0 || h <= 0) return;
    int ya = y, yb = y + h - 1;
    if (!row_range(c, &ya, &yb)) return;
    uint16_t v = store(c, color);
    for (int yy = ya; yy <= yb; ++yy) hspan(c, yy, x, x + w - 1, v);
}

static int isqrt(int n) {
    if (n <= 0) return 0;
    int x = n, y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}

/* Half-width of a filled circle of radius r at vertical offset dy.
 * Pixel centres with dx^2 + dy^2 <= r^2 + r (Adafruit fillCircle look). */
static int circle_half(int r, int dy) {
    int n = r * r + r - dy * dy;
    return n < 0 ? -1 : isqrt(n);
}

void ha_gfx_circle(ha_canvas_t *c, int cx, int cy, int r, uint16_t color) {
    int ya = cy - r, yb = cy + r;
    if (cx + r < c->x0 || cx - r >= c->x0 + c->w || !row_range(c, &ya, &yb)) return;
    uint16_t v = store(c, color);
    for (int y = ya; y <= yb; ++y) {
        int hw = circle_half(r, y - cy);
        if (hw >= 0) hspan(c, y, cx - hw, cx + hw, v);
    }
}

/* Largest h >= 0 with 4*h*h <= n4, or -1 if n4 < 0. */
static int max_half4(int n4) {
    if (n4 < 0) return -1;
    int h = isqrt(n4 / 4);
    while (4 * (h + 1) * (h + 1) <= n4) ++h;
    while (h > 0 && 4 * h * h > n4) --h;
    return h;
}

void ha_gfx_annulus(ha_canvas_t *c, int cx, int cy, int r_in, int r_out, uint16_t color) {
    int ya = cy - r_out, yb = cy + r_out;
    if (!row_range(c, &ya, &yb)) return;
    uint16_t v = store(c, color);
    // Pixel centres with (r_in - 0.5)^2 <= dx^2 + dy^2 <= (r_out + 0.5)^2, in 4x units.
    int lo4 = (2 * r_in - 1) * (2 * r_in - 1), hi4 = (2 * r_out + 1) * (2 * r_out + 1);
    for (int y = ya; y <= yb; ++y) {
        int dy4 = 4 * (y - cy) * (y - cy);
        int ho = max_half4(hi4 - dy4);
        if (ho < 0) continue;
        int hole = lo4 - dy4; // strictly inside the hole: 4dx^2 < hole
        int hi = hole > 0 ? max_half4(hole - 1) : -1;
        if (hi < 0) {
            hspan(c, y, cx - ho, cx + ho, v);
        } else {
            hspan(c, y, cx - ho, cx - hi - 1, v);
            hspan(c, y, cx + hi + 1, cx + ho, v);
        }
    }
}

void ha_gfx_round_rect(ha_canvas_t *c, int x, int y, int w, int h, int r, uint16_t color) {
    if (w <= 0 || h <= 0) return;
    if (r > w / 2) r = w / 2;
    if (r > h / 2) r = h / 2;
    int ya = y, yb = y + h - 1;
    if (!row_range(c, &ya, &yb)) return;
    uint16_t v = store(c, color);
    for (int yy = ya; yy <= yb; ++yy) {
        int inset = 0;
        int dy = 0;
        if (yy < y + r) dy = y + r - yy;
        else if (yy > y + h - 1 - r) dy = yy - (y + h - 1 - r);
        if (dy) {
            int hw = circle_half(r, dy);
            inset = hw < 0 ? w : r - hw;
        }
        hspan(c, yy, x + inset, x + w - 1 - inset, v);
    }
}

/* x where edge (xa,ya)-(xb,yb) crosses row y (ya != yb), rounded toward a. */
static int edge_x(int xa, int ya, int xb, int yb, int y) {
    return xa + (int)((long)(xb - xa) * (y - ya) / (yb - ya));
}

void ha_gfx_triangle(ha_canvas_t *c, int x0, int y0, int x1, int y1, int x2, int y2,
                     uint16_t color) {
    // Sort by y (Adafruit fillTriangle).
    int t;
#define SWAPI(a, b) (t = (a), (a) = (b), (b) = t)
    if (y0 > y1) { SWAPI(y0, y1); SWAPI(x0, x1); }
    if (y1 > y2) { SWAPI(y2, y1); SWAPI(x2, x1); }
    if (y0 > y1) { SWAPI(y0, y1); SWAPI(x0, x1); }
#undef SWAPI
    int ya = y0, yb = y2;
    if (!row_range(c, &ya, &yb)) return;
    uint16_t v = store(c, color);
    for (int y = ya; y <= yb; ++y) {
        int a, b;
        a = (y2 == y0) ? x0 : edge_x(x0, y0, x2, y2, y);
        if (y < y1 || (y == y1 && y1 != y0)) b = (y1 == y0) ? x1 : edge_x(x0, y0, x1, y1, y);
        else b = (y2 == y1) ? x1 : edge_x(x1, y1, x2, y2, y);
        if (y0 == y2) { // degenerate flat triangle
            a = x0 < x1 ? x0 : x1;
            a = a < x2 ? a : x2;
            b = x0 > x1 ? x0 : x1;
            b = b > x2 ? b : x2;
        }
        if (a > b) { int s = a; a = b; b = s; }
        hspan(c, y, a, b, v);
    }
}

/* ---------------------------------------------------------------- text */

static uint32_t utf8_next(const char **s) {
    const unsigned char *p = (const unsigned char *)*s;
    uint32_t cp;
    int n;
    if (p[0] < 0x80) { cp = p[0]; n = 1; }
    else if ((p[0] & 0xE0) == 0xC0) { cp = p[0] & 0x1Fu; n = 2; }
    else if ((p[0] & 0xF0) == 0xE0) { cp = p[0] & 0x0Fu; n = 3; }
    else { cp = p[0] & 0x07u; n = 4; }
    for (int i = 1; i < n; ++i) {
        if ((p[i] & 0xC0) != 0x80) { *s += i; return 0xFFFD; }
        cp = (cp << 6) | (p[i] & 0x3Fu);
    }
    *s += n;
    return cp;
}

const ha_glyph_t *ha_font_find(const ha_font_t *f, uint32_t cp) {
    int lo = 0, hi = (int)f->count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (f->glyphs[mid].cp == cp) return &f->glyphs[mid];
        if (f->glyphs[mid].cp < cp) lo = mid + 1;
        else hi = mid - 1;
    }
    return NULL;
}

void ha_gfx_text_bounds(const ha_font_t *f, const char *s, int *x1, int *y1, int *w, int *h) {
    int pen = 0, minx = 0, miny = 0, maxx = -1, maxy = -1;
    bool any = false;
    while (*s) {
        const ha_glyph_t *g = ha_font_find(f, utf8_next(&s));
        if (!g) continue;
        if (g->w && g->h) {
            int gx0 = pen + g->xo, gy0 = g->yo, gx1 = gx0 + g->w - 1, gy1 = gy0 + g->h - 1;
            if (!any || gx0 < minx) minx = gx0;
            if (!any || gy0 < miny) miny = gy0;
            if (!any || gx1 > maxx) maxx = gx1;
            if (!any || gy1 > maxy) maxy = gy1;
            any = true;
        }
        pen += g->adv;
    }
    *x1 = minx;
    *y1 = miny;
    *w = any ? maxx - minx + 1 : 0;
    *h = any ? maxy - miny + 1 : 0;
}

static uint16_t blend565(uint16_t fg, uint16_t bg, unsigned a /* 0..255 */) {
    unsigned rf = fg >> 11, gf = (fg >> 5) & 63, bf = fg & 31;
    unsigned rb = bg >> 11, gb = (bg >> 5) & 63, bb = bg & 31;
    unsigned r = (rf * a + rb * (255 - a) + 127) / 255;
    unsigned g = (gf * a + gb * (255 - a) + 127) / 255;
    unsigned b = (bf * a + bb * (255 - a) + 127) / 255;
    return (uint16_t)((r << 11) | (g << 5) | b);
}

static void draw_glyph(ha_canvas_t *c, const ha_font_t *f, const ha_glyph_t *g, int ox, int oy,
                       uint16_t color) {
    int gx = ox + g->xo, gy = oy + g->yo;
    int ya = gy, yb = gy + g->h - 1;
    if (!row_range(c, &ya, &yb)) return;
    int xa = gx < c->x0 ? c->x0 : gx;
    int xb = gx + g->w - 1;
    if (xb > c->x0 + c->w - 1) xb = c->x0 + c->w - 1;
    for (int y = ya; y <= yb; ++y) {
        uint32_t row = (uint32_t)(y - gy) * g->w;
        uint16_t *dst = c->px + (y - c->y0) * c->w;
        for (int x = xa; x <= xb; ++x) {
            uint32_t i = row + (uint32_t)(x - gx);
            uint8_t byte = f->bits[g->off + i / 2];
            unsigned a = (i & 1) ? (byte & 0x0Fu) : (byte >> 4);
            if (!a) continue;
            uint16_t *p = &dst[x - c->x0];
            if (a == 15) {
                *p = store(c, color);
            } else {
                uint16_t bg = c->big_endian ? swap16(*p) : *p;
                *p = store(c, blend565(color, bg, a * 17u));
            }
        }
    }
}

void ha_gfx_text(ha_canvas_t *c, const ha_font_t *f, const char *s, int x, int y,
                 uint16_t color) {
    while (*s) {
        const ha_glyph_t *g = ha_font_find(f, utf8_next(&s));
        if (!g) continue;
        if (g->w && g->h) draw_glyph(c, f, g, x, y, color);
        x += g->adv;
    }
}

void ha_gfx_text_center(ha_canvas_t *c, const ha_font_t *f, const char *s, int cx, int cy,
                        uint16_t color) {
    int x1, y1, w, h;
    ha_gfx_text_bounds(f, s, &x1, &y1, &w, &h);
    ha_gfx_text(c, f, s, cx - x1 - w / 2, cy - y1 - h / 2, color);
}
