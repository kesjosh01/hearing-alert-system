/*
 * GC9A01 over esp_lcd panel IO (SPI). The vendor register table is taken
 * verbatim from Espressif's esp_lcd_gc9a01 component (esp-bsp,
 * SPDX-FileCopyrightText: 2021-2026 Espressif Systems (Shanghai) CO LTD,
 * SPDX-License-Identifier: Apache-2.0). MADCTL 0x08 and INVON reproduce the
 * breadboard demo (Arduino_GFX GC9A01, rotation 0, ips=true).
 */
#include "ha_display.h"

#include <string.h>

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_heap_caps.h"
#include "esp_attr.h"
#include "esp_lcd_panel_io.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

static const char *TAG = "ha_display";

#define LCD_HOST SPI2_HOST
#define BAND_PIXELS (HA_LCD_W * 20) /* 20 full rows = 9600 bytes per buffer */

#define CMD_RDDPM 0x0A
#define CMD_SLPOUT 0x11
#define CMD_INVON 0x21
#define CMD_DISPOFF 0x28
#define CMD_DISPON 0x29
#define CMD_CASET 0x2A
#define CMD_RASET 0x2B
#define CMD_RAMWR 0x2C
#define CMD_MADCTL 0x36
#define CMD_COLMOD 0x3A

typedef struct {
    uint8_t cmd;
    uint8_t data[12];
    uint8_t len;
} init_cmd_t;

// esp_lcd_gc9a01 vendor_specific_init_default[] (all delays 0 ms).
static const init_cmd_t k_vendor_init[] = {
    {0xfe, {0x00}, 0},
    {0xef, {0x00}, 0},
    {0xeb, {0x14}, 1},
    {0x84, {0x60}, 1},
    {0x85, {0xff}, 1},
    {0x86, {0xff}, 1},
    {0x87, {0xff}, 1},
    {0x8e, {0xff}, 1},
    {0x8f, {0xff}, 1},
    {0x88, {0x0a}, 1},
    {0x89, {0x23}, 1},
    {0x8a, {0x00}, 1},
    {0x8b, {0x80}, 1},
    {0x8c, {0x01}, 1},
    {0x8d, {0x03}, 1},
    {0x90, {0x08, 0x08, 0x08, 0x08}, 4},
    {0xff, {0x60, 0x01, 0x04}, 3},
    {0xC3, {0x13}, 1},
    {0xC4, {0x13}, 1},
    {0xC9, {0x30}, 1},
    {0xbe, {0x11}, 1},
    {0xe1, {0x10, 0x0e}, 2},
    {0xdf, {0x21, 0x0c, 0x02}, 3},
    {0xF0, {0x45, 0x09, 0x08, 0x08, 0x26, 0x2a}, 6},
    {0xF1, {0x43, 0x70, 0x72, 0x36, 0x37, 0x6f}, 6},
    {0xF2, {0x45, 0x09, 0x08, 0x08, 0x26, 0x2a}, 6},
    {0xF3, {0x43, 0x70, 0x72, 0x36, 0x37, 0x6f}, 6},
    {0xed, {0x1b, 0x0b}, 2},
    {0xae, {0x77}, 1},
    {0xcd, {0x63}, 1},
    {0x70, {0x07, 0x07, 0x04, 0x0e, 0x0f, 0x09, 0x07, 0x08, 0x03}, 9},
    {0xE8, {0x34}, 1},
    {0x60, {0x38, 0x0b, 0x6D, 0x6D, 0x39, 0xf0, 0x6D, 0x6D}, 8},
    {0x61, {0x38, 0xf4, 0x6D, 0x6D, 0x38, 0xf7, 0x6D, 0x6D}, 8},
    {0x62, {0x38, 0x0D, 0x71, 0xED, 0x70, 0x70, 0x38, 0x0F, 0x71, 0xEF, 0x70, 0x70}, 12},
    {0x63, {0x38, 0x11, 0x71, 0xF1, 0x70, 0x70, 0x38, 0x13, 0x71, 0xF3, 0x70, 0x70}, 12},
    {0x64, {0x28, 0x29, 0xF1, 0x01, 0xF1, 0x00, 0x07}, 7},
    {0x66, {0x3C, 0x00, 0xCD, 0x67, 0x45, 0x45, 0x10, 0x00, 0x00, 0x00}, 10},
    {0x67, {0x00, 0x3C, 0x00, 0x00, 0x00, 0x01, 0x54, 0x10, 0x32, 0x98}, 10},
    {0x74, {0x10, 0x45, 0x80, 0x00, 0x00, 0x4E, 0x00}, 7},
    {0x98, {0x3e, 0x07}, 2},
    {0x99, {0x3e, 0x07}, 2},
};

static esp_lcd_panel_io_handle_t s_io;
static uint16_t *s_buf[2];
static SemaphoreHandle_t s_free; /* counts buffers not owned by DMA */
static int s_blk = -1;
static ha_display_cfg_t s_cfg;
static bool s_io_ready; /* SPI bus + panel IO exist (re-init possible) */
static bool s_ok;       /* panel initialised */
static bool s_disp_on;  /* DISPON last sent (for the read-back check) */
static void probe_log(void);

static bool IRAM_ATTR on_trans_done(esp_lcd_panel_io_handle_t io,
                                    esp_lcd_panel_io_event_data_t *edata, void *ctx) {
    (void)io;
    (void)edata;
    (void)ctx;
    BaseType_t woken = pdFALSE;
    xSemaphoreGiveFromISR(s_free, &woken);
    return woken == pdTRUE;
}

static esp_err_t tx(uint8_t cmd, const void *data, size_t len) {
    return esp_lcd_panel_io_tx_param(s_io, cmd, data, len);
}

/* (Re)create the panel IO at the given SPI clock. The GC9A01 serial READ
 * cycle is much slower than its write cycle (about 150 ns, ~6.6 MHz), so
 * reads briefly reopen the IO at HA read speed and then go back. Only called
 * from the drawing task between frames (no colour transfer in flight). */
static esp_err_t open_io(int hz) {
    if (s_io) {
        esp_lcd_panel_io_del(s_io);
        s_io = NULL;
    }
    esp_lcd_panel_io_spi_config_t io_cfg = {
        .cs_gpio_num = s_cfg.cs,
        .dc_gpio_num = s_cfg.dc,
        .spi_mode = 0,
        .pclk_hz = (unsigned)hz,
        .trans_queue_depth = 4,
        .on_color_trans_done = on_trans_done,
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
    };
    // Read-back needs the bidirectional SDA line (3-wire); the pin map has no MISO.
    io_cfg.flags.sio_mode = s_cfg.readback ? 1 : 0;
    return esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &io_cfg, &s_io);
}

#define READ_HZ 4000000

/* Read `n` parameter bytes of `cmd` at the slow read clock. */
static esp_err_t rx_slow(uint8_t cmd, uint8_t *buf, size_t n) {
    esp_err_t err = open_io(READ_HZ);
    if (err == ESP_OK) err = esp_lcd_panel_io_rx_param(s_io, cmd, buf, n);
    esp_err_t back = open_io(s_cfg.spi_hz);
    if (back != ESP_OK) {
        s_io_ready = false; // cannot draw any more; reported by ha_display_ok()
        s_ok = false;
        ESP_LOGE(TAG, "panel IO reopen failed: %s", esp_err_to_name(back));
        return back;
    }
    return err;
}

bool ha_display_ok(void) { return s_ok; }

/* Hardware reset (if RES is wired) + the full init sequence. */
static esp_err_t panel_init(void) {
    esp_err_t err;
    if (s_cfg.rst >= 0) {
        gpio_set_level(s_cfg.rst, 0);
        vTaskDelay(pdMS_TO_TICKS(10));
        gpio_set_level(s_cfg.rst, 1);
        vTaskDelay(pdMS_TO_TICKS(120));
    }
    // Order as esp_lcd_gc9a01: SLPOUT, MADCTL, COLMOD, vendor table; then INVON, DISPON.
    if ((err = tx(CMD_SLPOUT, NULL, 0)) != ESP_OK) return err;
    vTaskDelay(pdMS_TO_TICKS(120));
    const uint8_t madctl = (uint8_t)s_cfg.madctl, colmod = 0x55; // RGB565
    if ((err = tx(CMD_MADCTL, &madctl, 1)) != ESP_OK) return err;
    if ((err = tx(CMD_COLMOD, &colmod, 1)) != ESP_OK) return err;
    for (size_t i = 0; i < sizeof(k_vendor_init) / sizeof(k_vendor_init[0]); ++i)
        if ((err = tx(k_vendor_init[i].cmd, k_vendor_init[i].data, k_vendor_init[i].len)) != ESP_OK)
            return err;
    if (s_cfg.invert && (err = tx(CMD_INVON, NULL, 0)) != ESP_OK) return err;
    if ((err = tx(CMD_DISPON, NULL, 0)) != ESP_OK) return err;
    s_disp_on = true;
    vTaskDelay(pdMS_TO_TICKS(20));
    return ESP_OK;
}

esp_err_t ha_display_start(const ha_display_cfg_t *cfg) {
    if (s_ok) return ESP_OK;
    if (cfg->sck < 0 || cfg->mosi < 0 || cfg->cs < 0 || cfg->dc < 0) return ESP_ERR_INVALID_ARG;
    esp_err_t err;
    s_cfg = *cfg;
    s_blk = cfg->blk;
    if (s_blk >= 0) { // backlight off until the first frame is drawn
        gpio_set_level(s_blk, 0);
        const gpio_config_t bl = {.pin_bit_mask = 1ULL << s_blk, .mode = GPIO_MODE_OUTPUT};
        if ((err = gpio_config(&bl)) != ESP_OK) return err;
    }
    if (cfg->rst >= 0) {
        gpio_set_level(cfg->rst, 1);
        const gpio_config_t rst = {.pin_bit_mask = 1ULL << cfg->rst, .mode = GPIO_MODE_OUTPUT};
        if ((err = gpio_config(&rst)) != ESP_OK) return err;
    }
    const spi_bus_config_t bus = {
        .sclk_io_num = cfg->sck,
        .mosi_io_num = cfg->mosi,
        .miso_io_num = -1,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = BAND_PIXELS * 2,
    };
    if ((err = spi_bus_initialize(LCD_HOST, &bus, SPI_DMA_CH_AUTO)) != ESP_OK) return err;
    s_free = xSemaphoreCreateCounting(2, 2);
    s_buf[0] = heap_caps_malloc(BAND_PIXELS * 2, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    s_buf[1] = heap_caps_malloc(BAND_PIXELS * 2, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    if (!s_free || !s_buf[0] || !s_buf[1]) return ESP_ERR_NO_MEM;
    esp_err_t io_err = open_io(cfg->spi_hz);
    if (io_err != ESP_OK) return io_err;
    if (cfg->readback) gpio_pullup_en(cfg->mosi); // unplugged LCD reads 0xFF, not noise
    s_io_ready = true;

    // The panel itself may need a second try (power ramp, loose wire).
    for (int attempt = 1; attempt <= 3; ++attempt) {
        if ((err = panel_init()) == ESP_OK) break;
        ESP_LOGW(TAG, "panel init attempt %d failed: %s", attempt, esp_err_to_name(err));
        vTaskDelay(pdMS_TO_TICKS(200));
    }
    if (err != ESP_OK) return err;
    s_ok = true;
    ESP_LOGI(TAG, "GC9A01 ready: SCK=%d MOSI=%d CS=%d DC=%d RST=%d BLK=%d %d Hz MADCTL=0x%02x%s%s",
             cfg->sck, cfg->mosi, cfg->cs, cfg->dc, cfg->rst, cfg->blk, cfg->spi_hz,
             (unsigned)cfg->madctl, cfg->invert ? " INVON" : "",
             cfg->readback ? " readback=on" : "");
    if (cfg->readback) {
        probe_log();       // raw values for bench verification
        ha_display_check();
    }
    return ESP_OK;
}

esp_err_t ha_display_reinit(void) {
    if (!s_io_ready) return ESP_ERR_INVALID_STATE;
    esp_err_t err = panel_init();
    s_ok = (err == ESP_OK);
    ESP_LOGW(TAG, "panel re-init (hardware reset%s): %s", s_cfg.rst >= 0 ? "" : " not wired",
             esp_err_to_name(err));
    return err;
}

esp_err_t ha_display_refresh(void) {
    if (!s_ok) return ESP_ERR_INVALID_STATE;
    // Idempotent on a healthy panel; brings back one that reset itself
    // (supply dip from the motor, ESD) to sleep-out / RGB565 / display-on.
    const uint8_t madctl = (uint8_t)s_cfg.madctl, colmod = 0x55;
    esp_err_t err = tx(CMD_SLPOUT, NULL, 0);
    if (err == ESP_OK) vTaskDelay(pdMS_TO_TICKS(5)); // datasheet: 5 ms before next command
    if (err == ESP_OK) err = tx(CMD_MADCTL, &madctl, 1);
    if (err == ESP_OK) err = tx(CMD_COLMOD, &colmod, 1);
    if (err == ESP_OK && s_cfg.invert) err = tx(CMD_INVON, NULL, 0);
    if (err == ESP_OK && s_disp_on) err = tx(CMD_DISPON, NULL, 0);
    return err;
}

/* Bench diagnostic: read a few status registers (one extra byte each, so a
 * one-bit dummy-clock shift can be seen) and log them. MADCTL and COLMOD were
 * written by us, so their read-back tells whether reading works at all. */
static void probe_log(void) {
    static const struct { uint8_t cmd; uint8_t n; const char *name; } k_regs[] = {
        {0x04, 4, "RDDID"}, {0x09, 5, "RDDST"}, {0x0A, 2, "RDDPM"}, {0x0B, 2, "RDDMADCTL"},
        {0x0C, 2, "RDDCOLMOD"}, {0xDA, 2, "RDID1"}, {0xDB, 2, "RDID2"}, {0xDC, 2, "RDID3"},
    };
    for (size_t i = 0; i < sizeof(k_regs) / sizeof(k_regs[0]); ++i) {
        uint8_t b[5] = {0};
        esp_err_t err = rx_slow(k_regs[i].cmd, b, k_regs[i].n);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "probe %s(0x%02x): %s", k_regs[i].name, k_regs[i].cmd, esp_err_to_name(err));
            continue;
        }
        // First data byte as read, and as it would be after one dummy clock.
        uint8_t shifted = (uint8_t)((b[0] << 1) | (b[1] >> 7));
        ESP_LOGW(TAG, "probe %-9s(0x%02x): raw %02x %02x %02x %02x %02x | byte0=0x%02x "
                 "byte0_if_dummy_bit=0x%02x", k_regs[i].name, k_regs[i].cmd, b[0], b[1], b[2],
                 b[3], b[4], b[0], shifted);
    }
    ESP_LOGW(TAG, "probe read at %d Hz; no LCD reads 0xff (SDA pull-up). expects: RDDMADCTL=0x%02x RDDCOLMOD=0x55/0x05 RDDPM=0x9c (bits 0x14 set)",
             READ_HZ, (unsigned)s_cfg.madctl);
}

esp_err_t ha_display_check(void) {
    if (!s_cfg.readback) return ESP_ERR_NOT_SUPPORTED;
    if (!s_io_ready) return ESP_ERR_INVALID_STATE;
    uint8_t v = 0;
    esp_err_t err = rx_slow(CMD_RDDPM, &v, 1);
    if (err != ESP_OK) return err;
    // RDDPM: D4 sleep-out, D2 display-on, D6 idle, D5 partial. A missing panel
    // reads 0x00 or 0xFF (floating/pulled line); both fail this mask.
    const uint8_t want = (uint8_t)(0x10 | (s_disp_on ? 0x04 : 0x00));
    bool healthy = (v & 0x74) == want;
    ESP_LOG_LEVEL(healthy ? ESP_LOG_DEBUG : ESP_LOG_WARN, TAG, "RDDPM=0x%02x -> %s", v,
                  healthy ? "ok" : "panel NOT responding as expected");
    return healthy ? ESP_OK : ESP_ERR_INVALID_RESPONSE;
}

esp_err_t ha_display_draw(const ha_scene_t *scene, const ha_rect_t *area) {
    if (!s_ok) return ESP_ERR_INVALID_STATE;
    ha_rect_t r = area ? *area : (ha_rect_t){0, 0, HA_LCD_W, HA_LCD_H};
    if (r.w <= 0 || r.h <= 0) return ESP_OK;
    const int rows_per_band = BAND_PIXELS / r.w;
    esp_err_t err = ESP_OK;
    int which = 0;
    for (int y = r.y; y < r.y + r.h && err == ESP_OK; y += rows_per_band) {
        int h = (y + rows_per_band > r.y + r.h) ? r.y + r.h - y : rows_per_band;
        if (xSemaphoreTake(s_free, pdMS_TO_TICKS(1000)) != pdTRUE) {
            err = ESP_ERR_TIMEOUT;
            break;
        }
        // Render this band while the previous one is still being sent.
        ha_canvas_t c = {s_buf[which], r.x, y, r.w, h, true};
        ha_scene_draw(scene, &c);
        const int x1 = r.x + r.w - 1, y1 = y + h - 1;
        const uint8_t caset[4] = {r.x >> 8, r.x & 0xff, x1 >> 8, x1 & 0xff};
        const uint8_t raset[4] = {y >> 8, y & 0xff, y1 >> 8, y1 & 0xff};
        err = tx(CMD_CASET, caset, 4); // waits for queued colour data first
        if (err == ESP_OK) err = tx(CMD_RASET, raset, 4);
        if (err == ESP_OK)
            err = esp_lcd_panel_io_tx_color(s_io, CMD_RAMWR, s_buf[which], (size_t)r.w * h * 2);
        if (err != ESP_OK) xSemaphoreGive(s_free); // not queued: buffer is ours again
        which ^= 1;
    }
    // Wait until both buffers are back: every pixel has reached the panel.
    for (int i = 0; i < 2; ++i)
        if (xSemaphoreTake(s_free, pdMS_TO_TICKS(1000)) != pdTRUE) err = ESP_ERR_TIMEOUT;
    for (int i = 0; i < 2; ++i) xSemaphoreGive(s_free);
    if (err != ESP_OK) ESP_LOGE(TAG, "draw failed: %s", esp_err_to_name(err));
    return err;
}

void ha_display_backlight(bool on) {
    if (s_blk >= 0) gpio_set_level(s_blk, on ? 1 : 0);
    if (!s_ok) return;
    tx(on ? CMD_DISPON : CMD_DISPOFF, NULL, 0);
    s_disp_on = on;
}
