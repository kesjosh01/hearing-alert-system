/*
 * GC9A01 1.28" 240x240 round SPI LCD driver for the C6 receiver.
 *
 * No frame buffer: ha_display_draw() renders the scene band by band
 * (ha_ui_core rasteriser) into two small DMA buffers and streams them, so one
 * band is drawn while the previous one is being sent. It returns only when
 * every pixel has been transferred to the panel, which is what the UI uses
 * as "the alert is on screen" before sending the ACK.
 */
#pragma once

#include <stdbool.h>

#include "esp_err.h"
#include "ha_screens.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int sck, mosi, cs, dc, rst, blk; /* GPIOs; blk < 0 = backlight hard-wired */
    int spi_hz;
    unsigned madctl;                 /* 0x08 = demo orientation (Arduino_GFX rotation 0, BGR) */
    bool invert;                     /* IPS panel: INVON (demo ips=true) */
    bool readback;                   /* read panel status over SDA (3-wire); verify on bench first */
} ha_display_cfg_t;

/* Reset + init the panel. The backlight stays off until ha_display_backlight(true). */
esp_err_t ha_display_start(const ha_display_cfg_t *cfg);
bool ha_display_ok(void); /* panel initialised (last init/re-init succeeded) */

/* Recovery ------------------------------------------------------------- */
/* Hardware reset (RES pin) + full init sequence. Content is lost: redraw. */
esp_err_t ha_display_reinit(void);
/* Light, idempotent re-send of SLPOUT/MADCTL/COLMOD/INVON/DISPON (~6 ms).
 * Done before each alert so a panel that reset itself shows the alert. */
esp_err_t ha_display_refresh(void);
/* Read-back health check (RDDPM). ESP_ERR_NOT_SUPPORTED if readback is off;
 * ESP_ERR_INVALID_RESPONSE if the panel does not answer as expected
 * (unplugged, reset, asleep). Without read-back an unplugged LCD cannot be
 * detected: SPI writes "succeed" with nothing attached. */
esp_err_t ha_display_check(void);

/* Draw the scene into `area` (NULL = whole panel). Blocks until transferred. */
esp_err_t ha_display_draw(const ha_scene_t *scene, const ha_rect_t *area);

/* Backlight + panel display on/off (off = DISPOFF and backlight low). */
void ha_display_backlight(bool on);

#ifdef __cplusplus
}
#endif
