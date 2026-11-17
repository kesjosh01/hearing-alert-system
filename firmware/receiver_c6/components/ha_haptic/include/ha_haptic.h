/*
 * Coin vibration motor (DVM1027 via NPN transistor) pattern player.
 *
 * A dedicated task plays ha_vib_pattern_t step tables (ha_ui_core). play()/
 * stop() only post a command and return at once, so the UI task never waits
 * for a pattern and a fire alert can cut a doorbell pattern off immediately.
 * The motor is driven on/off by a plain GPIO (the demo patterns use duty
 * 0/255 only); the level is kept during automatic light-sleep.
 */
#pragma once

#include <stdbool.h>

#include "esp_err.h"
#include "ha_presenter.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Drive the motor pin to "off" right away (call first at boot). gpio < 0 = no motor. */
esp_err_t ha_haptic_init(int gpio, bool active_high);
/* Start the player task. max_play_ms > 0 stops any pattern after that long (safety). */
esp_err_t ha_haptic_start(uint32_t max_play_ms);
bool ha_haptic_ok(void);

void ha_haptic_play(ha_pattern_id_t id); /* restarts from step 0 */
void ha_haptic_stop(void);

#ifdef __cplusplus
}
#endif
