/*
 * C6 UI glue: receiver alerts + button + timers -> presenter -> display/motor,
 * and the ACK after the alert's screen has been drawn.
 *
 * Tasks
 *  - ui        (prio 5): the only task that draws. Blocks on its event queue
 *                        with a timeout = next deadline (blink step, hold time,
 *                        next clock minute), so it sleeps when nothing is due.
 *  - ui_pump   (prio 5): ha_receiver_wait_alert(portMAX_DELAY) -> event queue.
 *  - ui_button (prio 5): debounces the button after its interrupt.
 *  - ui_guard  (prio 6): display watchdog. If an accepted alert is still not
 *                        on screen (= not ACKed) after a few seconds it asks
 *                        the UI to re-initialise the LCD, then reboots the C6
 *                        (bounded; see ha_recovery.h). Independent of `ui`, so
 *                        a hung UI task is caught too.
 *  - ha_haptic (prio 4): plays vibration patterns (ha_haptic.c).
 */
#include "ha_ui.h"

#include <inttypes.h>
#include <string.h>

#include "driver/gpio.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_attr.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_sleep.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "ha_display.h"
#include "ha_events.h"
#include "ha_haptic.h"
#include "ha_presenter.h"
#include "ha_receiver.h"
#include "ha_recovery.h"
#include "ha_screens.h"

#if CONFIG_HA_UI_REBOOT_AFTER_MS <= CONFIG_HA_UI_RECOVER_AFTER_MS
#error "HA_UI_REBOOT_AFTER_MS must be larger than HA_UI_RECOVER_AFTER_MS"
#endif
#if CONFIG_HA_RX_DIAG_LOG_CONSUMER
#error "Disable HA_RX_DIAG_LOG_CONSUMER: the real UI consumes (and ACKs) the alerts"
#endif

static const char *TAG = "ha_ui";

typedef enum { EV_ALERT, EV_BUTTON, EV_RECOVER } ev_type_t;
typedef struct {
    ev_type_t type;
    ha_alert_t alert;
} ui_ev_t;

static QueueHandle_t s_q;
static TaskHandle_t s_button_task;
static ha_presenter_t s_pres;
static ha_scene_t s_scene;
static bool s_backlight;
static int64_t s_next_blink = INT64_MAX, s_next_watch = 0, s_backlight_off_at = INT64_MAX;
static uint32_t s_acked, s_not_acked;

/* Alerts accepted but not (yet) on screen. ACKed later if a recovery redraw
 * shows them while they are still represented on the current screen. */
#define UNSHOWN_MAX 8
static ha_alert_t s_unshown[UNSHOWN_MAX];
static int s_unshown_n;
static bool s_display_suspect;          /* an alert draw failed: keep retrying the LCD */
static int64_t s_next_display_retry = INT64_MAX;

/* Display watchdog state. The reboot counter survives esp_restart(). */
#define REC_MAGIC 0x48414443u /* "HADC" */
static RTC_NOINIT_ATTR uint32_t s_rtc_magic;
static RTC_NOINIT_ATTR uint32_t s_rtc_reboots;
static TaskHandle_t s_guard_task;
static volatile bool s_displayed_ok; /* ui -> guard: an alert reached the screen */
static bool s_degraded;

static int64_t now_ms(void) { return esp_timer_get_time() / 1000; }

/* ------------------------------------------------------------- battery */

static adc_oneshot_unit_handle_t s_adc;
static adc_cali_handle_t s_cali;
static adc_channel_t s_adc_ch;
static bool s_batt_ok;

static void battery_init(void) {
#if CONFIG_HA_UI_BATT_DIVIDER_X100 > 0
    if (CONFIG_HA_UI_GPIO_BATT_ADC < 0) return;
    adc_unit_t unit;
    if (adc_oneshot_io_to_channel(CONFIG_HA_UI_GPIO_BATT_ADC, &unit, &s_adc_ch) != ESP_OK) return;
    const adc_oneshot_unit_init_cfg_t ucfg = {.unit_id = unit};
    const adc_oneshot_chan_cfg_t ccfg = {.atten = ADC_ATTEN_DB_12, .bitwidth = ADC_BITWIDTH_DEFAULT};
    const adc_cali_curve_fitting_config_t cal = {
        .unit_id = unit, .chan = s_adc_ch, .atten = ADC_ATTEN_DB_12, .bitwidth = ADC_BITWIDTH_DEFAULT};
    if (adc_oneshot_new_unit(&ucfg, &s_adc) != ESP_OK ||
        adc_oneshot_config_channel(s_adc, s_adc_ch, &ccfg) != ESP_OK ||
        adc_cali_create_scheme_curve_fitting(&cal, &s_cali) != ESP_OK) {
        ESP_LOGW(TAG, "battery ADC unavailable");
        return;
    }
    s_batt_ok = true;
#endif
}

static int battery_pct(void) {
    if (!s_batt_ok) return -1;
    int sum = 0, n = 0;
    for (int i = 0; i < 8; ++i) {
        int mv;
        if (adc_oneshot_get_calibrated_result(s_adc, s_cali, s_adc_ch, &mv) == ESP_OK) {
            sum += mv;
            ++n;
        }
    }
    if (!n) return -1;
    int vbat = (sum / n) * CONFIG_HA_UI_BATT_DIVIDER_X100 / 100;
    return ha_battery_pct(vbat, CONFIG_HA_UI_BATT_EMPTY_MV, CONFIG_HA_UI_BATT_FULL_MV);
}

/* ---------------------------------------------------------------- time */

#if CONFIG_HA_UI_DEMO_BUILD_TIME
static int64_t s_build_utc;
static bool s_build_utc_ok;
#endif

/* Real time from S3 TIME_SYNC, else (demo option only) the build time. */
static bool current_utc(int64_t *utc) {
    uint32_t unc;
    ha_clock_state_t st = ha_receiver_get_time(utc, &unc);
    if (st == HA_CLOCK_SYNCED || st == HA_CLOCK_STALE) return true;
#if CONFIG_HA_UI_DEMO_BUILD_TIME
    if (s_build_utc_ok) {
        *utc = s_build_utc + (int64_t)CONFIG_HA_UI_DEMO_BUILD_TIME_OFFSET_S * 1000 + now_ms();
        return true;
    }
#endif
    return false;
}

/* Refresh clock/date/battery text. Returns which part of the watch changed:
 * 0 nothing, 1 text only, 2 battery (whole screen). */
static int watch_refresh(int64_t now) {
    char t[8], d[32];
    int64_t utc;
    if (current_utc(&utc)) {
        ha_watch_format(utc, CONFIG_HA_UI_UTC_OFFSET_MIN, t, d);
        s_next_watch = now + ha_watch_ms_to_next_minute(utc, CONFIG_HA_UI_UTC_OFFSET_MIN) + 20;
    } else {
        ha_watch_format_unknown(t, d);
        s_next_watch = now + 30000; // look for TIME_SYNC again later
    }
    int pct = battery_pct();
    int changed = 0;
    if (strcmp(t, s_scene.time) || strcmp(d, s_scene.date)) changed = 1;
    if (pct != s_scene.battery_pct) changed = 2;
    memcpy(s_scene.time, t, sizeof(t));
    memcpy(s_scene.date, d, sizeof(d));
    s_scene.battery_pct = pct;
    return changed;
}

/* ------------------------------------------------------------- output */

static void backlight(bool on, int64_t now) {
    if (on != s_backlight) {
        ha_display_backlight(on);
        s_backlight = on;
    }
    s_backlight_off_at = INT64_MAX;
#if CONFIG_HA_UI_IDLE_BACKLIGHT_OFF_S > 0
    if (on && s_pres.state == HA_UI_IDLE)
        s_backlight_off_at = now + (int64_t)CONFIG_HA_UI_IDLE_BACKLIGHT_OFF_S * 1000;
#else
    (void)now;
#endif
}

static void display_failed(int64_t now) {
    if (!s_display_suspect)
        ESP_LOGE(TAG, "display problem: retrying the LCD every %d s", CONFIG_HA_UI_DISPLAY_RETRY_S);
    s_display_suspect = true;
    if (s_next_display_retry == INT64_MAX)
        s_next_display_retry = now + (int64_t)CONFIG_HA_UI_DISPLAY_RETRY_S * 1000;
}

/* Plain redraw (blink, clock). Never re-initialises: the retry timer does. */
static bool draw(const ha_rect_t *area) {
    return ha_display_ok() && ha_display_draw(&s_scene, area) == ESP_OK;
}

/* Panel answered as expected (or read-back is off and cannot tell). */
static bool panel_healthy(void) {
    esp_err_t e = ha_display_check();
    return e == ESP_OK || e == ESP_ERR_NOT_SUPPORTED;
}

/* Draw an alert screen as reliably as we can, before ACKing it:
 * refresh the panel state -> (read-back check) -> draw -> on any failure
 * one hardware reset + full init + redraw. */
static bool draw_alert(void) {
    if (!ha_display_ok() && ha_display_reinit() != ESP_OK) return false;
    ha_display_refresh();
    if (!panel_healthy() && ha_display_reinit() != ESP_OK) return false;
    if (ha_display_draw(&s_scene, NULL) == ESP_OK && panel_healthy()) return true;
    if (ha_display_reinit() != ESP_OK) return false;
    return ha_display_draw(&s_scene, NULL) == ESP_OK && panel_healthy();
}

/* Apply a presenter command. Returns true if the screen was drawn. */
static bool apply(ha_ui_cmd_t cmd, int64_t now) {
    if (cmd.haptic == HA_HAP_STOP) ha_haptic_stop();
    else if (cmd.haptic >= 0) ha_haptic_play((ha_pattern_id_t)cmd.haptic);
    if (cmd.screen == HA_UI_KEEP) return false;
    s_scene.screen = (ha_screen_t)cmd.screen;
    s_scene.blink = false; // every alert starts with the demo's first frame
    bool ok;
    if (s_scene.screen == HA_SCR_WATCH) {
        s_next_blink = INT64_MAX;
        s_unshown_n = 0; // alert over: what was never shown stays un-ACKed
        watch_refresh(now);
        ok = draw(NULL);
    } else {
        s_next_blink = now + HA_UI_BLINK_MS;
        ok = draw_alert();
    }
    backlight(true, now);
    ESP_LOGI(TAG, "screen %s%s", ha_screen_name(s_scene.screen), ok ? "" : " (NOT drawn)");
    return ok;
}

/* Is this alert visible on the screen currently shown? */
static bool represented(uint32_t event_id) {
    switch (s_scene.screen) {
    case HA_SCR_FIRE: return event_id == HA_EVENT_FIRE_ALARM;
    case HA_SCR_DOORBELL: return event_id == HA_EVENT_DOORBELL;
    case HA_SCR_FIRE_DOORBELL:
        return event_id == HA_EVENT_FIRE_ALARM || event_id == HA_EVENT_DOORBELL;
    default: return false;
    }
}

static void acked_one(const ha_alert_t *a) {
    if (ha_receiver_alert_displayed(a) == ESP_OK) {
        ++s_acked;
        s_displayed_ok = true; // display proven: refill the watchdog reboot budget
        if (s_degraded) ESP_LOGW(TAG, "display works again; leaving vibration-only mode");
        s_degraded = false;
        s_display_suspect = false;
        s_next_display_retry = INT64_MAX;
    }
}

/* After a successful (re)draw: ACK earlier alerts that are now visible. */
static void ack_unshown(void) {
    int keep = 0;
    for (int i = 0; i < s_unshown_n; ++i) {
        if (represented(s_unshown[i].event_id)) {
            ESP_LOGW(TAG, "alert seq=%" PRIu32 " now on screen after recovery", s_unshown[i].seq);
            acked_one(&s_unshown[i]);
        } else {
            s_unshown[keep++] = s_unshown[i];
        }
    }
    s_unshown_n = keep;
}

static void on_alert(const ha_alert_t *a) {
    int64_t now = now_ms();
    ha_ui_cmd_t cmd = ha_presenter_on_alert(&s_pres, a->event_id, now);
    bool drawn = apply(cmd, now);
    if (cmd.shown && drawn) {
        acked_one(a); // ACK: the alert is on screen
        ack_unshown();
    } else if (a->event_id == HA_EVENT_TEST) {
        ESP_LOGW(TAG, "TEST event: no screen; confirmed for diagnostics only");
        ha_receiver_alert_displayed(a);
    } else {
        ++s_not_acked;
        if (cmd.shown && s_unshown_n < UNSHOWN_MAX) s_unshown[s_unshown_n++] = *a;
        display_failed(now);
        ESP_LOGE(TAG, "alert 0x%08" PRIx32 " seq=%" PRIu32 " NOT on screen -> no ACK yet "
                 "(vibration %s; watchdog will recover; total not shown=%" PRIu32 ")",
                 a->event_id, a->seq, ha_haptic_ok() ? "on" : "unavailable", s_not_acked);
    }
}

/* Full LCD re-init + redraw of whatever should be on screen now. */
static void recover_display(const char *why) {
    int64_t now = now_ms();
    ESP_LOGW(TAG, "display recovery (%s)", why);
    bool ok = ha_display_reinit() == ESP_OK;
    if (ok) {
        if (s_scene.screen == HA_SCR_WATCH) watch_refresh(now);
        ok = ha_display_draw(&s_scene, NULL) == ESP_OK && panel_healthy();
    }
    if (ok) {
        if (s_backlight) ha_display_backlight(true);
        else backlight(true, now);
        ack_unshown();
        if (s_unshown_n == 0) {
            s_display_suspect = false;
            s_next_display_retry = INT64_MAX;
        }
        ESP_LOGW(TAG, "display recovered");
    } else {
        s_display_suspect = true;
        s_next_display_retry = now + (int64_t)CONFIG_HA_UI_DISPLAY_RETRY_S * 1000;
        ESP_LOGE(TAG, "display recovery failed; next try in %d s", CONFIG_HA_UI_DISPLAY_RETRY_S);
    }
}

static void on_button(void) {
    int64_t now = now_ms();
#if CONFIG_HA_UI_DEMO_BUTTON_CYCLE
    // Demo: watch -> doorbell -> fire -> watch, local only (no radio, no ACK).
    if (s_pres.state == HA_UI_IDLE) {
        apply(ha_presenter_on_alert(&s_pres, HA_EVENT_DOORBELL, now), now);
        return;
    }
    if (s_pres.state == HA_UI_DOORBELL) {
        apply(ha_presenter_on_alert(&s_pres, HA_EVENT_FIRE_ALARM, now), now);
        return;
    }
#endif
    if (s_pres.state == HA_UI_IDLE) {
        backlight(true, now); // wake the watch face
        return;
    }
    apply(ha_presenter_on_button(&s_pres, now), now);
}

static void on_timers(int64_t now) {
    apply(ha_presenter_on_tick(&s_pres, now), now);
    if (s_pres.state != HA_UI_IDLE && now >= s_next_blink) {
        s_scene.blink = !s_scene.blink;
        draw(NULL);
        s_next_blink += HA_UI_BLINK_MS;
        if (s_next_blink <= now) s_next_blink = now + HA_UI_BLINK_MS; // fell behind
    }
    if (s_pres.state == HA_UI_IDLE && now >= s_next_watch) {
        int changed = watch_refresh(now);
        if (changed == 2) {
            draw(NULL);
        } else if (changed == 1) {
            ha_rect_t r = ha_scene_watch_text_rect();
            draw(&r);
        }
    }
    if (now >= s_next_display_retry) {
        s_next_display_retry = INT64_MAX;
        recover_display("periodic retry");
    }
    if (now >= s_backlight_off_at) {
        ha_display_backlight(false);
        s_backlight = false;
        s_backlight_off_at = INT64_MAX;
    }
}

static int64_t next_deadline(void) {
    int64_t d = ha_presenter_deadline(&s_pres);
    if (s_pres.state != HA_UI_IDLE) {
        if (s_next_blink < d) d = s_next_blink;
    } else if (s_next_watch < d) {
        d = s_next_watch;
    }
    if (s_backlight_off_at < d) d = s_backlight_off_at;
    if (s_next_display_retry < d) d = s_next_display_retry;
    return d;
}

static void ui_task(void *arg) {
    (void)arg;
    int64_t now = now_ms();
    watch_refresh(now);
    if (draw(NULL) && panel_healthy()) backlight(true, now);
    else display_failed(now); // no LCD / init failed: keep retrying in the background
    // Outputs are ready: from now on EVENTs are accepted (and ACKed once shown).
    ha_receiver_attach_consumer();
    for (;;) {
        int64_t d = next_deadline();
        TickType_t wait = portMAX_DELAY;
        if (d != INT64_MAX) {
            int64_t left = d - now_ms();
            wait = left > 0 ? pdMS_TO_TICKS(left) : 0;
        }
        ui_ev_t ev;
        if (xQueueReceive(s_q, &ev, wait) == pdTRUE) {
            if (ev.type == EV_ALERT) on_alert(&ev.alert);
            else if (ev.type == EV_RECOVER) recover_display("alert not on screen");
            else on_button();
        }
        on_timers(now_ms());
    }
}

static void pump_task(void *arg) {
    (void)arg;
    for (;;) {
        ui_ev_t ev = {.type = EV_ALERT};
        if (ha_receiver_wait_alert(&ev.alert, portMAX_DELAY)) {
            xQueueSend(s_q, &ev, portMAX_DELAY);
            if (s_guard_task) xTaskNotifyGive(s_guard_task); // start the display watchdog
        }
    }
}

/* ------------------------------------------------------- display watchdog */

static uint8_t boot_reboot_count(void) {
    esp_reset_reason_t why = esp_reset_reason();
    if (why == ESP_RST_SW && s_rtc_magic == REC_MAGIC && s_rtc_reboots <= 255) {
        ESP_LOGW(TAG, "restarted by the display watchdog (%u/%d consecutive)",
                 (unsigned)s_rtc_reboots, CONFIG_HA_UI_MAX_RECOVERY_REBOOTS);
        return (uint8_t)s_rtc_reboots;
    }
    s_rtc_magic = REC_MAGIC;
    s_rtc_reboots = 0;
    return 0;
}

#if CONFIG_HA_RX_ACK_ON_DISPLAY
static void guard_task(void *arg) {
    const ha_rec_cfg_t cfg = {CONFIG_HA_UI_RECOVER_AFTER_MS, CONFIG_HA_UI_REBOOT_AFTER_MS,
                              CONFIG_HA_UI_MAX_RECOVERY_REBOOTS};
    ha_rec_t rec;
    ha_rec_init(&rec, &cfg, (uint8_t)(uintptr_t)arg);
    for (;;) {
        if (s_displayed_ok) {
            s_displayed_ok = false;
            ha_rec_displayed_ok(&rec);
            s_rtc_reboots = 0;
        }
        int64_t since = 0, next, now = now_ms();
        bool waiting = ha_receiver_oldest_undisplayed(&since);
        ha_rec_action_t act = ha_rec_step(&rec, waiting, since, now, &next);
        if (act == HA_REC_ASK_UI) {
            ESP_LOGW(TAG, "watchdog: alert not on screen for %" PRId64 " ms -> LCD re-init",
                     now - since);
            ui_ev_t ev = {.type = EV_RECOVER};
            xQueueSend(s_q, &ev, 0);
        } else if (act == HA_REC_REBOOT) {
            ESP_LOGE(TAG, "watchdog: alert not on screen for %" PRId64 " ms -> REBOOT "
                     "(%u/%d); the S3 still holds the event and will resend it",
                     now - since, rec.reboots, CONFIG_HA_UI_MAX_RECOVERY_REBOOTS);
            s_rtc_magic = REC_MAGIC;
            s_rtc_reboots = rec.reboots;
            ha_haptic_stop();
            vTaskDelay(pdMS_TO_TICKS(100)); // let the log and motor-off go out
            esp_restart();
        } else if (act == HA_REC_DEGRADED) {
            s_degraded = true;
            ESP_LOGE(TAG, "watchdog: %d reboots did not bring the display back. "
                     "VIBRATION-ONLY mode: alerts still vibrate but are NOT ACKed; "
                     "LCD retried every %d s", CONFIG_HA_UI_MAX_RECOVERY_REBOOTS,
                     CONFIG_HA_UI_DISPLAY_RETRY_S);
        }
        TickType_t wait = portMAX_DELAY; // nothing waiting: sleep until the next alert
        if (next != INT64_MAX) {
            int64_t left = next - now_ms();
            wait = pdMS_TO_TICKS(left > 0 ? left + 10 : 10);
        }
        ulTaskNotifyTake(pdTRUE, wait);
    }
}
#endif /* CONFIG_HA_RX_ACK_ON_DISPLAY */

/* -------------------------------------------------------------- button */

static void button_isr(void *arg) {
    (void)arg;
    gpio_intr_disable(CONFIG_HA_UI_GPIO_BUTTON); // level interrupt: re-armed after release
    BaseType_t woken = pdFALSE;
    vTaskNotifyGiveFromISR(s_button_task, &woken);
    if (woken) portYIELD_FROM_ISR();
}

static void button_task(void *arg) {
    (void)arg;
    const int pin = CONFIG_HA_UI_GPIO_BUTTON;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        vTaskDelay(pdMS_TO_TICKS(30)); // debounce (demo: 30 ms)
        if (gpio_get_level(pin) == 0) {
            ui_ev_t ev = {.type = EV_BUTTON};
            xQueueSend(s_q, &ev, 0);
        }
        while (gpio_get_level(pin) == 0) vTaskDelay(pdMS_TO_TICKS(20)); // wait for release
        vTaskDelay(pdMS_TO_TICKS(30));
        gpio_intr_enable(pin);
    }
}

static esp_err_t button_init(void) {
    const int pin = CONFIG_HA_UI_GPIO_BUTTON;
    if (pin < 0) return ESP_ERR_NOT_SUPPORTED;
    // The module has its own pull-up; the internal one only helps if unplugged.
    const gpio_config_t io = {.pin_bit_mask = 1ULL << pin,
                              .mode = GPIO_MODE_INPUT,
                              .pull_up_en = GPIO_PULLUP_ENABLE,
                              .intr_type = GPIO_INTR_LOW_LEVEL};
    esp_err_t err = gpio_config(&io);
    if (err != ESP_OK) return err;
    if (xTaskCreate(button_task, "ui_button", 2048, NULL, 5, &s_button_task) != pdPASS)
        return ESP_ERR_NO_MEM;
    err = gpio_install_isr_service(0);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) return err;
    if ((err = gpio_isr_handler_add(pin, button_isr, NULL)) != ESP_OK) return err;
    // Wake from automatic light-sleep when pressed.
    gpio_wakeup_enable(pin, GPIO_INTR_LOW_LEVEL);
    esp_sleep_enable_gpio_wakeup();
    return ESP_OK;
}

/* ---------------------------------------------------------------- start */

esp_err_t ha_ui_early_init(void) {
    esp_err_t err = ha_haptic_init(CONFIG_HA_UI_GPIO_MOTOR, CONFIG_HA_UI_MOTOR_ACTIVE_HIGH);
    if (err != ESP_OK) ESP_LOGW(TAG, "motor pin: %s", esp_err_to_name(err));
    return err;
}

esp_err_t ha_ui_start(void) {
    if (s_q) return ESP_OK;
    s_q = xQueueCreate(CONFIG_HA_RX_ALERT_QUEUE_LEN + 2, sizeof(ui_ev_t));
    if (!s_q) return ESP_ERR_NO_MEM;

    const ha_display_cfg_t dcfg = {
        .sck = CONFIG_HA_DISP_GPIO_SCK, .mosi = CONFIG_HA_DISP_GPIO_MOSI,
        .cs = CONFIG_HA_DISP_GPIO_CS, .dc = CONFIG_HA_DISP_GPIO_DC,
        .rst = CONFIG_HA_DISP_GPIO_RST, .blk = CONFIG_HA_DISP_GPIO_BLK,
        .spi_hz = CONFIG_HA_DISP_SPI_HZ, .madctl = CONFIG_HA_DISP_MADCTL,
#if CONFIG_HA_DISP_INVERT
        .invert = true,
#endif
#if CONFIG_HA_DISP_READBACK
        .readback = true,
#endif
    };
    const uint8_t prior_reboots = boot_reboot_count();
    esp_err_t derr = ha_display_start(&dcfg);
    if (derr != ESP_OK)
        ESP_LOGE(TAG, "display FAILED (%s): alerts vibrate but are NOT ACKed until it "
                 "recovers (retry every %d s)", esp_err_to_name(derr), CONFIG_HA_UI_DISPLAY_RETRY_S);
    esp_err_t herr = ha_haptic_start(CONFIG_HA_UI_HAPTIC_MAX_MS);
    if (herr != ESP_OK) ESP_LOGE(TAG, "vibration FAILED (%s)", esp_err_to_name(herr));
    esp_err_t berr = button_init();
    if (berr != ESP_OK) ESP_LOGW(TAG, "button unavailable (%s)", esp_err_to_name(berr));
    battery_init();
#if CONFIG_HA_UI_DEMO_BUILD_TIME
    s_build_utc_ok = ha_watch_build_time_utc(__DATE__, __TIME__, CONFIG_HA_UI_UTC_OFFSET_MIN,
                                             &s_build_utc);
    ESP_LOGW(TAG, "DEMO clock from build time %s %s (+%ds); not a real time source",
             __DATE__, __TIME__, CONFIG_HA_UI_DEMO_BUILD_TIME_OFFSET_S);
#endif
#if CONFIG_HA_UI_SELFTEST
    if (ha_haptic_ok()) ha_haptic_play(HA_PAT_SELFTEST);
#endif
    if (!ha_display_ok() && !ha_haptic_ok()) {
        // Nothing can alert the user: stay detached so EVENTs are never ACKed.
        ESP_LOGE(TAG, "no output works; UI NOT attached (S3 log will show no ACK)");
        return ESP_FAIL;
    }

    const ha_presenter_cfg_t pcfg = {CONFIG_HA_UI_FIRE_HOLD_MS, CONFIG_HA_UI_DOORBELL_HOLD_MS};
    ha_presenter_init(&s_pres, &pcfg);
    s_scene.screen = HA_SCR_WATCH;
    s_scene.battery_pct = -1;
    if (xTaskCreate(ui_task, "ui", 4096, NULL, 5, NULL) != pdPASS)
        return ESP_ERR_NO_MEM;
#if CONFIG_HA_RX_ACK_ON_DISPLAY
    if (xTaskCreate(guard_task, "ui_guard", 3072, (void *)(uintptr_t)prior_reboots, 6,
                    &s_guard_task) != pdPASS)
        return ESP_ERR_NO_MEM;
#else
    (void)prior_reboots;
#endif
    if (xTaskCreate(pump_task, "ui_pump", 2048, NULL, 5, NULL) != pdPASS)
        return ESP_ERR_NO_MEM;
    ESP_LOGI(TAG, "UI started: display=%s motor=%s button=%s battery=%s; "
             "hold fire=%dms doorbell=%dms",
             ha_display_ok() ? "ok" : "FAIL", ha_haptic_ok() ? "ok" : "FAIL",
             berr == ESP_OK ? "ok" : "none", s_batt_ok ? "ok" : "not measured",
             CONFIG_HA_UI_FIRE_HOLD_MS, CONFIG_HA_UI_DOORBELL_HOLD_MS);
    return ESP_OK;
}
