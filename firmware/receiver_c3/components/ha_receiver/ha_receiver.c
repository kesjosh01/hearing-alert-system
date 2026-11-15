/*
 * C3 receiver glue around the portable ha_rx / ha_time core.
 *
 * Tasks:
 *  - rx task: blocks in recvfrom() (no timeout, no polling) so the CPU can
 *    auto light-sleep; owns s_rx (duplicate state) and sends ACKs.
 *  - housekeeping task: HELLO, POWER_STATE, PM_STATS every HELLO interval and
 *    immediately after (re)acquiring an IP; enables auto light-sleep after the
 *    boot grace period.
 *  - Wi-Fi/IP event handler: only calls the non-blocking esp_wifi_connect()
 *    and sets event bits.
 *
 * Power: WIFI_PS_MIN_MODEM keeps the association and wakes the radio for each
 * DTIM beacon of the S3 AP (100 TU x DTIM 3 = 307.2 ms nominal). Together with
 * PM + tickless idle the CPU enters light-sleep automatically when idle. No
 * manual esp_light_sleep_start() loop is used.
 */
#include "ha_receiver.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_pm.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"

#include "ha_config.h"
#include "ha_proto.h"

#if !CONFIG_IDF_TARGET_ESP32C3
#error "ha_receiver is the ESP32-C3 STA receiver"
#endif
#if !CONFIG_PM_ENABLE || !CONFIG_FREERTOS_USE_TICKLESS_IDLE
#error "C3 receiver requires CONFIG_PM_ENABLE and CONFIG_FREERTOS_USE_TICKLESS_IDLE"
#endif
#if CONFIG_HA_RX_PM_REPORT && !CONFIG_PM_PROFILING
#error "CONFIG_HA_RX_PM_REPORT needs CONFIG_PM_PROFILING for the SDK sleep counters"
#endif

static const char *TAG = "ha_receiver";

#define GOT_IP_BIT BIT0
#define HELLO_DUE_BIT BIT1

static EventGroupHandle_t s_bits;
static QueueHandle_t s_alert_q;
static SemaphoreHandle_t s_state_lock; /* stats copy + clock */
static volatile bool s_consumer;
static ha_rx_t s_rx;
static ha_clock_t s_clock;
static uint32_t s_c3_session;
static int s_fd = -1;
static struct sockaddr_in s_ap;
static uint32_t s_not_accepted_logged;

static int64_t now_ms(void) { return esp_timer_get_time() / 1000; }

/* ------------------------------------------------------------- public API */

esp_err_t ha_receiver_attach_consumer(void) {
    if (!s_alert_q) return ESP_ERR_INVALID_STATE;
    s_consumer = true;
    ESP_LOGI(TAG, "UI consumer attached; EVENTs will be accepted and ACKed");
    return ESP_OK;
}

void ha_receiver_detach_consumer(void) {
    s_consumer = false;
    ESP_LOGW(TAG, "UI consumer detached; EVENTs will not be ACKed");
}

bool ha_receiver_consumer_attached(void) { return s_consumer; }

bool ha_receiver_wait_alert(ha_alert_t *out, TickType_t wait) {
    return s_alert_q && xQueueReceive(s_alert_q, out, wait) == pdTRUE;
}

ha_clock_state_t ha_receiver_get_time(int64_t *utc_ms, uint32_t *uncertainty_ms) {
    if (!s_state_lock) return HA_CLOCK_UNSYNCED;
    xSemaphoreTake(s_state_lock, portMAX_DELAY);
    ha_clock_state_t st = ha_clock_now(&s_clock, now_ms(), utc_ms, uncertainty_ms);
    xSemaphoreGive(s_state_lock);
    return st;
}

void ha_receiver_get_stats(ha_rx_stats_t *out) {
    memset(out, 0, sizeof(*out));
    if (!s_state_lock) return;
    xSemaphoreTake(s_state_lock, portMAX_DELAY);
    *out = s_rx.stats;
    xSemaphoreGive(s_state_lock);
}

/* ------------------------------------------------------------------ power */

static void configure_pm(bool light_sleep) {
    const esp_pm_config_t cfg = {
        .max_freq_mhz = 160,
        .min_freq_mhz = 40,
        .light_sleep_enable = light_sleep,
    };
    ESP_ERROR_CHECK(esp_pm_configure(&cfg));
}

#if CONFIG_HA_RX_PM_REPORT
/* SDK successful/rejected light-sleep counters via the public profiling API
 * (same method as the 2026-10-30 verification). */
static bool get_sleep_counts(uint32_t *slept, uint32_t *rejected) {
    char *buffer = NULL;
    size_t length = 0;
    FILE *stream = open_memstream(&buffer, &length);
    if (!stream) return false;
    esp_err_t err = esp_pm_dump_locks(stream);
    fclose(stream);
    bool ok = false;
    if (err == ESP_OK && buffer) {
        const char *s = strstr(buffer, "light_sleep_counts:");
        ok = s && sscanf(s, "light_sleep_counts:%" SCNu32 " light_sleep_reject_counts:%" SCNu32,
                         slept, rejected) == 2;
    }
    free(buffer);
    return ok;
}
#endif

/* ------------------------------------------------------------------- send */

static void send_v1(uint8_t type, uint32_t session, uint32_t seq, uint32_t value) {
    ha_v1_frame_t f = {type, session, seq, value};
    uint8_t buf[HA_V1_LEN];
    ha_v1_encode(&f, buf);
    if (sendto(s_fd, buf, sizeof(buf), 0, (struct sockaddr *)&s_ap, sizeof(s_ap)) != sizeof(buf))
        ESP_LOGW(TAG, "sendto type=%u errno=%d", type, errno);
}

/* -------------------------------------------------------------- receive */

/* Consumer hand-off used by ha_rx_handle_event(): must not block. */
static bool accept_alert(void *ctx, const ha_v1_frame_t *f) {
    (void)ctx;
    if (!s_consumer) return false;
    ha_alert_t a = {f->value, f->session, f->seq, now_ms()};
    return xQueueSend(s_alert_q, &a, 0) == pdTRUE;
}

static void handle_event(const ha_v1_frame_t *f) {
    xSemaphoreTake(s_state_lock, portMAX_DELAY);
    ha_rx_outcome_t o = ha_rx_handle_event(&s_rx, f, accept_alert, NULL);
    xSemaphoreGive(s_state_lock);
    // ACK repeats the received three fields (only after acceptance/dup check).
    if (ha_rx_should_ack(o)) send_v1(HA_MSG_ACK, f->session, f->seq, f->value);
    switch (o) {
    case HA_RX_ACK_NEW:
        ESP_LOGI(TAG, "EVENT accepted id=0x%08" PRIx32 "(%s) session=%" PRIu32 " seq=%" PRIu32,
                 f->value, ha_event_name(f->value), f->session, f->seq);
        break;
    case HA_RX_ACK_DUPLICATE:
    case HA_RX_ACK_STALE:
        ESP_LOGI(TAG, "EVENT %s seq=%" PRIu32 "; re-ACK without new alert",
                 ha_rx_outcome_name(o), f->seq);
        break;
    case HA_RX_DROP_NOT_ACCEPTED:
        // Rate-limited: the S3 retransmits the same event until accepted.
        if (s_not_accepted_logged++ % 20 == 0)
            ESP_LOGW(TAG, "EVENT seq=%" PRIu32 " NOT accepted (%s); no ACK",
                     f->seq, s_consumer ? "alert queue full" : "no UI consumer attached");
        break;
    default:
        ESP_LOGW(TAG, "EVENT dropped (%s) session=%" PRIu32 " seq=%" PRIu32 " id=0x%08" PRIx32,
                 ha_rx_outcome_name(o), f->session, f->seq, f->value);
        break;
    }
}

static void handle_time(const uint8_t *buf, int n) {
    ha_time_sync_t m;
    ha_decode_result_t dr = ha_time_sync_decode(buf, (size_t)n, &m);
    if (dr != HA_DEC_OK) {
        ESP_LOGD(TAG, "TIME_SYNC dropped: %s", ha_decode_result_name(dr));
        return;
    }
    xSemaphoreTake(s_state_lock, portMAX_DELAY);
    ha_time_apply_t r = ha_clock_apply(&s_clock, &m, now_ms());
    xSemaphoreGive(s_state_lock);
    ESP_LOGI(TAG, "TIME_SYNC seq=%" PRIu32 " %s", m.seq,
             r == HA_TIME_APPLIED ? "applied" : "ignored (old)");
}

static void rx_task(void *arg) {
    (void)arg;
    for (;;) {
        uint8_t buf[HA_RX_BUFFER_LEN]; // one byte larger than any valid frame
        struct sockaddr_in from = {0};
        socklen_t from_len = sizeof(from);
        int n = recvfrom(s_fd, buf, sizeof(buf), 0, (struct sockaddr *)&from, &from_len);
        if (n < 0) {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                ESP_LOGW(TAG, "recvfrom errno=%d", errno);
                vTaskDelay(pdMS_TO_TICKS(100));
            }
            continue;
        }
        // Only the S3 AP address and the protocol port are accepted.
        if (from.sin_addr.s_addr != s_ap.sin_addr.s_addr || from.sin_port != s_ap.sin_port)
            continue;
        int ver = ha_frame_version(buf, (size_t)n);
        if (ver == HA_V2_VERSION) {
            handle_time(buf, n);
            continue;
        }
        ha_v1_frame_t f;
        ha_decode_result_t dr = ha_v1_decode(buf, (size_t)n, &f);
        if (dr != HA_DEC_OK) {
            ESP_LOGD(TAG, "drop len=%d: %s", n, ha_decode_result_name(dr));
            continue;
        }
        if (f.type == HA_MSG_EVENT) handle_event(&f);
        // HELLO/ACK/PM/POWER are C3->S3 only: wrong direction, ignored.
    }
}

/* ---------------------------------------------------------- housekeeping */

static void housekeeping_task(void *arg) {
    (void)arg;
    bool light_sleep = false;
    for (;;) {
        int64_t now = now_ms();
        TickType_t wait = pdMS_TO_TICKS(HA_HELLO_INTERVAL_MS);
        if (!light_sleep) {
            int64_t left = CONFIG_HA_RX_LIGHT_SLEEP_DELAY_MS - now;
            if (left <= 0) {
                configure_pm(true);
                light_sleep = true;
                ESP_LOGI(TAG, "auto light-sleep enabled; USB console may pause (see S3 C3_PM)");
            } else if (left < HA_HELLO_INTERVAL_MS) {
                wait = pdMS_TO_TICKS(left);
            }
        }
        if (xEventGroupGetBits(s_bits) & GOT_IP_BIT) {
            send_v1(HA_MSG_HELLO, s_c3_session, 0, 0);
            wifi_ps_type_t ps = WIFI_PS_NONE;
            esp_wifi_get_ps(&ps);
            send_v1(HA_MSG_POWER_STATE, s_c3_session, light_sleep ? 1 : 0, (uint32_t)ps);
#if CONFIG_HA_RX_PM_REPORT
            uint32_t slept = 0, rejected = 0;
            if (light_sleep && get_sleep_counts(&slept, &rejected))
                send_v1(HA_MSG_PM_STATS, s_c3_session, slept, rejected);
#endif
        }
        xEventGroupWaitBits(s_bits, HELLO_DUE_BIT, pdTRUE, pdFALSE, wait);
    }
}

/* ------------------------------------------------------------------ Wi-Fi */

static void net_event(void *arg, esp_event_base_t base, int32_t id, void *data) {
    (void)arg;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_bits, GOT_IP_BIT);
        const wifi_event_sta_disconnected_t *e = data;
        ESP_LOGW(TAG, "STA disconnected reason=%u rssi=%d", e->reason, e->rssi);
    }
    if (base == WIFI_EVENT && (id == WIFI_EVENT_STA_START || id == WIFI_EVENT_STA_DISCONNECTED)) {
        esp_err_t err = esp_wifi_connect(); // non-blocking; result arrives as an event
        if (err != ESP_OK) ESP_LOGW(TAG, "connect: %s", esp_err_to_name(err));
    }
    if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(s_bits, GOT_IP_BIT | HELLO_DUE_BIT); // HELLO immediately
        ESP_LOGI(TAG, "STA got IP; sending HELLO");
    }
}

static esp_err_t start_sta(void) {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    if (err != ESP_OK) return err;
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, net_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, net_event, NULL));

    size_t ssid_len = strlen(HA_WIFI_SSID), pass_len = strlen(HA_WIFI_PASSWORD);
    if (ssid_len == 0 || ssid_len > 32 || pass_len < 8 || pass_len > 63) {
        ESP_LOGE(TAG, "invalid SSID/password length (password not printed)");
        return ESP_ERR_INVALID_ARG;
    }
    wifi_config_t cfg = {0};
    memcpy(cfg.sta.ssid, HA_WIFI_SSID, ssid_len);
    memcpy(cfg.sta.password, HA_WIFI_PASSWORD, pass_len);
    cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    cfg.sta.channel = HA_WIFI_CHANNEL; // scan hint only
    // Only used by WIFI_PS_MAX_MODEM; advertised to the AP. MIN_MODEM follows DTIM.
    cfg.sta.listen_interval = HA_AP_DTIM_PERIOD;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &cfg));
    ESP_ERROR_CHECK(esp_wifi_start());
    // This SuperMini associated reliably at 11 dBm in the 2026-10 bench comparison.
    ESP_ERROR_CHECK(esp_wifi_set_max_tx_power(44));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_MIN_MODEM));
    ESP_LOGI(TAG, "C3 STA ready; SSID=%s UDP=%d MIN_MODEM (wake per AP DTIM; expected %" PRIu32
             "us with S3 defaults) tx_power=11dBm", HA_WIFI_SSID, HA_UDP_PORT,
             (uint32_t)HA_NOMINAL_WAKE_US);
    return ESP_OK;
}

#if CONFIG_HA_RX_DIAG_LOG_CONSUMER
/* DIAGNOSTIC ONLY (default off): drains alerts and logs them so radio tests
 * can run without UI hardware. This is NOT a display/vibration output. */
static void diag_consumer_task(void *arg) {
    (void)arg;
    ha_receiver_attach_consumer();
    for (;;) {
        ha_alert_t a;
        if (ha_receiver_wait_alert(&a, portMAX_DELAY))
            ESP_LOGW(TAG, "DIAG_CONSUMER id=0x%08" PRIx32 "(%s) seq=%" PRIu32
                     " (log only; no UI output)", a.event_id, ha_event_name(a.event_id), a.seq);
    }
}
#endif

esp_err_t ha_receiver_start(void) {
    if (s_alert_q) return ESP_OK;
    configure_pm(false); // boot grace window for flashing/logs; enabled later
    s_bits = xEventGroupCreate();
    s_alert_q = xQueueCreate(HA_RX_ALERT_QUEUE_LEN, sizeof(ha_alert_t));
    s_state_lock = xSemaphoreCreateMutex();
    if (!s_bits || !s_alert_q || !s_state_lock) return ESP_ERR_NO_MEM;
    ha_rx_init(&s_rx, HA_ALLOW_TEST_EVENT);
    ha_clock_config_t ccfg;
    ha_clock_default_config(&ccfg);
    ha_clock_init(&s_clock, &ccfg);
    s_c3_session = esp_random();
    s_ap = (struct sockaddr_in){.sin_family = AF_INET, .sin_port = htons(HA_UDP_PORT)};
    inet_pton(AF_INET, HA_AP_IP_STR, &s_ap.sin_addr);

    esp_err_t err = start_sta();
    if (err != ESP_OK) return err;
    s_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (s_fd < 0) return ESP_FAIL;
    struct sockaddr_in local = {.sin_family = AF_INET, .sin_port = htons(HA_UDP_PORT),
                                .sin_addr.s_addr = htonl(INADDR_ANY)};
    if (bind(s_fd, (struct sockaddr *)&local, sizeof(local)) != 0) {
        ESP_LOGE(TAG, "bind errno=%d", errno);
        close(s_fd);
        return ESP_FAIL;
    }
    if (xTaskCreate(rx_task, "ha_rx", 4096, NULL, 6, NULL) != pdPASS ||
        xTaskCreate(housekeeping_task, "ha_hk", 3072, NULL, 4, NULL) != pdPASS)
        return ESP_ERR_NO_MEM;
    ESP_LOGI(TAG, "receiver started c3_session=%" PRIu32 " alert_queue=%d", s_c3_session,
             HA_RX_ALERT_QUEUE_LEN);
#if CONFIG_HA_RX_DIAG_LOG_CONSUMER
    ESP_LOGW(TAG, "DIAGNOSTIC log consumer ENABLED (no real UI output)");
    xTaskCreate(diag_consumer_task, "ha_diag_ui", 3072, NULL, 3, NULL);
#endif
#if CONFIG_HA_DIAG_ALLOW_TEST_EVENT
    ESP_LOGW(TAG, "DIAGNOSTIC TEST(1) event id ENABLED (not a product build)");
#endif
    return ESP_OK;
}
