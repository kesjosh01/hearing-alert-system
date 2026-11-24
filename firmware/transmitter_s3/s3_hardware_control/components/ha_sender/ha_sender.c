/*
 * S3 sender glue: SoftAP, UDP socket, FreeRTOS ownership around the portable
 * ha_tx core (firmware/common/ha_comm).
 *
 * Ownership:
 *  - s_tx (queue + peer state) is guarded by s_lock; every critical section is
 *    short and never contains a socket call or a user callback.
 *  - The sender task owns the socket and is the only sender of datagrams.
 *  - Wi-Fi event handlers only post to s_net_q (no waiting, no sending).
 *  - Outcome reports are queued (s_report_q) and delivered by the sender task,
 *    so a status callback may call ha_publish_event() without deadlocking.
 */
#include "ha_sender.h"

#include <errno.h>
#include <inttypes.h>
#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"

#include "ha_config.h"
#include "ha_proto.h"

#if !CONFIG_IDF_TARGET_ESP32S3
#error "ha_sender is the ESP32-S3 SoftAP transmitter"
#endif

static const char *TAG = "ha_sender";

#define LOCK_WAIT pdMS_TO_TICKS(20)
#define MAX_POLL_MS 20 /* S3 is mains powered; bounded idle wake for queue/timers */

typedef enum { NET_STA_JOINED, NET_STA_LEFT } net_evt_t;

static SemaphoreHandle_t s_lock;
static QueueHandle_t s_net_q;
static QueueHandle_t s_report_q;
static ha_tx_t s_tx;
static bool s_started;
static uint32_t s_reports_dropped;

static ha_sender_status_cb_t s_status_cb;
static void *s_status_ctx;
static ha_local_alert_hook_t s_local_hook;
static void *s_local_ctx;
static ha_time_provider_t s_time_provider;
static void *s_time_ctx;

static int64_t now_ms(void) { return esp_timer_get_time() / 1000; }

static uint32_t rand32(void *ctx) {
    (void)ctx;
    return esp_random();
}

/* Called by the core while s_lock is held: only enqueue. */
static void core_report(void *ctx, const ha_tx_report_t *r) {
    (void)ctx;
    if (xQueueSend(s_report_q, r, 0) != pdTRUE) ++s_reports_dropped;
}

static ha_publish_result_t map_pub(ha_pub_status_t s) {
    switch (s) {
    case HA_PUB_OK: return HA_PUBLISH_OK;
    case HA_PUB_OK_EVICTED: return HA_PUBLISH_OK_EVICTED_OLDEST;
    case HA_PUB_ERR_INVALID_ID: return HA_PUBLISH_ERR_INVALID_ID;
    case HA_PUB_ERR_QUEUE_FULL: return HA_PUBLISH_ERR_QUEUE_FULL;
    }
    return HA_PUBLISH_ERR_INVALID_ID;
}

static void local_alert(uint32_t id) {
    // S3's own indicator gets the same canonical ID and never waits for C3.
    if (s_local_hook && ha_event_allowed(id, HA_ALLOW_TEST_EVENT)) s_local_hook(id, s_local_ctx);
}

ha_publish_result_t ha_publish_event(uint32_t event_id, ha_event_token_t *token_out) {
    if (token_out) *token_out = 0;
    if (!s_started) return HA_PUBLISH_ERR_NOT_STARTED;
    local_alert(event_id);
    if (xSemaphoreTake(s_lock, LOCK_WAIT) != pdTRUE) return HA_PUBLISH_ERR_BUSY;
    uint32_t tok = 0;
    ha_pub_status_t st = ha_tx_publish(&s_tx, event_id, now_ms(), &tok);
    uint32_t queued = s_tx.count;
    xSemaphoreGive(s_lock);
    ha_publish_result_t r = map_pub(st);
    if (token_out) *token_out = tok;
    if (r == HA_PUBLISH_OK || r == HA_PUBLISH_OK_EVICTED_OLDEST)
        ESP_LOGI(TAG, "PUBLISH id=0x%08" PRIx32 "(%s) token=%" PRIu32 " queued=%" PRIu32 " %s",
                 event_id, ha_event_name(event_id), tok, queued, ha_publish_result_name(r));
    else
        ESP_LOGW(TAG, "PUBLISH_REJECTED id=0x%08" PRIx32 "(%s) reason=%s", event_id,
                 ha_event_name(event_id), ha_publish_result_name(r));
    return r;
}

size_t ha_publish_events(const uint32_t *ids, size_t count, ha_publish_result_t *results,
                         ha_event_token_t *tokens) {
    if (count > HA_TX_QUEUE_MAX) count = HA_TX_QUEUE_MAX;
    if (!s_started) {
        for (size_t i = 0; results && i < count; ++i) results[i] = HA_PUBLISH_ERR_NOT_STARTED;
        return 0;
    }
    for (size_t i = 0; i < count; ++i) local_alert(ids[i]);
    ha_pub_status_t st[HA_TX_QUEUE_MAX];
    uint32_t tok[HA_TX_QUEUE_MAX];
    if (xSemaphoreTake(s_lock, LOCK_WAIT) != pdTRUE) {
        for (size_t i = 0; results && i < count; ++i) results[i] = HA_PUBLISH_ERR_BUSY;
        return 0;
    }
    size_t accepted = ha_tx_publish_batch(&s_tx, ids, count, now_ms(), tok, st);
    xSemaphoreGive(s_lock);
    for (size_t i = 0; i < count; ++i) {
        ha_publish_result_t r = map_pub(st[i]);
        if (results) results[i] = r;
        if (tokens) tokens[i] = tok[i];
        ESP_LOGI(TAG, "PUBLISH_BATCH[%u] id=0x%08" PRIx32 "(%s) token=%" PRIu32 " %s",
                 (unsigned)i, ids[i], ha_event_name(ids[i]), tok[i], ha_publish_result_name(r));
    }
    return accepted;
}

esp_err_t ha_cancel_event(ha_event_token_t token) {
    if (!s_started) return ESP_ERR_INVALID_STATE;
    if (xSemaphoreTake(s_lock, LOCK_WAIT) != pdTRUE) return ESP_ERR_TIMEOUT;
    bool ok = ha_tx_cancel(&s_tx, token, now_ms());
    xSemaphoreGive(s_lock);
    return ok ? ESP_OK : ESP_ERR_NOT_FOUND;
}

void ha_sender_set_status_callback(ha_sender_status_cb_t cb, void *ctx) {
    s_status_ctx = ctx;
    s_status_cb = cb;
}

void ha_sender_set_local_alert_hook(ha_local_alert_hook_t hook, void *ctx) {
    s_local_ctx = ctx;
    s_local_hook = hook;
}

void ha_sender_set_time_provider(ha_time_provider_t provider, void *ctx) {
    s_time_ctx = ctx;
    s_time_provider = provider;
}

esp_err_t ha_sender_get_info(ha_sender_info_t *out) {
    if (!s_started) return ESP_ERR_INVALID_STATE;
    if (xSemaphoreTake(s_lock, LOCK_WAIT) != pdTRUE) return ESP_ERR_TIMEOUT;
    out->tx = s_tx.stats;
    out->queued = s_tx.count;
    out->peer_registered = s_tx.peer.registered;
    out->peer_associated = s_tx.peer.associated;
    out->peer_stale = s_tx.peer.stale;
    out->s3_session = s_tx.session;
    out->reports_dropped = s_reports_dropped;
    xSemaphoreGive(s_lock);
    return ESP_OK;
}

const char *ha_publish_result_name(ha_publish_result_t r) {
    switch (r) {
    case HA_PUBLISH_OK: return "ok";
    case HA_PUBLISH_OK_EVICTED_OLDEST: return "ok_evicted_oldest";
    case HA_PUBLISH_ERR_INVALID_ID: return "invalid_id";
    case HA_PUBLISH_ERR_QUEUE_FULL: return "queue_full";
    case HA_PUBLISH_ERR_NOT_STARTED: return "not_started";
    case HA_PUBLISH_ERR_BUSY: return "busy";
    }
    return "?";
}

/* ------------------------------------------------------------------ Wi-Fi */

static void wifi_event(void *arg, esp_event_base_t base, int32_t id, void *data) {
    (void)arg;
    (void)data;
    if (base != WIFI_EVENT) return;
    net_evt_t e;
    if (id == WIFI_EVENT_AP_STACONNECTED) e = NET_STA_JOINED;
    else if (id == WIFI_EVENT_AP_STADISCONNECTED) e = NET_STA_LEFT;
    else return;
    (void)xQueueSend(s_net_q, &e, 0); // never block the event loop
}

static esp_err_t start_softap(void) {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    if (err != ESP_OK) return err;
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();
    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event, NULL));

    size_t ssid_len = strlen(HA_WIFI_SSID), pass_len = strlen(HA_WIFI_PASSWORD);
    if (ssid_len == 0 || ssid_len > 32 || pass_len < 8 || pass_len > 63) {
        ESP_LOGE(TAG, "invalid SSID/password length (password not printed)");
        return ESP_ERR_INVALID_ARG;
    }
    wifi_config_t cfg = {0};
    memcpy(cfg.ap.ssid, HA_WIFI_SSID, ssid_len);
    memcpy(cfg.ap.password, HA_WIFI_PASSWORD, pass_len);
    cfg.ap.ssid_len = (uint8_t)ssid_len;
    cfg.ap.channel = HA_WIFI_CHANNEL;
    cfg.ap.max_connection = 1; // single C3 receiver in this phase
    cfg.ap.authmode = WIFI_AUTH_WPA2_PSK;
    // C3 (MIN_MODEM) wakes for every DTIM beacon: 100 TU x 3 = 307.2 ms nominal.
    cfg.ap.beacon_interval = HA_AP_BEACON_INTERVAL_TU;
    cfg.ap.dtim_period = HA_AP_DTIM_PERIOD;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &cfg));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_LOGI(TAG, "S3 AP ready; SSID=%s ch=%d UDP=%d beacon=%dTU DTIM=%d nominal_C3_wake=%" PRIu32
             "us", HA_WIFI_SSID, HA_WIFI_CHANNEL, HA_UDP_PORT, HA_AP_BEACON_INTERVAL_TU,
             HA_AP_DTIM_PERIOD, (uint32_t)HA_NOMINAL_WAKE_US);
    return ESP_OK;
}

/* ------------------------------------------------------------ sender task */

static void deliver_reports(void) {
    ha_tx_report_t r;
    while (xQueueReceive(s_report_q, &r, 0) == pdTRUE) {
        if (r.outcome == HA_TXR_ACKED)
            ESP_LOGI(TAG, "ACK seq=%" PRIu32 " id=0x%08" PRIx32 "(%s) token=%" PRIu32
                     " attempts=%" PRIu32 " last_tx_to_ack=%lldms age=%lldms",
                     r.seq, r.event_id, ha_event_name(r.event_id), r.token, r.attempts_total,
                     (long long)r.ack_after_last_tx_ms, (long long)r.age_ms);
        else
            ESP_LOGW(TAG, "%s seq=%" PRIu32 " id=0x%08" PRIx32 "(%s) token=%" PRIu32
                     " attempts=%" PRIu32 " age=%lldms",
                     ha_tx_outcome_name(r.outcome), r.seq, r.event_id, ha_event_name(r.event_id),
                     r.token, r.attempts_total, (long long)r.age_ms);
        ha_sender_status_cb_t cb = s_status_cb;
        if (cb) cb(&r, s_status_ctx);
    }
}

static void handle_datagram(const uint8_t *buf, int n, const struct sockaddr_in *from) {
    ha_v1_frame_t f;
    ha_decode_result_t dr = ha_v1_decode(buf, (size_t)n, &f);
    if (dr != HA_DEC_OK) {
        ESP_LOGD(TAG, "drop datagram len=%d: %s", n, ha_decode_result_name(dr));
        return;
    }
    uint32_t ip = from->sin_addr.s_addr;
    uint16_t port = from->sin_port;
    int64_t now = now_ms();
    xSemaphoreTake(s_lock, portMAX_DELAY);
    switch (f.type) {
    case HA_MSG_HELLO: {
        ha_hello_result_t hr = ha_tx_on_hello(&s_tx, &f, ip, port, now);
        uint32_t queued = s_tx.count;
        xSemaphoreGive(s_lock);
        if (hr != HA_HELLO_REFRESH && hr != HA_HELLO_INVALID)
            ESP_LOGI(TAG, "receiver registered (%s) c3_session=%" PRIu32 " queued=%" PRIu32,
                     ha_hello_result_name(hr), f.session, queued);
        return;
    }
    case HA_MSG_ACK: {
        ha_ack_result_t ar = ha_tx_on_ack(&s_tx, &f, ip, port, now);
        xSemaphoreGive(s_lock);
        if (ar != HA_ACK_MATCHED)
            ESP_LOGW(TAG, "ACK ignored (%s) session=%" PRIu32 " seq=%" PRIu32 " id=0x%08" PRIx32,
                     ha_ack_result_name(ar), f.session, f.seq, f.value);
        return;
    }
    case HA_MSG_PM_STATS:
    case HA_MSG_POWER_STATE: {
        bool ok = ha_tx_from_peer(&s_tx, &f, ip, port);
        xSemaphoreGive(s_lock);
        if (!ok) return;
        if (f.type == HA_MSG_PM_STATS)
            ESP_LOGI(TAG, "C3_PM session=%" PRIu32 " sleep_count=%" PRIu32 " reject_count=%" PRIu32,
                     f.session, f.seq, f.value);
        else if (f.seq <= 1 && f.value == WIFI_PS_MIN_MODEM)
            ESP_LOGI(TAG, "C3_STATE modem_sleep=MIN auto_light_sleep=%" PRIu32, f.seq);
        return;
    }
    default:
        xSemaphoreGive(s_lock); // EVENT is S3->C3 only: wrong direction
        return;
    }
}

static void maybe_send_time(int fd, int64_t now, int64_t *next_at, bool force) {
#if CONFIG_HA_TIME_SYNC_TX
    static uint32_t time_seq;
    ha_time_provider_t provider = s_time_provider;
    if (!provider || (!force && now < *next_at)) return;
    xSemaphoreTake(s_lock, portMAX_DELAY);
    // Lower priority than alerts: only on an idle, reachable link.
    bool ok = ha_tx_link_up(&s_tx) && ha_tx_idle(&s_tx);
    struct sockaddr_in to = {.sin_family = AF_INET, .sin_port = s_tx.peer.port,
                             .sin_addr.s_addr = s_tx.peer.ip};
    uint32_t session = s_tx.session;
    xSemaphoreGive(s_lock);
    if (!ok) return;
    int64_t utc = 0;
    uint32_t acc = 0;
    if (!provider(s_time_ctx, &utc, &acc) || utc < HA_UTC_MIN_VALID_MS) {
        *next_at = now + 10000; // no valid UTC yet: alerts are unaffected
        return;
    }
    ha_time_sync_t m = {0, session, ++time_seq, utc, acc};
    if (time_seq == 0) m.seq = time_seq = 1;
    uint8_t buf[HA_V2_TIME_SYNC_LEN];
    ha_time_sync_encode(&m, buf);
    if (sendto(fd, buf, sizeof(buf), 0, (struct sockaddr *)&to, sizeof(to)) == sizeof(buf))
        ESP_LOGI(TAG, "TIME_SYNC seq=%" PRIu32 " utc_ms=%lld acc=%" PRIu32 "ms", m.seq,
                 (long long)utc, acc);
    *next_at = now + (int64_t)CONFIG_HA_TIME_SYNC_PERIOD_S * 1000;
#else
    (void)fd; (void)now; (void)next_at; (void)force;
#endif
}

static void sender_task(void *arg) {
    int fd = (int)(intptr_t)arg;
    int64_t time_next_at = 0;
    bool was_registered = false;
    for (;;) {
        net_evt_t ne;
        while (xQueueReceive(s_net_q, &ne, 0) == pdTRUE) {
            xSemaphoreTake(s_lock, portMAX_DELAY);
            if (ne == NET_STA_LEFT) ha_tx_on_station_left(&s_tx, now_ms());
            else ha_tx_on_station_joined(&s_tx, now_ms());
            uint32_t queued = s_tx.count;
            xSemaphoreGive(s_lock);
            if (ne == NET_STA_LEFT)
                ESP_LOGW(TAG, "station left; pausing EVENT tx, keeping %" PRIu32 " queued", queued);
            else
                ESP_LOGI(TAG, "station joined; waiting for HELLO");
        }

        uint8_t buf[HA_RX_BUFFER_LEN]; // one byte larger than any valid frame
        struct sockaddr_in from = {0};
        socklen_t from_len = sizeof(from);
        int n = recvfrom(fd, buf, sizeof(buf), 0, (struct sockaddr *)&from, &from_len);
        if (n > 0) handle_datagram(buf, n, &from);
        else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
            ESP_LOGW(TAG, "recvfrom errno=%d", errno);

        int64_t now = now_ms();
        uint8_t out[HA_V1_LEN];
        xSemaphoreTake(s_lock, portMAX_DELAY);
        ha_tx_action_t a = ha_tx_poll(&s_tx, now, out);
        struct sockaddr_in to = {.sin_family = AF_INET, .sin_port = s_tx.peer.port,
                                 .sin_addr.s_addr = s_tx.peer.ip};
        bool registered = s_tx.peer.registered && s_tx.peer.associated;
        int64_t due = ha_tx_ms_until_due(&s_tx, now);
        xSemaphoreGive(s_lock);

        if (a.peer_went_stale)
            ESP_LOGW(TAG, "receiver heartbeat expired; keeping peer and queue, retrying");
        if (a.send) {
            if (a.new_burst)
                ESP_LOGW(TAG, "RETRY_PENDING seq=%" PRIu32 "; new retry burst", a.seq);
            if (sendto(fd, out, sizeof(out), 0, (struct sockaddr *)&to, sizeof(to)) !=
                sizeof(out))
                ESP_LOGW(TAG, "sendto failed errno=%d", errno);
            ESP_LOGI(TAG, "TX seq=%" PRIu32 " id=0x%08" PRIx32 "(%s) attempt=%" PRIu32, a.seq,
                     a.event_id, ha_event_name(a.event_id), a.burst_attempt);
        }
        deliver_reports();
        maybe_send_time(fd, now, &time_next_at, registered && !was_registered);
        was_registered = registered;

        int64_t wait = due < 1 ? 1 : (due > MAX_POLL_MS ? MAX_POLL_MS : due);
        struct timeval tv = {.tv_sec = 0, .tv_usec = (suseconds_t)(wait * 1000)};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    }
}

#if CONFIG_HA_DIAG_INJECT_EVENTS
/* DIAGNOSTIC ONLY (default off): synthetic events to exercise the radio path.
 * This is NOT sound recognition and must never be reported as such. */
static void diag_inject_task(void *arg) {
    (void)arg;
    uint32_t n = 0;
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(CONFIG_HA_DIAG_INJECT_PERIOD_MS));
        ++n;
        if (n % 5 == 0) {
            const uint32_t both[2] = {HA_EVENT_DOORBELL, HA_EVENT_FIRE_ALARM};
            ESP_LOGW(TAG, "DIAG_INJECT batch FIRE+DOORBELL (synthetic, not a detection)");
            ha_publish_events(both, 2, NULL, NULL);
        } else {
            uint32_t id = (n & 1) ? HA_EVENT_DOORBELL : HA_EVENT_FIRE_ALARM;
            ESP_LOGW(TAG, "DIAG_INJECT %s (synthetic, not a detection)", ha_event_name(id));
            ha_publish_event(id, NULL);
        }
    }
}
#endif

esp_err_t ha_sender_start(void) {
    if (s_started) return ESP_OK;
    s_lock = xSemaphoreCreateMutex();
    s_net_q = xQueueCreate(8, sizeof(net_evt_t));
    s_report_q = xQueueCreate(HA_TX_QUEUE_MAX * 2, sizeof(ha_tx_report_t));
    if (!s_lock || !s_net_q || !s_report_q) return ESP_ERR_NO_MEM;

    ha_tx_config_t cfg;
    ha_tx_default_config(&cfg);
    cfg.peer_port = htons(HA_UDP_PORT);
#if CONFIG_HA_TX_FIRE_FIRST
    cfg.fire_first = true;
#endif
    cfg.rand32 = rand32;
    cfg.report = core_report;
    ha_tx_init(&s_tx, &cfg, esp_random());

    esp_err_t err = start_softap();
    if (err != ESP_OK) return err;

    int fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (fd < 0) return ESP_FAIL;
    struct sockaddr_in local = {.sin_family = AF_INET, .sin_port = htons(HA_UDP_PORT),
                                .sin_addr.s_addr = htonl(INADDR_ANY)};
    struct timeval tv = {.tv_sec = 0, .tv_usec = MAX_POLL_MS * 1000};
    if (bind(fd, (struct sockaddr *)&local, sizeof(local)) != 0 ||
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) {
        ESP_LOGE(TAG, "socket setup errno=%d", errno);
        close(fd);
        return ESP_FAIL;
    }
    s_started = true; // publish may queue before the C3 registers
    if (xTaskCreate(sender_task, "ha_sender", 4096, (void *)(intptr_t)fd, 6, NULL) != pdPASS) {
        s_started = false;
        close(fd);
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "sender started session=%" PRIu32 " queue=%" PRIu32 " ack_wait=%" PRIu32
             "ms burst=%" PRIu32 " backoff=%" PRIu32 "..%" PRIu32 "ms ttl fire=%" PRIu32
             "s door=%" PRIu32 "s",
             s_tx.session, cfg.capacity, cfg.ack_wait_ms, cfg.attempts_per_burst,
             cfg.backoff_min_ms, cfg.backoff_max_ms, HA_TTL_FIRE_ALARM_MS / 1000,
             HA_TTL_DOORBELL_MS / 1000);
#if CONFIG_HA_DIAG_INJECT_EVENTS
    ESP_LOGW(TAG, "DIAGNOSTIC event injection ENABLED (not a product build)");
    xTaskCreate(diag_inject_task, "ha_diag_inj", 3072, NULL, 3, NULL);
#endif
#if CONFIG_HA_DIAG_ALLOW_TEST_EVENT
    ESP_LOGW(TAG, "DIAGNOSTIC TEST(1) event id ENABLED (not a product build)");
#endif
    return ESP_OK;
}
