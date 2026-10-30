#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_pm.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"

// Test credentials only. Change both boards together before deployment.
#define SSID "HearingAlert-Test"
#define PASSWORD "hearing-test-2026"
#define PORT 4210
#if CONFIG_IDF_TARGET_ESP32S3
#define AP_ROLE 1
#elif CONFIG_IDF_TARGET_ESP32C3
#define AP_ROLE 0
#if !CONFIG_PM_ENABLE || !CONFIG_PM_PROFILING || !CONFIG_FREERTOS_USE_TICKLESS_IDLE
#error "C3 requires PM, profiling and tickless idle; use sdkconfig.defaults.esp32c3"
#endif
#else
#error "Only ESP32-S3 and ESP32-C3 are supported"
#endif
#if CONFIG_IDF_TARGET_ESP32C3 && CONFIG_USJ_NO_AUTO_LS_ON_CONNECTION
#error "C3 sleep validation requires CONFIG_USJ_NO_AUTO_LS_ON_CONNECTION=n"
#endif
#define HELLO 1
#define EVENT 2
#define ACK 3
#define PM_STATS 4
#define POWER_STATE 5
#define PACKET_SIZE 16
#define EVENT_INTERVAL_MS 3000
#define ACK_TIMEOUT_MS 300
#define MAX_SEND_ATTEMPTS 4
#define RETRY_BATCH_INTERVAL_MS 2000
#define HELLO_INTERVAL_MS 10000
#define PEER_TIMEOUT_MS 35000
#define POWER_REPORT_MS 10000
#define LIGHT_SLEEP_START_MS 15000
static const char *TAG = "softap_test";
static EventGroupHandle_t wifi_bits;
#define GOT_IP_BIT BIT0
#define HELLO_DUE_BIT BIT1
static int64_t now_ms(void) { return esp_timer_get_time() / 1000; }

#if CONFIG_IDF_TARGET_ESP32C3
static void configure_receiver_pm(bool light_sleep) {
    const esp_pm_config_t config = {
        .max_freq_mhz = 160, .min_freq_mhz = 40,
        .light_sleep_enable = light_sleep,
    };
    ESP_ERROR_CHECK(esp_pm_configure(&config));
}

// Read the SDK's successful/rejected sleep counters, not a count of idle loops.
// Using the public profiling API avoids a dependency on private PM internals.
static bool get_sleep_counts(uint32_t *slept, uint32_t *rejected) {
    char *buffer = NULL;
    size_t length = 0;
    FILE *stream = open_memstream(&buffer, &length);
    if (!stream) return false;
    esp_err_t err = esp_pm_dump_locks(stream);
    fclose(stream);
    bool valid = false;
    if (err == ESP_OK && buffer) {
        const char *stats = strstr(buffer, "light_sleep_counts:");
        valid = stats && sscanf(stats,
            "light_sleep_counts:%" SCNu32 " light_sleep_reject_counts:%" SCNu32,
            slept, rejected) == 2;
    }
    free(buffer);
    return valid;
}
#endif

// Explicit network byte order; never send a native C struct.
static void put32(uint8_t *p, uint32_t n) {
    p[0] = n >> 24; p[1] = n >> 16; p[2] = n >> 8; p[3] = n;
}
static uint32_t get32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | p[3];
}
static void send_packet(int fd, const struct sockaddr_in *peer, uint8_t type,
                        uint32_t session, uint32_t seq, uint32_t value) {
    uint8_t p[PACKET_SIZE] = {'H', 'A', 1, type};
    put32(p + 4, session); put32(p + 8, seq); put32(p + 12, value);
    if (sendto(fd, p, sizeof(p), 0, (const struct sockaddr *)peer, sizeof(*peer)) != sizeof(p))
        ESP_LOGW(TAG, "sendto failed errno=%d", errno);
}

static void wifi_event(void *arg, esp_event_base_t base, int32_t id, void *data) {
    (void)arg;
    if (!AP_ROLE && base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(wifi_bits, GOT_IP_BIT);
        const wifi_event_sta_disconnected_t *event = data;
        ESP_LOGW(TAG, "STA disconnected reason=%u rssi=%d", event->reason, event->rssi);
    }
    if (!AP_ROLE && base == WIFI_EVENT &&
        (id == WIFI_EVENT_STA_START || id == WIFI_EVENT_STA_DISCONNECTED)) {
        // Defer reconnect to event delivery; no blocking retry loop.
        esp_err_t err = esp_wifi_connect();
        if (err != ESP_OK) ESP_LOGW(TAG, "connect: %s", esp_err_to_name(err));
    }
    if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(wifi_bits, GOT_IP_BIT | HELLO_DUE_BIT);
        ESP_LOGI(TAG, "STA got IP; UDP discovery enabled");
    }
}

#if CONFIG_IDF_TARGET_ESP32C3
static int discovery_fd;
static uint32_t discovery_session;
// Wake on every IP acquisition, even if disconnect/reconnect happened while
// the receive task was blocked. No receive-timeout or heartbeat delay.
static void discovery_task(void *arg) {
    (void)arg;
    struct sockaddr_in ap = {.sin_family = AF_INET, .sin_port = htons(PORT)};
    inet_pton(AF_INET, "192.168.4.1", &ap.sin_addr);
    for (;;) {
        xEventGroupWaitBits(wifi_bits, HELLO_DUE_BIT, pdTRUE, pdFALSE,
                            pdMS_TO_TICKS(HELLO_INTERVAL_MS));
        if (xEventGroupGetBits(wifi_bits) & GOT_IP_BIT)
            send_packet(discovery_fd, &ap, HELLO, discovery_session, 0, 0);
    }
}
#endif

static void init_wifi(void) {
    wifi_bits = xEventGroupCreate();
    assert(wifi_bits);
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    if (AP_ROLE) esp_netif_create_default_wifi_ap();
    else esp_netif_create_default_wifi_sta();
    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event, NULL));
    wifi_config_t config = {0};
    if (AP_ROLE) {
        strcpy((char *)config.ap.ssid, SSID);
        strcpy((char *)config.ap.password, PASSWORD);
        config.ap.ssid_len = strlen(SSID);
        config.ap.channel = 6;
        config.ap.max_connection = 1;
        config.ap.authmode = WIFI_AUTH_WPA2_PSK;
        // DTIM=1 and 100 TU beacons: low-latency unicast delivery to sleeping C3.
        config.ap.beacon_interval = 100;
        config.ap.dtim_period = 1;
    } else {
        strcpy((char *)config.sta.ssid, SSID);
        strcpy((char *)config.sta.password, PASSWORD);
        config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    }
    ESP_ERROR_CHECK(esp_wifi_set_mode(AP_ROLE ? WIFI_MODE_AP : WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(AP_ROLE ? WIFI_IF_AP : WIFI_IF_STA, &config));
    ESP_ERROR_CHECK(esp_wifi_start());
    // This SuperMini connected reliably at 11 dBm in the bench comparison.
    if (!AP_ROLE) {
        ESP_ERROR_CHECK(esp_wifi_set_max_tx_power(44));
        ESP_LOGI(TAG, "C3 TX power limit: 11 dBm");
    }
    ESP_ERROR_CHECK(esp_wifi_set_ps(AP_ROLE ? WIFI_PS_NONE : WIFI_PS_MIN_MODEM));
    ESP_LOGI(TAG, "%s ready; SSID=%s UDP=%d", AP_ROLE ? "S3 AP" : "C3 STA", SSID, PORT);
}

void app_main(void) {
#if CONFIG_IDF_TARGET_ESP32C3
    // Leave a short USB flashing/boot-log window, then enable auto Light-sleep.
    configure_receiver_pm(false);
#endif
    init_wifi();
    int fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (fd < 0) { ESP_LOGE(TAG, "socket errno=%d", errno); return; }
    struct sockaddr_in local = {.sin_family = AF_INET, .sin_port = htons(PORT),
                                .sin_addr.s_addr = htonl(INADDR_ANY)};
    // C3 blocks instead of polling every 50ms; socket receive wakes its task.
    struct timeval timeout = {.tv_sec = AP_ROLE ? 0 : 1,
                              .tv_usec = AP_ROLE ? 50000 : 0};
    if (bind(fd, (struct sockaddr *)&local, sizeof(local)) != 0 ||
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0) {
        ESP_LOGE(TAG, "socket setup errno=%d", errno); close(fd); return;
    }
    struct sockaddr_in peer = {.sin_family = AF_INET, .sin_port = htons(PORT)};
    inet_pton(AF_INET, "192.168.4.1", &peer.sin_addr);
    uint32_t session = esp_random(), seq = 0, peer_session = 0;
    uint32_t last_session = 0, last_seq = 0;
    bool have_last = false, have_peer = false, pending = false, peer_stale = false;
#if CONFIG_IDF_TARGET_ESP32C3
    bool light_sleep_started = false;
    bool was_online = false;
    int64_t power_at = 0;
    discovery_fd = fd;
    discovery_session = session;
    if (xTaskCreate(discovery_task, "discovery", 3072, NULL, 5, NULL) != pdPASS) {
        ESP_LOGE(TAG, "discovery task allocation failed");
        close(fd);
        return;
    }
#endif
    int attempts = 0;
    int64_t event_at = 0, sent_at = 0, first_sent = 0, seen_at = 0;
    for (;;) {
        uint8_t p[PACKET_SIZE + 1]; // Extra byte rejects oversized UDP datagrams.
        struct sockaddr_in from = {0}; socklen_t from_len = sizeof(from);
        int n = recvfrom(fd, p, sizeof(p), 0, (struct sockaddr *)&from, &from_len);
        int64_t now = now_ms();
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
            ESP_LOGW(TAG, "recvfrom errno=%d", errno);
        if (n == PACKET_SIZE && p[0] == 'H' && p[1] == 'A' && p[2] == 1 &&
            from.sin_port == htons(PORT)) {
            uint32_t s = get32(p + 4), q = get32(p + 8), value = get32(p + 12);
            if (AP_ROLE && p[3] == HELLO && q == 0 && value == 0) {
                if (!have_peer || peer.sin_addr.s_addr != from.sin_addr.s_addr || peer_session != s) {
                    peer = from; peer_session = s; have_peer = true;
                    event_at = now;
                    if (pending) sent_at = now - RETRY_BATCH_INTERVAL_MS;
                    ESP_LOGI(TAG, "receiver registered session=%" PRIu32, s);
                }
                if (peer_stale && pending) sent_at = now - RETRY_BATCH_INTERVAL_MS;
                peer_stale = false;
                seen_at = now;
            } else if (AP_ROLE && have_peer && s == peer_session &&
                       from.sin_addr.s_addr == peer.sin_addr.s_addr && p[3] == PM_STATS) {
                ESP_LOGI(TAG, "C3_PM session=%" PRIu32 " sleep_count=%" PRIu32
                         " reject_count=%" PRIu32, s, q, value);
            } else if (AP_ROLE && have_peer && s == peer_session &&
                       from.sin_addr.s_addr == peer.sin_addr.s_addr && p[3] == POWER_STATE &&
                       q <= 1 && value == WIFI_PS_MIN_MODEM) {
                ESP_LOGI(TAG, "C3_STATE modem_sleep=MIN auto_light_sleep=%" PRIu32, q);
            } else if (AP_ROLE && have_peer && pending && p[3] == ACK &&
                       from.sin_addr.s_addr == peer.sin_addr.s_addr &&
                       s == session && q == seq && value == 1) {
                pending = false;
                ESP_LOGI(TAG, "ACK seq=%" PRIu32 " elapsed=%lldms attempts=%d",
                         seq, (long long)(now - first_sent), attempts);
            } else if (!AP_ROLE && p[3] == EVENT &&
                       from.sin_addr.s_addr == peer.sin_addr.s_addr && value == 1 && q != 0) {
                // Stop-and-wait sender: older packets within a boot are never delivered twice.
                bool fresh = !have_last || s != last_session || q > last_seq;
                if (fresh) {
                    have_last = true; last_session = s; last_seq = q;
                    ESP_LOGI(TAG, "TEST event session=%" PRIu32 " seq=%" PRIu32, s, q);
                } else ESP_LOGI(TAG, "duplicate/stale seq=%" PRIu32, q);
                send_packet(fd, &peer, ACK, s, q, value);
            }
        }
#if CONFIG_IDF_TARGET_ESP32C3
        if (!light_sleep_started && now >= LIGHT_SLEEP_START_MS) {
            configure_receiver_pm(true);
            light_sleep_started = true;
            power_at = 0;
            ESP_LOGI(TAG, "C3 auto Light-sleep enabled; USB console may pause; see S3 C3_PM");
        }
        bool online = (xEventGroupGetBits(wifi_bits) & GOT_IP_BIT) != 0;
        if (online && !was_online) power_at = 0;
        was_online = online;
        if (online && now >= power_at) {
            esp_pm_config_t pm;
            wifi_ps_type_t ps;
            ESP_ERROR_CHECK(esp_pm_get_configuration(&pm));
            ESP_ERROR_CHECK(esp_wifi_get_ps(&ps));
            send_packet(fd, &peer, POWER_STATE, session, pm.light_sleep_enable, ps);
            uint32_t slept = 0, rejected = 0;
            if (pm.light_sleep_enable && get_sleep_counts(&slept, &rejected)) {
                send_packet(fd, &peer, PM_STATS, session, slept, rejected);
            }
            power_at = now + POWER_REPORT_MS;
        }
#endif
        if (AP_ROLE) {
            if (have_peer && !peer_stale && now - seen_at > PEER_TIMEOUT_MS) {
                peer_stale = true;
                ESP_LOGW(TAG, "receiver heartbeat expired; retaining peer and pending event");
            }
            if (have_peer && !pending && now >= event_at) {
                if (++seq == 0) { session = esp_random(); seq = 1; }
                pending = true; attempts = 0; first_sent = now; sent_at = now - ACK_TIMEOUT_MS;
                event_at = now + EVENT_INTERVAL_MS;
            }
            if (pending && now - sent_at >=
                (attempts >= MAX_SEND_ATTEMPTS ? RETRY_BATCH_INTERVAL_MS : ACK_TIMEOUT_MS)) {
                if (attempts == MAX_SEND_ATTEMPTS) {
                    ESP_LOGW(TAG, "RETRY_PENDING seq=%" PRIu32 "; retrying retained event", seq);
                    attempts = 0;
                    send_packet(fd, &peer, EVENT, session, seq, 1);
                    sent_at = now; ++attempts;
                    ESP_LOGI(TAG, "TX seq=%" PRIu32 " attempt=%d", seq, attempts);
                } else {
                    send_packet(fd, &peer, EVENT, session, seq, 1);
                    sent_at = now; ++attempts;
                    ESP_LOGI(TAG, "TX seq=%" PRIu32 " attempt=%d", seq, attempts);
                }
            }
        }
    }
}
