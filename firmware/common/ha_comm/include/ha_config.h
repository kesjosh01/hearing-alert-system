/*
 * Shared communication settings.
 *
 * In ESP-IDF builds the values come from Kconfig (menuconfig: "Hearing Alert
 * communication"). Host unit tests compile without sdkconfig.h and get the same
 * defaults from the #ifndef fallbacks below, so both see one source of truth.
 *
 * Values marked "design default" were chosen for the 2026-11-15 implementation
 * and have NOT been measured on the boards with this firmware.
 */
#pragma once

#if defined(ESP_PLATFORM)
#include "sdkconfig.h"
#endif

/* ---- Network (re-used, previously verified with the TEST firmware) ---- */
#ifdef CONFIG_HA_WIFI_SSID
#define HA_WIFI_SSID CONFIG_HA_WIFI_SSID
#else
#define HA_WIFI_SSID "HearingAlert-Test"
#endif

#ifdef CONFIG_HA_WIFI_PASSWORD
#define HA_WIFI_PASSWORD CONFIG_HA_WIFI_PASSWORD
#else
#define HA_WIFI_PASSWORD "hearing-test-2026"
#endif

#ifdef CONFIG_HA_WIFI_CHANNEL
#define HA_WIFI_CHANNEL CONFIG_HA_WIFI_CHANNEL
#else
#define HA_WIFI_CHANNEL 6
#endif

#ifdef CONFIG_HA_UDP_PORT
#define HA_UDP_PORT CONFIG_HA_UDP_PORT
#else
#define HA_UDP_PORT 4210
#endif

/* S3 SoftAP address (ESP-IDF default AP netif). */
#define HA_AP_IP_STR "192.168.4.1"

/* ---- 300 ms receive opportunity ----
 * beacon_interval is in TU (1 TU = 1024 us) and must be a multiple of 100.
 * 100 TU * DTIM 3 = 307.2 ms nominal DTIM interval; C3 in WIFI_PS_MIN_MODEM
 * wakes for every DTIM beacon. Exactly 300 ms is not configurable. */
#ifdef CONFIG_HA_AP_BEACON_INTERVAL_TU
#define HA_AP_BEACON_INTERVAL_TU CONFIG_HA_AP_BEACON_INTERVAL_TU
#else
#define HA_AP_BEACON_INTERVAL_TU 100
#endif

#ifdef CONFIG_HA_AP_DTIM_PERIOD
#define HA_AP_DTIM_PERIOD CONFIG_HA_AP_DTIM_PERIOD
#else
#define HA_AP_DTIM_PERIOD 3
#endif

/* Nominal C3 receive interval in microseconds (for logs/docs only). */
#define HA_NOMINAL_WAKE_US ((uint32_t)HA_AP_BEACON_INTERVAL_TU * 1024u * HA_AP_DTIM_PERIOD)

/* ---- Liveness ---- */
#ifdef CONFIG_HA_HELLO_INTERVAL_MS
#define HA_HELLO_INTERVAL_MS CONFIG_HA_HELLO_INTERVAL_MS
#else
#define HA_HELLO_INTERVAL_MS 10000
#endif

#ifdef CONFIG_HA_PEER_STALE_MS
#define HA_PEER_STALE_MS CONFIG_HA_PEER_STALE_MS
#else
#define HA_PEER_STALE_MS 35000
#endif

/* ---- S3 sender (design defaults) ---- */
#ifdef CONFIG_HA_TX_QUEUE_CAPACITY
#define HA_TX_QUEUE_CAPACITY CONFIG_HA_TX_QUEUE_CAPACITY
#else
#define HA_TX_QUEUE_CAPACITY 8
#endif

/* Per-attempt ACK wait. Must cover: AP buffering until the next C3 DTIM wake
 * (<= 307.2 ms, or ~614 ms if one beacon is missed) + C3 processing + ACK. */
#ifdef CONFIG_HA_TX_ACK_WAIT_MS
#define HA_TX_ACK_WAIT_MS CONFIG_HA_TX_ACK_WAIT_MS
#else
#define HA_TX_ACK_WAIT_MS 700
#endif

#ifdef CONFIG_HA_TX_ATTEMPTS_PER_BURST
#define HA_TX_ATTEMPTS_PER_BURST CONFIG_HA_TX_ATTEMPTS_PER_BURST
#else
#define HA_TX_ATTEMPTS_PER_BURST 4
#endif

#ifdef CONFIG_HA_TX_BACKOFF_MIN_MS
#define HA_TX_BACKOFF_MIN_MS CONFIG_HA_TX_BACKOFF_MIN_MS
#else
#define HA_TX_BACKOFF_MIN_MS 2000
#endif

#ifdef CONFIG_HA_TX_BACKOFF_MAX_MS
#define HA_TX_BACKOFF_MAX_MS CONFIG_HA_TX_BACKOFF_MAX_MS
#else
#define HA_TX_BACKOFF_MAX_MS 16000
#endif

/* Per-event expiry, user-selected development defaults (2026-11-15). */
#ifdef CONFIG_HA_TTL_FIRE_ALARM_S
#define HA_TTL_FIRE_ALARM_MS ((uint32_t)CONFIG_HA_TTL_FIRE_ALARM_S * 1000u)
#else
#define HA_TTL_FIRE_ALARM_MS (600u * 1000u)
#endif

#ifdef CONFIG_HA_TTL_DOORBELL_S
#define HA_TTL_DOORBELL_MS ((uint32_t)CONFIG_HA_TTL_DOORBELL_S * 1000u)
#else
#define HA_TTL_DOORBELL_MS (120u * 1000u)
#endif

/* TEST(1) is only valid when this diagnostic option is enabled on BOTH boards. */
#ifdef CONFIG_HA_DIAG_ALLOW_TEST_EVENT
#define HA_ALLOW_TEST_EVENT 1
#else
#define HA_ALLOW_TEST_EVENT 0
#endif

/* ---- C3 receiver ---- */
#ifdef CONFIG_HA_RX_ALERT_QUEUE_LEN
#define HA_RX_ALERT_QUEUE_LEN CONFIG_HA_RX_ALERT_QUEUE_LEN
#else
#define HA_RX_ALERT_QUEUE_LEN 8
#endif

/* Duplicate-suppression memory (RAM only; lost on C3 reboot). */
#define HA_RX_SESSION_SLOTS 4
#define HA_RX_HISTORY_LEN 8

/* ---- Time extension (optional, default off on S3) ---- */
#ifdef CONFIG_HA_TIME_DELIVERY_BOUND_MS
#define HA_TIME_DELIVERY_BOUND_MS CONFIG_HA_TIME_DELIVERY_BOUND_MS
#else
#define HA_TIME_DELIVERY_BOUND_MS 1000
#endif

#ifdef CONFIG_HA_TIME_DRIFT_PPM
#define HA_TIME_DRIFT_PPM CONFIG_HA_TIME_DRIFT_PPM
#else
#define HA_TIME_DRIFT_PPM 1000
#endif

#ifdef CONFIG_HA_TIME_STALE_AFTER_S
#define HA_TIME_STALE_AFTER_MS ((int64_t)CONFIG_HA_TIME_STALE_AFTER_S * 1000)
#else
#define HA_TIME_STALE_AFTER_MS ((int64_t)24 * 3600 * 1000)
#endif
