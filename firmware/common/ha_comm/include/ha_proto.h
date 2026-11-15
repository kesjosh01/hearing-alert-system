/*
 * Hearing Alert S3–C3 wire protocol.
 *
 * v1 (16 bytes, docs/communication/README.md): HELLO / EVENT / ACK / PM_STATS /
 *     POWER_STATE. Layout and field meaning are frozen; never send a C struct.
 * v2 TIME_SYNC (28 bytes): an independent control message with its own version
 *     and explicit length field. v1 receivers reject it by version/length, so the
 *     v1 alert path stays compatible. event_id is never reused as a timestamp.
 *
 * This header is portable C (no ESP-IDF dependency) so that host unit tests run
 * against exactly the same encoder/decoder that the firmware links.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HA_MAGIC0 'H'
#define HA_MAGIC1 'A'

#define HA_V1_VERSION 1u
#define HA_V1_LEN 16u

#define HA_V2_VERSION 2u
#define HA_V2_TIME_SYNC_LEN 28u

/* Largest frame any receiver accepts. Receive buffers must be at least
 * HA_RX_BUFFER_LEN so that an oversized datagram is detected instead of being
 * silently truncated to a valid length. */
#define HA_MAX_FRAME_LEN HA_V2_TIME_SYNC_LEN
#define HA_RX_BUFFER_LEN (HA_MAX_FRAME_LEN + 1u)

/* v1 message types (byte 3). */
typedef enum {
    HA_MSG_HELLO = 1,       /* C3 -> S3: session=C3 boot session, seq=0, value=0 */
    HA_MSG_EVENT = 2,       /* S3 -> C3: session=S3 boot session, seq>=1, value=event_id */
    HA_MSG_ACK = 3,         /* C3 -> S3: copy of the accepted EVENT's three fields */
    HA_MSG_PM_STATS = 4,    /* C3 -> S3: seq=light-sleep successes, value=rejects */
    HA_MSG_POWER_STATE = 5, /* C3 -> S3: seq=auto light-sleep 0/1, value=MIN_MODEM(1) */
} ha_msg_type_t;

/* v2 message types (byte 3). Kept in a separate number range from v1. */
#define HA_MSG_V2_TIME_SYNC 0x10u

typedef struct {
    uint8_t type;
    uint32_t session;
    uint32_t seq;
    uint32_t value;
} ha_v1_frame_t;

/*
 * TIME_SYNC (S3 -> C3), all integers big-endian:
 *   0  2 magic "HA"
 *   2  1 version = 2
 *   3  1 type = 0x10
 *   4  2 length = 28 (total datagram length)
 *   6  2 flags  = 0 (reserved; non-zero is rejected by this version)
 *   8  4 session: S3 boot session (same value as v1 EVENT session)
 *  12  4 seq: TIME_SYNC counter, >= 1, independent from the EVENT seq space
 *  16  8 utc_ms: Unix epoch milliseconds (UTC) sampled by S3 right before send
 *  24  4 accuracy_ms: S3's own uncertainty of utc_ms at send time
 * Delivery delay (C3 sleep, AP buffering) is NOT included; the receiver adds
 * its own bound. Time zones are a display-layer concern.
 */
typedef struct {
    uint16_t flags;
    uint32_t session;
    uint32_t seq;
    int64_t utc_ms;
    uint32_t accuracy_ms;
} ha_time_sync_t;

/* Sanity window for utc_ms. Anything earlier is treated as "S3 clock not set";
 * this prevents an unsynchronized epoch-near value from being shown as time. */
#define HA_UTC_MIN_VALID_MS 1735689600000LL /* 2025-01-01T00:00:00Z */
#define HA_UTC_MAX_VALID_MS 4102444800000LL /* 2100-01-01T00:00:00Z */

typedef enum {
    HA_DEC_OK = 0,
    HA_DEC_ERR_LENGTH,   /* datagram length does not match the version */
    HA_DEC_ERR_MAGIC,
    HA_DEC_ERR_VERSION,  /* unknown version, or wrong decoder for this version */
    HA_DEC_ERR_TYPE,     /* unknown type for this version */
    HA_DEC_ERR_FIELD,    /* v2 length field / reserved flags / value range */
} ha_decode_result_t;

/* Encode one v1 frame into exactly 16 bytes. Returns HA_V1_LEN. */
size_t ha_v1_encode(const ha_v1_frame_t *frame, uint8_t out[HA_V1_LEN]);

/* Strictly decode a v1 datagram: len must be exactly 16, magic/version/type
 * must be known. Field semantics (seq != 0, allowed id, direction) are checked
 * by the caller-specific validators below. */
ha_decode_result_t ha_v1_decode(const uint8_t *buf, size_t len, ha_v1_frame_t *out);

/* Version byte of a datagram with a valid magic, or -1. Used for routing. */
int ha_frame_version(const uint8_t *buf, size_t len);

size_t ha_time_sync_encode(const ha_time_sync_t *msg, uint8_t out[HA_V2_TIME_SYNC_LEN]);
ha_decode_result_t ha_time_sync_decode(const uint8_t *buf, size_t len, ha_time_sync_t *out);

/* Field rules from docs/communication/README.md section 4. */
bool ha_v1_hello_fields_valid(const ha_v1_frame_t *f);  /* seq==0 && value==0 */
bool ha_v1_event_fields_valid(const ha_v1_frame_t *f);  /* seq!=0 (id checked separately) */

void ha_put_u16(uint8_t *p, uint16_t v);
void ha_put_u32(uint8_t *p, uint32_t v);
void ha_put_u64(uint8_t *p, uint64_t v);
uint16_t ha_get_u16(const uint8_t *p);
uint32_t ha_get_u32(const uint8_t *p);
uint64_t ha_get_u64(const uint8_t *p);

const char *ha_decode_result_name(ha_decode_result_t r);

#ifdef __cplusplus
}
#endif
