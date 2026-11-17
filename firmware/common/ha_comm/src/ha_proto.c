#include "ha_proto.h"

/* Explicit network byte order; never memcpy a native struct onto the wire. */
void ha_put_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v >> 8);
    p[1] = (uint8_t)v;
}

void ha_put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)v;
}

void ha_put_u64(uint8_t *p, uint64_t v) {
    ha_put_u32(p, (uint32_t)(v >> 32));
    ha_put_u32(p + 4, (uint32_t)v);
}

uint16_t ha_get_u16(const uint8_t *p) {
    return (uint16_t)(((uint16_t)p[0] << 8) | p[1]);
}

uint32_t ha_get_u32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

uint64_t ha_get_u64(const uint8_t *p) {
    return ((uint64_t)ha_get_u32(p) << 32) | ha_get_u32(p + 4);
}

static bool magic_ok(const uint8_t *buf, size_t len) {
    return len >= 2 && buf[0] == HA_MAGIC0 && buf[1] == HA_MAGIC1;
}

int ha_frame_version(const uint8_t *buf, size_t len) {
    if (!buf || len < 3 || !magic_ok(buf, len)) return -1;
    return buf[2];
}

size_t ha_v1_encode(const ha_v1_frame_t *f, uint8_t out[HA_V1_LEN]) {
    out[0] = HA_MAGIC0;
    out[1] = HA_MAGIC1;
    out[2] = HA_V1_VERSION;
    out[3] = f->type;
    ha_put_u32(out + 4, f->session);
    ha_put_u32(out + 8, f->seq);
    ha_put_u32(out + 12, f->value);
    return HA_V1_LEN;
}

static bool v1_type_known(uint8_t t) {
    return t >= HA_MSG_HELLO && t <= HA_MSG_POWER_STATE;
}

ha_decode_result_t ha_v1_decode(const uint8_t *buf, size_t len, ha_v1_frame_t *out) {
    // Length first: a longer datagram is never truncated into a valid frame.
    if (!buf || len != HA_V1_LEN) return HA_DEC_ERR_LENGTH;
    if (!magic_ok(buf, len)) return HA_DEC_ERR_MAGIC;
    if (buf[2] != HA_V1_VERSION) return HA_DEC_ERR_VERSION;
    if (!v1_type_known(buf[3])) return HA_DEC_ERR_TYPE;
    out->type = buf[3];
    out->session = ha_get_u32(buf + 4);
    out->seq = ha_get_u32(buf + 8);
    out->value = ha_get_u32(buf + 12);
    return HA_DEC_OK;
}

size_t ha_time_sync_encode(const ha_time_sync_t *m, uint8_t out[HA_V2_TIME_SYNC_LEN]) {
    out[0] = HA_MAGIC0;
    out[1] = HA_MAGIC1;
    out[2] = HA_V2_VERSION;
    out[3] = HA_MSG_V2_TIME_SYNC;
    ha_put_u16(out + 4, HA_V2_TIME_SYNC_LEN);
    ha_put_u16(out + 6, m->flags);
    ha_put_u32(out + 8, m->session);
    ha_put_u32(out + 12, m->seq);
    ha_put_u64(out + 16, (uint64_t)m->utc_ms);
    ha_put_u32(out + 24, m->accuracy_ms);
    return HA_V2_TIME_SYNC_LEN;
}

ha_decode_result_t ha_time_sync_decode(const uint8_t *buf, size_t len, ha_time_sync_t *out) {
    if (!buf || len != HA_V2_TIME_SYNC_LEN) return HA_DEC_ERR_LENGTH;
    if (!magic_ok(buf, len)) return HA_DEC_ERR_MAGIC;
    if (buf[2] != HA_V2_VERSION) return HA_DEC_ERR_VERSION;
    if (buf[3] != HA_MSG_V2_TIME_SYNC) return HA_DEC_ERR_TYPE;
    // The embedded length must agree with the datagram length.
    if (ha_get_u16(buf + 4) != HA_V2_TIME_SYNC_LEN) return HA_DEC_ERR_FIELD;
    uint16_t flags = ha_get_u16(buf + 6);
    uint32_t seq = ha_get_u32(buf + 12);
    int64_t utc = (int64_t)ha_get_u64(buf + 16);
    if (flags != 0 || seq == 0) return HA_DEC_ERR_FIELD;
    if (utc < HA_UTC_MIN_VALID_MS || utc > HA_UTC_MAX_VALID_MS) return HA_DEC_ERR_FIELD;
    out->flags = flags;
    out->session = ha_get_u32(buf + 8);
    out->seq = seq;
    out->utc_ms = utc;
    out->accuracy_ms = ha_get_u32(buf + 24);
    return HA_DEC_OK;
}

bool ha_v1_hello_fields_valid(const ha_v1_frame_t *f) {
    return f->type == HA_MSG_HELLO && f->seq == 0 && f->value == 0;
}

bool ha_v1_event_fields_valid(const ha_v1_frame_t *f) {
    return f->type == HA_MSG_EVENT && f->seq != 0;
}

const char *ha_decode_result_name(ha_decode_result_t r) {
    switch (r) {
    case HA_DEC_OK: return "ok";
    case HA_DEC_ERR_LENGTH: return "length";
    case HA_DEC_ERR_MAGIC: return "magic";
    case HA_DEC_ERR_VERSION: return "version";
    case HA_DEC_ERR_TYPE: return "type";
    case HA_DEC_ERR_FIELD: return "field";
    }
    return "?";
}
