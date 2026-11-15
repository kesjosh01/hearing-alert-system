/*
 * Public sound-event registry (canonical IDs shared by S3, C3 and ML glue).
 *
 * To add a sound: append ONE row in ha_events.c with a new, never-used ID
 * (next free: 0x00000102), add its constant here, add its vectors to the host
 * tests, and roll out C3 firmware before S3 starts sending it. Older C3
 * firmware rejects unknown IDs (no alert, no ACK, no duplicate-state change).
 * Never reuse or re-purpose an assigned ID.
 *
 * Model class indices are NOT event IDs. The ML/control layer maps its final
 * decision (e.g. "fire_alarm") to one of these constants before publishing.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HA_EVENT_RESERVED 0x00000000u  /* never an event */
#define HA_EVENT_TEST 0x00000001u      /* communication experiments only */
#define HA_EVENT_FIRE_ALARM 0x00000100u
#define HA_EVENT_DOORBELL 0x00000101u

typedef struct {
    uint32_t id;
    const char *name;
    bool product;        /* sent/accepted in the default product build */
    bool evict_protected; /* queue saturation: not evicted by a non-protected event */
    uint32_t ttl_ms;     /* S3 queue lifetime from publish; 0 = no expiry */
} ha_event_def_t;

/* Registry lookup (includes TEST). NULL if the ID is not registered. */
const ha_event_def_t *ha_event_find(uint32_t id);

/* True if the ID may be published/accepted in this build. TEST is allowed only
 * when allow_test is true (diagnostic option); RESERVED never. */
bool ha_event_allowed(uint32_t id, bool allow_test);

/* Canonical order inside one detection batch (FIRE_ALARM before DOORBELL).
 * Lower rank first. Unregistered IDs get UINT32_MAX. */
uint32_t ha_event_batch_rank(uint32_t id);

const char *ha_event_name(uint32_t id);

size_t ha_event_count(void);
const ha_event_def_t *ha_event_at(size_t index);

#ifdef __cplusplus
}
#endif
