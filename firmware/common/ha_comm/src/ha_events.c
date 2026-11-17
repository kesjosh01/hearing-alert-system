#include "ha_events.h"

#include "ha_config.h"

/* Row order == canonical batch order. TEST is last and non-product. */
static const ha_event_def_t s_events[] = {
    {HA_EVENT_FIRE_ALARM, "FIRE_ALARM", true, true, HA_TTL_FIRE_ALARM_MS},
    {HA_EVENT_DOORBELL, "DOORBELL", true, false, HA_TTL_DOORBELL_MS},
    {HA_EVENT_TEST, "TEST", false, false, 60u * 1000u},
};

#define EVENT_COUNT (sizeof(s_events) / sizeof(s_events[0]))

const ha_event_def_t *ha_event_find(uint32_t id) {
    for (size_t i = 0; i < EVENT_COUNT; ++i)
        if (s_events[i].id == id) return &s_events[i];
    return NULL;
}

bool ha_event_allowed(uint32_t id, bool allow_test) {
    const ha_event_def_t *def = ha_event_find(id);
    if (!def) return false;
    return def->product || (allow_test && id == HA_EVENT_TEST);
}

uint32_t ha_event_batch_rank(uint32_t id) {
    for (size_t i = 0; i < EVENT_COUNT; ++i)
        if (s_events[i].id == id) return (uint32_t)i;
    return UINT32_MAX;
}

const char *ha_event_name(uint32_t id) {
    if (id == HA_EVENT_RESERVED) return "RESERVED";
    const ha_event_def_t *def = ha_event_find(id);
    return def ? def->name : "UNREGISTERED";
}

size_t ha_event_count(void) { return EVENT_COUNT; }

const ha_event_def_t *ha_event_at(size_t index) {
    return index < EVENT_COUNT ? &s_events[index] : NULL;
}
