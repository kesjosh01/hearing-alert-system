#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t votes[2];   // One bit per window, newest in bit 0.
} ha_presence_t;

typedef struct {
    unsigned alerts;
    bool cooling;
    int64_t quiet_since_ms;
    int64_t last_alert_ms;
} ha_episode_t;

void ha_presence_update(ha_presence_t *state, const uint8_t score[3],
                        const uint8_t cuts[2], bool present[2]);
bool ha_episode_update(ha_episode_t *state, bool present, int64_t now_ms,
                       unsigned rearm_ms, unsigned repeat_ms, unsigned max_alerts);
void ha_episode_gap(ha_episode_t *state, int64_t now_ms);

#ifdef __cplusplus
}
#endif
