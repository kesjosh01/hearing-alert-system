#include "detection.h"

void ha_presence_update(ha_presence_t *state, const uint8_t score[3],
                        const uint8_t cuts[2], bool present[2]) {
    unsigned winner = 0;
    for (unsigned i = 1; i < 3; ++i)
        if (score[i] > score[winner]) winner = i;

    for (unsigned i = 0; i < 2; ++i) {
        bool hit = winner == i && score[i] > 0 && score[i] >= cuts[i];
        uint8_t votes = ((state->votes[i] << 1) | hit) & 7;
        state->votes[i] = votes;
        // A second-highest score >= cut means at least two passing windows.
        present[i] = (votes & (votes - 1)) != 0;
    }
}

bool ha_episode_update(ha_episode_t *state, bool present, int64_t now_ms,
                       unsigned rearm_ms, unsigned repeat_ms, unsigned max_alerts) {
    if (state->cooling && now_ms - state->quiet_since_ms >= rearm_ms)
        *state = (ha_episode_t){0};
    if (!present) {
        if (state->alerts && !state->cooling) {
            state->cooling = true;
            state->quiet_since_ms = now_ms;
        }
        return false;
    }
    state->cooling = false;
    if (state->alerts && (!repeat_ms || state->alerts >= max_alerts ||
                          now_ms - state->last_alert_ms < repeat_ms))
        return false;
    ++state->alerts;
    state->last_alert_ms = now_ms;
    return true;
}

void ha_episode_gap(ha_episode_t *state, int64_t now_ms) {
    // Missing samples prove neither presence nor silence. Keep the episode and
    // start measuring absence again from the first valid negative decision.
    state->cooling = false;
    if (state->alerts) state->last_alert_ms = now_ms;
}
