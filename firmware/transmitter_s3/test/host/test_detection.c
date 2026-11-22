#include <assert.h>
#include <stdio.h>
#include "detection.h"

static void presence(void) {
    const uint8_t cuts[] = {190, 162};
    const uint8_t rows[][3] = {
        {190, 20, 46}, {189, 20, 47}, {191, 20, 45}, {0, 162, 94},
        {0, 162, 94}, {0, 161, 95}, {0, 162, 94}, {0, 0, 255},
        {0, 0, 255}, {200, 200, 0}, {200, 200, 0}
    };
    const bool want[][2] = {
        {0,0}, {0,0}, {1,0}, {0,0}, {0,1}, {0,1}, {0,1},
        {0,0}, {0,0}, {0,0}, {1,0}
    };
    ha_presence_t history = {0};
    for (unsigned i = 0; i < sizeof(rows)/sizeof(rows[0]); ++i) {
        bool present[2];
        ha_presence_update(&history, rows[i], cuts, present);
        assert(present[0] == want[i][0] && present[1] == want[i][1]);
    }
}

static void episodes(void) {
    ha_episode_t fire = {0}, bell = {0};
    for (int ms = 0; ms <= 360000; ms += 500) {
        assert(ha_episode_update(&fire, true, ms, 10000, 30000, 10) ==
               (ms < 300000 && ms % 30000 == 0));
        assert(ha_episode_update(&bell, true, ms, 5000, 0, 1) == (ms == 0));
    }
    assert(!ha_episode_update(&bell, false, 361000, 5000, 0, 1));
    assert(!ha_episode_update(&bell, true, 365999, 5000, 0, 1));
    assert(!ha_episode_update(&bell, false, 366000, 5000, 0, 1));
    assert(ha_episode_update(&bell, true, 371000, 5000, 0, 1));
    assert(!ha_episode_update(&fire, false, 370000, 10000, 30000, 10));
    ha_episode_gap(&fire, 400000);
    assert(!ha_episode_update(&fire, true, 401000, 10000, 30000, 10));
    assert(!ha_episode_update(&fire, false, 402000, 10000, 30000, 10));
    assert(ha_episode_update(&fire, true, 412000, 10000, 30000, 10));
    ha_episode_gap(&fire, 450000);
    assert(!ha_episode_update(&fire, true, 451000, 10000, 30000, 10));
}

// detector.cc restores the pre-window state when the sender rejects an alert.
static void rejected_publish_retries(void) {
    ha_episode_t bell = {0}, before = bell;
    assert(ha_episode_update(&bell, true, 0, 5000, 0, 1));
    bell = before; // queue full
    before = bell;
    assert(ha_episode_update(&bell, true, 500, 5000, 0, 1)); // retried, not used up
    assert(!ha_episode_update(&bell, true, 1000, 5000, 0, 1));

    // A rejected rearm after cooling is retried too.
    assert(!ha_episode_update(&bell, false, 2000, 5000, 0, 1));
    before = bell;
    assert(ha_episode_update(&bell, true, 7000, 5000, 0, 1));
    bell = before;
    assert(ha_episode_update(&bell, true, 7500, 5000, 0, 1));

    ha_episode_t fire = {0};
    assert(ha_episode_update(&fire, true, 0, 10000, 30000, 10));
    for (int ms = 500; ms < 30000; ms += 500)
        assert(!ha_episode_update(&fire, true, ms, 10000, 30000, 10));
    before = fire;
    assert(ha_episode_update(&fire, true, 30000, 10000, 30000, 10));
    fire = before; // busy
    before = fire;
    assert(ha_episode_update(&fire, true, 30500, 10000, 30000, 10));
    assert(fire.alerts == 2 && fire.last_alert_ms == 30500);
    // The schedule continues from the alert that was actually queued.
    assert(!ha_episode_update(&fire, true, 60000, 10000, 30000, 10));
    assert(ha_episode_update(&fire, true, 60500, 10000, 30000, 10));
}

int main(void) {
    presence();
    episodes();
    rejected_publish_retries();
    puts("HA_DETECTION_HOST_TESTS_PASS");
}
