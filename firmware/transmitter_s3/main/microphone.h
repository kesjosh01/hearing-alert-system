#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { HA_HOP_SAMPLES = 8000 };

typedef struct {
    int16_t samples[HA_HOP_SAMPLES];
    uint32_t seq;
    bool gap;
    int64_t end_ms;
} ha_audio_hop_t;

esp_err_t ha_mic_start(void);
// Single consumer. Return each hop before waiting for the next one.
ha_audio_hop_t *ha_mic_take(void);
void ha_mic_release(ha_audio_hop_t *hop);

#ifdef __cplusplus
}
#endif
