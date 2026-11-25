#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "detection.h"
#include "dscnn_runner.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "ha_sender.h"
#include "microphone.h"
#include "model_int8.h"

namespace {
constexpr char TAG[] = "ha_detect";
constexpr uint8_t cuts[] = {HA_FIRE_CUT, HA_BELL_CUT};
static_assert(HA_HOP_SAMPLES * 2 == hearing::kAudioSamples);
int16_t window[hearing::kAudioSamples];
hearing::FrontendWorkspace frontend;

struct Decisions {
    ha_presence_t presence{};
    ha_episode_t fire{}, bell{};

    void gap(int64_t now) {
        presence = {};
        ha_episode_gap(&fire, now);
        ha_episode_gap(&bell, now);
    }

    void update(const uint8_t scores[3], int64_t now) {
        bool present[2];
        ha_presence_update(&presence, scores, cuts, present);
        uint32_t ids[2];
        ha_episode_t *episodes[2];
        ha_episode_t before[2];
        size_t n = 0;
        before[n] = fire;
        if (ha_episode_update(&fire, present[0], now, CONFIG_HA_POL_FIRE_REARM_MS,
                              CONFIG_HA_POL_FIRE_REALERT_MS, CONFIG_HA_POL_FIRE_MAX_ALERTS)) {
            episodes[n] = &fire;
            ids[n++] = HA_EVENT_FIRE_ALARM;
        }
        before[n] = bell;
        if (ha_episode_update(&bell, present[1], now, CONFIG_HA_POL_DOORBELL_REARM_MS, 0, 1)) {
            episodes[n] = &bell;
            ids[n++] = HA_EVENT_DOORBELL;
        }
        if (n) {
            ha_publish_result_t results[2];
            ha_publish_events(ids, n, results, nullptr);
            for (size_t i = 0; i < n; ++i) {
                if (results[i] == HA_PUBLISH_OK || results[i] == HA_PUBLISH_OK_EVICTED_OLDEST) {
                    ESP_LOGI(TAG, "detected %s at %lld ms -> %s", ha_event_name(ids[i]),
                             (long long)now, ha_publish_result_name(results[i]));
                } else {
                    // Not queued: forget this window so the alert is not used up.
                    // The next window that still detects the sound retries it.
                    *episodes[i] = before[i];
                    ESP_LOGW(TAG, "detected %s at %lld ms, publish rejected: %s; retry next window",
                             ha_event_name(ids[i]), (long long)now, ha_publish_result_name(results[i]));
                }
            }
        }
    }
};

void run(uint8_t *arena) {
    hearing::DscnnRunner runner;
    auto status = runner.Initialize(hearing::kModelData, hearing::kModelBytes,
                                    hearing::kModelPipelineOnly, hearing::kModelFrontendSha256,
                                    arena, CONFIG_HA_DET_ARENA_BYTES);
    if (status != hearing::DscnnStatus::kOk) {
        ESP_LOGE(TAG, "%s initialization failed: status=%d", HA_MODEL_NAME, (int)status);
        return;
    }
    ESP_LOGI(TAG, "%s ready, tensor arena %u/%u bytes", HA_MODEL_NAME, (unsigned)runner.ArenaUsedBytes(),
             (unsigned)CONFIG_HA_DET_ARENA_BYTES);
    esp_err_t err = ha_mic_start();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "microphone startup failed: %s", esp_err_to_name(err));
        return;
    }

    Decisions decisions;
    bool have_previous = false;
    uint32_t expected = 0;
    unsigned windows = 0;
    int64_t total_us = 0, max_us = 0;
    for (;;) {
        ha_audio_hop_t *hop = ha_mic_take();
        int64_t end_ms = hop->end_ms;
        if (hop->gap || hop->seq != expected) {
            if (have_previous) ESP_LOGW(TAG, "audio discontinuity at hop %lu", (unsigned long)hop->seq);
            decisions.gap(end_ms);
            have_previous = false;
        }
        expected = hop->seq + 1;
        constexpr size_t hop_bytes = sizeof(hop->samples);
        if (have_previous) std::memcpy(window, window + HA_HOP_SAMPLES, hop_bytes);
        std::memcpy(window + HA_HOP_SAMPLES, hop->samples, hop_bytes);
        ha_mic_release(hop);
        if (!have_previous) {
            have_previous = true;
            continue;
        }

        float probabilities[3];
        int64_t begin = esp_timer_get_time();
        status = runner.Classify(window, hearing::kAudioSamples, frontend, probabilities);
        int64_t elapsed = esp_timer_get_time() - begin;
        if (status != hearing::DscnnStatus::kOk) {
            ESP_LOGE(TAG, "classification failed: status=%d", (int)status);
            decisions.gap(end_ms);
            continue;
        }
        uint8_t scores[3];
        for (unsigned i = 0; i < 3; ++i)
            scores[i] = static_cast<uint8_t>(std::lround(probabilities[i] * 256.0f));
#if CONFIG_HA_DET_SCORE_LOG
        int peak = 0;
        for (int16_t sample : window) peak = std::max(peak, std::abs((int)sample));
        ESP_LOGI(TAG, "fire=%u bell=%u other=%u PCM peak=%d", scores[0], scores[1], scores[2], peak);
#endif
        decisions.update(scores, end_ms);
        max_us = std::max(max_us, elapsed);
        total_us += elapsed;
        if (++windows == 60) {
            ESP_LOGI(TAG, "inference avg=%lld ms max=%lld ms (500 ms hop)",
                     (long long)(total_us / windows / 1000), (long long)(max_us / 1000));
            windows = 0;
            max_us = total_us = 0;
        }
        if (elapsed >= 450000) {
            ESP_LOGW(TAG, "slow inference: %lld ms", (long long)(elapsed / 1000));
            vTaskDelay(1); // Let the core's idle task run even with a sustained backlog.
        }
    }
}
} // namespace

extern "C" void ha_detector_task(void *) {
    auto *arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(
        16, CONFIG_HA_DET_ARENA_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (arena) {
        run(arena);
        heap_caps_free(arena);
    } else {
        ESP_LOGE(TAG, "cannot allocate %u bytes of PSRAM; check T7 S3 PSRAM configuration",
                 (unsigned)CONFIG_HA_DET_ARENA_BYTES);
    }
    vTaskDelete(nullptr);
}
