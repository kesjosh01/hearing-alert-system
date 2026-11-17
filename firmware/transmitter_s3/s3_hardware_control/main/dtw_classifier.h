
#ifndef DTW_CLASSIFIER_H
#define DTW_CLASSIFIER_H

#include <stdint.h>

#define DTW_AUDIO_SAMPLE_RATE 16000
#define DTW_WINDOW_SAMPLES 16000
#define DTW_FRAME_COUNT 100
#define DTW_FEATURE_DIM 32

typedef enum {
    SOUND_UNKNOWN = 0,
    SOUND_FIRE_ALARM,
    SOUND_DOORBELL
} sound_class_t;

typedef struct {
    sound_class_t sound_class;
    float fire_distance;
    float doorbell_distance;
} dtw_result_t;

/**
 * @brief Classify one second of PCM audio.
 *
 * @param pcm Input PCM16 mono audio (16000 samples).
 * @return Classification result and DTW distances.
 */
dtw_result_t dtw_classify(const int16_t *pcm);

#endif
