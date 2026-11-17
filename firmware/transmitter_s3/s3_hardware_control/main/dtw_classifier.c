
#include "dtw_classifier.h"
#include "dtw_reference.h"

#include <math.h>
#include <float.h>
#include <limits.h>
#include <stddef.h>

#define DTW_INF FLT_MAX

// 16-bit PCM 기준 0.5% full scale
#define SILENCE_THRESHOLD 163.0f

static bool extract_features(
    const int16_t *pcm,
    int8_t features[DTW_FRAME_COUNT][DTW_FEATURE_DIM]
)
{
    // 1. DC 성분 제거를 위한 평균 계산
    double mean = 0.0;

    for (int i = 0; i < DTW_WINDOW_SAMPLES; i++) {
        mean += (double)pcm[i];
    }

    mean /= (double)DTW_WINDOW_SAMPLES;

    // 2. RMS 및 peak 계산
    double sum_squared = 0.0;
    float peak = 0.0f;

    for (int i = 0; i < DTW_WINDOW_SAMPLES; i++) {
        float value = (float)((double)pcm[i] - mean);
        float absolute = fabsf(value);

        sum_squared += (double)value * (double)value;

        if (absolute > peak) {
            peak = absolute;
        }
    }

    float rms = sqrtf(
        (float)(sum_squared / (double)DTW_WINDOW_SAMPLES)
    );

    // 3. 무음 판정
    if (rms < SILENCE_THRESHOLD || peak < 1.0f) {
        return false;
    }

    // 4. 기존 Python 특징 추출 방식 유지
    for (int frame = 0; frame < DTW_FRAME_COUNT; frame++) {
        for (int feature = 0; feature < DTW_FEATURE_DIM; feature++) {

            float pooled = 0.0f;

            for (int k = 0; k < 5; k++) {
                int sample_index =
                    frame * 160 + feature * 5 + k;

                float normalized =
                    ((float)pcm[sample_index] - (float)mean) / peak;

                pooled += normalized;
            }

            pooled /= 5.0f;

            int quantized = (int)roundf(pooled * 127.0f);

            if (quantized > 127) quantized = 127;
            if (quantized < -127) quantized = -127;

            features[frame][feature] = (int8_t)quantized;
        }
    }

    return true;
}

static float dtw_distance(
    const int8_t reference[DTW_FRAME_COUNT][DTW_FEATURE_DIM],
    const int8_t test[DTW_FRAME_COUNT][DTW_FEATURE_DIM]
)
{
    static float previous[DTW_FRAME_COUNT + 1];
    static float current[DTW_FRAME_COUNT + 1];

    for (int j = 0; j <= DTW_FRAME_COUNT; j++) {
        previous[j] = DTW_INF;
    }

    previous[0] = 0.0f;

    for (int i = 1; i <= DTW_FRAME_COUNT; i++) {

        for (int j = 0; j <= DTW_FRAME_COUNT; j++) {
            current[j] = DTW_INF;
        }

        int j_start = i - DTW_RADIUS;
        int j_end = i + DTW_RADIUS;

        if (j_start < 1) j_start = 1;
        if (j_end > DTW_FRAME_COUNT) {
            j_end = DTW_FRAME_COUNT;
        }

        for (int j = j_start; j <= j_end; j++) {

            int difference_sum = 0;

            for (int k = 0; k < DTW_FEATURE_DIM; k++) {

                int difference =
                    (int)reference[i - 1][k] -
                    (int)test[j - 1][k];

                if (difference < 0) {
                    difference = -difference;
                }

                difference_sum += difference;
            }

            float local_cost =
                (float)difference_sum / (float)DTW_FEATURE_DIM;

            float minimum = previous[j];

            if (current[j - 1] < minimum) {
                minimum = current[j - 1];
            }

            if (previous[j - 1] < minimum) {
                minimum = previous[j - 1];
            }

            current[j] = local_cost + minimum;
        }

        for (int j = 0; j <= DTW_FRAME_COUNT; j++) {
            previous[j] = current[j];
        }
    }

    return previous[DTW_FRAME_COUNT] /
           (float)DTW_FRAME_COUNT;
}

static float find_best_distance(
    const int8_t references[][DTW_FRAME_COUNT][DTW_FEATURE_DIM],
    int reference_count,
    const int8_t test[DTW_FRAME_COUNT][DTW_FEATURE_DIM]
)
{
    float best_distance = DTW_INF;

    for (int i = 0; i < reference_count; i++) {

        float distance = dtw_distance(references[i], test);

        if (distance < best_distance) {
            best_distance = distance;
        }
    }

    return best_distance;
}

dtw_result_t dtw_classify(const int16_t *pcm)
{
    dtw_result_t result = {
        .sound_class = SOUND_UNKNOWN,
        .fire_distance = DTW_INF,
        .doorbell_distance = DTW_INF
    };

    if (pcm == NULL) {
        return result;
    }

    static int8_t test_features[DTW_FRAME_COUNT][DTW_FEATURE_DIM];

    // 무음이면 DTW 실행 없이 UNKNOWN 반환
    if (!extract_features(pcm, test_features)) {
        return result;
    }

    result.fire_distance = find_best_distance(
        fire_refs,
        FIRE_REF_COUNT,
        test_features
    );

    result.doorbell_distance = find_best_distance(
        doorbell_refs,
        DOORBELL_REF_COUNT,
        test_features
    );

    if (result.fire_distance < result.doorbell_distance) {
        result.sound_class = SOUND_FIRE_ALARM;
    }
    else if (result.doorbell_distance < result.fire_distance) {
        result.sound_class = SOUND_DOORBELL;
    }

    return result;
}
