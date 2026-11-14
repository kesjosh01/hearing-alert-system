#pragma once

#include <cstddef>
#include <cstdint>

namespace hearing {

constexpr size_t kAudioSamples = 16000;
constexpr size_t kMelFrames = 98;
constexpr size_t kMelBins = 40;
constexpr size_t kFeatureElements = kMelFrames * kMelBins;

// Caller owns the workspace and input/output buffers. No heap allocation.
struct FrontendWorkspace {
  float real[512];
  float imag[512];
  float power[257];
};

bool LogMelFloat(const int16_t* pcm, size_t samples, float* output,
                 size_t elements, FrontendWorkspace& workspace);
bool LogMelInt8(const int16_t* pcm, size_t samples, int8_t* output,
                size_t elements, float scale, int zero_point,
                FrontendWorkspace& workspace);

}  // namespace hearing
