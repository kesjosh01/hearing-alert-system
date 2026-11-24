#include "log_mel_frontend.h"

#include <cmath>
#include <utility>

#include "frontend_tables.h"

namespace hearing {
namespace {

void Fft(FrontendWorkspace& w) {
  // In-place, unnormalized radix-2 FFT. Tables use the negative-angle convention.
  for (unsigned i = 1, j = 0; i < 512; ++i) {
    unsigned bit = 256;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      std::swap(w.real[i], w.real[j]);
      std::swap(w.imag[i], w.imag[j]);
    }
  }
  for (unsigned length = 2; length <= 512; length <<= 1) {
    const unsigned half = length / 2;
    for (unsigned start = 0; start < 512; start += length) {
      for (unsigned j = 0; j < half; ++j) {
        const unsigned a = start + j, b = a + half;
        const unsigned phase = j * (512 / length);
        const float re = w.real[b] * tables::kCos[phase] - w.imag[b] * tables::kSin[phase];
        const float im = w.real[b] * tables::kSin[phase] + w.imag[b] * tables::kCos[phase];
        w.real[b] = w.real[a] - re;
        w.imag[b] = w.imag[a] - im;
        w.real[a] += re;
        w.imag[a] += im;
      }
    }
  }
}

template <typename Store>
bool Compute(const int16_t* pcm, size_t samples, size_t elements,
             FrontendWorkspace& w, Store store) {
  if (!pcm || samples != kAudioSamples || elements != kFeatureElements) return false;
  for (size_t frame = 0; frame < kMelFrames; ++frame) {
    for (unsigned n = 0; n < 512; ++n) {
      w.real[n] = n < 400 ? (static_cast<float>(pcm[frame * 160 + n]) / 32768.0f) * tables::kHann[n] : 0.0f;
      w.imag[n] = 0.0f;
    }
    Fft(w);
    for (unsigned n = 0; n < 257; ++n) w.power[n] = w.real[n] * w.real[n] + w.imag[n] * w.imag[n];
    for (size_t bin = 0; bin < kMelBins; ++bin) {
      float energy = 0.0f;
      for (unsigned n = tables::kOffsets[bin]; n < tables::kOffsets[bin + 1]; ++n) {
        energy += w.power[tables::kFftBins[n]] * tables::kMelWeights[n];
      }
      const float value = std::log(energy < tables::kLogFloor ? tables::kLogFloor : energy);
      store(frame * kMelBins + bin, value);
    }
  }
  return true;
}
}  // namespace

bool LogMelFloat(const int16_t* pcm, size_t samples, float* output,
                 size_t elements, FrontendWorkspace& workspace) {
  if (!output) return false;
  return Compute(pcm, samples, elements, workspace,
                 [output](size_t i, float value) { output[i] = value; });
}

bool LogMelInt8(const int16_t* pcm, size_t samples, int8_t* output,
                size_t elements, float scale, int zero_point,
                FrontendWorkspace& workspace) {
  if (!output || !std::isfinite(scale) || scale <= 0 || zero_point < -128 || zero_point > 127) return false;
  return Compute(pcm, samples, elements, workspace, [=](size_t i, float value) {
    const float q = std::round(value / scale) + static_cast<float>(zero_point);
    output[i] = static_cast<int8_t>(q < -128 ? -128 : q > 127 ? 127 : q);
  });
}

}  // namespace hearing
