#include "integer_margin_readout.h"

extern "C" int hearing_integer_margin_trace(
    const std::int8_t* logits, int windows, const std::int16_t* thresholds,
    const int* need, const int* width, std::uint8_t* output) {
  if (!logits || !thresholds || !need || !width || !output || windows < 0) return -1;
  for (int k = 0; k < 2; ++k)
    if (thresholds[k] < 1 || thresholds[k] > 256 || need[k] < 1 ||
        need[k] > width[k] || width[k] > 3) return -2;
  hearing_research::IntegerMarginReadout state(thresholds, need, width);
  state.Reset();
  for (int i = 0; i < windows; ++i) state.Step(logits + i * 3, output + i * 2);
  return 0;
}
