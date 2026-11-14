#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace hearing_research {
// Only for a three-class, per-tensor int8 RAW-LOGIT output contract.
// This returns scores/flags only; notification cooldown and device IO are not
// implemented here. Thresholds must be fitted on the actual delivered model.
class IntegerMarginReadout {
 public:
  IntegerMarginReadout(const std::int16_t threshold[2],
                      const int need[2], const int width[2]) {
    for (int k = 0; k < 2; ++k) {
      threshold_[k] = threshold[k]; need_[k] = need[k]; width_[k] = width[k];
    }
  }
  void Reset() {
    for (int k = 0; k < 2; ++k) {
      count_[k] = 0; head_[k] = 0;
      for (int j = 0; j < 3; ++j) history_[k][j] = 0;
    }
  }
  void Step(const std::int8_t logits[3], std::uint8_t flags[2]) {
    for (int k = 0; k < 2; ++k) {
      std::int16_t other = -128;
      for (int j = 0; j < 3; ++j)
        if (j != k) other = std::max(other, static_cast<std::int16_t>(logits[j]));
      const auto margin = std::max<std::int16_t>(0, static_cast<std::int16_t>(logits[k]) - other);
      history_[k][head_[k]] = margin; head_[k] = (head_[k] + 1) % width_[k];
      count_[k] = std::min(count_[k] + 1, width_[k]);
      std::int16_t sorted[3] = {0, 0, 0};
      for (int j = 0; j < count_[k]; ++j) sorted[j] = history_[k][j];
      for (int a = 0; a < count_[k]; ++a)
        for (int b = a + 1; b < count_[k]; ++b)
          if (sorted[b] > sorted[a]) std::swap(sorted[a], sorted[b]);
      const auto score = count_[k] >= need_[k] ? sorted[need_[k] - 1] : 0;
      flags[k] = score > 0 && score >= threshold_[k];
    }
  }
 private:
  std::int16_t threshold_[2]; int need_[2]; int width_[2];
  std::int16_t history_[2][3] = {}; int count_[2] = {}; int head_[2] = {};
};
}  // namespace hearing_research

