#include "log_mel_frontend.h"

extern "C" int pursuit_logmel(const int16_t* pcm, size_t samples,
                              float* output, size_t elements) {
  hearing::FrontendWorkspace workspace;
  return hearing::LogMelFloat(pcm, samples, output, elements, workspace) ? 1 : 0;
}
