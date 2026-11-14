// Host-only numerical check utility. Never opens a microphone or device.
#include "log_mel_frontend.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  if (argc != 3 && argc != 5) return 2;
  int16_t pcm[hearing::kAudioSamples];
  hearing::FrontendWorkspace workspace;
  FILE* input = std::fopen(argv[1], "rb");
  if (!input) return 3;
  const size_t count = std::fread(pcm, sizeof(int16_t), hearing::kAudioSamples, input);
  const int extra = std::fgetc(input);
  std::fclose(input);
  if (count != hearing::kAudioSamples || extra != EOF) return 4;
  FILE* output = std::fopen(argv[2], "wb");
  if (!output) return 5;
  bool ok = false;
  size_t written = 0;
  if (argc == 3) {
    float feature[hearing::kFeatureElements];
    ok = hearing::LogMelFloat(pcm, count, feature, hearing::kFeatureElements, workspace);
    if (ok) written = std::fwrite(feature, sizeof(float), hearing::kFeatureElements, output);
  } else {
    int8_t feature[hearing::kFeatureElements];
    ok = hearing::LogMelInt8(pcm, count, feature, hearing::kFeatureElements,
                            std::strtof(argv[3], nullptr), std::atoi(argv[4]), workspace);
    if (ok) written = std::fwrite(feature, sizeof(int8_t), hearing::kFeatureElements, output);
  }
  std::fclose(output);
  return ok && written == hearing::kFeatureElements ? 0 : 6;
}
