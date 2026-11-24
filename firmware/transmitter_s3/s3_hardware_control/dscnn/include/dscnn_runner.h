#pragma once

#include <cstddef>
#include <cstdint>

#include "log_mel_frontend.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace hearing {

// This component has no app_main, microphone, radio, GPIO or alert actions.
// It must be integrated and measured with the actual ESP-IDF/TFLM SDK later.
enum class DscnnStatus {
  kOk, kPipelineOnlyModel, kFrontendContractMismatch, kInvalidBuffer, kInvalidFlatbuffer, kSchemaMismatch,
  kOperatorRegistrationFailed, kAllocateFailed, kTensorContractMismatch,
  kNotInitialized, kFrontendFailed, kInvokeFailed
};

class DscnnRunner {
 public:
  DscnnRunner();
  ~DscnnRunner();
  DscnnRunner(const DscnnRunner&) = delete;
  DscnnRunner& operator=(const DscnnRunner&) = delete;

  // model_data and 16-byte-aligned arena must outlive this object. The caller
  // chooses arena_bytes after establishing the board's total RAM budget.
  DscnnStatus Initialize(const uint8_t* model_data, size_t model_bytes,
                         bool pipeline_only, const char* frontend_sha256,
                         uint8_t* arena, size_t arena_bytes);
  // Output order: fire_alarm=0, doorbell=1, other=2. No alert threshold policy.
  DscnnStatus Classify(const int16_t* pcm, size_t samples,
                       FrontendWorkspace& workspace, float probabilities[3]);
  size_t ArenaUsedBytes() const;

 private:
  void Reset();
  tflite::MicroMutableOpResolver<6> resolver_;
  bool registrations_ok_;
  alignas(tflite::MicroInterpreter) uint8_t interpreter_storage_[sizeof(tflite::MicroInterpreter)];
  tflite::MicroInterpreter* interpreter_ = nullptr;
  TfLiteTensor* input_ = nullptr;
  TfLiteTensor* output_ = nullptr;
};

}  // namespace hearing
