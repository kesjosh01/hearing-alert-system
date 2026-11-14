#include "dscnn_runner.h"

#include <cmath>
#include <cstring>
#include <new>

#include "frontend_tables.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

namespace hearing {

DscnnRunner::DscnnRunner() {
  registrations_ok_ = resolver_.AddConv2D() == kTfLiteOk;
  registrations_ok_ = (resolver_.AddDepthwiseConv2D() == kTfLiteOk) && registrations_ok_;
  registrations_ok_ = (resolver_.AddAveragePool2D() == kTfLiteOk) && registrations_ok_;
  registrations_ok_ = (resolver_.AddReshape() == kTfLiteOk) && registrations_ok_;
  registrations_ok_ = (resolver_.AddFullyConnected() == kTfLiteOk) && registrations_ok_;
  registrations_ok_ = (resolver_.AddSoftmax() == kTfLiteOk) && registrations_ok_;
}

DscnnRunner::~DscnnRunner() { Reset(); }

void DscnnRunner::Reset() {
  input_ = nullptr;
  output_ = nullptr;
  if (interpreter_) {
    interpreter_->~MicroInterpreter();
    interpreter_ = nullptr;
  }
}

DscnnStatus DscnnRunner::Initialize(const uint8_t* data, size_t bytes,
                                    bool pipeline_only, const char* frontend_sha256,
                                    uint8_t* arena, size_t arena_bytes) {
  Reset();
  if (pipeline_only) return DscnnStatus::kPipelineOnlyModel;
  if (!frontend_sha256 || std::strcmp(frontend_sha256, tables::kFrontendSha256) != 0) {
    return DscnnStatus::kFrontendContractMismatch;
  }
  if (!data || bytes < 8 || !arena || !arena_bytes || reinterpret_cast<uintptr_t>(arena) % 16 != 0) {
    return DscnnStatus::kInvalidBuffer;
  }
  flatbuffers::Verifier verifier(data, bytes);
  if (!tflite::VerifyModelBuffer(verifier)) return DscnnStatus::kInvalidFlatbuffer;
  const tflite::Model* model = tflite::GetModel(data);
  if (model->version() != TFLITE_SCHEMA_VERSION) return DscnnStatus::kSchemaMismatch;
  if (!registrations_ok_) return DscnnStatus::kOperatorRegistrationFailed;
  interpreter_ = new (interpreter_storage_) tflite::MicroInterpreter(model, resolver_, arena, arena_bytes);
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    Reset();
    return DscnnStatus::kAllocateFailed;
  }
  if (interpreter_->inputs_size() != 1 || interpreter_->outputs_size() != 1) {
    Reset();
    return DscnnStatus::kTensorContractMismatch;
  }
  input_ = interpreter_->input(0);
  output_ = interpreter_->output(0);
  if (!input_ || !output_ ||
      input_->type != kTfLiteInt8 || output_->type != kTfLiteInt8 ||
      input_->dims->size != 4 || input_->dims->data[0] != 1 ||
      input_->dims->data[1] != 98 || input_->dims->data[2] != 40 || input_->dims->data[3] != 1 ||
      output_->dims->size != 2 || output_->dims->data[0] != 1 || output_->dims->data[1] != 3 ||
      !std::isfinite(input_->params.scale) || input_->params.scale <= 0 ||
      !std::isfinite(output_->params.scale) || output_->params.scale <= 0) {
    Reset();
    return DscnnStatus::kTensorContractMismatch;
  }
  return DscnnStatus::kOk;
}

DscnnStatus DscnnRunner::Classify(const int16_t* pcm, size_t samples,
                                  FrontendWorkspace& workspace, float probabilities[3]) {
  if (!interpreter_) return DscnnStatus::kNotInitialized;
  if (!probabilities || !LogMelInt8(pcm, samples, input_->data.int8, kFeatureElements,
                                   input_->params.scale, input_->params.zero_point, workspace)) {
    return DscnnStatus::kFrontendFailed;
  }
  if (interpreter_->Invoke() != kTfLiteOk) return DscnnStatus::kInvokeFailed;
  for (size_t i = 0; i < 3; ++i) {
    probabilities[i] = (static_cast<int>(output_->data.int8[i]) - output_->params.zero_point) * output_->params.scale;
  }
  return DscnnStatus::kOk;
}

size_t DscnnRunner::ArenaUsedBytes() const {
  return interpreter_ ? interpreter_->arena_used_bytes() : 0;
}

}  // namespace hearing
