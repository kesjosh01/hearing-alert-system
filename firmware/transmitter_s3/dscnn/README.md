# DS-CNN 추론 연결 컴포넌트

1초 PCM16 → 98×40 log-Mel → int8 TFLM 모델의 연결 코드다. 실제 마이크·ESP-IDF 앱·알림·통신 로직은 포함하지 않는다. [Python 모델과 개발 규약](../../../ml/training/dscnn/README.md)을 먼저 확인한다.

2026-11-14 최신 후보와 실제 통신 ID 매핑은 [모델 연결 안내](../../../docs/ml/MODEL_HANDOFF.md)에 정리했다. [`native40_r06_v1`](../../../ml/models/pursuit_20261114/README.md)의 98×40 전처리·확률 출력은 이 runner의 입력·출력 계약과 맞는다. `Classify()` 이후의 2/3 시간 판정·이벤트 생성·반복 알림 제어는 앱에서 연결해야 한다. logits 출력 후보와 96×64 후보는 이 확률 경로에 그대로 대입하지 않는다.

- `log_mel_frontend.cc`: float32 radix-2 FFT, sparse Mel 필터, 자연로그, int8 양자화. 작업 버퍼는 호출자가 소유하고 내부 heap을 사용하지 않는다. Python과 같은 periodic Hann·진폭·round-half-away-from-zero 규약이다.
- `include/frontend_tables.h`: Python frontend에서 생성한 LUT와 규약 SHA256. 모델의 frontend 해시가 다르면 runner가 중단한다.
- `dscnn_runner.cc`: Conv2D, DepthwiseConv2D, AveragePool2D, Reshape, FullyConnected, Softmax만 등록한다. flatbuffer/schema·입출력 shape/type·pipeline-only 여부를 검사한다.
- `CMakeLists.txt`: 기존 ESP-IDF 프로젝트에 추가할 수 있는 컴포넌트 정의다. 공식 `esp-tflite-micro`가 이미 제공되어야 하며 의존성을 다운로드하지 않는다.

**검증된 범위:** Ubuntu `g++`로 portable frontend를 컴파일하고 합성 PCM 27개를 Python과 수치 비교했다. C++ frontend가 만든 int8 입력을 host TFLite에서 확인한다. **미검증:** 이 runner의 ESP-IDF 크로스 컴파일, ESP-NN kernels, `AllocateTensors`, 실보드 추론·지연·최대 RAM. TFLM SDK 헤더/API는 실제 통합 때 선택한 공식 컴포넌트 버전으로 확인해야 한다. 현재 코드는 [Espressif 공식 MicroInterpreter API](https://github.com/espressif/esp-tflite-micro/blob/master/tensorflow/lite/micro/micro_interpreter.h)를 대상으로 작성했다.

모델 헤더는 학습/검증 출력 폴더에 생성하며 이 폴더로 자동 복사하지 않는다. `pipeline_only_int8.h`는 합성 경로 확인용이다. runner의 `Initialize(..., kModelPipelineOnly, kModelFrontendSha256, arena, arena_bytes)`에 실제 값을 전달하면 이 합성 모델은 `kPipelineOnlyModel`로 거부된다. 배포용 모델로 이름만 바꿔 사용하지 않는다.

추후 통합 시 호출자가 모델 버퍼, 16-byte 정렬 tensor arena와 크기, `FrontendWorkspace`, 16,000개 mono PCM16 입력을 제공한다. 입력은 frontend가 model tensor에 직접 양자화하므로 전체 float Mel 출력 버퍼를 둘 필요는 없다. 출력 순서는 `fire_alarm`, `doorbell`, `other`다. 이 runner 자체에는 경보 임계값·시간 누적 처리가 없으며 모델 묶음의 정책을 별도로 연결한다.

RAM 예산을 임의로 정하지 않았다. frontend workspace 5,124 bytes, 1초 PCM 32,000 bytes, int8 입력 3,920 bytes 외에도 arena·FFT/LUT 저장 위치·TFLM scratch·stack·I2S 버퍼·통신 메모리가 필요하다. arena는 `AllocateTensors()` 성공 및 `ArenaUsedBytes()` 측정 후 전체 시스템 메모리와 함께 판단한다. PSRAM 배치나 동작 시간은 이번에 추정하지 않는다.

보드·SDK·마이크가 준비되고 실행이 승인되기 전에는 `idf.py build/flash/monitor`를 수행하지 않는다. 새 SDK/컴포넌트 설치도 별도 허가 대상이다.
