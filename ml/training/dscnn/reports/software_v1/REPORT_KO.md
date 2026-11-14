# ESP32-S3용 DS-CNN 소프트웨어 개발 결과

기록: 2026-11-10T02:32:42.481550+09:00

요청한 범위의 전처리·모델·학습/변환 코드 개발과 합성 입력 로컬 검증을 완료했다. **실제 음원으로 학습한 DS-CNN 및 ESP32-S3 동작 검증은 아직 없다.** 모든 생성 모델은 `pipeline_only=true`, `deployment_ready=false`다. 개발 완료와 분류 성능 적합성을 별도로 판단한다.

## 선행 조건과 보존

YAMNet 기준 모델의 refit·최종 평가 완료를 기존 `FINAL_VERIFICATION.json`과 실제 분류기 SHA256으로 확인했다. 기준 모델 가중치 1, seed 42, refit 18 epoch 기록을 변경하지 않았다. 모델 SHA256은 `70b9da2319abecf27fb55ef748ca3e65ceb2dcb21207eb9e32f3bb9f446982b2`다. 기존 test 결과는 이번 DS-CNN의 설계·수치 허용오차 선택이나 성능 비교에 사용하지 않았다.

작업 전후 YAMNet data/raw·split·processed 임베딩·설정·코드·모델/runs·기존 보고서 등 **9,339개 파일의 SHA256**이 같고 `.venv`, `.venv-gpu`, `.venv-gpu-official221`의 패키지 목록이 동일하다. 기존 metadata CSV 2개의 작업 전 변경도 그대로 보존했다. 새 설치·환경 생성·시스템/드라이버 변경은 0회다. 공식 가상환경을 읽어 사용하고 런타임 캐시는 DS-CNN 하위에 둔다.

고정 test의 기존 YAMNet 임베딩 **411개를 보존**했다. 새 DS-CNN test 특징 0개, test 음원 디코딩 0개, test 재평가 0회다. 실제 마이크 접근, GPU 학습/추론, 보드/ESP-IDF 실행도 0회다.

## 구현

같은 `fire_alarm=0`, `doorbell=1`, `other=2`와 원본 그룹 프로토콜 `ko-source-cv4-20261108-v1`을 직접 읽는다. 모든 4 folds의 그룹 격리·고정 test 멤버십을 기존 검사로 확인했으며 split을 재생성하지 않았다. 테스트 행은 DS-CNN 특징 추출 단계에서도 거부한다. 개발용 타깃 음원 32개의 이벤트 구간 주석이 필요하며, 현재 빈 CSV 양식만 만들었다.

mono 16 kHz PCM16 1초 입력에서 25 ms Hann/10 ms hop, float32 FFT512, HTK Mel40(80–7,600 Hz), 자연로그로 `[98,40,1]`을 만든다. 짧은 파일은 오른쪽 0 padding, 긴 파일은 0.5초 hop과 마지막 완전한 창을 사용한다. peak normalization·증강·모델 distillation은 적용하지 않았다.

모델은 24채널 stem과 depthwise/pointwise 3블록, average pool, 3출력 softmax다. 학습 코드는 주석 확인 후 원본 GroupSampler로 학습하고 새 모델/optimizer, seed `42+fold`, 국내 가중치 1을 기본으로 사용한다. 기존 검증 가드를 유지하며 train-only calibration 후 int8를 내보낸다. 실제 데이터 학습 코드는 이번에 실행 검증하지 않았다.

| 정적 크기 | 값 | 해석 |
|---|---:|---|
| 파라미터(BN 통계 포함) | 3,339 | 학습 파라미터만의 수와 다름 |
| conv/dense MAC/창 | 2,540,232 | frontend·pool·activation 제외 |
| 합성 검증 int8 모델 | 13,656 bytes | 실제 학습 후 재측정 |
| 최대 단일 int8 activation | 23,520 bytes | tensor arena 전체 크기가 아님 |
| 모델 int8 입력 | 3,920 bytes | arena 내 입력 |
| C++ FFT workspace | 5,124 bytes | caller 소유, heap 없음 |
| PCM16 1초 | 32,000 bytes | 스트리밍 버퍼 추가 가능 |
| 보드 전체 RAM / 창 처리 시간 | 미정 / 미측정 | 임의 예산을 넣지 않음 |

## 소프트웨어 검증

| 검사 | 결과 |
|---|---|
| `pip check` | 통과, `No broken requirements found.` |
| 단위 테스트 | 13개 통과 |
| 라벨·4-fold 그룹/holdout 검사 | 통과, metadata만 사용 |
| 합성 CPU 순전파·gradient·Adam update | 2회, 순전파와 gradient의 CPU 배치 기록 |
| `.keras` 저장/재로딩 | 최대 확률 차이 0.0 |
| 전 정수형 변환 | 입력/출력 int8, 중간 int8/int32, float/Flex 없음 |
| 변환 연산 | Conv2D, DepthwiseConv2D, AveragePool2D, FullyConnected, Softmax |
| 합성 float/int8 최대 확률 차이 | 0.0019466579, 사전 smoke 기준 ≤0.05 |
| C++/Python log-Mel | 27개 합성 PCM, 최대 절대 차이 9.536743164e-07, 기준 ≤0.001 |
| C++/Python int8 모델 입력 | 27개 모두 0 LSB 차이, 기준 ≤1 LSB |
| C++ 입력 → host TFLite 출력 | 27개 모두 Python 입력 사용 출력과 동일 |
| LUT 최신성·Python 구문·CLI help | 통과 |

위 확률 오차 기준은 합성 경로의 소프트웨어 점검용이며 실제 데이터의 양자화 수용 기준이 아니다. 합성 float/int8 argmax는 **20/24 일치**로, [3, 6, 15, 18]에서 변경됐다. 이 창들의 float 상위 2개 확률 차이는 최대 0.0007236600로 output quantization 1/256보다 작다. 작은 모델에 임의 숫자 라벨로 2회 업데이트한 결과여서 세 클래스 인식률이나 정확도로 해석할 수 없다. 모델이 실제 소리를 구분한다는 증거는 아직 없다.

최종 합성 학습/저장/변환 검증은 3.780초, C++ 컴파일·전처리·host 모델 비교 전체는 1.950초였다. TF import·컴파일·변환을 포함한 host 작업 시간이며 ESP32-S3 추론 지연이나 처리량이 아니다.

## 최초 오류와 수정

첫 C++ frontend 비교는 합성 signal 11에서 log-Mel max abs **0.0015010834**가 기준 0.001을 넘어 종료 코드 1이었다. Python의 NumPy 기본 FFT(complex128)와 C++의 float32 butterfly 사이의 오차가 낮은 에너지 로그 값에서 나타났다. `frontend_host/status.json`과 최초 로그를 보존했다.

Python FFT를 C++와 같은 float32 radix-2 순서로 명시했다. 별도 단위 테스트에서 NumPy FFT와 power spectrum의 상대 L2 차이 <1e-6을 확인하고, **기존 허용오차를 유지**하여 27개 신호를 재검증했다. 합성 경로를 수정한 것이며 실제 음원·기존 YAMNet 특징·체크포인트는 변경하지 않았다. 첫 합성 검증, 수정 후 검증, 최종 export 헤더 검증의 출력 폴더를 모두 보존했다.

TensorFlow가 CPU 선택 상태에서 남긴 `CUDA_ERROR_NO_DEVICE` 및 `tf.lite.Interpreter` deprecation 경고는 원 로그에 보존했다. 새 패키지를 설치하거나 GPU로 전환하지 않았다. 최종 순전파·gradient device는 `/device:CPU:0`이다. 저장된 int8 파일은 host reference kernel로만 검사했다.

전체 `git diff --check`에는 작업 전부터 수정되어 있던 metadata CSV의 CRLF가 trailing whitespace로 보고됐다. 해당 파일은 수정하지 않았고 이번 변경 README 두 개에 한정한 검사는 통과했다. 파일 보존 검사는 작업 전후 동일함을 별도로 확인했다.

## 결과물과 다음 단계

- `../../dscnn_baseline/`: 전처리·모델·개발 학습·변환·합성 검증 CLI.
- `../../config/dscnn_v1.json`: 특징과 모델/학습 규약.
- `../../annotations/event_intervals.template.csv`: 헤더만 있는 주석 양식.
- `../../../../../firmware/transmitter_s3/dscnn/`: portable frontend, 생성 LUT, TFLM runner와 CMake 연결 정의.
- `final_selfcheck/status.json`, `final_selfcheck/packages.txt`, `final_selfcheck/export/`: 모델 구조·패키지·재로딩·int8 검증 증거.
- `final_frontend/status.json`: 실제 컴파일 명령, 27개 수치 비교·host 모델 출력 일치.
- `protected_before.json`, `protected_after.json`, `integrity.json`: 기존 파일/환경 보존 증거.
- `FINAL_VERIFICATION.json`: 완료 범위·미검증 사항·실행 종료 코드·코드 해시.

ESP32-S3 runner는 구간 주석·실제 학습·SDK 빌드·보드 검증 전의 연결 코드다. 공식 SDK 헤더와의 실제 크로스 컴파일, ESP-NN/TFLM 연산 실행, arena 할당 및 최대 RAM·처리 시간은 미검증이다. `pipeline_only` 모델을 초기화하려 하면 거부하도록 했다.

성능 비교 단계에서는 개발 타깃 구간 주석, 독립적인 실제 마이크 녹음, 보드 전체 RAM·처리 시간 예산을 먼저 준비해야 한다. 이때 source group 단위 평가와 이벤트/연속 오탐 정의를 확정하고, YAMNet의 파일 라벨 감독과 달라지는 점 및 공유 ESC-50 validation 중복을 구분해 보고한다. 이번에 이 비교나 최종 가중치 변경·refit·고정 test 평가를 진행하지 않았다.

공식 구현 참고: [Espressif TFLM](https://github.com/espressif/esp-tflite-micro), [LiteRT integer quantization](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant). 공식 라이브러리 문서 확인과 host 수치 검증은 실제 보드 성능 검증을 대신하지 않는다.
