# 마이크 입력·모델 출력·통신 연결 점검 — 2026-11-20

## 결론과 점검 범위

**마이크 수집 형식은 native40 DS-CNN에 연결 가능한 구성이다. 그러나 현재 앱은 DTW 데모이며, 학습한 DS-CNN을 거쳐 무선 알림을 보내는 통합 앱은 아직 없다.** 출력의 이벤트 ID 변환·무선 프레임·수신 화면과 진동까지 계획되어 있고 통신/UI 쪽 코드는 구현되어 있다. 모델 판정과 통신 API 사이의 연결이 남았다.

점검 기준은 GitHub `main`의 `a8eacbcf85b84b43a3552ca30bcd27b485635568`이다. 전체 추적 파일 **1,381개**의 경로를 확인하고, 입력·출력과 연결되는 앱 진입점, 분류기, 전처리, 모델 메타데이터, 빌드 구성, 통신 코어, 수신 UI, 보조 도구와 시험 기록을 교차 확인했다. 영역은 `firmware/` 86개, `ml/` 1,254개, `docs/` 23개, `communication_tests/` 8개, `tools/` 6개 및 루트 파일이다. 모든 과거 체크포인트의 재추론을 뜻하지 않는다.

이번 점검은 소스·설정·저장 학습/검증 기록과 파일 무결성 확인이다. 마이크 수집, GPU/CPU 모델 추론, 학습, ESP-IDF 빌드, 펌웨어 업로드, 무선·LCD·진동 시험을 새로 실행하지 않았다. 이 문서는 필요한 후속 작업을 정리하며 펌웨어 구현을 변경하지 않는다.

## 1. 현재 서로 분리된 세 경로

| 경로 | 실제 하는 일 | 아직 없는 연결 |
|---|---|---|
| [`s3_hardware_control/main/main.c`](../../firmware/transmitter_s3/s3_hardware_control/main/main.c) | I2S 마이크 → PCM → `dtw_classify()` → RGB LED | DS-CNN 호출·모델 판정·`ha_publish_event()` 호출 없음 |
| [`transmitter_s3/main/main.c`](../../firmware/transmitter_s3/main/main.c) | `ha_sender_start()`로 SoftAP/UDP 송신부 시작 | 오디오·추론 이벤트 공급자 없음. 기본 앱은 API 입력을 기다림 |
| [`dscnn_runner.cc`](../../firmware/transmitter_s3/dscnn/dscnn_runner.cc) | int8 TFLite 초기화, PCM → log-Mel → 추론 → 3개 값 반환 | 앱 진입점·마이크·알림 판정·통신을 소유하지 않는 독립 컴포넌트 |

제품 [`transmitter_s3/CMakeLists.txt`](../../firmware/transmitter_s3/CMakeLists.txt)는 공통 통신 컴포넌트만 추가하고 DS-CNN은 명시적으로 제외한다. [`main/CMakeLists.txt`](../../firmware/transmitter_s3/main/CMakeLists.txt)도 `ha_sender`만 요구한다. 따라서 같은 저장소에 코드가 있다는 사실만으로 한 펌웨어에 연결되지는 않는다.

## 2. 마이크 입력 적합성

### 원시 소리 형식은 조건부로 적합

현재 마이크 코드의 설정은 16,000 Hz, 왼쪽 mono, Philips I2S, 32-bit 수신 슬롯이다. INMP441 L/R=GND를 가정하고 GPIO4/5/6을 BCLK/WS/DIN으로 쓴다. 읽은 `int32_t`를 `raw >> 16`으로 변환해 signed PCM16 버퍼에 넣고 16,000 samples를 모은다. 코드 위치는 `main.c:290–403`, `447–517`이다.

INMP441은 24-bit I2S 마이크이며, 표준 I2S의 슬롯과 Philips 한 비트 지연 설정에 맞춰 읽어야 한다. 32-bit 워드 상위에 유효 데이터가 정렬되어 들어온다는 전제에서 현재 상위 16-bit 추출은 PCM16 변환으로 사용할 수 있다. 다만 실제 배선·샘플 정렬·부호·소리 크기·클리핑·샘플 연속성은 설정만으로 검증되지 않는다. 근거: [TDK INMP441 데이터시트](https://product.tdk.com/system/files/dam/doc/product/sw_piezo/mic/mems-mic/data_sheet/inmp441.pdf), [Espressif ESP32-S3 I2S 문서](https://docs.espressif.com/projects/esp-idf/en/release-v5.5/esp32s3/api-reference/peripherals/i2s.html).

학습 데이터와 마이크의 거리·감도·배경 소음이 다르면 같은 PCM 형식이어도 성능은 달라진다. 현재 저장소에는 이 마이크의 새 녹음을 해당 체크포인트로 평가해 그 차이를 확인한 결과가 없다. `receive_wav.py`는 16 kHz·10초 PCM16 전송을 기대하지만, 현재 `main.c`에는 그 도구가 기다리는 `S3WAV001`·`R` 녹음 응답 경로가 없다. 과거 녹음 도구가 남아 있다는 이유로 현재 펌웨어에서 녹음 검증이 된 것으로 판단하지 않는다.

### DTW 특징을 DS-CNN에 넣으면 맞지 않음

| 항목 | 현재 마이크 DTW 데모 | native40 DS-CNN 계약 |
|---|---|---|
| 원시 창 | PCM16 16,000 samples, 1초 | 동일 |
| 창 이동 | 3,200 samples, **0.2초** | 8,000 samples, **0.5초** |
| 진폭 처리 | DC 평균 제거 후 창 peak로 나눔 | PCM/32768, peak 정규화 없음 |
| 특징 | 5개씩 파형 평균, **100×32** int8 | STFT power → **98×40** log-Mel |
| 분류 | DTW 기준 파형 거리 비교 | 학습한 신경망의 3종 출력 |
| 후처리 | 범주 투표 2/3, UNKNOWN에서 이력 초기화 | 선택한 모델의 점수 임계값·argmax·클래스별 시간 이력 |

`dtw_classifier.c:20–77`과 [`generate_dtw_reference.py`](../../firmware/transmitter_s3/s3_hardware_control/generate_dtw_reference.py)의 특징은 log-Mel이 아니다. `fire_alarm_reference.npy`, `doorbell_reference.npy`, `dtw_reference.h`도 DS-CNN 체크포인트나 입력 특징이 아니다. **PCM 버퍼 단계에서 DS-CNN 전처리로 분기**해야 한다. 창 이동 0.2초를 그대로 쓰면 2/3 판정의 시간 의미도 달라지므로 기존 0.5초 평가 결과를 인용할 수 없다.

DTW는 무음이 아니면 두 목표 중 거리가 작은 쪽을 택하고, 두 거리 모두 멀 때 기타로 거부하는 기준이 없다(`dtw_classifier.c:173–212`). `UNKNOWN`은 DS-CNN의 학습된 `other` 확률과 같지 않다. DTW의 투표를 DS-CNN 판정 정책으로 재사용하지 않는다.

### DS-CNN에 필요한 정확한 입력

[`log_mel_frontend.cc`](../../firmware/transmitter_s3/dscnn/log_mel_frontend.cc)는 mono PCM16 16 kHz, 1초 창을 PCM/32768로 바꾼다. periodic Hann 400 samples, frame hop 160, FFT 512, 40개 HTK Mel bin(80–7,600 Hz), power spectrum과 `ln(max(power,1e-6))`를 사용해 98×40 특징을 만든다. 전처리 계약 SHA-256은 `b0cfbccaf11745e46aa07d6b6451ec94d1dc3230faa4de12979ea0416ecc640d`다.

- PC용 새 과적합 `.keras`: float32 `[batch,98,40,1]` 입력.
- 기존 MCU용 `native40_r06_v1`: int8 `[1,98,40,1]`, 해당 모델 tensor의 scale/zero point로 양자화. 입력 scale `0.09342911839485168`, zero point `20`.
- 96×64 입력인 `general_prefix64_v1`에는 위 98×40 전처리를 사용할 수 없다.

마이크 수집과 분류가 같은 루프에서 순차 실행되고 DMA 버퍼는 유한하다. 분류·무선·LED 처리가 길어질 때 오디오 누락이 없는지 측정한 자료나 오버런 감지/복구 처리는 확인되지 않았다. 누락이 발생한다고 단정할 수는 없지만, 현재 코드만으로 연속 실시간 입력을 보장할 수 없다. 최종 앱에는 샘플 수·창 시작 시각·단절 감지를 명시하고 수집과 처리를 분리할 필요가 있다.

## 3. 모델 output에서 수신부 알림까지의 계획과 구현

원래 계획은 [`MODEL_HANDOFF.md` 4–5절](MODEL_HANDOFF.md)과 [`communication/README.md`](../communication/README.md)에 명시되어 있다. **출력 변환을 고려한 설계이며 통신 쪽 API 이후의 경로도 구현되어 있다.**

### 출력·판정·이벤트 ID

| 모델 결과 | 모델 인덱스 | 최종 판정 뒤 통신 ID | 수신부 표현 |
|---|---:|---:|---|
| 화재 `fire_alarm` | 0 | `0x00000100` / 256 | 화재 화면·화재 진동 패턴 |
| 초인종 `doorbell` | 1 | `0x00000101` / 257 | 벨 화면·초인종 진동 패턴 |
| 기타 `other` | 2 | 없음 | 송신하지 않음 |

모델 인덱스를 그대로 송신하면 안 된다. DTW enum도 `UNKNOWN=0, FIRE=1, DOORBELL=2`여서 DS-CNN의 클래스 순서와 다르다. TEST ID `1`은 과거 통신 진단용이며 화재 ID가 아니다. 등록 ID는 [`ha_events.h`](../../firmware/common/ha_comm/include/ha_events.h)·[`ha_events.c`](../../firmware/common/ha_comm/src/ha_events.c)가 공유한다.

기존 int8 후보의 정책은 `q+128` 점수에서 argmax로 선택된 클래스만 남기고, 클래스별 최근 3창의 두 번째로 큰 점수가 화재 190·초인종 162 이상인지 보는 방식이다. 출력은 `(q+128)/256`이다. 이 후보는 T=4로 logits를 조정한 **확률 모델**이다. raw logits를 내는 증류 후보와 새 T=1 과적합 `.keras`에는 이 수식을 그대로 적용하지 않는다.

양성 상태가 되었다고 매 창 새 이벤트를 보내는 것은 아니다. 문서상 ML/제어 계층이 상승 순간 감지, 같은 소리의 재알림 간격, 재무장, 오디오 단절 시 이력 초기화를 정한 뒤 `ha_publish_event()`를 호출해야 한다. 이 실제 어댑터는 아직 없다.

### API부터 전파·수신 알림까지

1. `ha_publish_event(HA_EVENT_FIRE_ALARM, ...)` 또는 `ha_publish_events(...)`로 **확정된 새 알림**을 접수한다. 같은 canonical ID를 송신부 로컬 알림 hook에도 전달한다. hook은 연결 지점이며 현재 DTW LED를 자동으로 제어하지 않는다.
2. [`ha_sender.c`](../../firmware/transmitter_s3/components/ha_sender/ha_sender.c) → [`ha_tx.c`](../../firmware/common/ha_comm/src/ha_tx.c)의 RAM 큐(기본 8개, 활성 포함)에 넣는다. 한 배치의 화재와 벨은 별도 이벤트로 화재 먼저 접수한다. 단순 bit OR로 합치지 않는다.
3. 기존 SoftAP + UDP 4210번 포트로 **16바이트 v1 EVENT**를 보낸다. 구조는 `HA(2B) | version=1(1B) | type=EVENT/2(1B) | session(4B) | seq(4B) | event_id(4B)`이며 다중 바이트 정수는 big-endian이다. 확률 벡터·음원·모델 인덱스는 전송하지 않는다. [`ha_proto.c`](../../firmware/common/ha_comm/src/ha_proto.c)의 명시적 인코더를 사용한다.
4. stop-and-wait로 원래 `(session, seq, event_id)`를 유지하며 ACK를 기다린다. ACK 대기 기본 700ms, 한 burst 4회, 이후 backoff 2→4→8→16초 상한이다. 기본 TTL은 화재 10분, 벨 2분이다. 값은 소프트웨어 설계 기본값이며 실측 보증 시간이 아니다. 큐가 가득 차거나 TTL이 끝났을 때의 실패/폐기 상태를 앱에서 처리해야 한다.
5. 현재 수신부는 **XIAO ESP32-C6**다. [`ha_receiver.c`](../../firmware/receiver_c6/components/ha_receiver/ha_receiver.c)는 프레임·등록 ID·세션/순번·중복을 검사한 뒤 UI 큐에 넣는다. UI 소비자가 없거나 큐가 가득 차면 접수하지 않고 ACK도 보내지 않는다. 중복 수신은 새 알림을 만들지 않는다.
6. [`ha_ui.c`](../../firmware/receiver_c6/components/ha_ui/ha_ui.c) → [`ha_presenter.c`](../../firmware/receiver_c6/components/ha_ui_core/src/ha_presenter.c)가 화면과 진동을 선택한다. 화재가 벨보다 우선하며 둘 다 있으면 화재 화면에 벨 배지가 붙는다. 기본 화재 진동은 500ms 켜짐/200ms 꺼짐 반복, 초인종은 200ms/150ms/200ms/2,500ms 리듬이다.
7. 기본 `ACK_ON_DISPLAY`에서는 화면 그리기 전송이 성공한 뒤 `ha_receiver_alert_displayed()`로 원래 세 필드를 되돌린다. S3는 세 값과 송신 상대를 모두 확인한다. ACK는 사용자가 보았다는 확인이나 진동 완료 확인이 아니다. 기본 LCD 읽기 확인이 꺼져 있으므로 실제 패널 표시까지 검증한 하드웨어 보증도 아니다. 진단 옵션은 ACK 시점을 바꿀 수 있다.

## 4. 빌드 구성·문서와 실제 코드의 차이

- `s3_hardware_control/CMakeLists.txt` 최상위가 `idf_component_register()`만 포함한다. 독립 ESP-IDF 프로젝트에 필요한 `cmake_minimum_required`, `project.cmake` include, `project()`가 없고 소스 경로도 루트의 `main.c`를 가리킨다. 내부 `main/CMakeLists.txt`에는 컴포넌트 선언이 있지만, 현재 루트에서 README의 `idf.py build`가 재현되는 구성은 아니다. 이번에는 빌드하지 않았으며 이는 정적 구성 판단이다.
- 마이크 데모 README는 WS2812B+레벨 시프터를 설명하지만 현재 `main.c`는 공통 음극 RGB LED를 GPIO7/9/10의 LEDC PWM으로 구동한다. 문서 핀맵을 그대로 현재 LED 코드에 대응시킬 수 없다.
- 제품 S3는 정상 통신용 빌드 진입점을 갖지만 DS-CNN·마이크 앱을 포함하지 않는다. `esp-tflite-micro` 의존성도 최종 프로젝트에서 제공해야 한다.
- `communication_tests/softap_udp`는 C3·TEST 전용 과거 실험이다. 제품은 C6로 옮겨졌고 일부 내부 변수·로그·문서에 C3 이름이 남아 있다. 과거 TEST 송수신 성공은 소리 인식→현재 C6 화면/진동 성공의 근거가 아니다.
- `tools/communication/analyze_log.py`는 TEST 로그용이다. 제품 로그·마이크 추론 로그의 정확도/종단 지연 분석기로 간주하지 않는다.

## 5. 저장된 검증 기록과 남은 일

[`receiver_c6_2026-11-17.md`](../test_results/receiver_c6_2026-11-17.md)는 당시 PC 통신 시험 735 checks·UI 시험 124 checks 통과, C6/S3 빌드 성공을 기록한다. C6 벤치는 **LCD·모터·버튼 없이 USB만 연결한 시험**이다. 실제 화면/진동, S3→C6 제품 알림, 마이크→모델→사용자 알림, 전체 RAM·지연·오알림은 그 기록에서도 미검증이다. 이번에는 해당 시험을 재실행하지 않았다.

다음 작업이 순서상 필요하다. 이는 후속 계획이며 이번 변경에 구현하거나 실행하지 않았다.

1. 사용할 모델 버전과 전처리·출력 의미를 고정한다. 새 과적합 모델을 보드에서 쓸 경우 해당 가중치로 int8 변환·비교 검증·헤더 생성을 별도로 한다.
2. 실제 마이크 PCM의 샘플 정렬·레벨·클리핑·연속성을 확인하고 같은 소리를 PC 전처리와 대조한다. DTW의 peak 정규화 특징 대신 원시 PCM을 보존한다.
3. 한 제품 앱에 마이크 수집, 1초/0.5초 창 관리, DS-CNN 초기화와 호출, 모델별 판정, canonical ID 매핑, 상승 순간/재무장/반복 억제, `ha_publish_event()` 상태 처리를 연결한다.
4. 선택한 가중치·양자화·시간 정책의 감지와 비목표 오알림을 확인한다. 새 모델의 학습 파일 43/43을 실사용 감지율로 대신하지 않는다.
5. 그다음 보드에서 입력 수집→모델 판정→API→UDP→C6 화면/진동→ACK 전 과정을 같은 이벤트 기록으로 검증하고 RAM·처리 시간·누락·재전송·단절·중복·오알림을 측정한다.

새 과적합 모델의 파일·학습 기록·범위는 [`domestic_overfit_20261119`](../../ml/models/domestic_overfit_20261119/README.md)에 있다. 이번 게시에는 이 모델과 설명/기록을 추가했으며 기존 펌웨어·모델·통신 기록을 변경하지 않았다.
