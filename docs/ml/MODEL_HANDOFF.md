# 학습 모델과 통신 연결 안내

2026-11-14 기준으로 통신 연결을 검토할 최신 후보는 **`native40_r06_v1`**이다. 실제 학습·int8 변환을 마쳤으며 기존 portable 98×40 전처리와 맞는다. 제품 성능을 검증한 모델은 아니므로, 연결 시험과 실제 감지 성능 검증을 구분한다.

## 1. 먼저 받을 파일

저장소 루트 기준 경로다. 아래 모델 파일과 정책을 같은 버전으로 사용한다.

| 파일 | 용도 |
|---|---|
| [`model_int8.h`](../../ml/models/pursuit_20261114/candidates/native40_r06_v1/model_int8.h) | ESP32-S3 펌웨어에 포함할 모델 바이트와 상수 |
| [`model_int8.tflite`](../../ml/models/pursuit_20261114/candidates/native40_r06_v1/model_int8.tflite) | 같은 모델의 실제 FlatBuffer, 88,280 bytes |
| [`model_metadata.json`](../../ml/models/pursuit_20261114/candidates/native40_r06_v1/model_metadata.json) | 입력·출력 shape, 양자화 scale/zero point, 연산 목록, 전처리 해시 |
| [`delivered_output_contract.json`](../../ml/models/pursuit_20261114/candidates/native40_r06_v1/delivered_output_contract.json) | 클래스 순서, 정수 임계값, 시간 판정 정책 |
| [`candidate.json`](../../ml/models/pursuit_20261114/candidates/native40_r06_v1/candidate.json) | 원본 학습 체크포인트와 모델 파일 SHA-256 |
| [`classifier.keras`](../../ml/models/checkpoints_20261114/dscnn/pursuit_20261114_r06_native40_supervised_bnrefit/classifier.keras) | PC용 원본 float 체크포인트 |

int8 모델의 SHA-256은 `f572fb19749774e5586f541839c7879ad43c63091950002b27430093bdb1b7fd`다. `native40_r06_supervised_bnrefit_v1`는 같은 모델·판정 정책을 가진 비교용 이름이다. 공개 시험 전에 이 후보를 선택했으며 시험 결과로 다시 선택하지 않았다.

원본 `.keras`는 온도 변환 전 모델이다. 배포 int8은 최종 Dense의 가중치·bias를 4로 나눈 뒤 변환했으므로, `.keras`의 확률에 int8 전용 임계값을 그대로 적용하면 안 된다. 펌웨어에는 위 int8 파일과 그 계약을 사용한다.

자체 수집한 국내 학습 음원으로 연결 경로를 점검할 때 비교할 수 있도록 [`domestic_known_v1`](../../ml/models/pursuit_20261114/candidates/domestic_known_v1/README.md)도 보관했다. 이 모델은 15,600 bytes이며 학습 원음 화재 17/17·초인종 26/26 감지를 기록했다. 최신 일반 후보와 같은 모델은 아니다. 확률 출력의 int8 `q` 임계값은 화재 **34**, 초인종 **93**, 시간 정책은 각각 **1/1**이며 해당 폴더의 계약을 따른다. 이 결과를 새로운 국내 기기나 실제 현장의 성능으로 제시하지 않는다.

다른 모델과 학습 체크포인트는 [모델 목록](../../ml/models/README.md)과 [체크포인트 색인](../../ml/models/checkpoints_20261114/README.md)에서 찾는다. `general_prefix64_v1`는 **96×64 magnitude-Mel** 입력이어서 기존 98×40 runner에 넣을 수 없다. `native40_r06_distilled*`는 **raw logits** 출력이므로 확률 모델의 판정식을 적용하면 안 된다.

## 2. 입력 계약

구현은 [`firmware/transmitter_s3/dscnn/`](../../firmware/transmitter_s3/dscnn/README.md)에 있다. 오디오 수집과 I2S 설정은 별도로 연결해야 한다.

| 항목 | 최신 후보의 값 |
|---|---|
| PCM | mono signed PCM16, 16 kHz |
| 창 / hop | 16,000 samples / 8,000 samples, 즉 1초 / 0.5초 |
| 진폭 | PCM을 32768로 나눔, 창별 peak normalization 없음 |
| STFT | periodic Hann 400 samples, hop 160, FFT 512 |
| Mel·로그 | power spectrum, 40 HTK Mel bins, 80–7,600 Hz, `ln(max(power, 1e-6))` |
| 모델 입력 | int8 `[1,98,40,1]` |
| 입력 양자화 | scale `0.09342911839485168`, zero point `20` |
| 모델 출력 | int8 `[1,3]`, 순서 `fire_alarm`, `doorbell`, `other` |
| 출력 양자화 | scale `1/256`, zero point `-128` |
| 전처리 SHA-256 | `b0cfbccaf11745e46aa07d6b6451ec94d1dc3230faa4de12979ea0416ecc640d` |

`LogMelInt8()`는 입력 tensor의 실제 scale/zero point로 양자화한다. 모델별 입력 scale을 다른 버전에서 복사하지 않는다. 짧은 파일의 zero padding과 마지막 완전한 창 추가는 저장 음원 평가 규칙이며, 실시간 스트림에서는 정상적으로 수집한 연속 창의 시작 시각을 유지해야 한다.

## 3. S3 추론 컴포넌트 연결

[`DscnnRunner`](../../firmware/transmitter_s3/dscnn/include/dscnn_runner.h)는 아래 순서로 사용하도록 작성돼 있다. 이 부분은 통합 시의 호출 계약이며 실제 ESP-IDF 앱의 구현·실행 완료를 뜻하지 않는다.

1. 모델 헤더 **한 개**를 포함한다. 여러 모델의 헤더는 같은 `hearing::kModelData` 등의 이름을 사용하므로 한 translation unit에 함께 넣지 않는다.
2. 모델 버퍼와 16-byte 정렬 tensor arena를 runner보다 오래 유지한다. arena 크기는 보드에서 `AllocateTensors()` 성공과 전체 RAM 사용량을 측정해 정한다.
3. `Initialize(kModelData, kModelBytes, kModelPipelineOnly, kModelFrontendSha256, arena, arena_bytes)`를 호출하고 `DscnnStatus::kOk`인지 확인한다. 초기화 오류가 있으면 해당 원인을 기록하고 추론 결과를 만들지 않는다.
4. 16,000 samples와 호출자가 소유한 `FrontendWorkspace`로 `Classify(pcm, samples, workspace, probabilities)`를 호출한다. 성공한 결과만 판정기에 넘긴다.
5. 창마다 아래 시간 판정을 갱신하고, 알림으로 접수할 결과만 통신 이벤트 큐에 전달한다. 오디오 단절·샘플 누락 때의 상태 초기화와 재개 정책도 앱에서 명시한다.

runner가 반환하는 `probabilities[k]`는 `(q[k] + 128) / 256`이다. 다음 절의 정수 점수는 정확한 int8 값에서 계산하는 것이 기준이다. 현재 runner의 dequantized 확률로 옮길 때는 `round(probabilities[k] * 256)`로 점수를 복원하고 같은 판정 결과인지 비교한다.

등록 연산은 `Conv2D`, `DepthwiseConv2D`, `AveragePool2D`, `Reshape`, `FullyConnected`, `Softmax`다. 이 후보의 실제 연산 목록과 일치한다. 컴포넌트는 공식 `esp-tflite-micro`가 제공되는 ESP-IDF 프로젝트에 추가하는 구성이고, SDK 의존성을 자동 설치하지 않는다.

## 4. 최신 후보의 시간 판정

**클래스 argmax만으로 송신하지 않는다.** 이 후보는 클래스별 정수 임계값과 최근 창의 2/3 정책을 함께 사용한다.

1. int8 출력 `q[0..2]`의 첫 번째 최댓값을 선택한다. 동점이면 인덱스가 작은 클래스가 이긴다.
2. 화재·초인종 각각에 대해, 그 클래스가 argmax이면 `q[k] + 128`, 아니면 `0`을 해당 창의 점수로 기록한다. `other`가 이긴 창은 두 점수가 모두 0이다.
3. 클래스마다 최근 최대 3개 창의 점수를 유지하고 **두 번째로 큰 값**을 시간 점수로 사용한다. 아직 2개 창이 없으면 시간 점수는 0이다.
4. 화재 시간 점수가 **190 이상**, 초인종이 **162 이상**이면 각각 양성이다. 두 클래스의 이력을 따로 유지한다.

동일 시각의 창에서는 argmax가 하나지만, 최근 3개 창의 이력에 따라 두 시간 판정이 함께 참일 수 있다. 원시 확률을 수정하거나 임계값을 낮춰 통과시키지 않는다. 원본 구현은 [`delivered_integer.scores()`](../../ml/training/dscnn/pursuit/delivered_integer.py)에 있다. `integer_margin_readout.h`는 logits 전용이므로 이 확률 후보의 판정기로 사용하지 않는다.

이 정책은 감지 상태를 계산한다. 같은 소리가 지속될 때 알림을 몇 번 보낼지, rising edge·재알림 간격·재무장·큐 포화·장기 단절·유효시간을 어떻게 처리할지는 최종 앱에서 결정하고 검증해야 한다. 매 hop마다 같은 이벤트를 새로 송신하면 반복 알림이 생길 수 있다. 모델 평가의 파일 감지 성공은 실제 사용자 알림 횟수의 검증이 아니다.

## 5. 모델 클래스와 기존 UDP 이벤트의 연결

[통신 프로토콜 v1](../communication/README.md)의 ID와 16바이트 프레임을 그대로 사용한다.

| 모델 결과 | 모델 인덱스 | 통신 이벤트 ID | 처리 |
|---|---:|---:|---|
| `fire_alarm` | 0 | `0x00000100` / 256 | 화재 시간 판정과 앱의 접수 조건을 통과한 이벤트 |
| `doorbell` | 1 | `0x00000101` / 257 | 초인종 시간 판정과 앱의 접수 조건을 통과한 이벤트 |
| `other` | 2 | 없음 | 송신하지 않음 |
| 통신 실험 `TEST` | 모델 클래스 아님 | `0x00000001` / 1 | 기존 통신 시험에만 사용 |

모델 인덱스 `0`, `1`, `2`를 UDP의 `event_id`로 바로 넣지 않는다. confidence나 오디오를 기존 프레임에 추가하지 않는다. 모델 판정 → 명시적 ID 매핑 → S3 자체 알림 접수·송신 대기열 → 기존 big-endian v1 인코더 → UDP → C3 접수·ACK 순서다.

두 결과가 함께 새 이벤트로 접수되면 FIRE_ALARM, DOORBELL 순서로 별도 순번을 부여하고 stop-and-wait로 전송한다. 재전송은 원래 `(session, seq, event_id)`를 유지한다. ACK 대기 중 새 감지를 덮어쓰지 않도록 큐가 필요하다.

현재 [`communication_tests/softap_udp/main/main.c`](../../communication_tests/softap_udp/main/main.c)는 **TEST 전용**이다. 아래 작업은 소리 이벤트 통합 시 양 보드에서 함께 수행해야 하며, 이번 모델 정리에서 기존 통신 코드는 변경하지 않았다.

| 위치 | 통합할 내용 |
|---|---|
| S3 TEST 생성 부분 | 모델 판정과 ID 매핑·대기열을 이벤트 생성 입력으로 연결 |
| S3 ACK 검사 | 기다리는 session·seq·event_id 세 값을 모두 비교 |
| C3의 `value == 1` 검사 | 소리 ID 허용 목록과 실제 알림 처리기 접수에 연결 |
| C3 ACK | 처리기 접수 성공 뒤 원래 세 값으로 응답, 중복 이벤트는 알림 반복 없이 ACK |

현재 예제에 새 ID만 보내면 수신되지 않는다. ACK는 소프트웨어 접수 확인이며 LED·진동 동작 완료 확인은 아니다. 상세 길이·세션·중복·절전·오류 벡터는 기존 프로토콜 문서를 따른다.

## 6. 저장소를 받은 뒤 모델 무결성 확인

호스트: 팀원의 개발 PC. 작업 디렉터리: clone한 `hearing-alert-system` 루트. 기존 `python3`로 파일 해시만 확인하며 모델·GPU·보드는 실행하지 않는다. foreground 명령으로, 종료 코드 0과 마지막 pass 문자열을 확인하면 터미널을 유지할 필요가 없다.

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('ml/models/pursuit_20261114')
manifest = json.loads((root / 'release_manifest.json').read_text())
for name, digest in manifest['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
print('MODEL_RELEASE_HASHES_PASS')
PY
```

## 7. 확인된 성능과 남은 시험

최신 후보의 고정 공개 시험은 559개 파일이다. 화재경보 **8/41**, 초인종 **1/6**, 기타 오감지 **0/512**였다. 학습 국내 원음 진단에서는 화재 **0/17**, 초인종 **16/26**으로 국내 경보 감지에도 한계가 있다. 독립된 실제 마이크 사건 시험이나 월간 오알림 측정 결과는 아니다. [진행 상황과 비교표](DEVELOPMENT_STATUS_2026-11-14.md)를 함께 확인한다.

다음 통합 시험은 마이크 PCM 규약, 전처리·int8 입력 일치, TFLM tensor 할당·추론 성공, S3/C3 소리 ID·큐·ACK, 실제 LED·진동, 절전 중 지연·전체 RAM을 각각 기록해야 한다. 현재 이 소리 감지 경로의 ESP-IDF 전체 빌드·플래시·실보드 추론·물리 알림 시험은 완료하지 않았다. 기존 TEST 통신의 실물 검증 기록은 그대로 보존한다.
