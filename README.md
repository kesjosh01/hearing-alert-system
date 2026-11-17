# 난청 노인용 청각 알림 시스템

## 1. 프로젝트 소개

난청 노인을 위해 초인종, 화재경보 등 중요한 소리를 감지하고 시각 및 촉각 알림으로 전달하는 대학교 캡스톤 프로젝트입니다. 소리를 듣기 어려워 생활 속 주요 신호를 놓치는 문제를 해결하는 것을 목표로 합니다.

저비용 하드웨어, 로컬 동작, 오픈소스 공개, 사용자 친화성을 지향합니다. 통신 방식은 Wi-Fi SoftAP + UDP로 확정했으며, 현재는 실제 보드에서 기초 통신과 수신부 절전을 검증한 단계입니다. S3·C3 간 이벤트 전송·ACK와 C3 자동 Light-sleep을 구현했고, 절전 확인 후 연속 100개 이벤트에서 ACK 100/100·재전송 0·실패 0을 기록했습니다. 이 결과는 TEST 이벤트를 이용한 해당 실험의 결과이며, 실제 마이크 입력에 따른 보드 소리 감지·분류·알림 통합은 미검증입니다. 실제 화재경보 등의 감지 성능과 신뢰성은 아직 검증되지 않았습니다. 2026-11-15에는 제품용 S3·C3 통신 펌웨어(이벤트 입력 API·대기열·ACK·중복 억제·300ms 목표 절전 설정)를 `firmware/`에 구현하고 빌드·자동 검증했습니다. 2026-11-17에는 수신부를 XIAO ESP32-C6으로 바꾸고 원형 LCD 화면·진동 알림 UI와 화면 복구를 구현했습니다. 실기기 시험은 남아 있습니다. 오픈소스 라이선스 종류는 TBD입니다.

## 2. 시스템 개요

- 송신부 (Transmitter / Main module): ESP32-S3에서 마이크 입력, 소리 분석 및 이벤트 분류를 수행할 예정입니다.
- 수신부 (Receiver / Child module): ESP32-C6(Seeed Studio XIAO ESP32-C6)에서 이벤트 정보를 수신하고 원형 LCD 화면·진동으로 알립니다(펌웨어 구현, 실기기 검증 전). 2026-11-17에 설계상의 이유로 ESP32-C3 SuperMini에서 변경했습니다.
- 통신: Wi-Fi SoftAP + UDP로 확정했습니다. 반복 송신 + ACK 방식이며, 기초 실험(C3)에서 절전·이벤트 수신·ACK를 실물 검증했습니다.
- 머신러닝: Ubuntu PC에서 학습·평가·변환을 수행하고, 경량 모델을 ESP32-S3에 배포하는 것을 목표로 합니다.

계획된 전체 데이터 흐름은 다음과 같습니다. PC 모델 개발과 S3용 추론 컴포넌트는 준비됐고 S3 보드의 오디오·추론 통합은 구현 예정입니다. 수신부 화면·진동 알림은 펌웨어로 구현했으며 실기기 검증 전입니다.

```text
소리 입력 (예정)
→ ESP32-S3 오디오 입력 (예정)
→ 소리 분석 (예정)
→ 이벤트 분류 (예정)
→ Wi-Fi SoftAP + UDP 무선 통신 (실물 이벤트·ACK·C3 절전 검증)
→ ESP32-C6 이벤트 수신·UI 큐 접수 (제품 펌웨어 구현·C6 빌드 확인, 실기기 미검증)
→ 원형 LCD 화면 + 진동 알림, 화면 표시 후 ACK (구현·PC 시험, 실기기 미검증)
```

시스템은 송신부에서 소리를 분석하고 이벤트 결과만 수신부로 전달합니다. 원시 오디오는 보내지 않습니다. 이벤트 형식은 [통신 프로토콜 v1](docs/communication/README.md)에 정의했습니다.

## 3. 하드웨어 계획

### Transmitter

- MCU: LILYGO T7 S3 v1.1 (ESP32-S3), 통신 실험에 사용 중
- 마이크: TBD
- 오디오 입력·전처리 및 머신러닝 추론 (ML inference): 구현 예정

### Receiver

- MCU: Seeed Studio XIAO ESP32-C6 (2026-11-17 변경). 이전 통신·절전 실험은 ESP32-C3 SuperMini로 진행했으며, C6 실기기 검증은 아직 하지 않았습니다
- 진동모터: 코인 진동모터 DVM1027, NPN 구동, D2(GPIO2) — 펌웨어 구현
- 디스플레이: GC9A01 1.28" 원형 SPI LCD, D3·D4·D5·D7·D8·D10 — 펌웨어 구현(데모 디자인)
- 버튼 D1(GPIO1) 구현, 배터리 분압 ADC D0(GPIO0)은 저항값 확인 후 사용. 핀 배치는 [수신부 안내](firmware/receiver_c6/README.md) 참고

## 4. 통신 방식

외부 공유기 없이 S3가 Wi-Fi SoftAP를 열고 수신부(XIAO ESP32-C6)가 접속합니다. 2.4GHz·채널 6·WPA2-PSK, UDP 포트 4210, 수신부 1대입니다. 이벤트는 16바이트 v1 프레임 하나로 보냅니다.

- **S3:** 소리 이벤트를 대기열(8개)에 쌓고, ACK가 올 때까지 같은 내용으로 반복 송신합니다(시도당 700ms, 4회 묶음 뒤 2~16초 간격). 화재는 10분, 초인종은 2분이 지나면 포기합니다.
- **수신부:** 검증·중복 억제 뒤 화면·진동으로 알리고, 알림 화면을 LCD로 보낸 뒤 ACK합니다. 화면이 안 뜨면 LCD 재초기화 → 재부팅 순으로 복구합니다.
- **절전:** 수신부는 Wi-Fi 연결을 유지한 채 Modem-sleep(MIN_MODEM)과 자동 Light-sleep을 쓰며, 명목 307.2ms(DTIM 3)마다 깨어 수신합니다. Deep-sleep은 쓰지 않습니다. 10초마다 HELLO·절전 상태를 보고하고, S3는 35초 넘게 HELLO가 없으면 응답 없음으로 표시하지만 재전송은 계속합니다.

규격은 [통신 프로토콜 v1](docs/communication/README.md), 사용법은 [펌웨어 안내](firmware/README.md), 검증은 [시험 기록](docs/test_results/README.md)에 있습니다. 제품 펌웨어의 실기기 무선 시험은 아직 하지 않았습니다.

### 기초 통신 실험 결과 (TEST 펌웨어, C3 SuperMini, 2026-10-30)

제품 펌웨어 이전의 [SoftAP UDP 실험](communication_tests/softap_udp/README.md)(3초마다 TEST 이벤트, DTIM 1) 결과입니다. 원본 로그와 분석 JSON은 [실험 기록](docs/test_results/low_power_2026-10-30.md#근거)에 연결되어 있습니다.

| 항목 | 결과 |
| --- | --- |
| C3 실제 Light-sleep | SDK 절전 성공 횟수 111 → 3519 증가 |
| 절전 확인 후 연속 100개 TEST 이벤트 | ACK 100/100, 재전송 0, FAILED 0 |
| ACK 왕복 시간 | 최소 9ms / 평균 55.79ms / 최대 130ms |
| S3 재시작 후 C3 자동 재연결 | S3 부팅 후 약 5.5초에 재등록 및 첫 ACK |
| C3 자체 재시작 후 복구 | 미확인 |
| 소비전류·배터리 사용 시간 | 미측정 |

이 실험의 ACK는 C3 소프트웨어의 이벤트 수신 처리 확인이었습니다. ACK 유실 주입·중복 억제·거리 시험은 수행하지 않았습니다. 2026-11-11 약 4m 초기 시험은 [통신 시험 로그](docs/communication/261111_통신원거리테스트v1.md)에 있습니다.

[통신 모니터·로그 분석 도구](tools/communication/README.md)로 두 보드의 로그와 무선 상태 보고를 확인합니다. 모니터 로그는 기본적으로 `/tmp/hearing-alert-monitor.*`에 저장하므로, 보관할 원본은 `docs/test_results/evidence/`에 별도로 복사하고 실험 조건과 결과를 문서로 남깁니다.

`communication_tests/espnow/`는 이전 검토 후보의 참고 영역으로 보존합니다. ESP-NOW는 현재 개발 대상에서 제외합니다.

## 5. 머신러닝 개발 현황

2026-11-14까지 YAMNet 기준 학습·고정 평가, 경량 DS-CNN baseline, YAMNet distillation, 공개 정답 확장 및 후속 모델 비교·int8 변환을 완료했습니다. 후속 연구의 optimizer 학습 33회와 BN 통계 보정 결과를 [현재 진행 상황](docs/ml/DEVELOPMENT_STATUS_2026-11-14.md)에 정리했습니다. 저장된 YAMNet·DS-CNN 계열 체크포인트 127개는 smoke/integration 점검용 실행을 구분해 [색인과 원본 파일](ml/models/checkpoints_20261114/README.md)로 보관합니다.

최신 전체 후보는 [native40_r06_v1](ml/models/pursuit_20261114/README.md)입니다. 실제 학습한 88,280-byte int8 모델과 C++ 헤더, 전처리·출력 계약·SHA-256을 제공합니다. 기존 16 kHz mono PCM16 → 1초 98×40 log-Mel 입력과 맞으며, 통신 담당자는 **[모델 연결 안내](docs/ml/MODEL_HANDOFF.md)**에서 파일 위치·시간 판정·기존 UDP 이벤트 ID 매핑·양 보드의 추가 통합 지점을 확인할 수 있습니다.

최신 고정 공개 시험에서 이 후보는 화재경보 8/41개·초인종 1/6개 파일을 감지했고 기타 오감지는 0/512개였습니다. 국내 원음 진단의 화재경보 감지도 부족해 실제 제품 신뢰도를 달성한 모델로 표시하지 않습니다. 학습 국내 원음 43/43 감지 후보, 증류·비증류·64-bin 비교와 이전 네 모델 묶음도 보존했습니다. 평가 자료·단위가 다른 수치를 직접 합쳐 순위를 정하지 않습니다.

[DS-CNN 학습 코드](ml/training/dscnn/README.md)와 [portable C++ 컴포넌트](firmware/transmitter_s3/dscnn/README.md)는 준비돼 있습니다. 실제 마이크·최종 ESP-IDF 빌드·실보드 추론·전체 RAM·지연·LED/진동 통합·장기 오알림은 별도 검증 대상입니다. 기존 통신 방식·프로토콜·실물 시험 기록은 유지합니다.

## 6. 저장소 구조

```text
.
├── firmware/
│   ├── common/ha_comm/    # S3·수신부 공통 통신 코어
│   ├── transmitter_s3/
│   └── receiver_c6/       # XIAO ESP32-C6 수신부
├── communication_tests/
│   ├── espnow/
│   └── softap_udp/
├── ml/
│   ├── datasets/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   ├── export/
│   └── models/            # 학습 모델·int8 후보·체크포인트
├── tools/
│   ├── audio_capture/
│   └── communication/
├── docs/
│   ├── architecture/
│   ├── communication/
│   ├── ml/
│   └── test_results/
├── .gitignore
├── README.md
└── LICENSE
```

각 디렉터리에는 역할을 안내하는 한국어 `README.md`가 있습니다.

| 디렉터리 | 역할 |
| --- | --- |
| `firmware/` | ESP32-S3·C6 제품 펌웨어와 공통 통신 코어(`common/ha_comm`). 수신부 화면·진동 UI 포함, S3 오디오·추론은 통합 예정 |
| `communication_tests/` | 확정된 Wi-Fi SoftAP 테스트 코드·빌드 설정 관리; ESP-NOW 영역은 이전 검토 후보로 보존 |
| `ml/` | 음향 데이터·YAMNet 기준 모델·DS-CNN 전처리·학습·평가·int8 변환 관리 |
| `tools/` | 통신 모니터·로그 분석 도구 구현, 오디오 수집 도구는 예정 |
| `docs/` | 시스템 구조·통신 프로토콜·실험 기록·모델 개발 현황·통신 연결 안내 |

## 7. 개발 환경

Ubuntu의 ESP-IDF v5.5.5를 사용합니다. `firmware/transmitter_s3/`와 `firmware/receiver_c6/`는 제품용 ESP-IDF 프로젝트이고, `communication_tests/softap_udp/`는 ESP32-S3와 ESP32-C3를 각각 빌드하는 TEST 실험 프로젝트(이전 수신부 기준)입니다. Python 기반 ML 작업은 기존 YAMNet 및 DS-CNN 전용 가상환경에서 수행합니다. ROS 2 또는 colcon 프로젝트는 아닙니다.

## 8. 개발 워크플로

현재 통신 프로젝트는 다음 흐름으로 개발·검증합니다.

```text
코드 수정 → 빌드 → 보드 플래시 → 실물 테스트 → 로그 분석·결과 기록
→ git status / git diff → git add → git commit → git push
```

- 빌드 (build): 소스코드를 컴파일하여 실행 가능한 결과물을 생성하고 컴파일·링크 오류를 확인하는 과정입니다. 실제 보드에서의 실행 가능 여부와 동작은 플래시 후 테스트로 확인합니다.
- 커밋 (commit): 현재 변경사항을 Git 이력으로 기록하는 과정입니다.
- 푸시 (push): 커밋을 GitHub 원격 저장소에 업로드하는 과정입니다.

제품 펌웨어(`firmware/`)를 빌드·플래시해 시험합니다. 전체 구조는 [시스템 구조](docs/architecture/README.md)에 정리했습니다. 커밋·푸시는 별도 요청 시 수행합니다.

## 9. 현재 진행 상태

Wi-Fi SoftAP + UDP 통신과 절전의 기초 실물 검증(C3)을 마쳤고, 제품 통신 펌웨어와 수신부(C6) 화면·진동 UI를 구현했습니다. S3 소리 입력·분류 통합, 제품 펌웨어 실기기 시험과 전력 측정이 남아 있습니다.

- [x] 로컬 Git 저장소 및 GitHub 원격 주소 설정 확인
- [x] 기본 디렉터리 구조 및 한국어 안내 문서 생성
- [x] GitHub 원격 저장소 접근 및 공개 상태 확인
- [ ] 오픈소스 라이선스 선정
- [x] ESP-IDF v5.5.5 환경 확인
- [x] 무선 통신 방식 선정: Wi-Fi SoftAP
- [x] 기초 이벤트 전송 프로토콜 정의: UDP + ACK·재전송
- [x] Wi-Fi SoftAP 연결 및 이벤트 전송 테스트
- [x] C3 Modem-sleep·자동 Light-sleep 및 수신·ACK 검증
- [x] 절전 확인 후 연속 100개 TEST 이벤트 ACK 확인 및 결과 기록
- [x] S3 재시작 후 C3 자동 재연결 검증
- [x] 통신 모니터·로그 분석 도구 및 원본 로그·분석 결과 정리
- [ ] 수신부 자체 재시작 후 복구 검증
- [ ] ACK 유실 주입·중복 억제·거리 및 장애물 시험
- [ ] 배터리 전원 소비전류·사용 시간 측정
- [x] 제품 통신 펌웨어: 이벤트 입력 API·큐·ACK·중복 억제·UI 접수 계약 구현 및 호스트 시험·양 보드 빌드
- [ ] 제품 통신 펌웨어 실기기 시험 (DTIM 3 절전·지연·전류 포함)
- [ ] 오디오 입력
- [x] 개발용 음향 데이터 수집 및 전처리
- [x] PC에서 ML 모델 선정·학습·평가 (감지 성능 개선 필요)
- [x] 경량 DS-CNN 및 전체 int8 양자화
- [ ] ESP32-S3 추론 (inference)
- [x] 수신부 XIAO ESP32-C6 전환, 화면·진동 알림 UI·화면 전송 후 ACK·화면 복구 구현 (PC 시험·빌드, C6 단독 부팅)
- [ ] 수신부 화면·진동 실기기 시험 (LCD·모터·버튼 연결, S3와 연동)
