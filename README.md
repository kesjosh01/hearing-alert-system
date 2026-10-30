# 난청 노인용 청각 알림 시스템

## 1. 프로젝트 소개

난청 노인을 위해 초인종, 화재경보 등 중요한 소리를 감지하고 시각 및 촉각 알림으로 전달하는 대학교 캡스톤 프로젝트입니다. 소리를 듣기 어려워 생활 속 주요 신호를 놓치는 문제를 해결하는 것을 목표로 합니다.

저비용 하드웨어, 로컬 동작, 오픈소스 공개, 사용자 친화성을 지향합니다. 현재는 SoftAP + UDP 통신 실험 단계입니다. 실제 S3·C3에서 이벤트·ACK와 C3 자동 Light-sleep을 구현·검증했으며, 소리 감지·분류·알림 기능은 구현 전입니다. 실제 화재경보 등의 감지 성능과 신뢰성은 아직 검증되지 않았습니다. 오픈소스 라이선스 종류는 TBD입니다.

## 2. 시스템 개요

- 송신부 (Transmitter / Main module): ESP32-S3에서 마이크 입력, 소리 분석 및 이벤트 분류를 수행할 예정입니다.
- 수신부 (Receiver / Child module): ESP32-C3에서 이벤트 정보를 수신하고 시각·촉각 알림을 제공할 예정입니다.
- 통신: Wi-Fi SoftAP 기반으로 확정했습니다. 기초 실험은 UDP + ACK·재전송 방식이며, 실제 보드에서 C3 절전·이벤트 수신·ACK를 검증합니다.
- 머신러닝: Ubuntu PC에서 학습·평가·변환을 수행하고, 경량 모델을 ESP32-S3에 배포하는 것을 목표로 합니다.

계획된 전체 데이터 흐름은 다음과 같습니다. 통신 테스트 코드 외의 기능은 구현 예정입니다.

```text
소리 입력 (예정)
→ ESP32-S3 오디오 입력 (예정)
→ 소리 분석 (예정)
→ 이벤트 분류 (예정)
→ Wi-Fi SoftAP + UDP 무선 통신 (실물 이벤트·ACK·C3 절전 검증)
→ ESP32-C3 TEST 이벤트 수신 (구현·검증, 실제 소리 이벤트는 향후 통합)
→ LED / 진동 / 화면 알림 (예정)
```

원시 오디오 데이터 (raw data) 전체를 지속적으로 전송하기보다 송신부에서 소리를 분석하고 이벤트 결과만 수신부로 전달하는 구조를 우선 고려합니다. 기초 테스트의 이벤트 형식은 [통신 프로토콜 v1](docs/communication/README.md)에 정의했습니다.

## 3. 하드웨어 계획

### Transmitter

- MCU: ESP32-S3 사용 계획
- 마이크: TBD
- 오디오 입력·전처리 및 머신러닝 추론 (ML inference): 구현 예정

### Receiver

- MCU: ESP32-C3 사용 계획
- LED: 연결 및 알림 구현 예정
- 진동모터: 부품 선정 및 연결 예정
- 디스플레이 및 기타 부품: TBD

## 4. 통신 방식

LILYGO T7 S3 v1.1을 SoftAP·송신부로, ESP32-C3 SuperMini를 STA·수신부로 구성했습니다. UDP로 TEST 이벤트를 보내고 ACK, 300ms 간격 최대 4회 시도, 중복 처리 방지, 자동 재접속을 적용합니다.

[기초 통신 코드 및 빌드·테스트 절차](communication_tests/softap_udp/README.md)와 [프로토콜 v1](docs/communication/README.md)을 준비했습니다. 실제 보드의 연결·이벤트·ACK·C3 절전 검증 결과는 [2026-10-30 실험](docs/test_results/low_power_2026-10-30.md)에 기록합니다. 상세 상태는 [검증 기록](docs/test_results/softap_udp.md)에 기록합니다.

`communication_tests/espnow/`는 이전 검토 후보의 참고 영역으로 보존합니다. ESP-NOW는 현재 개발 대상에서 제외합니다.

## 5. 머신러닝 계획

향후 음향 데이터 수집 → 전처리 → 학습 → 평가 → 경량화/양자화 → ESP32-S3 배포 순서로 개발할 예정입니다. 학습, 전처리, 평가 및 모델 변환은 초기에는 Ubuntu PC에서 수행할 계획입니다.

YAMNet 기반 접근 또는 YAMNet에서 아이디어를 가져온 경량 음향 이벤트 분류 모델을 검토할 예정입니다. YAMNet 적용 여부, 모델 구조, 데이터셋 및 도구는 TBD이며, 현재 모델 다운로드나 머신러닝 코드 구현은 수행하지 않았습니다. 최종 목표는 경량화된 모델을 ESP32-S3에서 추론 가능하도록 만드는 것입니다.

## 6. 저장소 구조

```text
.
├── firmware/
│   ├── transmitter_s3/
│   └── receiver_c3/
├── communication_tests/
│   ├── espnow/
│   └── softap_udp/
├── ml/
│   ├── datasets/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   ├── export/
│   └── models/
├── tools/
│   └── audio_capture/
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
| `firmware/` | ESP32-S3 및 ESP32-C3 펌웨어 관리 예정 |
| `communication_tests/` | 확정된 Wi-Fi SoftAP 통신 테스트 관리 예정; ESP-NOW 영역은 이전 검토 후보로 보존 |
| `ml/` | 음향 데이터, 전처리, 학습, 평가 및 모델 변환 관리 예정 |
| `tools/` | 오디오 수집 등 개발 보조 도구 관리 예정 |
| `docs/` | 시스템 설계, 통신 분석, ML 조사 및 실험 결과 기록 예정 |

## 7. 개발 환경

Ubuntu의 ESP-IDF v5.5.5를 사용합니다. `communication_tests/softap_udp/`는 ESP32-S3와 ESP32-C3를 각각 빌드하는 ESP-IDF 프로젝트입니다. Python 기반 ML 환경은 향후 구성 예정입니다. ROS 2 또는 colcon 프로젝트는 아닙니다.

## 8. 개발 워크플로

코드와 개발 환경을 준비한 뒤 다음 흐름으로 작업할 예정입니다.

```text
코드 수정 → 빌드 → 보드 플래시 → 테스트
→ git status / git diff → git add → git commit → git push
```

- 빌드 (build): 소스코드를 컴파일하여 실행 가능한 결과물을 생성하고 컴파일·링크 오류를 확인하는 과정입니다. 실제 보드에서의 실행 가능 여부와 동작은 플래시 후 테스트로 확인합니다.
- 커밋 (commit): 현재 변경사항을 Git 이력으로 기록하는 과정입니다.
- 푸시 (push): 커밋을 GitHub 원격 저장소에 업로드하는 과정입니다.

현재는 기초 통신 및 C3 절전 코드를 빌드·플래시해 실험합니다. 최종 펌웨어 통합은 [시스템 구조](docs/architecture/README.md)에 정리했습니다. 커밋·푸시는 별도 요청 시 수행합니다.

## 9. 현재 진행 상태

현재는 SoftAP + UDP 기초 통신 및 C3 절전 실물 검증 단계입니다.

- [x] 로컬 Git 저장소 및 GitHub 원격 주소 설정 확인
- [x] 기본 디렉터리 구조 및 한국어 안내 문서 생성
- [ ] GitHub 원격 저장소 접근 및 공개 상태 확인
- [ ] 오픈소스 라이선스 선정
- [x] ESP-IDF v5.5.5 환경 확인
- [ ] ESP32-S3 Hello World
- [ ] ESP32-C3 Hello World
- [x] 무선 통신 방식 선정: Wi-Fi SoftAP
- [x] 기초 이벤트 전송 프로토콜 정의: UDP + ACK·재전송
- [x] Wi-Fi SoftAP 연결 및 이벤트 전송 테스트
- [x] C3 Modem-sleep·자동 Light-sleep 및 수신·ACK 검증
- [ ] 배터리 전원 소비전류·사용 시간 측정
- [ ] 오디오 입력
- [ ] 음향 데이터 수집 및 전처리
- [ ] ML 모델 선정·학습·평가
- [ ] 경량화 및 양자화
- [ ] ESP32-S3 추론 (inference)
- [ ] 시각 및 촉각 알림 구현
