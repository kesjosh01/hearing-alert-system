# ESP32-S3 Transmitter

## 1. 프로젝트 개요

ESP32-S3 기반 청각 알림 시스템의 송신부(Transmitter)를 개발하는 프로젝트입니다.

INMP441 디지털 마이크를 통해 주변 소리를 수집하고, 도어벨 및 화재경보기와 같은 특정 소리를 구분하여 청각 알림 시스템에 활용하는 것을 목표로 합니다.

현재까지 진행한 작업은 실제 장치에서 소리 수집 및 구분이 가능한지 확인하기 위한 기초 하드웨어 테스트와 사전 실험입니다.

## 2. 개발 환경

- **MCU:** LILYGO T7_S3 V1.1
- **개발 환경:** ESP-IDF
- **프로그래밍 언어:** C
- **IDE:** Visual Studio Code

## 3. 사용 부품

- LILYGO T7_S3 V1.1
- INMP441 디지털 MEMS 마이크
- WS2812B RGB LED
- SN74AHCT125N 레벨 시프터
- 5V DC 어댑터

## 4. GPIO 핀맵

| ESP32-S3 GPIO | 연결 부품 | 연결 핀 |
|---|---|---|
| GPIO4 | INMP441 | SCK |
| GPIO5 | INMP441 | WS |
| GPIO6 | INMP441 | SD |
| GPIO10 | SN74AHCT125N | Pin 2 (Data Input) |

## 5. 전원 구성

- INMP441: 3.3V 전원
- WS2812B: 외부 5V 전원
- SN74AHCT125N: 5V 전원
- 모든 부품은 공통 GND 사용

## 6. 현재까지 진행한 작업

- ESP32-S3 초기화 및 기본 동작 확인
- INMP441을 이용한 I2S 오디오 데이터 수집
- PCM 오디오 데이터 녹음 및 파형 분석
- 도어벨과 화재경보기 소리 구분 가능성 테스트
- DTW 기반 기준 파형 비교 실험
- WS2812B RGB LED 제어 테스트

※ DTW는 소리 구분 가능성을 확인하기 위해 사용한 실험적인 방법이며, 최종 소리 분류 알고리즘으로 확정된 것은 아닙니다.

## 7. 향후 개발 방향

- 실제 장치 환경에 적합한 소리 분류 방식 선정
- 소리 종류별 판별 성능 및 정확도 개선
- 소리 발생 초기 구간에서의 빠른 판정
- LED를 통한 소리 종류 표시
- SoftAP 기반 무선 통신 및 수신부 통합

## 8. 빌드 및 실행 방법

프로젝트 루트 디렉터리에서 실행합니다.

빌드:

```bash
idf.py build
```

펌웨어 업로드:

```bash
idf.py -p PORT flash
```

시리얼 모니터:

```bash
idf.py -p PORT monitor
```

`PORT`에는 실제 ESP32-S3의 포트 번호를 입력합니다. 예: `COM8`

## 9. 프로젝트 구조

```text
s3_hardware_control/
├── main/
│   ├── main.c
│   ├── dtw_classifier.c
│   ├── dtw_classifier.h
│   ├── dtw_reference.h
│   └── CMakeLists.txt
├── doorbell_reference.npy
├── fire_alarm_reference.npy
├── generate_dtw_reference.py
├── compare_dtw.py
├── compare_waveforms.py
├── check_reference.py
├── convert_to_header.py
├── receive_wav.py
├── CMakeLists.txt
├── sdkconfig
└── README.md
```

