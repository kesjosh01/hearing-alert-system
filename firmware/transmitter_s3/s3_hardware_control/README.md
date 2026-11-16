# Transmitter 하드웨어 제어 테스트

## 1. 프로젝트 개요

ESP32-S3 기반 송신부(Transmitter)의 하드웨어 기본 동작을 테스트하기 위한 프로젝트입니다.

마이크 입력과 LED 제어 기능을 개별적으로 확인하고, 이후 소리 분류 및 무선 통신 기능을 통합하기 위한 기초 환경을 구성합니다.

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

- INMP441: 3.3V 전원 사용
- WS2812B: 외부 5V 전원 사용
- SN74AHCT125N: 5V 전원 사용
- 모든 부품은 공통 GND를 사용합니다.

## 6. 구현 기능

- INMP441 마이크의 I2S 오디오 데이터 입력
- 시리얼 모니터를 통한 오디오 신호 크기 확인
- WS2812B RGB LED 색상 제어
- LED 밝기 제어
- SN74AHCT125N을 이용한 데이터 신호 레벨 변환

## 7. 현재 개발 상태

| 기능 | 진행 상태 |
|---|---|
| ESP32-S3 초기화 | 완료 |
| INMP441 마이크 입력 테스트 | 완료 |
| 오디오 신호 크기 확인 | 완료 |
| WS2812B LED 제어 | 테스트 진행 |
| LED 밝기 제어 | 테스트 진행 |
| DS-CNN 소리 분류 통합 | 미진행 |
| SoftAP 무선 통신 통합 | 미진행 |

## 8. 빌드 및 실행 방법

프로젝트 루트 디렉터리에서 다음 명령어를 실행합니다.

```bash
idf.py build
```

펌웨어 업로드:

```bash
idf.py -p PORT flash
```

시리얼 모니터 실행:

```bash
idf.py -p PORT monitor
```

`PORT`에는 실제 ESP32-S3의 포트 번호를 입력합니다. 예: `COM8`

## 9. 프로젝트 구조

```text
s3_hardware_control/
├── main/
│   ├── main.c
│   └── CMakeLists.txt
├── CMakeLists.txt
├── sdkconfig
├── .gitignore
└── README.md
```

## 10. 참고 사항

본 프로젝트는 송신부 하드웨어 제어 기능을 독립적으로 테스트하기 위한 프로젝트입니다.

소리 분류 기능(DS-CNN)과 SoftAP 기반 무선 통신 기능은 추후 통합할 예정입니다.