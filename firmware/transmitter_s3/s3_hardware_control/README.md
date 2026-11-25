# T7 S3 하드웨어 제어 펌웨어

LILYGO T7 S3 v1.2에서 INMP441 마이크의 오디오를 수집하고, `native40_r06_v1` 모델로 소리를 분류해 WS2812B LED 색상으로 표시하는 ESP-IDF 펌웨어입니다. 팀 프로젝트의 기존 SoftAP/UDP 송신기도 시작하므로 웨어러블 수신기로 이벤트를 전달합니다.

## 하드웨어 연결

| T7 S3 v1.2 | 연결 대상 |
| --- | --- |
| GPIO4 | INMP441 SCK/BCLK |
| GPIO5 | INMP441 WS/LRCLK |
| GPIO6 | INMP441 SD/DOUT |
| 3V3 | INMP441 VDD |
| GND | INMP441 GND 및 LED 전원 공통 접지 |
| GPIO10 | SN74AHCT125N 2번 핀(1A) |

SN74AHCT125N 3번 핀(1Y)을 WS2812B의 DIN에 연결하고, 1번 핀(1OE)은 GND에 연결해 출력을 활성화합니다. 레벨시프터 VCC에는 5V를 공급합니다. LED 스트립도 별도의 적절한 5V 전원으로 구동하고, 그라운드는 T7 S3와 공통으로 연결합니다. LED 스트립에 T7의 3.3V 출력을 전원으로 사용하지 마세요. 레벨시프터 전원 핀 가까이에 100nF 바이패스 커패시터를 두는 것을 권장합니다.

INMP441의 L/R 핀이 GND이면 왼쪽 슬롯, 3.3V이면 오른쪽 슬롯을 선택합니다. 펌웨어 기본값은 왼쪽입니다. L/R을 3.3V에 연결한 경우 `idf.py menuconfig`의 **S3 hardware control** 메뉴에서 오른쪽 슬롯 옵션을 켜세요.

## LED 표시 동작

팀 분류기의 기존 2-of-3 창 판정과 모델 임계값을 사용합니다.

| 판정 결과 | LED 색상 |
| --- | --- |
| 화재경보 | 빨강 |
| 초인종 | 파랑 |
| 두 대상 소리 미감지 | 초록 |

화재경보와 초인종이 동시에 판정되면 화재경보 표시가 우선합니다. 오디오 입력이 끊기면 판정 이력을 초기화하고 LED를 초록색으로 되돌립니다. 기본 LED 데이터 핀은 GPIO10, 밝기는 255 기준 32입니다. 두 값은 `menuconfig`의 **S3 hardware control** 메뉴에서 변경할 수 있습니다.

## 빌드 및 플래시

이 프로젝트는 ESP-IDF 6.1 환경에서 빌드했습니다. ESP-IDF 환경을 활성화한 뒤 이 폴더에서 실행합니다.

```powershell
idf.py set-target esp32s3
idf.py build
idf.py -p COM포트 flash monitor
```

예를 들어 포트가 COM8이면 마지막 명령은 `idf.py -p COM8 flash monitor`입니다. 플래시 전에 장치 관리자에서 실제 포트를 확인하세요. 빌드 디렉터리와 `sdkconfig`는 이 프로젝트 안에 생성되어 팀 송신기 프로젝트 설정과 분리됩니다.

### 국내 음원 과적합 모델 선택

이 폴더에서 기본 빌드와 별도 경로를 지정합니다. 아래 예시의 `COM8`은 실제 S3 포트로 바꿉니다. C6에는 기존 제품용 수신 펌웨어를 사용합니다.

```powershell
idf.py -B build-domestic -D HA_S3_MODEL=domestic_overfit_20261119 build
idf.py -B build-domestic -p COM8 flash monitor
```

부팅 로그에 `domestic_overfit_20261119 ready`가 표시됩니다. 기존 모델은 `idf.py -B build -D HA_S3_MODEL=native40_r06_v1 build`로 선택합니다.

## 참고 사항

- 모델과 분류 로직은 현재 팀 송신기 프로젝트에서 사용하는 `native40_r06_v1`을 기반으로 합니다.
- 저장소의 기존 모델 평가에서 화재경보 및 초인종 인식 성능에 한계가 보고되어 있습니다. 실제 사용 전 대표적인 녹음으로 감지 성능을 확인해야 합니다.
- 빌드 성공은 실물 마이크 입력, LED 색상, 무선 수신 동작을 검증한 것이 아닙니다. 보드에 연결한 뒤 각 기능을 별도로 시험해야 합니다.
