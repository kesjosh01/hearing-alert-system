# 펌웨어

ESP-IDF v5.5.5 기반입니다. 2026-11-15부터 두 보드의 제품용 통신 펌웨어를 아래 디렉터리에서 각각 독립 빌드합니다.

- [`transmitter_s3/`](transmitter_s3/README.md): ESP32-S3 SoftAP·이벤트 송신부. ML 팀이 호출할 `ha_publish_event()` API, 대기열, 재전송, ACK 검증을 포함합니다. INMP441 수집과 native40 int8 추론을 연결해 화재·초인종 이벤트를 생성합니다.
- [`receiver_c6/`](receiver_c6/README.md): Seeed Studio XIAO ESP32-C6 STA 수신부(2026-11-17 C3 SuperMini에서 변경). 검증·중복 억제·UI 접수 큐, MIN_MODEM과 자동 Light-sleep(목표 수신 간격 307.2ms), GC9A01 원형 화면·진동모터·버튼 UI(데모 디자인), **알림 화면을 LCD로 보낸 뒤 ACK**, 화면 복구(재초기화·재부팅)를 포함합니다.
- [`common/ha_comm/`](common/ha_comm/README.md): 양 보드 공통 프로토콜·큐·중복 억제·시각 확장 코어와 호스트 단위 시험입니다.

## 제품 보드 플래시

S3와 C6를 데이터 USB 케이블로 연결하고 저장소 루트에서 실행합니다.

```bash
./tools/flash.sh
```

VS Code에서는 **터미널 → 작업 실행 → 제품 펌웨어 플래시**에서 `both`, `s3`, `c6` 중 하나를 선택합니다. 터미널에서도 `./tools/flash.sh s3` 또는 `./tools/flash.sh c6`로 한 대씩 올릴 수 있습니다.

도구는 ESP-IDF 환경을 불러오고 USB로 실제 칩을 확인합니다. 선택한 제품 펌웨어를 모두 빌드한 뒤 부트로더·파티션 표·앱을 해당 보드에 기록합니다. 두 대를 선택했는데 하나만 연결되어 있으면 쓰기 전에 중단합니다. 같은 칩이 여러 대이거나 별도 USB-UART 어댑터를 쓰면 `./tools/flash.sh s3 /dev/ttyUSB0`처럼 포트를 지정합니다. 칩 식별 과정에서 검색 대상 보드가 재시작될 수 있으므로 열려 있는 시리얼 모니터는 먼저 종료합니다.

기본 SDK 경로는 `~/esp/esp-idf`입니다. 다른 위치에 설치했다면 `IDF_PATH`를 지정합니다. USB 없이 빌드만 확인하려면 `./tools/flash.sh --build-only`를 실행합니다. 결과는 `build/ml_s3`와 `build/ml_c6`에 생성됩니다.

## 플래시 후 실기기 확인

플래시가 끝나면 출력된 `--monitor` 명령 두 개를 각각 별도 터미널에서 실행합니다. 실제 화재 경보음과 초인종 소리를 차례로 재생해 C6의 화면과 진동을 확인합니다. S3의 `TX`, C6의 `EVENT displayed`, S3의 `ACK`에 같은 이벤트 ID와 순번이 찍히면 해당 이벤트의 무선 왕복을 확인한 것입니다. 모니터 종료는 `Ctrl+]`입니다.

## 호스트 시험

```bash
bash firmware/common/ha_comm/test/host/run_tests.sh
bash firmware/receiver_c6/components/ha_ui_core/test/host/run_tests.sh
```

과거 TEST 실험 코드는 [SoftAP UDP 실험](../communication_tests/softap_udp/README.md)에 보존합니다. 제품 펌웨어는 이 코드를 include하지 않습니다.
