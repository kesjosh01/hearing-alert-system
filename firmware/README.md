# 펌웨어

ESP-IDF v5.5.5 기반입니다. 2026-11-15부터 두 보드의 제품용 통신 펌웨어를 아래 디렉터리에서 각각 독립 빌드합니다.

- [`transmitter_s3/`](transmitter_s3/README.md): ESP32-S3 SoftAP·이벤트 송신부. ML 팀이 호출할 `ha_publish_event()` API, 대기열, 재전송, ACK 검증을 포함합니다. 오디오·추론은 아직 통합하지 않았습니다.
- [`receiver_c6/`](receiver_c6/README.md): Seeed Studio XIAO ESP32-C6 STA 수신부(2026-11-17 C3 SuperMini에서 변경). 검증·중복 억제·UI 접수 큐, MIN_MODEM과 자동 Light-sleep(목표 수신 간격 307.2ms), GC9A01 원형 화면·진동모터·버튼 UI(데모 디자인), **알림 화면을 LCD로 보낸 뒤 ACK**, 화면 복구(재초기화·재부팅)를 포함합니다.
- [`common/ha_comm/`](common/ha_comm/README.md): 양 보드 공통 프로토콜·큐·중복 억제·시각 확장 코어와 호스트 단위 시험입니다.

```bash
bash firmware/common/ha_comm/test/host/run_tests.sh
bash firmware/receiver_c6/components/ha_ui_core/test/host/run_tests.sh
source ~/esp/esp-idf/export.sh
idf.py -C firmware/transmitter_s3 -B firmware/transmitter_s3/build build
idf.py -C firmware/receiver_c6 -B firmware/receiver_c6/build build
```

과거 TEST 실험 코드는 [SoftAP UDP 실험](../communication_tests/softap_udp/README.md)에 보존합니다. 제품 펌웨어는 이 코드를 include하지 않습니다.
