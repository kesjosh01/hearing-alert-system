# 펌웨어

ESP-IDF v5.5.5 기반입니다. 2026-11-15부터 두 보드의 제품용 통신 펌웨어를 아래 디렉터리에서 각각 독립 빌드합니다.

- [`transmitter_s3/`](transmitter_s3/README.md): ESP32-S3 SoftAP·이벤트 송신부. ML 팀이 호출할 `ha_publish_event()` API, 대기열, 재전송, ACK 검증을 포함합니다. 오디오·추론은 아직 통합하지 않았습니다.
- [`receiver_c3/`](receiver_c3/README.md): ESP32-C3 STA 수신부. 검증·중복 억제·UI 접수 큐·ACK, MIN_MODEM과 자동 Light-sleep(목표 수신 간격 307.2ms)을 포함합니다. 표시·진동 드라이버는 아직 없습니다.
- [`common/ha_comm/`](common/ha_comm/README.md): 양 보드 공통 프로토콜·큐·중복 억제·시각 확장 코어와 호스트 단위 시험입니다.

```bash
bash firmware/common/ha_comm/test/host/run_tests.sh
source ~/esp/esp-idf/export.sh
idf.py -C firmware/transmitter_s3 -B firmware/transmitter_s3/build build
idf.py -C firmware/receiver_c3 -B firmware/receiver_c3/build build
```

과거 TEST 실험 코드는 [SoftAP UDP 실험](../communication_tests/softap_udp/README.md)에 보존합니다. 제품 펌웨어는 이 코드를 include하지 않습니다.
