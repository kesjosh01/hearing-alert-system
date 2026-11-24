# 통신 실험

- [softap_udp](softap_udp/README.md): T7 S3 v1.1 SoftAP 송신부와 STA 수신부의 ESP-IDF 기초 실험. 수신부는 2026-11-24부터 XIAO ESP32-C6(이전 C3 SuperMini, C3 빌드도 유지). UDP + ACK·재전송 코드를 작성했으며 C3 실제 보드의 이벤트·ACK 및 자동 Light-sleep을 검증했습니다(C6는 미검증). [결과](../docs/test_results/low_power_2026-10-30.md)를 참고합니다.
- `espnow/`: 이전 검토 후보의 참고 영역. 현재 개발 대상에서 제외합니다.

[프로토콜](../docs/communication/README.md)과 [검증 기록](../docs/test_results/softap_udp.md)을 함께 관리합니다.
