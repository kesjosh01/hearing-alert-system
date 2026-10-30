# 펌웨어

- `transmitter_s3/`: ESP32-S3 송신부의 오디오 입력, 소리 분류 및 이벤트 전송 펌웨어를 관리할 예정입니다.
- `receiver_c3/`: ESP32-C3 수신부의 이벤트 수신 및 시각·촉각 알림 펌웨어를 관리할 예정입니다.

ESP-IDF 기반으로 개발합니다. 통합 펌웨어는 구현 전이며, 기초 통신 ESP-IDF 프로젝트는 [SoftAP UDP 실험](../communication_tests/softap_udp/README.md)에 있습니다.
