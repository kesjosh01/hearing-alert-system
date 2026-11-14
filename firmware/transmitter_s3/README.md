# LILYGO T7 S3 v1.1 송신부

최종 오디오·알림 통합 펌웨어는 구현 전입니다. 현재 기초 통신 코드는 [SoftAP UDP 실험](../../communication_tests/softap_udp/README.md)에 공통 ESP-IDF 프로젝트로 관리합니다. 칩 타깃에 따라 S3는 AP·송신, C3는 STA·수신 역할을 선택합니다.

[DS-CNN 컴포넌트](dscnn/README.md)에 1초 PCM용 전처리와 TFLM 실행 연결 코드를 개발했습니다. portable 전처리의 host 검증만 완료했으며 ESP-IDF 빌드·실제 마이크·보드 추론은 아직 수행하지 않았습니다.

최신 학습 후보의 파일 위치·입력 계약·시간 판정·통신 이벤트 ID 연결은 [모델 연결 안내](../../docs/ml/MODEL_HANDOFF.md)를 따릅니다. [`native40_r06_v1`](../../ml/models/pursuit_20261114/README.md)는 기존 98×40 frontend와 맞는 확률 출력 모델입니다. 기존 TEST 통신 예제는 유지하며 소리 이벤트 큐·양 보드 ID 지원·실제 알림은 통합해야 합니다.
