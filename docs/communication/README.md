# SoftAP UDP 프로토콜 v1 (기초 실험)

## 구성과 범위

| 항목 | 설정 |
| --- | --- |
| AP / 송신부 | LILYGO T7 S3 v1.1, ESP32-S3 |
| STA / 수신부 | ESP32-C3 SuperMini, DHCP 주소 할당 |
| 무선 | 2.4GHz Wi-Fi, 채널 6, WPA2-PSK, 수신부 1대 |
| AP 주소 | ESP-IDF 기본값 192.168.4.1/24 |
| 전송 | IPv4 UDP 유니캐스트, 양쪽 로컬 포트 4210 |
| 페이로드 | 원시 오디오 대신 TEST 이벤트 |

SoftAP는 접속 방식이고 UDP는 접속 후 데이터를 전달하는 방식입니다. 작은 이벤트에 대해 왕복 지연과 손실을 관찰하기 위해 UDP를 선택했습니다. 자체 ACK·제한된 재전송을 사용하며 전송 성공을 보장하지 않습니다. 실제 초인종·화재 이벤트 및 알림 장치 연결은 후속 단계입니다.

## 패킷 형식

UDP 데이터그램은 정확히 16바이트입니다. 모든 32비트 정수는 unsigned big-endian입니다. C 구조체 메모리를 직접 전송하지 않습니다.

| 오프셋 | 크기 | 내용 |
| --- | --- | --- |
| 0 | 2 | ASCII `HA` (0x48 0x41) |
| 2 | 1 | 버전 1 |
| 3 | 1 | 종류: HELLO=1, EVENT=2, ACK=3, PM_STATS=4, POWER_STATE=5 |
| 4 | 4 | 부팅 세션 ID (esp_random) |
| 8 | 4 | 순번 또는 상태 값(종류별 의미는 아래 참고) |
| 12 | 4 | 이벤트 코드 또는 상태 값(종류별 의미는 아래 참고) |

- HELLO: C3의 세션 ID, 순번 0, 코드 0.
- EVENT: S3의 세션 ID, 순번 1부터 증가, 코드 1.
- ACK: 수신 EVENT의 세션 ID·순번·코드를 그대로 복사.
- 예시 EVENT: `48 41 01 02 12 34 56 78 00 00 00 01 00 00 00 01`.
- 길이·매직·버전·종류·방향·포트·필드가 맞지 않는 패킷은 처리하지 않습니다. EVENT는 AP IP에서, ACK는 등록된 수신부 IP에서 온 패킷만 처리합니다.

## 절전 검증용 상태 패킷

기존 16바이트 형식과 버전 1을 유지하며 다음 종류를 확장했다. 기존 이벤트와 ACK의 형식은 동일하다. 상태 보고는 진단용이며 별도 ACK·재전송을 하지 않는다.

| 종류 | 방향 | session | seq 필드 | value 필드 |
| --- | --- | --- | --- | --- |
| PM_STATS=4 | C3 → S3 | C3 부팅 세션 | SDK 실제 Light-sleep 성공 횟수 | SDK Light-sleep 거절 횟수 |
| POWER_STATE=5 | C3 → S3 | C3 부팅 세션 | auto Light-sleep 설정(0/1) | Wi-Fi PS 모드(MIN_MODEM=1) |

S3는 등록된 C3 주소·포트·세션에서 온 상태만 기록한다. C3는 약 10초마다 상태를 전송한다. 성공 횟수는 유휴 루프 횟수가 아니라 SDK `esp_pm_dump_locks()`의 프로파일링 카운터다. 버전 1을 지원하더라도 이 확장이 없는 이전 펌웨어와 혼용하면 상태 보고와 heartbeat 정책이 다르므로 두 보드를 함께 업데이트한다.

## 연결과 전달

1. S3는 SoftAP와 UDP 소켓을 엽니다. C3는 Wi-Fi 연결 및 DHCP 이후 AP로 HELLO를 약 10초마다 전송합니다. IP 획득 전에는 전송하지 않으며, 재접속 후 HELLO 일정을 초기화합니다.
2. S3는 HELLO의 송신 IP/포트와 C3 세션을 등록합니다. 주소나 C3 부팅 세션이 바뀌면 이전 대기 이벤트를 취소하고 새 수신부에 테스트 이벤트를 시작합니다.
3. S3는 약 3초마다 새 TEST 이벤트를 보냅니다. 한 번에 확인을 기다리는 이벤트는 하나입니다.
4. 300ms 이내 ACK가 없으면 동일 패킷을 다시 보냅니다. 최초 송신 포함 최대 4회입니다. 네 번째 시도 후 300ms 동안에도 ACK가 없으면 FAILED를 기록합니다. 다음 테스트 이벤트로 진행하며 실패 이벤트를 영구 저장하지 않습니다.
5. C3는 현재 기록된 S3 세션에서 이전에 처리한 순번 이하의 패킷을 다시 실행하지 않으며 ACK만 다시 보냅니다. 새 세션은 새로운 S3 부팅으로 간주합니다.
6. S3는 HELLO가 35초 넘게 없으면 수신부 등록·대기 이벤트를 해제합니다. C3는 Wi-Fi 끊김 이벤트에서 자동 재접속합니다. S3의 수신 대기는 50ms, C3의 수신 대기는 1초입니다. 이벤트 수신 시에는 타임아웃을 기다리지 않고 처리하며, 주기 작업에는 스케줄링 오차가 있습니다.

## 제한

- 중복 처리 기록은 RAM의 마지막 S3 세션/순번뿐입니다. C3 재부팅이나 세션 간 지연 패킷까지 포함한 exactly-once 보장은 없습니다. 실제 제품에서는 세션 합의와 보존 정책을 추가해야 합니다.
- ACK 유실 시 C3가 처리했어도 S3에 FAILED가 남을 수 있습니다. ACK는 실제 사용자의 인지나 장치 작동을 증명하지 않습니다.
- 인증은 테스트용 WPA2 공유 암호에 의존합니다. 애플리케이션 인증, 암호화, 영구 큐, 복수 수신부, 오디오 전송은 구현하지 않았습니다.
- 세션 ID는 32비트 난수이며 충돌 가능성이 있습니다. 순번 wrap 시 새 세션을 만듭니다.
- C3는 Wi-Fi MIN_MODEM 및 자동 Light-sleep을 사용합니다. S3는 AP를 유지합니다. 실제 절전 성공 횟수와 ACK를 검증하지만, 소비전류·배터리 수명은 아직 측정하지 않았습니다. [절전 구조](../architecture/README.md)와 [검증 결과](../test_results/low_power_2026-10-30.md)를 참고합니다.

## 참고

- [Espressif SoftAP 공식 예제](https://github.com/espressif/esp-idf/tree/v5.5.5/examples/wifi/getting_started/softAP)
- [Espressif STA 공식 예제](https://github.com/espressif/esp-idf/tree/v5.5.5/examples/wifi/getting_started/station)
- [ESP-IDF BSD 소켓 API](https://docs.espressif.com/projects/esp-idf/en/v5.5.5/esp32s3/api-guides/lwip.html)
- [LILYGO T7-S3 공식 저장소](https://github.com/Xinyuan-LilyGO/T7-S3)

실행 방법은 [통신 프로젝트 README](../../communication_tests/softap_udp/README.md), 실제 검증 상태는 [검증 기록](../test_results/softap_udp.md)에 기록합니다.
