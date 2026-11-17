# ha_comm — S3·C3 공통 통신 코어

ESP32-S3 송신부(`firmware/transmitter_s3/`)와 XIAO ESP32-C6 수신부(`firmware/receiver_c6/`)가 함께 링크하는 ESP-IDF 컴포넌트입니다. 코드·문서의 "C3"(`c3_session`, `HA_HELLO_C3_REBOOTED` 등)는 2026-11-17 보드 변경 뒤에도 "수신부"를 뜻하는 이름으로 유지합니다. `src/`에는 ESP-IDF API가 없는 portable C만 두어, 펌웨어에 들어가는 코드와 **같은 파일**을 PC에서 단위 시험합니다. Wi-Fi·소켓·FreeRTOS 연결부는 각 보드의 `components/ha_sender`, `components/ha_receiver`에 있습니다.

| 파일 | 역할 |
| --- | --- |
| `include/ha_proto.h`, `src/ha_proto.c` | v1 16바이트 프레임과 v2 TIME_SYNC 28바이트 인코더·디코더, big-endian 처리 |
| `include/ha_events.h`, `src/ha_events.c` | 공개 이벤트 등록 테이블: ID·이름·제품 허용 여부·포화 보호·TTL·동시 감지 순서 |
| `include/ha_tx.h`, `src/ha_tx.c` | S3 대기열, stop-and-wait 재전송·backoff, ACK 세 필드 검증, 단일 수신기 등록 상태 |
| `include/ha_rx.h`, `src/ha_rx.c` | C3 EVENT 검증, 중복·충돌 판정, “접수 성공 → 기록 → ACK” 계약 |
| `include/ha_rx_pending.h`, `src/ha_rx_pending.c` | 수신부 "접수했지만 아직 화면에 안 뜬" 이벤트 표. 화면 전송 후 ACK 계약(2026-11-17)에서 그 사이 재전송에 ACK하지 않게 하고, 화면 감시에 가장 오래 기다린 항목을 알려 줌 |
| `include/ha_time.h`, `src/ha_time.c` | C3 시각 상태(미동기화/동기화/오래됨), 경과 시간·불확실성 계산 |
| `include/ha_config.h`, `Kconfig` | SSID·포트·비콘/DTIM·타이밍·큐 크기 등 공동 설정. 펌웨어는 Kconfig, 호스트 시험은 같은 기본값 사용 |
| `test/host/` | 호스트 단위 시험 (`run_tests.sh`) |

## 호스트 단위 시험

```bash
bash firmware/common/ha_comm/test/host/run_tests.sh
```

시스템 `cc`로 `-Wall -Wextra -Werror -Wconversion`, AddressSanitizer·UBSan을 켜고 빌드합니다. 성공하면 `HA_COMM_HOST_TESTS_PASS`와 검사 수를 출력하고 종료 코드 0을 반환합니다. 시험은 모의 시계와 모의 유실 링크를 쓰며, 무선·절전·전류 측정을 대신하지 않습니다.

## 프로토콜

### v1 (변경 없음)

`docs/communication/README.md` 3절의 16바이트 프레임을 그대로 사용합니다. 길이를 먼저 검사하므로 17바이트 이상의 데이터그램을 16바이트로 잘라 정상으로 처리하지 않습니다. 수신 버퍼는 `HA_RX_BUFFER_LEN`(최대 프레임 + 1바이트)입니다.

### v2 TIME_SYNC (신규 확장, 선택 기능)

v1 EVENT의 `event_id`를 timestamp로 재해석하지 않고, 버전과 길이 필드를 따로 둔 독립 제어 메시지로 정의했습니다. 구형 C3는 길이·버전 검사에서 이 메시지를 폐기하므로 v1 알림 경로와 호환됩니다.

| 오프셋 | 크기 | 필드 | 값 |
| ---: | ---: | --- | --- |
| 0 | 2 | magic | ASCII `HA` |
| 2 | 1 | version | `2` |
| 3 | 1 | type | `0x10` TIME_SYNC (S3 → C3) |
| 4 | 2 | length | 전체 길이 `28`, big-endian. 실제 데이터그램 길이와 달라도 폐기 |
| 6 | 2 | flags | `0` (예약). 0이 아니면 이 버전은 폐기 |
| 8 | 4 | session | S3 부팅 세션 (EVENT와 같은 값) |
| 12 | 4 | seq | TIME_SYNC 전용 순번, 1 이상. EVENT 순번과 별개 |
| 16 | 8 | utc_ms | Unix epoch 밀리초(UTC), 송신 직전 S3가 읽은 값 |
| 24 | 4 | accuracy_ms | 송신 시점 S3 시계 자체의 불확실성 |

- `utc_ms`가 2025-01-01 이전이거나 2100년 이후면 “S3 시계 미설정”으로 보고 폐기합니다. 컴파일 시각이나 epoch 근처 값이 현재 시각으로 표시되지 않습니다.
- 같은 S3 세션에서 seq가 증가하지 않으면 지연·중복 메시지로 무시합니다. 새 세션(S3 재부팅)은 수용합니다.
- 전송 지연은 메시지에 없습니다. C3는 `accuracy_ms + 전달 지연 상한(기본 1000ms)`을 기본 불확실성으로 두고, 경과 시간 × 드리프트(기본 1000ppm, **미측정 보수값**)만큼 키웁니다. 24시간 동안 동기화가 없으면 `STALE`입니다.
- 타임존·서머타임은 표시 계층에서 처리합니다.
- ACK가 없는 주기 송신입니다. 실패해도 알림 큐에 영향이 없습니다.

벡터 예: session `0x12345678`, seq 1, utc_ms `1763164800123`, accuracy 250ms →
`48 41 02 10 00 1C 00 00 12 34 56 78 00 00 00 01 00 00 01 9A 84 CF C4 7B 00 00 00 FA`
(호스트 시험에서 Python `int.to_bytes(8, "big")`로 따로 계산한 값과 대조).

## 소리 종류 추가 절차

1. `ha_events.h`에 새 상수를 추가합니다. 다음 미할당 값은 `0x00000102`입니다. 기존 ID를 재사용하거나 의미를 바꾸지 않습니다.
2. `ha_events.c` 테이블에 한 줄을 추가합니다: 이름, `product`, 포화 보호(`evict_protected`), TTL. 테이블 순서가 동시 감지 시 접수 순서입니다.
3. TTL을 설정으로 바꾸려면 `transmitter_s3/components/ha_sender/Kconfig`와 `ha_config.h`에 항목을 추가합니다.
4. `test/host/test_ha_comm.c`에 EVENT/ACK 벡터와 허용 시험을 추가합니다.
5. 수신부 화면·진동 매핑(`firmware/receiver_c6/components/ha_ui_core/`의 presenter·화면·글꼴)과 ML 팀의 라벨 → ID 매핑을 함께 갱신합니다.
6. **C3를 먼저 배포**합니다. 구형 C3는 미등록 ID를 폐기하므로(알림·ACK·중복 기록 없음) S3가 재전송만 하다가 TTL로 만료됩니다.
7. 프레임 필드·크기·의미를 바꿀 때만 새 프로토콜 버전을 만듭니다. ID 추가만으로는 v1을 유지합니다.

모델 재학습 기능은 이 모듈의 범위가 아닙니다.
