# LILYGO T7 S3 v1.1 송신부

2026-11-15부터 이 디렉터리가 **S3 제품용 통신 펌웨어의 ESP-IDF 프로젝트**입니다. S3가 SoftAP를 열고, ML·제어 계층이 넘긴 최종 이벤트를 UDP로 C3에 전달합니다. 전달 방식은 대기열 → 반복 송신 → ACK 확인입니다. 오디오 입력·추론·판정 정책은 이 펌웨어에 들어 있지 않으며, ML 입력이 없으면 이벤트를 만들지 않고 대기합니다.

| 경로 | 내용 |
| --- | --- |
| `main/main.c` | `ha_sender_start()`만 호출하는 진입점 |
| `components/ha_sender/` | 공개 API(`include/ha_sender.h`), SoftAP·UDP 작업, 진단 옵션, Kconfig |
| `../common/ha_comm/` | 프로토콜·대기열·재전송 공통 코어 ([설명](../common/ha_comm/README.md)) |
| `dscnn/` | ML 팀의 DS-CNN 컴포넌트. `esp-tflite-micro`가 필요하므로 이 빌드에는 **포함하지 않습니다** |

`communication_tests/softap_udp/`는 과거 TEST 실험·회귀 참고용으로 보존하며, 이 펌웨어는 그 코드를 include하지 않습니다.

## 빌드·설정

저장소 루트에서 실행합니다. sdkconfig는 `firmware/transmitter_s3/build/sdkconfig`에 생기며 Git에서 제외됩니다. 대상 칩은 `sdkconfig.defaults`에 `esp32s3`으로 지정돼 있습니다.

```bash
source ~/esp/esp-idf/export.sh            # ESP-IDF v5.5.5
idf.py -C firmware/transmitter_s3 -B firmware/transmitter_s3/build build
idf.py -C firmware/transmitter_s3 -B firmware/transmitter_s3/build menuconfig   # 설정 변경
idf.py -C firmware/transmitter_s3 -B firmware/transmitter_s3/build -p S3_PORT flash monitor
```

`menuconfig`에는 메뉴가 두 개 있습니다.
- **Hearing Alert communication:** SSID·비밀번호·채널·포트·비콘/DTIM·HELLO 주기를 설정합니다. 양 보드에서 같은 값을 써야 합니다.
- **Hearing Alert S3 sender:** 큐 크기, ACK 대기, 재시도, TTL, 화재 우선, TIME_SYNC, 진단 주입을 설정합니다.

기본 SSID와 비밀번호는 기존 시험용 값입니다. 제품별 값은 menuconfig로 바꾸고, 가정용 Wi-Fi 비밀번호를 소스·로그·Git에 남기지 마세요. 펌웨어는 비밀번호를 출력하지 않습니다.

## ML·제어 팀용 연결 안내

```c
#include "ha_sender.h"   // REQUIRES ha_sender

// 최종 판정(rising edge 등 상위 정책 통과) 1건 = 호출 1회
ha_event_token_t tok;
ha_publish_result_t r = ha_publish_event(HA_EVENT_FIRE_ALARM, &tok);

// 같은 판정 구간에서 두 소리가 함께 확정되면
const uint32_t ids[] = {HA_EVENT_FIRE_ALARM, HA_EVENT_DOORBELL};
ha_publish_events(ids, 2, results, tokens);   // 순서와 무관하게 FIRE → DOORBELL로 접수
```

| 항목 | 계약 |
| --- | --- |
| 입력 | 정규 이벤트 ID: `HA_EVENT_FIRE_ALARM`(0x100, 256), `HA_EVENT_DOORBELL`(0x101, 257). 모델 클래스 인덱스(0/1/2)를 그대로 넣지 마세요. `other`·배경음은 이벤트가 아닙니다 |
| 거부 ID | 0(예약), TEST(1, 진단 옵션 전용), 미등록 ID → `HA_PUBLISH_ERR_INVALID_ID` |
| 반환값 | `OK` / `OK_EVICTED_OLDEST`(접수됨, 오래된 대기 이벤트 1개 폐기) / `ERR_QUEUE_FULL` / `ERR_INVALID_ID` / `ERR_NOT_STARTED` / `ERR_BUSY` |
| 호출 문맥 | 아무 FreeRTOS 작업에서나 호출할 수 있습니다(ISR 제외). 네트워크와 ACK를 기다리지 않습니다. 내부 잠금은 최대 20ms이고, 넘기면 `ERR_BUSY`를 반환합니다 |
| 소유권·수명 | ID 값만 복사하므로 호출자 버퍼를 계속 유지할 필요가 없습니다. 대기열은 RAM에 있어서 S3가 재부팅되면 사라집니다 |
| 중복 생성 방지 | **호출자 책임**입니다. 호출 1회는 항상 새 알림이 됩니다. 매 추론 프레임마다 호출하지 말고, rising edge·재알림 간격·재무장을 ML/상위 제어에서 결정하세요. 같은 소리의 재전송은 내부에서 같은 `(session, seq, id)`로 처리합니다 |
| 결과 관찰 | `ha_sender_set_status_callback()`이 `ACKED`/`EXPIRED`/`EVICTED`/`CANCELED`를 token과 함께 알려 줍니다. 콜백은 송신 작업에서 실행되므로 블로킹하지 마세요 |
| 취소 | `ha_cancel_event(token)`. 이미 송신한 이벤트는 C3가 접수했을 수 있습니다 |
| S3 자체 조명 | `ha_sender_set_local_alert_hook()`. 유효한 ID면 큐 접수 결과와 관계없이 호출자 문맥에서 즉시 호출되고, C3 ACK를 기다리지 않습니다. 이 저장소에는 LED 드라이버가 없습니다 |
| ACK의 의미 | C3 소프트웨어가 알림 큐에 접수했다는 뜻입니다. 화면·진동이 끝났거나 사용자가 알아챘다는 뜻이 아닙니다 |

ML 통합(오디오, `dscnn/` 연결, `esp-tflite-micro` 의존성 추가, 판정 정책)은 이번 작업에서 **수행하지 않았습니다**. 통합할 때 `dscnn/`를 이 프로젝트의 `EXTRA_COMPONENT_DIRS`에 추가하고, 위 API로 이벤트를 넘기면 됩니다. 클래스 → ID 매핑은 `docs/ml/MODEL_HANDOFF.md` 5절 표와 같습니다.

### ML 컴포넌트 현황 (기존 안내 유지)

[DS-CNN 컴포넌트](dscnn/README.md)에는 1초 PCM 전처리와 TFLM 실행 연결 코드가 있습니다. portable 전처리의 host 검증만 끝났고, ESP-IDF 빌드·실제 마이크·보드 추론은 아직 수행하지 않았습니다. 최신 학습 후보의 파일 위치·입력 계약·시간 판정·통신 이벤트 ID 연결은 [모델 연결 안내](../../docs/ml/MODEL_HANDOFF.md)를 따릅니다. [`native40_r06_v1`](../../ml/models/pursuit_20261114/README.md)는 기존 98×40 frontend와 맞는 확률 출력 모델입니다. 모델 연결 안내 5절의 “S3 TEST 생성 부분·ACK 검사·C3 value==1 검사” 통합 항목은 이 펌웨어의 `ha_publish_event()`, ACK 세 필드 검증, C3 등록 ID 검사로 구현했습니다.

## 전송 정책 (개발 기본값)

| 항목 | 값 | 근거·한계 |
| --- | --- | --- |
| 전송 방식 | FIFO, stop-and-wait, 활성 이벤트 1개 | seq는 **송신을 시작할 때** 부여합니다. 폐기·취소·만료·순서 변경이 있어도 C3의 “seq 이하는 중복” 판정이 유지됩니다 |
| 큐 용량 | 8 (활성 포함) | 설계 기본값. 동시 감지 2건 + 단절 중 적재 여유 |
| 포화 처리 | 가장 오래된 대기 이벤트를 폐기하되 FIRE_ALARM은 FIRE_ALARM으로만 밀려남. 활성 이벤트는 폐기하지 않음. 남은 것이 모두 화재면 새 DOORBELL은 `QUEUE_FULL` | 2026-11-15 사용자 결정 |
| TTL | FIRE_ALARM 10분, DOORBELL 2분 (게시 시각부터, monotonic) | 2026-11-15 사용자 결정, 개발 기본값 |
| ACK 대기 | 시도당 700ms | 설계 기본값. C3 수신 간격 307.2ms에 비콘 1회 누락(약 614ms)과 처리 시간을 더한 값입니다. **미측정** |
| 재시도 | 4회 묶음, 실패하면 2s → 4s → … 최대 16s backoff | 기존 4회 묶음·2초를 유지하고 장기 단절에 대비해 backoff를 추가했습니다 |
| 화재 우선 | 기본 꺼짐. `HA_TX_FIRE_FIRST`를 켜면 다음 활성 이벤트로 가장 오래된 화재를 고름(선점은 하지 않음) | 향후 정책을 위한 경계 |
| C3 미연결 | 큐에 쌓기만 하고 송신하지 않음. seq도 소비하지 않음 | AP station leave 이벤트 기준 |
| HELLO 35s 없음 | stale로 표시하고, peer·큐를 유지한 채 backoff로 재전송을 계속함 | 기존 동작 유지 |
| C3 재접속·IP 변경·재부팅 | 새 HELLO 주소로 갱신하고, 큐와 **S3 세션은 유지**함. 현재 묶음을 즉시 재시작함 | 세션이 바뀌었다고 대기 이벤트를 지우지 않습니다 |
| 순번 소진 | 새 무작위 세션으로 바꾸고 seq=1부터 시작 | |

**보장 범위:** RAM 보관이라 S3가 재부팅되면 대기 이벤트가 사라집니다. C3가 재부팅되면 중복 기록이 사라져서, 접수했지만 ACK를 잃은 이벤트가 다시 알림될 수 있습니다. exactly-once 전달, 영구 보존, 기한 내 전달을 보장하지 않습니다. 수신기는 1대(`max_connection=1`)이고 peer 상태는 `ha_tx_peer_t` 하나입니다. 여러 수신기로 확장하려면 peer별 큐·ACK 상태로 분리해야 하며, 전역 ACK 하나가 모든 수신기의 성공을 뜻하지 않습니다.

## 시각 확장 (기본 꺼짐)

`HA_TIME_SYNC_TX`를 켜고 `ha_sender_set_time_provider()`로 실제 UTC 공급자를 등록하면 TIME_SYNC를 보냅니다. 보내는 시점은 C3가 등록된 직후와 10분마다이며, 대기 이벤트가 없을 때만 보냅니다. 형식은 [ha_comm](../common/ha_comm/README.md#v2-time_sync-신규-확장-선택-기능)에 정의돼 있습니다. SNTP·가정용 Wi-Fi(AP+STA) 연결과 설정 UI는 **구현하지 않았습니다**.

후속으로 AP+STA를 도입할 때 고려할 점이 있습니다. S3의 STA가 외부 공유기 채널로 옮겨 가면 SoftAP 채널도 함께 바뀌어 C3 연결이 끊길 수 있습니다. 그래서 다음 항목을 별도로 시험해야 합니다.
- 채널 변경·외부 AP 단절이 C3 재접속, 절전, 미전달 이벤트에 미치는 영향
- 인터넷 장애가 로컬 알림 장애로 번지지 않는지

시간 미동기화는 알림 큐·UDP 작업을 막지 않으며, 재전송·만료 타이머는 UTC와 무관한 monotonic clock만 씁니다.

## 진단 옵션 (기본 꺼짐, 제품 빌드 금지)

- `HA_DIAG_INJECT_EVENTS`: 주기적으로 FIRE/DOORBELL과 동시 2건 배치를 합성해 무선 경로를 시험합니다. 로그에 `DIAG_INJECT (synthetic, not a detection)`이 찍힙니다. **소리 인식 결과가 아닙니다.**
- `HA_DIAG_ALLOW_TEST_EVENT`: TEST(1)를 허용합니다. 양 보드에서 함께 켜야 합니다.

## 로그

`PUBLISH`, `TX seq=… id=… attempt=…`, `ACK seq=… attempts=… last_tx_to_ack=…ms`, `RETRY_PENDING`, `EXPIRED`/`EVICTED`/`CANCELED`, `receiver registered (…)`, `C3_PM`, `C3_STATE`. `last_tx_to_ack`는 마지막 송신부터 ACK까지의 시간으로, 이전 시도에 대한 ACK가 늦게 도착한 경우도 포함하는 상한값입니다. `tools/communication/analyze_log.py`는 TEST 펌웨어 로그용이라 이 형식을 분석하지 않습니다.
