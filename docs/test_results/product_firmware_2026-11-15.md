# 제품 통신 펌웨어 자동 검증 — 2026-11-15

대상: `firmware/transmitter_s3/`, `firmware/receiver_c3/`, `firmware/common/ha_comm/`. 기준 커밋 `cccf7dd`에 미커밋 변경을 더한 작업 트리. ESP-IDF v5.5.5, 호스트 `cc` (Ubuntu 11.4.0).

> **2026-11-17 이후:** 수신부는 XIAO ESP32-C6(`firmware/receiver_c6/`)으로 바뀌었고, ACK는 "화면 전송 후"로 바뀌었습니다. 이 문서는 2026-11-15 C3 기준 기록으로 그대로 둡니다. 현재 수신부 검증과 **실기기 시험 절차의 최신본**은 [C6 수신부 검증 — 2026-11-17](receiver_c6_2026-11-17.md)에 있습니다.

**보드 연결 없이 수행한 결과입니다.** 무선 송수신·절전·전류·UI 출력은 검증하지 않았습니다. 기존 TEST 펌웨어의 실측값(2026-10-30, 2026-11-11)은 이 펌웨어의 성능이 아닙니다.

## 결과

| 검증 | 결과 |
| --- | --- |
| 호스트 단위 시험 `bash firmware/common/ha_comm/test/host/run_tests.sh` | `HA_COMM_HOST_TESTS_PASS: 697 checks, 0 failed`. `-Werror -Wconversion`, ASan·UBSan 사용 |
| S3 제품 빌드 `idf.py -C firmware/transmitter_s3 -B firmware/transmitter_s3/build build` | 성공, 경고 없음. 앱 0xa7a10 bytes (1MB 파티션의 35% 여유) |
| C3 제품 빌드 `idf.py -C firmware/receiver_c3 -B firmware/receiver_c3/build build` | 성공, 경고 없음. 앱 0xb9b70 bytes (27% 여유). PM·tickless·profiling 활성 확인 |
| 진단 옵션 빌드 (S3: 주입·TIME_SYNC·화재 우선·TEST 허용 / C3: 로그 소비자·TEST 허용) | 양쪽 성공. 산출물은 세션 scratchpad에 두었고 저장소에 남기지 않았습니다 |
| 범위 분리 | 제품 ELF에 `DIAG_INJECT`·`DIAG_CONSUMER`·`tflite`·`Dscnn` 문자열 0건(진단 빌드에서는 해당 문자열 존재). S3 빌드 구성요소에 dscnn 없음 |
| `git diff --check` (신규 파일 포함) | 통과 |

## 호스트 시험이 다루는 항목

- **프로토콜:** 문서 6절의 화재/초인종 EVENT·ACK 4개 바이트 벡터와 실제 인코더 출력이 일치하고, 디코딩도 일치합니다. 15·17·0바이트, 잘못된 magic·version·type, seq=0, ID 0·1·258, HELLO 필드 조건을 거부합니다.
- **ACK:** session·seq·ID 중 하나만 달라도 무시합니다. 다른 IP·포트에서 온 ACK도 무시합니다. 완료 뒤 늦게 온 ACK는 무시합니다. 문서 벡터와 같은 실제 송신 바이트를 확인했습니다.
- **재전송:** ACK가 유실되면 같은 `(session, seq, id)`로 재전송하고, 4회 묶음 뒤 backoff는 최대 16s까지 늘어납니다.
- **큐:**
  - 화재·초인종 동시 입력(역순으로 넘겨도 FIRE가 seq 1)
  - 첫 ACK 대기 중 새 입력이 와도 활성 이벤트 유지, stop-and-wait 순서
  - 포화 4가지 경우(초인종 폐기, 화재 보호, 화재만 남았을 때 초인종 거부, 용량 1)
  - 오프라인 입력은 seq를 소비하지 않음
  - TTL 만료(대기 중·활성 모두), 취소
- **세션:**
  - station이 떠난 동안 송신을 멈추고, 장기 단절 중에도 큐에 적재
  - IP 변경·C3 재부팅 HELLO 뒤에도 S3 세션과 큐를 유지하고 즉시 재시도
  - stale이어도 재전송을 계속함
  - 순번 소진 시 새 세션
  - S3 재부팅(새 세션)을 수용하고, 지연된 이전 세션 패킷은 중복으로 처리
  - C3 재부팅 뒤 재접수는 한계로 명시
- **C3 접수:** 소비자가 없거나 거부하면 ACK·기록이 없고, 재전송이 오면 나중에 접수합니다. 중복은 ACK만 다시 보내고 소비자를 다시 호출하지 않습니다. 같은 session·seq에 다른 ID가 오면 충돌로 처리하고 ACK하지 않습니다. 잘못된 프레임은 상태를 바꾸지 않습니다.
- **종단 간 모의 시험:** ACK 3회를 유실시켜도 두 이벤트가 소비자에게 각각 정확히 1회 전달되고, ACK가 완료됩니다.
- **시각:** 미동기화 상태에서는 시각을 만들어 내지 않습니다. 경과 시간을 반영하고, 드리프트에 따라 불확실성이 커집니다. 같은 세션의 이전 seq는 무시하고, 새 세션은 수용합니다. 24시간 뒤 STALE이 됩니다. TIME_SYNC 벡터와 오류(길이 필드·flags·seq 0·시각 범위·type)를 검사하고, v1 디코더가 v2를 거부하는지 확인합니다. 시각 보정을 섞어도 송신 일정이 바이트 단위로 같습니다.

## 실기기 후속 시험 (미수행)

모니터: `HEARING_FIRMWARE=product bash tools/communication/monitor.sh S3_PORT C3_PORT`. 펌웨어 리비전·sdkconfig·원본 양측 로그와 결과를 분리해 기록합니다.

1. **전원 인가 자동 연결:** S3 `S3 AP ready … DTIM=3`, C3 `STA got IP` → S3 `receiver registered (new_peer)`.
2. **UI 미연결 확인:** 진단 주입을 켠 S3에서 EVENT를 보냅니다. 제품 C3는 `NOT accepted (no UI consumer attached)`를 남기고 ACK하지 않아야 합니다. S3는 backoff로 재전송하다가 TTL에 따라 `EXPIRED`가 되어야 합니다.
3. **API → 무선 → 접수 → ACK:** C3 진단 소비자 또는 실제 UI 소비자 빌드로 시험합니다. `PUBLISH` → `TX` → C3 `EVENT accepted` → S3 `ACK`가 이어지고 세 필드가 일치하는지 확인합니다. 동시 2건 배치는 FIRE seq N, DOORBELL seq N+1 순서여야 합니다. **진단 주입 결과를 소리 인식 성공으로 보고하지 않습니다.**
4. **ACK 유실·중복:** C3의 ACK 송신을 임시로 막은 빌드에서 `duplicate` 재ACK와 알림 1회를 확인합니다. 미등록 ID·잘못된 길이 주입도 시험합니다.
5. **단절·재부팅:** C3 전원을 끈 뒤 S3에서 `station left`·적재를 확인합니다. 복구 후 같은 seq로 ACK되는지 봅니다. C3 재부팅(`c3_rebooted`), S3 재부팅(새 세션), 35초 초과 stale, TTL 만료도 확인합니다.
6. **300ms 목표 절전:**
   - `C3_PM sleep_count`가 증가하는지(실제 절전 진입)
   - ACK 왕복(`last_tx_to_ack`) 분포가 약 0~310ms 범위에 들어가는지(수신 기회 간접 확인)
   - 전류계로 DTIM=1과 DTIM=3의 평균 전류 비교
   설정 활성화만으로는 절전 성공으로 판정하지 않습니다.
7. **UI 연결 후:** 표시·진동이 실제로 동작하는지, 긴 진동 재생 중 두 번째 이벤트도 접수·ACK되는지, 진동 부하에서 절전·지연이 어떤지 확인합니다.
8. **시각 확장:** 공급자 구현 뒤 TIME_SYNC 수신·`SYNCED` 상태, 인터넷 장애 시 알림이 정상인지, AP+STA 채널 변경 영향을 확인합니다.
