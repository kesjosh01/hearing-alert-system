# ESP32-C3 SuperMini 수신부

2026-11-15부터 이 디렉터리가 **C3 제품용 통신 펌웨어의 ESP-IDF 프로젝트**입니다. C3는 S3 SoftAP에 STA로 접속한 상태를 유지하면서 Modem-sleep(MIN_MODEM)과 자동 Light-sleep을 씁니다. EVENT를 받으면 검증·중복 확인을 거쳐 UI 알림 큐에 접수하고, 접수에 성공한 경우에만 ACK합니다. 표시·진동 드라이버는 이 저장소에 아직 없습니다.

| 경로 | 내용 |
| --- | --- |
| `main/main.c` | `ha_receiver_start()` 진입점. UI 미연결 상태를 경고로 표시 |
| `components/ha_receiver/` | 공개 API(`include/ha_receiver.h`), STA·절전·UDP 수신·HELLO 작업, Kconfig |
| `../common/ha_comm/` | 프로토콜·중복 억제·시각 상태 공통 코어 |

## 빌드

```bash
source ~/esp/esp-idf/export.sh            # ESP-IDF v5.5.5
idf.py -C firmware/receiver_c3 -B firmware/receiver_c3/build build
idf.py -C firmware/receiver_c3 -B firmware/receiver_c3/build -p C3_PORT flash monitor
```

`sdkconfig.defaults`에서 `CONFIG_PM_ENABLE`, `CONFIG_FREERTOS_USE_TICKLESS_IDLE`, `CONFIG_PM_PROFILING`을 켜고, USB 연결이 Light-sleep을 막지 않게 설정합니다. 이 값들이 꺼져 있으면 컴파일 오류가 나서, 절전 없는 빌드가 만들어지지 않습니다. 부팅 15초 뒤 자동 Light-sleep을 켜며, 그 뒤로 USB 콘솔이 멈출 수 있습니다. 상태는 S3 로그의 `C3_PM`·`C3_STATE`로 확인합니다.

## C3 UI(표시·진동) 팀용 접수 계약

```c
#include "ha_receiver.h"   // REQUIRES ha_receiver

static void ui_task(void *arg) {
    ha_receiver_attach_consumer();             // 이 호출 전에는 EVENT를 ACK하지 않음
    for (;;) {
        ha_alert_t a;
        if (ha_receiver_wait_alert(&a, portMAX_DELAY)) {
            // a.event_id: HA_EVENT_FIRE_ALARM / HA_EVENT_DOORBELL
            // 여기서 표시·진동 패턴 실행 (이 작업 안에서 오래 걸려도 됨)
        }
    }
}
```

| 항목 | 계약 |
| --- | --- |
| ACK 조건 | **UI 소비자가 attach된 상태**에서 이벤트가 알림 큐(기본 8개)에 들어가야 ACK합니다. 소비자가 없거나 큐가 차 있으면 ACK도, 중복 기록도 남기지 않습니다. S3는 같은 이벤트를 계속 재전송합니다(2026-11-15 결정) |
| 전달 데이터 | `event_id`, `s3_session`, `seq`, `received_mono_ms`. 모델 클래스 정보는 없습니다 |
| 중복 | ACK를 잃어 재전송된 이벤트는 다시 큐에 넣지 않습니다. C3가 재부팅되면 중복 기록이 사라져 한 번 더 들어올 수 있습니다 |
| 분리 | 수신·ACK는 `ha_rx` 작업이 하고, 표시·진동은 UI 작업이 합니다. 긴 진동 재생이 두 번째 이벤트의 접수나 ACK를 막지 않습니다. 화재·초인종이 함께 오면 FIFO로 두 건이 들어오며, 순차 실행·선점은 UI 정책입니다 |
| 절전 | UI 작업이 `portMAX_DELAY`로 블로킹하면 자동 Light-sleep이 유지됩니다. 짧은 주기 타이머·폴링·주기 로그를 쓰면 절전이 줄어듭니다. 재생 중에 잠들도록 강제하지 않습니다 |
| 미등록 ID | 등록 테이블에 없는 ID와 TEST·0은 폐기합니다(알림·ACK 없음) |
| 시각 | `ha_receiver_get_time()` → `UNSYNCED`/`SYNCED`/`STALE`과 UTC 추정값·불확실성. 동기화 전에는 시각을 표시하지 마세요. 타임존은 UI에서 처리합니다 |

**남은 연결 지점:** 디스플레이 종류·버스·GPIO, 진동 모터 구동 회로, 이벤트별 패턴은 정해지지 않아 구현하지 않았습니다. 하드웨어가 확정되면 UI 팀이 위 작업을 만들고 `ha_receiver_attach_consumer()`를 호출하면 됩니다. 확인되지 않은 GPIO를 기본으로 켜지 마세요.

진단용 `HA_RX_DIAG_LOG_CONSUMER`(기본 꺼짐)는 로그만 남기는 소비자를 붙여, UI 없이 무선·ACK 시험을 할 수 있게 해 줍니다. 로그에 `DIAG_CONSUMER … (log only; no UI output)`이 찍히며, **실제 알림 출력이 아닙니다**. 제품 빌드에서는 켜지 마세요.

## 300ms 목표 수신 간격의 구현

| 구분 | 내용 |
| --- | --- |
| 목표 | 약 300ms 간격의 주기적 Wake-up (2026-11-10 회의) |
| 실제 설정 | S3 AP `beacon_interval=100` TU, `dtim_period=3`. C3 `WIFI_PS_MIN_MODEM` (ESP-IDF v5.5.5 헤더: “station wakes up to receive beacon every DTIM period”) |
| 명목 간격 | 100 × 1.024ms × 3 = **307.2ms**. 정확히 300ms는 설정할 수 없습니다. `beacon_interval`은 100 TU의 배수만 허용되므로, 가장 가까운 값을 골랐습니다(+2.4%) |
| 이전 값 | DTIM=1, 102.4ms (TEST 펌웨어 2026-10-30 측정 조건) |
| `listen_interval` | 3으로 설정했지만 MAX_MODEM에서만 수신 간격을 결정합니다. MIN_MODEM에서는 AP의 DTIM을 따릅니다 |
| CPU | `esp_pm` 40~160MHz, tickless idle, 자동 Light-sleep. 수동 `esp_light_sleep_start()` 반복이나 Deep-sleep은 쓰지 않습니다 |
| 수신 구조 | AP가 절전 중인 STA로 가는 유니캐스트를 버퍼링하고, C3는 DTIM 비콘의 TIM을 보고 수신합니다. AP 버퍼의 크기·보존 시간은 보장되지 않으므로, S3의 반복 송신과 ACK로 보완합니다 |

**검증 상태:** 설정이 들어간 빌드만 확인했습니다. 다음 항목은 **실기기 미검증**입니다.
- 실제 절전 진입(`C3_PM sleep_count` 증가)
- 약 307ms 수신 기회가 실제로 생기는지(ACK 왕복 지연 분포로 간접 확인)
- 소비전류(전류계로 별도 측정)

`vTaskDelay(300)`이나 소켓 timeout은 절전 주기 설정이 아니어서 사용하지 않았습니다.
