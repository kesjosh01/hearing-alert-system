# SoftAP + UDP 기초 통신 실험

LILYGO T7 S3 v1.1을 AP·송신부로, ESP32-C3 SuperMini를 STA·수신부로 사용하는 독립 ESP-IDF 프로젝트입니다. `firmware/`의 오디오·알림 구현과 분리한 통신 실험이며 GPIO, LED, 모터는 사용하지 않습니다. S3가 3초마다 만드는 TEST 이벤트를 C3가 로그로 처리하고 ACK를 보냅니다.

- 코드: `main/main.c` (칩 타깃에 따라 역할 선택)
- 설계: [프로토콜 v1](../../docs/communication/README.md)
- 검증 및 실험 기록: [검증 기록](../../docs/test_results/softap_udp.md)
- 개발 환경: ESP-IDF v5.5.5. Arduino/PlatformIO 설치는 필요하지 않습니다.

## 빌드

저장소 루트에서 실행합니다. 타깃별 빌드 디렉터리와 sdkconfig를 분리하므로 `set-target`로 다른 보드 설정을 덮어쓰지 않습니다.

```bash
source /home/josh/esp/esp-idf/export.sh
cd communication_tests/softap_udp
idf.py -B build/s3 -D SDKCONFIG=build/sdkconfig.s3 -D IDF_TARGET=esp32s3 build
idf.py -B build/c3 -D SDKCONFIG=build/sdkconfig.c3 -D IDF_TARGET=esp32c3 build
```

기본 설정은 내부 USB Serial/JTAG 콘솔, 4MB 플래시, PSRAM 미사용입니다. 이는 작은 테스트 펌웨어의 보수적인 설정이며 T7 S3의 전체 메모리 사양을 지정한 것이 아닙니다. 두 보드 모두 외부 주변장치 핀 설정이 필요하지 않습니다. SuperMini 판매처별 USB·플래시 구성 차이는 실물에서 확인해야 합니다.

## 실제 테스트

1. 데이터 전송 가능한 USB 케이블로 두 보드를 연결하고 `ls -l /dev/serial/by-id/`로 포트를 확인합니다. 부팅 순서는 무관합니다.
2. 아래 명령의 `S3_PORT`, `C3_PORT`를 실제 경로로 교체합니다. 각각 다른 터미널에서 실행합니다.

```bash
idf.py -B build/s3 -D SDKCONFIG=build/sdkconfig.s3 -D IDF_TARGET=esp32s3 -p S3_PORT flash monitor
idf.py -B build/c3 -D SDKCONFIG=build/sdkconfig.c3 -D IDF_TARGET=esp32c3 -p C3_PORT flash monitor
```

3. 자동 다운로드 모드 진입에 실패하면 BOOT를 누른 상태에서 RESET을 눌렀다 떼고 BOOT를 놓은 뒤 재시도합니다. 모니터 종료는 `Ctrl+]`입니다.
4. S3의 `receiver registered`, `TX seq=...`, `ACK seq=...`와 C3의 `TEST event ...` 로그를 확인합니다. C3의 ACK는 소프트웨어 수신 처리 확인이며 실제 LED·진동 동작 확인은 아닙니다.
5. 100개 이벤트의 고유 수신 수·ACK 수·재시도 수·실패 수를 기록합니다. 단일 시도 ACK의 `elapsed`는 송신→C3 처리→ACK 왕복 시간입니다. 재시도 포함 값은 최초 시도 이후 총 확인 시간입니다. 편도 지연으로 해석하지 않습니다.
6. C3 전원 분리 후 S3에서 heartbeat 만료를 확인하고, 다시 켜서 자동 등록·수신되는지 확인합니다. S3 재부팅 후 C3 재접속도 확인합니다.
7. 중복 ACK/재전송 검증은 C3의 `send_packet(... ACK ...)` 호출을 잠시 막아 빌드·플래시한 뒤 진행할 수 있습니다. S3는 4회 시도 후 이벤트를 유지하고 2초 후 다음 묶음을 시작하며 RETRY_PENDING을 남기고, C3는 첫 수신 1회 처리 후 duplicate/stale 로그를 남겨야 합니다. 실험 후 반드시 코드를 복원합니다.

실험 중 PC를 이 AP에 연결할 필요는 없습니다. AP 접속 수는 C3 한 대로 제한합니다. 공유기·인터넷·클라우드 없이 동작합니다. 테스트 자격증명은 `main.c`의 SSID/PASSWORD에 있으며 양쪽을 함께 변경해야 합니다.


## C3 절전 빌드와 모니터

`sdkconfig.defaults.esp32c3`는 C3의 PM·PM 프로파일링·tickless idle을 활성화한다. S3에는 적용하지 않는다. 새 빌드는 타깃별 기본 설정을 자동으로 읽는다. 예전 `build/sdkconfig.c3`에 `CONFIG_PM_ENABLE`이 꺼져 있으면 기본 설정을 추가해도 기존 값이 우선하므로 `idf.py ... menuconfig`에서 아래 값을 켜야 한다. 누락 시 소스가 컴파일 오류를 내어 절전 없는 빌드를 방지한다.

```text
CONFIG_PM_ENABLE=y
CONFIG_PM_PROFILING=y
CONFIG_FREERTOS_USE_TICKLESS_IDLE=y
# CONFIG_USJ_NO_AUTO_LS_ON_CONNECTION is not set
```

C3는 연결 후 MIN_MODEM, 부팅 15초 후 자동 Light-sleep을 사용한다. S3는 100 TU 비콘/DTIM=1의 AP를 유지한다. 이 SuperMini의 인증 시간 초과가 송신 출력 11 dBm 제한 후 해소된 비교 결과에 따라 해당 제한을 유지한다. 다른 보드에서도 반드시 같은 값이 최적이라는 뜻은 아니다.

저장소 루트에서 두 보드가 이미 플래시된 상태로 다음을 실행한다. 포트는 첫 번째 인자가 S3, 두 번째가 C3다. 모니터를 중복 실행하면 같은 포트를 다투므로 이전 창을 종료한다.

```bash
bash tools/communication/monitor.sh /dev/ttyACM0 /dev/ttyACM1
```

Terminator 한 창의 왼쪽은 S3 송신, 오른쪽은 C3 USB 로그와 Wi-Fi 상태 보고를 표시한다. 오른쪽의 `[C3 USB · 표시 지연 가능]`는 실제 무선 수신 시각을 나타내지 않는다. `[C3→S3 Wi-Fi] ACK`와 `C3_PM`을 기준으로 전달과 절전을 확인한다. 로그 디렉터리는 실행 시 출력하며 `HEARING_MONITOR_LOG_DIR`로 지정할 수 있다. USB 포트가 재생성되면 포트 권한을 다시 적용해야 할 수 있다.

```bash
python3 tools/communication/analyze_log.py /실제/로그/디렉터리/s3.log --events 100
```

분석기는 실제 절전 성공 횟수가 처음 0보다 커진 뒤의 이벤트를 대상으로, 지정한 고유 이벤트 수와 ACK·실패·재전송·왕복 지연 및 절전 횟수 증가를 검사한다. 재부팅·재등록 실험 로그는 연속 전송 실험과 분리한다.

현재 결과: [2026-10-30 절전 검증](../../docs/test_results/low_power_2026-10-30.md).
구조와 최종 통합 순서: [시스템 아키텍처](../../docs/architecture/README.md).

## 재접속·미전달 이벤트 복구 변경

C3는 끊김 감지 즉시 재접속을 시도하고 IP 획득 시 전용 작업에서 HELLO를 보냅니다. S3는 heartbeat 만료·재등록에도 RAM의 대기 이벤트 1개를 유지하며 ACK 전에는 다음 TEST를 만들지 않습니다. S3 재부팅 보존과 여러 실제 알림의 큐는 미구현입니다.

실물 검증은 미수행입니다. C3를 35초 이상 분리한 뒤 복구해 같은 S3 세션·순번으로 ACK가 완료되는지, ACK 억제 후 복원 시 중복 처리 없이 ACK하는지 확인해야 합니다. C3 재부팅 시 RAM 중복 기록이 사라져 재처리될 수 있습니다. 기존 측정은 변경 전 펌웨어 기준입니다. attempts는 현재 묶음의 시도 수이므로 기존 분석기의 재전송 수는 여러 묶음의 누적 재전송 수로 해석하지 않습니다.
