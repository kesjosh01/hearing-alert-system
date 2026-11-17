# Hearing Alert 시스템 구조와 절전 통신

## 구성

제품 펌웨어는 두 개의 ESP-IDF 프로젝트입니다. `firmware/transmitter_s3/`(S3 SoftAP·송신)와 `firmware/receiver_c6/`(수신부 STA·화면·진동, 2026-11-17부터 Seeed XIAO ESP32-C6)가 공통 코어 `firmware/common/ha_comm/`을 함께 씁니다. `communication_tests/softap_udp/`는 같은 소스를 S3/C3로 빌드하던 TEST 실험 프로젝트이며, 과거 기록·회귀 참고용으로 보존합니다. 제품 펌웨어는 이 실험 코드를 포함하지 않습니다.

```text
S3 (LILYGO T7 S3, 상시 동작)                         수신부 (XIAO ESP32-C6, 연결 유지 + 절전)
┌───────────────────────────────────┐               ┌─────────────────────────────────────────┐
│ 마이크 → 추론 → 판정   (후속 통합) │               │ ha_receiver: 검증 · 중복 확인 · UI 큐    │
│        │ ha_publish_event(id)     │  EVENT v1 16B │        │                                 │
│        ▼                          │ ────────────▶ │        ▼                                 │
│ ha_sender: 대기열 · 반복 송신      │  (반복 송신)   │ ha_ui: 우선순위 → 진동(ha_haptic)        │
│            ACK 세 필드 검증        │               │                  → 화면(ha_display, LCD) │
│                                   │ ◀──────────── │        │ 화면을 LCD로 보낸 뒤             │
│                                   │      ACK      │        ▼                                 │
│                                   │               │ ha_receiver_alert_displayed() → ACK      │
└───────────────────────────────────┘               │ ui_guard: 화면이 안 뜨면 재초기화·재부팅  │
                                                    └─────────────────────────────────────────┘
```

S3 쪽 마이크·추론·판정은 아직 연결하지 않았습니다([모델 연결 안내](../ml/MODEL_HANDOFF.md)). 규격과 정책은 [통신 프로토콜](../communication/README.md)에 있습니다.

## 절전의 정확한 의미

- S3는 AP와 향후 소리 감지를 유지합니다. 절전 적용 대상은 수신부입니다.
- 수신부는 `WIFI_PS_MIN_MODEM`과 동적 CPU 주파수 40~160MHz, FreeRTOS tickless idle, 자동 Light-sleep을 사용합니다.
- S3는 100 TU(약 102.4ms) 비콘, DTIM 3으로 설정합니다. MIN_MODEM 수신부의 명목 수신 간격은 307.2ms입니다(2026-11-10 회의의 약 300ms 목표, 정확히 300ms는 설정할 수 없음). TEST 실험(2026-10-30)은 DTIM 1이었습니다.
- 수신부는 주기적으로 깨어나 Wi-Fi 연결을 유지하며, AP는 절전 STA에 대한 UDP 유니캐스트를 버퍼링합니다. 전파 한 번으로 완전히 꺼진 칩을 깨우는 방식은 아닙니다.
- 이벤트가 오면 CPU가 처리하고, 할 일이 없으면 다시 자동 절전합니다. 화면 깜빡임·진동 중에도 그 사이사이 절전할 수 있으며, 백라이트 전류는 별도입니다.
- Deep-sleep은 사용하지 않습니다. Wi-Fi가 완전히 꺼진 Deep-sleep 중에는 UDP 패킷으로 즉시 깨울 수 없습니다.
- 부팅 후 15초까지는 CPU 자동 Light-sleep을 미뤄 업로드·부팅 로그 확인 시간을 둡니다. Wi-Fi Modem-sleep은 접속 후부터 사용합니다.
- 수신부의 HELLO는 10초 간격이며 S3의 응답 없음 판단은 35초입니다. 주소·미전달 이벤트는 유지하고 재전송합니다. 수신부는 끊김 감지 즉시 재접속을 시도하고 IP를 받으면 HELLO를 바로 보냅니다. 낮은 전력 소비와 빠른 단절 감지 사이의 초기 절충값입니다.

## 절전 확인 방법

수신부는 ESP-IDF 공개 API `esp_pm_dump_locks()`의 프로파일링 결과에서 Light-sleep 성공/거절 횟수를 읽어 S3에 보고합니다. S3 로그의 `C3_STATE ... auto_light_sleep=1`은 설정 확인이고 `C3_PM ... sleep_count=N`의 증가가 실제 수면 진입 성공의 증거입니다(로그 이름의 "C3"는 수신부를 뜻함). 전류 감소율·배터리 수명은 전류계로 별도 측정해야 합니다.

내장 USB Serial/JTAG는 Light-sleep 중 정상 응답하지 않을 수 있습니다. 절전 검증 빌드는 USB 연결에 의한 절전 차단을 사용하지 않으므로, USB 로그가 멈춰도 S3가 받은 무선 상태 보고·ACK로 확인합니다([모니터 도구](../../tools/communication/README.md)).

## 남은 통합

1. S3 마이크·추론·판정을 `ha_publish_event()`에 연결합니다.
2. 실제 알림 부하(화면·진동)와 배터리 전원에서 지연·소비전류·재접속을 실기기로 검증합니다([C6 시험 절차](../test_results/receiver_c6_2026-11-17.md#실기기-시험-절차-미수행)).

참고: [Espressif Wi-Fi 저전력 가이드](https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32c6/api-guides/low-power-mode/low-power-mode-wifi.html), [USB Serial/JTAG의 절전 제한](https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32c6/api-guides/usb-serial-jtag-console.html).
