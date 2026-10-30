# 통신 모니터와 로그 분석

저장소 루트에서 실행합니다. ESP-IDF v5.5.5와 두 타깃의 빌드 결과가 필요하며, 모니터는 Terminator를 사용합니다.

```bash
bash tools/communication/monitor.sh /dev/ttyACM0 /dev/ttyACM1
python3 tools/communication/analyze_log.py docs/test_results/evidence/2026-10-30-continuous-s3.txt --events 100
```

- `monitor.sh`: 첫 인자는 S3, 둘째는 C3 포트입니다. `IDF_PATH`로 SDK 경로, `HEARING_MONITOR_LOG_DIR`로 로그 저장 경로를 지정할 수 있습니다. 기본 로그 경로는 `/tmp/hearing-alert-monitor.*`이며 보관할 로그는 별도로 복사합니다.
- `monitor-pane.sh`: 모니터 창 내부에서 실행하는 스크립트입니다. C3 USB 로그는 절전 중 멈출 수 있으므로 S3가 받은 ACK·상태 보고도 함께 표시합니다. 두 모니터는 시작 시 보드를 리셋하지 않습니다.
- `analyze_log.py`: 등록 이후 SDK 절전 성공 횟수가 처음 양수가 된 뒤부터 지정한 수의 고유 이벤트를 분석합니다. 첫 TX 시도가 없는 구간과 수집 시작 후 재등록·heartbeat 만료가 있는 로그는 거부하므로 실험별로 분리합니다.

분석 결과는 JSON이며, 지정한 이벤트 수만큼 ACK 수신, FAILED 없음, 단일 C3 세션, 절전 성공 횟수 증가를 충족하면 종료 코드 0, 그 외에는 1입니다. 재전송 수는 별도 지표이므로 `passed=true`만으로 재전송 0을 뜻하지 않습니다. 절전 카운터는 로그 전체의 보고를 사용하며 ACK 통계는 선택된 이벤트만 사용합니다.

이 분석은 C3 자체 재부팅 복구나 소비전류·배터리 수명을 검증하지 않습니다. 실물 검증 범위는 [2026-10-30 기록](../../docs/test_results/low_power_2026-10-30.md), 빌드·플래시 절차는 [통신 프로젝트](../../communication_tests/softap_udp/README.md)를 참고합니다.
