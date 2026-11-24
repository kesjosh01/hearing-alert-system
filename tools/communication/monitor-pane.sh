#!/usr/bin/env bash
set -uo pipefail
role=$1
source "$HEARING_IDF_PATH/export.sh" >/dev/null 2>&1
# HEARING_FIRMWARE=product selects firmware/{transmitter_s3,receiver_c6} (XIAO ESP32-C6
# receiver); default: TEST experiment with the XIAO ESP32-C6 receiver (build/c6);
# HEARING_TEST_RX=c3 selects the previous C3 SuperMini build (build/c3). The receiver
# role/port keep the historical "c3" name (HEARING_C3_PORT, c3.log) so commands still work.
if [[ "${HEARING_FIRMWARE:-test}" = product ]]; then
    rx_target=esp32c6
    if [[ "$role" = s3 ]]; then elf="$HEARING_PROJECT_ROOT/firmware/transmitter_s3/build/hearing_alert_s3.elf"
    else elf="$HEARING_PROJECT_ROOT/firmware/receiver_c6/build/hearing_alert_c6.elf"; fi
else
    test_rx=${HEARING_TEST_RX:-c6}
    rx_target=esp32$test_rx
    build_role=$role; [[ "$role" = s3 ]] || build_role=$test_rx
    elf="$HEARING_PROJECT_ROOT/communication_tests/softap_udp/build/$build_role/hearing_softap_test.elf"
fi
control="$HEARING_MONITOR_LOG_DIR/$role.control"
[[ -p "$control" ]] || mkfifo "$control"
# Keep the command pipe open; tests can request a reset without closing the GUI.
exec 3<>"$control"
if [[ "$role" = s3 ]]; then
    printf '\033[1;36mS3 송신 | TX → 수신부 처리 → ACK | C3_PM = 실제 Light-sleep 성공 횟수\033[0m\n'
    python -m esp_idf_monitor --no-reset --timestamps \
        --target esp32s3 -p "$HEARING_S3_PORT" -b 115200 "$elf" <&3 2>&1 | tee "$HEARING_MONITOR_LOG_DIR/s3.log"
else
    printf '\033[1;32m수신부 수신·절전 | USB 원본 + S3가 받은 Wi-Fi 상태 보고\033[0m\n'
    printf '[수신부 USB]는 늦게 표시될 수 있습니다. [수신부→S3 Wi-Fi]의 ACK가 실제 무선 응답입니다.\n'
    printf 'ACK는 S3에 도착한 소프트웨어 처리 확인이며, LED·진동 동작 확인은 아닙니다.\n'
    python -m esp_idf_monitor --no-reset --timestamps \
        --target "$rx_target" -p "$HEARING_C3_PORT" -b 115200 "$elf" \
        <&3 > "$HEARING_MONITOR_LOG_DIR/c3.log" 2>&1 &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    tail -n 0 -F "$HEARING_MONITOR_LOG_DIR/c3.log" "$HEARING_MONITOR_LOG_DIR/s3.log" \
        | stdbuf -oL awk -v s3="$HEARING_MONITOR_LOG_DIR/s3.log" '
            /^==> / { source = index($0, s3) ? "[수신부→S3 Wi-Fi] " : "[수신부 USB · 표시 지연 가능] "; next }
            /TEST event|EVENT accepted|EVENT dup|EVENT stale|NOT accepted|EVENT dropped|DIAG_CONSUMER|STA got IP|auto Light-sleep|auto light-sleep|C3_PM|C3_STATE|ACK seq=|EXPIRED|EVICTED|FAILED|ERROR|Waiting for|disconnected reason/ {
                print source $0; fflush();
            }'
fi
