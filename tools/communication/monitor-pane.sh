#!/usr/bin/env bash
set -uo pipefail
role=$1
source "$HEARING_IDF_PATH/export.sh" >/dev/null 2>&1
elf="$HEARING_PROJECT_ROOT/communication_tests/softap_udp/build/$role/hearing_softap_test.elf"
control="$HEARING_MONITOR_LOG_DIR/$role.control"
[[ -p "$control" ]] || mkfifo "$control"
# Keep the command pipe open; tests can request a reset without closing the GUI.
exec 3<>"$control"
if [[ "$role" = s3 ]]; then
    printf '\033[1;36mS3 송신 | TX → C3 처리 → ACK | C3_PM = 실제 Light-sleep 성공 횟수\033[0m\n'
    python -m esp_idf_monitor --no-reset --timestamps \
        --target esp32s3 -p "$HEARING_S3_PORT" -b 115200 "$elf" <&3 2>&1 | tee "$HEARING_MONITOR_LOG_DIR/s3.log"
else
    printf '\033[1;32mC3 수신·절전 | USB 원본 + S3가 받은 Wi-Fi 상태 보고\033[0m\n'
    printf '[C3 USB]는 늦게 표시될 수 있습니다. [C3→S3 Wi-Fi]의 ACK가 실제 무선 응답입니다.\n'
    printf 'ACK는 S3에 도착한 소프트웨어 처리 확인이며, LED·진동 동작 확인은 아닙니다.\n'
    python -m esp_idf_monitor --no-reset --timestamps \
        --target esp32c3 -p "$HEARING_C3_PORT" -b 115200 "$elf" \
        <&3 > "$HEARING_MONITOR_LOG_DIR/c3.log" 2>&1 &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    tail -n 0 -F "$HEARING_MONITOR_LOG_DIR/c3.log" "$HEARING_MONITOR_LOG_DIR/s3.log" \
        | stdbuf -oL awk -v s3="$HEARING_MONITOR_LOG_DIR/s3.log" '
            /^==> / { source = index($0, s3) ? "[C3→S3 Wi-Fi] " : "[C3 USB · 표시 지연 가능] "; next }
            /TEST event|STA got IP|auto Light-sleep|C3_PM|C3_STATE|ACK seq=|FAILED|ERROR|Waiting for|disconnected reason/ {
                print source $0; fflush();
            }'
fi
