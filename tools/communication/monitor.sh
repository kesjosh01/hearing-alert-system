#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HEARING_PROJECT_ROOT=$(cd -- "$script_dir/../.." && pwd)
export HEARING_IDF_PATH=${IDF_PATH:-"$HOME/esp/esp-idf"}
export HEARING_S3_PORT=${1:-/dev/ttyACM0}
export HEARING_C3_PORT=${2:-/dev/ttyACM1}
export HEARING_MONITOR_LOG_DIR=${HEARING_MONITOR_LOG_DIR:-$(mktemp -d /tmp/hearing-alert-monitor.XXXXXX)}
mkdir -p "$HEARING_MONITOR_LOG_DIR"
for port in "$HEARING_S3_PORT" "$HEARING_C3_PORT"; do
    if [[ ! -r "$port" || ! -w "$port" ]]; then
        printf '시리얼 포트 읽기/쓰기 권한 필요: %s\n' "$port" >&2
        printf 'PC 터미널에서 sudo setfacl -m u:%s:rw %q 실행 후 재시도하세요.\n' "$USER" "$port" >&2
        exit 1
    fi
done
command -v terminator >/dev/null
touch "$HEARING_MONITOR_LOG_DIR/s3.log" "$HEARING_MONITOR_LOG_DIR/c3.log"
cat > "$HEARING_MONITOR_LOG_DIR/terminator.conf" <<EOF
[global_config]
  suppress_multiple_term_dialog = True
[profiles]
  [[default]]
    use_system_font = False
    font = Monospace 10
    scrollback_lines = 10000
    scroll_on_output = True
[layouts]
  [[hearing]]
    [[[window0]]]
      type = Window
      parent = ""
      size = 1600, 850
    [[[split0]]]
      type = HPaned
      parent = window0
      position = 800
    [[[s3]]]
      type = Terminal
      parent = split0
      order = 0
      title = S3 송신 · TX / ACK
      command = bash "$script_dir/monitor-pane.sh" s3
    [[[c3]]]
      type = Terminal
      parent = split0
      order = 1
      title = C3 수신·절전 · USB + Wi-Fi 보고
      command = bash "$script_dir/monitor-pane.sh" c3
[plugins]
EOF
printf '로그 디렉터리: %s\n' "$HEARING_MONITOR_LOG_DIR"
exec terminator --no-dbus --config="$HEARING_MONITOR_LOG_DIR/terminator.conf" \
    --layout=hearing --title='Hearing Alert · SoftAP 절전 통신' --maximise
