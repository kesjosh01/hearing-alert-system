#!/usr/bin/env bash
# Build and flash the product firmware using the workspace's ESP-IDF projects.
set -eo pipefail

usage() {
    cat <<'HELP'
제품 펌웨어 (저장소 루트에서 실행)
  ./tools/flash.sh                         S3와 C6 자동 식별 후 두 보드 플래시
  ./tools/flash.sh s3|c6 [포트]             한 보드만 플래시 (포트 생략 시 자동 검색)
  ./tools/flash.sh --build-only [both|s3|c6]   USB 없이 빌드
  ./tools/flash.sh --monitor s3|c6 포트      플래시 후 로그 열기 (종료 Ctrl+])
HELP
}

fail() { printf '%s\n' "$*" >&2; exit 1; }

action=flash
case ${1:-} in
    -h|--help) usage; exit 0 ;;
    --build-only) action=build; shift ;;
    --monitor) action=monitor; shift ;;
esac
target=${1:-both}
port_arg=${2:-}
case "$target" in
    both) boards=(s3 c6) ;;
    s3|c6) boards=("$target") ;;
    *) usage >&2; exit 2 ;;
esac
if (( $# > 2 )) || [[ -n $port_arg && ($target == both || $action == build) ]] ||
   [[ $action == monitor && -z $port_arg ]]; then
    usage >&2; exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
sdk=${IDF_PATH:-"$HOME/esp/esp-idf"}
[[ -r "$sdk/export.sh" ]] || fail "ESP-IDF의 설치 디렉터리를 IDF_PATH로 지정해 주세요."
source "$sdk/export.sh" >/dev/null
set -u

declare -A projects=([s3]=transmitter_s3 [c6]=receiver_c6)
declare -A ports=()
run_idf() {
    local board=$1
    shift
    idf.py -C "$root/firmware/${projects[$board]}" -B "$root/build/ml_$board" \
        -D "IDF_TARGET=esp32$board" "$@"
}

if [[ $action == monitor ]]; then
    run_idf "$target" -p "$port_arg" monitor --no-reset
    exit
fi

if [[ $action == flash ]]; then
    if [[ -n $port_arg ]]; then
        candidates=("$port_arg")
    else
        # Both product boards use the Espressif native USB Serial/JTAG interface.
        found=$(python - <<'PY'
from pathlib import Path
from serial.tools.list_ports import comports

by_id = {p.resolve(): p for p in sorted(Path("/dev/serial/by-id").glob("*"))}
for port in sorted(comports()):
    if (port.vid, port.pid) == (0x303A, 0x1001):
        print(by_id.get(Path(port.device).resolve(), port.device))
PY
)
        [[ -n $found ]] || fail "ESP USB 보드가 없습니다. 데이터 USB 케이블로 연결하거나 ./tools/flash.sh s3 포트 또는 c6 포트로 지정하세요."
        mapfile -t candidates <<< "$found"
    fi
    for port in "${candidates[@]}"; do
        printf '칩 확인 중  %s\n' "$port"
        if ! probe=$(python -m esptool --chip auto --port "$port" --connect-attempts 2 --no-stub read_mac 2>&1); then
            printf '%s\n' "$probe" >&2
            fail "$port 칩 확인 실패. 열린 시리얼 모니터, 포트 접근 권한 및 USB 연결을 확인하세요."
        fi
        case "$probe" in
            *'Chip is ESP32-S3 '*|*'Chip is ESP32-S3-PICO-'*) board=s3 ;;
            *'Chip is ESP32-C6 '*|*'Chip is ESP32-C6FH'*) board=c6 ;;
            *) printf '대상 외 칩  %s\n' "$port"; continue ;;
        esac
        if [[ $target != both && $board != "$target" ]]; then
            [[ -z $port_arg ]] || fail "$port 칩은 $board 입니다. 요청한 $target 보드의 포트를 지정하세요."
            continue
        fi
        [[ -z ${ports[$board]:-} ]] || fail "$board 보드가 여러 대입니다. ./tools/flash.sh $board 포트로 한 대씩 지정하세요."
        ports[$board]=$port
        printf '  %s → %s\n' "$board" "$port"
    done
    for board in "${boards[@]}"; do
        [[ -n ${ports[$board]:-} ]] || fail "요청한 $board 칩을 찾지 못했습니다. 단독 플래시는 ./tools/flash.sh s3 또는 c6로 실행하세요."
    done
fi

# Build every requested project before writing to the first board.
for board in "${boards[@]}"; do
    printf '\n[%s] 제품 펌웨어 빌드\n' "$board"
    run_idf "$board" build
done
if [[ $action == flash ]]; then
    for board in "${boards[@]}"; do
        printf '\n[%s] %s 플래시\n' "$board" "${ports[$board]}"
        run_idf "$board" -p "${ports[$board]}" flash
        printf '[%s] 플래시 완료\n' "$board"
    done
    printf '\n로그 확인 (저장소 루트에서 보드별 터미널로 실행)\n'
    for board in "${boards[@]}"; do
        printf './tools/flash.sh --monitor %s %q\n' "$board" "${ports[$board]}"
    done
fi
