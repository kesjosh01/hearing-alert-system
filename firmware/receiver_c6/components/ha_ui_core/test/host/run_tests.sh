#!/usr/bin/env bash
# Build and run the ha_ui_core host tests with the system C compiler.
# Usage: bash firmware/receiver_c6/components/ha_ui_core/test/host/run_tests.sh [preview_dir]
# With preview_dir, PPM images of every screen are written there.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
comp="$(cd "$here/../.." && pwd)"
comm="$(cd "$comp/../../../common/ha_comm" && pwd)"
out="${TMPDIR:-/tmp}/ha_ui_host_tests.$$"
trap 'rm -f "$out"' EXIT
"${CC:-cc}" -std=c11 -Wall -Wextra -Werror -Wconversion -Wno-sign-conversion \
    -fsanitize=address,undefined -g -O1 \
    -I"$comp/include" -I"$comm/include" \
    "$comp"/src/*.c "$comm/src/ha_events.c" "$here/test_ui_core.c" -lm -o "$out"
"$out" "$@"
