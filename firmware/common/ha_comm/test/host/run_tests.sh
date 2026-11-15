#!/usr/bin/env bash
# Build and run the ha_comm host unit tests with the system C compiler.
# Usage (from anywhere): bash firmware/common/ha_comm/test/host/run_tests.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
comp="$(cd "$here/../.." && pwd)"
out="${TMPDIR:-/tmp}/ha_comm_host_tests.$$"
trap 'rm -f "$out"' EXIT
"${CC:-cc}" -std=c11 -Wall -Wextra -Werror -Wconversion -Wno-sign-conversion \
    -fsanitize=address,undefined -g -O1 \
    -I"$comp/include" \
    "$comp"/src/*.c "$here/test_ha_comm.c" -o "$out"
"$out"
