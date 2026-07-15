#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-riscv-test"
mkdir -p "$tmp_dir"

cc="${RISCV_CC:-riscv64-linux-gnu-gcc}"
qemu="${QEMU_RISCV:-qemu-riscv64}"

command -v "$cc" >/dev/null
command -v "$qemu" >/dev/null

make -C "$root" >/dev/null

run_case() {
    local src="$1"
    local expected="$2"
    local name
    name="$(basename "${src%.sy}")"
    local asm="$tmp_dir/$name.s"
    local exe="$tmp_dir/$name"

    "$root/compiler" "$src" -S -o "$asm"
    "$cc" -march=rv64gc -mcmodel=medany -static "$asm" -o "$exe"
    set +e
    "$qemu" "$exe" >/tmp/compiler2026-riscv.stdout 2>/tmp/compiler2026-riscv.stderr
    local status=$?
    set -e
    if [[ "$status" != "$expected" ]]; then
        echo "unexpected exit for $src: got $status expected $expected" >&2
        echo "--- asm ---" >&2
        sed -n '1,160p' "$asm" >&2
        echo "--- stdout ---" >&2
        cat /tmp/compiler2026-riscv.stdout >&2
        echo "--- stderr ---" >&2
        cat /tmp/compiler2026-riscv.stderr >&2
        exit 1
    fi
}

run_case "$root/tests/mini/return_0.sy" 0
run_case "$root/tests/mini/return_42.sy" 42
run_case "$root/tests/backend/int_arith.sy" 16
run_case "$root/tests/backend/control.sy" 7
run_case "$root/tests/backend/call.sy" 3
run_case "$root/tests/backend/array1.sy" 7
run_case "$root/tests/backend/global_array.sy" 5
run_case "$root/tests/backend/array2.sy" 13
run_case "$root/tests/backend/global_array2.sy" 10
run_case "$root/tests/backend/array_param.sy" 10
run_case "$root/tests/backend/array_param2.sy" 6
run_case "$root/tests/backend/short_circuit.sy" 2

echo "riscv execution tests passed"
