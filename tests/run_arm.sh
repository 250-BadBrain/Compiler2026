#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-aarch64-test"
mkdir -p "$tmp_dir"

cc="${AARCH64_CC:-aarch64-linux-gnu-gcc}"
qemu="${QEMU_AARCH64:-qemu-aarch64}"

if ! command -v "$cc" >/dev/null; then
    echo "missing AArch64 compiler: $cc" >&2
    exit 1
fi
if ! command -v "$qemu" >/dev/null; then
    echo "missing AArch64 qemu: $qemu" >&2
    exit 1
fi

make -C "$root" >/dev/null

run_case() {
    local src="$1"
    local expected="$2"
    local name
    name="$(basename "${src%.sy}")"
    local asm="$tmp_dir/$name.s"
    local exe="$tmp_dir/$name"

    "$root/compiler" "$src" -S -o "$asm"
    "$cc" -static -march=armv8-a "$asm" -o "$exe"
    set +e
    "$qemu" -L /usr/aarch64-linux-gnu "$exe" >"$tmp_dir/$name.stdout" 2>"$tmp_dir/$name.stderr"
    local status=$?
    set -e
    if [[ "$status" != "$expected" ]]; then
        echo "unexpected exit for $src: got $status expected $expected" >&2
        sed -n '1,160p' "$asm" >&2
        cat "$tmp_dir/$name.stderr" >&2
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

echo "aarch64 execution tests passed"
