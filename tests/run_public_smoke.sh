#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
limit="${1:-20}"
cases_dir="$root/compiler2026/2026初赛RISCV赛道功能用例/functional"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-public-smoke"
mkdir -p "$tmp_dir"

cc="${RISCV_CC:-riscv64-linux-gnu-gcc}"
qemu="${QEMU_RISCV:-qemu-riscv64}"

command -v "$cc" >/dev/null
command -v "$qemu" >/dev/null

make -C "$root" >/dev/null
"$cc" -march=rv64gc -mcmodel=medany -static -c "$root/tests/runtime/sylib.c" -o "$tmp_dir/sylib.o"

passed=0
total=0

while IFS= read -r case_file; do
    total=$((total + 1))
    name="$(basename "${case_file%.sy}")"
    asm="$tmp_dir/$name.s"
    exe="$tmp_dir/$name"
    actual="$tmp_dir/$name.actual"
    stdout_file="$tmp_dir/$name.stdout"
    input_file="${case_file%.sy}.in"
    expected="${case_file%.sy}.out"

    "$root/compiler" "$case_file" -S -o "$asm"
    "$cc" -march=rv64gc -mcmodel=medany -static "$asm" "$tmp_dir/sylib.o" -o "$exe"

    set +e
    if [[ -f "$input_file" ]]; then
        "$qemu" "$exe" <"$input_file" >"$stdout_file"
    else
        "$qemu" "$exe" >"$stdout_file"
    fi
    status=$?
    set -e

    cp "$stdout_file" "$actual"
    if [[ -s "$actual" ]] && [[ "$(tail -c 1 "$actual" | od -An -t u1 | tr -d ' ')" != "10" ]]; then
        printf '\n' >>"$actual"
    fi
    printf '%s\n' "$status" >>"$actual"

    if ! diff -u "$expected" "$actual"; then
        echo "public smoke failed: $case_file" >&2
        exit 1
    fi
    passed=$((passed + 1))
done < <(find "$cases_dir" -maxdepth 1 -name '*.sy' | sort | head -n "$limit")

echo "public smoke passed: $passed/$total"
