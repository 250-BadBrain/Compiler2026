#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cases_dir="${1:-$root/compiler2026/2026初赛ARM赛道功能用例}"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-public-functional"
mkdir -p "$tmp_dir"

cc="${ARM_CC:-arm-linux-gnueabihf-gcc}"
qemu="${QEMU_ARM:-qemu-arm}"

if ! command -v "$cc" >/dev/null; then
    echo "missing ARM compiler: $cc" >&2
    exit 1
fi
if ! command -v "$qemu" >/dev/null; then
    echo "missing ARM qemu: $qemu" >&2
    exit 1
fi

make -C "$root" >/dev/null
"$cc" -static -c "$root/tests/runtime/sylib.c" -o "$tmp_dir/sylib.o"

passed=0
total=0

while IFS= read -r -d '' case_file; do
    total=$((total + 1))
    rel="${case_file#"$cases_dir"/}"
    name="${rel%.sy}"
    work="$tmp_dir/$name"
    mkdir -p "$(dirname "$work")"
    asm="$work.s"
    exe="$work"
    actual="$work.actual"
    stdout_file="$work.stdout"
    input_file="${case_file%.sy}.in"
    expected="${case_file%.sy}.out"

    "$root/compiler" "$case_file" -S -o "$asm"
    "$cc" -static "$asm" "$tmp_dir/sylib.o" -o "$exe"

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
    if [[ -s "$expected" ]] && [[ "$(tail -c 1 "$expected" | od -An -t u1 | tr -d ' ')" != "10" ]]; then
        truncate -s -1 "$actual"
    fi

    if ! diff -u "$expected" "$actual"; then
        echo "public functional failed: $case_file" >&2
        exit 1
    fi
    passed=$((passed + 1))
done < <(find "$cases_dir" -type f -name '*.sy' -print0 | sort -z)

echo "public functional passed: $passed/$total"
