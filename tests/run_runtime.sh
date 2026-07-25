#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-runtime-test"
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

"$cc" -static -march=armv8-a -c "$root/tests/runtime/sylib.c" -o "$tmp_dir/sylib.o"

"$root/compiler" "$root/tests/runtime/int_io.sy" -S -o "$tmp_dir/int_io.s"
"$cc" -static -march=armv8-a "$tmp_dir/int_io.s" "$tmp_dir/sylib.o" -o "$tmp_dir/int_io"

set +e
printf '7 5\n' | "$qemu" -L /usr/aarch64-linux-gnu "$tmp_dir/int_io" >"$tmp_dir/int_io.out" 2>"$tmp_dir/int_io.err"
status=$?
set -e

if [[ "$status" != 2 ]]; then
    echo "unexpected exit for int_io: got $status expected 2" >&2
    cat "$tmp_dir/int_io.err" >&2
    exit 1
fi

if [[ "$(cat "$tmp_dir/int_io.out")" != "12" ]]; then
    echo "unexpected stdout for int_io" >&2
    cat "$tmp_dir/int_io.out" >&2
    exit 1
fi

echo "runtime tests passed"
