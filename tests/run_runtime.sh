#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-runtime-test"
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

"$root/compiler" "$root/tests/runtime/int_io.sy" -S -o "$tmp_dir/int_io.s"
"$cc" -static "$tmp_dir/int_io.s" "$tmp_dir/sylib.o" -o "$tmp_dir/int_io"

set +e
printf '7 5\n' | "$qemu" "$tmp_dir/int_io" >"$tmp_dir/int_io.out" 2>"$tmp_dir/int_io.err"
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
