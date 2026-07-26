#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 <compiler>" >&2
    exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compiler="$1"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-register-allocation-native"
mkdir -p "$tmp_dir"

asm="$tmp_dir/register_allocation.s"
exe="$tmp_dir/register_allocation"

"$compiler" "$root/tests/backend/register_allocation.sy" -S -o "$asm"
gcc -static -march=armv8-a "$asm" "$root/tests/runtime/sylib.c" -o "$exe"

set +e
"$exe" >/dev/null
status=$?
set -e

if [[ "$status" -ne 72 ]]; then
    echo "register allocation native test failed: expected status 72, got $status" >&2
    exit 1
fi

echo "register allocation native test passed"
