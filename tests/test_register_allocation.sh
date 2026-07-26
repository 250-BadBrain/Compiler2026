#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-register-allocation"
mkdir -p "$tmp_dir"

make -C "$root" >/dev/null
asm="$tmp_dir/register_allocation.s"
"$root/compiler" "$root/tests/backend/register_allocation.sy" -S -o "$asm"

grep -q '^allocate_values:' "$asm"
grep -q '^blend:' "$asm"
grep -q '^pressure_call:' "$asm"

echo "register allocation structural test passed"
