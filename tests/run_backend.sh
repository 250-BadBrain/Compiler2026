#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-backend-test"
mkdir -p "$tmp_dir"

make -C "$root" >/dev/null

"$root/compiler" "$root/tests/backend/int_arith.sy" -S -o "$tmp_dir/int_arith.s"
grep -q 'movw r0, #16' "$tmp_dir/int_arith.s"
grep -q 'pop {fp, pc}' "$tmp_dir/int_arith.s"

"$root/compiler" "$root/tests/backend/control.sy" -S -o "$tmp_dir/control.s"
grep -q '.Larm.main.while.cond' "$tmp_dir/control.s"
grep -Eq 'bne |beq |blt |bge |b ' "$tmp_dir/control.s"

"$root/compiler" "$root/tests/backend/call.sy" -S -o "$tmp_dir/call.s"
grep -Eq 'bl add|movw r0, #3' "$tmp_dir/call.s"

"$root/compiler" "$root/tests/backend/array1.sy" -S -o "$tmp_dir/array1.s"
grep -Eq 'str r[0-9]+, \[r[0-9]+\]' "$tmp_dir/array1.s"
grep -Eq 'add r[0-9]+, r[0-9]+, r1, lsl #2|sub r[0-9]+, fp, r[0-9]+' "$tmp_dir/array1.s"

"$root/compiler" "$root/tests/backend/global_array.sy" -S -o "$tmp_dir/global_array.s"
grep -q '^g:' "$tmp_dir/global_array.s"
grep -Eq 'movw r[0-9]+, #:lower16:g' "$tmp_dir/global_array.s"

echo "backend tests passed"
