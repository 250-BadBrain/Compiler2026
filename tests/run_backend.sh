#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-backend-test"
mkdir -p "$tmp_dir"

make -C "$root" >/dev/null

"$root/compiler" "$root/tests/backend/int_arith.sy" -S -o "$tmp_dir/int_arith.s"
grep -q 'mulw a0' "$tmp_dir/int_arith.s"
grep -q 'addiw a0' "$tmp_dir/int_arith.s"
grep -q 'ld ra' "$tmp_dir/int_arith.s"

"$root/compiler" "$root/tests/backend/control.sy" -S -o "$tmp_dir/control.s"
grep -q '.Lwhile.cond' "$tmp_dir/control.s"
grep -q 'bnez a0' "$tmp_dir/control.s"
grep -q 'beqz a0' "$tmp_dir/control.s"

"$root/compiler" "$root/tests/backend/call.sy" -S -o "$tmp_dir/call.s"
grep -q 'call add' "$tmp_dir/call.s"
grep -q 'sd a0' "$tmp_dir/call.s"
grep -q 'ld a1' "$tmp_dir/call.s"

"$root/compiler" "$root/tests/backend/array1.sy" -S -o "$tmp_dir/array1.s"
grep -q 'sw a0' "$tmp_dir/array1.s"
grep -q 'slli a0, a0, 2' "$tmp_dir/array1.s"

"$root/compiler" "$root/tests/backend/global_array.sy" -S -o "$tmp_dir/global_array.s"
grep -q '^g:' "$tmp_dir/global_array.s"
grep -q 'la t2, g' "$tmp_dir/global_array.s"

"$root/compiler" "$root/tests/backend/array2.sy" -S -o "$tmp_dir/array2.s"
grep -q 'li t0, 3' "$tmp_dir/array2.s"

"$root/compiler" "$root/tests/backend/global_array2.sy" -S -o "$tmp_dir/global_array2.s"
grep -q 'li t0, 3' "$tmp_dir/global_array2.s"

echo "backend tests passed"
