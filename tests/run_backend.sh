#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-backend-test"
mkdir -p "$tmp_dir"

make -C "$root" >/dev/null

"$root/compiler" "$root/tests/backend/int_arith.sy" -S -o "$tmp_dir/int_arith.s"
grep -q 'li a0, 16' "$tmp_dir/int_arith.s"
grep -q 'ld ra' "$tmp_dir/int_arith.s"

"$root/compiler" "$root/tests/backend/control.sy" -S -o "$tmp_dir/control.s"
grep -q '.Lir.main.while.cond' "$tmp_dir/control.s"
grep -Eq 'bnez a0|beqz a0' "$tmp_dir/control.s"

"$root/compiler" "$root/tests/backend/call.sy" -S -o "$tmp_dir/call.s"
grep -q 'call add' "$tmp_dir/call.s"
grep -q 'sd a0' "$tmp_dir/call.s"
grep -Eq 'ld a1|mv a1' "$tmp_dir/call.s"

"$root/compiler" "$root/tests/backend/array1.sy" -S -o "$tmp_dir/array1.s"
grep -q 'sw t2, 0(a0)' "$tmp_dir/array1.s"
grep -q 'addi a0, t2, 4' "$tmp_dir/array1.s"

"$root/compiler" "$root/tests/backend/global_array.sy" -S -o "$tmp_dir/global_array.s"
grep -q '^g:' "$tmp_dir/global_array.s"
grep -q 'la t2, g' "$tmp_dir/global_array.s"

echo "backend tests passed"
