#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-backend-test"
mkdir -p "$tmp_dir"

make -C "$root" >/dev/null

"$root/compiler" "$root/tests/backend/int_arith.sy" -S -o "$tmp_dir/int_arith.s"
grep -q $'\t.arch armv8-a' "$tmp_dir/int_arith.s"
grep -Eq $'\tmovz w0, #16|\tmov w0, #16' "$tmp_dir/int_arith.s"
grep -q $'\tret' "$tmp_dir/int_arith.s"

"$root/compiler" "$root/tests/backend/control.sy" -S -o "$tmp_dir/control.s"
grep -q '.La64.main.while.cond' "$tmp_dir/control.s"
grep -Eq $'\tb\\.(ne|eq|lt|le|gt|ge)|\tb ' "$tmp_dir/control.s"

"$root/compiler" "$root/tests/backend/call.sy" -S -o "$tmp_dir/call.s"
grep -Eq $'\tbl add|\tmovz w0, #3|\tmov w0, #3' "$tmp_dir/call.s"

"$root/compiler" "$root/tests/backend/array1.sy" -S -o "$tmp_dir/array1.s"
grep -Eq 'str w[0-9]+, \[x[0-9]+\]' "$tmp_dir/array1.s"
grep -Eq 'add x[0-9]+, x[0-9]+, x[0-9]+, lsl #2|sub x[0-9]+, x29, #[0-9]+' "$tmp_dir/array1.s"

"$root/compiler" "$root/tests/backend/global_array.sy" -S -o "$tmp_dir/global_array.s"
grep -q '^g:' "$tmp_dir/global_array.s"
grep -Eq 'adrp x[0-9]+, g|adr x[0-9]+, g' "$tmp_dir/global_array.s"

"$root/compiler" "$root/tests/backend/branch_frame_select.sy" -S -o "$tmp_dir/branch_frame_select.s"
grep -q '^branch_nonzero:' "$tmp_dir/branch_frame_select.s"

"$root/compiler" "$root/tests/backend/signed_constant_division.sy" -S -o "$tmp_dir/signed_constant_division.s"
grep -q '^divide_by_two:' "$tmp_dir/signed_constant_division.s"

echo "backend tests passed"
