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
grep -Eq 'str w[0-9]+, \[x[0-9]+\]|stur w[0-9]+, \[x29, #-[0-9]+\]' "$tmp_dir/array1.s"
grep -Eq 'add x[0-9]+, x[0-9]+, x[0-9]+, lsl #2|sub x[0-9]+, x29, #[0-9]+|stur w[0-9]+, \[x29, #-[0-9]+\]' "$tmp_dir/array1.s"

"$root/compiler" "$root/tests/backend/global_array.sy" -S -o "$tmp_dir/global_array.s"
grep -q '^g:' "$tmp_dir/global_array.s"
grep -Eq 'adrp x[0-9]+, g|adr x[0-9]+, g' "$tmp_dir/global_array.s"

"$root/compiler" "$root/tests/backend/branch_frame_select.sy" -S -o "$tmp_dir/branch_frame_select.s"
grep -q '^branch_nonzero:' "$tmp_dir/branch_frame_select.s"

"$root/compiler" "$root/tests/backend/signed_constant_division.sy" -S -o "$tmp_dir/signed_constant_division.s"
grep -q '^divide_by_two:' "$tmp_dir/signed_constant_division.s"

"$root/compiler" "$root/tests/backend/constant_mul_branch.sy" -S -o "$tmp_dir/constant_mul_branch.s"
grep -q $'\tadd w0, w0, w0, lsl #4' "$tmp_dir/constant_mul_branch.s"
grep -q $'\tcbz w0,' "$tmp_dir/constant_mul_branch.s"

"$root/compiler" "$root/tests/backend/direct_call_return.sy" -S -o "$tmp_dir/direct_call_return.s"
grep -q '^wrapper:' "$tmp_dir/direct_call_return.s"
grep -q $'\tbl callee' "$tmp_dir/direct_call_return.s"
! awk '/^wrapper:/{inside=1; after=0; next} /^\\t.size wrapper/{inside=0} inside && /\tbl callee/{after=1; next} inside && after && /st(u)?r w0|ld(u)?r w0/' "$tmp_dir/direct_call_return.s" | grep -q .

"$root/compiler" "$root/tests/backend/direct_expr_return.sy" -S -o "$tmp_dir/direct_expr_return.s"
grep -q '^plus_five:' "$tmp_dir/direct_expr_return.s"
grep -q $'\tadd w0, w0, #5' "$tmp_dir/direct_expr_return.s"
! awk '/^plus_five:/{inside=1; after=0; next} /^\\t.size plus_five/{inside=0} inside && /\tadd w0, w0, #5/{after=1; next} inside && after && /st(u)?r w0|ld(u)?r w0/' "$tmp_dir/direct_expr_return.s" | grep -q .

perf_case="$root/compiler2026/2026初赛ARM赛道性能用例/performance/conv2d-1.sy"
if [[ -f "$perf_case" ]]; then
    "$root/compiler" "$perf_case" -S -o "$tmp_dir/conv2d-1.s"
    ! awk '
        /^	stur [wxsd][0-9]+, \[x29, #-?[0-9]+\]$/ {
            prev=$0
            slot=$0
            sub(/^.*\[x29, #/, "", slot)
            sub(/\].*$/, "", slot)
            next
        }
        /^	ldur [wxsd][0-9]+, \[x29, #-?[0-9]+\]$/ {
            loadSlot=$0
            sub(/^.*\[x29, #/, "", loadSlot)
            sub(/\].*$/, "", loadSlot)
            if (prev != "" && slot == loadSlot) {
                print prev
                print $0
                found=1
            }
        }
        { prev="" }
        END { exit found ? 0 : 1 }
    ' "$tmp_dir/conv2d-1.s" | grep -q .
fi

fft_case="$root/compiler2026/2026初赛ARM赛道性能用例/performance/fft0.sy"
if [[ -f "$fft_case" ]]; then
    "$root/compiler" "$fft_case" -S -o "$tmp_dir/fft0.s"
    grep -Eq $'\tcb(n)?z w[0-9]+,' "$tmp_dir/fft0.s"
fi

echo "backend tests passed"
