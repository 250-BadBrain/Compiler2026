#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-ir-test"
mkdir -p "$tmp_dir"

make -C "$root" >/dev/null

"$root/compiler" --dump-ir "$root/tests/mini/return_42.sy" >"$tmp_dir/return_42.ir"
grep -q 'func @main()' "$tmp_dir/return_42.ir"
grep -q 'ret 42' "$tmp_dir/return_42.ir"

"$root/compiler" --dump-ir "$root/tests/parser/control.sy" >"$tmp_dir/control.ir"
grep -q 'while.cond' "$tmp_dir/control.ir"
grep -q 'condbr' "$tmp_dir/control.ir"
grep -q 'br while.end' "$tmp_dir/control.ir"

"$root/compiler" --dump-ir "$root/tests/ir/call.sy" >"$tmp_dir/call.ir"
grep -q 'func @add' "$tmp_dir/call.ir"
grep -q 'call add' "$tmp_dir/call.ir"

"$root/compiler" --dump-ir "$root/tests/ir/global.sy" >"$tmp_dir/global.ir"
grep -q 'const @N' "$tmp_dir/global.ir"
grep -q 'global @g' "$tmp_dir/global.ir"
grep -q 'gep @g' "$tmp_dir/global.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/store_forwarding.sy" >"$tmp_dir/store_forwarding.ir"
grep -q 'func @main' "$tmp_dir/store_forwarding.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/exact_gep_store_forwarding.sy" >"$tmp_dir/exact_gep_store_forwarding.ir"
grep -q 'ret 7' "$tmp_dir/exact_gep_store_forwarding.ir"
! grep -q ' = load ' "$tmp_dir/exact_gep_store_forwarding.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/const_global_load_fold.sy" >"$tmp_dir/const_global_load_fold.ir"
grep -q 'ret 21' "$tmp_dir/const_global_load_fold.ir"
! grep -q ' = load ' "$tmp_dir/const_global_load_fold.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/readonly_global_load_fold.sy" >"$tmp_dir/readonly_global_load_fold.ir"
grep -q 'ret 12' "$tmp_dir/readonly_global_load_fold.ir"
! grep -q ' = load ' "$tmp_dir/readonly_global_load_fold.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/written_global_load_not_fold.sy" >"$tmp_dir/written_global_load_not_fold.ir"
grep -q ' = load ' "$tmp_dir/written_global_load_not_fold.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/dynamic_global_write_not_fold.sy" >"$tmp_dir/dynamic_global_write_not_fold.ir"
grep -q ' = load ' "$tmp_dir/dynamic_global_write_not_fold.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/nonclobber_call_load_forward.sy" >"$tmp_dir/nonclobber_call_load_forward.ir"
test "$(grep -c ' = load ' "$tmp_dir/nonclobber_call_load_forward.ir")" -le 1

"$root/compiler" -O1 --dump-ir "$root/tests/ir/clobber_call_preserves_load.sy" >"$tmp_dir/clobber_call_preserves_load.ir"
test "$(grep -c ' = load ' "$tmp_dir/clobber_call_preserves_load.ir")" -ge 1

"$root/compiler" -O1 --dump-ir "$root/tests/ir/local_dead_store.sy" >"$tmp_dir/local_dead_store.ir"
grep -q 'store 2 @g' "$tmp_dir/local_dead_store.ir"
! grep -q 'store 1 @g' "$tmp_dir/local_dead_store.ir"
grep -q 'ret 2' "$tmp_dir/local_dead_store.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/local_dead_gep_store.sy" >"$tmp_dir/local_dead_gep_store.ir"
! grep -q 'store 1 ' "$tmp_dir/local_dead_gep_store.ir"
grep -q 'store 2 ' "$tmp_dir/local_dead_gep_store.ir"
grep -q 'store 3 ' "$tmp_dir/local_dead_gep_store.ir"
grep -q 'ret 2' "$tmp_dir/local_dead_gep_store.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/boolean_negation_canonical.sy" >"$tmp_dir/boolean_negation_canonical.ir"
grep -q 'icmp ge' "$tmp_dir/boolean_negation_canonical.ir"
grep -q 'icmp ne' "$tmp_dir/boolean_negation_canonical.ir"
! grep -q ' = not ' "$tmp_dir/boolean_negation_canonical.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/boolean_return_branch.sy" >"$tmp_dir/boolean_return_branch.ir"
grep -q 'ret %' "$tmp_dir/boolean_return_branch.ir"
! grep -q 'condbr' "$tmp_dir/boolean_return_branch.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/boolean_inverse_return_branch.sy" >"$tmp_dir/boolean_inverse_return_branch.ir"
grep -q 'icmp le' "$tmp_dir/boolean_inverse_return_branch.ir"
grep -q 'ret %' "$tmp_dir/boolean_inverse_return_branch.ir"
! grep -q 'condbr' "$tmp_dir/boolean_inverse_return_branch.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/linear_i32_simplify.sy" >"$tmp_dir/linear_i32_simplify.ir"
grep -q 'ret 0' "$tmp_dir/linear_i32_simplify.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/linear_negate_value.sy" >"$tmp_dir/linear_negate_value.ir"
grep -q ' = neg ' "$tmp_dir/linear_negate_value.ir"
! grep -q 'sub 0' "$tmp_dir/linear_negate_value.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/cfg_merge.sy" >"$tmp_dir/cfg_merge.ir"
grep -q 'ret 3' "$tmp_dir/cfg_merge.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/empty_jump_block.sy" >"$tmp_dir/empty_jump_block.ir"
grep -q 'ret 7' "$tmp_dir/empty_jump_block.ir"
! grep -q 'br if.end' "$tmp_dir/empty_jump_block.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/address_gvn.sy" >"$tmp_dir/address_gvn.ir"
grep -q 'gep @data' "$tmp_dir/address_gvn.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/global_div_gvn.sy" >"$tmp_dir/global_div_gvn.ir"
test "$(grep -c ' = div ' "$tmp_dir/global_div_gvn.ir")" -eq 1

"$root/compiler" -O1 --dump-ir "$root/tests/ir/licm_div.sy" >"$tmp_dir/licm_div.ir"
div_line="$(grep -n ' = div ' "$tmp_dir/licm_div.ir" | cut -d: -f1)"
loop_line="$(grep -n '^while.body' "$tmp_dir/licm_div.ir" | cut -d: -f1)"
test "$div_line" -lt "$loop_line"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/load_cse.sy" >"$tmp_dir/load_cse.ir"
grep -q ' = load ' "$tmp_dir/load_cse.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/counted_loop_side_effect.sy" >"$tmp_dir/counted_loop_side_effect.ir"
grep -q 'func @main' "$tmp_dir/counted_loop_side_effect.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/alias_store_invalidation.sy" >"$tmp_dir/alias_store_invalidation.ir"
grep -q 'func @read_after_possible_alias' "$tmp_dir/alias_store_invalidation.ir"

echo "ir tests passed"
