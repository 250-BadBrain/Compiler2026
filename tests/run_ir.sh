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

"$root/compiler" -O1 --dump-ir "$root/tests/ir/cfg_merge.sy" >"$tmp_dir/cfg_merge.ir"
grep -q 'ret 3' "$tmp_dir/cfg_merge.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/address_gvn.sy" >"$tmp_dir/address_gvn.ir"
grep -q 'gep @data' "$tmp_dir/address_gvn.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/load_cse.sy" >"$tmp_dir/load_cse.ir"
grep -q ' = load ' "$tmp_dir/load_cse.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/counted_loop_side_effect.sy" >"$tmp_dir/counted_loop_side_effect.ir"
grep -q 'func @main' "$tmp_dir/counted_loop_side_effect.ir"

"$root/compiler" -O1 --dump-ir "$root/tests/ir/alias_store_invalidation.sy" >"$tmp_dir/alias_store_invalidation.ir"
grep -q 'func @read_after_possible_alias' "$tmp_dir/alias_store_invalidation.ir"

echo "ir tests passed"
