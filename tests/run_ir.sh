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

echo "ir tests passed"
