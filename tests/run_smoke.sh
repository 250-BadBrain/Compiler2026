#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

make -C "$root"
"$root/compiler" "$root/tests/mini/return_42.sy" -S -o "$root/out_return_42.s"
grep -q $'\t.arch armv8-a' "$root/out_return_42.s"
grep -Eq $'\tmovz w0, #42|\tmov w0, #42' "$root/out_return_42.s"

echo "smoke tests passed"
