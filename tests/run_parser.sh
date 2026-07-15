#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

make -C "$root" >/dev/null

for case in "$root"/tests/parser/*.sy; do
    "$root/compiler" --parse-only "$case"
done

"$root/compiler" --parse-only "$root/tests/mini/return_42.sy"

echo "parser tests passed"
