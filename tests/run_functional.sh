#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cases_dir="${1:-$root/compiler2026/2026初赛ARM赛道功能用例}"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-functional-test"
mkdir -p "$tmp_dir"

make -C "$root" >/dev/null

if [[ ! -d "$cases_dir" ]]; then
    echo "case directory does not exist: $cases_dir" >&2
    exit 1
fi
if [[ -z "$(find "$cases_dir" -type f -name '*.sy' -print -quit)" ]]; then
    echo "case directory contains no .sy files: $cases_dir" >&2
    exit 1
fi

total=0
passed=0

while IFS= read -r -d '' case_file; do
    total=$((total + 1))
    asm_file="$tmp_dir/$(basename "${case_file%.sy}").s"
    if "$root/compiler" "$case_file" -S -o "$asm_file" >/tmp/compiler2026-functional.out 2>/tmp/compiler2026-functional.err; then
        passed=$((passed + 1))
    else
        echo "compile failed: $case_file" >&2
        cat /tmp/compiler2026-functional.err >&2
        exit 1
    fi
done < <(find "$cases_dir" -type f -name '*.sy' -print0 | sort -z)

echo "functional asm generation passed: $passed/$total"
