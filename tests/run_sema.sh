#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

make -C "$root" >/dev/null

"$root/compiler" --sema-only "$root/tests/sema/valid.sy"

for case in "$root"/tests/sema/invalid_*.sy; do
    if "$root/compiler" --sema-only "$case" >/tmp/compiler2026-sema.out 2>/tmp/compiler2026-sema.err; then
        echo "expected semantic failure for $case" >&2
        exit 1
    fi
done

echo "sema tests passed"
