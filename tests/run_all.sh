#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$root/tests/run_smoke.sh"
"$root/tests/run_lexer.sh"
"$root/tests/run_parser.sh"
"$root/tests/run_sema.sh"
"$root/tests/run_ir.sh"
"$root/tests/run_backend.sh"
"$root/tests/run_functional.sh"

if command -v aarch64-linux-gnu-gcc >/dev/null && command -v qemu-aarch64 >/dev/null; then
    "$root/tests/run_arm.sh"
    "$root/tests/run_runtime.sh"
    "$root/tests/run_public_smoke.sh" 94
else
    echo "skip aarch64 execution tests: missing aarch64-linux-gnu-gcc or qemu-aarch64"
fi

echo "all tests passed"
