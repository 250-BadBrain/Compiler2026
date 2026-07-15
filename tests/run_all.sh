#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$root/tests/run_smoke.sh"
"$root/tests/run_lexer.sh"
"$root/tests/run_parser.sh"
"$root/tests/run_sema.sh"
"$root/tests/run_ir.sh"
"$root/tests/run_backend.sh"
"$root/tests/run_riscv.sh"
"$root/tests/run_runtime.sh"
"$root/tests/run_public_smoke.sh" 94
"$root/tests/run_functional.sh"

echo "all tests passed"
