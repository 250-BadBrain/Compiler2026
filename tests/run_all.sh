#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

(cd "$root" && python3 -m unittest tests.test_perf_tools -v)
"$root/tests/test_harness_guards.sh"
bash "$root/tests/test_optimization_compliance.sh"
bash "$root/tests/test_restored_structural_optimizations.sh"
"$root/tests/run_smoke.sh"
"$root/tests/run_lexer.sh"
"$root/tests/run_parser.sh"
"$root/tests/run_sema.sh"
"$root/tests/run_ir.sh"
"$root/tests/run_backend.sh"
"$root/tests/test_register_allocation.sh"

public_functional="$root/compiler2026/2026初赛ARM赛道功能用例"
if [[ -d "$public_functional" ]] && [[ -n "$(find "$public_functional" -type f -name '*.sy' -print -quit)" ]]; then
    "$root/tests/run_functional.sh" "$public_functional"
else
    echo "skip public functional generation tests: cases not provided"
fi

if command -v aarch64-linux-gnu-gcc >/dev/null && command -v qemu-aarch64 >/dev/null; then
    "$root/tests/run_arm.sh"
    "$root/tests/run_runtime.sh"
    if [[ -d "$public_functional/functional" ]] &&
       [[ -n "$(find "$public_functional/functional" -maxdepth 1 -type f -name '*.sy' -print -quit)" ]]; then
        "$root/tests/run_public_smoke.sh" 94
    else
        echo "skip public AArch64 smoke tests: cases not provided"
    fi
else
    echo "skip aarch64 execution tests: missing aarch64-linux-gnu-gcc or qemu-aarch64"
fi

echo "all tests passed"
