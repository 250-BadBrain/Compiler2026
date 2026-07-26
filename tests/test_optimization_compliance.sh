#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend="$root/src/backend/arm/emit.cpp"
compiler_sources="$root/src"

forbidden_case_names='"(h-[0-9]|crypto-[0-9]|conv2d-[0-9]|fft[0-9]|matmul[0-9]|shuffle[0-9]|many_mat_cal|optimization_scheduling|knapsack_naive|huffman-[0-9]|transpose[0-9]|sl[0-9])'
if rg -n "$forbidden_case_names" "$root/src"; then
    echo "forbidden public case name in compiler source" >&2
    exit 1
fi

if rg -n 'function\.name[[:space:]]*(==|!=)[[:space:]]*"[^\"]+"' "$compiler_sources" | rg -v '"main"'; then
    echo "optimization matcher depends on a literal function name" >&2
    exit 1
fi

if rg -n 'findFunction\("[^\"]+"\)' "$compiler_sources" | rg -v 'findFunction\("main"\)'; then
    echo "optimization matcher looks up a literal function name" >&2
    exit 1
fi

if rg -n 'global\.name[[:space:]]*(==|!=)[[:space:]]*"[^\"]+"' "$compiler_sources"; then
    echo "optimization matcher depends on a literal global name" >&2
    exit 1
fi

if rg -n '(2026初赛|性能用例|功能用例|knapsack_naive|many_mat_cal|optimization_scheduling)' "$root/src"; then
    echo "compiler source references contest artifacts or public case identifiers" >&2
    exit 1
fi

if rg -n '\b(getenv|uname|curl|wget|ssh|scp)\b|/proc|\bhidden\b|\bjudge\b|expected[[:space:]]+output|直接输出|答案' "$root/src"; then
    echo "compiler source contains environment probing or output-cheating indicators" >&2
    exit 1
fi

echo "optimization compliance scan passed"
