#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-structural-optimizations"
mkdir -p "$tmp_dir"

make -C "$root" >/dev/null

perf_dir="$root/compiler2026/2026初赛ARM赛道性能用例/performance"
"$root/compiler" "$perf_dir/many_mat_cal-1.sy" -S -o "$tmp_dir/many_mat_cal-1.s"

grep -q '\.many\.' "$tmp_dir/many_mat_cal-1.s"

"$root/compiler" "$root/tests/backend/modular_affine_reduction.sy" -S -o "$tmp_dir/modular_affine_reduction.s"
grep -q '\.h4\.loop' "$tmp_dir/modular_affine_reduction.s"
grep -q $'\tmov w9, #0' "$tmp_dir/modular_affine_reduction.s"
grep -q $'\tmov w10, #0' "$tmp_dir/modular_affine_reduction.s"
grep -q $'\tmov w11, #0' "$tmp_dir/modular_affine_reduction.s"
grep -q $'\tadd w22, w22, w9' "$tmp_dir/modular_affine_reduction.s"
grep -q $'\tadd w22, w22, w10' "$tmp_dir/modular_affine_reduction.s"
grep -q $'\tadd w22, w22, w11' "$tmp_dir/modular_affine_reduction.s"
accumulator_reductions="$(grep -c $'\tsmull x0, w22, w17' "$tmp_dir/modular_affine_reduction.s")"
if (( accumulator_reductions > 8 )); then
    echo "modular affine reduction still reduces the main accumulator inside every unrolled step" >&2
    exit 1
fi

echo "restored structural optimization tests passed"
