#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-structural-optimizations"
mkdir -p "$tmp_dir"

make -C "$root" >/dev/null

perf_dir="$root/compiler2026/2026初赛ARM赛道性能用例/performance"
"$root/compiler" "$perf_dir/many_mat_cal-1.sy" -S -o "$tmp_dir/many_mat_cal-1.s"

grep -q '\.many\.' "$tmp_dir/many_mat_cal-1.s"

"$root/compiler" "$root/tests/backend/matrix_extra_global.sy" -S -o "$tmp_dir/matrix_extra_global.s"
grep -q '\.many\.' "$tmp_dir/matrix_extra_global.s"

"$root/compiler" "$root/tests/backend/matrix_driver_wrapper.sy" -S -o "$tmp_dir/matrix_driver_wrapper.s"
grep -q '\.many\.' "$tmp_dir/matrix_driver_wrapper.s"

"$root/compiler" "$root/tests/backend/modular_affine_reduction.sy" -S -o "$tmp_dir/modular_affine_reduction.s"
grep -q '\.h4\.loop' "$tmp_dir/modular_affine_reduction.s"
grep -q $'\tmov w9, #0' "$tmp_dir/modular_affine_reduction.s"
grep -q $'\tmov w10, #0' "$tmp_dir/modular_affine_reduction.s"
grep -q $'\tmov w11, #0' "$tmp_dir/modular_affine_reduction.s"
grep -q $'\tadd w22, w22, w9' "$tmp_dir/modular_affine_reduction.s"
grep -q $'\tadd w22, w22, w10' "$tmp_dir/modular_affine_reduction.s"
grep -q $'\tadd w22, w22, w11' "$tmp_dir/modular_affine_reduction.s"
accumulator_reductions="$(grep -c $'\tsmull x0, w22, w17' "$tmp_dir/modular_affine_reduction.s")"
if (( accumulator_reductions > 12 )); then
    echo "modular affine reduction still reduces the main accumulator inside every unrolled step" >&2
    exit 1
fi

if [[ -f "$perf_dir/h-5-01.sy" ]]; then
    sed 's/kernel_ludcmp/solve_block/g; 1i int unused_extra_matrix[8][8];' \
        "$perf_dir/h-5-01.sy" > "$tmp_dir/ludcmp_renamed_extra.sy"
    "$root/compiler" "$tmp_dir/ludcmp_renamed_extra.sy" -S -o "$tmp_dir/ludcmp_renamed_extra.s"
    grep -q '\.lud\.' "$tmp_dir/ludcmp_renamed_extra.s"
fi

if [[ -f "$perf_dir/h-8-01.sy" ]]; then
    sed 's/kernel_nussinov/interval_solver/g; 1i int unrelated_seq[16];' \
        "$perf_dir/h-8-01.sy" > "$tmp_dir/nussinov_renamed_extra.sy"
    "$root/compiler" "$tmp_dir/nussinov_renamed_extra.sy" -S -o "$tmp_dir/nussinov_renamed_extra.s"
    grep -q '\.nus\.' "$tmp_dir/nussinov_renamed_extra.s"
fi

if [[ -f "$perf_dir/fft0.sy" ]]; then
    sed 's/multiply(/modular_product(/g; s/power(/exponentiate(/g; s/fft(/number_transform(/g; 1i int unused_signal[32];' \
        "$perf_dir/fft0.sy" > "$tmp_dir/fft_renamed_extra.sy"
    "$root/compiler" "$tmp_dir/fft_renamed_extra.sy" -S -o "$tmp_dir/fft_renamed_extra.s"
    grep -q '\.fast\.loop' "$tmp_dir/fft_renamed_extra.s"
    grep -Eq $'\tcb(n)?z w[0-9]+,' "$tmp_dir/fft_renamed_extra.s"
fi

echo "restored structural optimization tests passed"
