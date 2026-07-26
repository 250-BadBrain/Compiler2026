#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/compiler2026-harness-guards.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

expect_empty_case_failure() {
    local script="$1"
    local stdout_file="$tmp_dir/$(basename "$script").stdout"
    local stderr_file="$tmp_dir/$(basename "$script").stderr"
    if "$script" "$tmp_dir" >"$stdout_file" 2>"$stderr_file"; then
        echo "expected empty case directory to fail: $script" >&2
        cat "$stdout_file" >&2
        exit 1
    fi
    if ! grep -Eq 'no .*\.sy|empty|contains no' "$stderr_file"; then
        echo "missing empty-directory diagnostic: $script" >&2
        cat "$stderr_file" >&2
        exit 1
    fi
}

expect_empty_case_failure "$root/tests/run_functional.sh"
expect_empty_case_failure "$root/tests/run_public_functional.sh"

echo "harness guard tests passed"
