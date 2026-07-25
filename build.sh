#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

if [[ -z "${CXX:-}" ]]; then
    if command -v clang++ >/dev/null 2>&1; then
        CXX="clang++"
    else
        CXX="g++"
    fi
fi

sources=(
    src/main.cpp
    src/frontend/lexer.cpp
    src/frontend/parser.cpp
    src/frontend/type.cpp
    src/frontend/sema.cpp
    src/frontend/symbol.cpp
    src/ir/builder.cpp
    src/ir/ir.cpp
    src/backend/arm/emit.cpp
    src/support/diagnostic.cpp
)

"$CXX" -std=c++17 -O2 -lm \
    "${sources[@]}" \
    -I src/frontend \
    -I src/ir \
    -I src/backend/arm \
    -I src/support \
    -o compiler

echo "built ./compiler with $CXX"
