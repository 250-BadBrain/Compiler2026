cmake_minimum_required(VERSION 3.16)

project(compiler2026 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(compiler
    src/main.cpp
    src/frontend/lexer.cpp
    src/frontend/parser.cpp
    src/frontend/type.cpp
    src/frontend/symbol.cpp
    src/frontend/sema.cpp
    src/ir/ir.cpp
    src/ir/builder.cpp
    src/backend/riscv/emit.cpp
    src/support/diagnostic.cpp
)

target_include_directories(compiler PRIVATE src)

target_compile_options(compiler PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
