#pragma once

#include "frontend/ast.hpp"

#include <iosfwd>

namespace sysyc::riscv {

void emitAssembly(const TranslationUnit &unit, std::ostream &out);

} // namespace sysyc::riscv
