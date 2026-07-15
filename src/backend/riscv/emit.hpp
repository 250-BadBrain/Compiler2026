#pragma once

#include "../../frontend/ast.hpp"
#include "../../ir/ir.hpp"

#include <iosfwd>

namespace sysyc::riscv {

void emitAssembly(const TranslationUnit &unit, std::ostream &out);
void emitAssembly(const ir::Module &module, std::ostream &out);

} // namespace sysyc::riscv
