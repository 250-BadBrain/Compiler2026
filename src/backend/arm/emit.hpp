#pragma once

#include "../../ir/ir.hpp"

#include <iosfwd>

namespace sysyc::arm {

void emitAssembly(const ir::Module &module, std::ostream &out);

} // namespace sysyc::arm
