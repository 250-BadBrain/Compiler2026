#include "ir.hpp"

#include <ostream>

namespace sysyc::ir {

void dumpModule(const Module &module, std::ostream &out) {
    for (const auto &global : module.globals) {
        out << (global.isConst ? "const " : "global ") << '@' << global.name << " : " << global.type << '\n';
    }
    if (!module.globals.empty() && !module.functions.empty()) {
        out << '\n';
    }
    for (const auto &function : module.functions) {
        out << "func @" << function.name << '(';
        for (std::size_t i = 0; i < function.params.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << function.params[i];
        }
        out << ") {\n";
        for (const auto &block : function.blocks) {
            out << block.name << ":\n";
            for (const auto &inst : block.instructions) {
                out << "  " << inst << '\n';
            }
        }
        out << "}\n";
    }
}

} // namespace sysyc::ir
