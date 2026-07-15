#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace sysyc::ir {

struct BasicBlock {
    std::string name;
    std::vector<std::string> instructions;
};

struct Function {
    std::string name;
    std::vector<std::string> params;
    std::vector<BasicBlock> blocks;
};

struct Global {
    std::string name;
    std::string type;
    bool isConst = false;
};

struct Module {
    std::vector<Global> globals;
    std::vector<Function> functions;
};

void dumpModule(const Module &module, std::ostream &out);

} // namespace sysyc::ir
