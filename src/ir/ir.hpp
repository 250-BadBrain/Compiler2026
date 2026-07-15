#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace sysyc::ir {

enum class TypeKind {
    Void,
    I32,
    F32,
    Ptr,
};

struct Type {
    TypeKind kind = TypeKind::Void;
};

enum class Opcode {
    Alloca,
    Load,
    Store,
    Gep,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,
    Not,
    ICmp,
    FCmp,
    Cast,
    Phi,
    Call,
    Br,
    CondBr,
    Ret,
};

struct Value {
    int id = -1;
    Type type;
    std::string name;
    bool constant = false;
};

struct Instruction {
    int result = -1;
    Type resultType;
    Opcode opcode = Opcode::Alloca;
    std::vector<Value> operands;
    std::string text;
};

struct BasicBlock {
    std::string name;
    std::vector<Instruction> instructions;
    std::vector<int> predecessors;
    std::vector<int> successors;
};

struct Function {
    std::string name;
    Type returnType;
    std::vector<Value> params;
    std::vector<BasicBlock> blocks;
};

struct Global {
    std::string name;
    Type type;
    std::vector<int> dimensions;
    std::vector<std::string> initValues;
    bool isConst = false;
};

struct Module {
    std::vector<Global> globals;
    std::vector<Function> functions;
};

std::string typeName(Type type);
std::string opcodeName(Opcode opcode);
void dumpModule(const Module &module, std::ostream &out);
bool optimize(Module &module);

} // namespace sysyc::ir
