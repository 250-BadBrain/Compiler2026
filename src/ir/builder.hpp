#pragma once

#include "../frontend/ast.hpp"
#include "ir.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace sysyc::ir {

class IRBuilder {
public:
    Module build(const TranslationUnit &unit);

private:
    struct Binding {
        Value address;
        Type valueType;
        std::vector<int> dims;
        bool isArray = false;
    };

    struct FunctionInfo {
        Type returnType;
        std::vector<Type> params;
    };

    void collectFunctionInfo(const TranslationUnit &unit);
    void buildDecl(const Decl &decl, bool global);
    void buildVarDecl(const VarDecl &decl, bool global);
    void buildFuncDef(const FuncDef &func);
    void buildBlock(const BlockStmt &block);
    void buildStmt(const Stmt &stmt);
    Value buildExpr(const Expr &expr);
    Value buildBinaryExpr(const BinaryExpr &expr);
    Value buildShortCircuit(const BinaryExpr &expr);
    Value convert(Value value, Type target);
    Value boolValue(Value value);
    Value buildLValue(const Expr &expr);
    const DeclRefExpr *collectArrayBase(const Expr &expr, std::vector<const Expr *> &indices) const;
    bool subscriptYieldsArray(const ArraySubscriptExpr &expr) const;
    Type lValueElementType(const Expr &expr) const;
    Value emitValue(Opcode opcode, Type type, std::vector<Value> operands = {}, std::string text = {});
    void emitVoid(Opcode opcode, std::vector<Value> operands = {}, std::string text = {});
    void startBlock(std::string name);
    bool currentBlockTerminated() const;
    Type typeFrom(TypeSpecifier spec) const;
    std::string newLabel(const std::string &prefix);
    Binding lookup(const std::string &name) const;
    Value constant(Type type, std::string text) const;
    int literalDimension(const Expr &expr) const;
    long long evalConstInt(const Expr &expr) const;
    double evalConstFloat(const Expr &expr) const;
    void collectArrayInitializer(const InitVal &init, const std::vector<int> &dims, int level, int &index, std::vector<const Expr *> &values) const;
    static int productFrom(const std::vector<int> &dims, int level);
    std::string staticInitValue(const Expr *expr, Type type) const;
    static std::string floatWord(double value);

    Module module_;
    Function *function_ = nullptr;
    BasicBlock *block_ = nullptr;
    int nextValue_ = 0;
    int nextLabel_ = 0;
    std::vector<std::unordered_map<std::string, Binding>> scopes_;
    std::unordered_map<std::string, Binding> globals_;
    std::unordered_map<std::string, long long> constInts_;
    std::unordered_map<std::string, double> constFloats_;
    std::unordered_map<std::string, FunctionInfo> functions_;
    std::vector<std::pair<std::string, std::string>> loops_;
};

} // namespace sysyc::ir
