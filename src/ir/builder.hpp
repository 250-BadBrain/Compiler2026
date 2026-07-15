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
    struct LoopLabels {
        std::string continueLabel;
        std::string breakLabel;
    };

    void buildDecl(const Decl &decl, bool global);
    void buildVarDecl(const VarDecl &decl, bool global);
    void buildFuncDef(const FuncDef &func);
    void buildBlock(const BlockStmt &block);
    void buildStmt(const Stmt &stmt);
    std::string buildExpr(const Expr &expr);
    std::string buildLValue(const Expr &expr);
    std::string buildInitValue(const InitVal &init);
    std::string typeName(TypeSpecifier type) const;
    std::string newTemp();
    std::string newLabel(const std::string &prefix);
    void emit(std::string inst);
    void startBlock(std::string name);
    bool currentBlockTerminated() const;
    std::string scopedName(const std::string &name) const;

    Module module_;
    Function *function_ = nullptr;
    BasicBlock *block_ = nullptr;
    int nextTemp_ = 0;
    int nextLabel_ = 0;
    std::vector<std::unordered_map<std::string, std::string>> scopes_;
    std::vector<LoopLabels> loops_;
};

} // namespace sysyc::ir
