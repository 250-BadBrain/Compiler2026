#pragma once

#include "ast.hpp"
#include "symbol.hpp"
#include "../support/diagnostic.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace sysyc {

struct ConstValue {
    TypeKind kind = TypeKind::Error;
    long long intValue = 0;
    double floatValue = 0.0;
};

class SemanticAnalyzer {
public:
    SemanticAnalyzer();

    bool analyze(const TranslationUnit &unit);
    const std::vector<CompileError> &errors() const;

private:
    void installRuntime();
    void pushScope();
    void popScope();
    Scope &scope();
    const Scope &scope() const;
    void report(SourceLocation loc, const std::string &message);

    void analyzeDecl(const Decl &decl, bool isGlobal);
    void analyzeVarDecl(const VarDecl &decl, bool isGlobal);
    void analyzeFuncDef(const FuncDef &func);
    void analyzeBlock(const BlockStmt &block);
    void analyzeStmt(const Stmt &stmt);
    Type analyzeExpr(const Expr &expr);
    Type analyzeLValue(const Expr &expr);
    std::optional<ConstValue> evalConstExpr(const Expr &expr);
    int evalArrayDimension(const Expr &expr);
    Type typeFromSpecifier(TypeSpecifier spec) const;
    Type arrayElementAfterSubscript(const Type &type, SourceLocation loc);
    bool isLValueExpr(const Expr &expr) const;
    void checkMainExists(const TranslationUnit &unit);

    std::vector<std::unique_ptr<Scope>> scopes_;
    Scope *current_ = nullptr;
    std::vector<CompileError> errors_;
    const FuncDef *currentFunction_ = nullptr;
    int loopDepth_ = 0;
};

} // namespace sysyc
