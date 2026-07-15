#pragma once

#include "ast.hpp"
#include "token.hpp"
#include "../support/diagnostic.hpp"

#include <memory>
#include <vector>

namespace sysyc {

class Parser {
public:
    explicit Parser(std::vector<Token> tokens);

    TranslationUnit parse();
    const std::vector<CompileError> &errors() const;

private:
    const Token &peek(int offset = 0) const;
    const Token &advance();
    bool at(TokenKind kind) const;
    bool match(TokenKind kind);
    bool expect(TokenKind kind, const char *message);
    bool isTypeStart(TokenKind kind) const;
    bool isFunctionDefinition() const;

    TypeSpecifier parseTypeSpecifier();
    std::unique_ptr<Decl> parseTopLevelDecl();
    std::unique_ptr<Decl> parseVarDecl(bool isConst);
    std::unique_ptr<VarDef> parseVarDef();
    std::unique_ptr<InitVal> parseInitVal();
    std::unique_ptr<FuncDef> parseFuncDef();
    std::unique_ptr<FuncParam> parseFuncParam();
    std::unique_ptr<BlockStmt> parseBlock();
    BlockItem parseBlockItem();
    std::unique_ptr<Stmt> parseStmt();
    std::unique_ptr<Expr> parseLValue();
    std::unique_ptr<Expr> parseExpr();
    std::unique_ptr<Expr> parseLogicalOr();
    std::unique_ptr<Expr> parseLogicalAnd();
    std::unique_ptr<Expr> parseEquality();
    std::unique_ptr<Expr> parseRelational();
    std::unique_ptr<Expr> parseAdditive();
    std::unique_ptr<Expr> parseMultiplicative();
    std::unique_ptr<Expr> parseUnary();
    std::unique_ptr<Expr> parsePostfixOrPrimary();
    void skipToRecoveryPoint();
    void updateBootstrapMain(Function &bootstrap, const FuncDef &func);

    std::vector<Token> tokens_;
    std::size_t pos_ = 0;
    std::vector<CompileError> errors_;
};

} // namespace sysyc
