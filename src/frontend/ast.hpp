#pragma once

#include "../support/diagnostic.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sysyc {

enum class TypeSpecifier {
    Void,
    Int,
    Float,
};

enum class UnaryOp {
    Plus,
    Minus,
    LogicalNot,
};

enum class BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    Equal,
    NotEqual,
    LogicalAnd,
    LogicalOr,
};

struct AstNode {
    explicit AstNode(SourceLocation loc) : location(loc) {}
    virtual ~AstNode() = default;

    SourceLocation location;
};

struct Expr : AstNode {
    using AstNode::AstNode;
};

struct IntegerLiteral : Expr {
    IntegerLiteral(SourceLocation loc, std::string value) : Expr(loc), value(std::move(value)) {}
    std::string value;
};

struct FloatLiteral : Expr {
    FloatLiteral(SourceLocation loc, std::string value) : Expr(loc), value(std::move(value)) {}
    std::string value;
};

struct StringLiteral : Expr {
    StringLiteral(SourceLocation loc, std::string value) : Expr(loc), value(std::move(value)) {}
    std::string value;
};

struct DeclRefExpr : Expr {
    DeclRefExpr(SourceLocation loc, std::string name) : Expr(loc), name(std::move(name)) {}
    std::string name;
};

struct ArraySubscriptExpr : Expr {
    ArraySubscriptExpr(SourceLocation loc, std::unique_ptr<Expr> base, std::unique_ptr<Expr> index)
        : Expr(loc), base(std::move(base)), index(std::move(index)) {}
    std::unique_ptr<Expr> base;
    std::unique_ptr<Expr> index;
};

struct CallExpr : Expr {
    CallExpr(SourceLocation loc, std::string callee) : Expr(loc), callee(std::move(callee)) {}
    std::string callee;
    std::vector<std::unique_ptr<Expr>> args;
};

struct UnaryExpr : Expr {
    UnaryExpr(SourceLocation loc, UnaryOp op, std::unique_ptr<Expr> operand)
        : Expr(loc), op(op), operand(std::move(operand)) {}
    UnaryOp op;
    std::unique_ptr<Expr> operand;
};

struct BinaryExpr : Expr {
    BinaryExpr(SourceLocation loc, BinaryOp op, std::unique_ptr<Expr> lhs, std::unique_ptr<Expr> rhs)
        : Expr(loc), op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
    BinaryOp op;
    std::unique_ptr<Expr> lhs;
    std::unique_ptr<Expr> rhs;
};

struct InitVal : AstNode {
    explicit InitVal(SourceLocation loc) : AstNode(loc) {}
    std::unique_ptr<Expr> expr;
    std::vector<std::unique_ptr<InitVal>> elements;
};

struct Decl : AstNode {
    explicit Decl(SourceLocation loc) : AstNode(loc) {}
};

struct VarDef : AstNode {
    VarDef(SourceLocation loc, std::string name) : AstNode(loc), name(std::move(name)) {}
    std::string name;
    std::vector<std::unique_ptr<Expr>> dimensions;
    std::unique_ptr<InitVal> init;
};

struct VarDecl : Decl {
    VarDecl(SourceLocation loc, bool isConst, TypeSpecifier type) : Decl(loc), isConst(isConst), type(type) {}
    bool isConst = false;
    TypeSpecifier type = TypeSpecifier::Int;
    std::vector<std::unique_ptr<VarDef>> defs;
};

struct Stmt : AstNode {
    using AstNode::AstNode;
};

struct BlockItem {
    std::unique_ptr<Decl> decl;
    std::unique_ptr<Stmt> stmt;
};

struct BlockStmt : Stmt {
    using Stmt::Stmt;
    std::vector<BlockItem> items;
};

struct DeclStmt : Stmt {
    DeclStmt(SourceLocation loc, std::unique_ptr<Decl> decl) : Stmt(loc), decl(std::move(decl)) {}
    std::unique_ptr<Decl> decl;
};

struct ExprStmt : Stmt {
    ExprStmt(SourceLocation loc, std::unique_ptr<Expr> expr) : Stmt(loc), expr(std::move(expr)) {}
    std::unique_ptr<Expr> expr;
};

struct AssignStmt : Stmt {
    AssignStmt(SourceLocation loc, std::unique_ptr<Expr> target, std::unique_ptr<Expr> value)
        : Stmt(loc), target(std::move(target)), value(std::move(value)) {}
    std::unique_ptr<Expr> target;
    std::unique_ptr<Expr> value;
};

struct IfStmt : Stmt {
    explicit IfStmt(SourceLocation loc) : Stmt(loc) {}
    std::unique_ptr<Expr> condition;
    std::unique_ptr<Stmt> thenBranch;
    std::unique_ptr<Stmt> elseBranch;
};

struct WhileStmt : Stmt {
    explicit WhileStmt(SourceLocation loc) : Stmt(loc) {}
    std::unique_ptr<Expr> condition;
    std::unique_ptr<Stmt> body;
};

struct BreakStmt : Stmt {
    using Stmt::Stmt;
};

struct ContinueStmt : Stmt {
    using Stmt::Stmt;
};

struct ReturnStmt : Stmt {
    ReturnStmt(SourceLocation loc, std::unique_ptr<Expr> value) : Stmt(loc), value(std::move(value)) {}
    std::unique_ptr<Expr> value;
};

struct FuncParam : AstNode {
    FuncParam(SourceLocation loc, TypeSpecifier type, std::string name)
        : AstNode(loc), type(type), name(std::move(name)) {}
    TypeSpecifier type = TypeSpecifier::Int;
    std::string name;
    bool isArray = false;
    std::vector<std::unique_ptr<Expr>> dimensions;
};

struct FuncDef : Decl {
    FuncDef(SourceLocation loc, TypeSpecifier returnType, std::string name)
        : Decl(loc), returnType(returnType), name(std::move(name)) {}
    TypeSpecifier returnType = TypeSpecifier::Int;
    std::string name;
    std::vector<std::unique_ptr<FuncParam>> params;
    std::unique_ptr<BlockStmt> body;
};

struct Function {
    std::string name;
    int returnValue = 0;
};

struct TranslationUnit {
    std::vector<std::unique_ptr<Decl>> declarations;
    Function mainFunction;
};

} // namespace sysyc
