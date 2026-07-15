#include "parser.hpp"

#include <cstdlib>
#include <utility>

namespace sysyc {

Parser::Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

TranslationUnit Parser::parse() {
    TranslationUnit unit;
    while (!at(TokenKind::End)) {
        auto decl = parseTopLevelDecl();
        if (decl) {
            if (auto *func = dynamic_cast<FuncDef *>(decl.get())) {
                updateBootstrapMain(unit.mainFunction, *func);
            }
            unit.declarations.push_back(std::move(decl));
        } else {
            skipToRecoveryPoint();
        }
    }
    return unit;
}

const std::vector<CompileError> &Parser::errors() const {
    return errors_;
}

const Token &Parser::peek(int offset) const {
    const std::size_t index = pos_ + static_cast<std::size_t>(offset);
    if (index < tokens_.size()) {
        return tokens_[index];
    }
    return tokens_.back();
}

const Token &Parser::advance() {
    if (pos_ < tokens_.size()) {
        return tokens_[pos_++];
    }
    return tokens_.back();
}

bool Parser::at(TokenKind kind) const {
    return peek().kind == kind;
}

bool Parser::match(TokenKind kind) {
    if (!at(kind)) {
        return false;
    }
    advance();
    return true;
}

bool Parser::expect(TokenKind kind, const char *message) {
    if (match(kind)) {
        return true;
    }
    errors_.emplace_back(peek().location, message);
    return false;
}

bool Parser::isTypeStart(TokenKind kind) const {
    return kind == TokenKind::KeywordInt || kind == TokenKind::KeywordFloat || kind == TokenKind::KeywordVoid;
}

bool Parser::isFunctionDefinition() const {
    int offset = 0;
    if (!isTypeStart(peek(offset).kind)) {
        return false;
    }
    ++offset;
    if (peek(offset).kind != TokenKind::Identifier) {
        return false;
    }
    ++offset;
    return peek(offset).kind == TokenKind::LParen;
}

TypeSpecifier Parser::parseTypeSpecifier() {
    if (match(TokenKind::KeywordInt)) {
        return TypeSpecifier::Int;
    }
    if (match(TokenKind::KeywordFloat)) {
        return TypeSpecifier::Float;
    }
    if (match(TokenKind::KeywordVoid)) {
        return TypeSpecifier::Void;
    }
    errors_.emplace_back(peek().location, "expected type specifier");
    return TypeSpecifier::Int;
}

std::unique_ptr<Decl> Parser::parseTopLevelDecl() {
    if (at(TokenKind::KeywordConst)) {
        return parseVarDecl(true);
    }
    if (isFunctionDefinition()) {
        return parseFuncDef();
    }
    if (isTypeStart(peek().kind)) {
        return parseVarDecl(false);
    }
    errors_.emplace_back(peek().location, "expected declaration or function definition");
    return nullptr;
}

std::unique_ptr<Decl> Parser::parseVarDecl(bool isConst) {
    const SourceLocation loc = peek().location;
    if (isConst) {
        advance();
    }
    TypeSpecifier type = parseTypeSpecifier();
    if (type == TypeSpecifier::Void) {
        errors_.emplace_back(peek().location, "variable declaration cannot use void type");
    }

    auto decl = std::make_unique<VarDecl>(loc, isConst, type);
    decl->defs.push_back(parseVarDef());
    while (match(TokenKind::Comma)) {
        decl->defs.push_back(parseVarDef());
    }
    expect(TokenKind::Semicolon, "expected ';' after declaration");
    return decl;
}

std::unique_ptr<VarDef> Parser::parseVarDef() {
    const SourceLocation loc = peek().location;
    std::string name;
    if (at(TokenKind::Identifier)) {
        name = advance().text;
    } else {
        errors_.emplace_back(peek().location, "expected identifier in declaration");
    }

    auto def = std::make_unique<VarDef>(loc, std::move(name));
    while (match(TokenKind::LBracket)) {
        def->dimensions.push_back(parseExpr());
        expect(TokenKind::RBracket, "expected ']' after array dimension");
    }
    if (match(TokenKind::Assign)) {
        def->init = parseInitVal();
    }
    return def;
}

std::unique_ptr<InitVal> Parser::parseInitVal() {
    const SourceLocation loc = peek().location;
    auto init = std::make_unique<InitVal>(loc);
    if (match(TokenKind::LBrace)) {
        if (!at(TokenKind::RBrace)) {
            init->elements.push_back(parseInitVal());
            while (match(TokenKind::Comma)) {
                if (at(TokenKind::RBrace)) {
                    break;
                }
                init->elements.push_back(parseInitVal());
            }
        }
        expect(TokenKind::RBrace, "expected '}' after initializer list");
    } else {
        init->expr = parseExpr();
    }
    return init;
}

std::unique_ptr<FuncDef> Parser::parseFuncDef() {
    const SourceLocation loc = peek().location;
    TypeSpecifier returnType = parseTypeSpecifier();
    std::string name;
    if (at(TokenKind::Identifier)) {
        name = advance().text;
    } else {
        errors_.emplace_back(peek().location, "expected function name");
    }

    auto func = std::make_unique<FuncDef>(loc, returnType, std::move(name));
    expect(TokenKind::LParen, "expected '(' after function name");
    if (!at(TokenKind::RParen)) {
        func->params.push_back(parseFuncParam());
        while (match(TokenKind::Comma)) {
            func->params.push_back(parseFuncParam());
        }
    }
    expect(TokenKind::RParen, "expected ')' after parameter list");
    func->body = parseBlock();
    return func;
}

std::unique_ptr<FuncParam> Parser::parseFuncParam() {
    const SourceLocation loc = peek().location;
    TypeSpecifier type = parseTypeSpecifier();
    std::string name;
    if (at(TokenKind::Identifier)) {
        name = advance().text;
    } else {
        errors_.emplace_back(peek().location, "expected parameter name");
    }

    auto param = std::make_unique<FuncParam>(loc, type, std::move(name));
    if (match(TokenKind::LBracket)) {
        param->isArray = true;
        expect(TokenKind::RBracket, "expected ']' after array parameter first dimension");
        while (match(TokenKind::LBracket)) {
            param->dimensions.push_back(parseExpr());
            expect(TokenKind::RBracket, "expected ']' after array parameter dimension");
        }
    }
    return param;
}

std::unique_ptr<BlockStmt> Parser::parseBlock() {
    const SourceLocation loc = peek().location;
    auto block = std::make_unique<BlockStmt>(loc);
    expect(TokenKind::LBrace, "expected '{' to start block");
    while (!at(TokenKind::RBrace) && !at(TokenKind::End)) {
        block->items.push_back(parseBlockItem());
    }
    expect(TokenKind::RBrace, "expected '}' to end block");
    return block;
}

BlockItem Parser::parseBlockItem() {
    BlockItem item;
    if (at(TokenKind::KeywordConst)) {
        item.decl = parseVarDecl(true);
    } else if (peek().kind == TokenKind::KeywordInt || peek().kind == TokenKind::KeywordFloat) {
        item.decl = parseVarDecl(false);
    } else {
        item.stmt = parseStmt();
    }
    return item;
}

std::unique_ptr<Stmt> Parser::parseStmt() {
    const SourceLocation loc = peek().location;
    if (at(TokenKind::LBrace)) {
        return parseBlock();
    }
    if (match(TokenKind::KeywordIf)) {
        auto stmt = std::make_unique<IfStmt>(loc);
        expect(TokenKind::LParen, "expected '(' after if");
        stmt->condition = parseExpr();
        expect(TokenKind::RParen, "expected ')' after if condition");
        stmt->thenBranch = parseStmt();
        if (match(TokenKind::KeywordElse)) {
            stmt->elseBranch = parseStmt();
        }
        return stmt;
    }
    if (match(TokenKind::KeywordWhile)) {
        auto stmt = std::make_unique<WhileStmt>(loc);
        expect(TokenKind::LParen, "expected '(' after while");
        stmt->condition = parseExpr();
        expect(TokenKind::RParen, "expected ')' after while condition");
        stmt->body = parseStmt();
        return stmt;
    }
    if (match(TokenKind::KeywordBreak)) {
        expect(TokenKind::Semicolon, "expected ';' after break");
        return std::make_unique<BreakStmt>(loc);
    }
    if (match(TokenKind::KeywordContinue)) {
        expect(TokenKind::Semicolon, "expected ';' after continue");
        return std::make_unique<ContinueStmt>(loc);
    }
    if (match(TokenKind::KeywordReturn)) {
        std::unique_ptr<Expr> value;
        if (!at(TokenKind::Semicolon)) {
            value = parseExpr();
        }
        expect(TokenKind::Semicolon, "expected ';' after return");
        return std::make_unique<ReturnStmt>(loc, std::move(value));
    }
    if (match(TokenKind::Semicolon)) {
        return std::make_unique<ExprStmt>(loc, nullptr);
    }

    if (at(TokenKind::Identifier)) {
        const std::size_t checkpoint = pos_;
        auto target = parseLValue();
        if (match(TokenKind::Assign)) {
            auto value = parseExpr();
            expect(TokenKind::Semicolon, "expected ';' after assignment");
            return std::make_unique<AssignStmt>(loc, std::move(target), std::move(value));
        }
        pos_ = checkpoint;
    }

    auto expr = parseExpr();
    expect(TokenKind::Semicolon, "expected ';' after expression");
    return std::make_unique<ExprStmt>(loc, std::move(expr));
}

std::unique_ptr<Expr> Parser::parseLValue() {
    const SourceLocation loc = peek().location;
    std::string name;
    if (at(TokenKind::Identifier)) {
        name = advance().text;
    } else {
        errors_.emplace_back(peek().location, "expected identifier");
    }

    std::unique_ptr<Expr> expr = std::make_unique<DeclRefExpr>(loc, std::move(name));
    while (match(TokenKind::LBracket)) {
        auto index = parseExpr();
        expect(TokenKind::RBracket, "expected ']' after subscript");
        expr = std::make_unique<ArraySubscriptExpr>(loc, std::move(expr), std::move(index));
    }
    return expr;
}

std::unique_ptr<Expr> Parser::parseExpr() {
    return parseLogicalOr();
}

std::unique_ptr<Expr> Parser::parseLogicalOr() {
    auto lhs = parseLogicalAnd();
    while (match(TokenKind::PipePipe)) {
        const SourceLocation loc = lhs->location;
        auto rhs = parseLogicalAnd();
        lhs = std::make_unique<BinaryExpr>(loc, BinaryOp::LogicalOr, std::move(lhs), std::move(rhs));
    }
    return lhs;
}

std::unique_ptr<Expr> Parser::parseLogicalAnd() {
    auto lhs = parseEquality();
    while (match(TokenKind::AmpAmp)) {
        const SourceLocation loc = lhs->location;
        auto rhs = parseEquality();
        lhs = std::make_unique<BinaryExpr>(loc, BinaryOp::LogicalAnd, std::move(lhs), std::move(rhs));
    }
    return lhs;
}

std::unique_ptr<Expr> Parser::parseEquality() {
    auto lhs = parseRelational();
    while (at(TokenKind::Equal) || at(TokenKind::NotEqual)) {
        const SourceLocation loc = lhs->location;
        const BinaryOp op = match(TokenKind::Equal) ? BinaryOp::Equal : (advance(), BinaryOp::NotEqual);
        auto rhs = parseRelational();
        lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
    }
    return lhs;
}

std::unique_ptr<Expr> Parser::parseRelational() {
    auto lhs = parseAdditive();
    while (at(TokenKind::Less) || at(TokenKind::Greater) || at(TokenKind::LessEqual) || at(TokenKind::GreaterEqual)) {
        const SourceLocation loc = lhs->location;
        BinaryOp op = BinaryOp::Less;
        if (match(TokenKind::Less)) {
            op = BinaryOp::Less;
        } else if (match(TokenKind::Greater)) {
            op = BinaryOp::Greater;
        } else if (match(TokenKind::LessEqual)) {
            op = BinaryOp::LessEqual;
        } else {
            expect(TokenKind::GreaterEqual, "expected relational operator");
            op = BinaryOp::GreaterEqual;
        }
        auto rhs = parseAdditive();
        lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
    }
    return lhs;
}

std::unique_ptr<Expr> Parser::parseAdditive() {
    auto lhs = parseMultiplicative();
    while (at(TokenKind::Plus) || at(TokenKind::Minus)) {
        const SourceLocation loc = lhs->location;
        const BinaryOp op = match(TokenKind::Plus) ? BinaryOp::Add : (advance(), BinaryOp::Sub);
        auto rhs = parseMultiplicative();
        lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
    }
    return lhs;
}

std::unique_ptr<Expr> Parser::parseMultiplicative() {
    auto lhs = parseUnary();
    while (at(TokenKind::Star) || at(TokenKind::Slash) || at(TokenKind::Percent)) {
        const SourceLocation loc = lhs->location;
        BinaryOp op = BinaryOp::Mul;
        if (match(TokenKind::Star)) {
            op = BinaryOp::Mul;
        } else if (match(TokenKind::Slash)) {
            op = BinaryOp::Div;
        } else {
            expect(TokenKind::Percent, "expected multiplicative operator");
            op = BinaryOp::Mod;
        }
        auto rhs = parseUnary();
        lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), std::move(rhs));
    }
    return lhs;
}

std::unique_ptr<Expr> Parser::parseUnary() {
    const SourceLocation loc = peek().location;
    if (match(TokenKind::Plus)) {
        return std::make_unique<UnaryExpr>(loc, UnaryOp::Plus, parseUnary());
    }
    if (match(TokenKind::Minus)) {
        return std::make_unique<UnaryExpr>(loc, UnaryOp::Minus, parseUnary());
    }
    if (match(TokenKind::Bang)) {
        return std::make_unique<UnaryExpr>(loc, UnaryOp::LogicalNot, parseUnary());
    }
    return parsePostfixOrPrimary();
}

std::unique_ptr<Expr> Parser::parsePostfixOrPrimary() {
    const SourceLocation loc = peek().location;
    std::unique_ptr<Expr> expr;
    if (match(TokenKind::LParen)) {
        expr = parseExpr();
        expect(TokenKind::RParen, "expected ')' after expression");
    } else if (at(TokenKind::IntLiteral)) {
        expr = std::make_unique<IntegerLiteral>(loc, advance().text);
    } else if (at(TokenKind::FloatLiteral)) {
        expr = std::make_unique<FloatLiteral>(loc, advance().text);
    } else if (at(TokenKind::StringLiteral)) {
        expr = std::make_unique<StringLiteral>(loc, advance().text);
    } else if (at(TokenKind::Identifier)) {
        std::string name = advance().text;
        if (match(TokenKind::LParen)) {
            auto call = std::make_unique<CallExpr>(loc, std::move(name));
            if (!at(TokenKind::RParen)) {
                call->args.push_back(parseExpr());
                while (match(TokenKind::Comma)) {
                    call->args.push_back(parseExpr());
                }
            }
            expect(TokenKind::RParen, "expected ')' after function arguments");
            expr = std::move(call);
        } else {
            expr = std::make_unique<DeclRefExpr>(loc, std::move(name));
        }
    } else {
        errors_.emplace_back(peek().location, "expected expression");
        advance();
        expr = std::make_unique<IntegerLiteral>(loc, "0");
    }

    while (match(TokenKind::LBracket)) {
        auto index = parseExpr();
        expect(TokenKind::RBracket, "expected ']' after subscript");
        expr = std::make_unique<ArraySubscriptExpr>(loc, std::move(expr), std::move(index));
    }
    return expr;
}

void Parser::skipToRecoveryPoint() {
    while (!at(TokenKind::End) && !at(TokenKind::Semicolon) && !at(TokenKind::RBrace)) {
        advance();
    }
    if (at(TokenKind::Semicolon)) {
        advance();
    }
}

void Parser::updateBootstrapMain(Function &bootstrap, const FuncDef &func) {
    if (func.name != "main") {
        return;
    }
    bootstrap.name = "main";
    bootstrap.returnValue = 0;
    if (!func.body) {
        return;
    }
    for (const auto &item : func.body->items) {
        const auto *ret = item.stmt ? dynamic_cast<const ReturnStmt *>(item.stmt.get()) : nullptr;
        const auto *literal = ret && ret->value ? dynamic_cast<const IntegerLiteral *>(ret->value.get()) : nullptr;
        if (literal) {
            bootstrap.returnValue = static_cast<int>(std::strtol(literal->value.c_str(), nullptr, 0));
            return;
        }
    }
}

} // namespace sysyc
