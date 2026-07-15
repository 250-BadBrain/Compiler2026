#include "frontend/sema.hpp"

#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>

namespace sysyc {

namespace {

Type resultNumericType(const Type &lhs, const Type &rhs) {
    if (lhs.isError() || rhs.isError()) {
        return Type::errorType();
    }
    if (!lhs.isNumeric() || !rhs.isNumeric()) {
        return Type::errorType();
    }
    return lhs.kind == TypeKind::Float || rhs.kind == TypeKind::Float ? Type::floatType() : Type::intType();
}

TypeKind typeKindFromSpecifier(TypeSpecifier spec) {
    switch (spec) {
    case TypeSpecifier::Void:
        return TypeKind::Void;
    case TypeSpecifier::Int:
        return TypeKind::Int;
    case TypeSpecifier::Float:
        return TypeKind::Float;
    }
    return TypeKind::Error;
}

} // namespace

SemanticAnalyzer::SemanticAnalyzer() {
    pushScope();
    installRuntime();
}

bool SemanticAnalyzer::analyze(const TranslationUnit &unit) {
    for (const auto &decl : unit.declarations) {
        analyzeDecl(*decl, true);
    }
    checkMainExists(unit);
    return errors_.empty();
}

const std::vector<CompileError> &SemanticAnalyzer::errors() const {
    return errors_;
}

void SemanticAnalyzer::installRuntime() {
    auto add = [&](std::string name, TypeKind ret, std::vector<Type> params, bool variadic = false) {
        auto symbol = std::make_unique<Symbol>();
        symbol->name = std::move(name);
        symbol->kind = SymbolKind::Func;
        symbol->type = Type::functionType(ret, std::move(params), variadic);
        symbol->isGlobal = true;
        symbol->location = SourceLocation{1, 1};
        scope().insert(std::move(symbol));
    };

    add("getint", TypeKind::Int, {});
    add("getch", TypeKind::Int, {});
    add("getfloat", TypeKind::Float, {});
    add("getarray", TypeKind::Int, {Type::arrayType(TypeKind::Int, {-1})});
    add("getfarray", TypeKind::Int, {Type::arrayType(TypeKind::Float, {-1})});
    add("putint", TypeKind::Void, {Type::intType()});
    add("putch", TypeKind::Void, {Type::intType()});
    add("putfloat", TypeKind::Void, {Type::floatType()});
    add("putarray", TypeKind::Void, {Type::intType(), Type::arrayType(TypeKind::Int, {-1})});
    add("putfarray", TypeKind::Void, {Type::intType(), Type::arrayType(TypeKind::Float, {-1})});
    add("putf", TypeKind::Void, {Type::stringType()}, true);
    add("starttime", TypeKind::Void, {});
    add("stoptime", TypeKind::Void, {});
}

void SemanticAnalyzer::pushScope() {
    scopes_.push_back(std::make_unique<Scope>(current_));
    current_ = scopes_.back().get();
}

void SemanticAnalyzer::popScope() {
    current_ = current_->parent();
}

Scope &SemanticAnalyzer::scope() {
    return *current_;
}

const Scope &SemanticAnalyzer::scope() const {
    return *current_;
}

void SemanticAnalyzer::report(SourceLocation loc, const std::string &message) {
    errors_.emplace_back(loc, message);
}

void SemanticAnalyzer::analyzeDecl(const Decl &decl, bool isGlobal) {
    if (const auto *var = dynamic_cast<const VarDecl *>(&decl)) {
        analyzeVarDecl(*var, isGlobal);
        return;
    }
    if (const auto *func = dynamic_cast<const FuncDef *>(&decl)) {
        analyzeFuncDef(*func);
        return;
    }
    report(decl.location, "unknown declaration");
}

void SemanticAnalyzer::analyzeVarDecl(const VarDecl &decl, bool isGlobal) {
    const Type base = typeFromSpecifier(decl.type);
    if (base.kind == TypeKind::Void) {
        report(decl.location, "variable cannot have void type");
    }

    for (const auto &def : decl.defs) {
        std::vector<int> dims;
        for (const auto &dimExpr : def->dimensions) {
            dims.push_back(evalArrayDimension(*dimExpr));
        }

        Type type = dims.empty() ? base : Type::arrayType(base.kind, std::move(dims));
        std::optional<ConstValue> scalarConst;
        if (def->init && def->init->expr) {
            scalarConst = evalConstExpr(*def->init->expr);
        }
        if (scope().lookupCurrent(def->name)) {
            report(def->location, "redefinition of '" + def->name + "'");
        } else {
            auto symbol = std::make_unique<Symbol>();
            symbol->name = def->name;
            symbol->kind = decl.isConst ? SymbolKind::Const : SymbolKind::Var;
            symbol->type = type;
            symbol->location = def->location;
            symbol->isGlobal = isGlobal;
            if (decl.isConst && scalarConst && !type.isArray()) {
                symbol->hasConstValue = true;
                symbol->constKind = base.kind;
                if (base.kind == TypeKind::Float) {
                    symbol->floatValue = scalarConst->kind == TypeKind::Float ? scalarConst->floatValue : static_cast<double>(scalarConst->intValue);
                } else if (base.kind == TypeKind::Int) {
                    symbol->intValue = scalarConst->kind == TypeKind::Float ? static_cast<long long>(scalarConst->floatValue) : scalarConst->intValue;
                }
            }
            scope().insert(std::move(symbol));
        }

        if (decl.isConst && !def->init) {
            report(def->location, "const declaration requires initializer");
        }
        if (def->init && def->init->expr) {
            Type initType = analyzeExpr(*def->init->expr);
            if (type.isArray()) {
                report(def->init->location, "array initializer must use braces");
            } else if (!canConvert(initType, type)) {
                report(def->init->location, "cannot initialize " + type.str() + " with " + initType.str());
            }
            if (isGlobal && !scalarConst) {
                report(def->init->location, "global initializer must be constant");
            }
        }
    }
}

void SemanticAnalyzer::analyzeFuncDef(const FuncDef &func) {
    if (scope().lookupCurrent(func.name)) {
        report(func.location, "redefinition of '" + func.name + "'");
        return;
    }

    std::vector<Type> params;
    for (const auto &param : func.params) {
        Type base = typeFromSpecifier(param->type);
        if (param->isArray) {
            std::vector<int> dims = {-1};
            for (const auto &dim : param->dimensions) {
                dims.push_back(evalArrayDimension(*dim));
            }
            params.push_back(Type::arrayType(base.kind, std::move(dims)));
        } else {
            params.push_back(base);
        }
    }

    auto symbol = std::make_unique<Symbol>();
    symbol->name = func.name;
    symbol->kind = SymbolKind::Func;
    symbol->type = Type::functionType(typeKindFromSpecifier(func.returnType), params);
    symbol->location = func.location;
    symbol->isGlobal = true;
    scope().insert(std::move(symbol));

    const FuncDef *previousFunction = currentFunction_;
    currentFunction_ = &func;
    pushScope();
    for (std::size_t i = 0; i < func.params.size(); ++i) {
        const auto &param = func.params[i];
        if (scope().lookupCurrent(param->name)) {
            report(param->location, "redefinition of parameter '" + param->name + "'");
            continue;
        }
        auto paramSymbol = std::make_unique<Symbol>();
        paramSymbol->name = param->name;
        paramSymbol->kind = SymbolKind::Param;
        paramSymbol->type = params[i];
        paramSymbol->location = param->location;
        scope().insert(std::move(paramSymbol));
    }
    if (func.body) {
        analyzeBlock(*func.body);
    }
    popScope();
    currentFunction_ = previousFunction;
}

void SemanticAnalyzer::analyzeBlock(const BlockStmt &block) {
    pushScope();
    for (const auto &item : block.items) {
        if (item.decl) {
            analyzeDecl(*item.decl, false);
        } else if (item.stmt) {
            analyzeStmt(*item.stmt);
        }
    }
    popScope();
}

void SemanticAnalyzer::analyzeStmt(const Stmt &stmt) {
    if (const auto *block = dynamic_cast<const BlockStmt *>(&stmt)) {
        analyzeBlock(*block);
    } else if (const auto *expr = dynamic_cast<const ExprStmt *>(&stmt)) {
        if (expr->expr) {
            analyzeExpr(*expr->expr);
        }
    } else if (const auto *assign = dynamic_cast<const AssignStmt *>(&stmt)) {
        Type target = analyzeLValue(*assign->target);
        Type value = analyzeExpr(*assign->value);
        if (!canConvert(value, target)) {
            report(assign->location, "cannot assign " + value.str() + " to " + target.str());
        }
    } else if (const auto *ifStmt = dynamic_cast<const IfStmt *>(&stmt)) {
        analyzeExpr(*ifStmt->condition);
        analyzeStmt(*ifStmt->thenBranch);
        if (ifStmt->elseBranch) {
            analyzeStmt(*ifStmt->elseBranch);
        }
    } else if (const auto *whileStmt = dynamic_cast<const WhileStmt *>(&stmt)) {
        analyzeExpr(*whileStmt->condition);
        ++loopDepth_;
        analyzeStmt(*whileStmt->body);
        --loopDepth_;
    } else if (dynamic_cast<const BreakStmt *>(&stmt)) {
        if (loopDepth_ == 0) {
            report(stmt.location, "break statement is not inside a loop");
        }
    } else if (dynamic_cast<const ContinueStmt *>(&stmt)) {
        if (loopDepth_ == 0) {
            report(stmt.location, "continue statement is not inside a loop");
        }
    } else if (const auto *ret = dynamic_cast<const ReturnStmt *>(&stmt)) {
        Type expected = currentFunction_ ? typeFromSpecifier(currentFunction_->returnType) : Type::errorType();
        if (expected.kind == TypeKind::Void) {
            if (ret->value) {
                report(ret->location, "void function should not return a value");
            }
        } else {
            if (!ret->value) {
                report(ret->location, "non-void function should return a value");
            } else {
                Type actual = analyzeExpr(*ret->value);
                if (!canConvert(actual, expected)) {
                    report(ret->location, "return type mismatch: expected " + expected.str() + ", got " + actual.str());
                }
            }
        }
    }
}

Type SemanticAnalyzer::analyzeExpr(const Expr &expr) {
    if (dynamic_cast<const IntegerLiteral *>(&expr)) {
        return Type::intType();
    }
    if (dynamic_cast<const FloatLiteral *>(&expr)) {
        return Type::floatType();
    }
    if (dynamic_cast<const StringLiteral *>(&expr)) {
        return Type::stringType();
    }
    if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&expr)) {
        auto *symbol = scope().lookup(ref->name);
        if (!symbol) {
            report(expr.location, "use of undeclared identifier '" + ref->name + "'");
            return Type::errorType();
        }
        return symbol->type;
    }
    if (const auto *sub = dynamic_cast<const ArraySubscriptExpr *>(&expr)) {
        Type base = analyzeExpr(*sub->base);
        Type index = analyzeExpr(*sub->index);
        if (index.kind != TypeKind::Int && !index.isError()) {
            report(sub->index->location, "array subscript must be int");
        }
        return arrayElementAfterSubscript(base, expr.location);
    }
    if (const auto *call = dynamic_cast<const CallExpr *>(&expr)) {
        auto *symbol = scope().lookup(call->callee);
        if (!symbol) {
            report(expr.location, "call to undeclared function '" + call->callee + "'");
            for (const auto &arg : call->args) {
                analyzeExpr(*arg);
            }
            return Type::errorType();
        }
        if (symbol->kind != SymbolKind::Func || symbol->type.kind != TypeKind::Function) {
            report(expr.location, "'" + call->callee + "' is not a function");
            return Type::errorType();
        }
        const Type &funcType = symbol->type;
        if (!funcType.variadic && call->args.size() != funcType.params.size()) {
            report(expr.location, "wrong number of arguments in call to '" + call->callee + "'");
        } else if (funcType.variadic && call->args.size() < funcType.params.size()) {
            report(expr.location, "too few arguments in call to '" + call->callee + "'");
        }
        for (std::size_t i = 0; i < call->args.size(); ++i) {
            Type actual = analyzeExpr(*call->args[i]);
            if (i < funcType.params.size() && !canConvert(actual, funcType.params[i])) {
                report(call->args[i]->location, "argument type mismatch: expected " + funcType.params[i].str() + ", got " + actual.str());
            }
        }
        switch (funcType.returnKind) {
        case TypeKind::Void:
            return Type::voidType();
        case TypeKind::Int:
            return Type::intType();
        case TypeKind::Float:
            return Type::floatType();
        default:
            return Type::errorType();
        }
    }
    if (const auto *unary = dynamic_cast<const UnaryExpr *>(&expr)) {
        Type operand = analyzeExpr(*unary->operand);
        if (unary->op == UnaryOp::LogicalNot) {
            if (!operand.isNumeric() && !operand.isError()) {
                report(expr.location, "operator ! requires numeric operand");
            }
            return Type::intType();
        }
        if (!operand.isNumeric() && !operand.isError()) {
            report(expr.location, "unary operator requires numeric operand");
        }
        return operand;
    }
    if (const auto *binary = dynamic_cast<const BinaryExpr *>(&expr)) {
        Type lhs = analyzeExpr(*binary->lhs);
        Type rhs = analyzeExpr(*binary->rhs);
        switch (binary->op) {
        case BinaryOp::Add:
        case BinaryOp::Sub:
        case BinaryOp::Mul:
        case BinaryOp::Div: {
            Type result = resultNumericType(lhs, rhs);
            if (result.isError() && !lhs.isError() && !rhs.isError()) {
                report(expr.location, "binary arithmetic requires numeric operands");
            }
            return result;
        }
        case BinaryOp::Mod:
            if ((lhs.kind != TypeKind::Int || rhs.kind != TypeKind::Int) && !lhs.isError() && !rhs.isError()) {
                report(expr.location, "operator % requires int operands");
            }
            return Type::intType();
        case BinaryOp::Less:
        case BinaryOp::Greater:
        case BinaryOp::LessEqual:
        case BinaryOp::GreaterEqual:
        case BinaryOp::Equal:
        case BinaryOp::NotEqual:
        case BinaryOp::LogicalAnd:
        case BinaryOp::LogicalOr:
            if ((!lhs.isNumeric() || !rhs.isNumeric()) && !lhs.isError() && !rhs.isError()) {
                report(expr.location, "comparison or logical operator requires numeric operands");
            }
            return Type::intType();
        }
    }
    report(expr.location, "unknown expression");
    return Type::errorType();
}

Type SemanticAnalyzer::analyzeLValue(const Expr &expr) {
    if (!isLValueExpr(expr)) {
        report(expr.location, "assignment target is not an lvalue");
        return Type::errorType();
    }
    if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&expr)) {
        auto *symbol = scope().lookup(ref->name);
        if (!symbol) {
            report(expr.location, "use of undeclared identifier '" + ref->name + "'");
            return Type::errorType();
        }
        if (symbol->kind == SymbolKind::Const) {
            report(expr.location, "cannot assign to const '" + ref->name + "'");
        }
        return symbol->type;
    }
    return analyzeExpr(expr);
}

std::optional<ConstValue> SemanticAnalyzer::evalConstExpr(const Expr &expr) {
    if (const auto *integer = dynamic_cast<const IntegerLiteral *>(&expr)) {
        return ConstValue{TypeKind::Int, std::strtoll(integer->value.c_str(), nullptr, 0), 0.0};
    }
    if (const auto *floating = dynamic_cast<const FloatLiteral *>(&expr)) {
        return ConstValue{TypeKind::Float, 0, std::strtod(floating->value.c_str(), nullptr)};
    }
    if (const auto *unary = dynamic_cast<const UnaryExpr *>(&expr)) {
        auto value = evalConstExpr(*unary->operand);
        if (!value) {
            return std::nullopt;
        }
        if (unary->op == UnaryOp::Plus) {
            return value;
        }
        if (unary->op == UnaryOp::Minus) {
            if (value->kind == TypeKind::Float) {
                value->floatValue = -value->floatValue;
            } else {
                value->intValue = -value->intValue;
            }
            return value;
        }
        const bool isZero = value->kind == TypeKind::Float ? (value->floatValue == 0.0) : (value->intValue == 0);
        value->kind = TypeKind::Int;
        value->intValue = isZero;
        return value;
    }
    if (const auto *binary = dynamic_cast<const BinaryExpr *>(&expr)) {
        auto lhs = evalConstExpr(*binary->lhs);
        auto rhs = evalConstExpr(*binary->rhs);
        if (!lhs || !rhs) {
            return std::nullopt;
        }
        const bool useFloat = lhs->kind == TypeKind::Float || rhs->kind == TypeKind::Float;
        const double lf = lhs->kind == TypeKind::Float ? lhs->floatValue : static_cast<double>(lhs->intValue);
        const double rf = rhs->kind == TypeKind::Float ? rhs->floatValue : static_cast<double>(rhs->intValue);
        const long long li = lhs->kind == TypeKind::Float ? static_cast<long long>(lhs->floatValue) : lhs->intValue;
        const long long ri = rhs->kind == TypeKind::Float ? static_cast<long long>(rhs->floatValue) : rhs->intValue;
        switch (binary->op) {
        case BinaryOp::Add:
            return useFloat ? ConstValue{TypeKind::Float, 0, lf + rf} : ConstValue{TypeKind::Int, li + ri, 0.0};
        case BinaryOp::Sub:
            return useFloat ? ConstValue{TypeKind::Float, 0, lf - rf} : ConstValue{TypeKind::Int, li - ri, 0.0};
        case BinaryOp::Mul:
            return useFloat ? ConstValue{TypeKind::Float, 0, lf * rf} : ConstValue{TypeKind::Int, li * ri, 0.0};
        case BinaryOp::Div:
            if ((useFloat && rf == 0.0) || (!useFloat && ri == 0)) {
                return std::nullopt;
            }
            return useFloat ? ConstValue{TypeKind::Float, 0, lf / rf} : ConstValue{TypeKind::Int, li / ri, 0.0};
        case BinaryOp::Mod:
            return ri == 0 ? std::optional<ConstValue>{} : ConstValue{TypeKind::Int, li % ri, 0.0};
        case BinaryOp::Less:
            return ConstValue{TypeKind::Int, lf < rf, 0.0};
        case BinaryOp::Greater:
            return ConstValue{TypeKind::Int, lf > rf, 0.0};
        case BinaryOp::LessEqual:
            return ConstValue{TypeKind::Int, lf <= rf, 0.0};
        case BinaryOp::GreaterEqual:
            return ConstValue{TypeKind::Int, lf >= rf, 0.0};
        case BinaryOp::Equal:
            return ConstValue{TypeKind::Int, lf == rf, 0.0};
        case BinaryOp::NotEqual:
            return ConstValue{TypeKind::Int, lf != rf, 0.0};
        case BinaryOp::LogicalAnd:
            return ConstValue{TypeKind::Int, (lf != 0.0) && (rf != 0.0), 0.0};
        case BinaryOp::LogicalOr:
            return ConstValue{TypeKind::Int, (lf != 0.0) || (rf != 0.0), 0.0};
        }
    }
    if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&expr)) {
        auto *symbol = scope().lookup(ref->name);
        if (symbol && symbol->kind == SymbolKind::Const && symbol->hasConstValue) {
            if (symbol->constKind == TypeKind::Float) {
                return ConstValue{TypeKind::Float, 0, symbol->floatValue};
            }
            return ConstValue{TypeKind::Int, symbol->intValue, 0.0};
        }
    }
    return std::nullopt;
}

int SemanticAnalyzer::evalArrayDimension(const Expr &expr) {
    auto value = evalConstExpr(expr);
    if (!value || value->kind != TypeKind::Int) {
        report(expr.location, "array dimension must be an int constant expression");
        return 1;
    }
    if (value->intValue < 0) {
        report(expr.location, "array dimension must be non-negative");
        return 1;
    }
    return static_cast<int>(value->intValue);
}

Type SemanticAnalyzer::typeFromSpecifier(TypeSpecifier spec) const {
    switch (spec) {
    case TypeSpecifier::Void:
        return Type::voidType();
    case TypeSpecifier::Int:
        return Type::intType();
    case TypeSpecifier::Float:
        return Type::floatType();
    }
    return Type::errorType();
}

Type SemanticAnalyzer::arrayElementAfterSubscript(const Type &type, SourceLocation loc) {
    if (type.kind != TypeKind::Array) {
        report(loc, "subscripted value is not an array");
        return Type::errorType();
    }
    if (type.dims.size() <= 1) {
        return type.element == TypeKind::Float ? Type::floatType() : Type::intType();
    }
    std::vector<int> remaining(type.dims.begin() + 1, type.dims.end());
    return Type::arrayType(type.element, std::move(remaining));
}

bool SemanticAnalyzer::isLValueExpr(const Expr &expr) const {
    return dynamic_cast<const DeclRefExpr *>(&expr) || dynamic_cast<const ArraySubscriptExpr *>(&expr);
}

void SemanticAnalyzer::checkMainExists(const TranslationUnit &unit) {
    (void)unit;
    auto *main = scopes_.front()->lookupCurrent("main");
    if (!main || main->kind != SymbolKind::Func || main->type.returnKind != TypeKind::Int || !main->type.params.empty()) {
        report(SourceLocation{1, 1}, "program must define exactly one int main() function");
    }
}

} // namespace sysyc
