#include "ir/builder.hpp"

#include <sstream>
#include <utility>

namespace sysyc::ir {

Module IRBuilder::build(const TranslationUnit &unit) {
    module_ = Module{};
    for (const auto &decl : unit.declarations) {
        buildDecl(*decl, true);
    }
    return std::move(module_);
}

void IRBuilder::buildDecl(const Decl &decl, bool global) {
    if (const auto *var = dynamic_cast<const VarDecl *>(&decl)) {
        buildVarDecl(*var, global);
    } else if (const auto *func = dynamic_cast<const FuncDef *>(&decl)) {
        buildFuncDef(*func);
    }
}

void IRBuilder::buildVarDecl(const VarDecl &decl, bool global) {
    for (const auto &def : decl.defs) {
        std::ostringstream ty;
        ty << typeName(decl.type);
        for (const auto &dim : def->dimensions) {
            ty << '[' << buildExpr(*dim) << ']';
        }
        if (global) {
            module_.globals.push_back(Global{def->name, ty.str(), decl.isConst});
            continue;
        }

        const std::string ptr = scopedName(def->name);
        scopes_.back()[def->name] = ptr;
        emit(ptr + " = alloca " + ty.str());
        if (def->init) {
            emit("store " + buildInitValue(*def->init) + ", " + ptr);
        }
    }
}

void IRBuilder::buildFuncDef(const FuncDef &func) {
    module_.functions.push_back(Function{});
    function_ = &module_.functions.back();
    function_->name = func.name;
    block_ = nullptr;
    nextTemp_ = 0;
    scopes_.clear();
    scopes_.push_back({});

    for (const auto &param : func.params) {
        std::string paramText = typeName(param->type) + " %" + param->name;
        if (param->isArray) {
            paramText += "[]";
        }
        function_->params.push_back(paramText);
        scopes_.back()[param->name] = "%" + param->name;
    }

    startBlock("entry");
    if (func.body) {
        buildBlock(*func.body);
    }
    if (!currentBlockTerminated()) {
        emit(func.returnType == TypeSpecifier::Void ? "ret void" : "ret 0");
    }

    function_ = nullptr;
    block_ = nullptr;
    scopes_.clear();
}

void IRBuilder::buildBlock(const BlockStmt &block) {
    scopes_.push_back({});
    for (const auto &item : block.items) {
        if (item.decl) {
            buildDecl(*item.decl, false);
        } else if (item.stmt) {
            buildStmt(*item.stmt);
        }
    }
    scopes_.pop_back();
}

void IRBuilder::buildStmt(const Stmt &stmt) {
    if (const auto *block = dynamic_cast<const BlockStmt *>(&stmt)) {
        buildBlock(*block);
    } else if (const auto *expr = dynamic_cast<const ExprStmt *>(&stmt)) {
        if (expr->expr) {
            (void)buildExpr(*expr->expr);
        }
    } else if (const auto *assign = dynamic_cast<const AssignStmt *>(&stmt)) {
        emit("store " + buildExpr(*assign->value) + ", " + buildLValue(*assign->target));
    } else if (const auto *ifStmt = dynamic_cast<const IfStmt *>(&stmt)) {
        const std::string thenLabel = newLabel("if.then");
        const std::string elseLabel = ifStmt->elseBranch ? newLabel("if.else") : newLabel("if.end");
        const std::string endLabel = ifStmt->elseBranch ? newLabel("if.end") : elseLabel;
        emit("condbr " + buildExpr(*ifStmt->condition) + ", " + thenLabel + ", " + elseLabel);
        startBlock(thenLabel);
        buildStmt(*ifStmt->thenBranch);
        if (!currentBlockTerminated()) {
            emit("br " + endLabel);
        }
        if (ifStmt->elseBranch) {
            startBlock(elseLabel);
            buildStmt(*ifStmt->elseBranch);
            if (!currentBlockTerminated()) {
                emit("br " + endLabel);
            }
        }
        startBlock(endLabel);
    } else if (const auto *whileStmt = dynamic_cast<const WhileStmt *>(&stmt)) {
        const std::string condLabel = newLabel("while.cond");
        const std::string bodyLabel = newLabel("while.body");
        const std::string endLabel = newLabel("while.end");
        emit("br " + condLabel);
        startBlock(condLabel);
        emit("condbr " + buildExpr(*whileStmt->condition) + ", " + bodyLabel + ", " + endLabel);
        startBlock(bodyLabel);
        loops_.push_back(LoopLabels{condLabel, endLabel});
        buildStmt(*whileStmt->body);
        loops_.pop_back();
        if (!currentBlockTerminated()) {
            emit("br " + condLabel);
        }
        startBlock(endLabel);
    } else if (dynamic_cast<const BreakStmt *>(&stmt)) {
        emit("br " + loops_.back().breakLabel);
    } else if (dynamic_cast<const ContinueStmt *>(&stmt)) {
        emit("br " + loops_.back().continueLabel);
    } else if (const auto *ret = dynamic_cast<const ReturnStmt *>(&stmt)) {
        emit(ret->value ? "ret " + buildExpr(*ret->value) : "ret void");
    }
}

std::string IRBuilder::buildExpr(const Expr &expr) {
    if (const auto *integer = dynamic_cast<const IntegerLiteral *>(&expr)) {
        return integer->value;
    }
    if (const auto *floating = dynamic_cast<const FloatLiteral *>(&expr)) {
        return floating->value;
    }
    if (const auto *string = dynamic_cast<const StringLiteral *>(&expr)) {
        return string->value;
    }
    if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&expr)) {
        const std::string value = newTemp();
        emit(value + " = load " + buildLValue(*ref));
        return value;
    }
    if (const auto *sub = dynamic_cast<const ArraySubscriptExpr *>(&expr)) {
        const std::string value = newTemp();
        emit(value + " = load " + buildLValue(*sub));
        return value;
    }
    if (const auto *call = dynamic_cast<const CallExpr *>(&expr)) {
        std::ostringstream os;
        const std::string result = newTemp();
        os << result << " = call @" << call->callee << '(';
        for (std::size_t i = 0; i < call->args.size(); ++i) {
            if (i != 0) {
                os << ", ";
            }
            os << buildExpr(*call->args[i]);
        }
        os << ')';
        emit(os.str());
        return result;
    }
    if (const auto *unary = dynamic_cast<const UnaryExpr *>(&expr)) {
        const std::string result = newTemp();
        const std::string op = unary->op == UnaryOp::Plus ? "uplus" : (unary->op == UnaryOp::Minus ? "neg" : "not");
        emit(result + " = " + op + " " + buildExpr(*unary->operand));
        return result;
    }
    if (const auto *binary = dynamic_cast<const BinaryExpr *>(&expr)) {
        const std::string result = newTemp();
        const char *op = "add";
        switch (binary->op) {
        case BinaryOp::Add:
            op = "add";
            break;
        case BinaryOp::Sub:
            op = "sub";
            break;
        case BinaryOp::Mul:
            op = "mul";
            break;
        case BinaryOp::Div:
            op = "div";
            break;
        case BinaryOp::Mod:
            op = "mod";
            break;
        case BinaryOp::Less:
            op = "lt";
            break;
        case BinaryOp::Greater:
            op = "gt";
            break;
        case BinaryOp::LessEqual:
            op = "le";
            break;
        case BinaryOp::GreaterEqual:
            op = "ge";
            break;
        case BinaryOp::Equal:
            op = "eq";
            break;
        case BinaryOp::NotEqual:
            op = "ne";
            break;
        case BinaryOp::LogicalAnd:
            op = "and";
            break;
        case BinaryOp::LogicalOr:
            op = "or";
            break;
        }
        emit(result + " = " + op + " " + buildExpr(*binary->lhs) + ", " + buildExpr(*binary->rhs));
        return result;
    }
    return "0";
}

std::string IRBuilder::buildLValue(const Expr &expr) {
    if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&expr)) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto found = it->find(ref->name);
            if (found != it->end()) {
                return found->second;
            }
        }
        return "@" + ref->name;
    }
    if (const auto *sub = dynamic_cast<const ArraySubscriptExpr *>(&expr)) {
        const std::string ptr = newTemp();
        emit(ptr + " = gep " + buildLValue(*sub->base) + ", " + buildExpr(*sub->index));
        return ptr;
    }
    return "<invalid-lvalue>";
}

std::string IRBuilder::buildInitValue(const InitVal &init) {
    if (init.expr) {
        return buildExpr(*init.expr);
    }
    std::ostringstream os;
    os << '{';
    for (std::size_t i = 0; i < init.elements.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << buildInitValue(*init.elements[i]);
    }
    os << '}';
    return os.str();
}

std::string IRBuilder::typeName(TypeSpecifier type) const {
    switch (type) {
    case TypeSpecifier::Void:
        return "void";
    case TypeSpecifier::Int:
        return "i32";
    case TypeSpecifier::Float:
        return "f32";
    }
    return "error";
}

std::string IRBuilder::newTemp() {
    return "%" + std::to_string(nextTemp_++);
}

std::string IRBuilder::newLabel(const std::string &prefix) {
    return prefix + "." + std::to_string(nextLabel_++);
}

void IRBuilder::emit(std::string inst) {
    if (block_) {
        block_->instructions.push_back(std::move(inst));
    }
}

void IRBuilder::startBlock(std::string name) {
    function_->blocks.push_back(BasicBlock{std::move(name), {}});
    block_ = &function_->blocks.back();
}

bool IRBuilder::currentBlockTerminated() const {
    if (!block_ || block_->instructions.empty()) {
        return false;
    }
    const std::string &last = block_->instructions.back();
    return last.rfind("ret", 0) == 0 || last.rfind("br ", 0) == 0 || last.rfind("condbr ", 0) == 0;
}

std::string IRBuilder::scopedName(const std::string &name) const {
    return "%" + name + "." + std::to_string(scopes_.size());
}

} // namespace sysyc::ir
