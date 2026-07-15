#include "builder.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace sysyc::ir {

Module IRBuilder::build(const TranslationUnit &unit) {
    module_ = Module{};
    globals_.clear();
    constInts_.clear();
    constFloats_.clear();
    functions_.clear();
    collectFunctionInfo(unit);
    for (const auto &decl : unit.declarations) {
        buildDecl(*decl, true);
    }
    return std::move(module_);
}

void IRBuilder::collectFunctionInfo(const TranslationUnit &unit) {
    auto runtime = [&](std::string name, Type ret, std::vector<Type> params) {
        functions_[std::move(name)] = FunctionInfo{ret, std::move(params)};
    };
    runtime("getint", Type{TypeKind::I32}, {});
    runtime("getch", Type{TypeKind::I32}, {});
    runtime("getfloat", Type{TypeKind::F32}, {});
    runtime("getarray", Type{TypeKind::I32}, {Type{TypeKind::Ptr}});
    runtime("getfarray", Type{TypeKind::I32}, {Type{TypeKind::Ptr}});
    runtime("putint", Type{TypeKind::Void}, {Type{TypeKind::I32}});
    runtime("putch", Type{TypeKind::Void}, {Type{TypeKind::I32}});
    runtime("putfloat", Type{TypeKind::Void}, {Type{TypeKind::F32}});
    runtime("putarray", Type{TypeKind::Void}, {Type{TypeKind::I32}, Type{TypeKind::Ptr}});
    runtime("putfarray", Type{TypeKind::Void}, {Type{TypeKind::I32}, Type{TypeKind::Ptr}});
    runtime("starttime", Type{TypeKind::Void}, {});
    runtime("stoptime", Type{TypeKind::Void}, {});

    for (const auto &decl : unit.declarations) {
        const auto *func = dynamic_cast<const FuncDef *>(decl.get());
        if (!func) {
            continue;
        }
        FunctionInfo info{typeFrom(func->returnType), {}};
        for (const auto &param : func->params) {
            info.params.push_back(param->isArray ? Type{TypeKind::Ptr} : typeFrom(param->type));
        }
        functions_[func->name] = std::move(info);
    }
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
        std::vector<int> dims;
        for (const auto &dim : def->dimensions) {
            dims.push_back(literalDimension(*dim));
        }
        int elements = 1;
        for (int dim : dims) {
            elements *= dim;
        }

        if (global) {
            std::vector<std::string> initValues;
            std::vector<const Expr *> initExprs(static_cast<std::size_t>(elements), nullptr);
            if (def->init) {
                int index = 0;
                collectArrayInitializer(*def->init, dims, 0, index, initExprs);
                initValues.reserve(initExprs.size());
                for (const Expr *expr : initExprs) {
                    initValues.push_back(staticInitValue(expr, typeFrom(decl.type)));
                }
            }
            if (decl.isConst && dims.empty() && !initValues.empty()) {
                if (decl.type == TypeSpecifier::Float) {
                    constFloats_[def->name] = evalConstFloat(*def->init->expr);
                } else {
                    constInts_[def->name] = evalConstInt(*def->init->expr);
                }
            }
            globals_[def->name] = Binding{Value{-1, Type{TypeKind::Ptr}, "@" + def->name, true}, typeFrom(decl.type), dims, !dims.empty()};
            module_.globals.push_back(Global{def->name, typeFrom(decl.type), std::move(dims), std::move(initValues), decl.isConst});
            continue;
        }

        const int bytes = dims.empty() ? 4 : elements * 4;
        Value address = emitValue(Opcode::Alloca, Type{TypeKind::Ptr}, {}, def->name + ":" + std::to_string(bytes));
        scopes_.back()[def->name] = Binding{address, typeFrom(decl.type), dims, !dims.empty()};
        if (def->init && def->init->expr && dims.empty()) {
            emitVoid(Opcode::Store, {convert(buildExpr(*def->init->expr), typeFrom(decl.type)), address});
        } else if (def->init && !dims.empty()) {
            std::vector<const Expr *> init(static_cast<std::size_t>(elements), nullptr);
            int index = 0;
            collectArrayInitializer(*def->init, dims, 0, index, init);
            for (std::size_t i = 0; i < init.size(); ++i) {
                Value ptr = emitValue(Opcode::Gep, Type{TypeKind::Ptr}, {address, constant(Type{TypeKind::I32}, std::to_string(i))});
                Value value = init[i] ? buildExpr(*init[i]) : constant(typeFrom(decl.type), "0");
                emitVoid(Opcode::Store, {convert(value, typeFrom(decl.type)), ptr});
            }
        }
    }
}

void IRBuilder::buildFuncDef(const FuncDef &func) {
    module_.functions.push_back(Function{});
    function_ = &module_.functions.back();
    function_->name = func.name;
    function_->returnType = typeFrom(func.returnType);
    block_ = nullptr;
    nextValue_ = 0;
    scopes_.clear();
    scopes_.push_back({});

    for (const auto &param : func.params) {
        Value value{nextValue_++, param->isArray ? Type{TypeKind::Ptr} : typeFrom(param->type), "%" + param->name, false};
        function_->params.push_back(value);
    }

    startBlock("entry");
    for (std::size_t i = 0; i < func.params.size(); ++i) {
        const auto &param = func.params[i];
        std::vector<int> dims;
        if (param->isArray) {
            dims.push_back(-1);
            for (const auto &dim : param->dimensions) {
                dims.push_back(literalDimension(*dim));
            }
            scopes_.back()[param->name] = Binding{function_->params[i], typeFrom(param->type), std::move(dims), true};
        } else {
            Value address = emitValue(Opcode::Alloca, Type{TypeKind::Ptr}, {}, param->name + ":4");
            emitVoid(Opcode::Store, {function_->params[i], address});
            scopes_.back()[param->name] = Binding{address, typeFrom(param->type), {}, false};
        }
    }
    if (func.body) {
        buildBlock(*func.body);
    }
    if (!currentBlockTerminated()) {
        if (func.returnType == TypeSpecifier::Void) {
            emitVoid(Opcode::Ret, {}, "void");
        } else {
            emitVoid(Opcode::Ret, {constant(typeFrom(func.returnType), "0")});
        }
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
        emitVoid(Opcode::Store, {convert(buildExpr(*assign->value), lValueElementType(*assign->target)), buildLValue(*assign->target)});
    } else if (const auto *ret = dynamic_cast<const ReturnStmt *>(&stmt)) {
        if (ret->value) {
            emitVoid(Opcode::Ret, {convert(buildExpr(*ret->value), function_->returnType)});
        } else {
            emitVoid(Opcode::Ret, {}, "void");
        }
    } else if (const auto *whileStmt = dynamic_cast<const WhileStmt *>(&stmt)) {
        const std::string cond = newLabel("while.cond");
        const std::string body = newLabel("while.body");
        const std::string end = newLabel("while.end");
        emitVoid(Opcode::Br, {}, cond);
        startBlock(cond);
        emitVoid(Opcode::CondBr, {boolValue(buildExpr(*whileStmt->condition))}, body + ", " + end);
        startBlock(body);
        loops_.push_back({cond, end});
        buildStmt(*whileStmt->body);
        loops_.pop_back();
        if (!currentBlockTerminated()) {
            emitVoid(Opcode::Br, {}, cond);
        }
        startBlock(end);
    } else if (const auto *ifStmt = dynamic_cast<const IfStmt *>(&stmt)) {
        const std::string thenLabel = newLabel("if.then");
        const std::string elseLabel = ifStmt->elseBranch ? newLabel("if.else") : newLabel("if.end");
        const std::string endLabel = ifStmt->elseBranch ? newLabel("if.end") : elseLabel;
        emitVoid(Opcode::CondBr, {boolValue(buildExpr(*ifStmt->condition))}, thenLabel + ", " + elseLabel);
        startBlock(thenLabel);
        buildStmt(*ifStmt->thenBranch);
        if (!currentBlockTerminated()) {
            emitVoid(Opcode::Br, {}, endLabel);
        }
        if (ifStmt->elseBranch) {
            startBlock(elseLabel);
            buildStmt(*ifStmt->elseBranch);
            if (!currentBlockTerminated()) {
                emitVoid(Opcode::Br, {}, endLabel);
            }
        }
        startBlock(endLabel);
    } else if (dynamic_cast<const BreakStmt *>(&stmt)) {
        emitVoid(Opcode::Br, {}, loops_.back().second);
    } else if (dynamic_cast<const ContinueStmt *>(&stmt)) {
        emitVoid(Opcode::Br, {}, loops_.back().first);
    }
}

Value IRBuilder::buildExpr(const Expr &expr) {
    if (const auto *integer = dynamic_cast<const IntegerLiteral *>(&expr)) {
        return constant(Type{TypeKind::I32}, integer->value);
    }
    if (const auto *floating = dynamic_cast<const FloatLiteral *>(&expr)) {
        return constant(Type{TypeKind::F32}, floating->value);
    }
    if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&expr)) {
        const Binding binding = lookup(ref->name);
        if (binding.isArray) {
            return binding.address;
        }
        return emitValue(Opcode::Load, binding.valueType, {binding.address});
    }
    if (const auto *sub = dynamic_cast<const ArraySubscriptExpr *>(&expr)) {
        if (subscriptYieldsArray(*sub)) {
            return buildLValue(*sub);
        }
        Value address = buildLValue(*sub);
        return emitValue(Opcode::Load, lValueElementType(*sub), {address});
    }
    if (const auto *unary = dynamic_cast<const UnaryExpr *>(&expr)) {
        if (unary->op == UnaryOp::Plus) {
            return buildExpr(*unary->operand);
        }
        Value operand = buildExpr(*unary->operand);
        if (unary->op == UnaryOp::LogicalNot) {
            return emitValue(Opcode::Not, Type{TypeKind::I32}, {boolValue(operand)});
        }
        return emitValue(unary->op == UnaryOp::Minus ? Opcode::Neg : Opcode::Not,
                         operand.type,
                         {operand});
    }
    if (const auto *binary = dynamic_cast<const BinaryExpr *>(&expr)) {
        return buildBinaryExpr(*binary);
    }
    if (const auto *call = dynamic_cast<const CallExpr *>(&expr)) {
        std::vector<Value> args;
        const auto found = functions_.find(call->callee);
        Type ret = found == functions_.end() ? Type{TypeKind::I32} : found->second.returnType;
        for (std::size_t i = 0; i < call->args.size(); ++i) {
            Value arg = buildExpr(*call->args[i]);
            if (found != functions_.end() && i < found->second.params.size()) {
                arg = convert(arg, found->second.params[i]);
            }
            args.push_back(arg);
        }
        if (ret.kind == TypeKind::Void) {
            emitVoid(Opcode::Call, std::move(args), call->callee);
            return constant(Type{TypeKind::Void}, "void");
        }
        return emitValue(Opcode::Call, ret, std::move(args), call->callee);
    }
    return constant(Type{TypeKind::I32}, "0");
}

Value IRBuilder::buildBinaryExpr(const BinaryExpr &expr) {
    if (expr.op == BinaryOp::LogicalAnd || expr.op == BinaryOp::LogicalOr) {
        return buildShortCircuit(expr);
    }

    Opcode opcode = Opcode::Add;
    std::string predicate;
    switch (expr.op) {
    case BinaryOp::Add: opcode = Opcode::Add; break;
    case BinaryOp::Sub: opcode = Opcode::Sub; break;
    case BinaryOp::Mul: opcode = Opcode::Mul; break;
    case BinaryOp::Div: opcode = Opcode::Div; break;
    case BinaryOp::Mod: opcode = Opcode::Mod; break;
    case BinaryOp::Less: opcode = Opcode::ICmp; predicate = "lt"; break;
    case BinaryOp::Greater: opcode = Opcode::ICmp; predicate = "gt"; break;
    case BinaryOp::LessEqual: opcode = Opcode::ICmp; predicate = "le"; break;
    case BinaryOp::GreaterEqual: opcode = Opcode::ICmp; predicate = "ge"; break;
    case BinaryOp::Equal: opcode = Opcode::ICmp; predicate = "eq"; break;
    case BinaryOp::NotEqual: opcode = Opcode::ICmp; predicate = "ne"; break;
    case BinaryOp::LogicalAnd:
    case BinaryOp::LogicalOr:
        break;
    }

    Value lhs = buildExpr(*expr.lhs);
    Value rhs = buildExpr(*expr.rhs);
    const bool isCompare = opcode == Opcode::ICmp;
    const bool useFloat = lhs.type.kind == TypeKind::F32 || rhs.type.kind == TypeKind::F32;
    if (useFloat) {
        lhs = convert(lhs, Type{TypeKind::F32});
        rhs = convert(rhs, Type{TypeKind::F32});
    }
    if (isCompare && useFloat) {
        opcode = Opcode::FCmp;
    }
    Type type = isCompare ? Type{TypeKind::I32} : (useFloat ? Type{TypeKind::F32} : Type{TypeKind::I32});
    return emitValue(opcode, type, {lhs, rhs}, predicate);
}

Value IRBuilder::buildShortCircuit(const BinaryExpr &expr) {
    Value resultPtr = emitValue(Opcode::Alloca, Type{TypeKind::Ptr}, {}, "logic.tmp");
    const bool isAnd = expr.op == BinaryOp::LogicalAnd;
    emitVoid(Opcode::Store, {constant(Type{TypeKind::I32}, isAnd ? "0" : "1"), resultPtr});

    const std::string rhsLabel = newLabel("logic.rhs");
    const std::string endLabel = newLabel("logic.end");
    Value lhs = boolValue(buildExpr(*expr.lhs));
    emitVoid(Opcode::CondBr, {lhs}, isAnd ? rhsLabel + ", " + endLabel : endLabel + ", " + rhsLabel);

    startBlock(rhsLabel);
    Value rhs = boolValue(buildExpr(*expr.rhs));
    emitVoid(Opcode::Store, {rhs, resultPtr});
    emitVoid(Opcode::Br, {}, endLabel);

    startBlock(endLabel);
    return emitValue(Opcode::Load, Type{TypeKind::I32}, {resultPtr});
}

Value IRBuilder::convert(Value value, Type target) {
    if (target.kind == TypeKind::Void || value.type.kind == target.kind) {
        return value;
    }
    if (value.constant) {
        if (target.kind == TypeKind::I32 && value.type.kind == TypeKind::F32) {
            return constant(target, std::to_string(static_cast<long long>(std::strtod(value.name.c_str(), nullptr))));
        }
        if (target.kind == TypeKind::F32 && value.type.kind == TypeKind::I32) {
            return constant(target, value.name);
        }
    }
    if (target.kind == TypeKind::I32 && value.type.kind == TypeKind::F32) {
        return emitValue(Opcode::Cast, target, {value}, "f2i");
    }
    if (target.kind == TypeKind::F32 && value.type.kind == TypeKind::I32) {
        return emitValue(Opcode::Cast, target, {value}, "i2f");
    }
    return value;
}

Value IRBuilder::boolValue(Value value) {
    if (value.type.kind == TypeKind::I32) {
        if (value.constant) {
            return constant(Type{TypeKind::I32}, std::strtoll(value.name.c_str(), nullptr, 0) == 0 ? "0" : "1");
        }
        return value;
    }
    if (value.type.kind == TypeKind::F32) {
        if (value.constant) {
            return constant(Type{TypeKind::I32}, std::strtof(value.name.c_str(), nullptr) == 0.0f ? "0" : "1");
        }
        return emitValue(Opcode::FCmp, Type{TypeKind::I32}, {value, constant(Type{TypeKind::F32}, "0")}, "ne");
    }
    return value;
}

Value IRBuilder::buildLValue(const Expr &expr) {
    if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&expr)) {
        return lookup(ref->name).address;
    }
    if (const auto *sub = dynamic_cast<const ArraySubscriptExpr *>(&expr)) {
        std::vector<const Expr *> indices;
        const DeclRefExpr *base = collectArrayBase(*sub, indices);
        if (!base) {
            return constant(Type{TypeKind::Ptr}, "<invalid-lvalue>");
        }
        const Binding binding = lookup(base->name);
        Value linear = constant(Type{TypeKind::I32}, "0");
        for (std::size_t i = 0; i < indices.size(); ++i) {
            Value term = buildExpr(*indices[i]);
            int stride = 1;
            for (std::size_t j = i + 1; j < binding.dims.size(); ++j) {
                stride *= binding.dims[j] < 0 ? 1 : binding.dims[j];
            }
            if (stride != 1) {
                term = emitValue(Opcode::Mul, Type{TypeKind::I32}, {term, constant(Type{TypeKind::I32}, std::to_string(stride))});
            }
            linear = emitValue(Opcode::Add, Type{TypeKind::I32}, {linear, term});
        }
        return emitValue(Opcode::Gep, Type{TypeKind::Ptr}, {binding.address, linear});
    }
    return constant(Type{TypeKind::Ptr}, "<invalid-lvalue>");
}

const DeclRefExpr *IRBuilder::collectArrayBase(const Expr &expr, std::vector<const Expr *> &indices) const {
    if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&expr)) {
        return ref;
    }
    if (const auto *sub = dynamic_cast<const ArraySubscriptExpr *>(&expr)) {
        const DeclRefExpr *base = collectArrayBase(*sub->base, indices);
        indices.push_back(sub->index.get());
        return base;
    }
    return nullptr;
}

bool IRBuilder::subscriptYieldsArray(const ArraySubscriptExpr &expr) const {
    std::vector<const Expr *> indices;
    const DeclRefExpr *base = collectArrayBase(expr, indices);
    if (!base) {
        return false;
    }
    const Binding binding = lookup(base->name);
    return indices.size() < binding.dims.size();
}

Type IRBuilder::lValueElementType(const Expr &expr) const {
    if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&expr)) {
        return lookup(ref->name).valueType;
    }
    if (const auto *sub = dynamic_cast<const ArraySubscriptExpr *>(&expr)) {
        return lValueElementType(*sub->base);
    }
    return Type{TypeKind::I32};
}

Value IRBuilder::emitValue(Opcode opcode, Type type, std::vector<Value> operands, std::string text) {
    Value result{nextValue_++, type, {}, false};
    if (block_) {
        block_->instructions.push_back(Instruction{result.id, type, opcode, std::move(operands), std::move(text)});
    }
    return result;
}

void IRBuilder::emitVoid(Opcode opcode, std::vector<Value> operands, std::string text) {
    if (block_) {
        block_->instructions.push_back(Instruction{-1, Type{TypeKind::Void}, opcode, std::move(operands), std::move(text)});
    }
}

void IRBuilder::startBlock(std::string name) {
    function_->blocks.push_back(BasicBlock{std::move(name), {}, {}, {}});
    block_ = &function_->blocks.back();
}

bool IRBuilder::currentBlockTerminated() const {
    if (!block_ || block_->instructions.empty()) {
        return false;
    }
    const Opcode opcode = block_->instructions.back().opcode;
    return opcode == Opcode::Ret || opcode == Opcode::Br || opcode == Opcode::CondBr;
}

Type IRBuilder::typeFrom(TypeSpecifier spec) const {
    switch (spec) {
    case TypeSpecifier::Void:
        return Type{TypeKind::Void};
    case TypeSpecifier::Int:
        return Type{TypeKind::I32};
    case TypeSpecifier::Float:
        return Type{TypeKind::F32};
    }
    return Type{TypeKind::Void};
}

std::string IRBuilder::newLabel(const std::string &prefix) {
    return prefix + "." + std::to_string(nextLabel_++);
}

IRBuilder::Binding IRBuilder::lookup(const std::string &name) const {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
        const auto found = it->find(name);
        if (found != it->end()) {
            return found->second;
        }
    }
    const auto global = globals_.find(name);
    if (global != globals_.end()) {
        return global->second;
    }
    return Binding{Value{-1, Type{TypeKind::Ptr}, "@" + name, true}, Type{TypeKind::I32}, {}, false};
}

Value IRBuilder::constant(Type type, std::string text) const {
    return Value{-1, type, std::move(text), true};
}

int IRBuilder::literalDimension(const Expr &expr) const {
    return static_cast<int>(evalConstInt(expr));
}

long long IRBuilder::evalConstInt(const Expr &expr) const {
    if (const auto *integer = dynamic_cast<const IntegerLiteral *>(&expr)) {
        return std::strtoll(integer->value.c_str(), nullptr, 0);
    }
    if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&expr)) {
        const auto found = constInts_.find(ref->name);
        if (found != constInts_.end()) {
            return found->second;
        }
        const auto floatFound = constFloats_.find(ref->name);
        return floatFound == constFloats_.end() ? 1 : static_cast<long long>(floatFound->second);
    }
    if (const auto *floating = dynamic_cast<const FloatLiteral *>(&expr)) {
        return static_cast<long long>(std::strtod(floating->value.c_str(), nullptr));
    }
    if (const auto *unary = dynamic_cast<const UnaryExpr *>(&expr)) {
        const long long value = evalConstInt(*unary->operand);
        if (unary->op == UnaryOp::Minus) {
            return -value;
        }
        if (unary->op == UnaryOp::LogicalNot) {
            return value == 0;
        }
        return value;
    }
    if (const auto *binary = dynamic_cast<const BinaryExpr *>(&expr)) {
        const long long lhs = evalConstInt(*binary->lhs);
        const long long rhs = evalConstInt(*binary->rhs);
        switch (binary->op) {
        case BinaryOp::Add: return lhs + rhs;
        case BinaryOp::Sub: return lhs - rhs;
        case BinaryOp::Mul: return lhs * rhs;
        case BinaryOp::Div: return rhs == 0 ? 1 : lhs / rhs;
        case BinaryOp::Mod: return rhs == 0 ? 1 : lhs % rhs;
        case BinaryOp::Less: return lhs < rhs;
        case BinaryOp::Greater: return lhs > rhs;
        case BinaryOp::LessEqual: return lhs <= rhs;
        case BinaryOp::GreaterEqual: return lhs >= rhs;
        case BinaryOp::Equal: return lhs == rhs;
        case BinaryOp::NotEqual: return lhs != rhs;
        case BinaryOp::LogicalAnd: return (lhs != 0) && (rhs != 0);
        case BinaryOp::LogicalOr: return (lhs != 0) || (rhs != 0);
        }
    }
    return 1;
}

double IRBuilder::evalConstFloat(const Expr &expr) const {
    if (const auto *floating = dynamic_cast<const FloatLiteral *>(&expr)) {
        return static_cast<double>(std::strtof(floating->value.c_str(), nullptr));
    }
    if (const auto *integer = dynamic_cast<const IntegerLiteral *>(&expr)) {
        return static_cast<double>(static_cast<float>(std::strtoll(integer->value.c_str(), nullptr, 0)));
    }
    if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&expr)) {
        const auto floatFound = constFloats_.find(ref->name);
        if (floatFound != constFloats_.end()) {
            return floatFound->second;
        }
        const auto intFound = constInts_.find(ref->name);
        return intFound == constInts_.end() ? 1.0 : static_cast<double>(intFound->second);
    }
    if (const auto *unary = dynamic_cast<const UnaryExpr *>(&expr)) {
        const float value = static_cast<float>(evalConstFloat(*unary->operand));
        if (unary->op == UnaryOp::Minus) {
            return static_cast<double>(-value);
        }
        if (unary->op == UnaryOp::LogicalNot) {
            return value == 0.0;
        }
        return static_cast<double>(value);
    }
    if (const auto *binary = dynamic_cast<const BinaryExpr *>(&expr)) {
        const float lhs = static_cast<float>(evalConstFloat(*binary->lhs));
        const float rhs = static_cast<float>(evalConstFloat(*binary->rhs));
        switch (binary->op) {
        case BinaryOp::Add: return static_cast<double>(static_cast<float>(lhs + rhs));
        case BinaryOp::Sub: return static_cast<double>(static_cast<float>(lhs - rhs));
        case BinaryOp::Mul: return static_cast<double>(static_cast<float>(lhs * rhs));
        case BinaryOp::Div: return rhs == 0.0f ? 1.0 : static_cast<double>(static_cast<float>(lhs / rhs));
        case BinaryOp::Less: return lhs < rhs;
        case BinaryOp::Greater: return lhs > rhs;
        case BinaryOp::LessEqual: return lhs <= rhs;
        case BinaryOp::GreaterEqual: return lhs >= rhs;
        case BinaryOp::Equal: return lhs == rhs;
        case BinaryOp::NotEqual: return lhs != rhs;
        case BinaryOp::LogicalAnd: return (lhs != 0.0) && (rhs != 0.0);
        case BinaryOp::LogicalOr: return (lhs != 0.0) || (rhs != 0.0);
        case BinaryOp::Mod: return 1.0;
        }
    }
    return 1.0;
}

void IRBuilder::collectArrayInitializer(const InitVal &init, const std::vector<int> &dims, int level, int &index, std::vector<const Expr *> &values) const {
    if (index >= static_cast<int>(values.size())) {
        return;
    }
    if (init.expr) {
        values[static_cast<std::size_t>(index++)] = init.expr.get();
        return;
    }
    if (dims.empty()) {
        return;
    }
    const int currentLimit = level < static_cast<int>(dims.size()) ? productFrom(dims, level) : 1;
    const int listEnd = std::min(static_cast<int>(values.size()), index + currentLimit);
    for (const auto &element : init.elements) {
        if (index >= listEnd) {
            break;
        }
        if (element->expr || level + 1 >= static_cast<int>(dims.size())) {
            collectArrayInitializer(*element, dims, level + 1, index, values);
        } else {
            const int childStart = index;
            collectArrayInitializer(*element, dims, level + 1, index, values);
            index = childStart + productFrom(dims, level + 1);
        }
    }
}

int IRBuilder::productFrom(const std::vector<int> &dims, int level) {
    int product = 1;
    for (std::size_t i = static_cast<std::size_t>(level); i < dims.size(); ++i) {
        product *= dims[i];
    }
    return product;
}

std::string IRBuilder::staticInitValue(const Expr *expr, Type type) const {
    if (!expr) {
        return "0";
    }
    if (type.kind == TypeKind::F32) {
        return floatWord(evalConstFloat(*expr));
    }
    return std::to_string(evalConstInt(*expr));
}

std::string IRBuilder::floatWord(double value) {
    const float narrowed = static_cast<float>(value);
    std::uint32_t bits = 0;
    std::memcpy(&bits, &narrowed, sizeof(bits));
    return std::to_string(bits);
}

} // namespace sysyc::ir
