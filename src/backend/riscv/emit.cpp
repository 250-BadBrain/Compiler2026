#include "emit.hpp"

#include <cstdint>
#include <cstring>
#include <optional>
#include <ostream>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sysyc::riscv {

namespace {

class CodeGen {
public:
    CodeGen(const TranslationUnit &unit, std::ostream &out) : unit_(unit), out_(out) {}

    void run() {
        collectFunctionInfo();
        emitGlobals();
        out_ << "\t.text\n";
        for (const auto &decl : unit_.declarations) {
            if (const auto *func = dynamic_cast<const FuncDef *>(decl.get())) {
                emitFunction(*func);
            }
        }
    }

private:
    struct LoopLabels {
        std::string continueLabel;
        std::string breakLabel;
    };

    struct ObjectInfo {
        bool found = false;
        int offset = 0;
        std::vector<int> dims;
        bool global = false;
        TypeSpecifier type = TypeSpecifier::Int;
    };

    struct FunctionInfo {
        TypeSpecifier returnType = TypeSpecifier::Int;
        std::vector<TypeSpecifier> params;
    };

    enum class ValueKind {
        Int,
        Float,
        Void,
    };

    void emitGlobals() {
        bool emitted = false;
        for (const auto &decl : unit_.declarations) {
            const auto *var = dynamic_cast<const VarDecl *>(decl.get());
            if (!var) {
                continue;
            }
            if (!emitted) {
                out_ << "\t.data\n";
                emitted = true;
            }
            for (const auto &def : var->defs) {
                globals_[def->name] = ObjectInfo{true, 0, dimensionsOf(*def), true, var->type};
                out_ << "\t.globl " << def->name << "\n";
                out_ << "\t.align 2\n";
                out_ << def->name << ":\n";
                if (def->dimensions.empty()) {
                    if (def->init && def->init->expr) {
                        if (var->type == TypeSpecifier::Float) {
                            const float value = static_cast<float>(evalConstFloat(*def->init->expr).value_or(0.0));
                            out_ << "\t.word " << floatBits(value) << "\n";
                            if (var->isConst) {
                                constFloatValues_[def->name] = value;
                            }
                        } else {
                            const auto value = evalConstInt(*def->init->expr).value_or(0);
                            out_ << "\t.word " << value << "\n";
                            if (var->isConst) {
                                constValues_[def->name] = value;
                            }
                        }
                    } else {
                        out_ << "\t.word 0\n";
                    }
                } else {
                    std::vector<const Expr *> values(static_cast<std::size_t>(arrayElementCount(*def)), nullptr);
                    if (def->init) {
                        int index = 0;
                        collectArrayInitializer(*def->init, dimensionsOf(*def), 0, index, values);
                    }
                    emitStaticArrayInitializer(values, var->type);
                }
            }
        }
    }

    void emitStaticArrayInitializer(const std::vector<const Expr *> &values, TypeSpecifier type) {
        int zeroRun = 0;
        const auto flushZeroRun = [&]() {
            if (zeroRun > 0) {
                out_ << "\t.zero " << zeroRun * 4 << "\n";
                zeroRun = 0;
            }
        };

        for (const Expr *expr : values) {
            const std::string value = staticInitializerValue(expr, type);
            if (value == "0") {
                ++zeroRun;
                continue;
            }
            flushZeroRun();
            out_ << "\t.word " << value << "\n";
        }
        flushZeroRun();
    }

    void emitFunction(const FuncDef &func) {
        currentFunction_ = &func;
        scopes_.clear();
        constScopes_.clear();
        loops_.clear();
        nextOffset_ = -24;
        currentFrameSize_ = estimateFrameSize(func);
        epilogueLabel_ = newLabel(".L" + func.name + ".ret");

        out_ << "\t.align 1\n";
        out_ << "\t.globl " << func.name << "\n";
        out_ << "\t.type " << func.name << ", @function\n";
        out_ << func.name << ":\n";
        out_ << "\tli t0, " << currentFrameSize_ << "\n";
        out_ << "\tsub sp, sp, t0\n";
        out_ << "\tadd t1, sp, t0\n";
        out_ << "\tsd ra, -8(t1)\n";
        out_ << "\tsd s0, -16(t1)\n";
        out_ << "\tmv s0, t1\n";

        pushScope();
        int intParam = 0;
        int floatParam = 0;
        int stackParam = 0;
        for (std::size_t i = 0; i < func.params.size(); ++i) {
            const TypeSpecifier paramType = func.params[i]->type;
            const int offset = func.params[i]->isArray
                                   ? allocateLocalBytes(func.params[i]->name, 8, paramDimensions(*func.params[i]), paramType)
                                   : allocateLocal(func.params[i]->name, paramType);
            if (func.params[i]->isArray || paramType != TypeSpecifier::Float) {
                if (intParam < 8) {
                    emitStore("a" + std::to_string(intParam), "s0", offset, "sd");
                } else {
                    emitLoad("t0", "s0", stackParam * 8, "ld");
                    emitStore("t0", "s0", offset, "sd");
                    ++stackParam;
                }
                ++intParam;
            } else {
                if (floatParam < 8) {
                    emitFStore("fa" + std::to_string(floatParam), "s0", offset);
                } else {
                    emitFLoad("ft0", "s0", stackParam * 8);
                    emitFStore("ft0", "s0", offset);
                    ++stackParam;
                }
                ++floatParam;
            }
        }
        if (func.body) {
            emitBlock(*func.body, false);
        }
        if (func.returnType == TypeSpecifier::Void) {
            out_ << "\tj " << epilogueLabel_ << "\n";
        } else {
            out_ << "\tli a0, 0\n";
            out_ << "\tj " << epilogueLabel_ << "\n";
        }
        popScope();

        out_ << epilogueLabel_ << ":\n";
        out_ << "\tmv t1, s0\n";
        out_ << "\tld ra, -8(t1)\n";
        out_ << "\tld s0, -16(t1)\n";
        out_ << "\tmv sp, t1\n";
        out_ << "\tret\n";
        out_ << "\t.size " << func.name << ", .-" << func.name << "\n";
        currentFunction_ = nullptr;
        constScopes_.clear();
    }

    void emitBlock(const BlockStmt &block, bool createScope = true) {
        if (createScope) {
            pushScope();
        }
        for (const auto &item : block.items) {
            if (item.decl) {
                if (const auto *var = dynamic_cast<const VarDecl *>(item.decl.get())) {
                    emitLocalDecl(*var);
                }
            } else if (item.stmt) {
                emitStmt(*item.stmt);
            }
        }
        if (createScope) {
            popScope();
        }
    }

    void emitLocalDecl(const VarDecl &decl) {
        for (const auto &def : decl.defs) {
            if (!def->dimensions.empty()) {
                const int bytes = arrayStorageBytes(*def);
                const int offset = allocateLocalBytes(def->name, bytes, dimensionsOf(*def), decl.type);
                if (def->init) {
                    auto values = std::vector<const Expr *>(static_cast<std::size_t>(arrayElementCount(*def)), nullptr);
                    int index = 0;
                    collectArrayInitializer(*def->init, dimensionsOf(*def), 0, index, values);
                    emitArrayInitializer(offset, values, decl.type);
                }
                continue;
            }
            const int offset = allocateLocal(def->name, decl.type);
            if (def->init && def->init->expr) {
                emitExpr(*def->init->expr);
                if (decl.type == TypeSpecifier::Float) {
                    ensureFloat(valueKind(*def->init->expr));
                    emitFStore("fa0", "s0", offset);
                } else {
                    ensureInt(valueKind(*def->init->expr));
                    emitStore("a0", "s0", offset, "sd");
                }
                if (decl.isConst) {
                    if (auto value = evalConstInt(*def->init->expr)) {
                        constScopes_.back()[def->name] = *value;
                    }
                }
            }
        }
    }

    void emitStmt(const Stmt &stmt) {
        if (const auto *block = dynamic_cast<const BlockStmt *>(&stmt)) {
            emitBlock(*block);
        } else if (const auto *expr = dynamic_cast<const ExprStmt *>(&stmt)) {
            if (expr->expr) {
                emitExpr(*expr->expr);
            }
        } else if (const auto *assign = dynamic_cast<const AssignStmt *>(&stmt)) {
            emitExpr(*assign->value);
            emitStoreToLValue(*assign->target, valueKind(*assign->value));
        } else if (const auto *ifStmt = dynamic_cast<const IfStmt *>(&stmt)) {
            const std::string elseLabel = newLabel(".Lelse");
            const std::string endLabel = newLabel(".Lifend");
            emitExpr(*ifStmt->condition);
            emitBoolFromValue(valueKind(*ifStmt->condition));
            out_ << "\tbeqz a0, " << (ifStmt->elseBranch ? elseLabel : endLabel) << "\n";
            emitStmt(*ifStmt->thenBranch);
            out_ << "\tj " << endLabel << "\n";
            if (ifStmt->elseBranch) {
                out_ << elseLabel << ":\n";
                emitStmt(*ifStmt->elseBranch);
            }
            out_ << endLabel << ":\n";
        } else if (const auto *whileStmt = dynamic_cast<const WhileStmt *>(&stmt)) {
            const std::string condLabel = newLabel(".Lwhile.cond");
            const std::string bodyLabel = newLabel(".Lwhile.body");
            const std::string endLabel = newLabel(".Lwhile.end");
            loops_.push_back(LoopLabels{condLabel, endLabel});
            out_ << condLabel << ":\n";
            emitExpr(*whileStmt->condition);
            emitBoolFromValue(valueKind(*whileStmt->condition));
            out_ << "\tbnez a0, " << bodyLabel << "\n";
            out_ << "\tj " << endLabel << "\n";
            out_ << bodyLabel << ":\n";
            emitStmt(*whileStmt->body);
            out_ << "\tj " << condLabel << "\n";
            out_ << endLabel << ":\n";
            loops_.pop_back();
        } else if (dynamic_cast<const BreakStmt *>(&stmt)) {
            out_ << "\tj " << loops_.back().breakLabel << "\n";
        } else if (dynamic_cast<const ContinueStmt *>(&stmt)) {
            out_ << "\tj " << loops_.back().continueLabel << "\n";
        } else if (const auto *ret = dynamic_cast<const ReturnStmt *>(&stmt)) {
            if (ret->value) {
                emitExpr(*ret->value);
                if (currentFunction_ && currentFunction_->returnType == TypeSpecifier::Float) {
                    ensureFloat(valueKind(*ret->value));
                } else {
                    ensureInt(valueKind(*ret->value));
                }
            }
            out_ << "\tj " << epilogueLabel_ << "\n";
        }
    }

    void emitExpr(const Expr &expr) {
        if (const auto *integer = dynamic_cast<const IntegerLiteral *>(&expr)) {
            out_ << "\tli a0, " << integer->value << "\n";
        } else if (const auto *floating = dynamic_cast<const FloatLiteral *>(&expr)) {
            emitFloatImmediate(static_cast<float>(std::strtof(floating->value.c_str(), nullptr)));
        } else if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&expr)) {
            const auto object = lookupLocal(ref->name);
            if (object.found && !object.dims.empty() && !object.global && isArrayParameter(ref->name)) {
                emitLoad("a0", "s0", object.offset, "ld");
            } else if (object.found && !object.dims.empty() && !object.global) {
                emitAddress("a0", "s0", object.offset);
            } else if (object.found && !object.dims.empty() && object.global) {
                out_ << "\tla a0, " << ref->name << "\n";
            } else if (object.found && !object.global && object.type == TypeSpecifier::Float) {
                emitFLoad("fa0", "s0", object.offset);
            } else if (object.found && !object.global) {
                emitLoad("a0", "s0", object.offset, "ld");
            } else if (object.type == TypeSpecifier::Float) {
                out_ << "\tla t0, " << ref->name << "\n";
                out_ << "\tflw fa0, 0(t0)\n";
            } else {
                out_ << "\tla t0, " << ref->name << "\n";
                out_ << "\tlw a0, 0(t0)\n";
            }
        } else if (const auto *sub = dynamic_cast<const ArraySubscriptExpr *>(&expr)) {
            emitArrayAddress(*sub);
            if (!subscriptYieldsArray(*sub)) {
                const auto kind = valueKind(expr);
                if (kind == ValueKind::Float) {
                    out_ << "\tflw fa0, 0(a0)\n";
                } else {
                    out_ << "\tlw a0, 0(a0)\n";
                }
            }
        } else if (const auto *call = dynamic_cast<const CallExpr *>(&expr)) {
            emitCall(*call);
        } else if (const auto *unary = dynamic_cast<const UnaryExpr *>(&expr)) {
            emitExpr(*unary->operand);
            if (valueKind(*unary->operand) == ValueKind::Float && unary->op == UnaryOp::Minus) {
                out_ << "\tfneg.s fa0, fa0\n";
            } else if (unary->op == UnaryOp::Minus) {
                out_ << "\tnegw a0, a0\n";
            } else if (unary->op == UnaryOp::LogicalNot) {
                if (valueKind(*unary->operand) == ValueKind::Float) {
                    out_ << "\tfcvt.w.s a0, fa0, rtz\n";
                }
                out_ << "\tseqz a0, a0\n";
            }
        } else if (const auto *binary = dynamic_cast<const BinaryExpr *>(&expr)) {
            emitBinary(*binary);
        } else {
            out_ << "\tli a0, 0\n";
        }
    }

    void emitBinary(const BinaryExpr &binary) {
        if (binary.op == BinaryOp::LogicalAnd || binary.op == BinaryOp::LogicalOr) {
            emitShortCircuit(binary);
            return;
        }
        if (valueKind(*binary.lhs) == ValueKind::Float || valueKind(*binary.rhs) == ValueKind::Float) {
            emitFloatBinary(binary);
            return;
        }
        if (emitIntImmediateBinary(binary)) {
            return;
        }
        emitExpr(*binary.lhs);
        pushA0();
        emitExpr(*binary.rhs);
        popTo("t0");
        switch (binary.op) {
        case BinaryOp::Add:
            out_ << "\taddw a0, t0, a0\n";
            break;
        case BinaryOp::Sub:
            out_ << "\tsubw a0, t0, a0\n";
            break;
        case BinaryOp::Mul:
            out_ << "\tmulw a0, t0, a0\n";
            break;
        case BinaryOp::Div:
            out_ << "\tdivw a0, t0, a0\n";
            break;
        case BinaryOp::Mod:
            out_ << "\tremw a0, t0, a0\n";
            break;
        case BinaryOp::Less:
            out_ << "\tslt a0, t0, a0\n";
            break;
        case BinaryOp::Greater:
            out_ << "\tslt a0, a0, t0\n";
            break;
        case BinaryOp::LessEqual:
            out_ << "\tslt a0, a0, t0\n";
            out_ << "\txori a0, a0, 1\n";
            break;
        case BinaryOp::GreaterEqual:
            out_ << "\tslt a0, t0, a0\n";
            out_ << "\txori a0, a0, 1\n";
            break;
        case BinaryOp::Equal:
            out_ << "\tsub a0, t0, a0\n";
            out_ << "\tseqz a0, a0\n";
            break;
        case BinaryOp::NotEqual:
            out_ << "\tsub a0, t0, a0\n";
            out_ << "\tsnez a0, a0\n";
            break;
        case BinaryOp::LogicalAnd:
        case BinaryOp::LogicalOr:
            break;
        }
    }

    bool emitIntImmediateBinary(const BinaryExpr &binary) {
        const auto rhs = evalConstInt(*binary.rhs);
        if (!rhs) {
            return false;
        }

        switch (binary.op) {
        case BinaryOp::Add:
            if (!fitsI12(*rhs)) {
                return false;
            }
            emitExpr(*binary.lhs);
            out_ << "\taddiw a0, a0, " << *rhs << "\n";
            return true;
        case BinaryOp::Sub:
            if (!fitsI12(-*rhs)) {
                return false;
            }
            emitExpr(*binary.lhs);
            out_ << "\taddiw a0, a0, " << -*rhs << "\n";
            return true;
        case BinaryOp::Mul:
            emitExpr(*binary.lhs);
            out_ << "\tli t0, " << *rhs << "\n";
            out_ << "\tmulw a0, a0, t0\n";
            return true;
        case BinaryOp::Div:
            if (*rhs == 0) {
                return false;
            }
            emitExpr(*binary.lhs);
            out_ << "\tli t0, " << *rhs << "\n";
            out_ << "\tdivw a0, a0, t0\n";
            return true;
        case BinaryOp::Mod:
            if (*rhs == 0) {
                return false;
            }
            emitExpr(*binary.lhs);
            out_ << "\tli t0, " << *rhs << "\n";
            out_ << "\tremw a0, a0, t0\n";
            return true;
        case BinaryOp::Less:
            if (!fitsI12(*rhs)) {
                return false;
            }
            emitExpr(*binary.lhs);
            out_ << "\tslti a0, a0, " << *rhs << "\n";
            return true;
        case BinaryOp::LessEqual:
            if (!fitsI12(*rhs + 1)) {
                return false;
            }
            emitExpr(*binary.lhs);
            out_ << "\tslti a0, a0, " << (*rhs + 1) << "\n";
            return true;
        case BinaryOp::GreaterEqual:
            if (!fitsI12(*rhs)) {
                return false;
            }
            emitExpr(*binary.lhs);
            out_ << "\tslti a0, a0, " << *rhs << "\n";
            out_ << "\txori a0, a0, 1\n";
            return true;
        case BinaryOp::Equal:
        case BinaryOp::NotEqual:
        case BinaryOp::Greater:
            emitExpr(*binary.lhs);
            out_ << "\tli t0, " << *rhs << "\n";
            if (binary.op == BinaryOp::Greater) {
                out_ << "\tslt a0, t0, a0\n";
            } else {
                out_ << "\tsubw a0, a0, t0\n";
                out_ << (binary.op == BinaryOp::Equal ? "\tseqz a0, a0\n" : "\tsnez a0, a0\n");
            }
            return true;
        case BinaryOp::LogicalAnd:
        case BinaryOp::LogicalOr:
            return false;
        }
        return false;
    }

    void emitFloatBinary(const BinaryExpr &binary) {
        emitExpr(*binary.lhs);
        ensureFloat(valueKind(*binary.lhs));
        pushFA0();
        emitExpr(*binary.rhs);
        ensureFloat(valueKind(*binary.rhs));
        popFTo("ft0");
        switch (binary.op) {
        case BinaryOp::Add:
            out_ << "\tfadd.s fa0, ft0, fa0\n";
            break;
        case BinaryOp::Sub:
            out_ << "\tfsub.s fa0, ft0, fa0\n";
            break;
        case BinaryOp::Mul:
            out_ << "\tfmul.s fa0, ft0, fa0\n";
            break;
        case BinaryOp::Div:
            out_ << "\tfdiv.s fa0, ft0, fa0\n";
            break;
        case BinaryOp::Less:
            out_ << "\tflt.s a0, ft0, fa0\n";
            break;
        case BinaryOp::Greater:
            out_ << "\tflt.s a0, fa0, ft0\n";
            break;
        case BinaryOp::LessEqual:
            out_ << "\tfle.s a0, ft0, fa0\n";
            break;
        case BinaryOp::GreaterEqual:
            out_ << "\tfle.s a0, fa0, ft0\n";
            break;
        case BinaryOp::Equal:
            out_ << "\tfeq.s a0, ft0, fa0\n";
            break;
        case BinaryOp::NotEqual:
            out_ << "\tfeq.s a0, ft0, fa0\n";
            out_ << "\txori a0, a0, 1\n";
            break;
        case BinaryOp::Mod:
        case BinaryOp::LogicalAnd:
        case BinaryOp::LogicalOr:
            out_ << "\tli a0, 0\n";
            break;
        }
    }

    void emitShortCircuit(const BinaryExpr &binary) {
        const std::string doneLabel = newLabel(".Llogic.done");
        const std::string rhsLabel = newLabel(".Llogic.rhs");
        if (binary.op == BinaryOp::LogicalAnd) {
            emitExpr(*binary.lhs);
            emitBoolFromValue(valueKind(*binary.lhs));
            out_ << "\tbnez a0, " << rhsLabel << "\n";
            out_ << "\tli a0, 0\n";
            out_ << "\tj " << doneLabel << "\n";
            out_ << rhsLabel << ":\n";
            emitExpr(*binary.rhs);
            emitBoolFromValue(valueKind(*binary.rhs));
            out_ << "\tsnez a0, a0\n";
            out_ << doneLabel << ":\n";
            return;
        }

        emitExpr(*binary.lhs);
        emitBoolFromValue(valueKind(*binary.lhs));
        out_ << "\tbeqz a0, " << rhsLabel << "\n";
        out_ << "\tli a0, 1\n";
        out_ << "\tj " << doneLabel << "\n";
        out_ << rhsLabel << ":\n";
        emitExpr(*binary.rhs);
        emitBoolFromValue(valueKind(*binary.rhs));
        out_ << "\tsnez a0, a0\n";
        out_ << doneLabel << ":\n";
    }

    void emitCall(const CallExpr &call) {
        if (call.callee == "starttime" || call.callee == "stoptime") {
            return;
        }
        int intCount = 0;
        int floatCount = 0;
        for (std::size_t i = 0; i < call.args.size(); ++i) {
            if (callArgKind(call, i) == ValueKind::Float) {
                ++floatCount;
            } else {
                ++intCount;
            }
        }
        const int stackIntCount = intCount > 8 ? intCount - 8 : 0;
        const int stackFloatCount = floatCount > 8 ? floatCount - 8 : 0;
        const int stackCount = stackIntCount + stackFloatCount;
        const int stackBytes = alignTo(stackCount * 8, 16);
        if (stackBytes != 0) {
            emitAdjustSp(-stackBytes);
        }
        out_ << "\tmv t5, sp\n";
        int intIndex = 0;
        int floatIndex = 0;
        int stackIndex = 0;
        std::vector<std::pair<ValueKind, int>> pushedRegs;
        for (std::size_t i = 0; i < call.args.size(); ++i) {
            const auto &arg = call.args[i];
            const ValueKind actualKind = valueKind(*arg);
            const ValueKind targetKind = callArgKind(call, i);
            emitExpr(*arg);
            if (targetKind == ValueKind::Float) {
                ensureFloat(actualKind);
                if (floatIndex < 8) {
                    pushFA0();
                    pushedRegs.push_back({ValueKind::Float, floatIndex});
                } else {
                    emitFStore("fa0", "t5", stackIndex * 8);
                    ++stackIndex;
                }
                ++floatIndex;
            } else {
                ensureInt(actualKind);
                if (intIndex < 8) {
                    pushA0();
                    pushedRegs.push_back({ValueKind::Int, intIndex});
                } else {
                    emitStore("a0", "t5", stackIndex * 8, "sd");
                    ++stackIndex;
                }
                ++intIndex;
            }
        }
        for (auto it = pushedRegs.rbegin(); it != pushedRegs.rend(); ++it) {
            if (it->first == ValueKind::Float) {
                popFTo("fa" + std::to_string(it->second));
            } else {
                popTo("a" + std::to_string(it->second));
            }
        }
        out_ << "\tcall " << call.callee << "\n";
        if (stackBytes != 0) {
            emitAdjustSp(stackBytes);
        }
    }

    void emitStoreToLValue(const Expr &target, ValueKind sourceKind) {
        if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&target)) {
            const auto object = lookupLocal(ref->name);
            if (object.type == TypeSpecifier::Float) {
                ensureFloat(sourceKind);
            } else {
                ensureInt(sourceKind);
            }
            if (object.found && !object.global && object.type == TypeSpecifier::Float) {
                emitFStore("fa0", "s0", object.offset);
            } else if (object.found && !object.global) {
                emitStore("a0", "s0", object.offset, "sd");
            } else if (object.type == TypeSpecifier::Float) {
                out_ << "\tla t0, " << ref->name << "\n";
                out_ << "\tfsw fa0, 0(t0)\n";
            } else {
                out_ << "\tla t0, " << ref->name << "\n";
                out_ << "\tsw a0, 0(t0)\n";
            }
        } else if (const auto *sub = dynamic_cast<const ArraySubscriptExpr *>(&target)) {
            const ValueKind targetKind = valueKind(target);
            if (!arrayAddressMayCall(*sub)) {
                if (targetKind == ValueKind::Float) {
                    ensureFloat(sourceKind);
                    out_ << "\tfmv.s ft1, fa0\n";
                    emitArrayAddress(*sub);
                    out_ << "\tfsw ft1, 0(a0)\n";
                } else {
                    ensureInt(sourceKind);
                    out_ << "\tmv t4, a0\n";
                    emitArrayAddress(*sub);
                    out_ << "\tsw t4, 0(a0)\n";
                }
                return;
            }
            if (targetKind == ValueKind::Float) {
                ensureFloat(sourceKind);
                pushFA0();
            } else {
                ensureInt(sourceKind);
                pushA0();
            }
            emitArrayAddress(*sub);
            if (targetKind == ValueKind::Float) {
                popFTo("ft0");
                out_ << "\tfsw ft0, 0(a0)\n";
            } else {
                popTo("t0");
                out_ << "\tsw t0, 0(a0)\n";
            }
        }
    }

    void emitArrayAddress(const ArraySubscriptExpr &sub) {
        std::vector<const Expr *> indices;
        const DeclRefExpr *base = collectArrayBase(sub, indices);
        if (!base) {
            out_ << "\tli a0, 0\n";
            return;
        }

        const auto object = lookupLocal(base->name);
        if (indices.size() == 1) {
            emitExpr(*indices[0]);
            int stride = 1;
            for (std::size_t j = 1; j < object.dims.size(); ++j) {
                stride *= object.dims[j];
            }
            if (stride != 1) {
                out_ << "\tli t0, " << stride << "\n";
                out_ << "\tmulw a0, a0, t0\n";
            }
            out_ << "\tslli a0, a0, 2\n";
            out_ << "\tmv t3, a0\n";
            emitArrayBaseAddress("t2", base->name, object);
            out_ << "\tadd a0, t2, t3\n";
            return;
        }

        emitArrayBaseAddress("a0", base->name, object);
        pushA0();
        for (std::size_t i = indices.size(); i > 0; --i) {
            emitExpr(*indices[i - 1]);
            pushA0();
        }

        out_ << "\tli t1, 0\n";
        for (std::size_t i = 0; i < indices.size(); ++i) {
            int stride = 1;
            for (std::size_t j = i + 1; j < object.dims.size(); ++j) {
                stride *= object.dims[j];
            }
            popTo("a0");
            if (stride != 1) {
                out_ << "\tli t0, " << stride << "\n";
                out_ << "\tmul a0, a0, t0\n";
            }
            out_ << "\tadd t1, t1, a0\n";
        }
        out_ << "\tslli t1, t1, 2\n";
        popTo("t0");
        out_ << "\tadd a0, t0, t1\n";
    }

    void emitArrayBaseAddress(const std::string &dst, const std::string &name, const ObjectInfo &object) {
        if (object.found && !object.global && isArrayParameter(name)) {
            emitLoad(dst, "s0", object.offset, "ld");
        } else if (object.found && !object.global) {
            emitAddress(dst, "s0", object.offset);
        } else {
            out_ << "\tla " << dst << ", " << name << "\n";
        }
    }

    const DeclRefExpr *collectArrayBase(const Expr &expr, std::vector<const Expr *> &indices) const {
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

    bool subscriptYieldsArray(const ArraySubscriptExpr &sub) const {
        std::vector<const Expr *> indices;
        const DeclRefExpr *base = collectArrayBase(sub, indices);
        if (!base) {
            return false;
        }
        const auto object = lookupLocal(base->name);
        return object.found && indices.size() < object.dims.size();
    }

    bool arrayAddressMayCall(const ArraySubscriptExpr &sub) const {
        std::vector<const Expr *> indices;
        collectArrayBase(sub, indices);
        for (const Expr *index : indices) {
            if (exprMayCall(*index)) {
                return true;
            }
        }
        return false;
    }

    bool exprMayCall(const Expr &expr) const {
        if (dynamic_cast<const CallExpr *>(&expr)) {
            return true;
        }
        if (const auto *unary = dynamic_cast<const UnaryExpr *>(&expr)) {
            return exprMayCall(*unary->operand);
        }
        if (const auto *binary = dynamic_cast<const BinaryExpr *>(&expr)) {
            return exprMayCall(*binary->lhs) || exprMayCall(*binary->rhs);
        }
        if (const auto *sub = dynamic_cast<const ArraySubscriptExpr *>(&expr)) {
            return exprMayCall(*sub->base) || exprMayCall(*sub->index);
        }
        return false;
    }

    void pushA0() {
        out_ << "\taddi sp, sp, -16\n";
        out_ << "\tsd a0, 8(sp)\n";
    }

    void pushFA0() {
        out_ << "\taddi sp, sp, -16\n";
        out_ << "\tfsw fa0, 12(sp)\n";
    }

    void popTo(const std::string &reg) {
        out_ << "\tld " << reg << ", 8(sp)\n";
        out_ << "\taddi sp, sp, 16\n";
    }

    void popFTo(const std::string &reg) {
        out_ << "\tflw " << reg << ", 12(sp)\n";
        out_ << "\taddi sp, sp, 16\n";
    }

    void emitFloatImmediate(float value) {
        out_ << "\tli t0, " << floatBits(value) << "\n";
        out_ << "\tfmv.w.x fa0, t0\n";
    }

    void ensureFloat(ValueKind from) {
        if (from == ValueKind::Int) {
            out_ << "\tfcvt.s.w fa0, a0\n";
        }
    }

    void ensureInt(ValueKind from) {
        if (from == ValueKind::Float) {
            out_ << "\tfcvt.w.s a0, fa0, rtz\n";
        }
    }

    void emitBoolFromValue(ValueKind from) {
        if (from == ValueKind::Float) {
            out_ << "\tli t0, 0\n";
            out_ << "\tfmv.w.x ft0, t0\n";
            out_ << "\tfeq.s a0, ft0, fa0\n";
            out_ << "\txori a0, a0, 1\n";
            return;
        }
        out_ << "\tsnez a0, a0\n";
    }

    void emitLoad(const std::string &dst, const std::string &base, int offset, const std::string &op) {
        if (fitsI12(offset)) {
            out_ << "\t" << op << ' ' << dst << ", " << offset << '(' << base << ")\n";
            return;
        }
        out_ << "\tli t6, " << offset << "\n";
        out_ << "\tadd t6, " << base << ", t6\n";
        out_ << "\t" << op << ' ' << dst << ", 0(t6)\n";
    }

    void emitFLoad(const std::string &dst, const std::string &base, int offset) {
        emitLoad(dst, base, offset, "flw");
    }

    void emitStore(const std::string &src, const std::string &base, int offset, const std::string &op) {
        if (fitsI12(offset)) {
            out_ << "\t" << op << ' ' << src << ", " << offset << '(' << base << ")\n";
            return;
        }
        out_ << "\tli t6, " << offset << "\n";
        out_ << "\tadd t6, " << base << ", t6\n";
        out_ << "\t" << op << ' ' << src << ", 0(t6)\n";
    }

    void emitFStore(const std::string &src, const std::string &base, int offset) {
        emitStore(src, base, offset, "fsw");
    }

    void emitAddress(const std::string &dst, const std::string &base, int offset) {
        if (fitsI12(offset)) {
            out_ << "\taddi " << dst << ", " << base << ", " << offset << "\n";
            return;
        }
        out_ << "\tli " << dst << ", " << offset << "\n";
        out_ << "\tadd " << dst << ", " << base << ", " << dst << "\n";
    }

    void emitAdjustSp(int delta) {
        if (fitsI12(delta)) {
            out_ << "\taddi sp, sp, " << delta << "\n";
            return;
        }
        out_ << "\tli t6, " << (delta < 0 ? -delta : delta) << "\n";
        out_ << '\t' << (delta < 0 ? "sub" : "add") << " sp, sp, t6\n";
    }

    static bool fitsI12(long long value) {
        return value >= -2048 && value <= 2047;
    }

    void pushScope() {
        scopes_.push_back({});
        constScopes_.push_back({});
    }

    void popScope() {
        scopes_.pop_back();
        constScopes_.pop_back();
    }

    int allocateLocal(const std::string &name, TypeSpecifier type = TypeSpecifier::Int) {
        return allocateLocalBytes(name, 8, {}, type);
    }

    int allocateLocalBytes(const std::string &name, int bytes, std::vector<int> dims = {}, TypeSpecifier type = TypeSpecifier::Int) {
        const int aligned = alignTo(bytes, 8);
        nextOffset_ -= aligned - 8;
        const int offset = nextOffset_;
        nextOffset_ -= 8;
        scopes_.back()[name] = ObjectInfo{true, offset, std::move(dims), false, type};
        return offset;
    }

    int arrayStorageBytes(const VarDef &def) const {
        return arrayElementCount(def) * 4;
    }

    int arrayElementCount(const VarDef &def) const {
        int elements = 1;
        for (int dim : dimensionsOf(def)) {
            elements *= dim;
        }
        return elements;
    }

    std::vector<int> dimensionsOf(const VarDef &def) const {
        std::vector<int> dims;
        for (const auto &dim : def.dimensions) {
            dims.push_back(static_cast<int>(evalConstInt(*dim).value_or(1)));
        }
        return dims;
    }

    std::vector<int> paramDimensions(const FuncParam &param) const {
        std::vector<int> dims;
        if (!param.isArray) {
            return dims;
        }
        dims.push_back(-1);
        for (const auto &dim : param.dimensions) {
            dims.push_back(static_cast<int>(evalConstInt(*dim).value_or(1)));
        }
        return dims;
    }

    void emitArrayInitializer(int baseOffset, const std::vector<const Expr *> &values, TypeSpecifier type) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (values[i]) {
                emitExpr(*values[i]);
            } else {
                if (type == TypeSpecifier::Float) {
                    emitFloatImmediate(0.0f);
                } else {
                    out_ << "\tli a0, 0\n";
                }
            }
            if (type == TypeSpecifier::Float) {
                ensureFloat(values[i] ? valueKind(*values[i]) : ValueKind::Float);
                emitFStore("fa0", "s0", baseOffset + static_cast<int>(i) * 4);
            } else {
                ensureInt(values[i] ? valueKind(*values[i]) : ValueKind::Int);
                emitStore("a0", "s0", baseOffset + static_cast<int>(i) * 4, "sw");
            }
        }
    }

    static int alignTo(int value, int align) {
        return ((value + align - 1) / align) * align;
    }

    void collectStaticInitializer(const InitVal &init, std::vector<std::string> &values) const {
        if (init.expr) {
            if (const auto *integer = dynamic_cast<const IntegerLiteral *>(init.expr.get())) {
                values.push_back(integer->value);
            } else {
                values.push_back("0");
            }
            return;
        }
        for (const auto &element : init.elements) {
            collectStaticInitializer(*element, values);
        }
    }

    void collectArrayInitializer(const InitVal &init, const std::vector<int> &dims, int level, int &index, std::vector<const Expr *> &values) const {
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

    static int productFrom(const std::vector<int> &dims, int level) {
        int product = 1;
        for (std::size_t i = static_cast<std::size_t>(level); i < dims.size(); ++i) {
            product *= dims[i];
        }
        return product;
    }

    std::string staticInitializerValue(const Expr *expr, TypeSpecifier type) const {
        if (!expr) {
            return "0";
        }
        if (type == TypeSpecifier::Float) {
            return std::to_string(floatBits(static_cast<float>(evalConstFloat(*expr).value_or(0.0))));
        }
        if (const auto *integer = dynamic_cast<const IntegerLiteral *>(expr)) {
            return integer->value;
        }
        return std::to_string(evalConstInt(*expr).value_or(0));
    }

    ObjectInfo lookupLocal(const std::string &name) const {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            const auto found = it->find(name);
            if (found != it->end()) {
                return found->second;
            }
        }
        const auto global = globals_.find(name);
        return global == globals_.end() ? ObjectInfo{} : global->second;
    }

    bool isArrayParameter(const std::string &name) const {
        if (!currentFunction_) {
            return false;
        }
        for (const auto &param : currentFunction_->params) {
            if (param->name == name && param->isArray) {
                return true;
            }
        }
        return false;
    }

    void collectFunctionInfo() {
        registerFunction("getint", TypeSpecifier::Int, {});
        registerFunction("getch", TypeSpecifier::Int, {});
        registerFunction("getfloat", TypeSpecifier::Float, {});
        registerFunction("getarray", TypeSpecifier::Int, {TypeSpecifier::Int});
        registerFunction("getfarray", TypeSpecifier::Int, {TypeSpecifier::Int});
        registerFunction("putint", TypeSpecifier::Void, {TypeSpecifier::Int});
        registerFunction("putch", TypeSpecifier::Void, {TypeSpecifier::Int});
        registerFunction("putfloat", TypeSpecifier::Void, {TypeSpecifier::Float});
        registerFunction("putarray", TypeSpecifier::Void, {TypeSpecifier::Int, TypeSpecifier::Int});
        registerFunction("putfarray", TypeSpecifier::Void, {TypeSpecifier::Int, TypeSpecifier::Int});
        registerFunction("starttime", TypeSpecifier::Void, {});
        registerFunction("stoptime", TypeSpecifier::Void, {});
        for (const auto &decl : unit_.declarations) {
            if (const auto *func = dynamic_cast<const FuncDef *>(decl.get())) {
                FunctionInfo info;
                info.returnType = func->returnType;
                for (const auto &param : func->params) {
                    info.params.push_back(param->isArray ? TypeSpecifier::Int : param->type);
                }
                functions_[func->name] = std::move(info);
            }
        }
    }

    void registerFunction(const std::string &name, TypeSpecifier returnType, std::vector<TypeSpecifier> params) {
        FunctionInfo info;
        info.returnType = returnType;
        info.params = std::move(params);
        functions_[name] = std::move(info);
    }

    ValueKind callArgKind(const CallExpr &call, std::size_t index) const {
        const auto found = functions_.find(call.callee);
        if (found != functions_.end() && index < found->second.params.size()) {
            return found->second.params[index] == TypeSpecifier::Float ? ValueKind::Float : ValueKind::Int;
        }
        return valueKind(*call.args[index]);
    }

    ValueKind valueKind(const Expr &expr) const {
        if (dynamic_cast<const FloatLiteral *>(&expr)) {
            return ValueKind::Float;
        }
        if (dynamic_cast<const IntegerLiteral *>(&expr) || dynamic_cast<const StringLiteral *>(&expr)) {
            return ValueKind::Int;
        }
        if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&expr)) {
            const auto object = lookupLocal(ref->name);
            if (!object.dims.empty()) {
                return ValueKind::Int;
            }
            return object.type == TypeSpecifier::Float ? ValueKind::Float : ValueKind::Int;
        }
        if (const auto *sub = dynamic_cast<const ArraySubscriptExpr *>(&expr)) {
            std::vector<const Expr *> indices;
            const DeclRefExpr *base = collectArrayBase(*sub, indices);
            if (!base) {
                return ValueKind::Int;
            }
            const auto object = lookupLocal(base->name);
            if (indices.size() < object.dims.size()) {
                return ValueKind::Int;
            }
            return object.type == TypeSpecifier::Float ? ValueKind::Float : ValueKind::Int;
        }
        if (const auto *call = dynamic_cast<const CallExpr *>(&expr)) {
            const auto found = functions_.find(call->callee);
            if (found != functions_.end()) {
                if (found->second.returnType == TypeSpecifier::Float) {
                    return ValueKind::Float;
                }
                if (found->second.returnType == TypeSpecifier::Void) {
                    return ValueKind::Void;
                }
            }
            if (call->callee == "getfloat") {
                return ValueKind::Float;
            }
            return ValueKind::Int;
        }
        if (const auto *unary = dynamic_cast<const UnaryExpr *>(&expr)) {
            return unary->op == UnaryOp::LogicalNot ? ValueKind::Int : valueKind(*unary->operand);
        }
        if (const auto *binary = dynamic_cast<const BinaryExpr *>(&expr)) {
            switch (binary->op) {
            case BinaryOp::Less:
            case BinaryOp::Greater:
            case BinaryOp::LessEqual:
            case BinaryOp::GreaterEqual:
            case BinaryOp::Equal:
            case BinaryOp::NotEqual:
            case BinaryOp::LogicalAnd:
            case BinaryOp::LogicalOr:
                return ValueKind::Int;
            default:
                return valueKind(*binary->lhs) == ValueKind::Float || valueKind(*binary->rhs) == ValueKind::Float ? ValueKind::Float : ValueKind::Int;
            }
        }
        return ValueKind::Int;
    }

    std::optional<long long> evalConstInt(const Expr &expr) const {
        if (const auto *integer = dynamic_cast<const IntegerLiteral *>(&expr)) {
            return std::strtoll(integer->value.c_str(), nullptr, 0);
        }
        if (const auto *floating = dynamic_cast<const FloatLiteral *>(&expr)) {
            return static_cast<long long>(std::strtod(floating->value.c_str(), nullptr));
        }
        if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&expr)) {
            for (auto it = constScopes_.rbegin(); it != constScopes_.rend(); ++it) {
                const auto found = it->find(ref->name);
                if (found != it->end()) {
                    return found->second;
                }
            }
            const auto global = constValues_.find(ref->name);
            return global == constValues_.end() ? std::optional<long long>{} : std::optional<long long>{global->second};
        }
        if (const auto *unary = dynamic_cast<const UnaryExpr *>(&expr)) {
            auto value = evalConstInt(*unary->operand);
            if (!value) {
                return std::nullopt;
            }
            if (unary->op == UnaryOp::Minus) {
                return -*value;
            }
            if (unary->op == UnaryOp::LogicalNot) {
                return *value == 0;
            }
            return value;
        }
        if (const auto *binary = dynamic_cast<const BinaryExpr *>(&expr)) {
            auto lhs = evalConstInt(*binary->lhs);
            auto rhs = evalConstInt(*binary->rhs);
            if (!lhs || !rhs) {
                return std::nullopt;
            }
            switch (binary->op) {
            case BinaryOp::Add:
                return *lhs + *rhs;
            case BinaryOp::Sub:
                return *lhs - *rhs;
            case BinaryOp::Mul:
                return *lhs * *rhs;
            case BinaryOp::Div:
                return *rhs == 0 ? std::optional<long long>{} : std::optional<long long>{*lhs / *rhs};
            case BinaryOp::Mod:
                return *rhs == 0 ? std::optional<long long>{} : std::optional<long long>{*lhs % *rhs};
            case BinaryOp::Less:
                return *lhs < *rhs;
            case BinaryOp::Greater:
                return *lhs > *rhs;
            case BinaryOp::LessEqual:
                return *lhs <= *rhs;
            case BinaryOp::GreaterEqual:
                return *lhs >= *rhs;
            case BinaryOp::Equal:
                return *lhs == *rhs;
            case BinaryOp::NotEqual:
                return *lhs != *rhs;
            case BinaryOp::LogicalAnd:
                return (*lhs != 0) && (*rhs != 0);
            case BinaryOp::LogicalOr:
                return (*lhs != 0) || (*rhs != 0);
            }
        }
        return std::nullopt;
    }

    std::optional<double> evalConstFloat(const Expr &expr) const {
        if (const auto *floating = dynamic_cast<const FloatLiteral *>(&expr)) {
            return std::strtod(floating->value.c_str(), nullptr);
        }
        if (const auto *integer = dynamic_cast<const IntegerLiteral *>(&expr)) {
            return static_cast<double>(std::strtoll(integer->value.c_str(), nullptr, 0));
        }
        if (const auto *ref = dynamic_cast<const DeclRefExpr *>(&expr)) {
            const auto f = constFloatValues_.find(ref->name);
            if (f != constFloatValues_.end()) {
                return f->second;
            }
            const auto i = constValues_.find(ref->name);
            if (i != constValues_.end()) {
                return static_cast<double>(i->second);
            }
            return std::nullopt;
        }
        if (const auto *unary = dynamic_cast<const UnaryExpr *>(&expr)) {
            auto value = evalConstFloat(*unary->operand);
            if (!value) {
                return std::nullopt;
            }
            if (unary->op == UnaryOp::Minus) {
                return -*value;
            }
            if (unary->op == UnaryOp::LogicalNot) {
                return *value == 0.0 ? 1.0 : 0.0;
            }
            return value;
        }
        if (const auto *binary = dynamic_cast<const BinaryExpr *>(&expr)) {
            auto lhs = evalConstFloat(*binary->lhs);
            auto rhs = evalConstFloat(*binary->rhs);
            if (!lhs || !rhs) {
                return std::nullopt;
            }
            switch (binary->op) {
            case BinaryOp::Add:
                return *lhs + *rhs;
            case BinaryOp::Sub:
                return *lhs - *rhs;
            case BinaryOp::Mul:
                return *lhs * *rhs;
            case BinaryOp::Div:
                return *lhs / *rhs;
            case BinaryOp::Less:
                return *lhs < *rhs;
            case BinaryOp::Greater:
                return *lhs > *rhs;
            case BinaryOp::LessEqual:
                return *lhs <= *rhs;
            case BinaryOp::GreaterEqual:
                return *lhs >= *rhs;
            case BinaryOp::Equal:
                return *lhs == *rhs;
            case BinaryOp::NotEqual:
                return *lhs != *rhs;
            case BinaryOp::LogicalAnd:
                return (*lhs != 0.0) && (*rhs != 0.0);
            case BinaryOp::LogicalOr:
                return (*lhs != 0.0) || (*rhs != 0.0);
            case BinaryOp::Mod:
                return std::nullopt;
            }
        }
        return std::nullopt;
    }

    static std::uint32_t floatBits(float value) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    std::string newLabel(const std::string &prefix) {
        return prefix + "." + std::to_string(nextLabel_++);
    }

    const TranslationUnit &unit_;
    std::ostream &out_;
    const FuncDef *currentFunction_ = nullptr;
    std::vector<std::unordered_map<std::string, ObjectInfo>> scopes_;
    std::vector<std::unordered_map<std::string, long long>> constScopes_;
    std::unordered_map<std::string, ObjectInfo> globals_;
    std::unordered_map<std::string, long long> constValues_;
    std::unordered_map<std::string, double> constFloatValues_;
    std::unordered_map<std::string, FunctionInfo> functions_;
    std::vector<LoopLabels> loops_;
    std::string epilogueLabel_;
    int nextOffset_ = -8;
    int nextLabel_ = 0;
    int estimateFrameSize(const FuncDef &func) const {
        int bytes = 16;
        bytes += static_cast<int>(func.params.size()) * 8;
        if (func.body) {
            bytes += estimateBlockBytes(*func.body);
        }
        bytes += 512;
        if (bytes < 4096) {
            bytes = 4096;
        }
        return alignTo(bytes, 16);
    }

    int estimateBlockBytes(const BlockStmt &block) const {
        int bytes = 0;
        for (const auto &item : block.items) {
            if (const auto *decl = item.decl ? dynamic_cast<const VarDecl *>(item.decl.get()) : nullptr) {
                for (const auto &def : decl->defs) {
                    bytes += def->dimensions.empty() ? 8 : alignTo(arrayStorageBytes(*def), 8);
                }
            } else if (item.stmt) {
                bytes += estimateStmtBytes(*item.stmt);
            }
        }
        return bytes;
    }

    int estimateStmtBytes(const Stmt &stmt) const {
        if (const auto *block = dynamic_cast<const BlockStmt *>(&stmt)) {
            return estimateBlockBytes(*block);
        }
        if (const auto *ifStmt = dynamic_cast<const IfStmt *>(&stmt)) {
            int bytes = ifStmt->thenBranch ? estimateStmtBytes(*ifStmt->thenBranch) : 0;
            bytes += ifStmt->elseBranch ? estimateStmtBytes(*ifStmt->elseBranch) : 0;
            return bytes;
        }
        if (const auto *whileStmt = dynamic_cast<const WhileStmt *>(&stmt)) {
            return whileStmt->body ? estimateStmtBytes(*whileStmt->body) : 0;
        }
        return 0;
    }

    int currentFrameSize_ = 4096;
};

} // namespace

void emitAssembly(const TranslationUnit &unit, std::ostream &out) {
    CodeGen(unit, out).run();
}

} // namespace sysyc::riscv
