#include "emit.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <ostream>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

class IRCodeGen {
public:
    IRCodeGen(const ir::Module &module, std::ostream &out) : module_(module), out_(out) {}

    void run() {
        emitGlobals();
        out_ << "\t.text\n";
        for (const auto &function : module_.functions) {
            emitFunction(function);
        }
    }

private:
    struct PhiCopy {
        int target = -1;
        ir::Type targetType;
        ir::Value source;
    };

    void emitGlobals() {
        if (module_.globals.empty()) {
            return;
        }
        out_ << "\t.data\n";
        for (const auto &global : module_.globals) {
            out_ << "\t.globl " << global.name << "\n";
            out_ << "\t.align 2\n";
            out_ << global.name << ":\n";
            int elements = 1;
            for (int dim : global.dimensions) {
                elements *= dim;
            }
            if (global.dimensions.empty()) {
                out_ << "\t.word " << (global.initValues.empty() ? "0" : global.initValues.front()) << "\n";
                continue;
            }
            int emitted = 0;
            int zeroRun = 0;
            auto flushZero = [&]() {
                if (zeroRun != 0) {
                    out_ << "\t.zero " << zeroRun * 4 << "\n";
                    zeroRun = 0;
                }
            };
            for (const auto &value : global.initValues) {
                if (value == "0") {
                    ++zeroRun;
                } else {
                    flushZero();
                    out_ << "\t.word " << value << "\n";
                }
                ++emitted;
            }
            zeroRun += elements - emitted;
            flushZero();
        }
    }

    void emitFunction(const ir::Function &function) {
        currentFunction_ = &function;
        currentFunctionName_ = function.name;
        valueOffsets_.clear();
        allocaOffsets_.clear();
        valueRegs_.clear();
        valueFRegs_.clear();
        scalarIntAllocas_.clear();
        scalarFloatAllocas_.clear();
        globalValueRegs_.clear();
        globalValueFRegs_.clear();
        activeSavedIntRegs_.clear();
        activeSavedFloatRegs_.clear();
        blockLocalValues_.clear();
        phiCopies_.clear();
        currentBlockLocalValues_ = nullptr;
        currentBlockName_.clear();
        nextIntReg_ = 0;
        nextFloatReg_ = 0;
        nextPhiEdgeLabel_ = 0;
        epilogueLabel_ = ".L" + function.name + ".ret";
        analyzeRegisterAllocas(function);
        buildPhiCopies(function);
        analyzeGlobalValueRegisters(function);
        prepareSavedRegisters(function);
        nextOffset_ = firstLocalOffset();
        collectValueSlots(function);
        analyzeBlockLocalValues(function);
        frameSize_ = alignTo(-nextOffset_ + 512, 16);
        if (frameSize_ < 4096) {
            frameSize_ = 4096;
        }

        out_ << "\t.align 1\n";
        out_ << "\t.globl " << function.name << "\n";
        out_ << "\t.type " << function.name << ", @function\n";
        out_ << function.name << ":\n";
        out_ << "\tli t0, " << frameSize_ << "\n";
        out_ << "\tsub sp, sp, t0\n";
        out_ << "\tadd t1, sp, t0\n";
        out_ << "\tsd ra, -8(t1)\n";
        out_ << "\tsd s0, -16(t1)\n";
        for (std::size_t i = 0; i < activeSavedIntRegs_.size(); ++i) {
            out_ << "\tsd " << activeSavedIntRegs_[i] << ", " << savedIntOffset(i) << "(t1)\n";
        }
        for (std::size_t i = 0; i < activeSavedFloatRegs_.size(); ++i) {
            out_ << "\tfsd " << activeSavedFloatRegs_[i] << ", " << savedFloatOffset(i) << "(t1)\n";
        }
        out_ << "\tmv s0, t1\n";

        int intParam = 0;
        int floatParam = 0;
        int stackParam = 0;
        for (std::size_t i = 0; i < function.params.size(); ++i) {
            if (function.params[i].type.kind == ir::TypeKind::F32) {
                if (floatParam < 8) {
                    storeParamFloat(function.params[i].id, "fa" + std::to_string(floatParam));
                } else {
                    loadFReg("fa0", stackParam * 8);
                    storeParamFloat(function.params[i].id, "fa0");
                    ++stackParam;
                }
                ++floatParam;
            } else {
                if (intParam < 8) {
                    storeParamInt(function.params[i].id, "a" + std::to_string(intParam));
                } else {
                    loadReg("a0", stackParam * 8);
                    storeParamInt(function.params[i].id, "a0");
                    ++stackParam;
                }
                ++intParam;
            }
        }

        for (std::size_t blockIndex = 0; blockIndex < function.blocks.size(); ++blockIndex) {
            const auto &block = function.blocks[blockIndex];
            valueRegs_.clear();
            valueFRegs_.clear();
            nextIntReg_ = 0;
            nextFloatReg_ = 0;
            currentBlockLocalValues_ = &blockLocalValues_[block.name];
            currentBlockName_ = block.name;
            nextBlockName_ = blockIndex + 1 < function.blocks.size() ? function.blocks[blockIndex + 1].name : std::string{};
            out_ << blockLabel(block.name) << ":\n";
            for (const auto &inst : block.instructions) {
                emitInst(inst);
            }
        }

        out_ << epilogueLabel_ << ":\n";
        out_ << "\tmv t1, s0\n";
        for (std::size_t i = 0; i < activeSavedFloatRegs_.size(); ++i) {
            out_ << "\tfld " << activeSavedFloatRegs_[i] << ", " << savedFloatOffset(i) << "(t1)\n";
        }
        for (std::size_t i = 0; i < activeSavedIntRegs_.size(); ++i) {
            out_ << "\tld " << activeSavedIntRegs_[i] << ", " << savedIntOffset(i) << "(t1)\n";
        }
        out_ << "\tld ra, -8(t1)\n";
        out_ << "\tld s0, -16(t1)\n";
        out_ << "\tmv sp, t1\n";
        out_ << "\tret\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
        currentFunction_ = nullptr;
        currentFunctionName_.clear();
        currentBlockLocalValues_ = nullptr;
        currentBlockName_.clear();
        nextBlockName_.clear();
    }

    static std::string edgeKey(const std::string &pred, const std::string &succ) {
        return pred + '\n' + succ;
    }

    static std::string trimLabel(std::string label) {
        const std::size_t first = label.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return {};
        }
        const std::size_t last = label.find_last_not_of(" \t\r\n");
        return label.substr(first, last - first + 1);
    }

    static std::vector<std::string> splitLabels(const std::string &text) {
        std::vector<std::string> labels;
        std::size_t start = 0;
        while (start <= text.size()) {
            const std::size_t comma = text.find(',', start);
            const std::size_t end = comma == std::string::npos ? text.size() : comma;
            labels.push_back(trimLabel(text.substr(start, end - start)));
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
        return labels;
    }

    void buildPhiCopies(const ir::Function &function) {
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Phi) {
                    break;
                }
                const std::vector<std::string> preds = splitLabels(inst.text);
                const std::size_t count = std::min(preds.size(), inst.operands.size());
                for (std::size_t i = 0; i < count; ++i) {
                    if (preds[i].empty()) {
                        continue;
                    }
                    phiCopies_[edgeKey(preds[i], block.name)].push_back(PhiCopy{inst.result, inst.resultType, inst.operands[i]});
                }
            }
        }
    }

    void collectValueSlots(const ir::Function &function) {
        for (const auto &param : function.params) {
            if (!globalValueRegs_.count(param.id) && !globalValueFRegs_.count(param.id)) {
                valueOffsets_[param.id] = allocateSlot();
            }
        }
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.result >= 0 && inst.opcode != ir::Opcode::Alloca &&
                    !globalValueRegs_.count(inst.result) && !globalValueFRegs_.count(inst.result)) {
                    valueOffsets_[inst.result] = allocateSlot();
                }
                if (inst.opcode == ir::Opcode::Alloca && inst.result >= 0) {
                    if (scalarIntAllocas_.count(inst.result) || scalarFloatAllocas_.count(inst.result)) {
                        continue;
                    }
                    allocaOffsets_[inst.result] = allocateBytes(allocaBytes(inst.text));
                }
            }
        }
    }

    void analyzeGlobalValueRegisters(const ir::Function &function) {
        struct Candidate {
            ir::TypeKind type = ir::TypeKind::I32;
            int uses = 0;
            bool crossBlock = false;
            bool phi = false;
        };

        std::unordered_map<int, Candidate> candidates;
        std::unordered_map<int, std::string> defBlock;
        auto addCandidate = [&](int id, ir::Type type, const std::string &blockName, bool isPhi) {
            if (id < 0 || type.kind == ir::TypeKind::Void) {
                return;
            }
            Candidate &candidate = candidates[id];
            candidate.type = type.kind;
            candidate.phi = candidate.phi || isPhi;
            defBlock[id] = blockName;
        };

        for (const auto &param : function.params) {
            addCandidate(param.id, param.type, "entry", false);
            candidates[param.id].crossBlock = true;
        }
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Alloca) {
                    addCandidate(inst.result, inst.resultType, block.name, inst.opcode == ir::Opcode::Phi);
                }
            }
        }

        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                for (const auto &operand : inst.operands) {
                    if (operand.constant || operand.id < 0) {
                        continue;
                    }
                    const auto found = candidates.find(operand.id);
                    if (found == candidates.end()) {
                        continue;
                    }
                    found->second.uses += inst.opcode == ir::Opcode::Phi ? 3 : 1;
                    const auto def = defBlock.find(operand.id);
                    if (def != defBlock.end() && def->second != block.name) {
                        found->second.crossBlock = true;
                    }
                }
            }
        }
        for (const auto &[_, copies] : phiCopies_) {
            for (const PhiCopy &copy : copies) {
                auto source = candidates.find(copy.source.id);
                if (!copy.source.constant && source != candidates.end()) {
                    source->second.uses += 2;
                    source->second.crossBlock = true;
                }
                auto target = candidates.find(copy.target);
                if (target != candidates.end()) {
                    target->second.uses += 4;
                    target->second.crossBlock = true;
                    target->second.phi = true;
                }
            }
        }

        std::unordered_set<std::string> usedIntRegs;
        std::unordered_set<std::string> usedFloatRegs;
        for (const auto &[_, reg] : scalarIntAllocas_) {
            usedIntRegs.insert(reg);
        }
        for (const auto &[_, reg] : scalarFloatAllocas_) {
            usedFloatRegs.insert(reg);
        }

        std::vector<std::pair<int, Candidate>> ints;
        std::vector<std::pair<int, Candidate>> floats;
        for (const auto &[id, candidate] : candidates) {
            if (!candidate.phi || !candidate.crossBlock || candidate.uses < 2) {
                continue;
            }
            if (candidate.type == ir::TypeKind::F32) {
                floats.push_back({id, candidate});
            } else if (candidate.type == ir::TypeKind::I32 || candidate.type == ir::TypeKind::Ptr) {
                ints.push_back({id, candidate});
            }
        }
        const auto hotter = [](const auto &lhs, const auto &rhs) {
            const int lhsScore = lhs.second.uses + (lhs.second.phi ? 100 : 0);
            const int rhsScore = rhs.second.uses + (rhs.second.phi ? 100 : 0);
            if (lhsScore != rhsScore) {
                return lhsScore > rhsScore;
            }
            return lhs.first < rhs.first;
        };
        std::sort(ints.begin(), ints.end(), hotter);
        std::sort(floats.begin(), floats.end(), hotter);

        std::size_t intIndex = 0;
        for (const auto &[id, _] : ints) {
            while (intIndex < globalIntRegs().size() && usedIntRegs.count(globalIntRegs()[intIndex])) {
                ++intIndex;
            }
            if (intIndex >= globalIntRegs().size()) {
                break;
            }
            globalValueRegs_[id] = globalIntRegs()[intIndex++];
        }

        std::size_t floatIndex = 0;
        for (const auto &[id, _] : floats) {
            while (floatIndex < globalFloatRegs().size() && usedFloatRegs.count(globalFloatRegs()[floatIndex])) {
                ++floatIndex;
            }
            if (floatIndex >= globalFloatRegs().size()) {
                break;
            }
            globalValueFRegs_[id] = globalFloatRegs()[floatIndex++];
        }
    }

    void analyzeRegisterAllocas(const ir::Function &function) {
        struct Candidate {
            ir::TypeKind type = ir::TypeKind::I32;
            bool escaped = false;
            int uses = 0;
        };
        std::unordered_map<int, Candidate> candidates;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Alloca && inst.result >= 0 && allocaBytes(inst.text) == 4) {
                    candidates[inst.result] = Candidate{};
                }
            }
        }
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                for (std::size_t i = 0; i < inst.operands.size(); ++i) {
                    const auto &operand = inst.operands[i];
                    const auto found = candidates.find(operand.id);
                    if (operand.constant || found == candidates.end()) {
                        continue;
                    }
                    const bool loadAddress = inst.opcode == ir::Opcode::Load && i == 0;
                    const bool storeAddress = inst.opcode == ir::Opcode::Store && i == 1;
                    if (loadAddress) {
                        found->second.type = inst.resultType.kind;
                        ++found->second.uses;
                    } else if (storeAddress) {
                        found->second.type = inst.operands[0].type.kind;
                        ++found->second.uses;
                    } else {
                        found->second.escaped = true;
                    }
                }
            }
        }

        std::vector<std::pair<int, Candidate>> ints;
        std::vector<std::pair<int, Candidate>> floats;
        for (const auto &[id, candidate] : candidates) {
            if (candidate.escaped || candidate.uses == 0) {
                continue;
            }
            if (candidate.type == ir::TypeKind::F32) {
                floats.push_back({id, candidate});
            } else {
                ints.push_back({id, candidate});
            }
        }
        const auto hotter = [](const auto &lhs, const auto &rhs) {
            if (lhs.second.uses != rhs.second.uses) {
                return lhs.second.uses > rhs.second.uses;
            }
            return lhs.first < rhs.first;
        };
        std::sort(ints.begin(), ints.end(), hotter);
        std::sort(floats.begin(), floats.end(), hotter);
        for (std::size_t i = 0; i < ints.size() && i < scalarIntRegs().size(); ++i) {
            scalarIntAllocas_[ints[i].first] = scalarIntRegs()[i];
        }
        for (std::size_t i = 0; i < floats.size() && i < scalarFloatRegs().size(); ++i) {
            scalarFloatAllocas_[floats[i].first] = scalarFloatRegs()[i];
        }
    }

    void prepareSavedRegisters(const ir::Function &function) {
        auto addInt = [&](const std::string &reg) {
            if (std::find(activeSavedIntRegs_.begin(), activeSavedIntRegs_.end(), reg) == activeSavedIntRegs_.end()) {
                activeSavedIntRegs_.push_back(reg);
            }
        };
        auto addFloat = [&](const std::string &reg) {
            if (std::find(activeSavedFloatRegs_.begin(), activeSavedFloatRegs_.end(), reg) == activeSavedFloatRegs_.end()) {
                activeSavedFloatRegs_.push_back(reg);
            }
        };

        for (const auto &[_, reg] : scalarIntAllocas_) {
            addInt(reg);
        }
        for (const auto &[_, reg] : scalarFloatAllocas_) {
            addFloat(reg);
        }
        for (const auto &[_, reg] : globalValueRegs_) {
            addInt(reg);
        }
        for (const auto &[_, reg] : globalValueFRegs_) {
            addFloat(reg);
        }
        for (const auto &reg : intCacheRegs()) {
            addInt(reg);
        }

        bool hasFloatValue = false;
        for (const auto &param : function.params) {
            hasFloatValue = hasFloatValue || param.type.kind == ir::TypeKind::F32;
        }
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                hasFloatValue = hasFloatValue || inst.resultType.kind == ir::TypeKind::F32;
                for (const auto &operand : inst.operands) {
                    hasFloatValue = hasFloatValue || operand.type.kind == ir::TypeKind::F32;
                }
            }
        }
        if (hasFloatValue) {
            for (const auto &reg : floatCacheRegs()) {
                addFloat(reg);
            }
        }
    }

    void analyzeBlockLocalValues(const ir::Function &function) {
        std::unordered_map<int, std::string> defBlock;
        std::unordered_map<int, std::unordered_set<std::string>> useBlocks;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.result >= 0) {
                    defBlock[inst.result] = block.name;
                }
                for (const auto &operand : inst.operands) {
                    if (!operand.constant && operand.id >= 0) {
                        useBlocks[operand.id].insert(block.name);
                    }
                }
            }
        }
        for (const auto &[id, blockName] : defBlock) {
            const auto found = useBlocks.find(id);
            if (found == useBlocks.end()) {
                continue;
            }
            bool local = true;
            for (const auto &useBlock : found->second) {
                if (useBlock != blockName) {
                    local = false;
                    break;
                }
            }
            if (local) {
                blockLocalValues_[blockName].insert(id);
            }
        }
    }

    void emitInst(const ir::Instruction &inst) {
        switch (inst.opcode) {
        case ir::Opcode::Alloca:
            break;
        case ir::Opcode::Load:
            if (!inst.operands[0].constant) {
                const auto intAlloca = scalarIntAllocas_.find(inst.operands[0].id);
                if (intAlloca != scalarIntAllocas_.end()) {
                    out_ << "\tmv a0, " << intAlloca->second << "\n";
                    storeResult(inst);
                    break;
                }
                const auto floatAlloca = scalarFloatAllocas_.find(inst.operands[0].id);
                if (floatAlloca != scalarFloatAllocas_.end()) {
                    out_ << "\tfmv.s fa0, " << floatAlloca->second << "\n";
                    storeFResult(inst);
                    break;
                }
            }
            emitAddressOperand(inst.operands[0]);
            if (inst.resultType.kind == ir::TypeKind::F32) {
                out_ << "\tflw fa0, 0(a0)\n";
                storeFResult(inst);
            } else {
                out_ << "\tlw a0, 0(a0)\n";
                storeResult(inst);
            }
            break;
        case ir::Opcode::Store:
            if (!inst.operands[1].constant) {
                const auto intAlloca = scalarIntAllocas_.find(inst.operands[1].id);
                if (intAlloca != scalarIntAllocas_.end()) {
                    emitValueOperand(inst.operands[0]);
                    out_ << "\tmv " << intAlloca->second << ", a0\n";
                    break;
                }
                const auto floatAlloca = scalarFloatAllocas_.find(inst.operands[1].id);
                if (floatAlloca != scalarFloatAllocas_.end()) {
                    emitValueOperand(inst.operands[0]);
                    out_ << "\tfmv.s " << floatAlloca->second << ", fa0\n";
                    break;
                }
            }
            if (inst.operands[0].type.kind == ir::TypeKind::F32) {
                emitValueOperand(inst.operands[0]);
                emitAddressOperand(inst.operands[1]);
                out_ << "\tfsw fa0, 0(a0)\n";
            } else {
                emitValueOperandTo("t2", inst.operands[0]);
                emitAddressOperand(inst.operands[1]);
                out_ << "\tsw t2, 0(a0)\n";
            }
            break;
        case ir::Opcode::Gep:
            emitAddressOperandTo("t2", inst.operands[0]);
            if (inst.operands[1].constant && inst.operands[1].type.kind == ir::TypeKind::I32) {
                const long long index = std::strtoll(inst.operands[1].name.c_str(), nullptr, 0);
                const long long offset = index * 4;
                if (fitsI12IR(static_cast<int>(offset))) {
                    out_ << "\taddi a0, t2, " << offset << "\n";
                } else {
                    out_ << "\tli a0, " << offset << "\n";
                    out_ << "\tadd a0, t2, a0\n";
                }
            } else {
                emitValueOperand(inst.operands[1]);
                out_ << "\tslli a0, a0, 2\n";
                out_ << "\tadd a0, t2, a0\n";
            }
            storeResult(inst);
            break;
        case ir::Opcode::Add:
        case ir::Opcode::Sub:
        case ir::Opcode::Mul:
        case ir::Opcode::Div:
        case ir::Opcode::Mod:
        case ir::Opcode::ICmp:
        case ir::Opcode::FCmp:
            emitBinary(inst);
            break;
        case ir::Opcode::Call:
            emitCall(inst);
            break;
        case ir::Opcode::Neg:
            emitValueOperand(inst.operands[0]);
            if (inst.resultType.kind == ir::TypeKind::F32) {
                out_ << "\tfneg.s fa0, fa0\n";
                storeFResult(inst);
            } else {
                out_ << "\tnegw a0, a0\n";
                storeResult(inst);
            }
            break;
        case ir::Opcode::Not:
            emitValueOperand(inst.operands[0]);
            if (inst.operands[0].type.kind == ir::TypeKind::F32) {
                out_ << "\tfmv.w.x ft0, zero\n";
                out_ << "\tfeq.s a0, fa0, ft0\n";
                storeResult(inst);
                break;
            }
            out_ << "\tseqz a0, a0\n";
            storeResult(inst);
            break;
        case ir::Opcode::Br:
            emitPhiCopies(currentBlockName_, inst.text);
            if (inst.text != nextBlockName_) {
                emitJump(blockLabel(inst.text));
            }
            break;
        case ir::Opcode::CondBr:
            emitValueOperand(inst.operands[0]);
            emitCondBranchWithPhi(inst.text);
            break;
        case ir::Opcode::Ret:
            if (!inst.operands.empty()) {
                emitValueOperand(inst.operands[0]);
            }
            emitJump(epilogueLabel_);
            break;
        case ir::Opcode::Cast:
            emitCast(inst);
            break;
        case ir::Opcode::Phi:
            break;
        }
    }

    void emitBinary(const ir::Instruction &inst) {
        if (inst.opcode == ir::Opcode::FCmp ||
            inst.resultType.kind == ir::TypeKind::F32 ||
            inst.operands[0].type.kind == ir::TypeKind::F32 ||
            inst.operands[1].type.kind == ir::TypeKind::F32) {
            emitFloatBinary(inst);
            return;
        }
        if (emitImmediateBinary(inst)) {
            return;
        }
        emitValueOperandTo("t0", inst.operands[0]);
        emitValueOperand(inst.operands[1]);
        switch (inst.opcode) {
        case ir::Opcode::Add: out_ << "\taddw a0, t0, a0\n"; break;
        case ir::Opcode::Sub: out_ << "\tsubw a0, t0, a0\n"; break;
        case ir::Opcode::Mul: out_ << "\tmulw a0, t0, a0\n"; break;
        case ir::Opcode::Div: out_ << "\tdivw a0, t0, a0\n"; break;
        case ir::Opcode::Mod: out_ << "\tremw a0, t0, a0\n"; break;
        case ir::Opcode::ICmp:
            if (inst.text == "lt") out_ << "\tslt a0, t0, a0\n";
            else if (inst.text == "gt") out_ << "\tslt a0, a0, t0\n";
            else if (inst.text == "eq") { out_ << "\tsubw a0, t0, a0\n"; out_ << "\tseqz a0, a0\n"; }
            else if (inst.text == "ne") { out_ << "\tsubw a0, t0, a0\n"; out_ << "\tsnez a0, a0\n"; }
            else if (inst.text == "le") { out_ << "\tslt a0, a0, t0\n"; out_ << "\txori a0, a0, 1\n"; }
            else if (inst.text == "ge") { out_ << "\tslt a0, t0, a0\n"; out_ << "\txori a0, a0, 1\n"; }
            break;
        default:
            break;
        }
        storeResult(inst);
    }

    void emitCondBranch(const std::string &text) {
        const std::size_t comma = text.find(',');
        const std::string trueLabel = trimLabel(text.substr(0, comma));
        const std::string falseLabel = comma == std::string::npos ? std::string{} : trimLabel(text.substr(comma + 1));
        const std::string trueLocal = newPhiEdgeLabel();
        out_ << "\tbnez a0, " << trueLocal << "\n";
        emitJump(blockLabel(falseLabel));
        out_ << trueLocal << ":\n";
        emitJump(blockLabel(trueLabel));
    }

    void emitCondBranchWithPhi(const std::string &text) {
        const std::size_t comma = text.find(',');
        const std::string trueLabel = trimLabel(text.substr(0, comma));
        const std::string falseLabel = comma == std::string::npos ? std::string{} : trimLabel(text.substr(comma + 1));
        if (!hasPhiCopies(currentBlockName_, trueLabel) && !hasPhiCopies(currentBlockName_, falseLabel)) {
            emitCondBranch(text);
            return;
        }

        const std::string trueEdge = newPhiEdgeLabel();
        const std::string falseEdge = newPhiEdgeLabel();
        const std::string trueLocal = newPhiEdgeLabel();
        out_ << "\tbnez a0, " << trueLocal << "\n";
        emitJump(falseEdge);
        out_ << trueLocal << ":\n";
        emitJump(trueEdge);

        out_ << trueEdge << ":\n";
        emitPhiCopies(currentBlockName_, trueLabel);
        emitJump(blockLabel(trueLabel));

        out_ << falseEdge << ":\n";
        emitPhiCopies(currentBlockName_, falseLabel);
        emitJump(blockLabel(falseLabel));
    }

    bool hasPhiCopies(const std::string &pred, const std::string &succ) const {
        const auto found = phiCopies_.find(edgeKey(pred, succ));
        return found != phiCopies_.end() && !found->second.empty();
    }

    void emitPhiCopies(const std::string &pred, const std::string &succ) {
        const auto found = phiCopies_.find(edgeKey(pred, succ));
        if (found == phiCopies_.end()) {
            return;
        }
        for (const PhiCopy &copy : found->second) {
            if (copy.targetType.kind == ir::TypeKind::F32) {
                const auto targetReg = globalValueFRegs_.find(copy.target);
                if (targetReg != globalValueFRegs_.end()) {
                    emitFloatOperandTo(targetReg->second, copy.source);
                } else {
                    const auto sourceReg = !copy.source.constant ? globalValueFRegs_.find(copy.source.id) : globalValueFRegs_.end();
                    if (sourceReg != globalValueFRegs_.end()) {
                        storeFReg(sourceReg->second, valueOffsets_[copy.target]);
                    } else {
                        emitValueOperand(copy.source);
                        storeFReg("fa0", valueOffsets_[copy.target]);
                    }
                }
            } else {
                const auto targetReg = globalValueRegs_.find(copy.target);
                if (targetReg != globalValueRegs_.end()) {
                    emitValueOperandTo(targetReg->second, copy.source);
                } else {
                    const auto sourceReg = !copy.source.constant ? globalValueRegs_.find(copy.source.id) : globalValueRegs_.end();
                    if (sourceReg != globalValueRegs_.end()) {
                        storeReg(sourceReg->second, valueOffsets_[copy.target]);
                    } else {
                        emitValueOperand(copy.source);
                        storeReg("a0", valueOffsets_[copy.target]);
                    }
                }
            }
        }
    }

    std::string newPhiEdgeLabel() {
        return ".Lir." + currentFunctionName_ + ".phi_edge." + std::to_string(nextPhiEdgeLabel_++);
    }

    void emitJump(const std::string &label) {
        out_ << "\tlla t6, " << label << "\n";
        out_ << "\tjr t6\n";
    }

    bool emitImmediateBinary(const ir::Instruction &inst) {
        if (inst.operands.size() != 2 || !inst.operands[1].constant || inst.operands[1].type.kind != ir::TypeKind::I32) {
            return false;
        }
        const long long rhs = std::strtoll(inst.operands[1].name.c_str(), nullptr, 0);
        switch (inst.opcode) {
        case ir::Opcode::Add:
            if (!fitsI12IR(static_cast<int>(rhs))) return false;
            emitValueOperand(inst.operands[0]);
            out_ << "\taddiw a0, a0, " << rhs << "\n";
            storeResult(inst);
            return true;
        case ir::Opcode::Sub:
            if (!fitsI12IR(static_cast<int>(-rhs))) return false;
            emitValueOperand(inst.operands[0]);
            out_ << "\taddiw a0, a0, " << -rhs << "\n";
            storeResult(inst);
            return true;
        case ir::Opcode::Mul:
            if (rhs > 0 && isPowerOfTwo(rhs)) {
                emitValueOperand(inst.operands[0]);
                out_ << "\tslliw a0, a0, " << log2Int(rhs) << "\n";
                storeResult(inst);
                return true;
            }
            return false;
        case ir::Opcode::ICmp:
            if (inst.text == "lt" && fitsI12IR(static_cast<int>(rhs))) {
                emitValueOperand(inst.operands[0]);
                out_ << "\tslti a0, a0, " << rhs << "\n";
                storeResult(inst);
                return true;
            }
            if (inst.text == "ge" && fitsI12IR(static_cast<int>(rhs))) {
                emitValueOperand(inst.operands[0]);
                out_ << "\tslti a0, a0, " << rhs << "\n";
                out_ << "\txori a0, a0, 1\n";
                storeResult(inst);
                return true;
            }
            return false;
        default:
            return false;
        }
    }

    void emitFloatBinary(const ir::Instruction &inst) {
        emitValueOperand(inst.operands[0]);
        out_ << "\tfmv.s ft0, fa0\n";
        emitValueOperand(inst.operands[1]);
        switch (inst.opcode) {
        case ir::Opcode::Add: out_ << "\tfadd.s fa0, ft0, fa0\n"; storeFResult(inst); break;
        case ir::Opcode::Sub: out_ << "\tfsub.s fa0, ft0, fa0\n"; storeFResult(inst); break;
        case ir::Opcode::Mul: out_ << "\tfmul.s fa0, ft0, fa0\n"; storeFResult(inst); break;
        case ir::Opcode::Div: out_ << "\tfdiv.s fa0, ft0, fa0\n"; storeFResult(inst); break;
        case ir::Opcode::FCmp:
            if (inst.text == "lt") out_ << "\tflt.s a0, ft0, fa0\n";
            else if (inst.text == "gt") out_ << "\tflt.s a0, fa0, ft0\n";
            else if (inst.text == "le") out_ << "\tfle.s a0, ft0, fa0\n";
            else if (inst.text == "ge") out_ << "\tfle.s a0, fa0, ft0\n";
            else if (inst.text == "eq") out_ << "\tfeq.s a0, ft0, fa0\n";
            else if (inst.text == "ne") { out_ << "\tfeq.s a0, ft0, fa0\n"; out_ << "\txori a0, a0, 1\n"; }
            storeResult(inst);
            break;
        default:
            break;
        }
    }

    void emitCast(const ir::Instruction &inst) {
        emitValueOperand(inst.operands[0]);
        if (inst.text == "i2f") {
            out_ << "\tfcvt.s.w fa0, a0\n";
            storeFResult(inst);
        } else if (inst.text == "f2i") {
            out_ << "\tfcvt.w.s a0, fa0, rtz\n";
            storeResult(inst);
        }
    }

    void emitCall(const ir::Instruction &inst) {
        if (inst.text == "starttime" || inst.text == "stoptime") {
            return;
        }
        int intArgs = 0;
        int floatArgs = 0;
        for (const auto &operand : inst.operands) {
            if (operand.type.kind == ir::TypeKind::F32) {
                ++floatArgs;
            } else {
                ++intArgs;
            }
        }
        const std::size_t stackArgs = static_cast<std::size_t>((intArgs > 8 ? intArgs - 8 : 0) + (floatArgs > 8 ? floatArgs - 8 : 0));
        const int stackBytes = alignTo(static_cast<int>(stackArgs) * 8, 16);
        if (stackBytes != 0) {
            out_ << "\tli t6, " << stackBytes << "\n";
            out_ << "\tsub sp, sp, t6\n";
        }
        out_ << "\tmv t5, sp\n";
        int intIndex = 0;
        int floatIndex = 0;
        int stackIndex = 0;
        std::vector<std::pair<ir::TypeKind, int>> saved;
        for (const auto &operand : inst.operands) {
            emitValueOperand(operand);
            if (operand.type.kind == ir::TypeKind::F32) {
                if (floatIndex < 8) {
                    out_ << "\taddi sp, sp, -16\n";
                    out_ << "\tfsw fa0, 8(sp)\n";
                    saved.push_back({ir::TypeKind::F32, floatIndex});
                } else {
                    storeFBase("fa0", "t5", stackIndex * 8);
                    ++stackIndex;
                }
                ++floatIndex;
            } else {
                if (intIndex < 8) {
                    out_ << "\taddi sp, sp, -16\n";
                    out_ << "\tsd a0, 8(sp)\n";
                    saved.push_back({ir::TypeKind::I32, intIndex});
                } else {
                    storeBase("a0", "t5", stackIndex * 8);
                    ++stackIndex;
                }
                ++intIndex;
            }
        }
        for (std::size_t i = saved.size(); i > 0; --i) {
            const auto [kind, index] = saved[i - 1];
            if (kind == ir::TypeKind::F32) {
                out_ << "\tflw fa" << index << ", 8(sp)\n";
            } else {
                out_ << "\tld a" << index << ", 8(sp)\n";
            }
            out_ << "\taddi sp, sp, 16\n";
        }
        out_ << "\tcall " << inst.text << "\n";
        if (stackBytes != 0) {
            out_ << "\tli t6, " << stackBytes << "\n";
            out_ << "\tadd sp, sp, t6\n";
        }
        if (inst.result >= 0) {
            if (inst.resultType.kind == ir::TypeKind::F32) {
                storeFResult(inst);
            } else {
                storeResult(inst);
            }
        }
    }

    void storeParamInt(int id, const std::string &reg) {
        storeValueInt(id, reg);
    }

    void storeParamFloat(int id, const std::string &reg) {
        storeValueFloat(id, reg);
    }

    void storeValueInt(int id, const std::string &reg) {
        const auto global = globalValueRegs_.find(id);
        if (global != globalValueRegs_.end()) {
            if (global->second != reg) {
                out_ << "\tmv " << global->second << ", " << reg << "\n";
            }
            return;
        }
        storeReg(reg, valueOffsets_[id]);
    }

    void storeValueFloat(int id, const std::string &reg) {
        const auto global = globalValueFRegs_.find(id);
        if (global != globalValueFRegs_.end()) {
            if (global->second != reg) {
                out_ << "\tfmv.s " << global->second << ", " << reg << "\n";
            }
            return;
        }
        storeFReg(reg, valueOffsets_[id]);
    }

    void emitValueOperand(const ir::Value &value) {
        if (value.constant) {
            if (!value.name.empty() && value.name[0] == '@') {
                out_ << "\tla a0, " << value.name.substr(1) << "\n";
            } else if (value.type.kind == ir::TypeKind::F32) {
                const float f = static_cast<float>(std::strtof(value.name.c_str(), nullptr));
                out_ << "\tli t0, " << irFloatBits(f) << "\n";
                out_ << "\tfmv.w.x fa0, t0\n";
            } else {
                out_ << "\tli a0, " << value.name << "\n";
            }
            return;
        }
        const auto alloca = allocaOffsets_.find(value.id);
        if (alloca != allocaOffsets_.end()) {
            addressReg("a0", alloca->second);
            return;
        }
        const auto globalInt = globalValueRegs_.find(value.id);
        if (globalInt != globalValueRegs_.end()) {
            out_ << "\tmv a0, " << globalInt->second << "\n";
            return;
        }
        const auto globalFloat = globalValueFRegs_.find(value.id);
        if (globalFloat != globalValueFRegs_.end()) {
            out_ << "\tfmv.s fa0, " << globalFloat->second << "\n";
            return;
        }
        const auto intReg = valueRegs_.find(value.id);
        if (intReg != valueRegs_.end()) {
            out_ << "\tmv a0, " << intReg->second << "\n";
            return;
        }
        const auto floatReg = valueFRegs_.find(value.id);
        if (floatReg != valueFRegs_.end()) {
            out_ << "\tfmv.s fa0, " << floatReg->second << "\n";
            return;
        }
        if (value.type.kind == ir::TypeKind::F32) {
            loadFReg("fa0", valueOffsets_[value.id]);
        } else {
            loadReg("a0", valueOffsets_[value.id]);
        }
    }

    void emitValueOperandTo(const std::string &reg, const ir::Value &value) {
        if (reg == "a0") {
            emitValueOperand(value);
            return;
        }
        if (value.constant) {
            if (!value.name.empty() && value.name[0] == '@') {
                out_ << "\tla " << reg << ", " << value.name.substr(1) << "\n";
            } else {
                out_ << "\tli " << reg << ", " << value.name << "\n";
            }
            return;
        }
        const auto alloca = allocaOffsets_.find(value.id);
        if (alloca != allocaOffsets_.end()) {
            addressReg(reg, alloca->second);
            return;
        }
        const auto globalInt = globalValueRegs_.find(value.id);
        if (globalInt != globalValueRegs_.end()) {
            if (globalInt->second != reg) {
                out_ << "\tmv " << reg << ", " << globalInt->second << "\n";
            }
            return;
        }
        const auto intReg = valueRegs_.find(value.id);
        if (intReg != valueRegs_.end()) {
            if (intReg->second != reg) {
                out_ << "\tmv " << reg << ", " << intReg->second << "\n";
            }
            return;
        }
        loadReg(reg, valueOffsets_[value.id]);
    }

    void emitFloatOperandTo(const std::string &reg, const ir::Value &value) {
        if (reg == "fa0") {
            emitValueOperand(value);
            return;
        }
        if (value.constant) {
            const float f = static_cast<float>(std::strtof(value.name.c_str(), nullptr));
            out_ << "\tli t0, " << irFloatBits(f) << "\n";
            out_ << "\tfmv.w.x " << reg << ", t0\n";
            return;
        }
        const auto globalFloat = globalValueFRegs_.find(value.id);
        if (globalFloat != globalValueFRegs_.end()) {
            if (globalFloat->second != reg) {
                out_ << "\tfmv.s " << reg << ", " << globalFloat->second << "\n";
            }
            return;
        }
        const auto floatReg = valueFRegs_.find(value.id);
        if (floatReg != valueFRegs_.end()) {
            if (floatReg->second != reg) {
                out_ << "\tfmv.s " << reg << ", " << floatReg->second << "\n";
            }
            return;
        }
        loadFReg(reg, valueOffsets_[value.id]);
    }

    void emitAddressOperand(const ir::Value &value) {
        if (value.constant && !value.name.empty() && value.name[0] == '@') {
            out_ << "\tla a0, " << value.name.substr(1) << "\n";
            return;
        }
        const auto alloca = allocaOffsets_.find(value.id);
        if (alloca != allocaOffsets_.end()) {
            addressReg("a0", alloca->second);
            return;
        }
        const auto globalInt = globalValueRegs_.find(value.id);
        if (globalInt != globalValueRegs_.end()) {
            out_ << "\tmv a0, " << globalInt->second << "\n";
            return;
        }
        const auto intReg = valueRegs_.find(value.id);
        if (intReg != valueRegs_.end()) {
            out_ << "\tmv a0, " << intReg->second << "\n";
            return;
        }
        loadReg("a0", valueOffsets_[value.id]);
    }

    void emitAddressOperandTo(const std::string &reg, const ir::Value &value) {
        if (reg == "a0") {
            emitAddressOperand(value);
            return;
        }
        if (value.constant && !value.name.empty() && value.name[0] == '@') {
            out_ << "\tla " << reg << ", " << value.name.substr(1) << "\n";
            return;
        }
        const auto alloca = allocaOffsets_.find(value.id);
        if (alloca != allocaOffsets_.end()) {
            addressReg(reg, alloca->second);
            return;
        }
        const auto globalInt = globalValueRegs_.find(value.id);
        if (globalInt != globalValueRegs_.end()) {
            if (globalInt->second != reg) {
                out_ << "\tmv " << reg << ", " << globalInt->second << "\n";
            }
            return;
        }
        const auto intReg = valueRegs_.find(value.id);
        if (intReg != valueRegs_.end()) {
            if (intReg->second != reg) {
                out_ << "\tmv " << reg << ", " << intReg->second << "\n";
            }
            return;
        }
        loadReg(reg, valueOffsets_[value.id]);
    }

    void storeResult(const ir::Instruction &inst) {
        const auto global = globalValueRegs_.find(inst.result);
        if (global != globalValueRegs_.end()) {
            if (global->second != "a0") {
                out_ << "\tmv " << global->second << ", a0\n";
            }
            return;
        }
        if (cacheResult(inst.result, inst.resultType) && isCurrentBlockLocal(inst.result)) {
            return;
        }
        storeReg("a0", valueOffsets_[inst.result]);
    }

    void storeFResult(const ir::Instruction &inst) {
        const auto global = globalValueFRegs_.find(inst.result);
        if (global != globalValueFRegs_.end()) {
            if (global->second != "fa0") {
                out_ << "\tfmv.s " << global->second << ", fa0\n";
            }
            return;
        }
        if (cacheResult(inst.result, inst.resultType) && isCurrentBlockLocal(inst.result)) {
            return;
        }
        storeFReg("fa0", valueOffsets_[inst.result]);
    }

    bool cacheResult(int id, ir::Type type) {
        if (id < 0) {
            return false;
        }
        if (type.kind == ir::TypeKind::F32) {
            if (nextFloatReg_ >= floatCacheRegs().size()) {
                return false;
            }
            const std::string &reg = floatCacheRegs()[nextFloatReg_++];
            out_ << "\tfmv.s " << reg << ", fa0\n";
            valueFRegs_[id] = reg;
            return true;
        }
        if (type.kind == ir::TypeKind::I32 || type.kind == ir::TypeKind::Ptr) {
            if (nextIntReg_ >= intCacheRegs().size()) {
                return false;
            }
            const std::string &reg = intCacheRegs()[nextIntReg_++];
            out_ << "\tmv " << reg << ", a0\n";
            valueRegs_[id] = reg;
            return true;
        }
        return false;
    }

    bool isCurrentBlockLocal(int id) const {
        return currentBlockLocalValues_ && currentBlockLocalValues_->count(id) != 0;
    }

    int allocateSlot() {
        return allocateBytes(8);
    }

    int allocateBytes(int bytes) {
        const int aligned = ((bytes + 7) / 8) * 8;
        nextOffset_ -= aligned - 8;
        const int offset = nextOffset_;
        nextOffset_ -= 8;
        return offset;
    }

    int allocaBytes(const std::string &text) const {
        const std::size_t colon = text.find(':');
        if (colon == std::string::npos) {
            return 8;
        }
        return std::strtol(text.c_str() + colon + 1, nullptr, 10);
    }

    void loadReg(const std::string &reg, int offset) {
        if (fitsI12IR(offset)) {
            out_ << "\tld " << reg << ", " << offset << "(s0)\n";
            return;
        }
        out_ << "\tli t6, " << offset << "\n";
        out_ << "\tadd t6, s0, t6\n";
        out_ << "\tld " << reg << ", 0(t6)\n";
    }

    void storeReg(const std::string &reg, int offset) {
        if (fitsI12IR(offset)) {
            out_ << "\tsd " << reg << ", " << offset << "(s0)\n";
            return;
        }
        out_ << "\tli t6, " << offset << "\n";
        out_ << "\tadd t6, s0, t6\n";
        out_ << "\tsd " << reg << ", 0(t6)\n";
    }

    void loadFReg(const std::string &reg, int offset) {
        if (fitsI12IR(offset)) {
            out_ << "\tflw " << reg << ", " << offset << "(s0)\n";
            return;
        }
        out_ << "\tli t6, " << offset << "\n";
        out_ << "\tadd t6, s0, t6\n";
        out_ << "\tflw " << reg << ", 0(t6)\n";
    }

    void storeFReg(const std::string &reg, int offset) {
        if (fitsI12IR(offset)) {
            out_ << "\tfsw " << reg << ", " << offset << "(s0)\n";
            return;
        }
        out_ << "\tli t6, " << offset << "\n";
        out_ << "\tadd t6, s0, t6\n";
        out_ << "\tfsw " << reg << ", 0(t6)\n";
    }

    void storeBase(const std::string &reg, const std::string &base, int offset) {
        if (fitsI12IR(offset)) {
            out_ << "\tsd " << reg << ", " << offset << '(' << base << ")\n";
            return;
        }
        out_ << "\tli t6, " << offset << "\n";
        out_ << "\tadd t6, " << base << ", t6\n";
        out_ << "\tsd " << reg << ", 0(t6)\n";
    }

    void storeFBase(const std::string &reg, const std::string &base, int offset) {
        if (fitsI12IR(offset)) {
            out_ << "\tfsw " << reg << ", " << offset << '(' << base << ")\n";
            return;
        }
        out_ << "\tli t6, " << offset << "\n";
        out_ << "\tadd t6, " << base << ", t6\n";
        out_ << "\tfsw " << reg << ", 0(t6)\n";
    }

    void addressReg(const std::string &reg, int offset) {
        if (fitsI12IR(offset)) {
            out_ << "\taddi " << reg << ", s0, " << offset << "\n";
            return;
        }
        out_ << "\tli " << reg << ", " << offset << "\n";
        out_ << "\tadd " << reg << ", s0, " << reg << "\n";
    }

    std::string blockLabel(const std::string &name) const {
        return ".Lir." + currentFunctionName_ + "." + name;
    }

    static bool fitsI12IR(int value) {
        return value >= -2048 && value <= 2047;
    }

    static bool isPowerOfTwo(long long value) {
        return value > 0 && (value & (value - 1)) == 0;
    }

    static int log2Int(long long value) {
        int shift = 0;
        while (value > 1) {
            value >>= 1;
            ++shift;
        }
        return shift;
    }

    static int alignTo(int value, int align) {
        return ((value + align - 1) / align) * align;
    }

    static const std::vector<std::string> &savedIntRegs() {
        static const std::vector<std::string> regs = {"s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"};
        return regs;
    }

    static const std::vector<std::string> &savedFloatRegs() {
        static const std::vector<std::string> regs = {"fs0", "fs1", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11"};
        return regs;
    }

    static const std::vector<std::string> &scalarIntRegs() {
        static const std::vector<std::string> regs = {"s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"};
        return regs;
    }

    static const std::vector<std::string> &scalarFloatRegs() {
        static const std::vector<std::string> regs = {"fs0", "fs1", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7"};
        return regs;
    }

    static const std::vector<std::string> &globalIntRegs() {
        static const std::vector<std::string> regs = {"s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"};
        return regs;
    }

    static const std::vector<std::string> &globalFloatRegs() {
        static const std::vector<std::string> regs = {"fs0", "fs1", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7"};
        return regs;
    }

    static const std::vector<std::string> &intCacheRegs() {
        static const std::vector<std::string> regs = {"s9", "s10", "s11"};
        return regs;
    }

    static const std::vector<std::string> &floatCacheRegs() {
        static const std::vector<std::string> regs = {"fs8", "fs9", "fs10", "fs11"};
        return regs;
    }

    int savedIntOffset(std::size_t index) const {
        return -24 - static_cast<int>(index) * 8;
    }

    int savedFloatOffset(std::size_t index) const {
        return -24 - static_cast<int>(activeSavedIntRegs_.size()) * 8 - static_cast<int>(index) * 8;
    }

    int firstLocalOffset() const {
        return -24 - static_cast<int>(activeSavedIntRegs_.size() + activeSavedFloatRegs_.size()) * 8;
    }

    static std::uint32_t irFloatBits(float value) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    const ir::Module &module_;
    std::ostream &out_;
    const ir::Function *currentFunction_ = nullptr;
    std::string currentFunctionName_;
    std::string currentBlockName_;
    std::string nextBlockName_;
    std::unordered_map<int, int> valueOffsets_;
    std::unordered_map<int, int> allocaOffsets_;
    std::unordered_map<int, std::string> valueRegs_;
    std::unordered_map<int, std::string> valueFRegs_;
    std::unordered_map<int, std::string> scalarIntAllocas_;
    std::unordered_map<int, std::string> scalarFloatAllocas_;
    std::unordered_map<int, std::string> globalValueRegs_;
    std::unordered_map<int, std::string> globalValueFRegs_;
    std::vector<std::string> activeSavedIntRegs_;
    std::vector<std::string> activeSavedFloatRegs_;
    std::unordered_map<std::string, std::unordered_set<int>> blockLocalValues_;
    std::unordered_map<std::string, std::vector<PhiCopy>> phiCopies_;
    const std::unordered_set<int> *currentBlockLocalValues_ = nullptr;
    std::string epilogueLabel_;
    std::size_t nextIntReg_ = 0;
    std::size_t nextFloatReg_ = 0;
    int nextPhiEdgeLabel_ = 0;
    int nextOffset_ = -24;
    int frameSize_ = 4096;
};

} // namespace

void emitAssembly(const TranslationUnit &unit, std::ostream &out) {
    CodeGen(unit, out).run();
}

void emitAssembly(const ir::Module &module, std::ostream &out) {
    IRCodeGen(module, out).run();
}

} // namespace sysyc::riscv
