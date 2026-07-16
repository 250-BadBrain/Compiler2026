#include "emit.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <optional>
#include <ostream>
#include <cstdlib>
#include <sstream>
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

    struct Interval {
        int id = -1;
        ir::TypeKind type = ir::TypeKind::I32;
        int start = 0;
        int end = 0;
        int weight = 0;
        bool crossBlock = false;
        bool phi = false;
        bool param = false;
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
        transientRegs_.clear();
        transientFRegs_.clear();
        valueUseCounts_.clear();
        activeSavedIntRegs_.clear();
        activeSavedFloatRegs_.clear();
        blockLocalValues_.clear();
        phiCopies_.clear();
        currentBlockLocalValues_ = nullptr;
        currentBlockName_.clear();
        nextIntReg_ = 0;
        nextFloatReg_ = 0;
        nextPhiEdgeLabel_ = 0;
        elideResultStoreId_ = -1;
        useLongJumps_ = needsLongJumps(function);
        epilogueLabel_ = ".L" + function.name + ".ret";
        const bool saveReturnAddress = functionHasRealCall(function);
        currentGlobalIntRegs_ = savedIntRegs();
        currentIntCacheRegs_ = saveReturnAddress ? savedIntCacheRegs() : leafIntRegs();
        countValueUses(function);
        analyzeRegisterAllocas(function);
        buildPhiCopies(function);
        analyzeGlobalValueRegisters(function);
        analyzeBlockLocalValues(function);
        prepareSavedRegisters(function);
        nextOffset_ = firstLocalOffset();
        collectValueSlots(function);
        omitFrame_ = !saveReturnAddress && activeSavedIntRegs_.empty() && activeSavedFloatRegs_.empty() &&
                     valueOffsets_.empty() && allocaOffsets_.empty() && maxStackParamSlots(function) == 0;
        frameSize_ = alignTo(-nextOffset_ + 512, 16);
        if (frameSize_ < 4096) {
            frameSize_ = 4096;
        }

        out_ << "\t.align 1\n";
        out_ << "\t.globl " << function.name << "\n";
        out_ << "\t.type " << function.name << ", @function\n";
        out_ << function.name << ":\n";
        if (!omitFrame_) {
            out_ << "\tli t0, " << frameSize_ << "\n";
            out_ << "\tsub sp, sp, t0\n";
            out_ << "\tadd t1, sp, t0\n";
            if (saveReturnAddress) {
                out_ << "\tsd ra, -8(t1)\n";
            }
            out_ << "\tsd s0, -16(t1)\n";
            for (std::size_t i = 0; i < activeSavedIntRegs_.size(); ++i) {
                out_ << "\tsd " << activeSavedIntRegs_[i] << ", " << savedIntOffset(i) << "(t1)\n";
            }
            for (std::size_t i = 0; i < activeSavedFloatRegs_.size(); ++i) {
                out_ << "\tfsd " << activeSavedFloatRegs_[i] << ", " << savedFloatOffset(i) << "(t1)\n";
            }
            out_ << "\tmv s0, t1\n";
        }

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
            transientRegs_.clear();
            transientFRegs_.clear();
            nextIntReg_ = 0;
            nextFloatReg_ = 0;
            currentBlockLocalValues_ = &blockLocalValues_[block.name];
            currentBlockName_ = block.name;
            nextBlockName_ = blockIndex + 1 < function.blocks.size() ? function.blocks[blockIndex + 1].name : std::string{};
            out_ << blockLabel(block.name) << ":\n";
            for (std::size_t instIndex = 0; instIndex < block.instructions.size(); ++instIndex) {
                const auto &inst = block.instructions[instIndex];
                if ((inst.opcode == ir::Opcode::ICmp || inst.opcode == ir::Opcode::FCmp) &&
                    inst.result >= 0 && valueUseCounts_[inst.result] == 1 &&
                    instIndex + 1 < block.instructions.size() &&
                    block.instructions[instIndex + 1].opcode == ir::Opcode::CondBr &&
                    !block.instructions[instIndex + 1].operands.empty() &&
                    !block.instructions[instIndex + 1].operands[0].constant &&
                    block.instructions[instIndex + 1].operands[0].id == inst.result) {
                    emitCompareCondBranchWithPhi(inst, block.instructions[instIndex + 1].text);
                    ++instIndex;
                    continue;
                }
                if (inst.opcode == ir::Opcode::Not && inst.result >= 0 && valueUseCounts_[inst.result] == 1 &&
                    instIndex + 1 < block.instructions.size() &&
                    block.instructions[instIndex + 1].opcode == ir::Opcode::CondBr &&
                    !block.instructions[instIndex + 1].operands.empty() &&
                    !block.instructions[instIndex + 1].operands[0].constant &&
                    block.instructions[instIndex + 1].operands[0].id == inst.result) {
                    emitNotCondBranchWithPhi(inst, block.instructions[instIndex + 1].text);
                    ++instIndex;
                    continue;
                }
                if (canReturnInstructionResult(block.instructions, instIndex)) {
                    elideResultStoreId_ = inst.result;
                    emitInst(inst);
                    elideResultStoreId_ = -1;
                    emitJump(epilogueLabel_);
                    ++instIndex;
                    continue;
                }
                if (canForwardResultToNext(block.instructions, instIndex)) {
                    elideResultStoreId_ = inst.result;
                    emitInst(inst);
                    elideResultStoreId_ = -1;
                    if (inst.resultType.kind == ir::TypeKind::F32) {
                        const auto global = globalValueFRegs_.find(inst.result);
                        transientFRegs_[inst.result] = global == globalValueFRegs_.end() ? "fa0" : global->second;
                    } else {
                        transientRegs_[inst.result] = resultIntRegister(inst);
                    }
                    emitInst(block.instructions[instIndex + 1]);
                    transientRegs_.erase(inst.result);
                    transientFRegs_.erase(inst.result);
                    ++instIndex;
                    continue;
                }
                if (canFoldGepStore(block.instructions, instIndex)) {
                    emitFoldedGepStore(inst, block.instructions[instIndex + 1]);
                    ++instIndex;
                    continue;
                }
                if (canFoldGepLoad(block.instructions, instIndex)) {
                    emitFoldedGepLoad(inst, block.instructions[instIndex + 1]);
                    ++instIndex;
                    continue;
                }
                emitInst(inst);
            }
        }

        out_ << epilogueLabel_ << ":\n";
        if (!omitFrame_) {
            for (std::size_t i = 0; i < activeSavedFloatRegs_.size(); ++i) {
                out_ << "\tfld " << activeSavedFloatRegs_[i] << ", " << savedFloatOffset(i) << "(s0)\n";
            }
            for (std::size_t i = 0; i < activeSavedIntRegs_.size(); ++i) {
                out_ << "\tld " << activeSavedIntRegs_[i] << ", " << savedIntOffset(i) << "(s0)\n";
            }
            if (saveReturnAddress) {
                out_ << "\tld ra, -8(s0)\n";
            }
            out_ << "\tmv sp, s0\n";
            out_ << "\tld s0, -16(sp)\n";
        }
        out_ << "\tret\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
        currentFunction_ = nullptr;
        currentFunctionName_.clear();
        currentBlockLocalValues_ = nullptr;
        currentBlockName_.clear();
        nextBlockName_.clear();
        currentGlobalIntRegs_.clear();
        currentIntCacheRegs_.clear();
        omitFrame_ = false;
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

    static bool functionHasRealCall(const ir::Function &function) {
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Call && inst.text != "starttime" && inst.text != "stoptime") {
                    return true;
                }
            }
        }
        return false;
    }

    bool canReturnInstructionResult(const std::vector<ir::Instruction> &instructions, std::size_t index) const {
        if (index + 1 >= instructions.size()) {
            return false;
        }
        const ir::Instruction &inst = instructions[index];
        const ir::Instruction &ret = instructions[index + 1];
        if (inst.result < 0 || inst.resultType.kind == ir::TypeKind::Void ||
            inst.opcode == ir::Opcode::Phi || inst.opcode == ir::Opcode::Alloca ||
            ret.opcode != ir::Opcode::Ret || ret.operands.size() != 1 ||
            ret.operands[0].constant || ret.operands[0].id != inst.result) {
            return false;
        }
        const auto uses = valueUseCounts_.find(inst.result);
        return uses != valueUseCounts_.end() && uses->second == 1;
    }

    bool canForwardResultToNext(const std::vector<ir::Instruction> &instructions, std::size_t index) const {
        if (index + 1 >= instructions.size()) {
            return false;
        }
        const ir::Instruction &producer = instructions[index];
        const ir::Instruction &consumer = instructions[index + 1];
        if (producer.result < 0 || producer.resultType.kind == ir::TypeKind::Void ||
            producer.opcode == ir::Opcode::Phi || producer.opcode == ir::Opcode::Alloca ||
            consumer.opcode == ir::Opcode::Phi || consumer.opcode == ir::Opcode::Call ||
            consumer.opcode == ir::Opcode::Load) {
            return false;
        }
        const auto uses = valueUseCounts_.find(producer.result);
        if (uses == valueUseCounts_.end() || uses->second != 1) {
            return false;
        }
        if (consumer.opcode == ir::Opcode::Store) {
            return !consumer.operands.empty() && !consumer.operands[0].constant &&
                   consumer.operands[0].id == producer.result;
        }
        if (consumer.opcode == ir::Opcode::CondBr || consumer.opcode == ir::Opcode::Ret ||
            consumer.opcode == ir::Opcode::Cast || consumer.opcode == ir::Opcode::Neg ||
            consumer.opcode == ir::Opcode::Not) {
            return !consumer.operands.empty() && !consumer.operands[0].constant &&
                   consumer.operands[0].id == producer.result;
        }
        if (consumer.opcode == ir::Opcode::Add || consumer.opcode == ir::Opcode::Sub ||
            consumer.opcode == ir::Opcode::Mul || consumer.opcode == ir::Opcode::Div ||
            consumer.opcode == ir::Opcode::Mod || consumer.opcode == ir::Opcode::ICmp ||
            consumer.opcode == ir::Opcode::FCmp) {
            if (!consumer.operands.empty() && !consumer.operands[0].constant &&
                consumer.operands[0].id == producer.result) {
                return true;
            }
            return producer.resultType.kind != ir::TypeKind::F32 && consumer.opcode != ir::Opcode::FCmp &&
                   consumer.operands.size() > 1 && !consumer.operands[1].constant &&
                   consumer.operands[1].id == producer.result;
        }
        if (consumer.opcode == ir::Opcode::Gep) {
            return producer.resultType.kind != ir::TypeKind::F32 &&
                   consumer.operands.size() > 1 && !consumer.operands[1].constant &&
                   consumer.operands[1].id == producer.result;
        }
        return false;
    }

    bool canFoldGepStore(const std::vector<ir::Instruction> &instructions, std::size_t index) const {
        if (index + 1 >= instructions.size()) {
            return false;
        }
        const ir::Instruction &gep = instructions[index];
        const ir::Instruction &store = instructions[index + 1];
        if (gep.opcode != ir::Opcode::Gep || gep.result < 0 ||
            store.opcode != ir::Opcode::Store || store.operands.size() < 2 ||
            store.operands[1].constant || store.operands[1].id != gep.result) {
            return false;
        }
        const auto uses = valueUseCounts_.find(gep.result);
        return uses != valueUseCounts_.end() && uses->second == 1;
    }

    bool canFoldGepLoad(const std::vector<ir::Instruction> &instructions, std::size_t index) const {
        if (index + 1 >= instructions.size()) {
            return false;
        }
        const ir::Instruction &gep = instructions[index];
        const ir::Instruction &load = instructions[index + 1];
        if (gep.opcode != ir::Opcode::Gep || gep.result < 0 ||
            load.opcode != ir::Opcode::Load || load.operands.empty() ||
            load.operands[0].constant || load.operands[0].id != gep.result) {
            return false;
        }
        const auto uses = valueUseCounts_.find(gep.result);
        return uses != valueUseCounts_.end() && uses->second == 1;
    }

    void countValueUses(const ir::Function &function) {
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                for (const auto &operand : inst.operands) {
                    if (!operand.constant && operand.id >= 0) {
                        ++valueUseCounts_[operand.id];
                    }
                }
            }
        }
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

    static int maxStackParamSlots(const ir::Function &function) {
        int intParams = 0;
        int floatParams = 0;
        int stackParams = 0;
        for (const auto &param : function.params) {
            if (param.type.kind == ir::TypeKind::F32) {
                if (floatParams >= 8) {
                    ++stackParams;
                }
                ++floatParams;
            } else {
                if (intParams >= 8) {
                    ++stackParams;
                }
                ++intParams;
            }
        }
        return stackParams;
    }

    static bool hugeControlFlowFunction(const ir::Function &function) {
        std::size_t instructions = 0;
        for (const auto &block : function.blocks) {
            instructions += block.instructions.size();
        }
        return function.blocks.size() > 1000 || instructions > 30000;
    }

    void analyzeGlobalValueRegisters(const ir::Function &function) {
        if (hugeControlFlowFunction(function)) {
            return;
        }
        std::unordered_map<int, Interval> intervals;
        std::unordered_map<int, std::string> defBlock;
        std::unordered_map<int, std::unordered_set<std::string>> useBlocks;
        std::unordered_map<std::string, int> blockStart;
        std::unordered_map<std::string, int> blockEnd;
        int position = 1;
        for (const auto &block : function.blocks) {
            blockStart[block.name] = position;
            position += std::max<std::size_t>(1, block.instructions.size());
            blockEnd[block.name] = position;
            ++position;
        }

        auto addInterval = [&](int id, ir::Type type, const std::string &blockName, int start, bool isPhi) {
            if (id < 0 || type.kind == ir::TypeKind::Void) {
                return;
            }
            Interval &interval = intervals[id];
            interval.id = id;
            interval.type = type.kind;
            interval.start = interval.start == 0 ? start : std::min(interval.start, start);
            interval.end = std::max(interval.end, start);
            interval.phi = interval.phi || isPhi;
            defBlock[id] = blockName;
        };

        for (const auto &param : function.params) {
            addInterval(param.id, param.type, function.blocks.empty() ? "entry" : function.blocks.front().name, 0, false);
            intervals[param.id].crossBlock = true;
            intervals[param.id].weight += 8;
            intervals[param.id].param = true;
        }
        for (const auto &block : function.blocks) {
            int instPos = blockStart[block.name];
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Alloca) {
                    const int defPos = inst.opcode == ir::Opcode::Phi ? blockStart[block.name] : instPos;
                    addInterval(inst.result, inst.resultType, block.name, defPos, inst.opcode == ir::Opcode::Phi);
                }
                ++instPos;
            }
        }

        for (const auto &block : function.blocks) {
            int instPos = blockStart[block.name];
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Phi) {
                    ++instPos;
                    continue;
                }
                for (const auto &operand : inst.operands) {
                    if (operand.constant || operand.id < 0) {
                        continue;
                    }
                    const auto found = intervals.find(operand.id);
                    if (found == intervals.end()) {
                        continue;
                    }
                    found->second.end = std::max(found->second.end, instPos);
                    found->second.weight += 1;
                    useBlocks[operand.id].insert(block.name);
                    const auto def = defBlock.find(operand.id);
                    if (def != defBlock.end() && def->second != block.name) {
                        found->second.crossBlock = true;
                    }
                }
                ++instPos;
            }
        }
        for (const auto &[_, copies] : phiCopies_) {
            const std::size_t separator = _.find('\n');
            const std::string pred = separator == std::string::npos ? std::string{} : _.substr(0, separator);
            const std::string succ = separator == std::string::npos ? std::string{} : _.substr(separator + 1);
            const int sourceUsePos = blockEnd.count(pred) ? blockEnd[pred] : position;
            const int targetDefPos = blockStart.count(succ) ? blockStart[succ] : 0;
            for (const PhiCopy &copy : copies) {
                auto source = intervals.find(copy.source.id);
                if (!copy.source.constant && source != intervals.end()) {
                    source->second.end = std::max(source->second.end, sourceUsePos);
                    source->second.weight += 3;
                    source->second.crossBlock = true;
                    useBlocks[copy.source.id].insert(pred);
                }
                auto target = intervals.find(copy.target);
                if (target != intervals.end()) {
                    target->second.start = std::min(target->second.start, targetDefPos);
                    target->second.end = std::max(target->second.end, targetDefPos);
                    target->second.weight += 6;
                    target->second.crossBlock = true;
                    target->second.phi = true;
                }
            }
        }

        std::unordered_map<std::string, std::vector<std::string>> successors;
        std::unordered_map<std::string, std::vector<std::string>> predecessors;
        for (const auto &block : function.blocks) {
            if (block.instructions.empty()) {
                continue;
            }
            const ir::Instruction &term = block.instructions.back();
            if (term.opcode == ir::Opcode::Br) {
                successors[block.name].push_back(term.text);
                predecessors[term.text].push_back(block.name);
            } else if (term.opcode == ir::Opcode::CondBr) {
                const std::size_t comma = term.text.find(',');
                if (comma != std::string::npos) {
                    const std::string trueLabel = trimLabel(term.text.substr(0, comma));
                    const std::string falseLabel = trimLabel(term.text.substr(comma + 1));
                    successors[block.name].push_back(trueLabel);
                    successors[block.name].push_back(falseLabel);
                    predecessors[trueLabel].push_back(block.name);
                    predecessors[falseLabel].push_back(block.name);
                }
            }
        }

        std::unordered_set<std::string> allBlocks;
        for (const auto &block : function.blocks) {
            allBlocks.insert(block.name);
        }
        std::unordered_map<std::string, std::unordered_set<std::string>> dominators;
        for (std::size_t i = 0; i < function.blocks.size(); ++i) {
            dominators[function.blocks[i].name] = i == 0 ? std::unordered_set<std::string>{function.blocks[i].name} : allBlocks;
        }
        bool domChanged = true;
        while (domChanged) {
            domChanged = false;
            for (std::size_t i = 1; i < function.blocks.size(); ++i) {
                const std::string &name = function.blocks[i].name;
                std::unordered_set<std::string> next = allBlocks;
                if (predecessors[name].empty()) {
                    next.clear();
                }
                for (const std::string &pred : predecessors[name]) {
                    std::unordered_set<std::string> intersection;
                    for (const std::string &candidate : next) {
                        if (dominators[pred].count(candidate)) {
                            intersection.insert(candidate);
                        }
                    }
                    next = std::move(intersection);
                }
                next.insert(name);
                if (next != dominators[name]) {
                    dominators[name] = std::move(next);
                    domChanged = true;
                }
            }
        }

        std::unordered_map<std::string, std::unordered_set<int>> use;
        std::unordered_map<std::string, std::unordered_set<int>> def;
        for (const auto &block : function.blocks) {
            std::unordered_set<int> seenDef;
            for (const auto &inst : block.instructions) {
                if (inst.result >= 0 && intervals.count(inst.result)) {
                    def[block.name].insert(inst.result);
                    seenDef.insert(inst.result);
                }
                if (inst.opcode == ir::Opcode::Phi) {
                    continue;
                }
                for (const auto &operand : inst.operands) {
                    if (operand.constant || operand.id < 0 || !intervals.count(operand.id)) {
                        continue;
                    }
                    if (!seenDef.count(operand.id)) {
                        use[block.name].insert(operand.id);
                    }
                }
            }
        }
        for (const auto &[key, copies] : phiCopies_) {
            const std::size_t separator = key.find('\n');
            const std::string pred = separator == std::string::npos ? std::string{} : key.substr(0, separator);
            for (const PhiCopy &copy : copies) {
                if (!copy.source.constant && intervals.count(copy.source.id) && !def[pred].count(copy.source.id)) {
                    use[pred].insert(copy.source.id);
                }
            }
        }

        std::unordered_map<std::string, std::unordered_set<int>> liveIn;
        std::unordered_map<std::string, std::unordered_set<int>> liveOut;
        bool liveChanged = true;
        while (liveChanged) {
            liveChanged = false;
            for (auto it = function.blocks.rbegin(); it != function.blocks.rend(); ++it) {
                const std::string &name = it->name;
                std::unordered_set<int> nextOut;
                for (const std::string &succ : successors[name]) {
                    nextOut.insert(liveIn[succ].begin(), liveIn[succ].end());
                }
                std::unordered_set<int> nextIn = use[name];
                for (int id : nextOut) {
                    if (!def[name].count(id)) {
                        nextIn.insert(id);
                    }
                }
                if (nextOut != liveOut[name]) {
                    liveOut[name] = std::move(nextOut);
                    liveChanged = true;
                }
                if (nextIn != liveIn[name]) {
                    liveIn[name] = std::move(nextIn);
                    liveChanged = true;
                }
            }
        }

        std::unordered_map<int, std::unordered_set<int>> conflicts;
        auto addConflict = [&](int lhs, int rhs) {
            if (lhs == rhs || lhs < 0 || rhs < 0) {
                return;
            }
            conflicts[lhs].insert(rhs);
            conflicts[rhs].insert(lhs);
        };
        auto addLiveClique = [&](const std::unordered_set<int> &live) {
            for (auto lhs = live.begin(); lhs != live.end(); ++lhs) {
                auto rhs = lhs;
                for (++rhs; rhs != live.end(); ++rhs) {
                    addConflict(*lhs, *rhs);
                }
            }
        };
        for (const auto &block : function.blocks) {
            std::unordered_set<int> live = liveOut[block.name];
            addLiveClique(live);
            for (auto instIt = block.instructions.rbegin(); instIt != block.instructions.rend(); ++instIt) {
                const ir::Instruction &inst = *instIt;
                if (inst.opcode == ir::Opcode::Phi) {
                    continue;
                }
                if (inst.result >= 0 && intervals.count(inst.result)) {
                    for (int liveId : live) {
                        addConflict(inst.result, liveId);
                    }
                    live.erase(inst.result);
                }
                for (const auto &operand : inst.operands) {
                    if (!operand.constant && operand.id >= 0 && intervals.count(operand.id)) {
                        live.insert(operand.id);
                    }
                }
                addLiveClique(live);
            }
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Phi) {
                    break;
                }
                if (inst.result >= 0 && intervals.count(inst.result)) {
                    for (int liveId : live) {
                        addConflict(inst.result, liveId);
                    }
                    live.erase(inst.result);
                }
            }
            addLiveClique(liveIn[block.name]);
        }
        auto phiPairKey = [](int lhs, int rhs) {
            if (lhs > rhs) {
                std::swap(lhs, rhs);
            }
            return std::to_string(lhs) + ":" + std::to_string(rhs);
        };
        for (const auto &[key, copies] : phiCopies_) {
            const std::size_t separator = key.find('\n');
            const std::string pred = separator == std::string::npos ? std::string{} : key.substr(0, separator);
            std::unordered_set<int> edgeLive = liveOut[pred];
            std::unordered_set<std::string> allowedCoalescePairs;
            for (const PhiCopy &copy : copies) {
                if (!copy.source.constant && copy.source.id >= 0 && intervals.count(copy.source.id)) {
                    edgeLive.insert(copy.source.id);
                    if (copy.target >= 0 && intervals.count(copy.target)) {
                        allowedCoalescePairs.insert(phiPairKey(copy.source.id, copy.target));
                    }
                }
                if (copy.target >= 0 && intervals.count(copy.target)) {
                    edgeLive.insert(copy.target);
                }
            }
            for (auto lhs = edgeLive.begin(); lhs != edgeLive.end(); ++lhs) {
                auto rhs = lhs;
                for (++rhs; rhs != edgeLive.end(); ++rhs) {
                    if (!allowedCoalescePairs.count(phiPairKey(*lhs, *rhs))) {
                        addConflict(*lhs, *rhs);
                    }
                }
            }
        }
        std::unordered_map<int, std::unordered_map<int, int>> affinities;
        auto addAffinity = [&](int lhs, int rhs, int weight) {
            if (lhs == rhs || lhs < 0 || rhs < 0) {
                return;
            }
            auto lhsInterval = intervals.find(lhs);
            auto rhsInterval = intervals.find(rhs);
            if (lhsInterval == intervals.end() || rhsInterval == intervals.end() ||
                lhsInterval->second.type != rhsInterval->second.type) {
                return;
            }
            const auto conflict = conflicts.find(lhs);
            if (conflict != conflicts.end() && conflict->second.count(rhs)) {
                return;
            }
            affinities[lhs][rhs] += weight;
            affinities[rhs][lhs] += weight;
        };
        for (const auto &[_, copies] : phiCopies_) {
            for (const PhiCopy &copy : copies) {
                if (!copy.source.constant && copy.source.id >= 0) {
                    addAffinity(copy.source.id, copy.target, 64);
                }
            }
        }

        for (const auto &block : function.blocks) {
            const int start = blockStart[block.name];
            const int end = blockEnd[block.name];
            for (int id : liveIn[block.name]) {
                auto found = intervals.find(id);
                if (found != intervals.end()) {
                    found->second.start = std::min(found->second.start, start);
                    found->second.end = std::max(found->second.end, end);
                }
            }
            for (int id : liveOut[block.name]) {
                auto found = intervals.find(id);
                if (found != intervals.end()) {
                    found->second.end = std::max(found->second.end, end);
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

        std::vector<Interval> ints;
        std::vector<Interval> floats;
        for (auto [id, interval] : intervals) {
            bool dominatesUses = interval.phi;
            if (!dominatesUses) {
                const auto def = defBlock.find(id);
                dominatesUses = def != defBlock.end();
                if (dominatesUses) {
                    for (const std::string &useBlock : useBlocks[id]) {
                        if (!dominators[useBlock].count(def->second)) {
                            dominatesUses = false;
                            break;
                        }
                    }
                }
            }
            const bool profitableCrossBlock = interval.phi || interval.param || interval.weight >= 1;
            if (!profitableCrossBlock || !dominatesUses || !interval.crossBlock ||
                interval.weight < 2 || interval.end <= interval.start) {
                continue;
            }
            interval.weight += interval.phi ? 120 : 0;
            if (interval.type == ir::TypeKind::F32) {
                floats.push_back(interval);
            } else if (interval.type == ir::TypeKind::I32 || interval.type == ir::TypeKind::Ptr) {
                ints.push_back(interval);
            }
        }
        graphColorAssign(ints, currentGlobalIntRegs_, usedIntRegs, conflicts, affinities, globalValueRegs_);
        graphColorAssign(floats, globalFloatRegs(), usedFloatRegs, conflicts, affinities, globalValueFRegs_);
    }

    struct ActiveInterval {
        int id = -1;
        int start = 0;
        int end = 0;
        int weight = 0;
        std::string reg;
    };

    static void linearScanAssign(std::vector<Interval> intervals, const std::vector<std::string> &registers,
                                 const std::unordered_set<std::string> &reserved,
                                 std::unordered_map<int, std::string> &assignment) {
        std::sort(intervals.begin(), intervals.end(), [](const Interval &lhs, const Interval &rhs) {
            if (lhs.weight != rhs.weight) {
                return lhs.weight > rhs.weight;
            }
            const int lhsLength = lhs.end - lhs.start;
            const int rhsLength = rhs.end - rhs.start;
            if (lhsLength != rhsLength) {
                return lhsLength > rhsLength;
            }
            if (lhs.start != rhs.start) {
                return lhs.start < rhs.start;
            }
            return lhs.id < rhs.id;
        });

        std::unordered_map<std::string, std::vector<ActiveInterval>> assigned;

        for (const Interval &interval : intervals) {
            std::string chosen;
            for (auto regIt = registers.rbegin(); regIt != registers.rend(); ++regIt) {
                const std::string &reg = *regIt;
                if (reserved.count(reg)) {
                    continue;
                }
                if (assigned[reg].empty()) {
                    chosen = reg;
                    break;
                }
            }
            if (!chosen.empty()) {
                assigned[chosen].push_back(ActiveInterval{interval.id, interval.start, interval.end, interval.weight, chosen});
                assignment[interval.id] = chosen;
            }
        }
    }

    static void graphColorAssign(std::vector<Interval> intervals, const std::vector<std::string> &registers,
                                 const std::unordered_set<std::string> &reserved,
                                 const std::unordered_map<int, std::unordered_set<int>> &conflicts,
                                 const std::unordered_map<int, std::unordered_map<int, int>> &affinities,
                                 std::unordered_map<int, std::string> &assignment) {
        std::sort(intervals.begin(), intervals.end(), [](const Interval &lhs, const Interval &rhs) {
            if (lhs.weight != rhs.weight) {
                return lhs.weight > rhs.weight;
            }
            const int lhsLength = lhs.end - lhs.start;
            const int rhsLength = rhs.end - rhs.start;
            if (lhsLength != rhsLength) {
                return lhsLength > rhsLength;
            }
            return lhs.id < rhs.id;
        });

        std::unordered_map<std::string, std::vector<int>> assigned;
        for (const Interval &interval : intervals) {
            std::string chosen;
            int bestOccupancy = 0;
            int bestAffinity = -1;
            for (auto regIt = registers.rbegin(); regIt != registers.rend(); ++regIt) {
                const std::string &reg = *regIt;
                if (reserved.count(reg)) {
                    continue;
                }
                bool ok = true;
                for (int other : assigned[reg]) {
                    const auto conflict = conflicts.find(interval.id);
                    if (conflict != conflicts.end() && conflict->second.count(other)) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) {
                    continue;
                }
                const int occupancy = static_cast<int>(assigned[reg].size());
                int affinityScore = 0;
                const auto affinity = affinities.find(interval.id);
                if (affinity != affinities.end()) {
                    for (int other : assigned[reg]) {
                        const auto related = affinity->second.find(other);
                        if (related != affinity->second.end()) {
                            affinityScore += related->second;
                        }
                    }
                }
                if (chosen.empty() || affinityScore > bestAffinity ||
                    (affinityScore == bestAffinity && occupancy < bestOccupancy)) {
                    chosen = reg;
                    bestOccupancy = occupancy;
                    bestAffinity = affinityScore;
                }
            }
            if (!chosen.empty()) {
                assigned[chosen].push_back(interval.id);
                assignment[interval.id] = chosen;
            }
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
            if (!isCalleeSavedIntReg(reg)) {
                return;
            }
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
        for (const auto &reg : requiredIntCacheRegs(function)) {
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

    std::vector<std::string> requiredIntCacheRegs(const ir::Function &function) const {
        std::unordered_map<int, ir::TypeKind> resultTypes;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.result >= 0) {
                    resultTypes[inst.result] = inst.resultType.kind;
                }
            }
        }
        std::size_t maxNeeded = 0;
        for (const auto &block : function.blocks) {
            const auto locals = blockLocalValues_.find(block.name);
            if (locals == blockLocalValues_.end()) {
                continue;
            }
            std::size_t localInts = 0;
            for (int id : locals->second) {
                const auto type = resultTypes.find(id);
                if (type != resultTypes.end() &&
                    (type->second == ir::TypeKind::I32 || type->second == ir::TypeKind::Ptr)) {
                    ++localInts;
                }
            }
            maxNeeded = std::max(maxNeeded, localInts);
        }

        std::vector<std::string> regs;
        for (const auto &reg : currentIntCacheRegs_) {
            if (!isReservedIntReg(reg)) {
                regs.push_back(reg);
                if (regs.size() >= maxNeeded) {
                    break;
                }
            }
        }
        return regs;
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
                    const std::string dst = prepareIntLoadResultRegister(inst);
                    if (dst != intAlloca->second) {
                        out_ << "\tmv " << dst << ", " << intAlloca->second << "\n";
                    }
                    if (dst == "a0") {
                        storeResult(inst);
                    }
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
                const std::string dst = prepareIntLoadResultRegister(inst);
                out_ << "\tlw " << dst << ", 0(a0)\n";
                if (dst == "a0") {
                    storeResult(inst);
                }
            }
            break;
        case ir::Opcode::Store:
            if (!inst.operands[1].constant) {
                const auto intAlloca = scalarIntAllocas_.find(inst.operands[1].id);
                if (intAlloca != scalarIntAllocas_.end()) {
                    emitValueOperandTo(intAlloca->second, inst.operands[0]);
                    break;
                }
                const auto floatAlloca = scalarFloatAllocas_.find(inst.operands[1].id);
                if (floatAlloca != scalarFloatAllocas_.end()) {
                    emitFloatOperandTo(floatAlloca->second, inst.operands[0]);
                    break;
                }
            }
            if (inst.operands[0].type.kind == ir::TypeKind::F32) {
                emitValueOperand(inst.operands[0]);
                emitAddressOperand(inst.operands[1]);
                out_ << "\tfsw fa0, 0(a0)\n";
            } else {
                std::string sourceReg = existingIntRegister(inst.operands[0]);
                if (sourceReg.empty() || sourceReg == "a0") {
                    emitValueOperandTo("t2", inst.operands[0]);
                    sourceReg = "t2";
                }
                emitAddressOperand(inst.operands[1]);
                out_ << "\tsw " << sourceReg << ", 0(a0)\n";
            }
            break;
        case ir::Opcode::Gep:
            {
                const std::string dst = prepareIntResultRegister(inst);
                emitGepAddressTo(dst, inst);
                if (dst == "a0") {
                    storeResult(inst);
                }
            }
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
            if (inst.resultType.kind == ir::TypeKind::F32) {
                emitValueOperand(inst.operands[0]);
                out_ << "\tfneg.s fa0, fa0\n";
                storeFResult(inst);
            } else {
                const std::string dst = prepareIntResultRegister(inst);
                emitValueOperandTo(dst, inst.operands[0]);
                out_ << "\tnegw " << dst << ", " << dst << "\n";
                if (dst == "a0") {
                    storeResult(inst);
                }
            }
            break;
        case ir::Opcode::Not:
            if (inst.operands[0].type.kind == ir::TypeKind::F32) {
                emitValueOperand(inst.operands[0]);
                out_ << "\tfmv.w.x ft0, zero\n";
                out_ << "\tfeq.s a0, fa0, ft0\n";
                storeResult(inst);
                break;
            }
            {
                const std::string dst = prepareIntResultRegister(inst);
                emitValueOperandTo(dst, inst.operands[0]);
                out_ << "\tseqz " << dst << ", " << dst << "\n";
                if (dst == "a0") {
                    storeResult(inst);
                }
            }
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

    void emitGepAddress(const ir::Instruction &inst) {
        emitGepAddressTo("a0", inst);
    }

    void emitGepAddressTo(const std::string &dst, const ir::Instruction &inst) {
        emitAddressOperandTo("t2", inst.operands[0]);
        if (inst.operands[1].constant && inst.operands[1].type.kind == ir::TypeKind::I32) {
            const long long index = std::strtoll(inst.operands[1].name.c_str(), nullptr, 0);
            const long long offset = index * 4;
            if (fitsI12IR(static_cast<int>(offset))) {
                out_ << "\taddi " << dst << ", t2, " << offset << "\n";
            } else {
                out_ << "\tli " << dst << ", " << offset << "\n";
                out_ << "\tadd " << dst << ", t2, " << dst << "\n";
            }
        } else {
            const std::string indexReg = existingIntRegister(inst.operands[1]);
            if (!indexReg.empty()) {
                out_ << "\tslli " << dst << ", " << indexReg << ", 2\n";
            } else {
                emitValueOperandTo(dst, inst.operands[1]);
                out_ << "\tslli " << dst << ", " << dst << ", 2\n";
            }
            out_ << "\tadd " << dst << ", t2, " << dst << "\n";
        }
    }

    void emitFoldedGepStore(const ir::Instruction &gep, const ir::Instruction &store) {
        if (store.operands[0].type.kind == ir::TypeKind::F32) {
            emitValueOperand(store.operands[0]);
            emitGepAddress(gep);
            out_ << "\tfsw fa0, 0(a0)\n";
            return;
        }
        std::string sourceReg = existingIntRegister(store.operands[0]);
        if (sourceReg.empty() || sourceReg == "a0" || sourceReg == "t2") {
            emitValueOperandTo("t3", store.operands[0]);
            sourceReg = "t3";
        }
        emitGepAddress(gep);
        out_ << "\tsw " << sourceReg << ", 0(a0)\n";
    }

    void emitFoldedGepLoad(const ir::Instruction &gep, const ir::Instruction &load) {
        emitGepAddress(gep);
        if (load.resultType.kind == ir::TypeKind::F32) {
            out_ << "\tflw fa0, 0(a0)\n";
            storeFResult(load);
        } else {
            out_ << "\tlw a0, 0(a0)\n";
            storeResult(load);
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
        if (emitRegisterOperandBinary(inst)) {
            return;
        }
        const std::string dst = prepareIntResultRegister(inst);
        emitValueOperandTo("t0", inst.operands[0]);
        emitValueOperand(inst.operands[1]);
        switch (inst.opcode) {
        case ir::Opcode::Add: out_ << "\taddw " << dst << ", t0, a0\n"; break;
        case ir::Opcode::Sub: out_ << "\tsubw " << dst << ", t0, a0\n"; break;
        case ir::Opcode::Mul: out_ << "\tmulw " << dst << ", t0, a0\n"; break;
        case ir::Opcode::Div: out_ << "\tdivw " << dst << ", t0, a0\n"; break;
        case ir::Opcode::Mod: out_ << "\tremw " << dst << ", t0, a0\n"; break;
        case ir::Opcode::ICmp:
            if (inst.text == "lt") out_ << "\tslt " << dst << ", t0, a0\n";
            else if (inst.text == "gt") out_ << "\tslt " << dst << ", a0, t0\n";
            else if (inst.text == "eq") { out_ << "\tsubw " << dst << ", t0, a0\n"; out_ << "\tseqz " << dst << ", " << dst << "\n"; }
            else if (inst.text == "ne") { out_ << "\tsubw " << dst << ", t0, a0\n"; out_ << "\tsnez " << dst << ", " << dst << "\n"; }
            else if (inst.text == "le") { out_ << "\tslt " << dst << ", a0, t0\n"; out_ << "\txori " << dst << ", " << dst << ", 1\n"; }
            else if (inst.text == "ge") { out_ << "\tslt " << dst << ", t0, a0\n"; out_ << "\txori " << dst << ", " << dst << ", 1\n"; }
            break;
        default:
            break;
        }
        if (dst == "a0") {
            storeResult(inst);
        }
    }

    bool emitRegisterOperandBinary(const ir::Instruction &inst) {
        if (inst.operands.size() != 2) {
            return false;
        }
        if (inst.opcode != ir::Opcode::Add && inst.opcode != ir::Opcode::Sub &&
            inst.opcode != ir::Opcode::Mul && inst.opcode != ir::Opcode::Div &&
            inst.opcode != ir::Opcode::Mod) {
            return false;
        }

        std::string lhs = existingIntRegister(inst.operands[0]);
        if (lhs.empty()) {
            emitValueOperandTo("t0", inst.operands[0]);
            lhs = "t0";
        }

        std::string rhs;
        if (inst.operands[1].constant && inst.operands[1].type.kind == ir::TypeKind::I32 &&
            std::strtoll(inst.operands[1].name.c_str(), nullptr, 0) == 0) {
            rhs = "zero";
        } else {
            rhs = existingIntRegister(inst.operands[1]);
        }
        if (rhs.empty()) {
            if (lhs == "a0") {
                out_ << "\tmv t0, a0\n";
                lhs = "t0";
            }
            emitValueOperand(inst.operands[1]);
            rhs = "a0";
        }

        const std::string dst = prepareIntResultRegister(inst);
        switch (inst.opcode) {
        case ir::Opcode::Add: out_ << "\taddw " << dst << ", " << lhs << ", " << rhs << "\n"; break;
        case ir::Opcode::Sub: out_ << "\tsubw " << dst << ", " << lhs << ", " << rhs << "\n"; break;
        case ir::Opcode::Mul: out_ << "\tmulw " << dst << ", " << lhs << ", " << rhs << "\n"; break;
        case ir::Opcode::Div: out_ << "\tdivw " << dst << ", " << lhs << ", " << rhs << "\n"; break;
        case ir::Opcode::Mod: out_ << "\tremw " << dst << ", " << lhs << ", " << rhs << "\n"; break;
        default: return false;
        }
        if (dst == "a0") {
            storeResult(inst);
        }
        return true;
    }

    std::string resultIntRegister(const ir::Instruction &inst) const {
        const auto global = globalValueRegs_.find(inst.result);
        if (global != globalValueRegs_.end()) {
            return global->second;
        }
        const auto local = valueRegs_.find(inst.result);
        if (local != valueRegs_.end()) {
            return local->second;
        }
        return "a0";
    }

    std::string prepareIntResultRegister(const ir::Instruction &inst) {
        const auto global = globalValueRegs_.find(inst.result);
        if (global != globalValueRegs_.end()) {
            return global->second;
        }
        const auto local = valueRegs_.find(inst.result);
        if (local != valueRegs_.end()) {
            return local->second;
        }
        if (inst.result == elideResultStoreId_ || !isCurrentBlockLocal(inst.result)) {
            return "a0";
        }
        if (inst.resultType.kind != ir::TypeKind::I32 && inst.resultType.kind != ir::TypeKind::Ptr) {
            return "a0";
        }
        return reserveIntCacheResult(inst.result);
    }

    std::string reserveIntCacheResult(int id) {
        if (id < 0) {
            return "a0";
        }
        const auto existing = valueRegs_.find(id);
        if (existing != valueRegs_.end()) {
            return existing->second;
        }
        while (nextIntReg_ < currentIntCacheRegs_.size() && isReservedIntReg(currentIntCacheRegs_[nextIntReg_])) {
            ++nextIntReg_;
        }
        if (nextIntReg_ >= currentIntCacheRegs_.size()) {
            return "a0";
        }
        const std::string &reg = currentIntCacheRegs_[nextIntReg_++];
        valueRegs_[id] = reg;
        return reg;
    }

    std::string prepareIntLoadResultRegister(const ir::Instruction &inst) {
        const auto global = globalValueRegs_.find(inst.result);
        if (global != globalValueRegs_.end()) {
            return global->second;
        }
        const auto local = valueRegs_.find(inst.result);
        if (local != valueRegs_.end()) {
            return local->second;
        }
        if (inst.result == elideResultStoreId_ || !isCurrentBlockLocal(inst.result)) {
            return "a0";
        }
        if (inst.resultType.kind != ir::TypeKind::I32 && inst.resultType.kind != ir::TypeKind::Ptr) {
            return "a0";
        }
        return reserveIntCacheResult(inst.result);
    }

    void emitCondBranch(const std::string &text) {
        const std::size_t comma = text.find(',');
        const std::string trueLabel = trimLabel(text.substr(0, comma));
        const std::string falseLabel = comma == std::string::npos ? std::string{} : trimLabel(text.substr(comma + 1));
        if (!useLongJumps_) {
            if (trueLabel == nextBlockName_) {
                out_ << "\tbeqz a0, " << blockLabel(falseLabel) << "\n";
                return;
            }
            if (falseLabel == nextBlockName_) {
                out_ << "\tbnez a0, " << blockLabel(trueLabel) << "\n";
                return;
            }
            out_ << "\tbnez a0, " << blockLabel(trueLabel) << "\n";
            emitJump(blockLabel(falseLabel));
            return;
        }
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

    static std::string inverseBranchOp(const std::string &op) {
        if (op == "beq") return "bne";
        if (op == "bne") return "beq";
        if (op == "blt") return "bge";
        if (op == "bge") return "blt";
        return {};
    }

    void emitRawBranch(const std::string &op, const std::string &lhs, const std::string &rhs, const std::string &label) {
        out_ << '\t' << op << ' ' << lhs << ", " << rhs << ", " << label << "\n";
    }

    void emitSelectedCondBranch(const std::string &op, const std::string &lhs, const std::string &rhs,
                                const std::string &trueLabel, const std::string &falseLabel) {
        if (!useLongJumps_) {
            if (trueLabel == nextBlockName_) {
                emitRawBranch(inverseBranchOp(op), lhs, rhs, blockLabel(falseLabel));
                return;
            }
            if (falseLabel == nextBlockName_) {
                emitRawBranch(op, lhs, rhs, blockLabel(trueLabel));
                return;
            }
            emitRawBranch(op, lhs, rhs, blockLabel(trueLabel));
            emitJump(blockLabel(falseLabel));
            return;
        }
        const std::string trueLocal = newPhiEdgeLabel();
        emitRawBranch(op, lhs, rhs, trueLocal);
        emitJump(blockLabel(falseLabel));
        out_ << trueLocal << ":\n";
        emitJump(blockLabel(trueLabel));
    }

    bool selectIntegerCompareBranch(const ir::Instruction &cmp, std::string &op, std::string &lhs, std::string &rhs) {
        if (cmp.opcode != ir::Opcode::ICmp || cmp.operands.size() != 2) {
            return false;
        }
        const auto selectOperand = [&](const ir::Value &value, const std::string &fallback) {
            if (value.constant && value.type.kind == ir::TypeKind::I32 &&
                std::strtoll(value.name.c_str(), nullptr, 0) == 0) {
                return std::string("zero");
            }
            const std::string existing = existingIntRegister(value);
            if (!existing.empty()) {
                return existing;
            }
            emitValueOperandTo(fallback, value);
            return fallback;
        };

        lhs = selectOperand(cmp.operands[0], "t0");
        rhs = selectOperand(cmp.operands[1], "a0");

        const std::string originalLhs = lhs;
        if (cmp.operands[0].constant && cmp.operands[0].type.kind == ir::TypeKind::I32 &&
            std::strtoll(cmp.operands[0].name.c_str(), nullptr, 0) == 0) {
            lhs = "zero";
        }
        if (cmp.text == "eq") {
            op = "beq";
        } else if (cmp.text == "ne") {
            op = "bne";
        } else if (cmp.text == "lt") {
            op = "blt";
        } else if (cmp.text == "ge") {
            op = "bge";
        } else if (cmp.text == "gt") {
            op = "blt";
            lhs = rhs;
            rhs = originalLhs;
        } else if (cmp.text == "le") {
            op = "bge";
            lhs = rhs;
            rhs = originalLhs;
        } else {
            return false;
        }
        return true;
    }

    void emitFloatCompareResult(const ir::Instruction &cmp) {
        emitValueOperand(cmp.operands[0]);
        out_ << "\tfmv.s ft0, fa0\n";
        emitValueOperand(cmp.operands[1]);
        if (cmp.text == "lt") out_ << "\tflt.s a0, ft0, fa0\n";
        else if (cmp.text == "gt") out_ << "\tflt.s a0, fa0, ft0\n";
        else if (cmp.text == "le") out_ << "\tfle.s a0, ft0, fa0\n";
        else if (cmp.text == "ge") out_ << "\tfle.s a0, fa0, ft0\n";
        else if (cmp.text == "eq") out_ << "\tfeq.s a0, ft0, fa0\n";
        else if (cmp.text == "ne") { out_ << "\tfeq.s a0, ft0, fa0\n"; out_ << "\txori a0, a0, 1\n"; }
    }

    void emitCompareCondBranchWithPhi(const ir::Instruction &cmp, const std::string &text) {
        const std::size_t comma = text.find(',');
        const std::string trueLabel = trimLabel(text.substr(0, comma));
        const std::string falseLabel = comma == std::string::npos ? std::string{} : trimLabel(text.substr(comma + 1));
        if (!hasPhiCopies(currentBlockName_, trueLabel) && !hasPhiCopies(currentBlockName_, falseLabel)) {
            if (cmp.opcode == ir::Opcode::ICmp) {
                std::string op;
                std::string lhs;
                std::string rhs;
                if (selectIntegerCompareBranch(cmp, op, lhs, rhs)) {
                    emitSelectedCondBranch(op, lhs, rhs, trueLabel, falseLabel);
                    return;
                }
                emitBinary(cmp);
                emitValueOperand(ir::Value{cmp.result, cmp.resultType, {}, false});
                emitCondBranch(text);
                return;
            }
            emitFloatCompareResult(cmp);
            emitCondBranch(text);
            return;
        }

        const std::string trueEdge = newPhiEdgeLabel();
        const std::string falseEdge = newPhiEdgeLabel();
        const std::string trueLocal = newPhiEdgeLabel();
        if (cmp.opcode == ir::Opcode::ICmp) {
            std::string op;
            std::string lhs;
            std::string rhs;
            if (selectIntegerCompareBranch(cmp, op, lhs, rhs)) {
                emitRawBranch(op, lhs, rhs, trueLocal);
            } else {
                emitBinary(cmp);
                emitValueOperand(ir::Value{cmp.result, cmp.resultType, {}, false});
                out_ << "\tbnez a0, " << trueLocal << "\n";
            }
        } else {
            emitFloatCompareResult(cmp);
            out_ << "\tbnez a0, " << trueLocal << "\n";
        }
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

    static std::string swappedBranchText(const std::string &text) {
        const std::size_t comma = text.find(',');
        if (comma == std::string::npos) {
            return text;
        }
        const std::string trueLabel = trimLabel(text.substr(0, comma));
        const std::string falseLabel = trimLabel(text.substr(comma + 1));
        return falseLabel + ", " + trueLabel;
    }

    void emitNotCondBranchWithPhi(const ir::Instruction &inst, const std::string &text) {
        if (inst.operands.empty()) {
            emitValueOperand(ir::Value{-1, ir::Type{ir::TypeKind::I32}, "0", true});
            emitCondBranchWithPhi(text);
            return;
        }
        if (inst.operands[0].type.kind == ir::TypeKind::F32) {
            emitValueOperand(inst.operands[0]);
            out_ << "\tfmv.w.x ft0, zero\n";
            out_ << "\tfeq.s a0, fa0, ft0\n";
            emitCondBranchWithPhi(text);
            return;
        }
        emitValueOperand(inst.operands[0]);
        emitCondBranchWithPhi(swappedBranchText(text));
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
        const auto &copies = found->second;
        if (copies.empty()) {
            return;
        }
        auto phiSourceReg = [&](const PhiCopy &copy) -> std::string {
            if (copy.source.constant || copy.source.id < 0) {
                return {};
            }
            if (copy.targetType.kind == ir::TypeKind::F32) {
                const auto global = globalValueFRegs_.find(copy.source.id);
                if (global != globalValueFRegs_.end()) {
                    return global->second;
                }
                const auto local = valueFRegs_.find(copy.source.id);
                return local == valueFRegs_.end() ? std::string{} : local->second;
            }
            const auto global = globalValueRegs_.find(copy.source.id);
            if (global != globalValueRegs_.end()) {
                return global->second;
            }
            const auto local = valueRegs_.find(copy.source.id);
            return local == valueRegs_.end() ? std::string{} : local->second;
        };
        auto phiTargetReg = [&](const PhiCopy &copy) -> std::string {
            if (copy.targetType.kind == ir::TypeKind::F32) {
                const auto target = globalValueFRegs_.find(copy.target);
                return target == globalValueFRegs_.end() ? std::string{} : target->second;
            }
            const auto target = globalValueRegs_.find(copy.target);
            return target == globalValueRegs_.end() ? std::string{} : target->second;
        };
        bool needsSnapshot = false;
        std::vector<std::string> targetRegs;
        targetRegs.reserve(copies.size());
        for (const PhiCopy &copy : copies) {
            const std::string target = phiTargetReg(copy);
            if (!target.empty()) {
                targetRegs.push_back(target);
            }
        }
        for (const PhiCopy &copy : copies) {
            const std::string source = phiSourceReg(copy);
            const std::string ownTarget = phiTargetReg(copy);
            if (source.empty()) {
                continue;
            }
            for (const std::string &target : targetRegs) {
                if (source == target && source != ownTarget) {
                    needsSnapshot = true;
                    break;
                }
            }
            if (needsSnapshot) {
                break;
            }
        }
        if (!needsSnapshot) {
            for (const PhiCopy &copy : copies) {
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
            return;
        }
        for (const PhiCopy &copy : copies) {
            if (copy.targetType.kind == ir::TypeKind::F32) {
                emitFloatOperandTo("fa0", copy.source);
                out_ << "\taddi sp, sp, -16\n";
                out_ << "\tfsw fa0, 8(sp)\n";
            } else {
                emitValueOperandTo("a0", copy.source);
                out_ << "\taddi sp, sp, -16\n";
                out_ << "\tsd a0, 8(sp)\n";
            }
        }
        for (std::size_t i = copies.size(); i > 0; --i) {
            const PhiCopy &copy = copies[i - 1];
            if (copy.targetType.kind == ir::TypeKind::F32) {
                out_ << "\tflw fa0, 8(sp)\n";
                out_ << "\taddi sp, sp, 16\n";
                const auto targetReg = globalValueFRegs_.find(copy.target);
                if (targetReg != globalValueFRegs_.end()) {
                    if (targetReg->second != "fa0") {
                        out_ << "\tfmv.s " << targetReg->second << ", fa0\n";
                    }
                }
            } else {
                const auto targetReg = globalValueRegs_.find(copy.target);
                out_ << "\tld a0, 8(sp)\n";
                out_ << "\taddi sp, sp, 16\n";
                if (targetReg != globalValueRegs_.end()) {
                    if (targetReg->second != "a0") {
                        out_ << "\tmv " << targetReg->second << ", a0\n";
                    }
                }
            }
            if (copy.targetType.kind == ir::TypeKind::F32) {
                if (!globalValueFRegs_.count(copy.target)) {
                    storeFReg("fa0", valueOffsets_[copy.target]);
                }
            } else if (!globalValueRegs_.count(copy.target)) {
                storeReg("a0", valueOffsets_[copy.target]);
            }
        }
    }

    std::string newPhiEdgeLabel() {
        return ".Lir." + currentFunctionName_ + ".phi_edge." + std::to_string(nextPhiEdgeLabel_++);
    }

    void emitJump(const std::string &label) {
        if (!useLongJumps_) {
            out_ << "\tj " << label << "\n";
            return;
        }
        out_ << "\tlla t6, " << label << "\n";
        out_ << "\tjr t6\n";
    }

    bool emitImmediateBinary(const ir::Instruction &inst) {
        if (inst.operands.size() != 2 || !inst.operands[1].constant || inst.operands[1].type.kind != ir::TypeKind::I32) {
            return false;
        }
        const long long rhs = std::strtoll(inst.operands[1].name.c_str(), nullptr, 0);
        const std::string dst = prepareIntResultRegister(inst);
        auto finish = [&]() {
            if (dst == "a0") {
                storeResult(inst);
            }
        };
        switch (inst.opcode) {
        case ir::Opcode::Add:
            if (!fitsI12IR(static_cast<int>(rhs))) return false;
            emitValueOperandTo(dst, inst.operands[0]);
            out_ << "\taddiw " << dst << ", " << dst << ", " << rhs << "\n";
            finish();
            return true;
        case ir::Opcode::Sub:
            if (!fitsI12IR(static_cast<int>(-rhs))) return false;
            emitValueOperandTo(dst, inst.operands[0]);
            out_ << "\taddiw " << dst << ", " << dst << ", " << -rhs << "\n";
            finish();
            return true;
        case ir::Opcode::Mul:
            if (rhs > 0 && isPowerOfTwo(rhs)) {
                emitValueOperandTo(dst, inst.operands[0]);
                out_ << "\tslliw " << dst << ", " << dst << ", " << log2Int(rhs) << "\n";
                finish();
                return true;
            }
            return false;
        case ir::Opcode::ICmp:
            if ((inst.text == "eq" || inst.text == "ne") && rhs == 0) {
                emitValueOperandTo(dst, inst.operands[0]);
                out_ << (inst.text == "eq" ? "\tseqz " : "\tsnez ") << dst << ", " << dst << "\n";
                finish();
                return true;
            }
            if ((inst.text == "eq" || inst.text == "ne") && fitsI12IR(static_cast<int>(rhs))) {
                emitValueOperandTo(dst, inst.operands[0]);
                out_ << "\txori " << dst << ", " << dst << ", " << rhs << "\n";
                out_ << (inst.text == "eq" ? "\tseqz " : "\tsnez ") << dst << ", " << dst << "\n";
                finish();
                return true;
            }
            if (inst.text == "lt" && fitsI12IR(static_cast<int>(rhs))) {
                emitValueOperandTo(dst, inst.operands[0]);
                out_ << "\tslti " << dst << ", " << dst << ", " << rhs << "\n";
                finish();
                return true;
            }
            if (inst.text == "ge" && fitsI12IR(static_cast<int>(rhs))) {
                emitValueOperandTo(dst, inst.operands[0]);
                out_ << "\tslti " << dst << ", " << dst << ", " << rhs << "\n";
                out_ << "\txori " << dst << ", " << dst << ", 1\n";
                finish();
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
        if (inst.operands.empty()) {
            out_ << "\tcall " << inst.text << "\n";
            if (inst.result >= 0) {
                if (inst.resultType.kind == ir::TypeKind::F32) {
                    storeFResult(inst);
                } else {
                    storeResult(inst);
                }
            }
            return;
        }
        if (inst.operands.size() == 1) {
            const auto &arg = inst.operands[0];
            if (arg.type.kind == ir::TypeKind::F32) {
                emitFloatOperandTo("fa0", arg);
            } else {
                emitValueOperandTo("a0", arg);
            }
            out_ << "\tcall " << inst.text << "\n";
            if (inst.result >= 0) {
                if (inst.resultType.kind == ir::TypeKind::F32) {
                    storeFResult(inst);
                } else {
                    storeResult(inst);
                }
            }
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
        if (stackArgs == 0) {
            int intIndex = 0;
            int floatIndex = 0;
            for (const auto &operand : inst.operands) {
                if (operand.type.kind == ir::TypeKind::F32) {
                    emitFloatOperandTo("fa" + std::to_string(floatIndex++), operand);
                } else {
                    emitValueOperandTo("a" + std::to_string(intIndex++), operand);
                }
            }
            out_ << "\tcall " << inst.text << "\n";
            if (inst.result >= 0) {
                if (inst.resultType.kind == ir::TypeKind::F32) {
                    storeFResult(inst);
                } else {
                    storeResult(inst);
                }
            }
            return;
        }
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
        const auto transientInt = transientRegs_.find(value.id);
        if (transientInt != transientRegs_.end()) {
            if (transientInt->second != "a0") {
                out_ << "\tmv a0, " << transientInt->second << "\n";
            }
            return;
        }
        const auto transientFloat = transientFRegs_.find(value.id);
        if (transientFloat != transientFRegs_.end()) {
            if (transientFloat->second != "fa0") {
                out_ << "\tfmv.s fa0, " << transientFloat->second << "\n";
            }
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

    std::string existingIntRegister(const ir::Value &value) const {
        if (value.constant || value.id < 0) {
            return {};
        }
        const auto transient = transientRegs_.find(value.id);
        if (transient != transientRegs_.end()) {
            return transient->second;
        }
        const auto globalInt = globalValueRegs_.find(value.id);
        if (globalInt != globalValueRegs_.end()) {
            return globalInt->second;
        }
        const auto intReg = valueRegs_.find(value.id);
        if (intReg != valueRegs_.end()) {
            return intReg->second;
        }
        return {};
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
        const auto transient = transientRegs_.find(value.id);
        if (transient != transientRegs_.end()) {
            if (transient->second != reg) {
                out_ << "\tmv " << reg << ", " << transient->second << "\n";
            }
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
        const auto transient = transientFRegs_.find(value.id);
        if (transient != transientFRegs_.end()) {
            if (transient->second != reg) {
                out_ << "\tfmv.s " << reg << ", " << transient->second << "\n";
            }
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
        if (inst.result == elideResultStoreId_) {
            return;
        }
        const auto global = globalValueRegs_.find(inst.result);
        if (global != globalValueRegs_.end()) {
            if (global->second != "a0") {
                out_ << "\tmv " << global->second << ", a0\n";
            }
            return;
        }
        if (valueRegs_.count(inst.result)) {
            return;
        }
        if (cacheResult(inst.result, inst.resultType) && isCurrentBlockLocal(inst.result)) {
            return;
        }
        storeReg("a0", valueOffsets_[inst.result]);
    }

    void storeFResult(const ir::Instruction &inst) {
        if (inst.result == elideResultStoreId_) {
            return;
        }
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
            while (nextFloatReg_ < floatCacheRegs().size() && isGlobalFloatReg(floatCacheRegs()[nextFloatReg_])) {
                ++nextFloatReg_;
            }
            if (nextFloatReg_ >= floatCacheRegs().size()) {
                return false;
            }
            const std::string &reg = floatCacheRegs()[nextFloatReg_++];
            out_ << "\tfmv.s " << reg << ", fa0\n";
            valueFRegs_[id] = reg;
            return true;
        }
        if (type.kind == ir::TypeKind::I32 || type.kind == ir::TypeKind::Ptr) {
            while (nextIntReg_ < currentIntCacheRegs_.size() && isReservedIntReg(currentIntCacheRegs_[nextIntReg_])) {
                ++nextIntReg_;
            }
            if (nextIntReg_ >= currentIntCacheRegs_.size()) {
                return false;
            }
            const std::string &reg = currentIntCacheRegs_[nextIntReg_++];
            out_ << "\tmv " << reg << ", a0\n";
            valueRegs_[id] = reg;
            return true;
        }
        return false;
    }

    bool isCurrentBlockLocal(int id) const {
        return currentBlockLocalValues_ && currentBlockLocalValues_->count(id) != 0;
    }

    bool isGlobalIntReg(const std::string &reg) const {
        for (const auto &[_, globalReg] : globalValueRegs_) {
            if (globalReg == reg) {
                return true;
            }
        }
        return false;
    }

    bool isScalarIntReg(const std::string &reg) const {
        for (const auto &[_, scalarReg] : scalarIntAllocas_) {
            if (scalarReg == reg) {
                return true;
            }
        }
        return false;
    }

    bool isReservedIntReg(const std::string &reg) const {
        return isGlobalIntReg(reg) || isScalarIntReg(reg);
    }

    bool isGlobalFloatReg(const std::string &reg) const {
        for (const auto &[_, globalReg] : globalValueFRegs_) {
            if (globalReg == reg) {
                return true;
            }
        }
        return false;
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

    static const std::vector<std::string> &globalFloatRegs() {
        return savedFloatRegs();
    }

    static const std::vector<std::string> &savedIntCacheRegs() {
        static const std::vector<std::string> regs = {"s9", "s10", "s11"};
        return regs;
    }

    static const std::vector<std::string> &leafIntRegs() {
        static const std::vector<std::string> regs = {"a2", "a3", "a4", "a5", "a6", "a7", "t4", "t5"};
        return regs;
    }

    static bool isCalleeSavedIntReg(const std::string &reg) {
        return reg.size() >= 2 && reg[0] == 's' && reg != "s0";
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

    bool needsLongJumps(const ir::Function &function) const {
        std::size_t estimated = 0;
        for (const auto &block : function.blocks) {
            estimated += 1;
            for (const auto &inst : block.instructions) {
                switch (inst.opcode) {
                case ir::Opcode::Alloca:
                case ir::Opcode::Phi:
                    break;
                case ir::Opcode::Call:
                    estimated += 24 + inst.operands.size() * 5;
                    break;
                case ir::Opcode::CondBr:
                    estimated += 8;
                    break;
                case ir::Opcode::Br:
                case ir::Opcode::Ret:
                    estimated += 4;
                    break;
                default:
                    estimated += 8 + inst.operands.size() * 2;
                    break;
                }
            }
        }
        return estimated > 12000;
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
    std::unordered_map<int, std::string> transientRegs_;
    std::unordered_map<int, std::string> transientFRegs_;
    std::unordered_map<int, int> valueUseCounts_;
    std::vector<std::string> activeSavedIntRegs_;
    std::vector<std::string> activeSavedFloatRegs_;
    std::vector<std::string> currentGlobalIntRegs_;
    std::vector<std::string> currentIntCacheRegs_;
    std::unordered_map<std::string, std::unordered_set<int>> blockLocalValues_;
    std::unordered_map<std::string, std::vector<PhiCopy>> phiCopies_;
    const std::unordered_set<int> *currentBlockLocalValues_ = nullptr;
    std::string epilogueLabel_;
    std::size_t nextIntReg_ = 0;
    std::size_t nextFloatReg_ = 0;
    int nextPhiEdgeLabel_ = 0;
    bool useLongJumps_ = false;
    bool omitFrame_ = false;
    int elideResultStoreId_ = -1;
    int nextOffset_ = -24;
    int frameSize_ = 4096;
};

enum class MachineLineKind {
    Raw,
    Inst,
};

struct MachineLine {
    MachineLineKind kind = MachineLineKind::Raw;
    std::string raw;
    std::string op;
    std::vector<std::string> args;

    bool instruction() const {
        return kind == MachineLineKind::Inst;
    }

    std::string render() const {
        if (kind == MachineLineKind::Raw) {
            return raw;
        }
        std::string line = "\t" + op;
        if (!args.empty()) {
            line += ' ';
            for (std::size_t i = 0; i < args.size(); ++i) {
                if (i != 0) {
                    line += ", ";
                }
                line += args[i];
            }
        }
        return line;
    }
};

struct MachineAccess {
    std::unordered_set<std::string> uses;
    std::unordered_set<std::string> defs;
    bool barrier = false;
};

bool isLoadOp(const std::string &op);
bool isStoreOp(const std::string &op);
bool isAsmLabel(const MachineLine &line);
std::string asmLabelName(const MachineLine &line);

bool isIntegerRegisterName(const std::string &name) {
    if (name == "zero" || name == "ra" || name == "sp" || name == "gp" || name == "tp") {
        return true;
    }
    if (name.size() == 2 && (name[0] == 'a' || name[0] == 's' || name[0] == 't') && std::isdigit(static_cast<unsigned char>(name[1]))) {
        return true;
    }
    if (name.size() == 3 && name[0] == 's' && name[1] == '1' && (name[2] == '0' || name[2] == '1')) {
        return true;
    }
    if (name.size() == 3 && name[0] == 't' && name[1] == '6') {
        return true;
    }
    return false;
}

bool isFloatRegisterName(const std::string &name) {
    if (name.size() == 3 && name[0] == 'f' && (name[1] == 'a' || name[1] == 't') && std::isdigit(static_cast<unsigned char>(name[2]))) {
        return true;
    }
    if (name.size() == 3 && name[0] == 'f' && name[1] == 's' && std::isdigit(static_cast<unsigned char>(name[2]))) {
        return true;
    }
    if (name.size() == 4 && name[0] == 'f' && name[1] == 's' && name[2] == '1' && (name[3] == '0' || name[3] == '1')) {
        return true;
    }
    return false;
}

bool isRegisterName(const std::string &name) {
    return isIntegerRegisterName(name) || isFloatRegisterName(name);
}

std::string memoryBaseRegister(const std::string &operand) {
    const std::size_t open = operand.find('(');
    const std::size_t close = operand.find(')', open == std::string::npos ? 0 : open);
    if (open == std::string::npos || close == std::string::npos || close <= open + 1) {
        return {};
    }
    return operand.substr(open + 1, close - open - 1);
}

bool replaceMemoryBaseRegister(std::string &operand, const std::unordered_map<std::string, std::string> &copies) {
    const std::size_t open = operand.find('(');
    const std::size_t close = operand.find(')', open == std::string::npos ? 0 : open);
    if (open == std::string::npos || close == std::string::npos || close <= open + 1) {
        return false;
    }
    const std::string base = operand.substr(open + 1, close - open - 1);
    const auto replacement = copies.find(base);
    if (replacement == copies.end()) {
        return false;
    }
    operand = operand.substr(0, open + 1) + replacement->second + operand.substr(close);
    return true;
}

void addReg(std::unordered_set<std::string> &regs, const std::string &reg) {
    if (isRegisterName(reg) && reg != "zero") {
        regs.insert(reg);
    }
}

MachineAccess accessOf(const MachineLine &line) {
    MachineAccess access;
    if (!line.instruction()) {
        access.barrier = true;
        return access;
    }
    const std::string &op = line.op;
    const auto &a = line.args;
    if (op == "call" || op == "ret" || op == "jr" || op == "j" ||
        op == "beqz" || op == "bnez" || op == "beq" || op == "bne" || op == "blt" || op == "bge") {
        access.barrier = true;
    }
    if (op == "call") {
        for (int i = 0; i < 8; ++i) {
            addReg(access.uses, "a" + std::to_string(i));
            addReg(access.uses, "fa" + std::to_string(i));
        }
        return access;
    }
    if (op == "ret") {
        addReg(access.uses, "a0");
        return access;
    }
    if (op == "j") {
        if (!a.empty() && a[0].find(".ret") != std::string::npos) {
            addReg(access.uses, "a0");
            addReg(access.uses, "fa0");
        }
        return access;
    }
    if (op == "jr") {
        if (!a.empty()) addReg(access.uses, a[0]);
        return access;
    }
    if (op == "beqz" || op == "bnez") {
        if (!a.empty()) addReg(access.uses, a[0]);
        return access;
    }
    if (op == "beq" || op == "bne" || op == "blt" || op == "bge") {
        if (a.size() > 0) addReg(access.uses, a[0]);
        if (a.size() > 1) addReg(access.uses, a[1]);
        return access;
    }
    if (op == "li" || op == "la" || op == "lla") {
        if (!a.empty()) addReg(access.defs, a[0]);
        return access;
    }
    if (op == "mv" || op == "fmv.s" || op == "fmv.w.x") {
        if (a.size() > 0) addReg(access.defs, a[0]);
        if (a.size() > 1) addReg(access.uses, a[1]);
        return access;
    }
    if (isLoadOp(op)) {
        if (a.size() > 0) addReg(access.defs, a[0]);
        if (a.size() > 1) addReg(access.uses, memoryBaseRegister(a[1]));
        return access;
    }
    if (isStoreOp(op)) {
        if (a.size() > 0) addReg(access.uses, a[0]);
        if (a.size() > 1) addReg(access.uses, memoryBaseRegister(a[1]));
        return access;
    }
    if (op == "fcvt.s.w") {
        if (a.size() > 0) addReg(access.defs, a[0]);
        if (a.size() > 1) addReg(access.uses, a[1]);
        return access;
    }
    if (op == "fcvt.w.s") {
        if (a.size() > 0) addReg(access.defs, a[0]);
        if (a.size() > 1) addReg(access.uses, a[1]);
        return access;
    }
    if (!a.empty()) {
        addReg(access.defs, a[0]);
        for (std::size_t i = 1; i < a.size(); ++i) {
            addReg(access.uses, a[i]);
        }
    }
    return access;
}

std::string trimAsm(std::string text) {
    const std::size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const std::size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::vector<std::string> splitAsmArgs(const std::string &text) {
    std::vector<std::string> args;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::size_t end = comma == std::string::npos ? text.size() : comma;
        args.push_back(trimAsm(text.substr(start, end - start)));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return args;
}

MachineLine parseMachineLine(const std::string &line) {
    MachineLine parsed;
    parsed.raw = line;
    if (line.empty() || line[0] != '\t') {
        return parsed;
    }
    const std::string body = trimAsm(line);
    if (body.empty() || body[0] == '.') {
        return parsed;
    }
    const std::size_t space = body.find_first_of(" \t");
    parsed.kind = MachineLineKind::Inst;
    parsed.op = space == std::string::npos ? body : body.substr(0, space);
    if (space != std::string::npos) {
        parsed.args = splitAsmArgs(body.substr(space + 1));
    }
    return parsed;
}

MachineLine machineInst(const std::string &op, const std::vector<std::string> &args) {
    MachineLine line;
    line.kind = MachineLineKind::Inst;
    line.op = op;
    line.args = args;
    return line;
}

bool isLoadOp(const std::string &op) {
    return op == "ld" || op == "lw" || op == "flw" || op == "fld";
}

bool isStoreOp(const std::string &op) {
    return op == "sd" || op == "sw" || op == "fsw" || op == "fsd";
}

bool parseAsmImmediate(const std::string &text, long long &value) {
    char *end = nullptr;
    value = std::strtoll(text.c_str(), &end, 0);
    return end != text.c_str() && end != nullptr && *end == '\0';
}

bool fitsI12Asm(long long value) {
    return value >= -2048 && value <= 2047;
}

bool positivePowerOfTwo(long long value) {
    return value > 0 && (value & (value - 1)) == 0;
}

int log2Asm(long long value) {
    int shift = 0;
    while (value > 1) {
        value >>= 1;
        ++shift;
    }
    return shift;
}

bool matchingLoadStore(const std::string &load, const std::string &store) {
    return (load == "ld" && store == "sd") || (load == "lw" && store == "sw") ||
           (load == "flw" && store == "fsw") || (load == "fld" && store == "fsd");
}

std::vector<MachineLine> parseMachineAssembly(const std::string &assembly) {
    std::vector<MachineLine> lines;
    std::istringstream in(assembly);
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(parseMachineLine(line));
    }
    return lines;
}

void peepholeMachine(std::vector<MachineLine> &lines) {
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<MachineLine> next;
        next.reserve(lines.size());
        for (std::size_t i = 0; i < lines.size(); ++i) {
            const MachineLine &current = lines[i];
            if (current.instruction() && (current.op == "mv" || current.op == "fmv.s") &&
                current.args.size() == 2 && current.args[0] == current.args[1]) {
                changed = true;
                continue;
            }

            if (i + 1 < lines.size()) {
                const MachineLine &following = lines[i + 1];
                if (current.instruction() && following.instruction()) {
                    if (isStoreOp(current.op) && isLoadOp(following.op) && current.args.size() == 2 &&
                        following.args.size() == 2 && matchingLoadStore(following.op, current.op) &&
                        current.args[1] == following.args[1]) {
                        if (current.args[0] == following.args[0]) {
                            next.push_back(lines[i]);
                            ++i;
                            changed = true;
                            continue;
                        }
                        const bool floatMove = current.op == "fsw" || current.op == "fsd";
                        next.push_back(lines[i]);
                        next.push_back(machineInst(floatMove ? "fmv.s" : "mv", {following.args[0], current.args[0]}));
                        ++i;
                        changed = true;
                        continue;
                    }
                    if (isLoadOp(current.op) && isStoreOp(following.op) && current.args.size() == 2 &&
                        following.args.size() == 2 && matchingLoadStore(current.op, following.op) &&
                        current.args[0] == following.args[0] && current.args[1] == following.args[1]) {
                        next.push_back(lines[i]);
                        ++i;
                        changed = true;
                        continue;
                    }
                    if (current.op == "mv" && following.op == "mv" && current.args.size() == 2 &&
                        following.args.size() == 2 && current.args[0] == following.args[1]) {
                        next.push_back(lines[i]);
                        next.push_back(machineInst("mv", {following.args[0], current.args[1]}));
                        ++i;
                        changed = true;
                        continue;
                    }
                }
            }
            next.push_back(lines[i]);
        }
        lines = std::move(next);
    }
}

void foldImmediateProducer(std::vector<MachineLine> &lines) {
    std::vector<MachineLine> next;
    next.reserve(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (i + 1 >= lines.size() || !lines[i].instruction() || !lines[i + 1].instruction() ||
            lines[i].op != "li" || lines[i].args.size() != 2) {
            next.push_back(std::move(lines[i]));
            continue;
        }

        const std::string immReg = lines[i].args[0];
        long long imm = 0;
        if (!parseAsmImmediate(lines[i].args[1], imm)) {
            next.push_back(std::move(lines[i]));
            continue;
        }

        MachineLine replacement;
        bool folded = false;
        const MachineLine &use = lines[i + 1];
        if ((use.op == "addw" || use.op == "subw" || use.op == "slt" || use.op == "mulw") && use.args.size() == 3) {
            const std::string &dst = use.args[0];
            const std::string &lhs = use.args[1];
            const std::string &rhs = use.args[2];
            if (use.op == "addw" && fitsI12Asm(imm)) {
                if (rhs == immReg && lhs != immReg) {
                    replacement = machineInst("addiw", {dst, lhs, std::to_string(imm)});
                    folded = true;
                } else if (lhs == immReg && rhs != immReg) {
                    replacement = machineInst("addiw", {dst, rhs, std::to_string(imm)});
                    folded = true;
                }
            } else if (use.op == "subw" && rhs == immReg && lhs != immReg && fitsI12Asm(-imm)) {
                replacement = machineInst("addiw", {dst, lhs, std::to_string(-imm)});
                folded = true;
            } else if (use.op == "slt" && rhs == immReg && lhs != immReg && fitsI12Asm(imm)) {
                replacement = machineInst("slti", {dst, lhs, std::to_string(imm)});
                folded = true;
            } else if (use.op == "mulw" && rhs == immReg && lhs != immReg) {
                if (imm == 0) {
                    replacement = machineInst("li", {dst, "0"});
                    folded = true;
                } else if (imm == 1) {
                    replacement = machineInst("mv", {dst, lhs});
                    folded = true;
                } else if (positivePowerOfTwo(imm)) {
                    replacement = machineInst("slliw", {dst, lhs, std::to_string(log2Asm(imm))});
                    folded = true;
                }
            }
        }

        if (folded) {
            next.push_back(std::move(replacement));
            ++i;
        } else {
            next.push_back(std::move(lines[i]));
        }
    }
    lines = std::move(next);
}

bool isMoveInstruction(const MachineLine &line) {
    return line.instruction() && (line.op == "mv" || line.op == "fmv.s") && line.args.size() == 2;
}

bool copyPropagatableRegister(const std::string &reg) {
    return isRegisterName(reg) && reg != "zero" && reg != "sp" && reg != "ra" &&
           reg != "gp" && reg != "tp" && reg != "s0";
}

bool sameRegisterClassForCopy(const std::string &dst, const std::string &src, const std::string &op) {
    if (op == "mv") {
        return isIntegerRegisterName(dst) && isIntegerRegisterName(src);
    }
    if (op == "fmv.s") {
        return isFloatRegisterName(dst) && isFloatRegisterName(src);
    }
    return false;
}

void killCopyAliases(std::unordered_map<std::string, std::string> &copies, const std::unordered_set<std::string> &defs) {
    if (defs.empty() || copies.empty()) {
        return;
    }
    for (auto it = copies.begin(); it != copies.end();) {
        if (defs.count(it->first) || defs.count(it->second)) {
            it = copies.erase(it);
        } else {
            ++it;
        }
    }
}

bool rewriteRegisterUse(std::string &arg, const std::unordered_map<std::string, std::string> &copies) {
    const auto found = copies.find(arg);
    if (found == copies.end()) {
        return false;
    }
    arg = found->second;
    return true;
}

bool rewriteCopiedUses(MachineLine &line, const std::unordered_map<std::string, std::string> &copies) {
    if (!line.instruction() || copies.empty()) {
        return false;
    }
    bool changed = false;
    const std::string &op = line.op;
    auto rewriteArg = [&](std::size_t index) {
        if (index < line.args.size()) {
            changed = rewriteRegisterUse(line.args[index], copies) || changed;
        }
    };
    if (op == "call" || op == "ret") {
        return false;
    }
    if (op == "jr" || op == "beqz" || op == "bnez") {
        rewriteArg(0);
        return changed;
    }
    if (op == "beq" || op == "bne" || op == "blt" || op == "bge") {
        rewriteArg(0);
        rewriteArg(1);
        return changed;
    }
    if (isLoadOp(op)) {
        if (line.args.size() > 1) {
            changed = replaceMemoryBaseRegister(line.args[1], copies) || changed;
        }
        return changed;
    }
    if (isStoreOp(op)) {
        rewriteArg(0);
        if (line.args.size() > 1) {
            changed = replaceMemoryBaseRegister(line.args[1], copies) || changed;
        }
        return changed;
    }
    if (op == "li" || op == "la" || op == "lla") {
        return false;
    }
    if (line.args.size() > 1) {
        for (std::size_t i = 1; i < line.args.size(); ++i) {
            rewriteArg(i);
        }
    }
    return changed;
}

void propagateRegisterCopies(std::vector<MachineLine> &lines) {
    std::unordered_map<std::string, std::string> copies;
    auto reset = [&]() {
        copies.clear();
    };

    for (MachineLine &line : lines) {
        if (!line.instruction() || isAsmLabel(line)) {
            reset();
            continue;
        }

        MachineAccess before = accessOf(line);
        if (before.barrier) {
            rewriteCopiedUses(line, copies);
            reset();
            continue;
        }

        rewriteCopiedUses(line, copies);
        MachineAccess after = accessOf(line);
        killCopyAliases(copies, after.defs);

        if (isMoveInstruction(line)) {
            const std::string &dst = line.args[0];
            const std::string &src = line.args[1];
            if (dst != src && copyPropagatableRegister(dst) && copyPropagatableRegister(src) &&
                sameRegisterClassForCopy(dst, src, line.op)) {
                copies[dst] = src;
            }
        }
    }
}

bool isPureDefinitionInstruction(const MachineLine &line) {
    if (!line.instruction()) {
        return false;
    }
    const std::string &op = line.op;
    if (op == "call" || op == "ret" || op == "jr" || op == "j" ||
        op == "beqz" || op == "bnez" || op == "beq" || op == "bne" ||
        op == "blt" || op == "bge" || isStoreOp(op)) {
        return false;
    }
    MachineAccess access = accessOf(line);
    return !access.barrier && access.defs.size() == 1;
}

bool isCrossBlockRegister(const std::string &reg) {
    return reg == "sp" || reg == "ra" || reg == "s0" || reg == "a0" || reg == "fa0" ||
           (reg.size() >= 2 && reg[0] == 's') ||
           (reg.size() >= 3 && reg[0] == 'f' && reg[1] == 's');
}

void eliminateDeadMoves(std::vector<MachineLine> &lines) {
    std::vector<bool> remove(lines.size(), false);
    std::size_t blockBegin = 0;
    auto flushBlock = [&](std::size_t blockEnd) {
        std::unordered_set<std::string> live;
        for (std::size_t i = blockEnd; i-- > blockBegin;) {
            if (!lines[i].instruction()) {
                continue;
            }
            MachineAccess access = accessOf(lines[i]);
            if (access.barrier) {
                live.clear();
            }
            if (isMoveInstruction(lines[i])) {
                const std::string &dst = lines[i].args[0];
                if (!isCrossBlockRegister(dst) && !live.count(dst)) {
                    remove[i] = true;
                    continue;
                }
            }
            for (const std::string &def : access.defs) {
                live.erase(def);
            }
            for (const std::string &use : access.uses) {
                live.insert(use);
            }
        }
    };

    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (!lines[i].instruction() || (lines[i].instruction() && lines[i].op.back() == ':')) {
            flushBlock(i);
            blockBegin = i + 1;
        } else if (lines[i].instruction()) {
            MachineAccess access = accessOf(lines[i]);
            if (access.barrier) {
                flushBlock(i + 1);
                blockBegin = i + 1;
            }
        }
    }
    flushBlock(lines.size());

    std::vector<MachineLine> kept;
    kept.reserve(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (!remove[i]) {
            kept.push_back(std::move(lines[i]));
        }
    }
    lines = std::move(kept);
}

void eliminateOverwrittenDefs(std::vector<MachineLine> &lines) {
    std::vector<bool> remove(lines.size(), false);
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (!isPureDefinitionInstruction(lines[i])) {
            continue;
        }
        MachineAccess current = accessOf(lines[i]);
        const std::string reg = *current.defs.begin();
        for (std::size_t j = i + 1; j < lines.size(); ++j) {
            if (!lines[j].instruction()) {
                break;
            }
            MachineAccess next = accessOf(lines[j]);
            if (next.barrier || next.uses.count(reg)) {
                break;
            }
            if (next.defs.count(reg)) {
                remove[i] = true;
                break;
            }
        }
    }

    std::vector<MachineLine> kept;
    kept.reserve(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (!remove[i]) {
            kept.push_back(std::move(lines[i]));
        }
    }
    lines = std::move(kept);
}

struct AvailableLoad {
    std::string op;
    std::string reg;
    std::string base;
};

std::string loadAvailabilityKey(const MachineLine &line) {
    if (!line.instruction() || !isLoadOp(line.op) || line.args.size() != 2) {
        return {};
    }
    return line.op + "|" + line.args[1];
}

void killAvailableLoads(std::unordered_map<std::string, AvailableLoad> &loads,
                        const std::unordered_set<std::string> &defs) {
    if (defs.empty() || loads.empty()) {
        return;
    }
    for (auto it = loads.begin(); it != loads.end();) {
        bool killed = false;
        for (const std::string &def : defs) {
            if (def == it->second.reg || def == it->second.base) {
                killed = true;
                break;
            }
        }
        if (killed) {
            it = loads.erase(it);
        } else {
            ++it;
        }
    }
}

void eliminateRedundantLoads(std::vector<MachineLine> &lines) {
    std::unordered_map<std::string, AvailableLoad> loads;
    std::vector<bool> remove(lines.size(), false);

    for (std::size_t i = 0; i < lines.size(); ++i) {
        MachineLine &line = lines[i];
        if (!line.instruction() || isAsmLabel(line)) {
            loads.clear();
            continue;
        }

        MachineAccess access = accessOf(line);
        if (access.barrier || isStoreOp(line.op)) {
            loads.clear();
            continue;
        }

        if (isLoadOp(line.op) && line.args.size() == 2 && line.op != "fld") {
            const std::string key = loadAvailabilityKey(line);
            const auto available = loads.find(key);
            if (available != loads.end() && available->second.reg != line.args[0]) {
                line = machineInst(line.op == "flw" ? "fmv.s" : "mv", {line.args[0], available->second.reg});
                access = accessOf(line);
            } else if (available != loads.end()) {
                remove[i] = true;
                continue;
            } else {
                const std::string base = memoryBaseRegister(line.args[1]);
                if (!base.empty() && copyPropagatableRegister(line.args[0])) {
                    loads[key] = AvailableLoad{line.op, line.args[0], base};
                }
            }
        }

        killAvailableLoads(loads, access.defs);
    }

    std::vector<MachineLine> kept;
    kept.reserve(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (!remove[i]) {
            kept.push_back(std::move(lines[i]));
        }
    }
    lines = std::move(kept);
}

std::string registerValueProducerKey(const MachineLine &line) {
    if (!line.instruction() || line.args.size() < 2) {
        return {};
    }
    if (line.op == "li" || line.op == "la" || line.op == "lla") {
        return line.op + "|" + line.args[1];
    }
    return {};
}

void eliminateRedundantValueProducers(std::vector<MachineLine> &lines) {
    std::unordered_map<std::string, std::string> regValue;
    std::unordered_map<std::string, std::unordered_set<std::string>> valueRegs;
    std::vector<bool> remove(lines.size(), false);

    auto clearAll = [&]() {
        regValue.clear();
        valueRegs.clear();
    };
    auto killReg = [&](const std::string &reg) {
        const auto old = regValue.find(reg);
        if (old != regValue.end()) {
            auto regs = valueRegs.find(old->second);
            if (regs != valueRegs.end()) {
                regs->second.erase(reg);
                if (regs->second.empty()) {
                    valueRegs.erase(regs);
                }
            }
            regValue.erase(old);
        }
    };
    auto remember = [&](const std::string &reg, const std::string &value) {
        killReg(reg);
        regValue[reg] = value;
        valueRegs[value].insert(reg);
    };

    for (std::size_t i = 0; i < lines.size(); ++i) {
        MachineLine &line = lines[i];
        if (!line.instruction() || isAsmLabel(line)) {
            clearAll();
            continue;
        }
        MachineAccess access = accessOf(line);
        if (access.barrier) {
            clearAll();
            continue;
        }

        const std::string produced = registerValueProducerKey(line);
        if (!produced.empty() && !line.args.empty()) {
            const std::string dst = line.args[0];
            const auto current = regValue.find(dst);
            if (current != regValue.end() && current->second == produced) {
                remove[i] = true;
                continue;
            }
        }

        for (const std::string &def : access.defs) {
            killReg(def);
        }
        if (!produced.empty() && !line.args.empty() && copyPropagatableRegister(line.args[0])) {
            remember(line.args[0], produced);
        }
    }

    std::vector<MachineLine> kept;
    kept.reserve(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (!remove[i]) {
            kept.push_back(std::move(lines[i]));
        }
    }
    lines = std::move(kept);
}

bool isAsmLabel(const MachineLine &line) {
    if (line.instruction() || line.raw.empty() || line.raw[0] == '\t') {
        return false;
    }
    const std::string text = trimAsm(line.raw);
    return text.size() > 1 && text.back() == ':';
}

std::string asmLabelName(const MachineLine &line) {
    if (!isAsmLabel(line)) {
        return {};
    }
    std::string text = trimAsm(line.raw);
    text.pop_back();
    return text;
}

bool skippableBeforeLabel(const MachineLine &line) {
    if (line.instruction()) {
        return false;
    }
    const std::string text = trimAsm(line.raw);
    return text.empty() || text == ".align 2" || text == ".p2align 2" ||
           text.rfind(".align ", 0) == 0 || text.rfind(".p2align ", 0) == 0;
}

std::optional<std::string> nextFallthroughLabel(const std::vector<MachineLine> &lines, std::size_t from) {
    for (std::size_t i = from + 1; i < lines.size(); ++i) {
        if (isAsmLabel(lines[i])) {
            return asmLabelName(lines[i]);
        }
        if (!skippableBeforeLabel(lines[i])) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

void eliminateJumpsToFallthrough(std::vector<MachineLine> &lines) {
    std::vector<bool> remove(lines.size(), false);
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (!lines[i].instruction()) {
            continue;
        }
        const MachineLine &line = lines[i];
        if ((line.op == "j" || line.op == "beqz" || line.op == "bnez") && line.args.size() >= 1) {
            const std::string &target = line.args.back();
            const std::optional<std::string> fallthrough = nextFallthroughLabel(lines, i);
            if (fallthrough && *fallthrough == target) {
                remove[i] = true;
            }
            continue;
        }
        if (line.op == "lla" && line.args.size() == 2 && line.args[0] == "t6" &&
            i + 1 < lines.size() && lines[i + 1].instruction() &&
            lines[i + 1].op == "jr" && lines[i + 1].args.size() == 1 && lines[i + 1].args[0] == "t6") {
            const std::optional<std::string> fallthrough = nextFallthroughLabel(lines, i + 1);
            if (fallthrough && *fallthrough == line.args[1]) {
                remove[i] = true;
                remove[i + 1] = true;
            }
        }
    }

    std::vector<MachineLine> kept;
    kept.reserve(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (!remove[i]) {
            kept.push_back(std::move(lines[i]));
        }
    }
    lines = std::move(kept);
}

std::optional<std::string> oppositeZeroBranch(const std::string &op) {
    if (op == "beqz") {
        return "bnez";
    }
    if (op == "bnez") {
        return "beqz";
    }
    return std::nullopt;
}

void foldBranchOverJump(std::vector<MachineLine> &lines) {
    std::vector<bool> remove(lines.size(), false);
    for (std::size_t i = 0; i + 1 < lines.size(); ++i) {
        if (!lines[i].instruction() || !lines[i + 1].instruction()) {
            continue;
        }
        MachineLine &branch = lines[i];
        const MachineLine &jump = lines[i + 1];
        const std::optional<std::string> opposite = oppositeZeroBranch(branch.op);
        if (!opposite || branch.args.size() != 2 || jump.op != "j" || jump.args.size() != 1) {
            continue;
        }
        const std::optional<std::string> fallthrough = nextFallthroughLabel(lines, i + 1);
        if (!fallthrough || *fallthrough != branch.args[1]) {
            continue;
        }
        branch.op = *opposite;
        branch.args[1] = jump.args[0];
        remove[i + 1] = true;
    }

    std::vector<MachineLine> kept;
    kept.reserve(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (!remove[i]) {
            kept.push_back(std::move(lines[i]));
        }
    }
    lines = std::move(kept);
}

std::string renderMachineAssembly(const std::vector<MachineLine> &lines) {
    std::ostringstream out;
    for (const auto &outLine : lines) {
        out << outLine.render() << '\n';
    }
    return out.str();
}

std::string peepholeAssembly(const std::string &assembly) {
    std::vector<MachineLine> lines = parseMachineAssembly(assembly);
    foldImmediateProducer(lines);
    peepholeMachine(lines);
    foldBranchOverJump(lines);
    eliminateJumpsToFallthrough(lines);
    eliminateRedundantValueProducers(lines);
    eliminateRedundantLoads(lines);
    propagateRegisterCopies(lines);
    eliminateDeadMoves(lines);
    eliminateOverwrittenDefs(lines);
    foldImmediateProducer(lines);
    peepholeMachine(lines);
    foldBranchOverJump(lines);
    eliminateJumpsToFallthrough(lines);
    eliminateRedundantValueProducers(lines);
    eliminateRedundantLoads(lines);
    propagateRegisterCopies(lines);
    eliminateDeadMoves(lines);
    eliminateOverwrittenDefs(lines);
    return renderMachineAssembly(lines);
}

} // namespace

void emitAssembly(const TranslationUnit &unit, std::ostream &out) {
    CodeGen(unit, out).run();
}

void emitAssembly(const ir::Module &module, std::ostream &out) {
    std::ostringstream raw;
    IRCodeGen(module, raw).run();
    out << peepholeAssembly(raw.str());
}

} // namespace sysyc::riscv
