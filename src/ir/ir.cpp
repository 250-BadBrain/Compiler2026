#include "ir.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <ostream>
#include <sstream>

namespace sysyc::ir {

std::string typeName(Type type) {
    switch (type.kind) {
    case TypeKind::Void:
        return "void";
    case TypeKind::I32:
        return "i32";
    case TypeKind::F32:
        return "f32";
    case TypeKind::Ptr:
        return "ptr";
    }
    return "unknown";
}

std::string opcodeName(Opcode opcode) {
    switch (opcode) {
    case Opcode::Alloca:
        return "alloca";
    case Opcode::Load:
        return "load";
    case Opcode::Store:
        return "store";
    case Opcode::Gep:
        return "gep";
    case Opcode::Add:
        return "add";
    case Opcode::Sub:
        return "sub";
    case Opcode::Mul:
        return "mul";
    case Opcode::Div:
        return "div";
    case Opcode::Mod:
        return "mod";
    case Opcode::Neg:
        return "neg";
    case Opcode::Not:
        return "not";
    case Opcode::ICmp:
        return "icmp";
    case Opcode::FCmp:
        return "fcmp";
    case Opcode::Cast:
        return "cast";
    case Opcode::Phi:
        return "phi";
    case Opcode::Call:
        return "call";
    case Opcode::Br:
        return "br";
    case Opcode::CondBr:
        return "condbr";
    case Opcode::Ret:
        return "ret";
    }
    return "unknown";
}

static void dumpValue(const Value &value, std::ostream &out) {
    if (value.constant || value.id < 0) {
        out << value.name;
        return;
    }
    out << '%' << value.id;
}

void dumpModule(const Module &module, std::ostream &out) {
    for (const auto &global : module.globals) {
        out << (global.isConst ? "const " : "global ") << '@' << global.name << " : " << typeName(global.type);
        for (int dim : global.dimensions) {
            out << '[' << dim << ']';
        }
        if (!global.initValues.empty()) {
            out << " = {";
            for (std::size_t i = 0; i < global.initValues.size(); ++i) {
                if (i != 0) {
                    out << ", ";
                }
                out << global.initValues[i];
            }
            out << '}';
        }
        out << '\n';
    }

    for (const auto &function : module.functions) {
        out << "func @" << function.name << '(';
        for (std::size_t i = 0; i < function.params.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << typeName(function.params[i].type) << ' ' << function.params[i].name;
        }
        out << ") -> " << typeName(function.returnType) << " {\n";
        for (const auto &block : function.blocks) {
            out << block.name << ":\n";
            for (const auto &inst : block.instructions) {
                out << "  ";
                if (inst.result >= 0) {
                    out << '%' << inst.result << " = ";
                }
                out << opcodeName(inst.opcode);
                if (!inst.text.empty()) {
                    out << ' ' << inst.text;
                }
                for (const auto &operand : inst.operands) {
                    out << ' ';
                    dumpValue(operand, out);
                }
                out << '\n';
            }
        }
        out << "}\n";
    }
}

namespace {

bool isPure(Opcode opcode) {
    switch (opcode) {
    case Opcode::Load:
    case Opcode::Gep:
    case Opcode::Add:
    case Opcode::Sub:
    case Opcode::Mul:
    case Opcode::Div:
    case Opcode::Mod:
    case Opcode::Neg:
    case Opcode::Not:
    case Opcode::ICmp:
    case Opcode::FCmp:
    case Opcode::Cast:
    case Opcode::Phi:
        return true;
    case Opcode::Alloca:
    case Opcode::Store:
    case Opcode::Call:
    case Opcode::Br:
    case Opcode::CondBr:
    case Opcode::Ret:
        return false;
    }
    return false;
}

Value resolve(Value value, const std::unordered_map<int, Value> &replacements) {
    while (!value.constant && value.id >= 0) {
        const auto found = replacements.find(value.id);
        if (found == replacements.end()) {
            break;
        }
        value = found->second;
    }
    return value;
}

bool foldInteger(const Instruction &inst, Value &result) {
    if (inst.operands.empty()) {
        return false;
    }
    for (const auto &operand : inst.operands) {
        if (!operand.constant || operand.type.kind != TypeKind::I32) {
            return false;
        }
    }

    const auto intValue = [](const Value &value) {
        return std::strtoll(value.name.c_str(), nullptr, 0);
    };
    long long folded = 0;
    if (inst.opcode == Opcode::Neg) {
        folded = -intValue(inst.operands[0]);
    } else if (inst.opcode == Opcode::Not) {
        folded = intValue(inst.operands[0]) == 0;
    } else if (inst.operands.size() >= 2) {
        const long long lhs = intValue(inst.operands[0]);
        const long long rhs = intValue(inst.operands[1]);
        if (inst.opcode == Opcode::Add) {
            folded = lhs + rhs;
        } else if (inst.opcode == Opcode::Sub) {
            folded = lhs - rhs;
        } else if (inst.opcode == Opcode::Mul) {
            folded = lhs * rhs;
        } else if (inst.opcode == Opcode::Div) {
            if (rhs == 0) {
                return false;
            }
            folded = lhs / rhs;
        } else if (inst.opcode == Opcode::Mod) {
            if (rhs == 0) {
                return false;
            }
            folded = lhs % rhs;
        } else if (inst.opcode == Opcode::ICmp) {
            if (inst.text == "lt") folded = lhs < rhs;
            else if (inst.text == "gt") folded = lhs > rhs;
            else if (inst.text == "le") folded = lhs <= rhs;
            else if (inst.text == "ge") folded = lhs >= rhs;
            else if (inst.text == "eq") folded = lhs == rhs;
            else if (inst.text == "ne") folded = lhs != rhs;
            else return false;
        } else {
            return false;
        }
    } else {
        return false;
    }

    result = Value{-1, Type{TypeKind::I32}, std::to_string(folded), true};
    return true;
}

bool foldFloat(const Instruction &inst, Value &result) {
    if (inst.operands.empty()) {
        return false;
    }
    for (const auto &operand : inst.operands) {
        if (!operand.constant || operand.type.kind != TypeKind::F32) {
            return false;
        }
    }

    const auto floatValue = [](const Value &value) {
        return std::strtof(value.name.c_str(), nullptr);
    };
    const auto floatText = [](float value) {
        std::ostringstream out;
        out.precision(9);
        out << value;
        return out.str();
    };

    if (inst.opcode == Opcode::Neg) {
        result = Value{-1, Type{TypeKind::F32}, floatText(-floatValue(inst.operands[0])), true};
        return true;
    }
    if (inst.opcode == Opcode::Not) {
        result = Value{-1, Type{TypeKind::I32}, floatValue(inst.operands[0]) == 0.0f ? "1" : "0", true};
        return true;
    }
    if (inst.opcode == Opcode::Cast && inst.text == "f2i") {
        result = Value{-1, Type{TypeKind::I32}, std::to_string(static_cast<long long>(floatValue(inst.operands[0]))), true};
        return true;
    }
    if (inst.operands.size() < 2) {
        return false;
    }

    const float lhs = floatValue(inst.operands[0]);
    const float rhs = floatValue(inst.operands[1]);
    if (inst.opcode == Opcode::Add) {
        result = Value{-1, Type{TypeKind::F32}, floatText(lhs + rhs), true};
    } else if (inst.opcode == Opcode::Sub) {
        result = Value{-1, Type{TypeKind::F32}, floatText(lhs - rhs), true};
    } else if (inst.opcode == Opcode::Mul) {
        result = Value{-1, Type{TypeKind::F32}, floatText(lhs * rhs), true};
    } else if (inst.opcode == Opcode::Div) {
        if (rhs == 0.0f) {
            return false;
        }
        result = Value{-1, Type{TypeKind::F32}, floatText(lhs / rhs), true};
    } else if (inst.opcode == Opcode::FCmp) {
        bool folded = false;
        if (inst.text == "lt") folded = lhs < rhs;
        else if (inst.text == "gt") folded = lhs > rhs;
        else if (inst.text == "le") folded = lhs <= rhs;
        else if (inst.text == "ge") folded = lhs >= rhs;
        else if (inst.text == "eq") folded = lhs == rhs;
        else if (inst.text == "ne") folded = lhs != rhs;
        else return false;
        result = Value{-1, Type{TypeKind::I32}, folded ? "1" : "0", true};
    } else {
        return false;
    }
    return true;
}

bool foldCast(const Instruction &inst, Value &result) {
    if (inst.opcode != Opcode::Cast || inst.operands.size() != 1 || !inst.operands[0].constant) {
        return false;
    }
    if (inst.text == "i2f" && inst.operands[0].type.kind == TypeKind::I32) {
        std::ostringstream out;
        out.precision(9);
        out << static_cast<float>(std::strtoll(inst.operands[0].name.c_str(), nullptr, 0));
        result = Value{-1, Type{TypeKind::F32}, out.str(), true};
        return true;
    }
    return false;
}

bool simplifyAlgebra(const Instruction &inst, Value &result) {
    if (inst.result < 0) {
        return false;
    }
    const auto isI32 = [](const Value &value, long long expected) {
        return value.constant && value.type.kind == TypeKind::I32 &&
               std::strtoll(value.name.c_str(), nullptr, 0) == expected;
    };
    const auto isF32 = [](const Value &value, float expected) {
        return value.constant && value.type.kind == TypeKind::F32 &&
               std::strtof(value.name.c_str(), nullptr) == expected;
    };

    if (inst.operands.size() == 2) {
        const Value &lhs = inst.operands[0];
        const Value &rhs = inst.operands[1];
        if (inst.opcode == Opcode::Add) {
            if (isI32(rhs, 0) || isF32(rhs, 0.0f)) { result = lhs; return true; }
            if (isI32(lhs, 0) || isF32(lhs, 0.0f)) { result = rhs; return true; }
        } else if (inst.opcode == Opcode::Sub) {
            if (isI32(rhs, 0) || isF32(rhs, 0.0f)) { result = lhs; return true; }
            if (!lhs.constant && !rhs.constant && lhs.id == rhs.id) {
                result = Value{-1, inst.resultType, inst.resultType.kind == TypeKind::F32 ? "0.0" : "0", true};
                return true;
            }
        } else if (inst.opcode == Opcode::Mul) {
            if (isI32(rhs, 1) || isF32(rhs, 1.0f)) { result = lhs; return true; }
            if (isI32(lhs, 1) || isF32(lhs, 1.0f)) { result = rhs; return true; }
            if (isI32(rhs, 0) || isI32(lhs, 0) || isF32(rhs, 0.0f) || isF32(lhs, 0.0f)) {
                result = Value{-1, inst.resultType, inst.resultType.kind == TypeKind::F32 ? "0.0" : "0", true};
                return true;
            }
        } else if (inst.opcode == Opcode::Div) {
            if (isI32(rhs, 1) || isF32(rhs, 1.0f)) { result = lhs; return true; }
            if (isI32(lhs, 0) || isF32(lhs, 0.0f)) {
                result = Value{-1, inst.resultType, inst.resultType.kind == TypeKind::F32 ? "0.0" : "0", true};
                return true;
            }
        } else if (inst.opcode == Opcode::Mod) {
            if (isI32(rhs, 1) || isI32(lhs, 0)) {
                result = Value{-1, Type{TypeKind::I32}, "0", true};
                return true;
            }
        } else if (inst.opcode == Opcode::ICmp && !lhs.constant && !rhs.constant && lhs.id == rhs.id) {
            const bool folded = inst.text == "eq" || inst.text == "le" || inst.text == "ge";
            if (inst.text == "eq" || inst.text == "ne" || inst.text == "lt" || inst.text == "gt" || inst.text == "le" || inst.text == "ge") {
                result = Value{-1, Type{TypeKind::I32}, folded ? "1" : "0", true};
                return true;
            }
        }
    }
    return false;
}

std::string valueKey(const Value &value) {
    if (value.constant) {
        return "c:" + typeName(value.type) + ":" + value.name;
    }
    return "v:" + std::to_string(value.id);
}

std::string instKey(const Instruction &inst) {
    std::vector<std::string> operands;
    operands.reserve(inst.operands.size());
    for (const auto &operand : inst.operands) {
        operands.push_back(valueKey(operand));
    }
    const bool commutative = inst.opcode == Opcode::Add || inst.opcode == Opcode::Mul ||
                             (inst.opcode == Opcode::ICmp && (inst.text == "eq" || inst.text == "ne")) ||
                             (inst.opcode == Opcode::FCmp && (inst.text == "eq" || inst.text == "ne"));
    if (commutative) {
        std::sort(operands.begin(), operands.end());
    }
    std::string key = opcodeName(inst.opcode) + ":" + typeName(inst.resultType) + ":" + inst.text;
    for (const auto &operand : operands) {
        key += "|" + operand;
    }
    return key;
}

int allocaBytes(const std::string &text) {
    const std::size_t colon = text.find(':');
    if (colon == std::string::npos) {
        return 8;
    }
    return static_cast<int>(std::strtol(text.c_str() + colon + 1, nullptr, 10));
}

Value zeroValue(Type type) {
    if (type.kind == TypeKind::F32) {
        return Value{-1, type, "0.0", true};
    }
    return Value{-1, type, "0", true};
}

bool promoteSingleBlockAllocas(Function &function) {
    struct Candidate {
        int block = -1;
        Type type;
        bool escaped = false;
        bool seenUse = false;
    };

    std::unordered_map<int, Candidate> candidates;
    for (std::size_t blockIndex = 0; blockIndex < function.blocks.size(); ++blockIndex) {
        const auto &block = function.blocks[blockIndex];
        for (const auto &inst : block.instructions) {
            if (inst.opcode == Opcode::Alloca && inst.result >= 0 && allocaBytes(inst.text) == 4) {
                candidates[inst.result] = Candidate{static_cast<int>(blockIndex), Type{TypeKind::I32}, false, false};
            }
        }
    }

    for (std::size_t blockIndex = 0; blockIndex < function.blocks.size(); ++blockIndex) {
        const auto &block = function.blocks[blockIndex];
        for (const auto &inst : block.instructions) {
            for (std::size_t operandIndex = 0; operandIndex < inst.operands.size(); ++operandIndex) {
                const Value &operand = inst.operands[operandIndex];
                if (operand.constant || operand.id < 0) {
                    continue;
                }
                auto found = candidates.find(operand.id);
                if (found == candidates.end()) {
                    continue;
                }
                const bool loadAddress = inst.opcode == Opcode::Load && operandIndex == 0;
                const bool storeAddress = inst.opcode == Opcode::Store && operandIndex == 1;
                if (!loadAddress && !storeAddress) {
                    found->second.escaped = true;
                    continue;
                }
                if (found->second.block != static_cast<int>(blockIndex)) {
                    found->second.escaped = true;
                    continue;
                }
                found->second.seenUse = true;
                found->second.type = loadAddress ? inst.resultType : inst.operands[0].type;
            }
        }
    }

    std::unordered_set<int> promotable;
    for (const auto &[id, candidate] : candidates) {
        if (!candidate.escaped && candidate.seenUse) {
            promotable.insert(id);
        }
    }
    if (promotable.empty()) {
        return false;
    }

    bool changed = false;
    std::unordered_map<int, Value> replacements;
    for (auto &block : function.blocks) {
        std::unordered_map<int, Value> current;
        std::vector<Instruction> kept;
        kept.reserve(block.instructions.size());

        for (auto inst : block.instructions) {
            for (auto &operand : inst.operands) {
                operand = resolve(operand, replacements);
            }

            if (inst.opcode == Opcode::Alloca && promotable.count(inst.result)) {
                changed = true;
                continue;
            }

            if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                const Value &address = inst.operands[1];
                if (!address.constant && promotable.count(address.id)) {
                    current[address.id] = inst.operands[0];
                    changed = true;
                    continue;
                }
            }

            if (inst.opcode == Opcode::Load && inst.result >= 0 && inst.operands.size() == 1) {
                const Value &address = inst.operands[0];
                if (!address.constant && promotable.count(address.id)) {
                    const auto found = current.find(address.id);
                    replacements[inst.result] = found == current.end() ? zeroValue(inst.resultType) : found->second;
                    changed = true;
                    continue;
                }
            }

            kept.push_back(std::move(inst));
        }
        block.instructions = std::move(kept);
    }

    if (changed) {
        for (auto &block : function.blocks) {
            for (auto &inst : block.instructions) {
                for (auto &operand : inst.operands) {
                    operand = resolve(operand, replacements);
                }
            }
        }
    }
    return changed;
}

bool foldConstants(Function &function) {
    bool changed = false;
    std::unordered_map<int, Value> replacements;
    for (auto &block : function.blocks) {
        std::vector<Instruction> kept;
        kept.reserve(block.instructions.size());
        for (auto inst : block.instructions) {
            for (auto &operand : inst.operands) {
                operand = resolve(operand, replacements);
            }
            if (inst.result >= 0) {
                Value folded;
                if (foldInteger(inst, folded) || foldFloat(inst, folded) ||
                    foldCast(inst, folded) || simplifyAlgebra(inst, folded)) {
                    replacements[inst.result] = folded;
                    changed = true;
                    continue;
                }
            }
            kept.push_back(std::move(inst));
        }
        block.instructions = std::move(kept);
    }
    return changed;
}

bool eliminateCommonSubexpressions(Function &function) {
    bool changed = false;
    std::unordered_map<int, Value> replacements;

    for (auto &block : function.blocks) {
        std::unordered_map<std::string, Value> available;
        std::vector<Instruction> kept;
        kept.reserve(block.instructions.size());
        for (auto inst : block.instructions) {
            for (auto &operand : inst.operands) {
                operand = resolve(operand, replacements);
            }

            const bool eligible = inst.result >= 0 &&
                                  (inst.opcode == Opcode::Gep || inst.opcode == Opcode::Add ||
                                   inst.opcode == Opcode::Sub || inst.opcode == Opcode::Mul ||
                                   inst.opcode == Opcode::Div || inst.opcode == Opcode::Mod ||
                                   inst.opcode == Opcode::Neg || inst.opcode == Opcode::Not ||
                                   inst.opcode == Opcode::ICmp || inst.opcode == Opcode::FCmp ||
                                   inst.opcode == Opcode::Cast);
            if (eligible) {
                const std::string key = instKey(inst);
                const auto found = available.find(key);
                if (found != available.end()) {
                    replacements[inst.result] = found->second;
                    changed = true;
                    continue;
                }
                available[key] = Value{inst.result, inst.resultType, {}, false};
            }

            if (inst.opcode == Opcode::Call || inst.opcode == Opcode::Store) {
                available.clear();
            }
            kept.push_back(std::move(inst));
        }
        block.instructions = std::move(kept);
    }

    if (changed) {
        for (auto &block : function.blocks) {
            for (auto &inst : block.instructions) {
                for (auto &operand : inst.operands) {
                    operand = resolve(operand, replacements);
                }
            }
        }
    }
    return changed;
}

bool simplifyBranches(Function &function) {
    bool changed = false;
    for (auto &block : function.blocks) {
        if (block.instructions.empty()) {
            continue;
        }
        Instruction &inst = block.instructions.back();
        if (inst.opcode != Opcode::CondBr || inst.operands.empty() || !inst.operands[0].constant) {
            continue;
        }
        const bool takeTrue = std::strtoll(inst.operands[0].name.c_str(), nullptr, 0) != 0;
        const std::size_t comma = inst.text.find(',');
        if (comma == std::string::npos) {
            continue;
        }
        const std::string trueLabel = inst.text.substr(0, comma);
        const std::string falseLabel = inst.text.substr(comma + 2);
        inst.opcode = Opcode::Br;
        inst.operands.clear();
        inst.text = takeTrue ? trueLabel : falseLabel;
        inst.result = -1;
        inst.resultType = Type{TypeKind::Void};
        changed = true;
    }
    return changed;
}

bool removeUnreachableBlocks(Function &function) {
    if (function.blocks.empty()) {
        return false;
    }
    std::unordered_map<std::string, std::size_t> index;
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        index[function.blocks[i].name] = i;
    }

    std::vector<std::size_t> stack{0};
    std::unordered_set<std::size_t> reachable;
    while (!stack.empty()) {
        const std::size_t current = stack.back();
        stack.pop_back();
        if (!reachable.insert(current).second || function.blocks[current].instructions.empty()) {
            continue;
        }
        const Instruction &term = function.blocks[current].instructions.back();
        if (term.opcode == Opcode::Br) {
            const auto found = index.find(term.text);
            if (found != index.end()) {
                stack.push_back(found->second);
            }
        } else if (term.opcode == Opcode::CondBr) {
            const std::size_t comma = term.text.find(',');
            if (comma != std::string::npos) {
                const auto t = index.find(term.text.substr(0, comma));
                const auto f = index.find(term.text.substr(comma + 2));
                if (t != index.end()) stack.push_back(t->second);
                if (f != index.end()) stack.push_back(f->second);
            }
        }
    }

    if (reachable.size() == function.blocks.size()) {
        return false;
    }
    std::vector<BasicBlock> kept;
    kept.reserve(reachable.size());
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        if (reachable.count(i)) {
            kept.push_back(std::move(function.blocks[i]));
        }
    }
    function.blocks = std::move(kept);
    return true;
}

bool eliminateDeadCode(Function &function) {
    std::unordered_set<int> used;
    for (const auto &block : function.blocks) {
        for (const auto &inst : block.instructions) {
            for (const auto &operand : inst.operands) {
                if (!operand.constant && operand.id >= 0) {
                    used.insert(operand.id);
                }
            }
        }
    }

    bool changed = false;
    for (auto &block : function.blocks) {
        std::vector<Instruction> kept;
        kept.reserve(block.instructions.size());
        for (auto &inst : block.instructions) {
            if (inst.result >= 0 && isPure(inst.opcode) && !used.count(inst.result)) {
                changed = true;
                continue;
            }
            kept.push_back(std::move(inst));
        }
        block.instructions = std::move(kept);
    }
    return changed;
}

bool forwardLocalMemory(Function &function) {
    bool changed = false;
    std::unordered_map<int, Value> replacements;

    for (auto &block : function.blocks) {
        std::unordered_map<int, Value> knownMemory;
        std::vector<Instruction> kept;
        kept.reserve(block.instructions.size());

        for (auto inst : block.instructions) {
            for (auto &operand : inst.operands) {
                operand = resolve(operand, replacements);
            }

            if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                const Value &address = inst.operands[1];
                if (!address.constant && address.id >= 0) {
                    knownMemory[address.id] = inst.operands[0];
                }
                kept.push_back(std::move(inst));
                continue;
            }

            if (inst.opcode == Opcode::Load && inst.result >= 0 && inst.operands.size() == 1) {
                const Value &address = inst.operands[0];
                if (!address.constant && address.id >= 0) {
                    const auto found = knownMemory.find(address.id);
                    if (found != knownMemory.end()) {
                        replacements[inst.result] = found->second;
                        changed = true;
                        continue;
                    }
                }
            }

            if (inst.opcode == Opcode::Call) {
                knownMemory.clear();
            }
            kept.push_back(std::move(inst));
        }

        block.instructions = std::move(kept);
    }

    if (changed) {
        for (auto &block : function.blocks) {
            for (auto &inst : block.instructions) {
                for (auto &operand : inst.operands) {
                    operand = resolve(operand, replacements);
                }
            }
        }
    }
    return changed;
}

bool eliminateDeadAllocas(Function &function) {
    std::unordered_set<int> allocas;
    std::unordered_set<int> escaped;
    std::unordered_set<int> loaded;

    for (const auto &block : function.blocks) {
        for (const auto &inst : block.instructions) {
            if (inst.opcode == Opcode::Alloca && inst.result >= 0) {
                allocas.insert(inst.result);
            }
        }
    }

    for (const auto &block : function.blocks) {
        for (const auto &inst : block.instructions) {
            for (std::size_t i = 0; i < inst.operands.size(); ++i) {
                const Value &operand = inst.operands[i];
                if (operand.constant || operand.id < 0 || !allocas.count(operand.id)) {
                    continue;
                }
                const bool addressOperand = (inst.opcode == Opcode::Load && i == 0) ||
                                            (inst.opcode == Opcode::Store && i == 1);
                if (inst.opcode == Opcode::Load && i == 0) {
                    loaded.insert(operand.id);
                }
                if (!addressOperand) {
                    escaped.insert(operand.id);
                }
            }
        }
    }

    std::unordered_set<int> removable;
    for (int id : allocas) {
        if (!escaped.count(id) && !loaded.count(id)) {
            removable.insert(id);
        }
    }

    if (removable.empty()) {
        return false;
    }

    bool changed = false;
    for (auto &block : function.blocks) {
        std::vector<Instruction> kept;
        kept.reserve(block.instructions.size());
        for (auto &inst : block.instructions) {
            if (inst.opcode == Opcode::Alloca && removable.count(inst.result)) {
                changed = true;
                continue;
            }
            if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                const Value &address = inst.operands[1];
                if (!address.constant && removable.count(address.id)) {
                    changed = true;
                    continue;
                }
            }
            kept.push_back(std::move(inst));
        }
        block.instructions = std::move(kept);
    }
    return changed;
}

} // namespace

bool optimize(Module &module) {
    bool changed = false;
    bool again = true;
    while (again) {
        again = false;
        for (auto &function : module.functions) {
            if (promoteSingleBlockAllocas(function)) {
                changed = true;
                again = true;
            }
            if (forwardLocalMemory(function)) {
                changed = true;
                again = true;
            }
            if (foldConstants(function)) {
                changed = true;
                again = true;
            }
            if (eliminateCommonSubexpressions(function)) {
                changed = true;
                again = true;
            }
            if (simplifyBranches(function)) {
                changed = true;
                again = true;
            }
            if (removeUnreachableBlocks(function)) {
                changed = true;
                again = true;
            }
            if (eliminateDeadCode(function)) {
                changed = true;
                again = true;
            }
            if (eliminateDeadAllocas(function)) {
                changed = true;
                again = true;
            }
        }
    }
    return changed;
}

} // namespace sysyc::ir
