#include "ir.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
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
    case Opcode::Add:
    case Opcode::Sub:
    case Opcode::Mul:
    case Opcode::Div:
    case Opcode::Mod:
    case Opcode::Neg:
    case Opcode::Not:
    case Opcode::ICmp:
    case Opcode::FCmp:
    case Opcode::Gep:
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

bool isTerminator(Opcode opcode) {
    return opcode == Opcode::Br || opcode == Opcode::CondBr || opcode == Opcode::Ret;
}

bool truncateAfterTerminators(Function &function) {
    bool changed = false;
    for (auto &block : function.blocks) {
        for (std::size_t i = 0; i < block.instructions.size(); ++i) {
            if (!isTerminator(block.instructions[i].opcode)) {
                continue;
            }
            if (i + 1 < block.instructions.size()) {
                block.instructions.resize(i + 1);
                changed = true;
            }
            break;
        }
    }
    return changed;
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

std::vector<std::vector<int>> computePredecessors(const Function &function) {
    std::unordered_map<std::string, int> blockIndex;
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        blockIndex[function.blocks[i].name] = static_cast<int>(i);
    }
    std::vector<std::vector<int>> predecessors(function.blocks.size());
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        if (function.blocks[i].instructions.empty()) {
            continue;
        }
        const Instruction &term = function.blocks[i].instructions.back();
        auto add = [&](const std::string &name) {
            const auto found = blockIndex.find(name);
            if (found != blockIndex.end()) {
                predecessors[static_cast<std::size_t>(found->second)].push_back(static_cast<int>(i));
            }
        };
        if (term.opcode == Opcode::Br) {
            add(term.text);
        } else if (term.opcode == Opcode::CondBr) {
            const std::size_t comma = term.text.find(',');
            if (comma != std::string::npos) {
                add(term.text.substr(0, comma));
                add(term.text.substr(comma + 2));
            }
        }
    }
    return predecessors;
}

std::vector<std::unordered_set<int>> computeDominators(const Function &function,
                                                       const std::vector<std::vector<int>> &predecessors) {
    std::unordered_set<int> allBlocks;
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        allBlocks.insert(static_cast<int>(i));
    }

    std::vector<std::unordered_set<int>> dominators(function.blocks.size());
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        dominators[i] = i == 0 ? std::unordered_set<int>{0} : allBlocks;
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t blockIndex = 1; blockIndex < function.blocks.size(); ++blockIndex) {
            std::unordered_set<int> next = allBlocks;
            if (predecessors[blockIndex].empty()) {
                next.clear();
            }
            for (int pred : predecessors[blockIndex]) {
                std::unordered_set<int> intersection;
                for (int candidate : next) {
                    if (dominators[static_cast<std::size_t>(pred)].count(candidate)) {
                        intersection.insert(candidate);
                    }
                }
                next = std::move(intersection);
            }
            next.insert(static_cast<int>(blockIndex));
            if (next != dominators[blockIndex]) {
                dominators[blockIndex] = std::move(next);
                changed = true;
            }
        }
    }
    return dominators;
}

std::vector<std::vector<int>> computeDominatorTree(const Function &function,
                                                   const std::vector<std::unordered_set<int>> &dominators) {
    std::vector<std::vector<int>> children(function.blocks.size());
    for (std::size_t blockIndex = 1; blockIndex < function.blocks.size(); ++blockIndex) {
        int idom = -1;
        for (int candidate : dominators[blockIndex]) {
            if (candidate == static_cast<int>(blockIndex)) {
                continue;
            }
            bool deepest = true;
            for (int other : dominators[blockIndex]) {
                if (other == candidate || other == static_cast<int>(blockIndex)) {
                    continue;
                }
                if (!dominators[static_cast<std::size_t>(candidate)].count(other)) {
                    deepest = false;
                    break;
                }
            }
            if (deepest) {
                idom = candidate;
                break;
            }
        }
        if (idom >= 0) {
            children[static_cast<std::size_t>(idom)].push_back(static_cast<int>(blockIndex));
        }
    }
    return children;
}

std::vector<int> estimateBlockValuePressure(const Function &function,
                                            const std::vector<std::vector<int>> &predecessors) {
    std::vector<std::vector<int>> successors(function.blocks.size());
    for (std::size_t blockIndex = 0; blockIndex < predecessors.size(); ++blockIndex) {
        for (int pred : predecessors[blockIndex]) {
            successors[static_cast<std::size_t>(pred)].push_back(static_cast<int>(blockIndex));
        }
    }

    std::vector<std::unordered_set<int>> use(function.blocks.size());
    std::vector<std::unordered_set<int>> def(function.blocks.size());
    for (std::size_t blockIndex = 0; blockIndex < function.blocks.size(); ++blockIndex) {
        std::unordered_set<int> seenDef;
        for (const auto &inst : function.blocks[blockIndex].instructions) {
            if (inst.result >= 0) {
                def[blockIndex].insert(inst.result);
                seenDef.insert(inst.result);
            }
            if (inst.opcode == Opcode::Phi) {
                continue;
            }
            for (const auto &operand : inst.operands) {
                if (!operand.constant && operand.id >= 0 && !seenDef.count(operand.id)) {
                    use[blockIndex].insert(operand.id);
                }
            }
        }
    }

    std::vector<std::unordered_set<int>> liveIn(function.blocks.size());
    std::vector<std::unordered_set<int>> liveOut(function.blocks.size());
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t blockIndex = function.blocks.size(); blockIndex-- > 0;) {
            std::unordered_set<int> nextOut;
            for (int succ : successors[blockIndex]) {
                nextOut.insert(liveIn[static_cast<std::size_t>(succ)].begin(),
                               liveIn[static_cast<std::size_t>(succ)].end());
            }
            std::unordered_set<int> nextIn = use[blockIndex];
            for (int id : nextOut) {
                if (!def[blockIndex].count(id)) {
                    nextIn.insert(id);
                }
            }
            if (nextOut != liveOut[blockIndex]) {
                liveOut[blockIndex] = std::move(nextOut);
                changed = true;
            }
            if (nextIn != liveIn[blockIndex]) {
                liveIn[blockIndex] = std::move(nextIn);
                changed = true;
            }
        }
    }

    std::vector<int> pressure(function.blocks.size(), 0);
    for (std::size_t blockIndex = 0; blockIndex < function.blocks.size(); ++blockIndex) {
        std::unordered_set<int> combined = liveIn[blockIndex];
        combined.insert(liveOut[blockIndex].begin(), liveOut[blockIndex].end());
        pressure[blockIndex] = static_cast<int>(combined.size());
    }
    return pressure;
}

bool sameValue(const Value &lhs, const Value &rhs) {
    return lhs.id == rhs.id && lhs.type.kind == rhs.type.kind &&
           lhs.name == rhs.name && lhs.constant == rhs.constant;
}

bool sameMap(const std::unordered_map<int, Value> &lhs, const std::unordered_map<int, Value> &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (const auto &[key, value] : lhs) {
        const auto found = rhs.find(key);
        if (found == rhs.end() || !sameValue(value, found->second)) {
            return false;
        }
    }
    return true;
}

bool sameStringValueMap(const std::unordered_map<std::string, Value> &lhs,
                        const std::unordered_map<std::string, Value> &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (const auto &[key, value] : lhs) {
        const auto found = rhs.find(key);
        if (found == rhs.end() || !sameValue(value, found->second)) {
            return false;
        }
    }
    return true;
}

std::unordered_set<int> promotableScalarAllocas(const Function &function) {
    std::unordered_set<int> allocas;
    std::unordered_set<int> escaped;
    for (const auto &block : function.blocks) {
        for (const auto &inst : block.instructions) {
            if (inst.opcode == Opcode::Alloca && inst.result >= 0 && allocaBytes(inst.text) == 4) {
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
                const bool addressUse = (inst.opcode == Opcode::Load && i == 0) ||
                                        (inst.opcode == Opcode::Store && i == 1);
                if (!addressUse) {
                    escaped.insert(operand.id);
                }
            }
        }
    }
    for (int id : escaped) {
        allocas.erase(id);
    }
    return allocas;
}

int nextValueId(const Function &function) {
    int next = 0;
    for (const auto &param : function.params) {
        next = std::max(next, param.id + 1);
    }
    for (const auto &block : function.blocks) {
        for (const auto &inst : block.instructions) {
            next = std::max(next, inst.result + 1);
            for (const auto &operand : inst.operands) {
                if (!operand.constant) {
                    next = std::max(next, operand.id + 1);
                }
            }
        }
    }
    return next;
}

bool hugeFunction(const Function &function) {
    std::size_t instructions = 0;
    for (const auto &block : function.blocks) {
        instructions += block.instructions.size();
    }
    return function.blocks.size() > 1000 || instructions > 30000;
}

std::size_t instructionCount(const Function &function) {
    std::size_t count = 0;
    for (const auto &block : function.blocks) {
        count += block.instructions.size();
    }
    return count;
}

bool canInlineSingleBlock(const Function &caller, const Function &callee) {
    if (caller.name == callee.name || callee.blocks.size() != 1) {
        return false;
    }
    const std::size_t size = instructionCount(callee);
    if (size == 0 || size > 24) {
        return false;
    }
    const auto &body = callee.blocks.front().instructions;
    if (body.empty() || body.back().opcode != Opcode::Ret) {
        return false;
    }
    for (std::size_t i = 0; i + 1 < body.size(); ++i) {
        if (isTerminator(body[i].opcode)) {
            return false;
        }
        if (body[i].opcode == Opcode::Call) {
            return false;
        }
        if (body[i].opcode == Opcode::Load && !body[i].operands.empty() && body[i].operands[0].constant) {
            return false;
        }
        if (body[i].opcode == Opcode::Store && body[i].operands.size() == 2 && body[i].operands[1].constant) {
            return false;
        }
        if (body[i].opcode == Opcode::Alloca && allocaBytes(body[i].text) != 4) {
            return false;
        }
    }
    return true;
}

Value remapValue(Value value, const std::unordered_map<int, Value> &valueMap) {
    if (value.constant || value.id < 0) {
        return value;
    }
    const auto found = valueMap.find(value.id);
    return found == valueMap.end() ? value : found->second;
}

bool inlineSmallFunctions(Module &module) {
    std::unordered_map<std::string, const Function *> functions;
    functions.reserve(module.functions.size());
    for (const auto &function : module.functions) {
        functions[function.name] = &function;
    }

    bool changed = false;
    for (auto &caller : module.functions) {
        if (hugeFunction(caller)) {
            continue;
        }
        int nextId = nextValueId(caller);
        std::unordered_map<int, Value> replacements;

        for (auto &block : caller.blocks) {
            std::vector<Instruction> kept;
            kept.reserve(block.instructions.size());

            for (auto inst : block.instructions) {
                for (auto &operand : inst.operands) {
                    operand = resolve(operand, replacements);
                }

                const auto found = functions.find(inst.text);
                if (inst.opcode != Opcode::Call || found == functions.end() ||
                    !canInlineSingleBlock(caller, *found->second)) {
                    kept.push_back(std::move(inst));
                    continue;
                }

                const Function &callee = *found->second;
                std::unordered_map<int, Value> valueMap;
                for (std::size_t i = 0; i < callee.params.size(); ++i) {
                    if (i < inst.operands.size()) {
                        valueMap[callee.params[i].id] = inst.operands[i];
                    }
                }

                Value returnValue;
                bool hasReturnValue = false;
                const auto &body = callee.blocks.front().instructions;
                for (std::size_t i = 0; i < body.size(); ++i) {
                    Instruction cloned = body[i];
                    if (cloned.opcode == Opcode::Ret) {
                        if (!cloned.operands.empty()) {
                            returnValue = remapValue(resolve(cloned.operands[0], replacements), valueMap);
                            hasReturnValue = true;
                        }
                        break;
                    }

                    for (auto &operand : cloned.operands) {
                        operand = remapValue(operand, valueMap);
                    }
                    if (cloned.result >= 0) {
                        const int oldResult = cloned.result;
                        cloned.result = nextId++;
                        valueMap[oldResult] = Value{cloned.result, cloned.resultType, {}, false};
                    }
                    kept.push_back(std::move(cloned));
                }

                if (inst.result >= 0 && hasReturnValue) {
                    replacements[inst.result] = returnValue;
                }
                changed = true;
            }

            block.instructions = std::move(kept);
        }

        if (!replacements.empty()) {
            for (auto &block : caller.blocks) {
                for (auto &inst : block.instructions) {
                    for (auto &operand : inst.operands) {
                        operand = resolve(operand, replacements);
                    }
                }
            }
        }
    }

    return changed;
}

std::unordered_set<std::string> pureFunctionNames(const Module &module) {
    std::unordered_set<std::string> candidates;
    std::unordered_map<std::string, std::vector<std::string>> callees;
    for (const auto &function : module.functions) {
        if (function.returnType.kind == TypeKind::Void) {
            continue;
        }
        bool ok = true;
        for (const auto &param : function.params) {
            if (param.type.kind == TypeKind::Ptr) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == Opcode::Store || inst.opcode == Opcode::Load) {
                    ok = false;
                    break;
                }
                if (inst.opcode == Opcode::Call) {
                    callees[function.name].push_back(inst.text);
                }
            }
            if (!ok) {
                break;
            }
        }
        if (ok) {
            candidates.insert(function.name);
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (auto it = candidates.begin(); it != candidates.end();) {
            bool ok = true;
            for (const auto &callee : callees[*it]) {
                if (!candidates.count(callee)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                it = candidates.erase(it);
                changed = true;
            } else {
                ++it;
            }
        }
    }
    return candidates;
}

bool eliminatePureCallCommonSubexpressions(Module &module) {
    const std::unordered_set<std::string> pure = pureFunctionNames(module);
    if (pure.empty()) {
        return false;
    }

    bool changed = false;
    for (auto &function : module.functions) {
        std::unordered_map<int, Value> replacements;

        for (auto &block : function.blocks) {
            std::unordered_map<std::string, Value> available;
            std::vector<Instruction> kept;
            kept.reserve(block.instructions.size());

            for (auto inst : block.instructions) {
                for (auto &operand : inst.operands) {
                    operand = resolve(operand, replacements);
                }

                if (inst.opcode == Opcode::Call && inst.result >= 0 && pure.count(inst.text)) {
                    const std::string key = instKey(inst);
                    const auto found = available.find(key);
                    if (found != available.end()) {
                        replacements[inst.result] = found->second;
                        changed = true;
                        continue;
                    }
                    available[key] = Value{inst.result, inst.resultType, {}, false};
                    kept.push_back(std::move(inst));
                    continue;
                }

                if (inst.opcode == Opcode::Call) {
                    available.clear();
                }
                kept.push_back(std::move(inst));
            }

            block.instructions = std::move(kept);
        }

        if (!replacements.empty()) {
            for (auto &block : function.blocks) {
                for (auto &inst : block.instructions) {
                    for (auto &operand : inst.operands) {
                        operand = resolve(operand, replacements);
                    }
                }
            }
        }
    }

    return changed;
}

bool removeUnreachableFunctions(Module &module) {
    std::unordered_set<std::string> defined;
    defined.reserve(module.functions.size());
    for (const auto &function : module.functions) {
        defined.insert(function.name);
    }
    if (!defined.count("main")) {
        return false;
    }

    std::unordered_map<std::string, std::vector<std::string>> calls;
    calls.reserve(module.functions.size());
    for (const auto &function : module.functions) {
        auto &edges = calls[function.name];
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == Opcode::Call && defined.count(inst.text)) {
                    edges.push_back(inst.text);
                }
            }
        }
    }

    std::unordered_set<std::string> reachable;
    std::vector<std::string> worklist{"main"};
    reachable.insert("main");
    while (!worklist.empty()) {
        std::string current = std::move(worklist.back());
        worklist.pop_back();
        for (const auto &callee : calls[current]) {
            if (reachable.insert(callee).second) {
                worklist.push_back(callee);
            }
        }
    }

    const std::size_t oldSize = module.functions.size();
    module.functions.erase(std::remove_if(module.functions.begin(), module.functions.end(),
                                          [&](const Function &function) {
                                              return !reachable.count(function.name);
                                          }),
                           module.functions.end());
    return module.functions.size() != oldSize;
}

std::unordered_map<int, Type> scalarAllocaTypes(const Function &function, const std::unordered_set<int> &allocas) {
    std::unordered_map<int, Type> types;
    for (int id : allocas) {
        types[id] = Type{TypeKind::I32};
    }
    for (const auto &block : function.blocks) {
        for (const auto &inst : block.instructions) {
            if (inst.opcode == Opcode::Load && inst.operands.size() == 1) {
                const Value &addr = inst.operands[0];
                if (!addr.constant && allocas.count(addr.id)) {
                    types[addr.id] = inst.resultType;
                }
            } else if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                const Value &addr = inst.operands[1];
                if (!addr.constant && allocas.count(addr.id)) {
                    types[addr.id] = inst.operands[0].type;
                }
            }
        }
    }
    return types;
}

bool promoteScalarAllocasToSSA(Function &function) {
    if (hugeFunction(function)) {
        return false;
    }
    const std::unordered_set<int> allocas = promotableScalarAllocas(function);
    if (allocas.empty() || function.blocks.empty()) {
        return false;
    }

    const auto predecessors = computePredecessors(function);
    const auto types = scalarAllocaTypes(function, allocas);
    int nextId = nextValueId(function);
    bool changed = false;

    std::vector<std::unordered_map<int, int>> blockPhiResult(function.blocks.size());
    std::vector<std::vector<int>> successors(function.blocks.size());
    for (std::size_t blockIndex = 0; blockIndex < predecessors.size(); ++blockIndex) {
        for (int pred : predecessors[blockIndex]) {
            successors[static_cast<std::size_t>(pred)].push_back(static_cast<int>(blockIndex));
        }
    }

    std::unordered_set<int> allBlocks;
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        allBlocks.insert(static_cast<int>(i));
    }
    std::vector<std::unordered_set<int>> dominators(function.blocks.size());
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        dominators[i] = i == 0 ? std::unordered_set<int>{0} : allBlocks;
    }
    bool domChanged = true;
    while (domChanged) {
        domChanged = false;
        for (std::size_t blockIndex = 1; blockIndex < function.blocks.size(); ++blockIndex) {
            std::unordered_set<int> next = allBlocks;
            if (predecessors[blockIndex].empty()) {
                next.clear();
            }
            for (int pred : predecessors[blockIndex]) {
                std::unordered_set<int> intersection;
                for (int candidate : next) {
                    if (dominators[static_cast<std::size_t>(pred)].count(candidate)) {
                        intersection.insert(candidate);
                    }
                }
                next = std::move(intersection);
            }
            next.insert(static_cast<int>(blockIndex));
            if (next != dominators[blockIndex]) {
                dominators[blockIndex] = std::move(next);
                domChanged = true;
            }
        }
    }

    std::vector<std::unordered_set<int>> dominanceFrontier(function.blocks.size());
    for (std::size_t domBlock = 0; domBlock < function.blocks.size(); ++domBlock) {
        for (std::size_t join = 0; join < function.blocks.size(); ++join) {
            bool dominatesPred = false;
            for (int pred : predecessors[join]) {
                if (dominators[static_cast<std::size_t>(pred)].count(static_cast<int>(domBlock))) {
                    dominatesPred = true;
                    break;
                }
            }
            const bool strictlyDominatesJoin = domBlock != join && dominators[join].count(static_cast<int>(domBlock));
            if (dominatesPred && !strictlyDominatesJoin) {
                dominanceFrontier[domBlock].insert(static_cast<int>(join));
            }
        }
    }

    std::unordered_map<int, std::unordered_set<int>> defBlocks;
    for (int alloca : allocas) {
        defBlocks[alloca].insert(0);
    }
    for (std::size_t blockIndex = 0; blockIndex < function.blocks.size(); ++blockIndex) {
        for (const auto &inst : function.blocks[blockIndex].instructions) {
            if (inst.opcode != Opcode::Store || inst.operands.size() != 2) {
                continue;
            }
            const Value &addr = inst.operands[1];
            if (!addr.constant && allocas.count(addr.id)) {
                defBlocks[addr.id].insert(static_cast<int>(blockIndex));
            }
        }
    }

    std::vector<std::unordered_set<int>> use(function.blocks.size());
    std::vector<std::unordered_set<int>> def(function.blocks.size());
    for (std::size_t blockIndex = 0; blockIndex < function.blocks.size(); ++blockIndex) {
        std::unordered_set<int> seenDef;
        for (const auto &inst : function.blocks[blockIndex].instructions) {
            if (inst.opcode == Opcode::Load && inst.operands.size() == 1) {
                const Value &addr = inst.operands[0];
                if (!addr.constant && allocas.count(addr.id) && !seenDef.count(addr.id)) {
                    use[blockIndex].insert(addr.id);
                }
            } else if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                const Value &addr = inst.operands[1];
                if (!addr.constant && allocas.count(addr.id)) {
                    def[blockIndex].insert(addr.id);
                    seenDef.insert(addr.id);
                }
            }
        }
    }
    std::vector<std::unordered_set<int>> liveIn(function.blocks.size());
    std::vector<std::unordered_set<int>> liveOut(function.blocks.size());
    bool liveChanged = true;
    while (liveChanged) {
        liveChanged = false;
        for (std::size_t blockIndex = function.blocks.size(); blockIndex-- > 0;) {
            std::unordered_set<int> nextOut;
            for (int succ : successors[blockIndex]) {
                nextOut.insert(liveIn[static_cast<std::size_t>(succ)].begin(), liveIn[static_cast<std::size_t>(succ)].end());
            }
            std::unordered_set<int> nextIn = use[blockIndex];
            for (int id : nextOut) {
                if (!def[blockIndex].count(id)) {
                    nextIn.insert(id);
                }
            }
            if (nextOut != liveOut[blockIndex]) {
                liveOut[blockIndex] = std::move(nextOut);
                liveChanged = true;
            }
            if (nextIn != liveIn[blockIndex]) {
                liveIn[blockIndex] = std::move(nextIn);
                liveChanged = true;
            }
        }
    }

    auto addPhi = [&](int alloca, int blockIndex) {
        if (blockIndex == 0 || blockPhiResult[static_cast<std::size_t>(blockIndex)].count(alloca)) {
            return false;
        }
        blockPhiResult[static_cast<std::size_t>(blockIndex)][alloca] = nextId++;
        return true;
    };

    for (int alloca : allocas) {
        std::vector<int> worklist(defBlocks[alloca].begin(), defBlocks[alloca].end());
        std::unordered_set<int> seen(worklist.begin(), worklist.end());
        while (!worklist.empty()) {
            const int block = worklist.back();
            worklist.pop_back();
            for (int frontierBlock : dominanceFrontier[static_cast<std::size_t>(block)]) {
                if (!liveIn[static_cast<std::size_t>(frontierBlock)].count(alloca)) {
                    continue;
                }
                if (addPhi(alloca, frontierBlock) && !seen.count(frontierBlock)) {
                    seen.insert(frontierBlock);
                    worklist.push_back(frontierBlock);
                }
            }
        }
    }
    for (std::size_t blockIndex = 1; blockIndex < function.blocks.size(); ++blockIndex) {
        if (predecessors[blockIndex].size() < 2) {
            continue;
        }
        for (int alloca : liveIn[blockIndex]) {
            if (allocas.count(alloca)) {
                addPhi(alloca, static_cast<int>(blockIndex));
            }
        }
    }

    for (std::size_t blockIndex = 1; blockIndex < function.blocks.size(); ++blockIndex) {
        if (blockPhiResult[blockIndex].empty()) {
            continue;
        }
        std::vector<Instruction> phis;
        phis.reserve(blockPhiResult[blockIndex].size());
        for (const auto &[alloca, result] : blockPhiResult[blockIndex]) {
            const auto typeFound = types.find(alloca);
            const Type type = typeFound == types.end() ? Type{TypeKind::I32} : typeFound->second;
            std::string labels;
            for (std::size_t i = 0; i < predecessors[blockIndex].size(); ++i) {
                if (i != 0) {
                    labels += ",";
                }
                labels += function.blocks[static_cast<std::size_t>(predecessors[blockIndex][i])].name;
            }
            phis.push_back(Instruction{result, type, Opcode::Phi, {}, labels});
        }
        auto &instructions = function.blocks[blockIndex].instructions;
        instructions.insert(instructions.begin(), std::make_move_iterator(phis.begin()), std::make_move_iterator(phis.end()));
    }

    std::vector<std::unordered_map<int, Value>> in(function.blocks.size());
    std::vector<std::unordered_map<int, Value>> out(function.blocks.size());

    bool dataChanged = true;
    while (dataChanged) {
        dataChanged = false;
        for (std::size_t blockIndex = 0; blockIndex < function.blocks.size(); ++blockIndex) {
            std::unordered_map<int, Value> current;
            if (blockIndex == 0 || predecessors[blockIndex].empty()) {
                for (int alloca : allocas) {
                    const auto typeFound = types.find(alloca);
                    current[alloca] = zeroValue(typeFound == types.end() ? Type{TypeKind::I32} : typeFound->second);
                }
            } else if (predecessors[blockIndex].size() == 1) {
                current = out[static_cast<std::size_t>(predecessors[blockIndex].front())];
            } else {
                for (int alloca : allocas) {
                    const auto typeFound = types.find(alloca);
                    const Type type = typeFound == types.end() ? Type{TypeKind::I32} : typeFound->second;
                    const auto phiFound = blockPhiResult[blockIndex].find(alloca);
                    if (phiFound != blockPhiResult[blockIndex].end()) {
                        current[alloca] = Value{phiFound->second, type, {}, false};
                        continue;
                    }
                    Value merged = zeroValue(type);
                    bool first = true;
                    bool same = true;
                    for (int pred : predecessors[blockIndex]) {
                        const auto found = out[static_cast<std::size_t>(pred)].find(alloca);
                        const Value value = found == out[static_cast<std::size_t>(pred)].end() ? zeroValue(type) : found->second;
                        if (first) {
                            merged = value;
                            first = false;
                        } else if (!sameValue(merged, value)) {
                            same = false;
                            break;
                        }
                    }
                    current[alloca] = same ? merged : zeroValue(type);
                }
            }

            std::unordered_map<int, Value> nextOut = current;
            for (const auto &inst : function.blocks[blockIndex].instructions) {
                if (inst.opcode == Opcode::Phi) {
                    continue;
                }
                if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                    const Value &addr = inst.operands[1];
                    if (!addr.constant && allocas.count(addr.id)) {
                        nextOut[addr.id] = inst.operands[0];
                    }
                }
            }

            if (!sameMap(in[blockIndex], current)) {
                in[blockIndex] = std::move(current);
                dataChanged = true;
            }
            if (!sameMap(out[blockIndex], nextOut)) {
                out[blockIndex] = std::move(nextOut);
                dataChanged = true;
            }
        }
    }

    for (std::size_t blockIndex = 0; blockIndex < function.blocks.size(); ++blockIndex) {
        for (auto &inst : function.blocks[blockIndex].instructions) {
            if (inst.opcode != Opcode::Phi) {
                continue;
            }
            int alloca = -1;
            for (const auto &[candidate, result] : blockPhiResult[blockIndex]) {
                if (result == inst.result) {
                    alloca = candidate;
                    break;
                }
            }
            if (alloca < 0) {
                continue;
            }
            inst.operands.clear();
            for (int pred : predecessors[blockIndex]) {
                auto found = out[static_cast<std::size_t>(pred)].find(alloca);
                if (found == out[static_cast<std::size_t>(pred)].end()) {
                    const auto typeFound = types.find(alloca);
                    inst.operands.push_back(zeroValue(typeFound == types.end() ? Type{TypeKind::I32} : typeFound->second));
                } else {
                    inst.operands.push_back(found->second);
                }
            }
        }
    }

    std::unordered_map<int, Value> replacements;
    for (std::size_t blockIndex = 0; blockIndex < function.blocks.size(); ++blockIndex) {
        std::unordered_map<int, Value> current = in[blockIndex];
        std::vector<Instruction> kept;
        kept.reserve(function.blocks[blockIndex].instructions.size());
        for (auto inst : function.blocks[blockIndex].instructions) {
            for (auto &operand : inst.operands) {
                operand = resolve(operand, replacements);
            }
            if (inst.opcode == Opcode::Alloca && allocas.count(inst.result)) {
                changed = true;
                continue;
            }
            if (inst.opcode == Opcode::Load && inst.result >= 0 && inst.operands.size() == 1) {
                const Value &addr = inst.operands[0];
                if (!addr.constant && allocas.count(addr.id)) {
                    replacements[inst.result] = current[addr.id];
                    changed = true;
                    continue;
                }
            }
            if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                const Value &addr = inst.operands[1];
                if (!addr.constant && allocas.count(addr.id)) {
                    current[addr.id] = inst.operands[0];
                    changed = true;
                    continue;
                }
            }
            kept.push_back(std::move(inst));
        }
        function.blocks[blockIndex].instructions = std::move(kept);
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

bool forwardCrossBlockMemory(Function &function) {
    if (hugeFunction(function)) {
        return false;
    }
    const std::unordered_set<int> candidates = promotableScalarAllocas(function);
    if (candidates.empty() || function.blocks.empty()) {
        return false;
    }

    const auto predecessors = computePredecessors(function);
    std::vector<std::unordered_map<int, Value>> in(function.blocks.size());
    std::vector<std::unordered_map<int, Value>> out(function.blocks.size());

    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t blockIndex = 0; blockIndex < function.blocks.size(); ++blockIndex) {
            std::unordered_map<int, Value> nextIn;
            if (blockIndex != 0 && !predecessors[blockIndex].empty()) {
                nextIn = out[static_cast<std::size_t>(predecessors[blockIndex].front())];
                for (std::size_t i = 1; i < predecessors[blockIndex].size(); ++i) {
                    const auto &predOut = out[static_cast<std::size_t>(predecessors[blockIndex][i])];
                    for (auto it = nextIn.begin(); it != nextIn.end();) {
                        const auto found = predOut.find(it->first);
                        if (found == predOut.end() || !sameValue(it->second, found->second)) {
                            it = nextIn.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
            }

            std::unordered_map<int, Value> nextOut = nextIn;
            for (const auto &inst : function.blocks[blockIndex].instructions) {
                if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                    const Value &address = inst.operands[1];
                    if (!address.constant && candidates.count(address.id)) {
                        nextOut[address.id] = inst.operands[0];
                    }
                }
            }

            if (!sameMap(in[blockIndex], nextIn)) {
                in[blockIndex] = std::move(nextIn);
                changed = true;
            }
            if (!sameMap(out[blockIndex], nextOut)) {
                out[blockIndex] = std::move(nextOut);
                changed = true;
            }
        }
    }

    bool replaced = false;
    std::unordered_map<int, Value> replacements;
    for (std::size_t blockIndex = 0; blockIndex < function.blocks.size(); ++blockIndex) {
        std::unordered_map<int, Value> current = in[blockIndex];
        std::vector<Instruction> kept;
        kept.reserve(function.blocks[blockIndex].instructions.size());
        for (auto inst : function.blocks[blockIndex].instructions) {
            for (auto &operand : inst.operands) {
                operand = resolve(operand, replacements);
            }
            if (inst.opcode == Opcode::Load && inst.result >= 0 && inst.operands.size() == 1) {
                const Value &address = inst.operands[0];
                if (!address.constant && candidates.count(address.id)) {
                    const auto found = current.find(address.id);
                    if (found != current.end()) {
                        replacements[inst.result] = found->second;
                        replaced = true;
                        continue;
                    }
                }
            }
            if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                const Value &address = inst.operands[1];
                if (!address.constant && candidates.count(address.id)) {
                    current[address.id] = inst.operands[0];
                }
            }
            kept.push_back(std::move(inst));
        }
        function.blocks[blockIndex].instructions = std::move(kept);
    }

    if (replaced) {
        for (auto &block : function.blocks) {
            for (auto &inst : block.instructions) {
                for (auto &operand : inst.operands) {
                    operand = resolve(operand, replacements);
                }
            }
        }
    }
    return replaced;
}

std::string memoryAddressKey(const Value &address);

bool forwardCrossBlockExactMemory(Function &function) {
    if (function.blocks.empty() || hugeFunction(function)) {
        return false;
    }

    const auto predecessors = computePredecessors(function);
    std::vector<std::unordered_map<std::string, Value>> in(function.blocks.size());
    std::vector<std::unordered_map<std::string, Value>> out(function.blocks.size());

    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t blockIndex = 0; blockIndex < function.blocks.size(); ++blockIndex) {
            std::unordered_map<std::string, Value> nextIn;
            if (blockIndex != 0 && !predecessors[blockIndex].empty()) {
                nextIn = out[static_cast<std::size_t>(predecessors[blockIndex].front())];
                for (std::size_t i = 1; i < predecessors[blockIndex].size(); ++i) {
                    const auto &predOut = out[static_cast<std::size_t>(predecessors[blockIndex][i])];
                    for (auto it = nextIn.begin(); it != nextIn.end();) {
                        const auto found = predOut.find(it->first);
                        if (found == predOut.end() || !sameValue(it->second, found->second)) {
                            it = nextIn.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
            }

            std::unordered_map<std::string, Value> nextOut = nextIn;
            for (const auto &inst : function.blocks[blockIndex].instructions) {
                if (inst.opcode == Opcode::Call) {
                    nextOut.clear();
                    continue;
                }
                if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                    const std::string key = memoryAddressKey(inst.operands[1]);
                    nextOut.clear();
                    if (!key.empty()) {
                        nextOut[key] = inst.operands[0];
                    }
                }
            }

            if (!sameStringValueMap(in[blockIndex], nextIn)) {
                in[blockIndex] = std::move(nextIn);
                changed = true;
            }
            if (!sameStringValueMap(out[blockIndex], nextOut)) {
                out[blockIndex] = std::move(nextOut);
                changed = true;
            }
        }
    }

    bool replaced = false;
    std::unordered_map<int, Value> replacements;
    for (std::size_t blockIndex = 0; blockIndex < function.blocks.size(); ++blockIndex) {
        std::unordered_map<std::string, Value> current = in[blockIndex];
        std::vector<Instruction> kept;
        kept.reserve(function.blocks[blockIndex].instructions.size());
        for (auto inst : function.blocks[blockIndex].instructions) {
            for (auto &operand : inst.operands) {
                operand = resolve(operand, replacements);
            }

            if (inst.opcode == Opcode::Load && inst.result >= 0 && inst.operands.size() == 1) {
                const std::string key = memoryAddressKey(inst.operands[0]);
                const auto found = key.empty() ? current.end() : current.find(key);
                if (found != current.end()) {
                    replacements[inst.result] = found->second;
                    replaced = true;
                    continue;
                }
            }

            if (inst.opcode == Opcode::Call) {
                current.clear();
            } else if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                const std::string key = memoryAddressKey(inst.operands[1]);
                current.clear();
                if (!key.empty()) {
                    current[key] = inst.operands[0];
                }
            }
            kept.push_back(std::move(inst));
        }
        function.blocks[blockIndex].instructions = std::move(kept);
    }

    if (replaced) {
        for (auto &block : function.blocks) {
            for (auto &inst : block.instructions) {
                for (auto &operand : inst.operands) {
                    operand = resolve(operand, replacements);
                }
            }
        }
    }
    return replaced;
}

bool isHoistableOpcode(Opcode opcode) {
    switch (opcode) {
    case Opcode::Add:
    case Opcode::Sub:
    case Opcode::Mul:
    case Opcode::Neg:
    case Opcode::Not:
    case Opcode::ICmp:
    case Opcode::Cast:
        return true;
    case Opcode::Alloca:
    case Opcode::Load:
    case Opcode::Store:
    case Opcode::Div:
    case Opcode::Mod:
    case Opcode::FCmp:
    case Opcode::Gep:
    case Opcode::Phi:
    case Opcode::Call:
    case Opcode::Br:
    case Opcode::CondBr:
    case Opcode::Ret:
        return false;
    }
    return false;
}

bool hoistLoopInvariants(Function &function) {
    if (function.blocks.size() < 2 || hugeFunction(function)) {
        return false;
    }

    const auto predecessors = computePredecessors(function);
    std::unordered_map<std::string, int> blockIndex;
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        blockIndex[function.blocks[i].name] = static_cast<int>(i);
    }

    std::unordered_map<int, int> defBlock;
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        for (const auto &inst : function.blocks[i].instructions) {
            if (inst.result >= 0) {
                defBlock[inst.result] = static_cast<int>(i);
            }
        }
    }

    bool changed = false;
    for (std::size_t tail = 0; tail < function.blocks.size(); ++tail) {
        if (function.blocks[tail].instructions.empty()) {
            continue;
        }
        const Instruction &term = function.blocks[tail].instructions.back();
        std::vector<int> targets;
        auto addTarget = [&](const std::string &name) {
            const auto found = blockIndex.find(name);
            if (found != blockIndex.end()) {
                targets.push_back(found->second);
            }
        };
        if (term.opcode == Opcode::Br) {
            addTarget(term.text);
        } else if (term.opcode == Opcode::CondBr) {
            const std::size_t comma = term.text.find(',');
            if (comma != std::string::npos) {
                addTarget(term.text.substr(0, comma));
                addTarget(term.text.substr(comma + 2));
            }
        }

        for (int header : targets) {
            if (header < 0 || header > static_cast<int>(tail)) {
                continue;
            }

            std::unordered_set<int> loopBlocks;
            loopBlocks.insert(header);
            loopBlocks.insert(static_cast<int>(tail));
            std::vector<int> stack{static_cast<int>(tail)};
            while (!stack.empty()) {
                const int current = stack.back();
                stack.pop_back();
                for (int pred : predecessors[static_cast<std::size_t>(current)]) {
                    if (!loopBlocks.count(pred)) {
                        loopBlocks.insert(pred);
                        stack.push_back(pred);
                    }
                }
            }

            int preheader = -1;
            for (int pred : predecessors[static_cast<std::size_t>(header)]) {
                if (!loopBlocks.count(pred)) {
                    if (preheader != -1) {
                        preheader = -2;
                        break;
                    }
                    preheader = pred;
                }
            }
            if (preheader < 0 || function.blocks[static_cast<std::size_t>(preheader)].instructions.empty()) {
                continue;
            }

            std::unordered_set<int> hoistedValues;
            std::vector<Instruction> hoisted;
            bool loopChanged = true;
            while (loopChanged) {
                loopChanged = false;
                for (int blockId : loopBlocks) {
                    auto &instructions = function.blocks[static_cast<std::size_t>(blockId)].instructions;
                    std::vector<Instruction> kept;
                    kept.reserve(instructions.size());
                    for (auto &inst : instructions) {
                        if (inst.result < 0 || !isHoistableOpcode(inst.opcode)) {
                            kept.push_back(std::move(inst));
                            continue;
                        }
                        bool invariant = true;
                        for (const auto &operand : inst.operands) {
                            if (operand.constant || operand.id < 0) {
                                continue;
                            }
                            const auto def = defBlock.find(operand.id);
                            if (def != defBlock.end() && loopBlocks.count(def->second) && !hoistedValues.count(operand.id)) {
                                invariant = false;
                                break;
                            }
                        }
                        if (!invariant) {
                            kept.push_back(std::move(inst));
                            continue;
                        }
                        hoistedValues.insert(inst.result);
                        hoisted.push_back(std::move(inst));
                        loopChanged = true;
                        changed = true;
                    }
                    instructions = std::move(kept);
                }
            }

            if (!hoisted.empty()) {
                auto &pre = function.blocks[static_cast<std::size_t>(preheader)].instructions;
                auto insertPos = pre.end();
                if (!pre.empty()) {
                    --insertPos;
                }
                pre.insert(insertPos, std::make_move_iterator(hoisted.begin()), std::make_move_iterator(hoisted.end()));
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

bool simplifyTrivialPhis(Function &function) {
    bool changed = false;
    std::unordered_map<int, Value> replacements;

    for (auto &block : function.blocks) {
        std::vector<Instruction> kept;
        kept.reserve(block.instructions.size());
        for (auto inst : block.instructions) {
            for (auto &operand : inst.operands) {
                operand = resolve(operand, replacements);
            }

            if (inst.opcode == Opcode::Phi && inst.result >= 0) {
                Value merged;
                bool haveMerged = false;
                bool allSame = true;

                for (const auto &operand : inst.operands) {
                    const Value value = operand;
                    if (!haveMerged) {
                        merged = value;
                        haveMerged = true;
                    } else if (!sameValue(merged, value)) {
                        allSame = false;
                        break;
                    }
                }

                if (allSame && haveMerged) {
                    replacements[inst.result] = merged;
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

bool globallyCseEligible(Opcode opcode) {
    switch (opcode) {
    case Opcode::Add:
    case Opcode::Sub:
    case Opcode::Mul:
    case Opcode::Neg:
    case Opcode::Not:
    case Opcode::ICmp:
    case Opcode::Cast:
        return true;
    case Opcode::Div:
    case Opcode::Mod:
    case Opcode::FCmp:
    case Opcode::Alloca:
    case Opcode::Load:
    case Opcode::Store:
    case Opcode::Gep:
    case Opcode::Phi:
    case Opcode::Call:
    case Opcode::Br:
    case Opcode::CondBr:
    case Opcode::Ret:
        return false;
    }
    return false;
}

int globalCseCost(Opcode opcode) {
    switch (opcode) {
    case Opcode::Mul:
        return 3;
    case Opcode::Gep:
    case Opcode::ICmp:
        return 2;
    case Opcode::Add:
    case Opcode::Sub:
    case Opcode::Neg:
    case Opcode::Not:
    case Opcode::Cast:
        return 1;
    case Opcode::Div:
    case Opcode::Mod:
    case Opcode::FCmp:
    case Opcode::Alloca:
    case Opcode::Load:
    case Opcode::Store:
    case Opcode::Phi:
    case Opcode::Call:
    case Opcode::Br:
    case Opcode::CondBr:
    case Opcode::Ret:
        return 0;
    }
    return 0;
}

struct GlobalAvailableExpr {
    Value value;
    int block = -1;
    int cost = 0;
};

bool pressureAllowsGlobalReuse(const GlobalAvailableExpr &available, int useBlock,
                               const std::vector<int> &blockPressure) {
    if (available.block == useBlock) {
        return true;
    }
    if (available.cost >= 2) {
        return true;
    }
    const int pressure = useBlock >= 0 && static_cast<std::size_t>(useBlock) < blockPressure.size()
                             ? blockPressure[static_cast<std::size_t>(useBlock)]
                             : 1000;
    return pressure <= 32;
}

bool eliminateGlobalCommonSubexpressions(Function &function) {
    if (function.blocks.empty() || hugeFunction(function)) {
        return false;
    }

    const auto predecessors = computePredecessors(function);
    const auto dominators = computeDominators(function, predecessors);
    const auto domTree = computeDominatorTree(function, dominators);
    const auto blockPressure = estimateBlockValuePressure(function, predecessors);

    bool changed = false;
    std::unordered_map<int, Value> replacements;

    std::function<void(int, std::unordered_map<std::string, GlobalAvailableExpr>)> visit =
        [&](int blockIndex, std::unordered_map<std::string, GlobalAvailableExpr> available) {
            auto &block = function.blocks[static_cast<std::size_t>(blockIndex)];
            std::vector<Instruction> kept;
            kept.reserve(block.instructions.size());

            for (auto inst : block.instructions) {
                for (auto &operand : inst.operands) {
                    operand = resolve(operand, replacements);
                }

                if (inst.result >= 0 && globallyCseEligible(inst.opcode)) {
                    const std::string key = instKey(inst);
                    const auto found = available.find(key);
                    if (found != available.end() && pressureAllowsGlobalReuse(found->second, blockIndex, blockPressure)) {
                        replacements[inst.result] = found->second.value;
                        changed = true;
                        continue;
                    }
                    available[key] = GlobalAvailableExpr{Value{inst.result, inst.resultType, {}, false},
                                                         blockIndex, globalCseCost(inst.opcode)};
                }

                kept.push_back(std::move(inst));
            }

            block.instructions = std::move(kept);
            for (int child : domTree[static_cast<std::size_t>(blockIndex)]) {
                visit(child, available);
            }
        };

    visit(0, {});

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

std::string memoryAddressKey(const Value &address) {
    if (address.constant) {
        if (!address.name.empty() && address.name[0] == '@') {
            return "global:" + address.name;
        }
        return {};
    }
    if (address.id >= 0) {
        return "value:" + std::to_string(address.id);
    }
    return {};
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
        std::unordered_map<std::string, Value> knownMemory;
        std::vector<Instruction> kept;
        kept.reserve(block.instructions.size());

        for (auto inst : block.instructions) {
            for (auto &operand : inst.operands) {
                operand = resolve(operand, replacements);
            }

            if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                const Value &address = inst.operands[1];
                const std::string key = memoryAddressKey(address);
                if (!key.empty()) {
                    knownMemory[key] = inst.operands[0];
                }
                kept.push_back(std::move(inst));
                continue;
            }

            if (inst.opcode == Opcode::Load && inst.result >= 0 && inst.operands.size() == 1) {
                const Value &address = inst.operands[0];
                const std::string key = memoryAddressKey(address);
                if (!key.empty()) {
                    const auto found = knownMemory.find(key);
                    if (found != knownMemory.end()) {
                        replacements[inst.result] = found->second;
                        changed = true;
                        continue;
                    }
                    knownMemory[key] = Value{inst.result, inst.resultType, {}, false};
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
        if (inlineSmallFunctions(module)) {
            changed = true;
            again = true;
        }
        if (removeUnreachableFunctions(module)) {
            changed = true;
            again = true;
        }
        if (eliminatePureCallCommonSubexpressions(module)) {
            changed = true;
            again = true;
        }
        for (auto &function : module.functions) {
            if (truncateAfterTerminators(function)) {
                changed = true;
                again = true;
            }
            if (promoteSingleBlockAllocas(function)) {
                changed = true;
                again = true;
            }
            if (forwardLocalMemory(function)) {
                changed = true;
                again = true;
            }
            if (forwardCrossBlockMemory(function)) {
                changed = true;
                again = true;
            }
            if (forwardCrossBlockExactMemory(function)) {
                changed = true;
                again = true;
            }
            if (foldConstants(function)) {
                changed = true;
                again = true;
            }
            if (simplifyTrivialPhis(function)) {
                changed = true;
                again = true;
            }
            if (eliminateCommonSubexpressions(function)) {
                changed = true;
                again = true;
            }
            if (eliminateGlobalCommonSubexpressions(function)) {
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
            if (hoistLoopInvariants(function)) {
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
    bool ssaAgain = true;
    while (ssaAgain) {
        ssaAgain = false;
        if (eliminatePureCallCommonSubexpressions(module)) {
            changed = true;
            ssaAgain = true;
        }
        for (auto &function : module.functions) {
            if (truncateAfterTerminators(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (promoteScalarAllocasToSSA(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (simplifyTrivialPhis(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (foldConstants(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (simplifyTrivialPhis(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (eliminateCommonSubexpressions(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (eliminateGlobalCommonSubexpressions(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (eliminateDeadCode(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (eliminateDeadAllocas(function)) {
                changed = true;
                ssaAgain = true;
            }
        }
    }
    return changed;
}

} // namespace sysyc::ir
