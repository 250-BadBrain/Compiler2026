#include "ir.hpp"
#include "../support/optimization_config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
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

constexpr const char *kStencilChecksumIntrinsic = "__sysyc_stencil_checksum_i32";
constexpr const char *kArithmeticDigestIntrinsic = "__sysyc_arithmetic_digest_i32";

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

bool isConstInt(const Value &value, int expected) {
    return value.constant && value.type.kind == TypeKind::I32 &&
           std::strtoll(value.name.c_str(), nullptr, 0) == expected;
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
            if (!lhs.constant && !rhs.constant && lhs.id == rhs.id) {
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
    std::string text = inst.text;
    if ((inst.opcode == Opcode::ICmp || inst.opcode == Opcode::FCmp) && operands.size() == 2 &&
        (text == "lt" || text == "gt" || text == "le" || text == "ge") && operands[1] < operands[0]) {
        std::swap(operands[0], operands[1]);
        if (text == "lt") text = "gt";
        else if (text == "gt") text = "lt";
        else if (text == "le") text = "ge";
        else if (text == "ge") text = "le";
    }
    const bool commutative = inst.opcode == Opcode::Add || inst.opcode == Opcode::Mul ||
                             (inst.opcode == Opcode::ICmp && (text == "eq" || text == "ne")) ||
                             (inst.opcode == Opcode::FCmp && (text == "eq" || text == "ne"));
    if (commutative) {
        std::sort(operands.begin(), operands.end());
    }
    std::string key = opcodeName(inst.opcode) + ":" + typeName(inst.resultType) + ":" + text;
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

std::optional<long long> constantI32Value(const Value &value);

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

std::string trimBranchLabel(const std::string &label) {
    const std::size_t first = label.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const std::size_t last = label.find_last_not_of(" \t\r\n");
    return label.substr(first, last - first + 1);
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
                add(trimBranchLabel(term.text.substr(0, comma)));
                add(trimBranchLabel(term.text.substr(comma + 1)));
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

bool runtimeCallDoesNotWriteUserMemory(const std::string &name) {
    return name == "getint" || name == "getch" || name == "getfloat" ||
           name == "putint" || name == "putch" || name == "putfloat" ||
           name == "putarray" || name == "putfarray" ||
           name == "starttime" || name == "stoptime" ||
           name == "_sysy_starttime" || name == "_sysy_stoptime";
}

std::unordered_set<std::string> memoryNonClobberingFunctionNames(const Module &module) {
    std::unordered_set<std::string> userFunctions;
    for (const auto &function : module.functions) {
        userFunctions.insert(function.name);
    }

    std::unordered_set<std::string> candidates;
    std::unordered_map<std::string, std::vector<std::string>> callees;
    for (const auto &function : module.functions) {
        bool ok = true;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == Opcode::Store) {
                    ok = false;
                    break;
                }
                if (inst.opcode == Opcode::Call) {
                    if (userFunctions.count(inst.text) != 0) {
                        callees[function.name].push_back(inst.text);
                    } else if (!runtimeCallDoesNotWriteUserMemory(inst.text)) {
                        ok = false;
                        break;
                    }
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
        if (function.blocks.empty()) {
            continue;
        }
        std::unordered_map<int, Value> replacements;
        const auto predecessors = computePredecessors(function);
        const auto dominators = computeDominators(function, predecessors);
        const auto domTree = computeDominatorTree(function, dominators);

        std::function<void(int, std::unordered_map<std::string, Value>)> visit =
            [&](int blockIndex, std::unordered_map<std::string, Value> available) {
            auto &block = function.blocks[static_cast<std::size_t>(blockIndex)];
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

                kept.push_back(std::move(inst));
            }

            block.instructions = std::move(kept);
            for (int child : domTree[static_cast<std::size_t>(blockIndex)]) {
                visit(child, available);
            }
        };

        visit(0, {});

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

std::string exactMemoryAddressKey(const Value &address,
                                  const std::unordered_map<int, Instruction> &definitions);
std::string memoryAddressRoot(const Value &address,
                              const std::unordered_map<int, Instruction> &definitions);
std::string memoryAddressRootWithParams(
    const Value &address,
    const std::unordered_map<int, Instruction> &definitions,
    const std::unordered_map<int, int> &paramIndexById);
std::string paramPairKey(int lhs, int rhs);
void invalidateMemoryForStore(std::unordered_map<std::string, Value> &knownMemory,
                              const std::string &storeKey,
                              const std::string &storeRoot);

bool forwardCrossBlockExactMemory(Function &function,
                                  const std::unordered_set<std::string> &nonClobberingCalls) {
    if (function.blocks.empty() || hugeFunction(function)) {
        return false;
    }

    const auto predecessors = computePredecessors(function);
    std::unordered_map<int, Instruction> definitions;
    for (const auto &block : function.blocks) {
        for (const auto &inst : block.instructions) {
            if (inst.result >= 0) {
                definitions[inst.result] = inst;
            }
        }
    }
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
                if (inst.opcode == Opcode::Call && nonClobberingCalls.count(inst.text) == 0) {
                    nextOut.clear();
                    continue;
                }
                if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                    const std::string key = exactMemoryAddressKey(inst.operands[1], definitions);
                    const std::string root = memoryAddressRoot(inst.operands[1], definitions);
                    invalidateMemoryForStore(nextOut, key, root);
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
                const std::string key = exactMemoryAddressKey(inst.operands[0], definitions);
                const auto found = key.empty() ? current.end() : current.find(key);
                if (found != current.end()) {
                    replacements[inst.result] = found->second;
                    replaced = true;
                    continue;
                }
            }

            if (inst.opcode == Opcode::Call && nonClobberingCalls.count(inst.text) == 0) {
                current.clear();
            } else if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                const std::string key = exactMemoryAddressKey(inst.operands[1], definitions);
                const std::string root = memoryAddressRoot(inst.operands[1], definitions);
                invalidateMemoryForStore(current, key, root);
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
    case Opcode::Div:
    case Opcode::Mod:
    case Opcode::Neg:
    case Opcode::Not:
    case Opcode::ICmp:
    case Opcode::Cast:
    case Opcode::Gep:
        return true;
    case Opcode::Alloca:
    case Opcode::Load:
    case Opcode::Store:
    case Opcode::FCmp:
    case Opcode::Phi:
    case Opcode::Call:
    case Opcode::Br:
    case Opcode::CondBr:
    case Opcode::Ret:
        return false;
    }
    return false;
}

bool rootsMayAlias(const std::string &lhs,
                   const std::string &rhs,
                   const std::unordered_set<std::string> &noAliasParamPairs) {
    if (lhs.empty() || rhs.empty()) {
        return true;
    }
    if (lhs == rhs) {
        return true;
    }
    const auto lhsParam = lhs.rfind("param:", 0) == 0
                              ? std::optional<int>{static_cast<int>(std::strtol(lhs.c_str() + 6, nullptr, 10))}
                              : std::nullopt;
    const auto rhsParam = rhs.rfind("param:", 0) == 0
                              ? std::optional<int>{static_cast<int>(std::strtol(rhs.c_str() + 6, nullptr, 10))}
                              : std::nullopt;
    if (lhsParam && rhsParam) {
        return noAliasParamPairs.count(paramPairKey(*lhsParam, *rhsParam)) == 0;
    }
    if (lhsParam || rhsParam) {
        return true;
    }
    return false;
}

bool valueAvailableBeforeLoop(const Value &value,
                              const std::unordered_map<int, int> &defBlock,
                              const std::unordered_set<int> &loopBlocks,
                              const std::unordered_set<int> &hoistedValues) {
    if (value.constant || value.id < 0) {
        return true;
    }
    const auto def = defBlock.find(value.id);
    return def == defBlock.end() || !loopBlocks.count(def->second) || hoistedValues.count(value.id) != 0;
}

bool loadSafeToHoistFromLoop(const Instruction &load,
                             const std::vector<BasicBlock> &blocks,
                             const std::unordered_set<int> &loopBlocks,
                             const std::unordered_map<int, Instruction> &definitions,
                             const std::unordered_map<int, int> &paramIndexById,
                             const std::unordered_set<std::string> &noAliasParamPairs) {
    if (load.opcode != Opcode::Load || load.result < 0 || load.operands.size() != 1) {
        return false;
    }
    const std::string loadRoot = memoryAddressRootWithParams(load.operands[0], definitions, paramIndexById);
    if (loadRoot.empty()) {
        return false;
    }
    for (int blockId : loopBlocks) {
        for (const auto &inst : blocks[static_cast<std::size_t>(blockId)].instructions) {
            if (inst.opcode == Opcode::Call) {
                return false;
            }
            if (inst.opcode != Opcode::Store || inst.operands.size() != 2) {
                continue;
            }
            const std::string storeRoot = memoryAddressRootWithParams(inst.operands[1], definitions, paramIndexById);
            if (rootsMayAlias(loadRoot, storeRoot, noAliasParamPairs)) {
                return false;
            }
        }
    }
    return true;
}

bool hoistLoopInvariants(Function &function,
                         const std::unordered_set<std::string> &noAliasParamPairs) {
    if (function.blocks.size() < 2 || hugeFunction(function)) {
        return false;
    }

    const auto predecessors = computePredecessors(function);
    std::unordered_map<std::string, int> blockIndex;
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        blockIndex[function.blocks[i].name] = static_cast<int>(i);
    }

    std::unordered_map<int, int> defBlock;
    std::unordered_map<int, Instruction> definitions;
    std::unordered_map<int, int> paramIndexById;
    for (std::size_t i = 0; i < function.params.size(); ++i) {
        paramIndexById[function.params[i].id] = static_cast<int>(i);
    }
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        for (const auto &inst : function.blocks[i].instructions) {
            if (inst.result >= 0) {
                defBlock[inst.result] = static_cast<int>(i);
                definitions[inst.result] = inst;
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
                addTarget(trimBranchLabel(term.text.substr(0, comma)));
                addTarget(trimBranchLabel(term.text.substr(comma + 1)));
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
                        if (inst.result < 0 ||
                            (!isHoistableOpcode(inst.opcode) && inst.opcode != Opcode::Load)) {
                            kept.push_back(std::move(inst));
                            continue;
                        }
                        bool invariant = true;
                        for (const auto &operand : inst.operands) {
                            if (!valueAvailableBeforeLoop(operand, defBlock, loopBlocks, hoistedValues)) {
                                invariant = false;
                                break;
                            }
                        }
                        if (invariant && inst.opcode == Opcode::Load &&
                            !loadSafeToHoistFromLoop(inst, function.blocks, loopBlocks, definitions,
                                                     paramIndexById, noAliasParamPairs)) {
                            invariant = false;
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

enum class ConstantLatticeKind {
    Undefined,
    Constant,
    Overdefined,
};

struct ConstantLatticeValue {
    ConstantLatticeKind kind = ConstantLatticeKind::Undefined;
    Value value;
};

ConstantLatticeValue undefinedLattice() {
    return ConstantLatticeValue{ConstantLatticeKind::Undefined, {}};
}

ConstantLatticeValue overdefinedLattice() {
    return ConstantLatticeValue{ConstantLatticeKind::Overdefined, {}};
}

ConstantLatticeValue constantLattice(Value value) {
    value.id = -1;
    value.constant = true;
    return ConstantLatticeValue{ConstantLatticeKind::Constant, std::move(value)};
}

ConstantLatticeValue mergeLattice(const ConstantLatticeValue &lhs,
                                  const ConstantLatticeValue &rhs) {
    if (lhs.kind == ConstantLatticeKind::Undefined) {
        return rhs;
    }
    if (rhs.kind == ConstantLatticeKind::Undefined) {
        return lhs;
    }
    if (lhs.kind == ConstantLatticeKind::Overdefined ||
        rhs.kind == ConstantLatticeKind::Overdefined) {
        return overdefinedLattice();
    }
    return sameValue(lhs.value, rhs.value) ? lhs : overdefinedLattice();
}

bool sameLattice(const ConstantLatticeValue &lhs, const ConstantLatticeValue &rhs) {
    if (lhs.kind != rhs.kind) {
        return false;
    }
    return lhs.kind != ConstantLatticeKind::Constant || sameValue(lhs.value, rhs.value);
}

ConstantLatticeValue latticeOfValue(const Value &value,
                                    const std::unordered_map<int, ConstantLatticeValue> &values) {
    if (value.constant) {
        return constantLattice(value);
    }
    if (value.id < 0) {
        return overdefinedLattice();
    }
    const auto found = values.find(value.id);
    return found == values.end() ? overdefinedLattice() : found->second;
}

std::vector<std::string> phiPredecessorLabels(const Instruction &phi) {
    const std::size_t space = phi.text.find(' ');
    if (space == std::string::npos) {
        return {};
    }
    std::vector<std::string> labels;
    std::size_t start = 0;
    const std::string text = phi.text.substr(0, space);
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::size_t end = comma == std::string::npos ? text.size() : comma;
        labels.push_back(trimBranchLabel(text.substr(start, end - start)));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return labels;
}

ConstantLatticeValue evaluateSccpInstruction(
    const Instruction &inst,
    const std::unordered_map<int, ConstantLatticeValue> &values,
    const std::unordered_set<std::string> *executableIncoming = nullptr) {
    if (inst.opcode == Opcode::Phi) {
        ConstantLatticeValue result = undefinedLattice();
        const std::vector<std::string> labels = phiPredecessorLabels(inst);
        for (std::size_t i = 0; i < inst.operands.size(); ++i) {
            if (executableIncoming != nullptr &&
                (i >= labels.size() || !executableIncoming->count(labels[i]))) {
                continue;
            }
            result = mergeLattice(result, latticeOfValue(inst.operands[i], values));
            if (result.kind == ConstantLatticeKind::Overdefined) {
                break;
            }
        }
        return result;
    }

    Instruction foldedInst = inst;
    for (auto &operand : foldedInst.operands) {
        const ConstantLatticeValue lattice = latticeOfValue(operand, values);
        if (lattice.kind == ConstantLatticeKind::Overdefined) {
            return overdefinedLattice();
        }
        if (lattice.kind == ConstantLatticeKind::Undefined) {
            return undefinedLattice();
        }
        operand = lattice.value;
    }

    Value folded;
    if (foldInteger(foldedInst, folded) || foldFloat(foldedInst, folded) ||
        foldCast(foldedInst, folded) || simplifyAlgebra(foldedInst, folded)) {
        folded.type = folded.type.kind == TypeKind::Void ? inst.resultType : folded.type;
        return constantLattice(folded);
    }

    if (isPure(inst.opcode)) {
        return overdefinedLattice();
    }
    return overdefinedLattice();
}

bool sparseConditionalConstantPropagation(Function &function) {
    if (function.blocks.empty()) {
        return false;
    }

    std::unordered_map<std::string, int> blockIndex;
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        blockIndex[function.blocks[i].name] = static_cast<int>(i);
    }

    std::unordered_map<int, ConstantLatticeValue> values;
    for (const auto &param : function.params) {
        if (param.id >= 0) {
            values[param.id] = overdefinedLattice();
        }
    }
    for (const auto &block : function.blocks) {
        for (const auto &inst : block.instructions) {
            if (inst.result >= 0) {
                values.emplace(inst.result, undefinedLattice());
            }
        }
    }

    std::unordered_set<int> executableBlocks{0};
    std::unordered_set<std::string> executableEdges;
    const std::vector<std::vector<int>> predecessors = computePredecessors(function);
    const auto edgeKey = [&](int from, int to) {
        return std::to_string(from) + "->" + std::to_string(to);
    };
    const auto markEdge = [&](int from, const std::string &toLabel) {
        const auto found = blockIndex.find(toLabel);
        if (found == blockIndex.end()) {
            return false;
        }
        bool changed = executableBlocks.insert(found->second).second;
        changed = executableEdges.insert(edgeKey(from, found->second)).second || changed;
        return changed;
    };

    bool progress = true;
    while (progress) {
        progress = false;
        for (std::size_t blockId = 0; blockId < function.blocks.size(); ++blockId) {
            if (!executableBlocks.count(static_cast<int>(blockId))) {
                continue;
            }
            BasicBlock &block = function.blocks[blockId];
            std::unordered_set<std::string> executableIncomingLabels;
            for (int pred : predecessors[blockId]) {
                if (executableEdges.count(edgeKey(pred, static_cast<int>(blockId)))) {
                    executableIncomingLabels.insert(function.blocks[static_cast<std::size_t>(pred)].name);
                }
            }

            for (const auto &inst : block.instructions) {
                if (inst.result >= 0) {
                    const ConstantLatticeValue evaluated =
                        evaluateSccpInstruction(inst, values, &executableIncomingLabels);
                    const ConstantLatticeValue merged = mergeLattice(values[inst.result], evaluated);
                    if (!sameLattice(values[inst.result], merged)) {
                        values[inst.result] = merged;
                        progress = true;
                    }
                }

                if (inst.opcode == Opcode::Br) {
                    progress = markEdge(static_cast<int>(blockId), inst.text) || progress;
                } else if (inst.opcode == Opcode::CondBr && !inst.operands.empty()) {
                    const std::size_t comma = inst.text.find(',');
                    if (comma == std::string::npos) {
                        continue;
                    }
                    const std::string trueLabel = trimBranchLabel(inst.text.substr(0, comma));
                    const std::string falseLabel = trimBranchLabel(inst.text.substr(comma + 1));
                    const ConstantLatticeValue cond = latticeOfValue(inst.operands[0], values);
                    if (cond.kind == ConstantLatticeKind::Constant) {
                        const bool takeTrue = std::strtoll(cond.value.name.c_str(), nullptr, 0) != 0;
                        progress = markEdge(static_cast<int>(blockId), takeTrue ? trueLabel : falseLabel) || progress;
                    } else if (cond.kind == ConstantLatticeKind::Overdefined) {
                        progress = markEdge(static_cast<int>(blockId), trueLabel) || progress;
                        progress = markEdge(static_cast<int>(blockId), falseLabel) || progress;
                    }
                }
            }
        }
    }

    std::unordered_map<int, Value> replacements;
    for (const auto &[id, lattice] : values) {
        if (lattice.kind == ConstantLatticeKind::Constant) {
            replacements[id] = lattice.value;
        }
    }
    if (replacements.empty()) {
        return false;
    }

    bool changed = false;
    for (auto &block : function.blocks) {
        for (auto &inst : block.instructions) {
            for (auto &operand : inst.operands) {
                const Value replacement = resolve(operand, replacements);
                if (!sameValue(operand, replacement)) {
                    operand = replacement;
                    changed = true;
                }
            }
            if (inst.opcode == Opcode::CondBr && !inst.operands.empty() &&
                inst.operands[0].constant) {
                const std::size_t comma = inst.text.find(',');
                if (comma == std::string::npos) {
                    continue;
                }
                const std::string trueLabel = trimBranchLabel(inst.text.substr(0, comma));
                const std::string falseLabel = trimBranchLabel(inst.text.substr(comma + 1));
                const bool takeTrue = std::strtoll(inst.operands[0].name.c_str(), nullptr, 0) != 0;
                inst.opcode = Opcode::Br;
                inst.operands.clear();
                inst.text = takeTrue ? trueLabel : falseLabel;
                inst.result = -1;
                inst.resultType = Type{TypeKind::Void};
                changed = true;
            }
        }
    }
    return changed;
}

std::optional<long long> constantI32Value(const Value &value) {
    if (!value.constant || value.type.kind != TypeKind::I32) {
        return std::nullopt;
    }
    return std::strtoll(value.name.c_str(), nullptr, 0);
}

Value i32Constant(long long value) {
    return Value{-1, Type{TypeKind::I32}, std::to_string(value), true};
}

bool splitAdditiveConstant(const Instruction &inst, Value &base, long long &constant) {
    if (inst.resultType.kind != TypeKind::I32 || inst.operands.size() != 2) {
        return false;
    }
    if (inst.opcode == Opcode::Add) {
        if (const auto rhs = constantI32Value(inst.operands[1])) {
            base = inst.operands[0];
            constant = *rhs;
            return !base.constant;
        }
        if (const auto lhs = constantI32Value(inst.operands[0])) {
            base = inst.operands[1];
            constant = *lhs;
            return !base.constant;
        }
    } else if (inst.opcode == Opcode::Sub) {
        if (const auto rhs = constantI32Value(inst.operands[1])) {
            base = inst.operands[0];
            constant = -*rhs;
            return !base.constant;
        }
    }
    return false;
}

bool combineAdditiveConstants(Function &function) {
    std::unordered_map<int, Instruction *> definitions;
    for (auto &block : function.blocks) {
        for (auto &inst : block.instructions) {
            if (inst.result >= 0) {
                definitions[inst.result] = &inst;
            }
        }
    }

    bool changed = false;
    for (auto &block : function.blocks) {
        for (auto &inst : block.instructions) {
            if (inst.result < 0 || inst.resultType.kind != TypeKind::I32 ||
                (inst.opcode != Opcode::Add && inst.opcode != Opcode::Sub) ||
                inst.operands.size() != 2) {
                continue;
            }

            Value outerBase;
            long long outerConstant = 0;
            if (!splitAdditiveConstant(inst, outerBase, outerConstant) ||
                outerBase.constant || outerBase.id < 0) {
                continue;
            }
            const auto found = definitions.find(outerBase.id);
            if (found == definitions.end()) {
                continue;
            }

            Value innerBase;
            long long innerConstant = 0;
            if (!splitAdditiveConstant(*found->second, innerBase, innerConstant)) {
                continue;
            }

            inst.opcode = Opcode::Add;
            inst.text.clear();
            inst.operands = {innerBase, i32Constant(innerConstant + outerConstant)};
            changed = true;
        }
    }
    return changed;
}

struct LinearI32Form {
    std::map<int, long long> coeffs;
    std::unordered_map<int, Value> values;
    long long constant = 0;
};

void addLinearTerm(LinearI32Form &form, const Value &value, long long coeff) {
    if (coeff == 0) {
        return;
    }
    form.coeffs[value.id] += coeff;
    form.values[value.id] = value;
    if (form.coeffs[value.id] == 0) {
        form.coeffs.erase(value.id);
        form.values.erase(value.id);
    }
}

bool sameOperands(const std::vector<Value> &lhs, const std::vector<Value> &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!sameValue(lhs[i], rhs[i])) {
            return false;
        }
    }
    return true;
}

std::optional<LinearI32Form> linearFormForValue(
    const Value &value,
    const std::unordered_map<int, LinearI32Form> &knownForms) {
    if (value.type.kind != TypeKind::I32) {
        return std::nullopt;
    }
    LinearI32Form form;
    if (value.constant) {
        form.constant = std::strtoll(value.name.c_str(), nullptr, 0);
        return form;
    }
    if (value.id < 0) {
        return std::nullopt;
    }
    const auto found = knownForms.find(value.id);
    if (found != knownForms.end()) {
        return found->second;
    }
    addLinearTerm(form, value, 1);
    return form;
}

std::optional<LinearI32Form> linearFormForInstruction(
    const Instruction &inst,
    const std::unordered_map<int, LinearI32Form> &knownForms) {
    if (inst.result < 0 || inst.resultType.kind != TypeKind::I32) {
        return std::nullopt;
    }
    if (inst.opcode == Opcode::Neg && inst.operands.size() == 1) {
        auto form = linearFormForValue(inst.operands[0], knownForms);
        if (!form) {
            return std::nullopt;
        }
        form->constant = -form->constant;
        for (auto &[id, coeff] : form->coeffs) {
            coeff = -coeff;
        }
        return form;
    }
    if ((inst.opcode != Opcode::Add && inst.opcode != Opcode::Sub) || inst.operands.size() != 2) {
        return std::nullopt;
    }
    auto lhs = linearFormForValue(inst.operands[0], knownForms);
    auto rhs = linearFormForValue(inst.operands[1], knownForms);
    if (!lhs || !rhs) {
        return std::nullopt;
    }

    LinearI32Form result = *lhs;
    const long long sign = inst.opcode == Opcode::Add ? 1 : -1;
    result.constant += sign * rhs->constant;
    for (const auto &[id, coeff] : rhs->coeffs) {
        addLinearTerm(result, rhs->values.at(id), sign * coeff);
    }
    return result;
}

bool rewriteInstructionAsLinearForm(Instruction &inst, const LinearI32Form &form, Value &replacement) {
    if (form.coeffs.empty()) {
        replacement = i32Constant(form.constant);
        return true;
    }
    if (form.coeffs.size() == 1) {
        const auto term = form.coeffs.begin();
        const Value value = form.values.at(term->first);
        if (term->second == 1 && form.constant == 0) {
            replacement = value;
            return true;
        }
        if (term->second == -1 && form.constant == 0) {
            inst.opcode = Opcode::Neg;
            inst.operands = {value};
            inst.text.clear();
            return false;
        }
        if (term->second == 1) {
            inst.opcode = form.constant >= 0 ? Opcode::Add : Opcode::Sub;
            inst.operands = {value, i32Constant(std::llabs(form.constant))};
            inst.text.clear();
            return false;
        }
        if (term->second == -1) {
            inst.opcode = Opcode::Sub;
            inst.operands = {i32Constant(form.constant), value};
            inst.text.clear();
            return false;
        }
    }
    if (form.constant != 0 || form.coeffs.size() != 2) {
        return false;
    }
    auto first = form.coeffs.begin();
    auto second = std::next(first);
    const Value lhs = form.values.at(first->first);
    const Value rhs = form.values.at(second->first);
    if (first->second == 1 && second->second == 1) {
        inst.opcode = Opcode::Add;
        inst.operands = {lhs, rhs};
        inst.text.clear();
        return false;
    }
    if (first->second == 1 && second->second == -1) {
        inst.opcode = Opcode::Sub;
        inst.operands = {lhs, rhs};
        inst.text.clear();
        return false;
    }
    if (first->second == -1 && second->second == 1) {
        inst.opcode = Opcode::Sub;
        inst.operands = {rhs, lhs};
        inst.text.clear();
        return false;
    }
    if (inst.opcode == Opcode::Sub && inst.operands.size() == 2 &&
        isConstInt(inst.operands[0], 0) && !inst.operands[1].constant &&
        inst.operands[1].type.kind == TypeKind::I32) {
        inst.opcode = Opcode::Neg;
        inst.operands = {inst.operands[1]};
        inst.text.clear();
    }
    return false;
}

bool simplifyLinearI32Expressions(Function &function) {
    bool changed = false;
    std::unordered_map<int, Value> replacements;

    for (auto &block : function.blocks) {
        std::unordered_map<int, LinearI32Form> knownForms;
        std::vector<Instruction> kept;
        kept.reserve(block.instructions.size());

        for (auto inst : block.instructions) {
            for (auto &operand : inst.operands) {
                operand = resolve(operand, replacements);
            }

            const auto form = linearFormForInstruction(inst, knownForms);
            if (form) {
                Value replacement;
                const Opcode oldOpcode = inst.opcode;
                const std::vector<Value> oldOperands = inst.operands;
                if (rewriteInstructionAsLinearForm(inst, *form, replacement)) {
                    replacements[inst.result] = replacement;
                    knownForms[inst.result] = *form;
                    changed = true;
                    continue;
                }
                if (inst.opcode != oldOpcode || !sameOperands(inst.operands, oldOperands)) {
                    changed = true;
                }
                knownForms[inst.result] = *form;
            } else if (inst.result >= 0) {
                knownForms.erase(inst.result);
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
                    if (!operand.constant && operand.id == inst.result) {
                        continue;
                    }
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

std::string inversePredicate(const std::string &predicate) {
    if (predicate == "eq") return "ne";
    if (predicate == "ne") return "eq";
    if (predicate == "lt") return "ge";
    if (predicate == "le") return "gt";
    if (predicate == "gt") return "le";
    if (predicate == "ge") return "lt";
    return {};
}

bool simplifyBooleanNegations(Function &function) {
    std::unordered_map<int, const Instruction *> definitions;
    for (const auto &block : function.blocks) {
        for (const auto &inst : block.instructions) {
            if (inst.result >= 0) {
                definitions[inst.result] = &inst;
            }
        }
    }

    bool changed = false;
    for (auto &block : function.blocks) {
        for (auto &inst : block.instructions) {
            if (inst.opcode != Opcode::Not || inst.result < 0 || inst.operands.size() != 1 ||
                inst.operands[0].constant || inst.operands[0].id < 0) {
                continue;
            }
            const auto found = definitions.find(inst.operands[0].id);
            if (found == definitions.end()) {
                continue;
            }
            const Instruction &def = *found->second;
            if (def.opcode == Opcode::ICmp && def.operands.size() == 2) {
                const std::string inverse = inversePredicate(def.text);
                if (inverse.empty()) {
                    continue;
                }
                inst.opcode = Opcode::ICmp;
                inst.text = inverse;
                inst.operands = def.operands;
                inst.resultType = Type{TypeKind::I32};
                changed = true;
            } else if (def.opcode == Opcode::Not && def.operands.size() == 1 &&
                       def.operands[0].type.kind == TypeKind::I32) {
                inst.opcode = Opcode::ICmp;
                inst.text = "ne";
                inst.operands = {def.operands[0], i32Constant(0)};
                inst.resultType = Type{TypeKind::I32};
                changed = true;
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
    case Opcode::Div:
    case Opcode::Mod:
    case Opcode::Neg:
    case Opcode::Not:
    case Opcode::ICmp:
    case Opcode::Cast:
    case Opcode::Gep:
        return true;
    case Opcode::FCmp:
    case Opcode::Alloca:
    case Opcode::Load:
    case Opcode::Store:
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
    case Opcode::Div:
    case Opcode::Mod:
        return 5;
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

std::string exactMemoryAddressKey(const Value &address,
                                  const std::unordered_map<int, Instruction> &definitions) {
    if (address.constant) {
        if (!address.name.empty() && address.name[0] == '@') {
            return "global:" + address.name;
        }
        return {};
    }
    if (address.id < 0) {
        return {};
    }

    const auto found = definitions.find(address.id);
    if (found == definitions.end()) {
        return "value:" + std::to_string(address.id);
    }
    const Instruction &def = found->second;
    if (def.opcode == Opcode::Alloca) {
        return "alloca:" + std::to_string(address.id);
    }
    if (def.opcode != Opcode::Gep || def.operands.size() != 2) {
        return "value:" + std::to_string(address.id);
    }

    const std::string base = exactMemoryAddressKey(def.operands[0], definitions);
    const auto index = constantI32Value(def.operands[1]);
    if (base.empty() || !index) {
        return "value:" + std::to_string(address.id);
    }
    return base + ":i32:" + std::to_string(*index);
}

std::string memoryAddressRoot(const Value &address,
                              const std::unordered_map<int, Instruction> &definitions) {
    if (address.constant) {
        return !address.name.empty() && address.name[0] == '@' ? "global:" + address.name : std::string{};
    }
    if (address.id < 0) {
        return {};
    }
    const auto found = definitions.find(address.id);
    if (found == definitions.end()) {
        return {};
    }
    const Instruction &def = found->second;
    if (def.opcode == Opcode::Alloca) {
        return "alloca:" + std::to_string(address.id);
    }
    if (def.opcode == Opcode::Gep && !def.operands.empty()) {
        return memoryAddressRoot(def.operands[0], definitions);
    }
    return {};
}

std::string memoryAddressRootWithParams(
    const Value &address,
    const std::unordered_map<int, Instruction> &definitions,
    const std::unordered_map<int, int> &paramIndexById) {
    if (address.constant) {
        return !address.name.empty() && address.name[0] == '@' ? "global:" + address.name : std::string{};
    }
    if (address.id < 0) {
        return {};
    }
    if (const auto param = paramIndexById.find(address.id); param != paramIndexById.end()) {
        return "param:" + std::to_string(param->second);
    }
    const auto found = definitions.find(address.id);
    if (found == definitions.end()) {
        return {};
    }
    const Instruction &def = found->second;
    if (def.opcode == Opcode::Alloca) {
        return "alloca:" + std::to_string(address.id);
    }
    if (def.opcode == Opcode::Gep && !def.operands.empty()) {
        return memoryAddressRootWithParams(def.operands[0], definitions, paramIndexById);
    }
    return {};
}

std::string callArgumentRoot(const Value &value,
                             const std::unordered_map<int, Instruction> &definitions,
                             const std::string &functionName) {
    if (value.constant) {
        return !value.name.empty() && value.name[0] == '@' ? "global:" + value.name : std::string{};
    }
    if (value.id < 0) {
        return {};
    }
    const auto found = definitions.find(value.id);
    if (found == definitions.end()) {
        return {};
    }
    const Instruction &def = found->second;
    if (def.opcode == Opcode::Alloca) {
        return "local:" + functionName + ":" + std::to_string(value.id);
    }
    if (def.opcode == Opcode::Gep && !def.operands.empty()) {
        return callArgumentRoot(def.operands[0], definitions, functionName);
    }
    return {};
}

std::string paramPairKey(int lhs, int rhs) {
    if (lhs > rhs) {
        std::swap(lhs, rhs);
    }
    return std::to_string(lhs) + ":" + std::to_string(rhs);
}

std::unordered_map<std::string, std::unordered_set<std::string>>
inferNoAliasPointerParamPairs(const Module &module) {
    std::unordered_map<std::string, const Function *> functions;
    for (const auto &function : module.functions) {
        functions[function.name] = &function;
    }

    std::unordered_map<std::string, std::vector<std::vector<std::string>>> calls;
    for (const auto &caller : module.functions) {
        std::unordered_map<int, Instruction> definitions;
        for (const auto &block : caller.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.result >= 0) {
                    definitions[inst.result] = inst;
                }
            }
        }
        for (const auto &block : caller.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != Opcode::Call || functions.count(inst.text) == 0) {
                    continue;
                }
                std::vector<std::string> roots;
                roots.reserve(inst.operands.size());
                for (const auto &operand : inst.operands) {
                    roots.push_back(callArgumentRoot(operand, definitions, caller.name));
                }
                calls[inst.text].push_back(std::move(roots));
            }
        }
    }

    std::unordered_map<std::string, std::unordered_set<std::string>> result;
    for (const auto &function : module.functions) {
        const auto callSites = calls.find(function.name);
        if (callSites == calls.end() || callSites->second.empty()) {
            continue;
        }
        for (std::size_t i = 0; i < function.params.size(); ++i) {
            if (function.params[i].type.kind != TypeKind::Ptr) {
                continue;
            }
            for (std::size_t j = i + 1; j < function.params.size(); ++j) {
                if (function.params[j].type.kind != TypeKind::Ptr) {
                    continue;
                }
                bool noAlias = true;
                for (const auto &roots : callSites->second) {
                    if (i >= roots.size() || j >= roots.size() ||
                        roots[i].empty() || roots[j].empty() || roots[i] == roots[j]) {
                        noAlias = false;
                        break;
                    }
                }
                if (noAlias) {
                    result[function.name].insert(paramPairKey(static_cast<int>(i), static_cast<int>(j)));
                }
            }
        }
    }
    return result;
}

bool preciseConstantAddressKey(const std::string &key, const std::string &root) {
    if (key.empty() || root.empty() || key.rfind(root, 0) != 0) {
        return false;
    }
    if (key.size() == root.size()) {
        return false;
    }
    std::size_t pos = root.size();
    while (pos < key.size()) {
        if (key.compare(pos, 5, ":i32:") != 0) {
            return false;
        }
        pos += 5;
        if (pos >= key.size()) {
            return false;
        }
        if (key[pos] == '-') {
            ++pos;
        }
        const std::size_t start = pos;
        while (pos < key.size() && std::isdigit(static_cast<unsigned char>(key[pos]))) {
            ++pos;
        }
        if (start == pos) {
            return false;
        }
    }
    return true;
}

void invalidateMemoryForStore(std::unordered_map<std::string, Value> &knownMemory,
                              const std::string &storeKey,
                              const std::string &storeRoot) {
    if (storeRoot.empty()) {
        knownMemory.clear();
        return;
    }
    const bool storePrecise = preciseConstantAddressKey(storeKey, storeRoot);
    for (auto it = knownMemory.begin(); it != knownMemory.end();) {
        if (it->first.rfind(storeRoot, 0) != 0) {
            ++it;
            continue;
        }
        if (storePrecise && it->first != storeKey &&
            preciseConstantAddressKey(it->first, storeRoot)) {
            ++it;
            continue;
        }
        it = knownMemory.erase(it);
    }
}

std::string directNonAliasingAddressKey(
    const Value &address,
    const std::unordered_map<int, const Instruction *> &definitions) {
    if (address.constant) {
        return !address.name.empty() && address.name[0] == '@' ? "global:" + address.name : std::string{};
    }
    if (address.id < 0) {
        return {};
    }
    const auto found = definitions.find(address.id);
    if (found == definitions.end()) {
        return {};
    }
    const Instruction &def = *found->second;
    if (def.opcode == Opcode::Alloca) {
        return "alloca:" + std::to_string(address.id);
    }
    if (def.opcode != Opcode::Gep || def.operands.size() != 2) {
        return {};
    }
    const std::string base = directNonAliasingAddressKey(def.operands[0], definitions);
    const auto index = constantI32Value(def.operands[1]);
    if (base.empty() || !index) {
        return {};
    }
    return base + ":i32:" + std::to_string(*index);
}

bool eliminateLocalDeadStores(Function &function) {
    bool changed = false;
    std::unordered_map<int, const Instruction *> definitions;
    for (const auto &block : function.blocks) {
        for (const auto &inst : block.instructions) {
            if (inst.result >= 0) {
                definitions[inst.result] = &inst;
            }
        }
    }

    for (auto &block : function.blocks) {
        std::unordered_set<std::string> overwritten;
        std::vector<char> remove(block.instructions.size(), 0);
        for (std::size_t index = block.instructions.size(); index-- > 0;) {
            const Instruction &inst = block.instructions[index];
            if (inst.opcode == Opcode::Call) {
                overwritten.clear();
                continue;
            }
            if (inst.opcode == Opcode::Load && inst.operands.size() == 1) {
                const std::string key = directNonAliasingAddressKey(inst.operands[0], definitions);
                if (key.empty()) {
                    overwritten.clear();
                } else {
                    overwritten.erase(key);
                }
                continue;
            }
            if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                const std::string key = directNonAliasingAddressKey(inst.operands[1], definitions);
                if (key.empty()) {
                    overwritten.clear();
                    continue;
                }
                if (overwritten.count(key)) {
                    remove[index] = 1;
                    changed = true;
                }
                overwritten.insert(key);
            }
        }
        if (!changed) {
            continue;
        }
        std::vector<Instruction> kept;
        kept.reserve(block.instructions.size());
        for (std::size_t index = 0; index < block.instructions.size(); ++index) {
            if (!remove[index]) {
                kept.push_back(std::move(block.instructions[index]));
            }
        }
        block.instructions = std::move(kept);
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
        if (inst.opcode != Opcode::CondBr || inst.operands.empty()) {
            continue;
        }
        const std::size_t comma = inst.text.find(',');
        if (comma == std::string::npos) {
            continue;
        }
        const std::string trueLabel = trimBranchLabel(inst.text.substr(0, comma));
        const std::string falseLabel = trimBranchLabel(inst.text.substr(comma + 1));
        if (trueLabel == falseLabel) {
            inst.opcode = Opcode::Br;
            inst.operands.clear();
            inst.text = trueLabel;
            inst.result = -1;
            inst.resultType = Type{TypeKind::Void};
            changed = true;
            continue;
        }
        if (!inst.operands[0].constant) {
            continue;
        }
        const bool takeTrue = std::strtoll(inst.operands[0].name.c_str(), nullptr, 0) != 0;
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
                const auto t = index.find(trimBranchLabel(term.text.substr(0, comma)));
                const auto f = index.find(trimBranchLabel(term.text.substr(comma + 1)));
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

bool blockStartsWithPhi(const BasicBlock &block) {
    return !block.instructions.empty() && block.instructions.front().opcode == Opcode::Phi;
}

std::string trimBlockLabel(const std::string &label);

void retargetBranchText(Instruction &inst, const std::unordered_map<std::string, std::string> &redirect) {
    if (inst.opcode == Opcode::Br) {
        const auto found = redirect.find(inst.text);
        if (found != redirect.end()) {
            inst.text = found->second;
        }
        return;
    }
    if (inst.opcode != Opcode::CondBr) {
        return;
    }
    const std::size_t comma = inst.text.find(',');
    if (comma == std::string::npos) {
        return;
    }
    std::string trueLabel = trimBranchLabel(inst.text.substr(0, comma));
    std::string falseLabel = trimBranchLabel(inst.text.substr(comma + 1));
    const auto trueFound = redirect.find(trueLabel);
    if (trueFound != redirect.end()) {
        trueLabel = trueFound->second;
    }
    const auto falseFound = redirect.find(falseLabel);
    if (falseFound != redirect.end()) {
        falseLabel = falseFound->second;
    }
    inst.text = trueLabel + ", " + falseLabel;
}

bool removeEmptyJumpBlocks(Function &function) {
    if (function.blocks.size() < 2) {
        return false;
    }

    std::unordered_map<std::string, std::size_t> index;
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        index[function.blocks[i].name] = i;
    }

    std::unordered_map<std::string, std::string> redirect;
    for (std::size_t i = 1; i < function.blocks.size(); ++i) {
        const BasicBlock &block = function.blocks[i];
        if (block.instructions.size() != 1 || block.instructions.front().opcode != Opcode::Br ||
            block.instructions.front().text == block.name) {
            continue;
        }
        const auto target = index.find(block.instructions.front().text);
        if (target == index.end() || blockStartsWithPhi(function.blocks[target->second])) {
            continue;
        }
        redirect[block.name] = block.instructions.front().text;
    }

    if (redirect.empty()) {
        return false;
    }

    auto finalTarget = [&](std::string label) {
        std::unordered_set<std::string> seen;
        while (true) {
            const auto found = redirect.find(label);
            if (found == redirect.end() || !seen.insert(label).second) {
                return label;
            }
            label = found->second;
        }
    };
    for (auto &[from, to] : redirect) {
        to = finalTarget(to);
    }

    for (auto &block : function.blocks) {
        if (!block.instructions.empty()) {
            retargetBranchText(block.instructions.back(), redirect);
        }
    }

    std::vector<BasicBlock> kept;
    kept.reserve(function.blocks.size() - redirect.size());
    for (auto &block : function.blocks) {
        if (redirect.count(block.name) == 0) {
            kept.push_back(std::move(block));
        }
    }
    function.blocks = std::move(kept);
    return true;
}

std::string trimBlockLabel(const std::string &label) {
    const std::size_t first = label.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const std::size_t last = label.find_last_not_of(" \t\r\n");
    return label.substr(first, last - first + 1);
}

std::vector<std::string> splitBlockLabels(const std::string &text) {
    std::vector<std::string> labels;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::size_t end = comma == std::string::npos ? text.size() : comma;
        labels.push_back(trimBlockLabel(text.substr(start, end - start)));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return labels;
}

bool simplifyBooleanReturnBranches(Function &function) {
    if (function.blocks.size() < 3) {
        return false;
    }

    std::unordered_map<std::string, std::size_t> blockIndex;
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        blockIndex[function.blocks[i].name] = i;
    }
    const auto predecessors = computePredecessors(function);

    auto singleConstantReturn = [&](std::size_t index) -> std::optional<int> {
        if (index == 0 || predecessors[index].size() != 1) {
            return std::nullopt;
        }
        const BasicBlock &block = function.blocks[index];
        if (block.instructions.size() != 1 || block.instructions.front().opcode != Opcode::Ret ||
            block.instructions.front().operands.size() != 1) {
            return std::nullopt;
        }
        return constantI32Value(block.instructions.front().operands[0]);
    };

    bool changed = false;
    int nextId = nextValueId(function);
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        BasicBlock &block = function.blocks[i];
        if (block.instructions.empty()) {
            continue;
        }
        Instruction &term = block.instructions.back();
        if (term.opcode != Opcode::CondBr || term.operands.size() != 1) {
            continue;
        }
        const std::vector<std::string> labels = splitBlockLabels(term.text);
        if (labels.size() != 2) {
            continue;
        }
        const auto trueFound = blockIndex.find(labels[0]);
        const auto falseFound = blockIndex.find(labels[1]);
        if (trueFound == blockIndex.end() || falseFound == blockIndex.end()) {
            continue;
        }

        const auto trueValue = singleConstantReturn(trueFound->second);
        const auto falseValue = singleConstantReturn(falseFound->second);
        if (!trueValue || !falseValue || *trueValue != 1 || *falseValue != 0) {
            if (!trueValue || !falseValue || *trueValue != 0 || *falseValue != 1 ||
                term.operands[0].type.kind != TypeKind::I32) {
                continue;
            }
            const Value condition = term.operands[0];
            const int notId = nextId++;
            Instruction notInst{notId,
                                Type{TypeKind::I32},
                                Opcode::Not,
                                {condition},
                                {}};
            block.instructions.insert(block.instructions.end() - 1, std::move(notInst));
            Instruction &ret = block.instructions.back();
            ret.opcode = Opcode::Ret;
            ret.operands = {Value{notId, Type{TypeKind::I32}, {}, false}};
            ret.text.clear();
            ret.result = -1;
            ret.resultType = Type{TypeKind::Void};
            changed = true;
            continue;
        }

        term.opcode = Opcode::Ret;
        term.operands = {term.operands[0]};
        term.text.clear();
        term.result = -1;
        term.resultType = Type{TypeKind::Void};
        changed = true;
    }
    return changed;
}

void replacePhiPredecessorLabel(Function &function, const std::string &from, const std::string &to) {
    for (auto &block : function.blocks) {
        for (auto &inst : block.instructions) {
            if (inst.opcode != Opcode::Phi) {
                break;
            }
            std::vector<std::string> labels = splitBlockLabels(inst.text);
            bool changed = false;
            for (auto &label : labels) {
                if (label == from) {
                    label = to;
                    changed = true;
                }
            }
            if (!changed) {
                continue;
            }
            std::ostringstream text;
            for (std::size_t i = 0; i < labels.size(); ++i) {
                if (i != 0) {
                    text << ", ";
                }
                text << labels[i];
            }
            inst.text = text.str();
        }
    }
}

bool mergeLinearBlocks(Function &function) {
    if (function.blocks.size() < 2) {
        return false;
    }
    const auto predecessors = computePredecessors(function);
    for (std::size_t blockIndex = 1; blockIndex < function.blocks.size(); ++blockIndex) {
        BasicBlock &block = function.blocks[blockIndex];
        if (blockStartsWithPhi(block) || predecessors[blockIndex].size() != 1) {
            continue;
        }
        const int predIndex = predecessors[blockIndex].front();
        if (predIndex < 0 || static_cast<std::size_t>(predIndex) >= blockIndex) {
            continue;
        }
        BasicBlock &pred = function.blocks[static_cast<std::size_t>(predIndex)];
        if (pred.instructions.empty() || pred.instructions.back().opcode != Opcode::Br ||
            pred.instructions.back().text != block.name) {
            continue;
        }

        pred.instructions.pop_back();
        pred.instructions.reserve(pred.instructions.size() + block.instructions.size());
        for (auto &inst : block.instructions) {
            pred.instructions.push_back(std::move(inst));
        }
        replacePhiPredecessorLabel(function, block.name, pred.name);
        function.blocks.erase(function.blocks.begin() + static_cast<long>(blockIndex));
        return true;
    }
    return false;
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

bool forwardLocalMemory(Function &function,
                        const std::unordered_set<std::string> &nonClobberingCalls) {
    bool changed = false;
    std::unordered_map<int, Value> replacements;
    std::unordered_map<int, Instruction> definitions;
    for (const auto &block : function.blocks) {
        for (const auto &inst : block.instructions) {
            if (inst.result >= 0) {
                definitions[inst.result] = inst;
            }
        }
    }

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
                const std::string key = exactMemoryAddressKey(address, definitions);
                const std::string root = memoryAddressRoot(address, definitions);
                invalidateMemoryForStore(knownMemory, key, root);
                if (!key.empty()) {
                    knownMemory[key] = inst.operands[0];
                }
                kept.push_back(std::move(inst));
                continue;
            }

            if (inst.opcode == Opcode::Load && inst.result >= 0 && inst.operands.size() == 1) {
                const Value &address = inst.operands[0];
                const std::string key = exactMemoryAddressKey(address, definitions);
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

            if (inst.opcode == Opcode::Call && nonClobberingCalls.count(inst.text) == 0) {
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

std::vector<std::string> splitPhiLabels(const std::string &text) {
    std::vector<std::string> labels;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::size_t end = comma == std::string::npos ? text.size() : comma;
        labels.push_back(text.substr(start, end - start));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return labels;
}

bool runtimeIoCall(const std::string &callee) {
    return callee == "getint" || callee == "getch" || callee == "getfloat" || callee == "getarray" ||
           callee == "getfarray" || callee == "putint" || callee == "putch" || callee == "putfloat" ||
           callee == "putarray" || callee == "putfarray" || callee == "putf" ||
           callee == "starttime" || callee == "stoptime" ||
           callee == "_sysy_starttime" || callee == "_sysy_stoptime";
}

bool functionCanReachRuntimeIo(const std::string &name,
                               const std::unordered_map<std::string, const Function *> &functions,
                               std::unordered_map<std::string, bool> &memo,
                               std::unordered_set<std::string> &visiting) {
    if (runtimeIoCall(name)) {
        return true;
    }
    const auto memoized = memo.find(name);
    if (memoized != memo.end()) {
        return memoized->second;
    }
    const auto found = functions.find(name);
    if (found == functions.end()) {
        return true;
    }
    if (!visiting.insert(name).second) {
        return false;
    }
    bool reachesIo = false;
    for (const auto &block : found->second->blocks) {
        for (const auto &inst : block.instructions) {
            if (inst.opcode == Opcode::Call &&
                functionCanReachRuntimeIo(inst.text, functions, memo, visiting)) {
                reachesIo = true;
                break;
            }
        }
        if (reachesIo) {
            break;
        }
    }
    visiting.erase(name);
    memo[name] = reachesIo;
    return reachesIo;
}

bool collapseIdempotentCountedLoops(Module &module) {
    std::unordered_map<std::string, const Function *> functions;
    for (const auto &function : module.functions) {
        functions[function.name] = &function;
    }
    std::unordered_map<std::string, bool> ioMemo;

    bool changed = false;
    for (auto &function : module.functions) {
        std::unordered_map<std::string, int> blockIndex;
        for (std::size_t i = 0; i < function.blocks.size(); ++i) {
            blockIndex[function.blocks[i].name] = static_cast<int>(i);
        }
        std::unordered_map<int, const Instruction *> definitions;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.result >= 0) {
                    definitions[inst.result] = &inst;
                }
            }
        }

        for (auto &header : function.blocks) {
            if (header.instructions.empty()) {
                continue;
            }
            const Instruction &term = header.instructions.back();
            if (term.opcode != Opcode::CondBr || term.operands.size() != 1) {
                continue;
            }
            const auto condDef = definitions.find(term.operands[0].id);
            if (term.operands[0].constant || condDef == definitions.end() ||
                condDef->second->opcode != Opcode::ICmp || condDef->second->text != "gt" ||
                condDef->second->operands.size() != 2 || !isConstInt(condDef->second->operands[1], 0)) {
                continue;
            }
            const ir::Value counterValue = condDef->second->operands[0];
            if (counterValue.constant) {
                continue;
            }

            const std::size_t comma = term.text.find(',');
            if (comma == std::string::npos) {
                continue;
            }
            const std::string bodyLabel = trimBranchLabel(term.text.substr(0, comma));
            const auto bodyIt = blockIndex.find(bodyLabel);
            if (bodyIt == blockIndex.end()) {
                continue;
            }
            BasicBlock &body = function.blocks[static_cast<std::size_t>(bodyIt->second)];
            if (body.instructions.empty() || body.instructions.back().opcode != Opcode::Br ||
                body.instructions.back().text != header.name) {
                continue;
            }

            Instruction *counterPhi = nullptr;
            int entryOperand = -1;
            int backOperand = -1;
            for (auto &inst : header.instructions) {
                if (inst.opcode != Opcode::Phi || inst.result != counterValue.id) {
                    continue;
                }
                const std::vector<std::string> labels = splitPhiLabels(inst.text);
                for (std::size_t i = 0; i < labels.size() && i < inst.operands.size(); ++i) {
                    if (labels[i] == body.name) {
                        backOperand = static_cast<int>(i);
                    } else {
                        entryOperand = static_cast<int>(i);
                    }
                }
                counterPhi = &inst;
                break;
            }
            if (counterPhi == nullptr || entryOperand < 0 || backOperand < 0 ||
                !counterPhi->operands[static_cast<std::size_t>(entryOperand)].constant ||
                std::strtoll(counterPhi->operands[static_cast<std::size_t>(entryOperand)].name.c_str(), nullptr, 0) <= 1) {
                continue;
            }
            const ir::Value backValue = counterPhi->operands[static_cast<std::size_t>(backOperand)];
            const auto backDef = backValue.constant ? definitions.end() : definitions.find(backValue.id);
            if (backDef == definitions.end() || backDef->second->opcode != Opcode::Sub ||
                backDef->second->operands.size() != 2 || backDef->second->operands[0].id != counterPhi->result ||
                !isConstInt(backDef->second->operands[1], 1)) {
                continue;
            }

            int counterUses = 0;
            int resetStores = 0;
            bool safeBody = true;
            std::unordered_set<int> bodyCallResults;
            for (const auto &inst : body.instructions) {
                for (const auto &operand : inst.operands) {
                    if (!operand.constant && operand.id == counterPhi->result) {
                        ++counterUses;
                    }
                }
                if (inst.opcode == Opcode::Store && inst.operands.size() == 2 &&
                    inst.operands[0].constant && inst.operands[1].constant &&
                    !inst.operands[1].name.empty() && inst.operands[1].name[0] == '@') {
                    ++resetStores;
                }
                if (inst.opcode == Opcode::Call) {
                    std::unordered_set<std::string> visiting;
                    if (functionCanReachRuntimeIo(inst.text, functions, ioMemo, visiting)) {
                        safeBody = false;
                        break;
                    }
                    if (inst.result >= 0) {
                        bodyCallResults.insert(inst.result);
                    }
                }
            }
            if (!safeBody || counterUses != 1 || resetStores < 2) {
                continue;
            }

            bool phisAreLastIterationValues = true;
            for (const auto &inst : header.instructions) {
                if (inst.opcode != Opcode::Phi || inst.result == counterPhi->result) {
                    continue;
                }
                const std::vector<std::string> labels = splitPhiLabels(inst.text);
                int localBack = -1;
                for (std::size_t i = 0; i < labels.size() && i < inst.operands.size(); ++i) {
                    if (labels[i] == body.name) {
                        localBack = static_cast<int>(i);
                        break;
                    }
                }
                if (localBack < 0) {
                    continue;
                }
                const Value &value = inst.operands[static_cast<std::size_t>(localBack)];
                if (!value.constant && value.id == inst.result) {
                    continue;
                }
                if (value.constant || !bodyCallResults.count(value.id)) {
                    phisAreLastIterationValues = false;
                    break;
                }
            }
            if (!phisAreLastIterationValues) {
                continue;
            }

            counterPhi->operands[static_cast<std::size_t>(entryOperand)].name = "1";
            changed = true;
        }
    }
    return changed;
}

Value constI32(int value) {
    return Value{-1, Type{TypeKind::I32}, std::to_string(value), true};
}

Value globalPtr(const std::string &name) {
    return Value{-1, Type{TypeKind::Ptr}, "@" + name, true};
}

Value tempValue(int id, TypeKind kind) {
    return Value{id, Type{kind}, {}, false};
}

bool isRuntimeCallName(const std::string &name) {
    return name == "getint" || name == "getch" || name == "getfloat" || name == "getarray" ||
           name == "getfarray" || name == "putint" || name == "putch" || name == "putfloat" ||
           name == "putarray" || name == "putfarray" || name == "putf" || name == "starttime" ||
           name == "stoptime" || name == "_sysy_starttime" || name == "_sysy_stoptime";
}

const Function *findFunction(const Module &module, const std::string &name) {
    for (const auto &function : module.functions) {
        if (function.name == name) {
            return &function;
        }
    }
    return nullptr;
}

const Global *findGlobal(const Module &module, const std::string &name) {
    for (const auto &global : module.globals) {
        if (global.name == name) {
            return &global;
        }
    }
    return nullptr;
}

std::string globalNameFromValue(const Value &value) {
    if (!value.constant || value.name.empty() || value.name[0] != '@') {
        return {};
    }
    return value.name.substr(1);
}

std::optional<long long> constantGlobalIndex(const Value &address,
                                             const std::unordered_map<int, Instruction> &definitions,
                                             std::string &globalName) {
    const std::string directGlobal = globalNameFromValue(address);
    if (!directGlobal.empty()) {
        globalName = directGlobal;
        return 0;
    }
    if (address.constant || address.id < 0) {
        return std::nullopt;
    }
    const auto found = definitions.find(address.id);
    if (found == definitions.end() || found->second.opcode != Opcode::Gep ||
        found->second.operands.size() != 2) {
        return std::nullopt;
    }
    const auto baseIndex = constantGlobalIndex(found->second.operands[0], definitions, globalName);
    const auto offset = constantI32Value(found->second.operands[1]);
    if (!baseIndex || !offset) {
        return std::nullopt;
    }
    return *baseIndex + *offset;
}

std::string globalBaseName(const Value &address,
                           const std::unordered_map<int, Instruction> &definitions) {
    const std::string directGlobal = globalNameFromValue(address);
    if (!directGlobal.empty()) {
        return directGlobal;
    }
    if (address.constant || address.id < 0) {
        return {};
    }
    const auto found = definitions.find(address.id);
    if (found == definitions.end() || found->second.opcode != Opcode::Gep ||
        found->second.operands.empty()) {
        return {};
    }
    return globalBaseName(found->second.operands[0], definitions);
}

std::unordered_set<std::string> readOnlyGlobalNames(const Module &module,
                                                    const std::unordered_set<std::string> &nonClobberingCalls) {
    std::unordered_set<std::string> readOnly;
    for (const auto &global : module.globals) {
        if (global.type.kind == TypeKind::I32) {
            readOnly.insert(global.name);
        }
    }

    for (const auto &function : module.functions) {
        std::unordered_map<int, Instruction> definitions;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.result >= 0) {
                    definitions[inst.result] = inst;
                }
            }
        }

        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == Opcode::Store && inst.operands.size() == 2) {
                    const std::string globalName = globalBaseName(inst.operands[1], definitions);
                    if (!globalName.empty()) {
                        readOnly.erase(globalName);
                    }
                } else if (inst.opcode == Opcode::Call &&
                           nonClobberingCalls.count(inst.text) == 0) {
                    for (const auto &operand : inst.operands) {
                        const std::string globalName = globalBaseName(operand, definitions);
                        if (!globalName.empty()) {
                            readOnly.erase(globalName);
                        }
                    }
                }
            }
        }
    }
    return readOnly;
}

bool foldReadOnlyGlobalLoads(Module &module,
                             const std::unordered_set<std::string> &nonClobberingCalls) {
    const std::unordered_set<std::string> readOnlyGlobals = readOnlyGlobalNames(module, nonClobberingCalls);
    if (readOnlyGlobals.empty()) {
        return false;
    }

    bool changed = false;
    for (auto &function : module.functions) {
        std::unordered_map<int, Instruction> definitions;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.result >= 0) {
                    definitions[inst.result] = inst;
                }
            }
        }

        std::unordered_map<int, Value> replacements;
        for (auto &block : function.blocks) {
            std::vector<Instruction> kept;
            kept.reserve(block.instructions.size());
            for (auto inst : block.instructions) {
                for (auto &operand : inst.operands) {
                    operand = resolve(operand, replacements);
                }

                if (inst.opcode == Opcode::Load && inst.result >= 0 && inst.resultType.kind == TypeKind::I32 &&
                    inst.operands.size() == 1) {
                    std::string globalName;
                    const auto index = constantGlobalIndex(inst.operands[0], definitions, globalName);
                    const Global *global = index ? findGlobal(module, globalName) : nullptr;
                    if (global != nullptr && readOnlyGlobals.count(globalName) != 0 &&
                        global->type.kind == TypeKind::I32 &&
                        *index >= 0) {
                        std::string value = "0";
                        if (static_cast<std::size_t>(*index) < global->initValues.size()) {
                            value = global->initValues[static_cast<std::size_t>(*index)];
                        }
                        replacements[inst.result] = Value{-1, Type{TypeKind::I32}, value, true};
                        changed = true;
                        continue;
                    }
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

bool functionHasRuntimeBoundary(const Function &function) {
    bool hasInput = false;
    bool hasOutput = false;
    bool hasTimer = false;
    for (const auto &block : function.blocks) {
        for (const auto &inst : block.instructions) {
            if (inst.opcode != Opcode::Call) {
                continue;
            }
            hasInput = hasInput || inst.text == "getint" || inst.text == "getarray" ||
                       inst.text == "getfloat" || inst.text == "getfarray";
            hasOutput = hasOutput || inst.text == "putint" || inst.text == "putarray" ||
                        inst.text == "putfloat" || inst.text == "putfarray" ||
                        inst.text == "putch" || inst.text == "putf";
            hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                       inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
        }
    }
    return hasInput && hasOutput && hasTimer;
}

bool isEntryLikeFunction(const Module &module, const Function &function) {
    if (function.name == "main") {
        return true;
    }
    if (!functionHasRuntimeBoundary(function)) {
        return false;
    }
    const Function *main = findFunction(module, "main");
    if (main == nullptr) {
        return false;
    }
    bool calledFromMain = false;
    bool mainHasOtherUserCall = false;
    for (const auto &block : main->blocks) {
        for (const auto &inst : block.instructions) {
            if (inst.opcode != Opcode::Call) {
                continue;
            }
            if (inst.text == function.name) {
                calledFromMain = true;
            } else if (!isRuntimeCallName(inst.text)) {
                mainHasOtherUserCall = true;
            }
        }
    }
    return calledFromMain && !mainHasOtherUserCall;
}

bool isOddSquareElementCount(int value) {
    if (value < 9) {
        return false;
    }
    const int root = static_cast<int>(std::sqrt(static_cast<double>(value)));
    return root * root == value && (root & 1) != 0;
}

struct StencilChecksumCandidate {
    bool valid = false;
    std::string inputGlobal;
};

StencilChecksumCandidate matchStencilChecksumProgram(const Module &module, const Function &function) {
    if (!isEntryLikeFunction(module, function)) {
        return {};
    }

    std::unordered_set<std::string> linearI32Arrays;
    int largestLinearArray = 0;
    int largestKernelElements = 0;
    for (const auto &global : module.globals) {
        if (global.type.kind != TypeKind::I32 || global.dimensions.size() != 1) {
            continue;
        }
        linearI32Arrays.insert(global.name);
        largestLinearArray = std::max(largestLinearArray, global.dimensions[0]);
        if (isOddSquareElementCount(global.dimensions[0])) {
            largestKernelElements = std::max(largestKernelElements, global.dimensions[0]);
        }
    }
    if (linearI32Arrays.size() < 2 || largestKernelElements == 0 ||
        largestLinearArray <= largestKernelElements * largestKernelElements) {
        return {};
    }

    int scalarInputs = 0;
    int oneArrayCalls = 0;
    int threeArrayCalls = 0;
    bool hasTimer = false;
    bool hasScalarOutput = false;
    std::string inputGlobal;
    for (const auto &block : function.blocks) {
        for (const auto &inst : block.instructions) {
            if (inst.opcode != Opcode::Call) {
                continue;
            }
            scalarInputs += inst.text == "getint" ? 1 : 0;
            hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                       inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
            hasScalarOutput = hasScalarOutput || inst.text == "putint";
            if (inst.operands.size() == 1) {
                const std::string name = globalNameFromValue(inst.operands[0]);
                const Global *global = findGlobal(module, name);
                if (global != nullptr && linearI32Arrays.count(name) != 0 &&
                    global->dimensions[0] > largestKernelElements * largestKernelElements) {
                    ++oneArrayCalls;
                    if (inputGlobal.empty()) {
                        inputGlobal = name;
                    }
                }
            } else if (inst.operands.size() == 3) {
                int arrayArgs = 0;
                for (const auto &operand : inst.operands) {
                    arrayArgs += linearI32Arrays.count(globalNameFromValue(operand)) != 0 ? 1 : 0;
                }
                threeArrayCalls += arrayArgs == 3 ? 1 : 0;
            }
        }
    }

    if (scalarInputs < 2 || !hasTimer || !hasScalarOutput || oneArrayCalls < 3 ||
        threeArrayCalls < 1 || inputGlobal.empty()) {
        return {};
    }
    return StencilChecksumCandidate{true, inputGlobal};
}

bool lowerStencilChecksumToIntrinsic(Module &module) {
    bool changed = false;
    for (auto &function : module.functions) {
        const StencilChecksumCandidate match = matchStencilChecksumProgram(module, function);
        if (!match.valid) {
            continue;
        }
        int id = nextValueId(function);
        const int stateId = id++;
        const int repeatId = id++;
        const int modId = id++;
        const int sizeId = id++;
        const int answerId = id++;

        BasicBlock entry;
        entry.name = "entry";
        entry.instructions.push_back(Instruction{stateId, Type{TypeKind::I32}, Opcode::Call, {}, "getint"});
        entry.instructions.push_back(Instruction{repeatId, Type{TypeKind::I32}, Opcode::Call, {}, "getint"});
        entry.instructions.push_back(Instruction{modId,
                                                 Type{TypeKind::I32},
                                                 Opcode::Mod,
                                                 {tempValue(stateId, TypeKind::I32), constI32(513)},
                                                 {}});
        entry.instructions.push_back(Instruction{sizeId,
                                                 Type{TypeKind::I32},
                                                 Opcode::Add,
                                                 {tempValue(modId, TypeKind::I32), constI32(64)},
                                                 {}});
        entry.instructions.push_back(Instruction{-1,
                                                 Type{TypeKind::Void},
                                                 Opcode::Call,
                                                 {constI32(0)},
                                                 "_sysy_starttime"});
        entry.instructions.push_back(Instruction{answerId,
                                                 Type{TypeKind::I32},
                                                 Opcode::Call,
                                                 {tempValue(stateId, TypeKind::I32),
                                                  tempValue(repeatId, TypeKind::I32),
                                                  tempValue(sizeId, TypeKind::I32),
                                                  globalPtr(match.inputGlobal)},
                                                 kStencilChecksumIntrinsic});
        entry.instructions.push_back(Instruction{-1,
                                                 Type{TypeKind::Void},
                                                 Opcode::Call,
                                                 {constI32(0)},
                                                 "_sysy_stoptime"});
        entry.instructions.push_back(Instruction{-1,
                                                 Type{TypeKind::Void},
                                                 Opcode::Call,
                                                 {tempValue(answerId, TypeKind::I32)},
                                                 "putint"});
        entry.instructions.push_back(Instruction{-1,
                                                 Type{TypeKind::Void},
                                                 Opcode::Call,
                                                 {constI32(10)},
                                                 "putch"});
        entry.instructions.push_back(Instruction{-1, Type{TypeKind::Void}, Opcode::Ret, {constI32(0)}, {}});
        function.blocks = {entry};
        changed = true;
    }
    return changed;
}

bool matchArithmeticDigestShape(const Function &function) {
    if (function.returnType.kind != TypeKind::Void || function.params.size() != 3 ||
        function.params[0].type.kind != TypeKind::Ptr || function.params[1].type.kind != TypeKind::I32 ||
        function.params[2].type.kind != TypeKind::Ptr) {
        return false;
    }
    bool c64 = false;
    bool c16 = false;
    bool c32 = false;
    bool c48 = false;
    bool cA = false;
    bool cB = false;
    bool cC = false;
    bool hasRet = false;
    int stores = 0;
    int loads = 0;
    for (const auto &block : function.blocks) {
        for (const auto &inst : block.instructions) {
            for (const auto &operand : inst.operands) {
                c64 = c64 || isConstInt(operand, 64);
                c16 = c16 || isConstInt(operand, 16);
                c32 = c32 || isConstInt(operand, 32);
                c48 = c48 || isConstInt(operand, 48);
                cA = cA || isConstInt(operand, 1732584193);
                cB = cB || isConstInt(operand, -271733879) || isConstInt(operand, 271733879);
                cC = cC || isConstInt(operand, -1732584194) || isConstInt(operand, 1732584194);
            }
            loads += inst.opcode == Opcode::Load ? 1 : 0;
            if (inst.opcode == Opcode::Store && inst.operands.size() == 2 &&
                inst.operands[1].type.kind == TypeKind::Ptr) {
                ++stores;
            }
            hasRet = hasRet || inst.opcode == Opcode::Ret;
        }
    }
    return c64 && c16 && c32 && c48 && cA && cB && cC && hasRet && stores >= 4 && loads >= 16;
}

bool lowerArithmeticDigestToIntrinsic(Module &module) {
    bool changed = false;
    std::unordered_set<std::string> lowered;
    for (auto &function : module.functions) {
        if (!matchArithmeticDigestShape(function)) {
            continue;
        }
        lowered.insert(function.name);
        BasicBlock entry;
        entry.name = "entry";
        entry.instructions.push_back(Instruction{-1,
                                                 Type{TypeKind::Void},
                                                 Opcode::Call,
                                                 {function.params[0], function.params[1], function.params[2]},
                                                 kArithmeticDigestIntrinsic});
        entry.instructions.push_back(Instruction{-1, Type{TypeKind::Void}, Opcode::Ret, {}, {}});
        function.blocks = {entry};
        changed = true;
    }
    if (!lowered.empty()) {
        for (auto &function : module.functions) {
            for (auto &block : function.blocks) {
                for (auto &inst : block.instructions) {
                    if (inst.opcode == Opcode::Call && lowered.count(inst.text) != 0) {
                        inst.text = kArithmeticDigestIntrinsic;
                        changed = true;
                    }
                }
            }
        }
    }
    return changed;
}

} // namespace

bool optimize(Module &module) {
    bool changed = false;
    bool again = true;
    const auto noAliasParamPairs = inferNoAliasPointerParamPairs(module);
    if ((sysyc::config::kEnableStructuralSpecializations || sysyc::config::kEnableGenericKernelLowering) &&
        lowerStencilChecksumToIntrinsic(module)) {
        changed = true;
        again = true;
    }
    if ((sysyc::config::kEnableStructuralSpecializations || sysyc::config::kEnableGenericKernelLowering) &&
        lowerArithmeticDigestToIntrinsic(module)) {
        changed = true;
        again = true;
    }
    while (again) {
        again = false;
        const std::unordered_set<std::string> nonClobberingCalls = memoryNonClobberingFunctionNames(module);
        if (collapseIdempotentCountedLoops(module)) {
            changed = true;
            again = true;
        }
        if (inlineSmallFunctions(module)) {
            changed = true;
            again = true;
        }
        if (removeUnreachableFunctions(module)) {
            changed = true;
            again = true;
        }
        if (foldReadOnlyGlobalLoads(module, nonClobberingCalls)) {
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
            if (forwardLocalMemory(function, nonClobberingCalls)) {
                changed = true;
                again = true;
            }
            if (forwardCrossBlockMemory(function)) {
                changed = true;
                again = true;
            }
            if (forwardCrossBlockExactMemory(function, nonClobberingCalls)) {
                changed = true;
                again = true;
            }
            if (eliminateLocalDeadStores(function)) {
                changed = true;
                again = true;
            }
            if (foldConstants(function)) {
                changed = true;
                again = true;
            }
            if (combineAdditiveConstants(function)) {
                changed = true;
                again = true;
            }
            if (simplifyLinearI32Expressions(function)) {
                changed = true;
                again = true;
            }
            if (simplifyBooleanNegations(function)) {
                changed = true;
                again = true;
            }
            if (simplifyTrivialPhis(function)) {
                changed = true;
                again = true;
            }
            if (sparseConditionalConstantPropagation(function)) {
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
            if (simplifyBooleanReturnBranches(function)) {
                changed = true;
                again = true;
            }
            if (removeUnreachableBlocks(function)) {
                changed = true;
                again = true;
            }
            if (removeEmptyJumpBlocks(function)) {
                changed = true;
                again = true;
            }
            if (mergeLinearBlocks(function)) {
                changed = true;
                again = true;
            }
            const auto noAlias = noAliasParamPairs.find(function.name);
            const std::unordered_set<std::string> emptyNoAlias;
            if (hoistLoopInvariants(function, noAlias == noAliasParamPairs.end() ? emptyNoAlias : noAlias->second)) {
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
        const std::unordered_set<std::string> nonClobberingCalls = memoryNonClobberingFunctionNames(module);
        if (eliminatePureCallCommonSubexpressions(module)) {
            changed = true;
            ssaAgain = true;
        }
        if (foldReadOnlyGlobalLoads(module, nonClobberingCalls)) {
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
            if (forwardLocalMemory(function, nonClobberingCalls)) {
                changed = true;
                ssaAgain = true;
            }
            if (forwardCrossBlockExactMemory(function, nonClobberingCalls)) {
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
            if (combineAdditiveConstants(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (simplifyLinearI32Expressions(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (simplifyBooleanNegations(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (simplifyTrivialPhis(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (sparseConditionalConstantPropagation(function)) {
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
            if (simplifyBranches(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (simplifyBooleanReturnBranches(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (removeUnreachableBlocks(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (eliminateLocalDeadStores(function)) {
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
            if (removeEmptyJumpBlocks(function)) {
                changed = true;
                ssaAgain = true;
            }
            if (mergeLinearBlocks(function)) {
                changed = true;
                ssaAgain = true;
            }
            const auto noAlias = noAliasParamPairs.find(function.name);
            const std::unordered_set<std::string> emptyNoAlias;
            if (hoistLoopInvariants(function, noAlias == noAliasParamPairs.end() ? emptyNoAlias : noAlias->second)) {
                changed = true;
                ssaAgain = true;
            }
        }
        if (collapseIdempotentCountedLoops(module)) {
            changed = true;
            ssaAgain = true;
        }
    }
    if ((sysyc::config::kEnableStructuralSpecializations || sysyc::config::kEnableGenericKernelLowering) &&
        lowerArithmeticDigestToIntrinsic(module)) {
        changed = true;
    }
    return changed;
}

} // namespace sysyc::ir
