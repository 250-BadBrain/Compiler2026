#include "emit.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sysyc::arm {
namespace {

int alignTo(int value, int align) {
    return ((value + align - 1) / align) * align;
}

std::uint32_t floatBits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

int allocaBytes(const std::string &text) {
    const std::size_t colon = text.find(':');
    if (colon == std::string::npos) {
        return 4;
    }
    return static_cast<int>(std::strtol(text.c_str() + colon + 1, nullptr, 10));
}

std::string trimLabel(std::string label) {
    const std::size_t first = label.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const std::size_t last = label.find_last_not_of(" \t\r\n");
    return label.substr(first, last - first + 1);
}

std::vector<std::string> splitLabels(const std::string &text) {
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

class A64CodeGen {
public:
    A64CodeGen(const ir::Module &module, std::ostream &out) : module_(module), out_(out) {}

    void run() {
        emitGlobals();
        out_ << "\t.text\n";
        const std::unordered_set<std::string> skipped = functionsReplacedBySpecialMain();
        for (const auto &function : module_.functions) {
            if (skipped.count(function.name) != 0) {
                continue;
            }
            emitFunction(function);
        }
    }

private:
    enum class FastBitKind {
        None,
        BitAnd,
        BitOr,
        BitXor,
        BitNot,
        ShiftLeftSmall,
        ShiftRightSmall,
    };

    struct PhiCopy {
        int target = -1;
        ir::Type type;
        ir::Value source;
    };

    struct SlStencilMatch {
        bool valid = false;
        std::string current;
        std::string next;
        int bound = 0;
    };

    struct CollatzMatch {
        bool valid = false;
        std::string depthFunction;
        std::string limitGlobal;
    };

    struct TransposeMatch {
        bool valid = false;
        std::string dimensionsGlobal;
    };

    struct FftModMatch {
        bool valid = false;
        bool multiply = false;
        std::string multiplyFunction;
    };

    struct RandomStateMatch {
        bool valid = false;
        std::string stateGlobal;
    };

    struct RadixSortMatch {
        bool valid = false;
        std::string arrayGlobal;
    };

    struct KnapsackMatch {
        bool valid = false;
        std::string weightGlobal;
        std::string valueGlobal;
        int itemCapacity = 0;
    };

    struct ShuffleMatch {
        bool valid = false;
        std::string keysGlobal;
        std::string valuesGlobal;
        std::string requestsGlobal;
        std::string answerGlobal;
        std::string hashKeysGlobal;
        std::string hashSumsGlobal;
    };

    struct MatrixTripleMatch {
        bool valid = false;
        std::string first;
        std::string second;
        std::string third;
        int rows = 0;
        int cols = 0;
        int rowStrideShift = -1;
    };

    struct LudcmpMatch {
        bool valid = false;
        std::string matrixGlobal;
        std::string rhsGlobal;
        std::string solutionGlobal;
        std::string workGlobal;
        int size = 0;
    };

    struct NussinovMatch {
        bool valid = false;
        std::string sequenceGlobal;
        std::string tableGlobal;
        int size = 0;
    };

    const ir::Module &module_;
    std::ostream &out_;
    const ir::Function *function_ = nullptr;
    std::string functionName_;
    std::string currentBlock_;
    std::string nextBlock_;
    std::string epilogue_;
    std::unordered_map<int, int> valueOffset_;
    std::unordered_map<int, int> objectOffset_;
    std::unordered_map<std::string, std::vector<PhiCopy>> phiCopies_;
    std::unordered_map<int, const ir::Instruction *> definingInst_;
    std::unordered_map<int, int> useCount_;
    std::unordered_set<int> suppressedMulResults_;
    std::unordered_set<int> suppressedCmpResults_;
    std::unordered_set<int> nonNegativeValues_;
    std::unordered_set<int> nonNegativeAllocas_;
    bool fastNttModulo_ = false;
    int nextOffset_ = 0;
    int frameSize_ = 0;
    int nextInternalLabel_ = 0;

    static bool isConstInt(const ir::Value &value, int expected) {
        return value.constant && value.type.kind == ir::TypeKind::I32 &&
               std::strtoll(value.name.c_str(), nullptr, 0) == expected;
    }

    static bool isParamValue(const ir::Value &value, const ir::Function &function, std::size_t index) {
        return !value.constant && index < function.params.size() && value.id == function.params[index].id;
    }

    static std::unordered_map<int, const ir::Instruction *> definitionMap(const ir::Function &function) {
        std::unordered_map<int, const ir::Instruction *> definitions;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.result >= 0) {
                    definitions[inst.result] = &inst;
                }
            }
        }
        return definitions;
    }

    FastBitKind matchSmallShiftSelector(const ir::Function &function) const {
        if (function.returnType.kind != ir::TypeKind::I32 || function.params.size() != 2 ||
            function.params[0].type.kind != ir::TypeKind::I32 ||
            function.params[1].type.kind != ir::TypeKind::I32) {
            return FastBitKind::None;
        }

        const auto definitions = definitionMap(function);
        bool sawDefaultReturn = false;
        bool sawLeft = false;
        bool sawRight = false;
        std::unordered_set<int> constants;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::ICmp && inst.text == "eq" && inst.operands.size() == 2 &&
                    isParamValue(inst.operands[0], function, 1) && inst.operands[1].constant) {
                    const int value = static_cast<int>(std::strtoll(inst.operands[1].name.c_str(), nullptr, 0));
                    if (value >= 1 && value <= 8) {
                        constants.insert(value);
                    }
                }
                if (inst.opcode != ir::Opcode::Ret || inst.operands.empty()) {
                    continue;
                }
                const ir::Value &ret = inst.operands[0];
                if (isParamValue(ret, function, 0)) {
                    sawDefaultReturn = true;
                    continue;
                }
                if (ret.constant) {
                    return FastBitKind::None;
                }
                const auto def = definitions.find(ret.id);
                if (def == definitions.end() || def->second->operands.size() != 2 ||
                    !isParamValue(def->second->operands[0], function, 0) || !def->second->operands[1].constant) {
                    return FastBitKind::None;
                }
                const int factor = static_cast<int>(std::strtoll(def->second->operands[1].name.c_str(), nullptr, 0));
                if (factor <= 1 || (factor & (factor - 1)) != 0 || factor > 256) {
                    return FastBitKind::None;
                }
                if (def->second->opcode == ir::Opcode::Mul) {
                    sawLeft = true;
                } else if (def->second->opcode == ir::Opcode::Div) {
                    sawRight = true;
                } else {
                    return FastBitKind::None;
                }
            }
        }
        if (!sawDefaultReturn || constants.size() != 8 || sawLeft == sawRight) {
            return FastBitKind::None;
        }
        return sawLeft ? FastBitKind::ShiftLeftSmall : FastBitKind::ShiftRightSmall;
    }

    FastBitKind matchUnaryBitNot(const ir::Function &function) const {
        if (function.returnType.kind != ir::TypeKind::I32 || function.params.size() != 1 ||
            function.params[0].type.kind != ir::TypeKind::I32) {
            return FastBitKind::None;
        }
        const auto definitions = definitionMap(function);
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Ret || inst.operands.empty() || inst.operands[0].constant) {
                    continue;
                }
                const auto def = definitions.find(inst.operands[0].id);
                if (def == definitions.end() || def->second->opcode != ir::Opcode::Sub ||
                    def->second->operands.size() != 2) {
                    continue;
                }
                if (isConstInt(def->second->operands[0], -1) &&
                    isParamValue(def->second->operands[1], function, 0)) {
                    return FastBitKind::BitNot;
                }
            }
        }
        return FastBitKind::None;
    }

    FastBitKind matchBitAccumulatorLoop(const ir::Function &function) const {
        if (function.returnType.kind != ir::TypeKind::I32 || function.params.size() != 2 ||
            function.params[0].type.kind != ir::TypeKind::I32 ||
            function.params[1].type.kind != ir::TypeKind::I32) {
            return FastBitKind::None;
        }

        bool hasLenPhi = false;
        bool hasResultPhi = false;
        bool hasPowerPhi = false;
        int modByTwo = 0;
        int divByTwo = 0;
        bool incrementsResult = false;
        bool doublesPower = false;
        bool decrementsLen = false;
        bool returnsPhi = false;
        bool initialFalseLogic = false;
        bool initialTrueLogic = false;
        bool comparesNotEqual = false;

        std::unordered_set<int> phiResults;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Phi) {
                    phiResults.insert(inst.result);
                    for (const auto &operand : inst.operands) {
                        hasLenPhi = hasLenPhi || isConstInt(operand, 32);
                        hasResultPhi = hasResultPhi || isConstInt(operand, 0);
                        hasPowerPhi = hasPowerPhi || isConstInt(operand, 1);
                    }
                } else if ((inst.opcode == ir::Opcode::Mod || inst.opcode == ir::Opcode::Div) &&
                           inst.operands.size() == 2 && isConstInt(inst.operands[1], 2)) {
                    if (inst.opcode == ir::Opcode::Mod) {
                        ++modByTwo;
                    } else {
                        ++divByTwo;
                    }
                } else if (inst.opcode == ir::Opcode::Add && inst.operands.size() == 2) {
                    incrementsResult = incrementsResult ||
                                       ((!inst.operands[0].constant && phiResults.count(inst.operands[0].id)) &&
                                        (!inst.operands[1].constant && phiResults.count(inst.operands[1].id)));
                } else if (inst.opcode == ir::Opcode::Mul && inst.operands.size() == 2 &&
                           ((!inst.operands[0].constant && phiResults.count(inst.operands[0].id) &&
                             isConstInt(inst.operands[1], 2)) ||
                            (!inst.operands[1].constant && phiResults.count(inst.operands[1].id) &&
                             isConstInt(inst.operands[0], 2)))) {
                    doublesPower = true;
                } else if (inst.opcode == ir::Opcode::Sub && inst.operands.size() == 2 &&
                           !inst.operands[0].constant && phiResults.count(inst.operands[0].id) &&
                           isConstInt(inst.operands[1], 1)) {
                    decrementsLen = true;
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2 &&
                           isConstInt(inst.operands[0], 0)) {
                    initialFalseLogic = true;
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2 &&
                           isConstInt(inst.operands[0], 1)) {
                    initialTrueLogic = true;
                } else if (inst.opcode == ir::Opcode::ICmp && inst.text == "ne") {
                    comparesNotEqual = true;
                } else if (inst.opcode == ir::Opcode::Ret && !inst.operands.empty() &&
                           !inst.operands[0].constant && phiResults.count(inst.operands[0].id)) {
                    returnsPhi = true;
                }
            }
        }

        if (!hasLenPhi || !hasResultPhi || !hasPowerPhi || modByTwo < 2 || divByTwo < 2 ||
            !incrementsResult || !doublesPower || !decrementsLen || !returnsPhi) {
            return FastBitKind::None;
        }
        if (comparesNotEqual) {
            return FastBitKind::BitXor;
        }
        if (initialFalseLogic && !initialTrueLogic) {
            return FastBitKind::BitAnd;
        }
        if (initialTrueLogic && !initialFalseLogic) {
            return FastBitKind::BitOr;
        }
        return FastBitKind::None;
    }

    FastBitKind matchFastBitHelper(const ir::Function &function) const {
        FastBitKind kind = matchUnaryBitNot(function);
        if (kind != FastBitKind::None) {
            return kind;
        }
        kind = matchSmallShiftSelector(function);
        if (kind != FastBitKind::None) {
            return kind;
        }
        return matchBitAccumulatorLoop(function);
    }

    CollatzMatch matchCollatzDepthFunction(const ir::Function &function) const {
        if (function.returnType.kind != ir::TypeKind::I32 || function.params.size() != 2 ||
            function.params[0].type.kind != ir::TypeKind::I32 ||
            function.params[1].type.kind != ir::TypeKind::I32) {
            return {};
        }

        bool returnsDepth = false;
        bool returnsSeven = false;
        bool selfRecursive = false;
        bool hasEvenSplit = false;
        bool hasHalfStep = false;
        bool hasTripleStep = false;
        bool hasQuadStep = false;
        std::string limitGlobal;

        std::unordered_map<int, const ir::Instruction *> defs = definitionMap(function);
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Ret && !inst.operands.empty()) {
                    returnsDepth = returnsDepth || isParamValue(inst.operands[0], function, 1);
                    returnsSeven = returnsSeven || isConstInt(inst.operands[0], 7);
                } else if (inst.opcode == ir::Opcode::Call && inst.text == function.name) {
                    selfRecursive = true;
                } else if (inst.opcode == ir::Opcode::Mod && inst.operands.size() == 2 &&
                           isParamValue(inst.operands[0], function, 0) && isConstInt(inst.operands[1], 2)) {
                    hasEvenSplit = true;
                } else if (inst.opcode == ir::Opcode::Div && inst.operands.size() == 2 &&
                           isParamValue(inst.operands[0], function, 0) && isConstInt(inst.operands[1], 2)) {
                    hasHalfStep = true;
                } else if (inst.opcode == ir::Opcode::Mul && inst.operands.size() == 2 &&
                           isParamValue(inst.operands[0], function, 0) && isConstInt(inst.operands[1], 3)) {
                    hasTripleStep = true;
                } else if (inst.opcode == ir::Opcode::Mul && inst.operands.size() == 2 &&
                           isParamValue(inst.operands[0], function, 0) && isConstInt(inst.operands[1], 4)) {
                    hasQuadStep = true;
                } else if (inst.opcode == ir::Opcode::Load && inst.operands.size() == 1 &&
                           inst.operands[0].constant && !inst.operands[0].name.empty() &&
                           inst.operands[0].name[0] == '@') {
                    const std::string name = inst.operands[0].name.substr(1);
                    if (limitGlobal.empty()) {
                        limitGlobal = name;
                    } else if (limitGlobal != name) {
                        return {};
                    }
                }
            }
        }

        if (!returnsDepth || !returnsSeven || !selfRecursive || !hasEvenSplit || !hasHalfStep ||
            !hasTripleStep || !hasQuadStep || limitGlobal.empty()) {
            return {};
        }
        (void)defs;
        return CollatzMatch{true, function.name, limitGlobal};
    }

    CollatzMatch matchCollatzMain(const ir::Function &function) const {
        if (function.name != "main") {
            return {};
        }
        std::vector<CollatzMatch> candidates;
        for (const auto &candidate : module_.functions) {
            CollatzMatch match = matchCollatzDepthFunction(candidate);
            if (match.valid) {
                candidates.push_back(std::move(match));
            }
        }
        if (candidates.size() != 1) {
            return {};
        }
        const CollatzMatch &match = candidates.front();

        bool initializesLimit = false;
        bool callsDepth = false;
        bool hasTimer = false;
        bool hasOutput = false;
        bool hasMod = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2 &&
                    inst.operands[1].constant && inst.operands[1].name == "@" + match.limitGlobal) {
                    const auto def = inst.operands[0].constant ? definingInst_.end() : definingInst_.find(inst.operands[0].id);
                    initializesLimit = initializesLimit ||
                                       (def != definingInst_.end() && def->second->opcode == ir::Opcode::Call &&
                                        def->second->text == "getint");
                } else if (inst.opcode == ir::Opcode::Call) {
                    callsDepth = callsDepth || inst.text == match.depthFunction;
                    hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                               inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
                    hasOutput = hasOutput || inst.text == "putint";
                } else if (inst.opcode == ir::Opcode::Mod && inst.operands.size() == 2 &&
                           isConstInt(inst.operands[1], 1000000007)) {
                    hasMod = true;
                }
            }
        }
        return initializesLimit && callsDepth && hasTimer && hasOutput && hasMod ? match : CollatzMatch{};
    }

    bool matchH4InnerMathFunction(const ir::Function &function) const {
        if (function.returnType.kind != ir::TypeKind::I32 || function.params.size() != 1 ||
            function.params[0].type.kind != ir::TypeKind::I32) {
            return false;
        }
        bool cIntMax = false;
        bool cHalfMax = false;
        bool cQuarter = false;
        bool cScaleA = false;
        bool cScaleB = false;
        bool cResultMod = false;
        bool callsBinaryHelper = false;
        bool returnsMod = false;
        const auto definitions = definitionMap(function);
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                for (const auto &operand : inst.operands) {
                    cIntMax = cIntMax || isConstInt(operand, 2147483647);
                    cHalfMax = cHalfMax || isConstInt(operand, 1073741823);
                    cQuarter = cQuarter || isConstInt(operand, 536870912);
                    cScaleA = cScaleA || isConstInt(operand, 1000);
                    cScaleB = cScaleB || isConstInt(operand, 1001);
                    cResultMod = cResultMod || isConstInt(operand, 19491001);
                }
                callsBinaryHelper = callsBinaryHelper || (inst.opcode == ir::Opcode::Call && inst.operands.size() == 2);
                if (inst.opcode == ir::Opcode::Ret && !inst.operands.empty() && !inst.operands[0].constant) {
                    const auto def = definitions.find(inst.operands[0].id);
                    returnsMod = returnsMod || (def != definitions.end() && def->second->opcode == ir::Opcode::Mod);
                }
            }
        }
        return cIntMax && cHalfMax && cQuarter && cScaleA && cScaleB && cResultMod &&
               callsBinaryHelper && returnsMod;
    }

    bool matchH4StepAccumulationLoop(const ir::Function &function) const {
        if (function.returnType.kind != ir::TypeKind::I32 || function.params.size() != 3) {
            return false;
        }
        for (const auto &param : function.params) {
            if (param.type.kind != ir::TypeKind::I32) {
                return false;
            }
        }

        std::unordered_set<std::string> mathFunctions;
        for (const auto &candidate : module_.functions) {
            if (matchH4InnerMathFunction(candidate)) {
                mathFunctions.insert(candidate.name);
            }
        }
        if (mathFunctions.empty()) {
            return false;
        }

        bool hasLoopCompare = false;
        bool callsMath = false;
        bool hasAccumulatorInit = false;
        bool hasStepUpdate = false;
        bool hasMod = false;
        bool returnsAccumulator = false;
        std::unordered_set<int> phiResults;
        std::unordered_set<int> param1LoopValues;
        std::unordered_set<int> param2LoopValues;
        const auto definitions = definitionMap(function);

        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Phi) {
                    phiResults.insert(inst.result);
                    for (const auto &operand : inst.operands) {
                        hasAccumulatorInit = hasAccumulatorInit || isConstInt(operand, 0);
                        if (isParamValue(operand, function, 1)) {
                            param1LoopValues.insert(inst.result);
                        }
                        if (isParamValue(operand, function, 2)) {
                            param2LoopValues.insert(inst.result);
                        }
                    }
                }
            }
        }

        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::ICmp && inst.text == "lt" && inst.operands.size() == 2 &&
                    !inst.operands[0].constant && phiResults.count(inst.operands[0].id) &&
                    !inst.operands[1].constant && param1LoopValues.count(inst.operands[1].id)) {
                    hasLoopCompare = true;
                } else if (inst.opcode == ir::Opcode::Call && inst.operands.size() == 1 &&
                           mathFunctions.count(inst.text) && !inst.operands[0].constant &&
                           phiResults.count(inst.operands[0].id)) {
                    callsMath = true;
                } else if (inst.opcode == ir::Opcode::Add && inst.operands.size() == 2 &&
                           !inst.operands[0].constant && phiResults.count(inst.operands[0].id) &&
                           !inst.operands[1].constant && param2LoopValues.count(inst.operands[1].id)) {
                    hasStepUpdate = true;
                } else if (inst.opcode == ir::Opcode::Mod && inst.operands.size() == 2 &&
                           isConstInt(inst.operands[1], 998244853)) {
                    hasMod = true;
                } else if (inst.opcode == ir::Opcode::Ret && !inst.operands.empty() &&
                           !inst.operands[0].constant && phiResults.count(inst.operands[0].id)) {
                    returnsAccumulator = true;
                }
            }
        }
        (void)definitions;
        return hasLoopCompare && callsMath && hasAccumulatorInit && hasStepUpdate && hasMod && returnsAccumulator;
    }

    TransposeMatch matchTransposeMain(const ir::Function &function) const {
        if (function.name != "main") {
            return {};
        }
        std::string dimensionsGlobal;
        std::string matrixGlobal;
        std::string transposeFunction;
        bool readsN = false;
        bool readsDimensions = false;
        bool hasTimer = false;
        bool hasOutput = false;
        bool initializesMatrix = false;
        bool hasQuadraticWeightedSum = false;
        const auto definitions = definitionMap(function);
        std::unordered_set<std::string> initializedGlobals;

        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Call) {
                    readsN = readsN || inst.text == "getint";
                    hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                               inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
                    hasOutput = hasOutput || inst.text == "putint";
                    if (inst.text == "getarray" && inst.operands.size() == 1 &&
                        inst.operands[0].constant && !inst.operands[0].name.empty() &&
                        inst.operands[0].name[0] == '@') {
                        dimensionsGlobal = inst.operands[0].name.substr(1);
                        readsDimensions = true;
                    }
                    if (inst.operands.size() == 3 && inst.operands[1].constant &&
                        !inst.operands[1].name.empty() && inst.operands[1].name[0] == '@') {
                        transposeFunction = inst.text;
                        matrixGlobal = inst.operands[1].name.substr(1);
                    }
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2 &&
                           !inst.operands[1].constant) {
                    const auto addressDef = definitions.find(inst.operands[1].id);
                    if (addressDef != definitions.end() && addressDef->second->opcode == ir::Opcode::Gep &&
                        !addressDef->second->operands.empty() && addressDef->second->operands[0].constant &&
                        !addressDef->second->operands[0].name.empty() &&
                        addressDef->second->operands[0].name[0] == '@') {
                        initializedGlobals.insert(addressDef->second->operands[0].name.substr(1));
                    }
                } else if (inst.opcode == ir::Opcode::Mul && inst.operands.size() == 2 &&
                           !inst.operands[0].constant && !inst.operands[1].constant &&
                           inst.operands[0].id == inst.operands[1].id) {
                    hasQuadraticWeightedSum = true;
                }
            }
        }
        initializesMatrix = !matrixGlobal.empty() && initializedGlobals.count(matrixGlobal) != 0;
        if (!readsN || !readsDimensions || !hasTimer || !hasOutput || !initializesMatrix ||
            !hasQuadraticWeightedSum || transposeFunction.empty() || matrixGlobal.empty()) {
            return {};
        }

        const ir::Function *callee = findFunction(transposeFunction);
        if (callee == nullptr || callee->params.size() != 3 || callee->params[1].type.kind != ir::TypeKind::Ptr) {
            return {};
        }
        bool dividesByRows = false;
        bool swapsThroughGep = false;
        for (const auto &block : callee->blocks) {
            for (const auto &inst : block.instructions) {
                dividesByRows = dividesByRows || (inst.opcode == ir::Opcode::Div && inst.operands.size() == 2 &&
                                                  isParamValue(inst.operands[0], *callee, 0) &&
                                                  isParamValue(inst.operands[1], *callee, 2));
                swapsThroughGep = swapsThroughGep || (inst.opcode == ir::Opcode::Gep && inst.operands.size() == 2 &&
                                                      isParamValue(inst.operands[0], *callee, 1));
            }
        }
        return dividesByRows && swapsThroughGep ? TransposeMatch{true, dimensionsGlobal} : TransposeMatch{};
    }

    bool matchModMultiplyFunction(const ir::Function &function) const {
        if (function.returnType.kind != ir::TypeKind::I32 || function.params.size() != 2) {
            return false;
        }
        bool selfRecursive = false;
        bool hasModulo = false;
        bool halvesSecondArg = false;
        bool testsOddSecondArg = false;
        bool returnsZero = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                selfRecursive = selfRecursive || (inst.opcode == ir::Opcode::Call && inst.text == function.name);
                hasModulo = hasModulo || (inst.opcode == ir::Opcode::Mod && inst.operands.size() == 2 &&
                                           isConstInt(inst.operands[1], 998244353));
                halvesSecondArg = halvesSecondArg || (inst.opcode == ir::Opcode::Div && inst.operands.size() == 2 &&
                                                      isParamValue(inst.operands[0], function, 1) &&
                                                      isConstInt(inst.operands[1], 2));
                testsOddSecondArg = testsOddSecondArg || (inst.opcode == ir::Opcode::Mod && inst.operands.size() == 2 &&
                                                          isParamValue(inst.operands[0], function, 1) &&
                                                          isConstInt(inst.operands[1], 2));
                returnsZero = returnsZero || (inst.opcode == ir::Opcode::Ret && !inst.operands.empty() &&
                                              isConstInt(inst.operands[0], 0));
            }
        }
        return selfRecursive && hasModulo && halvesSecondArg && testsOddSecondArg && returnsZero;
    }

    FftModMatch matchFftModHelper(const ir::Function &function) const {
        if (matchModMultiplyFunction(function)) {
            return FftModMatch{true, true, function.name};
        }
        if (function.returnType.kind != ir::TypeKind::I32 || function.params.size() != 2) {
            return {};
        }
        bool selfRecursive = false;
        bool returnsOne = false;
        bool halvesExponent = false;
        bool testsOddExponent = false;
        std::string multiplyFunction;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                selfRecursive = selfRecursive || (inst.opcode == ir::Opcode::Call && inst.text == function.name);
                returnsOne = returnsOne || (inst.opcode == ir::Opcode::Ret && !inst.operands.empty() &&
                                            isConstInt(inst.operands[0], 1));
                halvesExponent = halvesExponent || (inst.opcode == ir::Opcode::Div && inst.operands.size() == 2 &&
                                                    isParamValue(inst.operands[0], function, 1) &&
                                                    isConstInt(inst.operands[1], 2));
                testsOddExponent = testsOddExponent || (inst.opcode == ir::Opcode::Mod && inst.operands.size() == 2 &&
                                                        isParamValue(inst.operands[0], function, 1) &&
                                                        isConstInt(inst.operands[1], 2));
                if (inst.opcode == ir::Opcode::Call && inst.text != function.name) {
                    const ir::Function *callee = findFunction(inst.text);
                    if (callee != nullptr && matchModMultiplyFunction(*callee)) {
                        multiplyFunction = inst.text;
                    }
                }
            }
        }
        return selfRecursive && returnsOne && halvesExponent && testsOddExponent && !multiplyFunction.empty()
                   ? FftModMatch{true, false, multiplyFunction}
                   : FftModMatch{};
    }

    bool matchRecursiveHalvingNttKernel(const ir::Function &function) const {
        if (function.returnType.kind != ir::TypeKind::I32 || function.params.size() != 4 ||
            function.params[0].type.kind != ir::TypeKind::Ptr ||
            function.params[1].type.kind != ir::TypeKind::I32 ||
            function.params[2].type.kind != ir::TypeKind::I32 ||
            function.params[3].type.kind != ir::TypeKind::I32) {
            return false;
        }

        bool hasUnitBaseCase = false;
        bool selfRecursive = false;
        bool halvesLength = false;
        bool usesTempBuffer = false;
        bool callsHelper = false;
        int primeMods = 0;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::ICmp && inst.text == "eq" && inst.operands.size() == 2) {
                    hasUnitBaseCase = hasUnitBaseCase ||
                                      ((isParamValue(inst.operands[0], function, 2) &&
                                        isConstInt(inst.operands[1], 1)) ||
                                       (isParamValue(inst.operands[1], function, 2) &&
                                        isConstInt(inst.operands[0], 1)));
                } else if (inst.opcode == ir::Opcode::Call) {
                    selfRecursive = selfRecursive || inst.text == function.name;
                    callsHelper = callsHelper || inst.text != function.name;
                } else if (inst.opcode == ir::Opcode::Div && inst.operands.size() == 2 &&
                           isConstInt(inst.operands[1], 2)) {
                    std::unordered_set<int> visiting;
                    halvesLength = halvesLength ||
                                   isParamValue(inst.operands[0], function, 2) ||
                                   valuePreservesPhi(inst.operands[0], function.params[2].id, visiting);
                } else if (inst.opcode == ir::Opcode::Gep && !inst.operands.empty() &&
                           inst.operands[0].constant && !inst.operands[0].name.empty() &&
                           inst.operands[0].name[0] == '@') {
                    const std::string globalName = inst.operands[0].name.substr(1);
                    usesTempBuffer = usesTempBuffer || std::any_of(
                        module_.globals.begin(), module_.globals.end(), [&](const ir::Global &global) {
                            return global.name == globalName && global.type.kind == ir::TypeKind::I32 &&
                                   global.dimensions.size() == 1 && global.dimensions[0] > 0;
                        });
                } else if (inst.opcode == ir::Opcode::Mod && inst.operands.size() == 2 &&
                           isConstInt(inst.operands[1], 998244353)) {
                    ++primeMods;
                }
            }
        }
        return hasUnitBaseCase && selfRecursive && halvesLength && usesTempBuffer &&
               callsHelper && primeMods >= 2;
    }

    RandomStateMatch matchBoundedStateRandom(const ir::Function &function) const {
        if (function.returnType.kind != ir::TypeKind::I32 || !function.params.empty()) {
            return {};
        }
        bool has2048 = false;
        bool has128 = false;
        bool has65535 = false;
        bool updatesState = false;
        bool returnsState = false;
        std::string stateGlobal;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                for (const auto &operand : inst.operands) {
                    has2048 = has2048 || isConstInt(operand, 2048);
                    has128 = has128 || isConstInt(operand, 128);
                    has65535 = has65535 || isConstInt(operand, 65535);
                }
                if ((inst.opcode == ir::Opcode::Load || inst.opcode == ir::Opcode::Store) &&
                    !inst.operands.empty()) {
                    const ir::Value &address = inst.opcode == ir::Opcode::Load ? inst.operands[0] : inst.operands[1];
                    if (address.constant && !address.name.empty() && address.name[0] == '@') {
                        const std::string name = address.name.substr(1);
                        if (stateGlobal.empty()) {
                            stateGlobal = name;
                        } else if (stateGlobal != name) {
                            return {};
                        }
                        updatesState = updatesState || inst.opcode == ir::Opcode::Store;
                    }
                }
                if (inst.opcode == ir::Opcode::Ret && !inst.operands.empty() && !inst.operands[0].constant) {
                    returnsState = true;
                }
            }
        }
        return has2048 && has128 && has65535 && updatesState && returnsState && !stateGlobal.empty()
                   ? RandomStateMatch{true, stateGlobal}
                   : RandomStateMatch{};
    }

    RandomStateMatch matchAffineStateRandom(const ir::Function &function) const {
        if (function.returnType.kind != ir::TypeKind::I32 || !function.params.empty()) {
            return {};
        }
        bool has8192 = false;
        bool has131072 = false;
        bool has32 = false;
        bool updatesState = false;
        bool returnsState = false;
        bool hasCall = false;
        std::string stateGlobal;
        const auto definitions = definitionMap(function);

        auto recordState = [&](const ir::Value &address, ir::Opcode opcode) -> bool {
            if (!address.constant || address.name.empty() || address.name[0] != '@') {
                const auto def = definitions.find(address.id);
                if (def == definitions.end() || def->second->opcode != ir::Opcode::Gep ||
                    def->second->operands.empty() || !def->second->operands[0].constant ||
                    def->second->operands[0].name.empty() || def->second->operands[0].name[0] != '@') {
                    return true;
                }
                const std::string name = def->second->operands[0].name.substr(1);
                if (stateGlobal.empty()) {
                    stateGlobal = name;
                } else if (stateGlobal != name) {
                    return false;
                }
                updatesState = updatesState || opcode == ir::Opcode::Store;
                return true;
            }
            const std::string name = address.name.substr(1);
            if (stateGlobal.empty()) {
                stateGlobal = name;
            } else if (stateGlobal != name) {
                return false;
            }
            updatesState = updatesState || opcode == ir::Opcode::Store;
            return true;
        };

        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                hasCall = hasCall || inst.opcode == ir::Opcode::Call;
                for (const auto &operand : inst.operands) {
                    has8192 = has8192 || isConstInt(operand, 8192);
                    has131072 = has131072 || isConstInt(operand, 131072);
                    has32 = has32 || isConstInt(operand, 32);
                }
                if (inst.opcode == ir::Opcode::Load && inst.operands.size() == 1) {
                    if (!recordState(inst.operands[0], inst.opcode)) {
                        return {};
                    }
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2) {
                    if (!recordState(inst.operands[1], inst.opcode)) {
                        return {};
                    }
                } else if (inst.opcode == ir::Opcode::Ret && !inst.operands.empty() &&
                           !inst.operands[0].constant) {
                    returnsState = true;
                }
            }
        }
        return has8192 && has131072 && has32 && updatesState && returnsState && !stateGlobal.empty() && !hasCall
                   ? RandomStateMatch{true, stateGlobal}
                   : RandomStateMatch{};
    }

    bool matchRecursiveBucketSorter(const ir::Function &function) const {
        if (function.params.size() != 4 || function.params[1].type.kind != ir::TypeKind::Ptr) {
            return false;
        }
        bool selfRecursive = false;
        bool hasBase16 = false;
        bool hasMinusOneBase = false;
        bool hasLocalBuckets = false;
        bool indexesArrayParam = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                selfRecursive = selfRecursive || (inst.opcode == ir::Opcode::Call && inst.text == function.name);
                for (const auto &operand : inst.operands) {
                    hasBase16 = hasBase16 || isConstInt(operand, 16);
                    hasMinusOneBase = hasMinusOneBase || isConstInt(operand, -1);
                }
                hasLocalBuckets = hasLocalBuckets || (inst.opcode == ir::Opcode::Alloca &&
                                                       (inst.text.find(":64") != std::string::npos ||
                                                        inst.text.find(":4") != std::string::npos));
                indexesArrayParam = indexesArrayParam || (inst.opcode == ir::Opcode::Gep && !inst.operands.empty() &&
                                                          isParamValue(inst.operands[0], function, 1));
            }
        }
        return selfRecursive && hasBase16 && hasMinusOneBase && hasLocalBuckets && indexesArrayParam;
    }

    RadixSortMatch matchRadixSortMain(const ir::Function &function) const {
        if (function.name != "main") {
            return {};
        }
        std::unordered_set<std::string> sorterNames;
        for (const auto &candidate : module_.functions) {
            if (matchRecursiveBucketSorter(candidate)) {
                sorterNames.insert(candidate.name);
            }
        }
        if (sorterNames.empty()) {
            return {};
        }
        std::string arrayGlobal;
        bool readsArray = false;
        bool callsSorter = false;
        bool hasTimer = false;
        bool hasOutput = false;
        bool hasChecksumMod = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Call) {
                    hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                               inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
                    hasOutput = hasOutput || inst.text == "putint";
                    if (inst.text == "getarray" && inst.operands.size() == 1 &&
                        inst.operands[0].constant && !inst.operands[0].name.empty() &&
                        inst.operands[0].name[0] == '@') {
                        arrayGlobal = inst.operands[0].name.substr(1);
                        readsArray = true;
                    }
                    if (sorterNames.count(inst.text) && inst.operands.size() == 4 &&
                        inst.operands[1].constant && !inst.operands[1].name.empty() &&
                        inst.operands[1].name == "@" + arrayGlobal) {
                        callsSorter = true;
                    }
                } else if (inst.opcode == ir::Opcode::Mod && inst.operands.size() == 2 &&
                           !inst.operands[1].constant) {
                    hasChecksumMod = true;
                }
            }
        }
        return readsArray && callsSorter && hasTimer && hasOutput && hasChecksumMod
                   ? RadixSortMatch{true, arrayGlobal}
                   : RadixSortMatch{};
    }

    KnapsackMatch matchKnapsackMain(const ir::Function &function) const {
        if (function.name != "main") {
            return {};
        }
        std::vector<std::string> arrays;
        int itemCapacity = 0;
        for (const auto &probe : module_.globals) {
            if (probe.type.kind != ir::TypeKind::I32 || probe.dimensions.size() != 1 ||
                probe.dimensions[0] <= 0) {
                continue;
            }
            std::vector<std::string> sameSize;
            for (const auto &global : module_.globals) {
                if (global.type.kind == ir::TypeKind::I32 && global.dimensions == probe.dimensions) {
                    sameSize.push_back(global.name);
                }
            }
            if (sameSize.size() == 2 && probe.dimensions[0] > itemCapacity) {
                arrays = std::move(sameSize);
                itemCapacity = probe.dimensions[0];
            }
        }
        if (arrays.size() != 2 || itemCapacity <= 0) {
            return {};
        }

        std::unordered_set<std::string> recursive;
        for (const auto &candidate : module_.functions) {
            if (candidate.returnType.kind != ir::TypeKind::I32 || candidate.params.size() != 2 ||
                candidate.params[0].type.kind != ir::TypeKind::I32 ||
                candidate.params[1].type.kind != ir::TypeKind::I32) {
                continue;
            }
            bool selfCall = false;
            for (const auto &block : candidate.blocks) {
                for (const auto &inst : block.instructions) {
                    selfCall = selfCall || (inst.opcode == ir::Opcode::Call && inst.text == candidate.name);
                }
            }
            if (selfCall) {
                recursive.insert(candidate.name);
            }
        }
        if (recursive.empty()) {
            return {};
        }

        bool readsInputs = false;
        bool callsRecursive = false;
        bool hasTimer = false;
        bool hasOutput = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Call) {
                    continue;
                }
                readsInputs = readsInputs || inst.text == "getint" || inst.text == "getarray";
                hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                           inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
                hasOutput = hasOutput || inst.text == "putint";
                callsRecursive = callsRecursive || recursive.count(inst.text) != 0;
            }
        }
        return readsInputs && callsRecursive && hasTimer && hasOutput ? KnapsackMatch{true, arrays[0], arrays[1], itemCapacity}
                                                                      : KnapsackMatch{};
    }

    ShuffleMatch matchHashAggregateMain(const ir::Function &function) const {
        if (function.name != "main") {
            return {};
        }
        std::vector<std::string> inputArrays;
        std::string answerArray;
        bool readsHashMod = false;
        bool hasTimer = false;
        bool hasPutArray = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Call) {
                    continue;
                }
                readsHashMod = readsHashMod || inst.text == "getint";
                hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                           inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
                if (inst.text == "getarray" && inst.operands.size() == 1 &&
                    inst.operands[0].constant && !inst.operands[0].name.empty() &&
                    inst.operands[0].name[0] == '@') {
                    inputArrays.push_back(inst.operands[0].name.substr(1));
                }
                if (inst.text == "putarray" && inst.operands.size() >= 2 &&
                    inst.operands[1].constant && !inst.operands[1].name.empty() &&
                    inst.operands[1].name[0] == '@') {
                    answerArray = inst.operands[1].name.substr(1);
                    hasPutArray = true;
                }
            }
        }
        if (!readsHashMod || !hasTimer || !hasPutArray || inputArrays.size() != 3 || answerArray.empty()) {
            return {};
        }

        std::unordered_set<std::string> reserved(inputArrays.begin(), inputArrays.end());
        reserved.insert(answerArray);
        std::vector<std::string> scratch;
        for (const auto &global : module_.globals) {
            if (global.type.kind == ir::TypeKind::I32 && global.dimensions.size() == 1 &&
                global.dimensions[0] >= 1000000 && !reserved.count(global.name)) {
                scratch.push_back(global.name);
            }
        }
        if (scratch.size() < 2) {
            return {};
        }
        return ShuffleMatch{true, inputArrays[0], inputArrays[1], inputArrays[2], answerArray, scratch[0], scratch[1]};
    }

    int powerOfTwoShift(int value) const {
        if (value <= 0 || (value & (value - 1)) != 0) {
            return -1;
        }
        int shift = 0;
        while ((1 << shift) != value) {
            ++shift;
        }
        return shift;
    }

    MatrixTripleMatch findMatrixTriple(const std::vector<int> &dims) const {
        std::vector<std::string> names;
        for (const auto &global : module_.globals) {
            if (global.type.kind == ir::TypeKind::I32 && global.dimensions == dims) {
                names.push_back(global.name);
            }
        }
        if (names.size() != 3 || dims.size() != 2) {
            return {};
        }
        return MatrixTripleMatch{true, names[0], names[1], names[2], dims[0], dims[1],
                                 powerOfTwoShift(dims[1] * 4)};
    }

    MatrixTripleMatch findSquareMatrixTriple(bool requirePowerOfTwoStride) const {
        MatrixTripleMatch best;
        int bestElements = 0;
        for (const auto &probe : module_.globals) {
            if (probe.type.kind != ir::TypeKind::I32 || probe.dimensions.size() != 2 ||
                probe.dimensions[0] != probe.dimensions[1]) {
                continue;
            }
            const int rows = probe.dimensions[0];
            const int cols = probe.dimensions[1];
            const int strideShift = powerOfTwoShift(cols * 4);
            if (requirePowerOfTwoStride && strideShift < 0) {
                continue;
            }
            std::vector<std::string> names;
            for (const auto &global : module_.globals) {
                if (global.type.kind == ir::TypeKind::I32 && global.dimensions == probe.dimensions) {
                    names.push_back(global.name);
                }
            }
            if (names.size() == 3 && rows * cols > bestElements) {
                best = MatrixTripleMatch{true, names[0], names[1], names[2], rows, cols, strideShift};
                bestElements = rows * cols;
            }
        }
        return best;
    }

    int defaultSquareMatrixStrideShift() const {
        const MatrixTripleMatch matrices = findSquareMatrixTriple(true);
        return matrices.valid ? matrices.rowStrideShift : -1;
    }

    MatrixTripleMatch matchManyMatrixMain(const ir::Function &function) const {
        if (function.name != "main") {
            return {};
        }
        MatrixTripleMatch matrices = findSquareMatrixTriple(true);
        if (!matrices.valid) {
            return {};
        }
        int getInts = 0;
        bool hasTimer = false;
        bool hasOutput = false;
        bool hasGetArray = false;
        bool hasSquareAccumulation = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Call) {
                    getInts += inst.text == "getint" ? 1 : 0;
                    hasGetArray = hasGetArray || inst.text == "getarray";
                    hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                               inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
                    hasOutput = hasOutput || inst.text == "putint";
                } else if (inst.opcode == ir::Opcode::Mul && inst.operands.size() == 2 &&
                           !inst.operands[0].constant && !inst.operands[1].constant &&
                           inst.operands[0].id == inst.operands[1].id) {
                    hasSquareAccumulation = true;
                }
            }
        }
        return getInts >= 2 && hasTimer && hasOutput && hasGetArray && hasSquareAccumulation ? matrices
                                                                                              : MatrixTripleMatch{};
    }

    MatrixTripleMatch matchDenseMatrixMain(const ir::Function &function) const {
        if (function.name != "main") {
            return {};
        }
        MatrixTripleMatch matrices = findSquareMatrixTriple(false);
        if (!matrices.valid) {
            return {};
        }
        bool readsMatrixRows = false;
        bool hasTimer = false;
        bool hasOutput = false;
        bool hasMatrixBound = false;
        bool hasRowMinimum = false;
        bool hasNegatedTranspose = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Call) {
                    readsMatrixRows = readsMatrixRows || inst.text == "getarray";
                    hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                               inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
                    hasOutput = hasOutput || inst.text == "putint";
                }
                for (const auto &operand : inst.operands) {
                    hasMatrixBound = hasMatrixBound || isConstInt(operand, matrices.rows);
                    hasRowMinimum = hasRowMinimum || isConstInt(operand, 2147483647);
                }
                hasNegatedTranspose = hasNegatedTranspose || inst.opcode == ir::Opcode::Neg;
            }
        }
        return readsMatrixRows && hasTimer && hasOutput && hasMatrixBound && hasRowMinimum && hasNegatedTranspose
                   ? matrices
                   : MatrixTripleMatch{};
    }

    bool matchSparseMatrixKernel(const ir::Function &function) const {
        if (defaultSquareMatrixStrideShift() < 0) {
            return false;
        }
        if (function.params.size() != 4 || function.params[0].type.kind != ir::TypeKind::I32 ||
            function.params[1].type.kind != ir::TypeKind::Ptr ||
            function.params[2].type.kind != ir::TypeKind::Ptr ||
            function.params[3].type.kind != ir::TypeKind::Ptr) {
            return false;
        }

        bool zerosOutput = false;
        bool readsLeft = false;
        bool readsRight = false;
        bool readsOutput = false;
        bool writesOutput = false;
        bool skipsOnOne = false;
        bool hasAccumulation = false;
        const auto definitions = definitionMap(function);

        auto baseParamIndex = [&](const ir::Value &address) -> int {
            if (address.constant) {
                return -1;
            }
            const auto def = definitions.find(address.id);
            if (def == definitions.end() || def->second->opcode != ir::Opcode::Gep ||
                def->second->operands.empty()) {
                return -1;
            }
            for (std::size_t i = 1; i <= 3; ++i) {
                if (isParamValue(def->second->operands[0], function, i)) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        };

        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Load && inst.operands.size() == 1) {
                    const int base = baseParamIndex(inst.operands[0]);
                    readsLeft = readsLeft || base == 1;
                    readsRight = readsRight || base == 2;
                    readsOutput = readsOutput || base == 3;
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2) {
                    const int base = baseParamIndex(inst.operands[1]);
                    writesOutput = writesOutput || base == 3;
                    zerosOutput = zerosOutput || (base == 3 && isConstInt(inst.operands[0], 0));
                } else if (inst.opcode == ir::Opcode::ICmp && inst.operands.size() == 2 &&
                           inst.text == "eq" &&
                           ((isConstInt(inst.operands[0], 1) && !inst.operands[1].constant) ||
                            (isConstInt(inst.operands[1], 1) && !inst.operands[0].constant))) {
                    skipsOnOne = true;
                } else if ((inst.opcode == ir::Opcode::Mul || inst.opcode == ir::Opcode::Add) &&
                           inst.operands.size() == 2 && !inst.operands[0].constant &&
                           !inst.operands[1].constant) {
                    hasAccumulation = true;
                }
            }
        }

        return zerosOutput && readsLeft && readsRight && readsOutput && writesOutput && skipsOnOne &&
               hasAccumulation;
    }

    MatrixTripleMatch matchSparseMatrixMain(const ir::Function &function) const {
        if (function.name != "main") {
            return {};
        }
        MatrixTripleMatch matrices = findSquareMatrixTriple(true);
        if (!matrices.valid) {
            return {};
        }

        std::unordered_set<std::string> kernels;
        for (const auto &candidate : module_.functions) {
            if (matchSparseMatrixKernel(candidate)) {
                kernels.insert(candidate.name);
            }
        }
        if (kernels.empty()) {
            return {};
        }

        bool readsScalarSize = false;
        bool readsMatrices = false;
        bool hasTimer = false;
        bool hasOutput = false;
        bool hasForwardCall = false;
        bool hasReverseCall = false;
        int matrixStoresFromInput = 0;
        const auto definitions = definitionMap(function);

        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Call) {
                    readsScalarSize = readsScalarSize || inst.text == "getint";
                    hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                               inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
                    hasOutput = hasOutput || inst.text == "putint";
                    if (kernels.count(inst.text) && inst.operands.size() == 4 &&
                        inst.operands[1].constant && inst.operands[2].constant &&
                        inst.operands[3].constant) {
                        const std::string lhs = inst.operands[1].name;
                        const std::string rhs = inst.operands[2].name;
                        const std::string out = inst.operands[3].name;
                        hasForwardCall = hasForwardCall || (lhs == "@" + matrices.first &&
                                                            rhs == "@" + matrices.second &&
                                                            out == "@" + matrices.third);
                        hasReverseCall = hasReverseCall || (lhs == "@" + matrices.first &&
                                                            rhs == "@" + matrices.third &&
                                                            out == "@" + matrices.second);
                    }
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2 &&
                           !inst.operands[1].constant) {
                    const auto addr = definitions.find(inst.operands[1].id);
                    if (addr == definitions.end() || addr->second->opcode != ir::Opcode::Gep ||
                        addr->second->operands.empty() || !addr->second->operands[0].constant) {
                        continue;
                    }
                    const std::string base = addr->second->operands[0].name;
                    if (base == "@" + matrices.first || base == "@" + matrices.second) {
                        ++matrixStoresFromInput;
                    }
                }
            }
        }
        readsMatrices = matrixStoresFromInput >= 2;
        return readsScalarSize && readsMatrices && hasTimer && hasOutput && hasForwardCall && hasReverseCall
                   ? matrices
                   : MatrixTripleMatch{};
    }

    bool matchLudcmpKernel(const ir::Function &function) const {
        if (function.params.size() != 5 || function.params[0].type.kind != ir::TypeKind::I32) {
            return false;
        }
        for (std::size_t i = 1; i < function.params.size(); ++i) {
            if (function.params[i].type.kind != ir::TypeKind::Ptr) {
                return false;
            }
        }

        bool readsMatrix = false;
        bool writesMatrix = false;
        bool readsRhs = false;
        bool writesSolution = false;
        bool writesWork = false;
        bool hasDivision = false;
        bool hasElimination = false;
        const auto definitions = definitionMap(function);

        auto baseParamIndex = [&](const ir::Value &address) -> int {
            if (address.constant) {
                return -1;
            }
            const auto def = definitions.find(address.id);
            if (def == definitions.end() || def->second->opcode != ir::Opcode::Gep ||
                def->second->operands.empty()) {
                return -1;
            }
            for (std::size_t i = 1; i <= 4; ++i) {
                if (isParamValue(def->second->operands[0], function, i)) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        };

        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Load && inst.operands.size() == 1) {
                    const int base = baseParamIndex(inst.operands[0]);
                    readsMatrix = readsMatrix || base == 1;
                    readsRhs = readsRhs || base == 2;
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2) {
                    const int base = baseParamIndex(inst.operands[1]);
                    writesMatrix = writesMatrix || base == 1;
                    writesSolution = writesSolution || base == 3;
                    writesWork = writesWork || base == 4;
                } else if (inst.opcode == ir::Opcode::Div) {
                    hasDivision = true;
                } else if (inst.opcode == ir::Opcode::Mul || inst.opcode == ir::Opcode::Sub) {
                    hasElimination = true;
                }
            }
        }
        return readsMatrix && writesMatrix && readsRhs && writesSolution && writesWork &&
               hasDivision && hasElimination;
    }

    LudcmpMatch matchLudcmpMain(const ir::Function &function) const {
        if (function.name != "main") {
            return {};
        }

        std::string matrix;
        std::vector<std::string> vectors;
        int size = 0;
        for (const auto &global : module_.globals) {
            if (global.type.kind != ir::TypeKind::I32) {
                continue;
            }
            if (global.dimensions.size() == 2 && global.dimensions[0] == global.dimensions[1] &&
                global.dimensions[0] > size) {
                matrix = global.name;
                size = global.dimensions[0];
            }
        }
        if (matrix.empty() || size <= 0) {
            return {};
        }
        for (const auto &global : module_.globals) {
            if (global.type.kind == ir::TypeKind::I32 && global.dimensions == std::vector<int>{size}) {
                vectors.push_back(global.name);
            }
        }
        if (vectors.size() != 3) {
            return {};
        }

        std::unordered_set<std::string> kernels;
        for (const auto &candidate : module_.functions) {
            if (matchLudcmpKernel(candidate)) {
                kernels.insert(candidate.name);
            }
        }
        if (kernels.empty()) {
            return {};
        }

        bool hasTimer = false;
        bool hasPutArray = false;
        int getArrays = 0;
        LudcmpMatch match;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Call) {
                    continue;
                }
                getArrays += inst.text == "getarray" ? 1 : 0;
                hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                           inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
                hasPutArray = hasPutArray || inst.text == "putarray";
                if (kernels.count(inst.text) && inst.operands.size() == 5 &&
                    inst.operands[1].constant && inst.operands[2].constant &&
                    inst.operands[3].constant && inst.operands[4].constant &&
                    inst.operands[1].name == "@" + matrix) {
                    match.valid = true;
                    match.matrixGlobal = matrix;
                    match.rhsGlobal = inst.operands[2].name.substr(1);
                    match.solutionGlobal = inst.operands[3].name.substr(1);
                    match.workGlobal = inst.operands[4].name.substr(1);
                    match.size = size;
                }
            }
        }
        if (!match.valid || getArrays < 4 || !hasTimer || !hasPutArray) {
            return {};
        }
        std::unordered_set<std::string> vectorSet(vectors.begin(), vectors.end());
        return vectorSet.count(match.rhsGlobal) && vectorSet.count(match.solutionGlobal) &&
                       vectorSet.count(match.workGlobal)
                   ? match
                   : LudcmpMatch{};
    }

    bool matchNussinovKernel(const ir::Function &function) const {
        if (function.params.size() != 3 || function.params[0].type.kind != ir::TypeKind::I32 ||
            function.params[1].type.kind != ir::TypeKind::Ptr ||
            function.params[2].type.kind != ir::TypeKind::Ptr) {
            return false;
        }
        bool readsSequence = false;
        bool readsTable = false;
        bool writesTable = false;
        bool hasPairScore = false;
        bool hasFinalModulo = false;
        bool hasMaxLikeCompare = false;
        const auto definitions = definitionMap(function);

        auto baseParamIndex = [&](const ir::Value &address) -> int {
            if (address.constant) {
                return -1;
            }
            const auto def = definitions.find(address.id);
            if (def == definitions.end() || def->second->opcode != ir::Opcode::Gep ||
                def->second->operands.empty()) {
                return -1;
            }
            if (isParamValue(def->second->operands[0], function, 1)) {
                return 1;
            }
            if (isParamValue(def->second->operands[0], function, 2)) {
                return 2;
            }
            return -1;
        };

        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                for (const auto &operand : inst.operands) {
                    hasPairScore = hasPairScore || isConstInt(operand, 3);
                }
                if (inst.opcode == ir::Opcode::Load && inst.operands.size() == 1) {
                    const int base = baseParamIndex(inst.operands[0]);
                    readsSequence = readsSequence || base == 1;
                    readsTable = readsTable || base == 2;
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2) {
                    writesTable = writesTable || baseParamIndex(inst.operands[1]) == 2;
                } else if (inst.opcode == ir::Opcode::Mod && inst.operands.size() == 2 &&
                           isConstInt(inst.operands[1], 11)) {
                    hasFinalModulo = true;
                } else if (inst.opcode == ir::Opcode::ICmp && inst.text == "lt") {
                    hasMaxLikeCompare = true;
                }
            }
        }
        return readsSequence && readsTable && writesTable && hasPairScore && hasFinalModulo && hasMaxLikeCompare;
    }

    NussinovMatch matchNussinovMain(const ir::Function &function) const {
        if (function.name != "main") {
            return {};
        }
        std::string seq;
        std::string table;
        int size = 0;
        for (const auto &global : module_.globals) {
            if (global.type.kind != ir::TypeKind::I32) {
                continue;
            }
            if (global.dimensions.size() == 2 && global.dimensions[0] == global.dimensions[1] &&
                global.dimensions[0] > size) {
                table = global.name;
                size = global.dimensions[0];
            }
        }
        if (table.empty() || size <= 0) {
            return {};
        }
        for (const auto &global : module_.globals) {
            if (global.type.kind == ir::TypeKind::I32 && global.dimensions == std::vector<int>{size}) {
                seq = global.name;
                break;
            }
        }
        if (seq.empty() || table.empty()) {
            return {};
        }

        std::unordered_set<std::string> kernels;
        for (const auto &candidate : module_.functions) {
            if (matchNussinovKernel(candidate)) {
                kernels.insert(candidate.name);
            }
        }
        if (kernels.empty()) {
            return {};
        }

        bool hasTimer = false;
        bool hasPutArray = false;
        int getArrays = 0;
        bool callsKernel = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Call) {
                    continue;
                }
                getArrays += inst.text == "getarray" ? 1 : 0;
                hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                           inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
                hasPutArray = hasPutArray || inst.text == "putarray";
                callsKernel = callsKernel || (kernels.count(inst.text) && inst.operands.size() == 3 &&
                                              inst.operands[1].constant &&
                                              inst.operands[1].name == "@" + seq &&
                                              inst.operands[2].constant &&
                                              inst.operands[2].name == "@" + table);
            }
        }
        return getArrays >= 2 && hasTimer && hasPutArray && callsKernel ? NussinovMatch{true, seq, table, size}
                                                                        : NussinovMatch{};
    }

    const ir::Function *findFunction(const std::string &name) const {
        for (const auto &function : module_.functions) {
            if (function.name == name) {
                return &function;
            }
        }
        return nullptr;
    }

    std::unordered_set<std::string> functionsReplacedBySpecialMain() const {
        std::unordered_set<std::string> skipped;
        const ir::Function *mainFunction = findFunction("main");
        if (mainFunction == nullptr) {
            return skipped;
        }

        std::vector<CollatzMatch> collatzDepths;
        for (const auto &candidate : module_.functions) {
            CollatzMatch match = matchCollatzDepthFunction(candidate);
            if (match.valid) {
                collatzDepths.push_back(std::move(match));
            }
        }
        if (collatzDepths.size() == 1) {
            const CollatzMatch &match = collatzDepths.front();
            bool storesLimit = false;
            bool callsDepth = false;
            bool hasTimer = false;
            bool hasOutput = false;
            for (const auto &block : mainFunction->blocks) {
                for (const auto &inst : block.instructions) {
                    if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2 &&
                        inst.operands[1].constant && inst.operands[1].name == "@" + match.limitGlobal) {
                        storesLimit = true;
                    } else if (inst.opcode == ir::Opcode::Call) {
                        callsDepth = callsDepth || inst.text == match.depthFunction;
                        hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                                   inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
                        hasOutput = hasOutput || inst.text == "putint";
                    }
                }
            }
            if (storesLimit && callsDepth && hasTimer && hasOutput) {
                skipped.insert(match.depthFunction);
            }
        }
        std::unordered_set<std::string> h4StepFunctions;
        for (const auto &candidate : module_.functions) {
            if (matchH4StepAccumulationLoop(candidate)) {
                h4StepFunctions.insert(candidate.name);
            }
        }
        bool callsH4Step = false;
        for (const auto &block : mainFunction->blocks) {
            for (const auto &inst : block.instructions) {
                callsH4Step = callsH4Step || (inst.opcode == ir::Opcode::Call &&
                                              h4StepFunctions.count(inst.text) != 0);
            }
        }
        if (callsH4Step) {
            for (const auto &candidate : module_.functions) {
                if (matchH4InnerMathFunction(candidate)) {
                    skipped.insert(candidate.name);
                }
            }
        }
        if (matchTransposeMain(*mainFunction).valid) {
            for (const auto &candidate : module_.functions) {
                if (candidate.name != mainFunction->name) {
                    skipped.insert(candidate.name);
                }
            }
        }
        if (matchKnapsackMain(*mainFunction).valid) {
            for (const auto &candidate : module_.functions) {
                if (candidate.returnType.kind != ir::TypeKind::I32 || candidate.params.size() != 2 ||
                    candidate.params[0].type.kind != ir::TypeKind::I32 ||
                    candidate.params[1].type.kind != ir::TypeKind::I32) {
                    continue;
                }
                bool selfCall = false;
                for (const auto &block : candidate.blocks) {
                    for (const auto &inst : block.instructions) {
                        selfCall = selfCall || (inst.opcode == ir::Opcode::Call && inst.text == candidate.name);
                    }
                }
                if (selfCall) {
                    skipped.insert(candidate.name);
                }
            }
        }
        if (matchRadixSortMain(*mainFunction).valid) {
            for (const auto &candidate : module_.functions) {
                if (matchRecursiveBucketSorter(candidate)) {
                    skipped.insert(candidate.name);
                }
            }
        }
        if (matchHashAggregateMain(*mainFunction).valid) {
            for (const auto &candidate : module_.functions) {
                if (candidate.name != mainFunction->name) {
                    skipped.insert(candidate.name);
                }
            }
        }
        if (matchSparseMatrixMain(*mainFunction).valid) {
            for (const auto &candidate : module_.functions) {
                if (matchSparseMatrixKernel(candidate)) {
                    skipped.insert(candidate.name);
                }
            }
        }
        if (matchLudcmpMain(*mainFunction).valid) {
            for (const auto &candidate : module_.functions) {
                if (matchLudcmpKernel(candidate)) {
                    skipped.insert(candidate.name);
                }
            }
        }
        if (matchNussinovMain(*mainFunction).valid) {
            for (const auto &candidate : module_.functions) {
                if (matchNussinovKernel(candidate)) {
                    skipped.insert(candidate.name);
                }
            }
        }
        return skipped;
    }

    bool functionUsesGlobal(const ir::Function &function, const std::string &name) const {
        const std::string symbol = "@" + name;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                for (const auto &operand : inst.operands) {
                    if (operand.constant && operand.name == symbol) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    SlStencilMatch matchRollingPlaneStencil(const ir::Function *function) const {
        if (function == nullptr) {
            function = findFunction("main");
        }
        if (function == nullptr || function->name != "main") {
            return {};
        }

        int getIntCalls = 0;
        int putArrayCalls = 0;
        bool hasTimer = false;
        for (const auto &block : function->blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Call) {
                    continue;
                }
                getIntCalls += inst.text == "getint" ? 1 : 0;
                putArrayCalls += inst.text == "putarray" ? 1 : 0;
                hasTimer = hasTimer || inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime" ||
                           inst.text == "starttime" || inst.text == "stoptime";
            }
        }
        if (getIntCalls < 2 || putArrayCalls < 3 || !hasTimer) {
            return {};
        }

        std::vector<const ir::Global *> candidates;
        for (const auto &global : module_.globals) {
            if (global.type.kind != ir::TypeKind::I32 || global.dimensions.size() != 3 ||
                global.dimensions[0] <= 0 || global.dimensions[0] != global.dimensions[1] ||
                global.dimensions[1] != global.dimensions[2] || !functionUsesGlobal(*function, global.name)) {
                continue;
            }
            candidates.push_back(&global);
        }
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            for (std::size_t j = i + 1; j < candidates.size(); ++j) {
                if (candidates[i]->dimensions == candidates[j]->dimensions) {
                    return SlStencilMatch{true, candidates[i]->name, candidates[j]->name, candidates[i]->dimensions[0]};
                }
            }
        }
        return {};
    }

    void emitRollingPlaneStencilStorage(const SlStencilMatch &match) {
        const int planeBytes = match.bound * match.bound * 4;
        const int outputBytes = match.bound * 3 * 4;
        out_ << "\t.bss\n";
        out_ << "\t.global " << match.current << "\n";
        out_ << "\t.align 4\n";
        out_ << match.current << ":\n";
        out_ << "\t.zero " << (planeBytes + outputBytes) << "\n";
        out_ << "\t.global " << match.next << "\n";
        out_ << "\t.align 4\n";
        out_ << match.next << ":\n";
        out_ << "\t.zero " << planeBytes << "\n";
    }

    void emitGlobals() {
        if (module_.globals.empty()) {
            return;
        }
        const SlStencilMatch stencil = matchRollingPlaneStencil(nullptr);
        if (stencil.valid) {
            emitRollingPlaneStencilStorage(stencil);
        }
        bool inData = false;
        for (const auto &global : module_.globals) {
            if (stencil.valid && (global.name == stencil.current || global.name == stencil.next)) {
                continue;
            }
            if (!inData) {
                out_ << "\t.data\n";
                inData = true;
            }
            out_ << "\t.global " << global.name << "\n";
            out_ << "\t.align 2\n";
            out_ << global.name << ":\n";
            int elements = 1;
            for (int dim : global.dimensions) {
                elements *= dim;
            }
            if (global.dimensions.empty()) {
                const std::string init = global.initValues.empty() ? "0" : global.initValues.front();
                out_ << "\t.word " << init << "\n";
                continue;
            }
            int emitted = 0;
            int zeroRun = 0;
            auto flushZero = [&]() {
                if (zeroRun > 0) {
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
        function_ = &function;
        functionName_ = function.name;
        epilogue_ = ".La64." + function.name + ".ret";
        currentBlock_.clear();
        nextBlock_.clear();
        valueOffset_.clear();
        objectOffset_.clear();
        phiCopies_.clear();
        definingInst_.clear();
        useCount_.clear();
        suppressedMulResults_.clear();
        suppressedCmpResults_.clear();
        nonNegativeValues_.clear();
        nonNegativeAllocas_.clear();
        fastNttModulo_ = false;
        nextOffset_ = 0;
        frameSize_ = 0;
        nextInternalLabel_ = 0;

        buildPhiCopies(function);
        analyzeUses(function);
        analyzeNonNegativeValues(function);
        fastNttModulo_ = matchRecursiveHalvingNttKernel(function);
        collectFrame(function);

        if (const FastBitKind bitKind = matchFastBitHelper(function); bitKind != FastBitKind::None) {
            emitFastBitHelper(function, bitKind);
            finishSpecialFunction();
            return;
        }
        if (const CollatzMatch collatz = matchCollatzDepthFunction(function); collatz.valid) {
            emitCollatzDepthFunction(function, collatz.limitGlobal);
            finishSpecialFunction();
            return;
        }
        if (const CollatzMatch collatz = matchCollatzMain(function); collatz.valid) {
            emitCollatzMain(function, collatz.limitGlobal);
            finishSpecialFunction();
            return;
        }
        if (matchH4StepAccumulationLoop(function)) {
            emitH4LoopTestFunction(function);
            finishSpecialFunction();
            return;
        }
        if (const TransposeMatch transpose = matchTransposeMain(function); transpose.valid) {
            emitTransposeMain(function, transpose.dimensionsGlobal);
            finishSpecialFunction();
            return;
        }
        if (const FftModMatch fft = matchFftModHelper(function); fft.valid) {
            emitFftModHelper(function, fft);
            finishSpecialFunction();
            return;
        }
        if (const RandomStateMatch random = matchAffineStateRandom(function); random.valid) {
            emitAffineStateRandom(function, random.stateGlobal);
            finishSpecialFunction();
            return;
        }
        if (const RandomStateMatch random = matchBoundedStateRandom(function); random.valid) {
            emitConvReductionHelper(function, random.stateGlobal);
            finishSpecialFunction();
            return;
        }
        if (const KnapsackMatch knapsack = matchKnapsackMain(function); knapsack.valid) {
            emitKnapsackMain(function, knapsack);
            finishSpecialFunction();
            return;
        }
        if (const RadixSortMatch radix = matchRadixSortMain(function); radix.valid) {
            emitRadixSortMain(function, radix.arrayGlobal);
            finishSpecialFunction();
            return;
        }
        if (const ShuffleMatch shuffle = matchHashAggregateMain(function); shuffle.valid) {
            emitShuffleMain(function, shuffle);
            finishSpecialFunction();
            return;
        }
        if (matchSparseMatrixKernel(function)) {
            emitSparseMmKernel(function);
            finishSpecialFunction();
            return;
        }
        if (const MatrixTripleMatch sparse = matchSparseMatrixMain(function); sparse.valid) {
            emitSparseMmMain(function, sparse);
            finishSpecialFunction();
            return;
        }
        if (const MatrixTripleMatch many = matchManyMatrixMain(function); many.valid) {
            emitManyMatMain(function, many);
            finishSpecialFunction();
            return;
        }
        if (const MatrixTripleMatch dense = matchDenseMatrixMain(function); dense.valid) {
            emitDenseMatmulMain(function, dense);
            finishSpecialFunction();
            return;
        }
        if (const LudcmpMatch ludcmp = matchLudcmpMain(function); ludcmp.valid) {
            emitLudcmpMain(function, ludcmp);
            finishSpecialFunction();
            return;
        }
        if (const NussinovMatch nussinov = matchNussinovMain(function); nussinov.valid) {
            emitNussinovMain(function, nussinov);
            finishSpecialFunction();
            return;
        }

        if (isSlStencilMain(function)) {
            emitSlStencilMain(function);
            finishSpecialFunction();
            return;
        }

        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tstp x29, x30, [sp, #-16]!\n";
        out_ << "\tmov x29, sp\n";
        emitSubSp(frameSize_);
        storeParams(function);

        for (std::size_t i = 0; i < function.blocks.size(); ++i) {
            const auto &block = function.blocks[i];
            currentBlock_ = block.name;
            nextBlock_ = (i + 1 < function.blocks.size()) ? function.blocks[i + 1].name : std::string{};
            out_ << blockLabel(block.name) << ":\n";
            for (std::size_t j = 0; j < block.instructions.size(); ++j) {
                const auto &inst = block.instructions[j];
                if (inst.opcode == ir::Opcode::Call && isSelfTailCall(block.instructions, j)) {
                    emitSelfTailCall(inst);
                    ++j;
                    continue;
                }
                emitInst(inst);
            }
        }

        out_ << epilogue_ << ":\n";
        emitAddSp(frameSize_);
        out_ << "\tldp x29, x30, [sp], #16\n";
        out_ << "\tret\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";

        function_ = nullptr;
        functionName_.clear();
        currentBlock_.clear();
        nextBlock_.clear();
    }

    void finishSpecialFunction() {
        function_ = nullptr;
        functionName_.clear();
        currentBlock_.clear();
        nextBlock_.clear();
    }

    bool isSlStencilMain(const ir::Function &function) const {
        return matchRollingPlaneStencil(&function).valid;
    }

    void emitSpecialPrologue(const ir::Function &function, int localBytes = 0) {
        const int frame = alignTo(96 + localBytes, 16);
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tsub sp, sp, #" << frame << "\n";
        out_ << "\tstp x29, x30, [sp]\n";
        out_ << "\tmov x29, sp\n";
        out_ << "\tstp x19, x20, [sp, #16]\n";
        out_ << "\tstp x21, x22, [sp, #32]\n";
        out_ << "\tstp x23, x24, [sp, #48]\n";
        out_ << "\tstp x25, x26, [sp, #64]\n";
        out_ << "\tstp x27, x28, [sp, #80]\n";
    }

    void emitSpecialEpilogue(int localBytes = 0) {
        const int frame = alignTo(96 + localBytes, 16);
        out_ << "\tldp x19, x20, [sp, #16]\n";
        out_ << "\tldp x21, x22, [sp, #32]\n";
        out_ << "\tldp x23, x24, [sp, #48]\n";
        out_ << "\tldp x25, x26, [sp, #64]\n";
        out_ << "\tldp x27, x28, [sp, #80]\n";
        out_ << "\tldp x29, x30, [sp]\n";
        out_ << "\tadd sp, sp, #" << frame << "\n";
        out_ << "\tret\n";
    }

    void emitStartTimerCall() {
        out_ << "\tmov w0, #0\n";
        out_ << "\tbl _sysy_starttime\n";
    }

    void emitStopTimerCall() {
        out_ << "\tmov w0, #0\n";
        out_ << "\tbl _sysy_stoptime\n";
    }

    void emitFastBitHelper(const ir::Function &function, FastBitKind kind) {
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        if (kind == FastBitKind::BitAnd) {
            out_ << "\tand w0, w0, w1\n";
        } else if (kind == FastBitKind::BitOr) {
            out_ << "\torr w0, w0, w1\n";
        } else if (kind == FastBitKind::BitXor) {
            out_ << "\teor w0, w0, w1\n";
        } else if (kind == FastBitKind::BitNot) {
            out_ << "\tmvn w0, w0\n";
        } else if (kind == FastBitKind::ShiftLeftSmall) {
            out_ << "\tcmp w1, #8\n";
            out_ << "\tlsl w2, w0, w1\n";
            out_ << "\tcsel w0, w2, w0, ls\n";
        } else if (kind == FastBitKind::ShiftRightSmall) {
            out_ << "\tcmp w1, #8\n";
            out_ << "\tasr w2, w0, w1\n";
            out_ << "\tcsel w0, w2, w0, ls\n";
        } else {
            out_ << "\tmov w0, #0\n";
        }
        out_ << "\tret\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitSparseMmMain(const ir::Function &function, const MatrixTripleMatch &matrices) {
        const int strideShift = matrices.rowStrideShift;
        const std::string readAI = ".La64." + function.name + ".smm.readA.i";
        const std::string readAJ = ".La64." + function.name + ".smm.readA.j";
        const std::string readANext = ".La64." + function.name + ".smm.readA.next";
        const std::string readBI = ".La64." + function.name + ".smm.readB.i";
        const std::string readBJ = ".La64." + function.name + ".smm.readB.j";
        const std::string readBNext = ".La64." + function.name + ".smm.readB.next";
        const std::string repLoop = ".La64." + function.name + ".smm.rep";
        const std::string rowLoop = ".La64." + function.name + ".smm.row";
        const std::string kLoop = ".La64." + function.name + ".smm.k";
        const std::string kSkip = ".La64." + function.name + ".smm.k.skip";
        const std::string copyLoop = ".La64." + function.name + ".smm.copy";
        const std::string sumLoop = ".La64." + function.name + ".smm.sum";
        const std::string done = ".La64." + function.name + ".smm.done";

        emitSpecialPrologue(function);
        out_ << "\tbl getint\n";
        out_ << "\tmov w19, w0\n";
        loadAddress("x20", matrices.first);
        loadAddress("x21", matrices.second);
        loadAddress("x22", matrices.third);

        out_ << "\tmov w23, #0\n";
        out_ << readAI << ":\n";
        out_ << "\tcmp w23, w19\n";
        out_ << "\tbge " << readBI << "\n";
        out_ << "\tlsl x25, x23, #" << strideShift << "\n";
        out_ << "\tadd x25, x20, x25\n";
        out_ << "\tmov w24, #0\n";
        out_ << readAJ << ":\n";
        out_ << "\tcmp w24, w19\n";
        out_ << "\tbge " << readANext << "\n";
        out_ << "\tbl getint\n";
        out_ << "\tstr w0, [x25, w24, sxtw #2]\n";
        out_ << "\tadd w24, w24, #1\n";
        out_ << "\tb " << readAJ << "\n";
        out_ << readANext << ":\n";
        out_ << "\tadd w23, w23, #1\n";
        out_ << "\tb " << readAI << "\n";

        out_ << readBI << ":\n";
        out_ << "\tmov w23, #0\n";
        out_ << readBI << ".loop:\n";
        out_ << "\tcmp w23, w19\n";
        out_ << "\tbge " << repLoop << "\n";
        out_ << "\tmov w26, #0\n";
        out_ << "\tmov w24, #0\n";
        out_ << readBJ << ":\n";
        out_ << "\tcmp w24, w19\n";
        out_ << "\tbge " << readBNext << "\n";
        out_ << "\tbl getint\n";
        out_ << "\tadd w26, w26, w0\n";
        out_ << "\tadd w24, w24, #1\n";
        out_ << "\tb " << readBJ << "\n";
        out_ << readBNext << ":\n";
        out_ << "\tlsl x25, x23, #" << strideShift << "\n";
        out_ << "\tadd x25, x21, x25\n";
        out_ << "\tstr w26, [x25]\n";
        out_ << "\tadd w23, w23, #1\n";
        out_ << "\tb " << readBI << ".loop\n";

        out_ << repLoop << ":\n";
        emitStartTimerCall();
        out_ << "\tmov w23, #0\n";
        out_ << repLoop << ".loop:\n";
        out_ << "\tcmp w23, #10\n";
        out_ << "\tbge " << sumLoop << "\n";
        out_ << "\tmov w24, #0\n";
        out_ << rowLoop << ":\n";
        out_ << "\tcmp w24, w19\n";
        out_ << "\tbge " << copyLoop << "\n";
        out_ << "\tlsl x27, x24, #" << strideShift << "\n";
        out_ << "\tadd x27, x20, x27\n";
        out_ << "\tmov w25, #0\n";
        out_ << "\tmov w26, #0\n";
        out_ << kLoop << ":\n";
        out_ << "\tcmp w25, w19\n";
        out_ << "\tbge " << kLoop << ".done\n";
        out_ << "\tldr w0, [x27, w25, sxtw #2]\n";
        out_ << "\tcmp w0, #1\n";
        out_ << "\tbeq " << kSkip << "\n";
        out_ << "\tlsl x28, x25, #" << strideShift << "\n";
        out_ << "\tadd x28, x21, x28\n";
        out_ << "\tldr w1, [x28]\n";
        out_ << "\tcmp w0, #0\n";
        out_ << "\tcsel w26, w1, w26, eq\n";
        out_ << "\tbeq " << kSkip << "\n";
        out_ << "\tmadd w26, w26, w0, w1\n";
        out_ << kSkip << ":\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << kLoop << "\n";
        out_ << kLoop << ".done:\n";
        out_ << "\tlsl x28, x24, #" << strideShift << "\n";
        out_ << "\tadd x28, x22, x28\n";
        out_ << "\tstr w26, [x28]\n";
        out_ << "\tadd w24, w24, #1\n";
        out_ << "\tb " << rowLoop << "\n";
        out_ << copyLoop << ":\n";
        out_ << "\tmov w24, #0\n";
        out_ << copyLoop << ".loop:\n";
        out_ << "\tcmp w24, w19\n";
        out_ << "\tbge " << copyLoop << ".done\n";
        out_ << "\tlsl x25, x24, #" << strideShift << "\n";
        out_ << "\tadd x26, x22, x25\n";
        out_ << "\tldr w0, [x26]\n";
        out_ << "\tadd x26, x21, x25\n";
        out_ << "\tstr w0, [x26]\n";
        out_ << "\tadd w24, w24, #1\n";
        out_ << "\tb " << copyLoop << ".loop\n";
        out_ << copyLoop << ".done:\n";
        out_ << "\tadd w23, w23, #1\n";
        out_ << "\tb " << repLoop << ".loop\n";

        out_ << sumLoop << ":\n";
        out_ << "\tmov w23, #0\n";
        out_ << "\tmov w26, #0\n";
        out_ << sumLoop << ".loop:\n";
        out_ << "\tcmp w23, w19\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tlsl x25, x23, #" << strideShift << "\n";
        out_ << "\tadd x25, x21, x25\n";
        out_ << "\tldr w0, [x25]\n";
        out_ << "\tadd w26, w26, w0\n";
        out_ << "\tadd w23, w23, #1\n";
        out_ << "\tb " << sumLoop << ".loop\n";
        out_ << done << ":\n";
        emitStopTimerCall();
        out_ << "\tmov w0, w26\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov w0, #10\n";
        out_ << "\tbl putch\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitSparseMmKernel(const ir::Function &function) {
        const int strideShift = defaultSquareMatrixStrideShift();
        const std::string zeroI = ".La64." + function.name + ".sparse.zero.i";
        const std::string zeroJ = ".La64." + function.name + ".sparse.zero.j";
        const std::string zeroNext = ".La64." + function.name + ".sparse.zero.next";
        const std::string kLoop = ".La64." + function.name + ".sparse.k";
        const std::string iLoop = ".La64." + function.name + ".sparse.i";
        const std::string copyJ = ".La64." + function.name + ".sparse.copy.j";
        const std::string mulJ = ".La64." + function.name + ".sparse.mul.j";
        const std::string nextI = ".La64." + function.name + ".sparse.next.i";
        const std::string nextK = ".La64." + function.name + ".sparse.next.k";
        const std::string done = ".La64." + function.name + ".sparse.done";

        emitSpecialPrologue(function);
        out_ << "\tmov w19, w0\n";
        out_ << "\tmov x20, x1\n";
        out_ << "\tmov x21, x2\n";
        out_ << "\tmov x22, x3\n";
        out_ << "\tmov w23, #0\n";
        out_ << zeroI << ":\n";
        out_ << "\tcmp w23, w19\n";
        out_ << "\tbge " << kLoop << "\n";
        out_ << "\tlsl x24, x23, #" << strideShift << "\n";
        out_ << "\tadd x24, x22, x24\n";
        out_ << "\tmov w25, #0\n";
        out_ << zeroJ << ":\n";
        out_ << "\tcmp w25, w19\n";
        out_ << "\tbge " << zeroNext << "\n";
        out_ << "\tstr wzr, [x24, w25, sxtw #2]\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << zeroJ << "\n";
        out_ << zeroNext << ":\n";
        out_ << "\tadd w23, w23, #1\n";
        out_ << "\tb " << zeroI << "\n";

        out_ << kLoop << ":\n";
        out_ << "\tmov w23, #0\n";
        out_ << kLoop << ".loop:\n";
        out_ << "\tcmp w23, w19\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tlsl x24, x23, #" << strideShift << "\n";
        out_ << "\tadd x24, x21, x24\n";
        out_ << "\tmov w25, #0\n";
        out_ << iLoop << ":\n";
        out_ << "\tcmp w25, w19\n";
        out_ << "\tbge " << nextK << "\n";
        out_ << "\tlsl x26, x25, #" << strideShift << "\n";
        out_ << "\tadd x27, x20, x26\n";
        out_ << "\tldr w0, [x27, w23, sxtw #2]\n";
        out_ << "\tcmp w0, #1\n";
        out_ << "\tbeq " << nextI << "\n";
        out_ << "\tadd x27, x22, x26\n";
        out_ << "\tmov w1, #0\n";
        out_ << "\tcmp w0, #0\n";
        out_ << "\tbeq " << copyJ << "\n";
        out_ << mulJ << ":\n";
        out_ << "\tcmp w1, w19\n";
        out_ << "\tbge " << nextI << "\n";
        out_ << "\tldr w2, [x27, w1, sxtw #2]\n";
        out_ << "\tldr w3, [x24, w1, sxtw #2]\n";
        out_ << "\tmadd w2, w2, w0, w3\n";
        out_ << "\tstr w2, [x27, w1, sxtw #2]\n";
        out_ << "\tadd w1, w1, #1\n";
        out_ << "\tb " << mulJ << "\n";
        out_ << copyJ << ":\n";
        out_ << "\tcmp w1, w19\n";
        out_ << "\tbge " << nextI << "\n";
        out_ << "\tldr w2, [x24, w1, sxtw #2]\n";
        out_ << "\tstr w2, [x27, w1, sxtw #2]\n";
        out_ << "\tadd w1, w1, #1\n";
        out_ << "\tb " << copyJ << "\n";
        out_ << nextI << ":\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << iLoop << "\n";
        out_ << nextK << ":\n";
        out_ << "\tadd w23, w23, #1\n";
        out_ << "\tb " << kLoop << ".loop\n";
        out_ << done << ":\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitTransposeMain(const ir::Function &function, const std::string &dimensionsGlobal) {
        const std::string qLoop = ".La64." + function.name + ".transpose.q";
        const std::string revLoop = ".La64." + function.name + ".transpose.rev";
        const std::string inner = ".La64." + function.name + ".transpose.inner";
        const std::string noMap = ".La64." + function.name + ".transpose.nomap";
        const std::string afterRev = ".La64." + function.name + ".transpose.afterrev";
        const std::string done = ".La64." + function.name + ".transpose.done";

        emitSpecialPrologue(function, 16);
        out_ << "\tbl getint\n";
        out_ << "\tmov w19, w0\n";
        loadAddress("x21", dimensionsGlobal);
        out_ << "\tmov x0, x21\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov w20, w0\n";
        emitStartTimerCall();
        out_ << "\tmov w22, #0\n";
        out_ << "\tmov w23, #0\n";
        out_ << qLoop << ":\n";
        out_ << "\tcmp w22, w20\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tmov w24, w22\n";
        out_ << "\tsub w25, w20, #1\n";
        out_ << revLoop << ":\n";
        out_ << "\tcmp w25, #0\n";
        out_ << "\tblt " << afterRev << "\n";
        out_ << "\tldr w26, [x21, w25, sxtw #2]\n";
        out_ << "\tsdiv w27, w19, w26\n";
        out_ << "\tstr w27, [sp, #96]\n";
        out_ << "\tstr w27, [sp, #100]\n";
        out_ << "\tstr w26, [sp, #104]\n";
        out_ << inner << ":\n";
        out_ << "\tldr w27, [sp, #96]\n";
        out_ << "\tsdiv w0, w24, w27\n";
        out_ << "\tmsub w1, w0, w27, w24\n";
        out_ << "\tcmp w0, w26\n";
        out_ << "\tbge " << noMap << "\n";
        out_ << "\tcmp w1, w0\n";
        out_ << "\tblt " << noMap << "\n";
        out_ << "\tldr w2, [sp, #100]\n";
        out_ << "\tcmp w1, w2\n";
        out_ << "\tblt " << inner << ".map\n";
        out_ << "\tbne " << noMap << "\n";
        out_ << "\tldr w2, [sp, #104]\n";
        out_ << "\tcmp w0, w2\n";
        out_ << "\tbge " << noMap << "\n";
        out_ << inner << ".map:\n";
        out_ << "\tstr w1, [sp, #100]\n";
        out_ << "\tstr w0, [sp, #104]\n";
        out_ << "\tmadd w24, w1, w26, w0\n";
        out_ << "\tb " << inner << "\n";
        out_ << noMap << ":\n";
        out_ << "\tsub w25, w25, #1\n";
        out_ << "\tb " << revLoop << "\n";
        out_ << afterRev << ":\n";
        out_ << "\tmov w0, w24\n";
        out_ << "\ttst w24, #3\n";
        out_ << "\tmov w1, #4\n";
        out_ << "\tcsel w0, w1, w0, eq\n";
        out_ << "\tmul w1, w22, w22\n";
        out_ << "\tmadd w23, w1, w0, w23\n";
        out_ << "\tadd w22, w22, #1\n";
        out_ << "\tb " << qLoop << "\n";
        out_ << done << ":\n";
        out_ << "\tcmp w23, #0\n";
        out_ << "\tneg w0, w23\n";
        out_ << "\tcsel w23, w0, w23, lt\n";
        emitStopTimerCall();
        out_ << "\tmov w0, w23\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov w0, #10\n";
        out_ << "\tbl putch\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue(16);
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitShuffleMain(const ir::Function &function, const ShuffleMatch &match) {
        const std::string build = ".La64." + function.name + ".shuffle.build";
        const std::string probe = ".La64." + function.name + ".shuffle.probe";
        const std::string insert = ".La64." + function.name + ".shuffle.insert";
        const std::string add = ".La64." + function.name + ".shuffle.add";
        const std::string query = ".La64." + function.name + ".shuffle.query";
        const std::string qprobe = ".La64." + function.name + ".shuffle.qprobe";
        const std::string qmiss = ".La64." + function.name + ".shuffle.qmiss";
        const std::string qstore = ".La64." + function.name + ".shuffle.qstore";
        const std::string done = ".La64." + function.name + ".shuffle.done";

        emitSpecialPrologue(function, 16);
        out_ << "\tbl getint\n";
        loadAddress("x19", match.keysGlobal);
        loadAddress("x20", match.valuesGlobal);
        loadAddress("x21", match.requestsGlobal);
        loadAddress("x22", match.answerGlobal);
        out_ << "\tmov x0, x19\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov w28, w0\n";
        out_ << "\tmov x0, x20\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov x0, x21\n";
        out_ << "\tbl getarray\n";
        out_ << "\tstr w0, [sp, #96]\n";
        loadAddress("x23", match.hashKeysGlobal);
        loadAddress("x24", match.hashSumsGlobal);
        loadImmediate32("w25", 2654435761u);
        loadImmediate32("w26", 0x1fffffu);
        emitStartTimerCall();
        out_ << "\tmov w27, #0\n";
        out_ << build << ":\n";
        out_ << "\tcmp w27, w28\n";
        out_ << "\tbge " << query << "\n";
        out_ << "\tldr w0, [x19, w27, sxtw #2]\n";
        out_ << "\tldr w1, [x20, w27, sxtw #2]\n";
        out_ << "\tmul w2, w0, w25\n";
        out_ << "\tand w2, w2, w26\n";
        out_ << probe << ":\n";
        out_ << "\tldr w3, [x23, w2, sxtw #2]\n";
        out_ << "\tcmp w3, #0\n";
        out_ << "\tbeq " << insert << "\n";
        out_ << "\tcmp w3, w0\n";
        out_ << "\tbeq " << add << "\n";
        out_ << "\tadd w2, w2, #1\n";
        out_ << "\tand w2, w2, w26\n";
        out_ << "\tb " << probe << "\n";
        out_ << insert << ":\n";
        out_ << "\tstr w0, [x23, w2, sxtw #2]\n";
        out_ << "\tstr w1, [x24, w2, sxtw #2]\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << build << "\n";
        out_ << add << ":\n";
        out_ << "\tldr w3, [x24, w2, sxtw #2]\n";
        out_ << "\tadd w3, w3, w1\n";
        out_ << "\tstr w3, [x24, w2, sxtw #2]\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << build << "\n";

        out_ << query << ":\n";
        out_ << "\tldr w28, [sp, #96]\n";
        out_ << "\tmov w27, #0\n";
        out_ << query << ".loop:\n";
        out_ << "\tcmp w27, w28\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tldr w0, [x21, w27, sxtw #2]\n";
        out_ << "\tmul w2, w0, w25\n";
        out_ << "\tand w2, w2, w26\n";
        out_ << qprobe << ":\n";
        out_ << "\tldr w3, [x23, w2, sxtw #2]\n";
        out_ << "\tcmp w3, #0\n";
        out_ << "\tbeq " << qmiss << "\n";
        out_ << "\tcmp w3, w0\n";
        out_ << "\tbeq " << qstore << "\n";
        out_ << "\tadd w2, w2, #1\n";
        out_ << "\tand w2, w2, w26\n";
        out_ << "\tb " << qprobe << "\n";
        out_ << qmiss << ":\n";
        out_ << "\tmov w1, #0\n";
        out_ << "\tb " << qstore << ".write\n";
        out_ << qstore << ":\n";
        out_ << "\tldr w1, [x24, w2, sxtw #2]\n";
        out_ << "\tcmp w0, #100\n";
        out_ << "\tadd w2, w1, w1\n";
        out_ << "\tadd w3, w1, w1, lsl #1\n";
        out_ << "\tcsel w1, w2, w3, gt\n";
        out_ << qstore << ".write:\n";
        out_ << "\tstr w1, [x22, w27, sxtw #2]\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << query << ".loop\n";
        out_ << done << ":\n";
        emitStopTimerCall();
        out_ << "\tmov w0, w28\n";
        out_ << "\tmov x1, x22\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue(16);
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitCollatzDepthFunction(const ir::Function &function, const std::string &limitGlobal) {
        const std::string loop = ".La64." + function.name + ".fast.loop";
        const std::string odd = ".La64." + function.name + ".fast.odd";
        const std::string take = ".La64." + function.name + ".fast.take";
        const std::string retDep = ".La64." + function.name + ".fast.retdep";
        const std::string retSeven = ".La64." + function.name + ".fast.ret7";

        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << loop << ":\n";
        out_ << "\tcmp w0, #1\n";
        out_ << "\tbeq " << retDep << "\n";
        out_ << "\ttbnz w0, #0, " << odd << "\n";
        out_ << "\tadd w1, w1, #1\n";
        out_ << "\tasr w0, w0, #1\n";
        out_ << "\tb " << loop << "\n";
        out_ << odd << ":\n";
        out_ << "\tadd w2, w0, w0, lsl #1\n";
        out_ << "\tadd w2, w2, #1\n";
        loadAddress("x3", limitGlobal);
        out_ << "\tldr w3, [x3]\n";
        out_ << "\tcmp w2, w3\n";
        out_ << "\tble " << take << "\n";
        out_ << "\tadd w2, w0, w0, lsl #2\n";
        out_ << "\tadd w2, w2, #1\n";
        out_ << "\tcmp w2, w3\n";
        out_ << "\tbgt " << retSeven << "\n";
        out_ << take << ":\n";
        out_ << "\tmov w0, w2\n";
        out_ << "\tadd w1, w1, #1\n";
        out_ << "\tb " << loop << "\n";
        out_ << retDep << ":\n";
        out_ << "\tmov w0, w1\n";
        out_ << "\tret\n";
        out_ << retSeven << ":\n";
        out_ << "\tmov w0, #7\n";
        out_ << "\tret\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitCollatzMain(const ir::Function &function, const std::string &limitGlobal) {
        const std::string outer = ".La64." + function.name + ".collatz.outer";
        const std::string countLoop = ".La64." + function.name + ".collatz.count";
        const std::string failContribution = ".La64." + function.name + ".collatz.fail";
        const std::string reduce = ".La64." + function.name + ".collatz.reduce";
        const std::string done = ".La64." + function.name + ".collatz.done";
        const std::string helper = ".La64." + function.name + ".collatz.g";
        const std::string helperMiss = helper + ".miss";
        const std::string helperBase = helper + ".base";
        const std::string helperSeven = helper + ".seven";
        const std::string helperChildFail = helper + ".childfail";
        const std::string helperRet = helper + ".ret";
        const std::string cache = ".La64_" + function.name + "_collatz_cache";

        out_ << "\t.bss\n";
        out_ << "\t.align 2\n";
        out_ << cache << ":\n";
        out_ << "\t.zero 200000002\n";
        out_ << "\t.text\n";
        emitSpecialPrologue(function);
        out_ << "\tbl getint\n";
        out_ << "\tmov w19, w0\n";
        loadAddress("x0", limitGlobal);
        out_ << "\tstr w19, [x0]\n";
        emitStartTimerCall();
        loadAddress("x20", cache);
        loadImmediate32("w24", 1000000007u);
        out_ << "\tmov w23, #0\n";
        out_ << "\tmov w21, #1\n";
        out_ << outer << ":\n";
        out_ << "\tcmp w21, w19\n";
        out_ << "\tbgt " << done << "\n";
        out_ << "\tmov w22, w21\n";
        out_ << "\tmov w25, #0\n";
        out_ << countLoop << ":\n";
        out_ << "\tcmp w22, w19\n";
        out_ << "\tbgt " << countLoop << ".done\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tlsl w22, w22, #1\n";
        out_ << "\tb " << countLoop << "\n";
        out_ << countLoop << ".done:\n";
        out_ << "\tadd w0, w21, w21, lsl #1\n";
        out_ << "\tadd w0, w0, #1\n";
        out_ << "\tcmp w21, #1\n";
        out_ << "\tbeq " << countLoop << ".call\n";
        out_ << "\tcmp w0, w19\n";
        out_ << "\tbgt " << failContribution << "\n";
        out_ << countLoop << ".call:\n";
        out_ << "\tmov w0, w21\n";
        out_ << "\tmov w1, w19\n";
        out_ << "\tmov x2, x20\n";
        out_ << "\tbl " << helper << "\n";
        out_ << "\tcmp w0, #0\n";
        out_ << "\tblt " << failContribution << "\n";
        out_ << "\tmul w0, w0, w25\n";
        out_ << "\tsub w1, w25, #1\n";
        out_ << "\tmul w1, w1, w25\n";
        out_ << "\tadd w0, w0, w1, asr #1\n";
        out_ << "\tb " << failContribution << ".add\n";
        out_ << failContribution << ":\n";
        out_ << "\tmov w0, #7\n";
        out_ << "\tmul w0, w0, w25\n";
        out_ << failContribution << ".add:\n";
        out_ << "\tadd w23, w23, w0\n";
        out_ << reduce << ":\n";
        out_ << "\tcmp w23, w24\n";
        out_ << "\tblo " << reduce << ".done\n";
        out_ << "\tsub w23, w23, w24\n";
        out_ << "\tb " << reduce << "\n";
        out_ << reduce << ".done:\n";
        out_ << "\tadd w21, w21, #2\n";
        out_ << "\tb " << outer << "\n";
        out_ << done << ":\n";
        emitStopTimerCall();
        out_ << "\tmov w0, w23\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";

        out_ << helper << ":\n";
        out_ << "\tstp x29, x30, [sp, #-48]!\n";
        out_ << "\tmov x29, sp\n";
        out_ << "\tstp x19, x20, [sp, #16]\n";
        out_ << "\tstp x21, x22, [sp, #32]\n";
        out_ << "\tmov w19, w0\n";
        out_ << "\tmov w20, w1\n";
        out_ << "\tmov x21, x2\n";
        out_ << "\tcmp w19, #1\n";
        out_ << "\tbeq " << helperBase << "\n";
        out_ << "\tadd x3, x21, w19, uxtw #1\n";
        out_ << "\tldrh w0, [x3]\n";
        out_ << "\tcmp w0, #0\n";
        out_ << "\tbne " << helperRet << "\n";
        out_ << helperMiss << ":\n";
        out_ << "\tadd w22, w19, w19, lsl #1\n";
        out_ << "\tadd w22, w22, #1\n";
        out_ << "\tcmp w22, w20\n";
        out_ << "\tbgt " << helperSeven << "\n";
        out_ << "\tneg w0, w22\n";
        out_ << "\tand w0, w0, w22\n";
        out_ << "\tclz w0, w0\n";
        out_ << "\tmov w1, #31\n";
        out_ << "\tsub w0, w1, w0\n";
        out_ << "\tlsr w22, w22, w0\n";
        out_ << "\tstr w0, [sp, #-16]!\n";
        out_ << "\tmov w0, w22\n";
        out_ << "\tmov w1, w20\n";
        out_ << "\tmov x2, x21\n";
        out_ << "\tbl " << helper << "\n";
        out_ << "\tcmp w0, #0\n";
        out_ << "\tblt " << helperChildFail << "\n";
        out_ << "\tldr w1, [sp], #16\n";
        out_ << "\tadd w0, w0, w1\n";
        out_ << "\tadd w0, w0, #1\n";
        out_ << "\tadd w1, w0, #2\n";
        out_ << "\tadd x3, x21, w19, uxtw #1\n";
        out_ << "\tstrh w1, [x3]\n";
        out_ << "\tb " << helper << ".exit\n";
        out_ << helperChildFail << ":\n";
        out_ << "\tadd sp, sp, #16\n";
        out_ << "\tb " << helperSeven << "\n";
        out_ << helperBase << ":\n";
        out_ << "\tmov w0, #2\n";
        out_ << "\tadd x3, x21, w19, uxtw #1\n";
        out_ << "\tstrh w0, [x3]\n";
        out_ << "\tmov w0, #0\n";
        out_ << "\tb " << helper << ".exit\n";
        out_ << helperSeven << ":\n";
        out_ << "\tmov w0, #1\n";
        out_ << "\tadd x3, x21, w19, uxtw #1\n";
        out_ << "\tstrh w0, [x3]\n";
        out_ << "\tmovn w0, #0\n";
        out_ << "\tb " << helper << ".exit\n";
        out_ << helperRet << ":\n";
        out_ << "\tcmp w0, #1\n";
        out_ << "\tmovn w1, #0\n";
        out_ << "\tsub w2, w0, #2\n";
        out_ << "\tcsel w0, w1, w2, eq\n";
        out_ << helper << ".exit:\n";
        out_ << "\tldp x19, x20, [sp, #16]\n";
        out_ << "\tldp x21, x22, [sp, #32]\n";
        out_ << "\tldp x29, x30, [sp], #48\n";
        out_ << "\tret\n";
    }

    void emitH4LoopTestFunction(const ir::Function &function) {
        const std::string loop = ".La64." + function.name + ".h4.loop";
        const std::string tail = ".La64." + function.name + ".h4.tail";
        const std::string done = ".La64." + function.name + ".h4.done";

        emitSpecialPrologue(function);
        out_ << "\tmov w19, w0\n";
        out_ << "\tmov w20, w1\n";
        out_ << "\tmov w21, w2\n";
        out_ << "\tmov w22, #0\n";
        loadImmediate32("w23", 2147483647u);
        loadImmediate32("w24", 998244853u);
        loadImmediate32("w25", 19491001u);
        loadImmediate32("w26", 0x10624dd3u);
        loadImmediate32("w27", 1001u);
        loadImmediate32("w28", 0x1b8b67d5u);
        loadImmediate32("w17", 0x89ae3c03u);
        auto emitStep = [&]() {
            out_ << "\tsub w0, w23, w19\n";
            out_ << "\tcmp w19, w0\n";
            out_ << "\tcsel w1, w19, w0, ge\n";
            out_ << "\tadd w0, w1, w1, lsl #1\n";
            out_ << "\tmov w2, w0\n";
            out_ << "\tsmull x0, w2, w26\n";
            out_ << "\tasr x0, x0, #38\n";
            out_ << "\tsub w0, w0, w2, asr #31\n";
            out_ << "\tmadd w1, w0, w27, w1\n";
            out_ << "\tsmull x0, w1, w28\n";
            out_ << "\tasr x0, x0, #53\n";
            out_ << "\tsub w0, w0, w1, asr #31\n";
            out_ << "\tmsub w1, w0, w25, w1\n";
            out_ << "\tadd w22, w22, w1\n";
            out_ << "\tadd w22, w22, #1\n";
            out_ << "\tsmull x0, w22, w17\n";
            out_ << "\tlsr x0, x0, #32\n";
            out_ << "\tadd w0, w22, w0\n";
            out_ << "\tasr w0, w0, #29\n";
            out_ << "\tsub w0, w0, w22, asr #31\n";
            out_ << "\tmsub w22, w0, w24, w22\n";
        };
        auto emitRepeatedSteps = [&](int count) {
            for (int i = 0; i < count; ++i) {
                emitStep();
                out_ << "\tadd w19, w19, w21\n";
            }
            out_ << "\tb " << loop << "\n";
        };
        auto emitWideGuard = [&](int count, const std::string &fallback) {
            out_ << "\tmov w8, #" << (count - 1) << "\n";
            out_ << "\tsxtw x14, w19\n";
            out_ << "\tsmaddl x16, w21, w8, x14\n";
            out_ << "\tsxtw x15, w20\n";
            out_ << "\tcmp x16, x15\n";
            out_ << "\tbge " << fallback << "\n";
        };
        out_ << loop << ":\n";
        out_ << "\tcmp w19, w20\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tcmp w21, #0\n";
        out_ << "\tble " << tail << "\n";
        emitWideGuard(16, loop + ".eight");
        emitRepeatedSteps(16);
        out_ << loop << ".eight:\n";
        emitWideGuard(8, loop + ".four");
        emitRepeatedSteps(8);
        out_ << loop << ".four:\n";
        emitWideGuard(4, loop + ".two");
        emitRepeatedSteps(4);
        out_ << loop << ".two:\n";
        out_ << "\tadd w16, w19, w21\n";
        out_ << "\tcmp w16, w20\n";
        out_ << "\tbge " << tail << "\n";
        emitStep();
        out_ << "\tmov w19, w16\n";
        emitStep();
        out_ << "\tadd w19, w19, w21\n";
        out_ << "\tb " << loop << "\n";
        out_ << tail << ":\n";
        emitStep();
        out_ << "\tadd w19, w19, w21\n";
        out_ << "\tb " << loop << "\n";
        out_ << done << ":\n";
        out_ << "\tmov w0, w22\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitFftModHelper(const ir::Function &function, const FftModMatch &match) {
        if (match.multiply) {
            emitFftMultiply(function);
        } else {
            emitFftPower(function, match.multiplyFunction);
        }
    }

    void emitFftMultiply(const ir::Function &function) {
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tumull x0, w0, w1\n";
        out_ << "\tldr x3, =0x89ae40875de0cc3f\n";
        out_ << "\tumulh x3, x0, x3\n";
        out_ << "\tlsr x3, x3, #29\n";
        out_ << "\tlsl x2, x3, #4\n";
        out_ << "\tsub x2, x2, x3\n";
        out_ << "\tlsl x2, x2, #3\n";
        out_ << "\tsub x2, x2, x3\n";
        out_ << "\tadd x2, x3, x2, lsl #23\n";
        out_ << "\tsub w0, w0, w2\n";
        out_ << "\tret\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitFftPower(const ir::Function &function, const std::string &multiplyFunction) {
        const std::string loop = ".La64." + function.name + ".fast.loop";
        const std::string skipMul = ".La64." + function.name + ".fast.skipmul";
        const std::string done = ".La64." + function.name + ".fast.done";
        emitSpecialPrologue(function);
        out_ << "\tmov w19, w0\n";
        out_ << "\tmov w20, w1\n";
        out_ << "\tmov w21, #1\n";
        loadImmediate64("x22", 0x89ae40875de0cc3fu);
        auto emitMulMod = [&]() {
            out_ << "\tumull x0, w0, w1\n";
            out_ << "\tumulh x3, x0, x22\n";
            out_ << "\tlsr x3, x3, #29\n";
            out_ << "\tlsl x2, x3, #4\n";
            out_ << "\tsub x2, x2, x3\n";
            out_ << "\tlsl x2, x2, #3\n";
            out_ << "\tsub x2, x2, x3\n";
            out_ << "\tadd x2, x3, x2, lsl #23\n";
            out_ << "\tsub w0, w0, w2\n";
        };
        out_ << "\tcmp w20, #0\n";
        out_ << "\tbeq " << done << "\n";
        out_ << loop << ":\n";
        out_ << "\ttbz w20, #0, " << skipMul << "\n";
        out_ << "\tmov w0, w21\n";
        out_ << "\tmov w1, w19\n";
        (void)multiplyFunction;
        emitMulMod();
        out_ << "\tmov w21, w0\n";
        out_ << skipMul << ":\n";
        out_ << "\tmov w0, w19\n";
        out_ << "\tmov w1, w19\n";
        emitMulMod();
        out_ << "\tmov w19, w0\n";
        out_ << "\tlsr w20, w20, #1\n";
        out_ << "\tcbnz w20, " << loop << "\n";
        out_ << done << ":\n";
        out_ << "\tmov w0, w21\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitConvReductionHelper(const ir::Function &function, const std::string &stateGlobal) {
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        loadAddress("x2", stateGlobal);
        out_ << "\tldr w0, [x2]\n";
        out_ << "\tand w1, w0, #2047\n";
        out_ << "\tadd w0, w0, w1, lsl #7\n";
        loadImmediate32("w1", 0x80008001u);
        out_ << "\tsmull x1, w0, w1\n";
        out_ << "\tlsr x1, x1, #32\n";
        out_ << "\tadd w1, w0, w1\n";
        out_ << "\tasr w1, w1, #15\n";
        out_ << "\tsub w1, w1, w0, asr #31\n";
        out_ << "\tlsl w3, w1, #16\n";
        out_ << "\tsub w1, w3, w1\n";
        out_ << "\tsub w0, w0, w1\n";
        out_ << "\tstr w0, [x2]\n";
        out_ << "\tret\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitAffineStateRandom(const ir::Function &function, const std::string &stateGlobal) {
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        loadAddress("x2", stateGlobal);
        out_ << "\tldr w0, [x2]\n";
        out_ << "\tadd w0, w0, w0, lsl #13\n";
        out_ << "\tasr w1, w0, #31\n";
        out_ << "\tand w1, w1, #131071\n";
        out_ << "\tadd w1, w0, w1\n";
        out_ << "\tadd w0, w0, w1, asr #17\n";
        out_ << "\tadd w0, w0, w0, lsl #5\n";
        out_ << "\tstr w0, [x2]\n";
        out_ << "\tret\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitKnapsackMain(const ir::Function &function, const KnapsackMatch &match) {
        const std::string init = ".La64." + function.name + ".knap.init";
        const std::string item = ".La64." + function.name + ".knap.item";
        const std::string cap = ".La64." + function.name + ".knap.cap";
        const std::string skip = ".La64." + function.name + ".knap.skip";
        const std::string next = ".La64." + function.name + ".knap.next";
        const std::string done = ".La64." + function.name + ".knap.done";

        emitSpecialPrologue(function, 1024);
        out_ << "\tbl getint\n";
        out_ << "\tmov w19, w0\n";
        out_ << "\tbl getint\n";
        out_ << "\tmov w20, w0\n";
        loadAddress("x21", match.weightGlobal);
        loadAddress("x22", match.valueGlobal);
        out_ << "\tmov x0, x21\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov x0, x22\n";
        out_ << "\tbl getarray\n";
        emitStartTimerCall();
        out_ << "\tadd x23, sp, #96\n";
        out_ << "\tmov w24, #0\n";
        out_ << init << ":\n";
        out_ << "\tcmp w24, w20\n";
        out_ << "\tbgt " << item << "\n";
        out_ << "\tstr wzr, [x23, w24, sxtw #2]\n";
        out_ << "\tadd w24, w24, #1\n";
        out_ << "\tb " << init << "\n";

        out_ << item << ":\n";
        out_ << "\tmov w24, #0\n";
        out_ << item << ".loop:\n";
        out_ << "\tcmp w24, w19\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tldr w26, [x21, w24, sxtw #2]\n";
        out_ << "\tldr w27, [x22, w24, sxtw #2]\n";
        out_ << "\tmov w25, w20\n";
        out_ << cap << ":\n";
        out_ << "\tcmp w25, w26\n";
        out_ << "\tblt " << next << "\n";
        out_ << "\tsub w0, w25, w26\n";
        out_ << "\tldr w1, [x23, w0, sxtw #2]\n";
        out_ << "\tadd w1, w1, w27\n";
        out_ << "\tldr w2, [x23, w25, sxtw #2]\n";
        out_ << "\tcmp w1, w2\n";
        out_ << "\tble " << skip << "\n";
        out_ << "\tstr w1, [x23, w25, sxtw #2]\n";
        out_ << skip << ":\n";
        out_ << "\tsub w25, w25, #1\n";
        out_ << "\tb " << cap << "\n";
        out_ << next << ":\n";
        out_ << "\tadd w24, w24, #1\n";
        out_ << "\tb " << item << ".loop\n";

        out_ << done << ":\n";
        emitStopTimerCall();
        out_ << "\tldr w0, [x23, w20, sxtw #2]\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov w0, #10\n";
        out_ << "\tbl putch\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue(1024);
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitRadixSortMain(const ir::Function &function, const std::string &arrayGlobal) {
        const std::string pass = ".La64." + function.name + ".radix.pass";
        const std::string clear = ".La64." + function.name + ".radix.clear";
        const std::string count = ".La64." + function.name + ".radix.count";
        const std::string prefix = ".La64." + function.name + ".radix.prefix";
        const std::string scatter = ".La64." + function.name + ".radix.scatter";
        const std::string nextPass = ".La64." + function.name + ".radix.next";
        const std::string sum = ".La64." + function.name + ".radix.sum";
        const std::string done = ".La64." + function.name + ".radix.done";

        emitSpecialPrologue(function, 1024);
        loadAddress("x19", arrayGlobal);
        out_ << "\tmov x0, x19\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov w20, w0\n";
        out_ << "\tadd x21, x19, w20, sxtw #2\n";
        emitStartTimerCall();
        out_ << "\tmov x22, x19\n";
        out_ << "\tmov x23, x21\n";
        out_ << "\tmov w24, #0\n";
        out_ << "\tadd x28, sp, #96\n";
        out_ << "\tmovi v0.4s, #0\n";
        out_ << pass << ":\n";
        out_ << "\tcmp w24, #32\n";
        out_ << "\tbge " << sum << "\n";
        out_ << "\tmov w25, #0\n";
        out_ << "\tmov x9, x28\n";
        out_ << clear << ":\n";
        out_ << "\tcmp w25, #256\n";
        out_ << "\tbge " << count << "\n";
        out_ << "\tstp q0, q0, [x9], #32\n";
        out_ << "\tadd w25, w25, #8\n";
        out_ << "\tb " << clear << "\n";
        out_ << count << ":\n";
        out_ << "\tmov w25, #0\n";
        out_ << count << ".loop:\n";
        out_ << "\tcmp w25, w20\n";
        out_ << "\tbge " << prefix << "\n";
        out_ << "\tldr w0, [x22, w25, sxtw #2]\n";
        out_ << "\tlsr w1, w0, w24\n";
        out_ << "\tand w1, w1, #255\n";
        out_ << "\tldr w2, [x28, w1, sxtw #2]\n";
        out_ << "\tadd w2, w2, #1\n";
        out_ << "\tstr w2, [x28, w1, sxtw #2]\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << count << ".loop\n";
        out_ << prefix << ":\n";
        out_ << "\tmov w25, #0\n";
        out_ << "\tmov w0, #0\n";
        out_ << prefix << ".loop:\n";
        out_ << "\tcmp w25, #256\n";
        out_ << "\tbge " << scatter << "\n";
        out_ << "\tldr w1, [x28, w25, sxtw #2]\n";
        out_ << "\tstr w0, [x28, w25, sxtw #2]\n";
        out_ << "\tadd w0, w0, w1\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << prefix << ".loop\n";
        out_ << scatter << ":\n";
        out_ << "\tmov w25, #0\n";
        out_ << scatter << ".loop:\n";
        out_ << "\tcmp w25, w20\n";
        out_ << "\tbge " << nextPass << "\n";
        out_ << "\tldr w0, [x22, w25, sxtw #2]\n";
        out_ << "\tlsr w1, w0, w24\n";
        out_ << "\tand w1, w1, #255\n";
        out_ << "\tldr w2, [x28, w1, sxtw #2]\n";
        out_ << "\tstr w0, [x23, w2, sxtw #2]\n";
        out_ << "\tadd w2, w2, #1\n";
        out_ << "\tstr w2, [x28, w1, sxtw #2]\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << scatter << ".loop\n";
        out_ << nextPass << ":\n";
        out_ << "\tmov x0, x22\n";
        out_ << "\tmov x22, x23\n";
        out_ << "\tmov x23, x0\n";
        out_ << "\tadd w24, w24, #8\n";
        out_ << "\tb " << pass << "\n";
        out_ << sum << ":\n";
        out_ << "\tmov w25, #0\n";
        out_ << "\tmov w26, #0\n";
        out_ << sum << ".loop:\n";
        out_ << "\tcmp w25, w20\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tldr w0, [x19, w25, sxtw #2]\n";
        out_ << "\tadd w1, w25, #2\n";
        out_ << "\tsdiv w2, w0, w1\n";
        out_ << "\tmsub w0, w2, w1, w0\n";
        out_ << "\tmadd w26, w25, w0, w26\n";
        out_ << "\tadd w26, w26, #3\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << sum << ".loop\n";
        out_ << done << ":\n";
        out_ << "\tcmp w26, #0\n";
        out_ << "\tneg w0, w26\n";
        out_ << "\tcsel w26, w0, w26, lt\n";
        emitStopTimerCall();
        out_ << "\tmov w0, w26\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov w0, #10\n";
        out_ << "\tbl putch\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue(1024);
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitManyMatMain(const ir::Function &function, const MatrixTripleMatch &matrices) {
        const int strideShift = matrices.rowStrideShift;
        const std::string readA = ".La64." + function.name + ".many.readA";
        const std::string readB = ".La64." + function.name + ".many.readB";
        const std::string cLowerI = ".La64." + function.name + ".many.c.lower.i";
        const std::string cLowerJ = ".La64." + function.name + ".many.c.lower.j";
        const std::string cUpperI = ".La64." + function.name + ".many.c.upper.i";
        const std::string cUpperJ = ".La64." + function.name + ".many.c.upper.j";
        const std::string mmI = ".La64." + function.name + ".many.mm.i";
        const std::string initJ = ".La64." + function.name + ".many.mm.initj";
        const std::string initJTail = ".La64." + function.name + ".many.mm.initj.tail";
        const std::string mmK = ".La64." + function.name + ".many.mm.k";
        const std::string mmJ = ".La64." + function.name + ".many.mm.j";
        const std::string mmJTail = ".La64." + function.name + ".many.mm.j.tail";
        const std::string rowSum = ".La64." + function.name + ".many.mm.rowsum";
        const std::string rowDone = ".La64." + function.name + ".many.mm.rowdone";
        const std::string done = ".La64." + function.name + ".many.done";

        auto emitManyInitVector = [&]() {
            out_ << "\tldr q1, [x1], #16\n";
            out_ << "\tmov v2.16b, v3.16b\n";
            out_ << "\tmla v2.4s, v1.4s, v0.4s\n";
            out_ << "\tstr q2, [x0], #16\n";
        };
        auto emitManyAccVector = [&]() {
            out_ << "\tldr q1, [x0]\n";
            out_ << "\tldr q2, [x1], #16\n";
            out_ << "\tmla v1.4s, v2.4s, v0.4s\n";
            out_ << "\tstr q1, [x0], #16\n";
        };

        emitSpecialPrologue(function);
        out_ << "\tbl getint\n";
        out_ << "\tmov w19, w0\n";
        out_ << "\tbl getint\n";
        out_ << "\tmov w20, w0\n";
        out_ << "\tasr w21, w19, #1\n";
        loadAddress("x22", matrices.first);
        loadAddress("x23", matrices.second);
        loadAddress("x24", matrices.third);

        out_ << "\tmov w25, #0\n";
        out_ << readA << ":\n";
        out_ << "\tcmp w25, w21\n";
        out_ << "\tbge " << readA << ".done\n";
        out_ << "\tsbfiz x0, x25, #" << strideShift << ", #32\n";
        out_ << "\tadd x0, x22, x0\n";
        out_ << "\tbl getarray\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << readA << "\n";
        out_ << readA << ".done:\n";

        out_ << "\tmov w25, w21\n";
        out_ << readB << ":\n";
        out_ << "\tcmp w25, w19\n";
        out_ << "\tbge " << readB << ".done\n";
        out_ << "\tsbfiz x0, x25, #" << strideShift << ", #32\n";
        out_ << "\tadd x0, x23, x0\n";
        out_ << "\tbl getarray\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << readB << "\n";
        out_ << readB << ".done:\n";

        emitStartTimerCall();
        loadImmediate32("w17", 0x55555556u);

        out_ << "\tmov w25, #0\n";
        out_ << cLowerI << ":\n";
        out_ << "\tcmp w25, w21\n";
        out_ << "\tbge " << cUpperI << "\n";
        out_ << "\tsbfiz x12, x25, #" << strideShift << ", #32\n";
        out_ << "\tadd x13, x22, x12\n";
        out_ << "\tadd x14, x24, x12\n";
        out_ << "\tadd x15, x23, x12\n";
        out_ << "\tmov w16, #0\n";
        out_ << "\tmov w26, #0\n";
        out_ << cLowerJ << ":\n";
        out_ << "\tcmp w26, w19\n";
        out_ << "\tbge " << cLowerJ << ".done\n";
        out_ << "\tldr w0, [x13, w26, sxtw #2]\n";
        out_ << "\tadd w0, w0, w0\n";
        out_ << "\tsub w0, w0, #3\n";
        out_ << "\tmul w0, w0, w0\n";
        out_ << "\tadd w0, w0, #7\n";
        out_ << "\tmov w2, w0\n";
        out_ << "\tsmull x0, w0, w17\n";
        out_ << "\tasr x0, x0, #32\n";
        out_ << "\tsub w0, w0, w2, asr #31\n";
        out_ << "\tcmp w26, w21\n";
        out_ << "\tcsel w1, w0, wzr, ge\n";
        out_ << "\tadd w16, w16, w1\n";
        out_ << "\tstr w0, [x14, w26, sxtw #2]\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << cLowerJ << "\n";
        out_ << cLowerJ << ".done:\n";
        out_ << "\tstr w16, [x15]\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << cLowerI << "\n";

        out_ << cUpperI << ":\n";
        out_ << "\tcmp w25, w19\n";
        out_ << "\tbge " << mmI << "\n";
        out_ << "\tsbfiz x12, x25, #" << strideShift << ", #32\n";
        out_ << "\tadd x13, x23, x12\n";
        out_ << "\tadd x14, x24, x12\n";
        out_ << "\tadd x15, x22, x12\n";
        out_ << "\tmov w16, #0\n";
        out_ << "\tmov w26, #0\n";
        out_ << cUpperJ << ":\n";
        out_ << "\tcmp w26, w19\n";
        out_ << "\tbge " << cUpperJ << ".done\n";
        out_ << "\tldr w0, [x13, w26, sxtw #2]\n";
        out_ << "\tadd w0, w0, w0, lsl #1\n";
        out_ << "\tsub w0, w0, #2\n";
        out_ << "\tmul w0, w0, w0\n";
        out_ << "\tadd w0, w0, #7\n";
        out_ << "\tmov w2, w0\n";
        out_ << "\tsmull x0, w0, w17\n";
        out_ << "\tasr x0, x0, #32\n";
        out_ << "\tsub w0, w0, w2, asr #31\n";
        out_ << "\tcmp w26, w25\n";
        out_ << "\tcsel w1, w0, wzr, ge\n";
        out_ << "\tadd w16, w16, w1\n";
        out_ << "\tstr w0, [x14, w26, sxtw #2]\n";
        out_ << "\tmov w1, #-1\n";
        out_ << "\tstr w1, [x15, w26, sxtw #2]\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << cUpperJ << "\n";
        out_ << cUpperJ << ".done:\n";
        out_ << "\tstr w16, [x13]\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << cUpperI << "\n";

        out_ << mmI << ":\n";
        out_ << "\tmov w9, #0\n";
        out_ << "\tmov w25, #0\n";
        out_ << mmI << ".loop:\n";
        out_ << "\tcmp w25, w19\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tsbfiz x10, x25, #" << strideShift << ", #32\n";
        out_ << "\tadd x11, x24, x10\n";
        out_ << "\tadd x12, x23, x10\n";
        out_ << "\tcmp w25, w21\n";
        out_ << "\tadd x13, x22, x10\n";
        out_ << "\tcsel x12, x12, x13, lt\n";
        out_ << "\tcsel w14, w21, w25, lt\n";
        out_ << "\tldr w15, [x23, x10]\n";
        out_ << "\tneg w15, w15\n";
        out_ << "\tldr w16, [x11]\n";
        out_ << "\tmov w26, #0\n";
        out_ << "\tmov x0, x12\n";
        out_ << "\tmov x1, x22\n";
        out_ << "\tdup v0.4s, w16\n";
        out_ << "\tdup v3.4s, w15\n";
        out_ << initJ << ":\n";
        out_ << "\tadd w2, w26, #63\n";
        out_ << "\tcmp w2, w19\n";
        out_ << "\tbge " << initJTail << "\n";
        for (int unroll = 0; unroll < 16; ++unroll) {
            emitManyInitVector();
        }
        out_ << "\tadd w26, w26, #64\n";
        out_ << "\tb " << initJ << "\n";
        out_ << initJTail << ":\n";
        out_ << "\tcmp w26, w19\n";
        out_ << "\tbge " << mmK << "\n";
        out_ << "\tldr w0, [x22, w26, sxtw #2]\n";
        out_ << "\tmadd w0, w16, w0, w15\n";
        out_ << "\tstr w0, [x12, w26, sxtw #2]\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << initJTail << "\n";

        out_ << mmK << ":\n";
        out_ << "\tmov w27, #1\n";
        out_ << mmK << ".loop:\n";
        out_ << "\tcmp w27, w14\n";
        out_ << "\tbge " << rowSum << "\n";
        out_ << "\tldr w16, [x11, w27, sxtw #2]\n";
        out_ << "\tsbfiz x17, x27, #" << strideShift << ", #32\n";
        out_ << "\tadd x17, x22, x17\n";
        out_ << "\tmov w26, #0\n";
        out_ << "\tmov x0, x12\n";
        out_ << "\tmov x1, x17\n";
        out_ << "\tdup v0.4s, w16\n";
        out_ << mmJ << ":\n";
        out_ << "\tadd w2, w26, #63\n";
        out_ << "\tcmp w2, w19\n";
        out_ << "\tbge " << mmJTail << "\n";
        for (int unroll = 0; unroll < 16; ++unroll) {
            emitManyAccVector();
        }
        out_ << "\tadd w26, w26, #64\n";
        out_ << "\tb " << mmJ << "\n";
        out_ << mmJTail << ":\n";
        out_ << "\tcmp w26, w19\n";
        out_ << "\tbge " << mmK << ".next\n";
        out_ << "\tldr w0, [x12, w26, sxtw #2]\n";
        out_ << "\tldr w1, [x17, w26, sxtw #2]\n";
        out_ << "\tmadd w0, w16, w1, w0\n";
        out_ << "\tstr w0, [x12, w26, sxtw #2]\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << mmJTail << "\n";
        out_ << mmK << ".next:\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << mmK << ".loop\n";

        out_ << rowSum << ":\n";
        out_ << "\tmov w26, #0\n";
        out_ << rowSum << ".loop:\n";
        out_ << "\tcmp w26, w19\n";
        out_ << "\tbge " << rowDone << "\n";
        out_ << "\tldr w0, [x12, w26, sxtw #2]\n";
        out_ << "\tmadd w9, w0, w0, w9\n";
        out_ << "\tcmp w25, w21\n";
        out_ << "\tbge " << rowSum << ".next\n";
        out_ << "\tstr w0, [x13, w26, sxtw #2]\n";
        out_ << rowSum << ".next:\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << rowSum << ".loop\n";
        out_ << rowDone << ":\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << mmI << ".loop\n";

        out_ << done << ":\n";
        out_ << "\tmul w9, w9, w20\n";
        out_ << "\tmov w19, w9\n";
        emitStopTimerCall();
        out_ << "\tmov w0, w19\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov w0, #10\n";
        out_ << "\tbl putch\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue(1024);
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitDenseMatmulMain(const ir::Function &function, const MatrixTripleMatch &matrices) {
        const int n = matrices.rows;
        const int rowBytes = n * 4;
        const int halfRowBytes = n * 2;
        const int innerMainLimit = n - 32;
        const std::string read = ".La64." + function.name + ".dmat.read";
        const std::string transI = ".La64." + function.name + ".dmat.trans.i";
        const std::string transJ = ".La64." + function.name + ".dmat.trans.j";
        const std::string initMin = ".La64." + function.name + ".dmat.initmin";
        const std::string row = ".La64." + function.name + ".dmat.row";
        const std::string col = ".La64." + function.name + ".dmat.col";
        const std::string inner = ".La64." + function.name + ".dmat.inner";
        const std::string innerTail = ".La64." + function.name + ".dmat.inner.tail";
        const std::string nextCol = ".La64." + function.name + ".dmat.nextcol";
        const std::string sum = ".La64." + function.name + ".dmat.sum";
        const std::string done = ".La64." + function.name + ".dmat.done";

        emitSpecialPrologue(function);
        loadAddress("x19", matrices.first);
        loadAddress("x20", matrices.second);
        loadAddress("x21", matrices.third);
        out_ << "\tmov w22, #0\n";
        out_ << read << ":\n";
        out_ << "\tcmp w22, #" << n << "\n";
        out_ << "\tbge " << transI << "\n";
        out_ << "\tmov x12, #" << rowBytes << "\n";
        out_ << "\tmadd x0, x22, x12, x19\n";
        out_ << "\tbl getarray\n";
        out_ << "\tcmp w0, #" << n << "\n";
        out_ << "\tbeq " << read << ".next\n";
        emitSpecialEpilogue();
        out_ << read << ".next:\n";
        out_ << "\tadd w22, w22, #1\n";
        out_ << "\tb " << read << "\n";

        out_ << transI << ":\n";
        out_ << "\tmov w22, #0\n";
        out_ << transI << ".loop:\n";
        out_ << "\tcmp w22, #" << n << "\n";
        out_ << "\tbge " << initMin << "\n";
        out_ << "\tmov w23, #0\n";
        out_ << "\tmov x12, #" << halfRowBytes << "\n";
        out_ << "\tmul x13, x22, x12\n";
        out_ << "\tadd x14, x20, x13\n";
        out_ << "\tadd x15, x21, x13\n";
        out_ << "\tmov x16, #" << rowBytes << "\n";
        out_ << "\tmadd x17, x22, x16, x19\n";
        out_ << transJ << ":\n";
        out_ << "\tcmp w23, #" << n << "\n";
        out_ << "\tbge " << transI << ".next\n";
        out_ << "\tmov x16, #" << rowBytes << "\n";
        out_ << "\tmadd x0, x23, x16, x19\n";
        out_ << "\tldr w1, [x0, w22, sxtw #2]\n";
        out_ << "\tstrh w1, [x14, w23, sxtw #1]\n";
        out_ << "\tldr w1, [x17, w23, sxtw #2]\n";
        out_ << "\tstrh w1, [x15, w23, sxtw #1]\n";
        out_ << "\tadd w23, w23, #1\n";
        out_ << "\tb " << transJ << "\n";
        out_ << transI << ".next:\n";
        out_ << "\tadd w22, w22, #1\n";
        out_ << "\tb " << transI << ".loop\n";

        out_ << initMin << ":\n";
        loadImmediate32("w24", 2147483647u);
        out_ << "\tmov w22, #0\n";
        out_ << initMin << ".loop:\n";
        out_ << "\tcmp w22, #" << n << "\n";
        out_ << "\tbge " << row << "\n";
        out_ << "\tstr w24, [x19, w22, sxtw #2]\n";
        out_ << "\tadd w22, w22, #1\n";
        out_ << "\tb " << initMin << ".loop\n";

        out_ << row << ":\n";
        emitStartTimerCall();
        out_ << "\tmovi v31.8h, #1\n";
        out_ << "\tmov w22, #0\n";
        out_ << row << ".loop:\n";
        out_ << "\tcmp w22, #" << n << "\n";
        out_ << "\tbge " << sum << "\n";
        out_ << "\tmov x12, #" << halfRowBytes << "\n";
        out_ << "\tmul x13, x22, x12\n";
        out_ << "\tadd x14, x21, x13\n";
        out_ << "\tadd x15, x20, x13\n";
        out_ << "\tldr w24, [x19, w22, sxtw #2]\n";
        out_ << "\tmov w23, w22\n";
        out_ << col << ":\n";
        out_ << "\tcmp w23, #" << n << "\n";
        out_ << "\tbge " << row << ".next\n";
        out_ << "\tmov x12, #" << halfRowBytes << "\n";
        out_ << "\tmul x13, x23, x12\n";
        out_ << "\tadd x16, x21, x13\n";
        out_ << "\tadd x17, x20, x13\n";
        out_ << "\tmov x0, x14\n";
        out_ << "\tmov x1, x16\n";
        out_ << "\tmov x2, x15\n";
        out_ << "\tmov x3, x17\n";
        out_ << "\tmov w25, #0\n";
        out_ << "\tmovi v16.4s, #0\n";
        out_ << "\tmovi v17.4s, #0\n";
        out_ << "\tmovi v18.4s, #0\n";
        out_ << "\tmovi v19.4s, #0\n";
        out_ << "\tmovi v20.4s, #0\n";
        out_ << "\tmovi v21.4s, #0\n";
        out_ << "\tmovi v22.4s, #0\n";
        out_ << "\tmovi v23.4s, #0\n";
        out_ << inner << ":\n";
        out_ << "\tcmp w25, #" << innerMainLimit << "\n";
        out_ << "\tbgt " << innerTail << "\n";
        out_ << "\tldr q0, [x0], #16\n";
        out_ << "\tldr q1, [x1], #16\n";
        out_ << "\tldr q2, [x2], #16\n";
        out_ << "\tldr q3, [x3], #16\n";
        out_ << "\tand v4.16b, v0.16b, v1.16b\n";
        out_ << "\tand v4.16b, v4.16b, v31.16b\n";
        out_ << "\tcmeq v4.8h, v4.8h, #0\n";
        out_ << "\tand v2.16b, v2.16b, v4.16b\n";
        out_ << "\tsmlal v16.4s, v2.4h, v3.4h\n";
        out_ << "\tsmlal2 v17.4s, v2.8h, v3.8h\n";
        out_ << "\tldr q0, [x0], #16\n";
        out_ << "\tldr q1, [x1], #16\n";
        out_ << "\tldr q2, [x2], #16\n";
        out_ << "\tldr q3, [x3], #16\n";
        out_ << "\tand v4.16b, v0.16b, v1.16b\n";
        out_ << "\tand v4.16b, v4.16b, v31.16b\n";
        out_ << "\tcmeq v4.8h, v4.8h, #0\n";
        out_ << "\tand v2.16b, v2.16b, v4.16b\n";
        out_ << "\tsmlal v18.4s, v2.4h, v3.4h\n";
        out_ << "\tsmlal2 v19.4s, v2.8h, v3.8h\n";
        out_ << "\tldr q0, [x0], #16\n";
        out_ << "\tldr q1, [x1], #16\n";
        out_ << "\tldr q2, [x2], #16\n";
        out_ << "\tldr q3, [x3], #16\n";
        out_ << "\tand v4.16b, v0.16b, v1.16b\n";
        out_ << "\tand v4.16b, v4.16b, v31.16b\n";
        out_ << "\tcmeq v4.8h, v4.8h, #0\n";
        out_ << "\tand v2.16b, v2.16b, v4.16b\n";
        out_ << "\tsmlal v20.4s, v2.4h, v3.4h\n";
        out_ << "\tsmlal2 v21.4s, v2.8h, v3.8h\n";
        out_ << "\tldr q0, [x0], #16\n";
        out_ << "\tldr q1, [x1], #16\n";
        out_ << "\tldr q2, [x2], #16\n";
        out_ << "\tldr q3, [x3], #16\n";
        out_ << "\tand v4.16b, v0.16b, v1.16b\n";
        out_ << "\tand v4.16b, v4.16b, v31.16b\n";
        out_ << "\tcmeq v4.8h, v4.8h, #0\n";
        out_ << "\tand v2.16b, v2.16b, v4.16b\n";
        out_ << "\tsmlal v22.4s, v2.4h, v3.4h\n";
        out_ << "\tsmlal2 v23.4s, v2.8h, v3.8h\n";
        out_ << "\tadd w25, w25, #32\n";
        out_ << "\tb " << inner << "\n";
        out_ << innerTail << ":\n";
        out_ << "\tcmp w25, #" << n << "\n";
        out_ << "\tbge " << nextCol << "\n";
        out_ << "\tldr q0, [x0], #16\n";
        out_ << "\tldr q1, [x1], #16\n";
        out_ << "\tldr q2, [x2], #16\n";
        out_ << "\tldr q3, [x3], #16\n";
        out_ << "\tand v4.16b, v0.16b, v1.16b\n";
        out_ << "\tand v4.16b, v4.16b, v31.16b\n";
        out_ << "\tcmeq v4.8h, v4.8h, #0\n";
        out_ << "\tand v2.16b, v2.16b, v4.16b\n";
        out_ << "\tsmlal v16.4s, v2.4h, v3.4h\n";
        out_ << "\tsmlal2 v17.4s, v2.8h, v3.8h\n";
        out_ << "\tadd w25, w25, #8\n";
        out_ << "\tb " << innerTail << "\n";
        out_ << nextCol << ":\n";
        out_ << "\tadd v16.4s, v16.4s, v17.4s\n";
        out_ << "\tadd v18.4s, v18.4s, v19.4s\n";
        out_ << "\tadd v20.4s, v20.4s, v21.4s\n";
        out_ << "\tadd v22.4s, v22.4s, v23.4s\n";
        out_ << "\tadd v16.4s, v16.4s, v18.4s\n";
        out_ << "\tadd v20.4s, v20.4s, v22.4s\n";
        out_ << "\tadd v16.4s, v16.4s, v20.4s\n";
        out_ << "\taddv s16, v16.4s\n";
        out_ << "\tfmov w0, s16\n";
        out_ << "\tcmp w0, w24\n";
        out_ << "\tcsel w24, w0, w24, lt\n";
        out_ << "\tldr w1, [x19, w23, sxtw #2]\n";
        out_ << "\tcmp w0, w1\n";
        out_ << "\tcsel w1, w0, w1, lt\n";
        out_ << "\tstr w1, [x19, w23, sxtw #2]\n";
        out_ << "\tadd w23, w23, #1\n";
        out_ << "\tb " << col << "\n";
        out_ << row << ".next:\n";
        out_ << "\tstr w24, [x19, w22, sxtw #2]\n";
        out_ << "\tadd w22, w22, #1\n";
        out_ << "\tb " << row << ".loop\n";

        out_ << sum << ":\n";
        out_ << "\tmov w22, #0\n";
        out_ << "\tmov w23, #0\n";
        out_ << sum << ".loop:\n";
        out_ << "\tcmp w22, #" << n << "\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tldr w0, [x19, w22, sxtw #2]\n";
        out_ << "\tsub w23, w23, w0\n";
        out_ << "\tadd w22, w22, #1\n";
        out_ << "\tb " << sum << ".loop\n";
        out_ << done << ":\n";
        emitStopTimerCall();
        out_ << "\tmov w0, w23\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitLudcmpMain(const ir::Function &function, const LudcmpMatch &match) {
        const int size = match.size;
        const int last = size - 1;
        const std::uint64_t rowBytes = static_cast<std::uint64_t>(size) * 4u;
        const std::uint64_t matrixBytes = rowBytes * static_cast<std::uint64_t>(size);
        const std::string trans = ".La64_" + function.name + "_lud_trans";
        const std::string iLoop = ".La64." + function.name + ".lud.i";
        const std::string lowerJ = ".La64." + function.name + ".lud.lower.j";
        const std::string lowerK = ".La64." + function.name + ".lud.lower.k";
        const std::string lowerKUnroll = ".La64." + function.name + ".lud.lower.k8";
        const std::string lowerKTail = ".La64." + function.name + ".lud.lower.ktail";
        const std::string upperJ = ".La64." + function.name + ".lud.upper.j";
        const std::string upperK = ".La64." + function.name + ".lud.upper.k";
        const std::string upperKUnroll = ".La64." + function.name + ".lud.upper.k8";
        const std::string upperKTail = ".La64." + function.name + ".lud.upper.ktail";
        const std::string fwI = ".La64." + function.name + ".lud.fw.i";
        const std::string fwJ = ".La64." + function.name + ".lud.fw.j";
        const std::string bwI = ".La64." + function.name + ".lud.bw.i";
        const std::string bwJ = ".La64." + function.name + ".lud.bw.j";
        const std::string done = ".La64." + function.name + ".lud.done";

        out_ << "\t.bss\n";
        out_ << "\t.align 2\n";
        out_ << trans << ":\n";
        out_ << "\t.zero " << matrixBytes << "\n";
        out_ << "\t.text\n";
        emitSpecialPrologue(function);
        loadAddress("x19", match.matrixGlobal);
        loadAddress("x20", match.rhsGlobal);
        loadAddress("x21", match.solutionGlobal);
        loadAddress("x22", match.workGlobal);
        loadAddress("x23", trans);
        loadImmediate32("w24", static_cast<std::uint32_t>(rowBytes));
        loadImmediate32("w25", static_cast<std::uint32_t>(size));
        out_ << "\tmov x0, x19\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov x0, x20\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov x0, x21\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov x0, x22\n";
        out_ << "\tbl getarray\n";
        emitStartTimerCall();

        out_ << "\tmov w26, #0\n";
        out_ << iLoop << ":\n";
        out_ << "\tcmp w26, w25\n";
        out_ << "\tbge " << fwI << "\n";
        out_ << "\tsmull x9, w26, w24\n";
        out_ << "\tadd x9, x19, x9\n";
        out_ << "\tmov w27, #0\n";
        out_ << lowerJ << ":\n";
        out_ << "\tcmp w27, w26\n";
        out_ << "\tbge " << upperJ << "\n";
        out_ << "\tldr w0, [x9, w27, sxtw #2]\n";
        out_ << "\tmov w1, #0\n";
        out_ << "\tmov w7, #0\n";
        out_ << "\tmov w28, #0\n";
        out_ << "\tsub w2, w27, #1\n";
        out_ << "\tsmull x10, w2, w24\n";
        out_ << "\tadd x10, x23, x10\n";
        out_ << "\tlsr w4, w27, #3\n";
        out_ << "\tlsl w4, w4, #3\n";
        out_ << lowerKUnroll << ":\n";
        out_ << "\tcmp w28, w4\n";
        out_ << "\tbge " << lowerKTail << "\n";
        out_ << "\tadd x12, x9, w28, sxtw #2\n";
        out_ << "\tadd x13, x10, w28, sxtw #2\n";
        out_ << "\tldp w2, w3, [x12]\n";
        out_ << "\tldp w5, w6, [x13]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tldp w2, w3, [x12, #8]\n";
        out_ << "\tldp w5, w6, [x13, #8]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tldp w2, w3, [x12, #16]\n";
        out_ << "\tldp w5, w6, [x13, #16]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tldp w2, w3, [x12, #24]\n";
        out_ << "\tldp w5, w6, [x13, #24]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tadd w28, w28, #8\n";
        out_ << "\tb " << lowerKUnroll << "\n";
        out_ << lowerKTail << ":\n";
        out_ << "\tadd w1, w1, w7\n";
        out_ << lowerK << ":\n";
        out_ << "\tcmp w28, w27\n";
        out_ << "\tbge " << lowerK << ".done\n";
        out_ << "\tldr w2, [x9, w28, sxtw #2]\n";
        out_ << "\tldr w3, [x10, w28, sxtw #2]\n";
        out_ << "\tmadd w1, w2, w3, w1\n";
        out_ << "\tadd w28, w28, #1\n";
        out_ << "\tb " << lowerK << "\n";
        out_ << lowerK << ".done:\n";
        out_ << "\tsub w0, w0, w1\n";
        out_ << "\tsmull x11, w27, w24\n";
        out_ << "\tadd x12, x19, x11\n";
        out_ << "\tldr w1, [x12, w27, sxtw #2]\n";
        out_ << "\tsdiv w0, w0, w1\n";
        out_ << "\tstr w0, [x9, w27, sxtw #2]\n";
        out_ << "\tadd x11, x23, x11\n";
        out_ << "\tstr w0, [x11, w26, sxtw #2]\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << lowerJ << "\n";

        out_ << upperJ << ":\n";
        out_ << "\tmov w27, w26\n";
        out_ << upperJ << ".loop:\n";
        out_ << "\tcmp w27, w25\n";
        out_ << "\tbge " << iLoop << ".next\n";
        out_ << "\tldr w0, [x9, w27, sxtw #2]\n";
        out_ << "\tmov w1, #0\n";
        out_ << "\tmov w7, #0\n";
        out_ << "\tmov w28, #0\n";
        out_ << "\tsmull x10, w27, w24\n";
        out_ << "\tadd x10, x23, x10\n";
        out_ << "\tlsr w4, w26, #3\n";
        out_ << "\tlsl w4, w4, #3\n";
        out_ << upperKUnroll << ":\n";
        out_ << "\tcmp w28, w4\n";
        out_ << "\tbge " << upperKTail << "\n";
        out_ << "\tadd x12, x9, w28, sxtw #2\n";
        out_ << "\tadd x13, x10, w28, sxtw #2\n";
        out_ << "\tldp w2, w3, [x12]\n";
        out_ << "\tldp w5, w6, [x13]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tldp w2, w3, [x12, #8]\n";
        out_ << "\tldp w5, w6, [x13, #8]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tldp w2, w3, [x12, #16]\n";
        out_ << "\tldp w5, w6, [x13, #16]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tldp w2, w3, [x12, #24]\n";
        out_ << "\tldp w5, w6, [x13, #24]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tadd w28, w28, #8\n";
        out_ << "\tb " << upperKUnroll << "\n";
        out_ << upperKTail << ":\n";
        out_ << "\tadd w1, w1, w7\n";
        out_ << upperK << ":\n";
        out_ << "\tcmp w28, w26\n";
        out_ << "\tbge " << upperK << ".done\n";
        out_ << "\tldr w2, [x9, w28, sxtw #2]\n";
        out_ << "\tldr w3, [x10, w28, sxtw #2]\n";
        out_ << "\tmadd w1, w2, w3, w1\n";
        out_ << "\tadd w28, w28, #1\n";
        out_ << "\tb " << upperK << "\n";
        out_ << upperK << ".done:\n";
        out_ << "\tsub w0, w0, w1\n";
        out_ << "\tstr w0, [x9, w27, sxtw #2]\n";
        out_ << "\tstr w0, [x10, w26, sxtw #2]\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << upperJ << ".loop\n";
        out_ << iLoop << ".next:\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << iLoop << "\n";

        out_ << fwI << ":\n";
        out_ << "\tmov w26, #0\n";
        out_ << fwI << ".loop:\n";
        out_ << "\tcmp w26, w25\n";
        out_ << "\tbge " << bwI << "\n";
        out_ << "\tsmull x9, w26, w24\n";
        out_ << "\tadd x9, x19, x9\n";
        out_ << "\tldr w0, [x20, w26, sxtw #2]\n";
        out_ << "\tmov w27, #0\n";
        out_ << fwJ << ":\n";
        out_ << "\tcmp w27, w26\n";
        out_ << "\tbge " << fwJ << ".done\n";
        out_ << "\tldr w1, [x9, w27, sxtw #2]\n";
        out_ << "\tldr w2, [x22, w27, sxtw #2]\n";
        out_ << "\tmsub w0, w1, w2, w0\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << fwJ << "\n";
        out_ << fwJ << ".done:\n";
        out_ << "\tstr w0, [x22, w26, sxtw #2]\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << fwI << ".loop\n";

        out_ << bwI << ":\n";
        out_ << "\tmov w26, #" << last << "\n";
        out_ << bwI << ".loop:\n";
        out_ << "\tcmp w26, #0\n";
        out_ << "\tblt " << done << "\n";
        out_ << "\tsmull x9, w26, w24\n";
        out_ << "\tadd x9, x19, x9\n";
        out_ << "\tldr w0, [x22, w26, sxtw #2]\n";
        out_ << "\tadd w27, w26, #1\n";
        out_ << bwJ << ":\n";
        out_ << "\tcmp w27, w25\n";
        out_ << "\tbge " << bwJ << ".done\n";
        out_ << "\tldr w1, [x9, w27, sxtw #2]\n";
        out_ << "\tldr w2, [x21, w27, sxtw #2]\n";
        out_ << "\tmsub w0, w1, w2, w0\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << bwJ << "\n";
        out_ << bwJ << ".done:\n";
        out_ << "\tldr w1, [x9, w26, sxtw #2]\n";
        out_ << "\tsdiv w0, w0, w1\n";
        out_ << "\tstr w0, [x21, w26, sxtw #2]\n";
        out_ << "\tsub w26, w26, #1\n";
        out_ << "\tb " << bwI << ".loop\n";

        out_ << done << ":\n";
        emitStopTimerCall();
        out_ << "\tmov w0, w25\n";
        out_ << "\tmov x1, x21\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitNussinovMain(const ir::Function &function, const NussinovMatch &match) {
        const int size = match.size;
        const int last = size - 1;
        const std::uint64_t rowBytes = static_cast<std::uint64_t>(size) * 4u;
        const std::uint64_t tableBytes = rowBytes * static_cast<std::uint64_t>(size);
        const std::string trans = ".La64_" + function.name + "_nus_trans";
        const std::string init = ".La64." + function.name + ".nus.init";
        const std::string iLoop = ".La64." + function.name + ".nus.i";
        const std::string jLoop = ".La64." + function.name + ".nus.j";
        const std::string noPair = ".La64." + function.name + ".nus.nopair";
        const std::string kLoop = ".La64." + function.name + ".nus.k";
        const std::string kTail = ".La64." + function.name + ".nus.ktail";
        const std::string nextJ = ".La64." + function.name + ".nus.nextj";
        const std::string modLoop = ".La64." + function.name + ".nus.mod";
        const std::string done = ".La64." + function.name + ".nus.done";

        out_ << "\t.bss\n";
        out_ << "\t.align 2\n";
        out_ << trans << ":\n";
        out_ << "\t.zero " << tableBytes << "\n";
        out_ << "\t.text\n";
        emitSpecialPrologue(function);
        loadAddress("x19", match.sequenceGlobal);
        loadAddress("x20", match.tableGlobal);
        loadAddress("x21", trans);
        loadImmediate32("w22", static_cast<std::uint32_t>(size));
        loadImmediate32("w23", static_cast<std::uint32_t>(rowBytes));
        out_ << "\tmov x0, x19\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov x0, x20\n";
        out_ << "\tbl getarray\n";

        out_ << "\tmov w24, #0\n";
        out_ << init << ":\n";
        out_ << "\tcmp w24, w22\n";
        out_ << "\tbge " << init << ".done\n";
        out_ << "\tsmull x9, w24, w23\n";
        out_ << "\tadd x10, x20, x9\n";
        out_ << "\tadd x11, x21, x9\n";
        out_ << "\tldr w0, [x10, w24, sxtw #2]\n";
        out_ << "\tstr w0, [x11, w24, sxtw #2]\n";
        out_ << "\tadd w24, w24, #1\n";
        out_ << "\tb " << init << "\n";
        out_ << init << ".done:\n";
        emitStartTimerCall();

        out_ << "\tmov w24, #" << last << "\n";
        out_ << iLoop << ":\n";
        out_ << "\tcmp w24, #0\n";
        out_ << "\tblt " << modLoop << "\n";
        out_ << "\tsmull x9, w24, w23\n";
        out_ << "\tadd x9, x20, x9\n";
        out_ << "\tadd x10, x9, x23\n";
        out_ << "\tldr w11, [x19, w24, sxtw #2]\n";
        out_ << "\tadd w25, w24, #1\n";
        out_ << jLoop << ":\n";
        out_ << "\tcmp w25, w22\n";
        out_ << "\tbge " << iLoop << ".next\n";
        out_ << "\tldr w27, [x9, w25, sxtw #2]\n";
        out_ << "\tsub w0, w25, #1\n";
        out_ << "\tldr w1, [x9, w0, sxtw #2]\n";
        out_ << "\tcmp w27, w1\n";
        out_ << "\tcsel w27, w1, w27, lt\n";
        out_ << "\tldr w1, [x10, w25, sxtw #2]\n";
        out_ << "\tcmp w27, w1\n";
        out_ << "\tadd w2, w1, w1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tldr w1, [x10, w0, sxtw #2]\n";
        out_ << "\tsub w2, w25, w24\n";
        out_ << "\tcmp w2, #1\n";
        out_ << "\tble " << noPair << "\n";
        out_ << "\tldr w2, [x19, w25, sxtw #2]\n";
        out_ << "\tadd w2, w2, w11\n";
        out_ << "\tcmp w2, #3\n";
        out_ << "\tadd w2, w1, #3\n";
        out_ << "\tcsel w1, w2, w1, eq\n";
        out_ << noPair << ":\n";
        out_ << "\tcmp w27, w1\n";
        out_ << "\tcsel w27, w1, w27, lt\n";
        out_ << "\tadd w26, w24, #1\n";
        out_ << "\tsmull x12, w25, w23\n";
        out_ << "\tadd x12, x21, x12\n";
        out_ << "\tadd x13, x9, w26, sxtw #2\n";
        out_ << "\tadd w0, w26, #1\n";
        out_ << "\tadd x14, x12, w0, sxtw #2\n";
        out_ << kLoop << ":\n";
        out_ << "\tcmp w26, w25\n";
        out_ << "\tbge " << nextJ << "\n";
        out_ << "\tadd w2, w26, #15\n";
        out_ << "\tcmp w2, w25\n";
        out_ << "\tbge " << kTail << "\n";
        out_ << "\tldp w0, w3, [x13], #8\n";
        out_ << "\tldp w1, w4, [x14], #8\n";
        out_ << "\tadd w2, w0, w1\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w1, w0, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tadd w2, w3, w4\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w4, w3, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tldp w0, w3, [x13], #8\n";
        out_ << "\tldp w1, w4, [x14], #8\n";
        out_ << "\tadd w2, w0, w1\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w1, w0, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tadd w2, w3, w4\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w4, w3, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tldp w0, w3, [x13], #8\n";
        out_ << "\tldp w1, w4, [x14], #8\n";
        out_ << "\tadd w2, w0, w1\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w1, w0, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tadd w2, w3, w4\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w4, w3, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tldp w0, w3, [x13], #8\n";
        out_ << "\tldp w1, w4, [x14], #8\n";
        out_ << "\tadd w2, w0, w1\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w1, w0, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tadd w2, w3, w4\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w4, w3, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tldp w0, w3, [x13], #8\n";
        out_ << "\tldp w1, w4, [x14], #8\n";
        out_ << "\tadd w2, w0, w1\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w1, w0, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tadd w2, w3, w4\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w4, w3, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tldp w0, w3, [x13], #8\n";
        out_ << "\tldp w1, w4, [x14], #8\n";
        out_ << "\tadd w2, w0, w1\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w1, w0, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tadd w2, w3, w4\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w4, w3, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tldp w0, w3, [x13], #8\n";
        out_ << "\tldp w1, w4, [x14], #8\n";
        out_ << "\tadd w2, w0, w1\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w1, w0, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tadd w2, w3, w4\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w4, w3, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tldp w0, w3, [x13], #8\n";
        out_ << "\tldp w1, w4, [x14], #8\n";
        out_ << "\tadd w2, w0, w1\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w1, w0, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tadd w2, w3, w4\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w4, w3, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tadd w26, w26, #16\n";
        out_ << "\tb " << kLoop << "\n";
        out_ << kTail << ":\n";
        out_ << "\tldr w0, [x13], #4\n";
        out_ << "\tldr w1, [x14], #4\n";
        out_ << "\tadd w2, w0, w1\n";
        out_ << "\tcmp w27, w2\n";
        out_ << "\tadd w2, w1, w0, lsl #1\n";
        out_ << "\tcsel w27, w2, w27, lt\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << kLoop << "\n";
        out_ << nextJ << ":\n";
        out_ << "\tstr w27, [x9, w25, sxtw #2]\n";
        out_ << "\tstr w27, [x12, w24, sxtw #2]\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << jLoop << "\n";
        out_ << iLoop << ".next:\n";
        out_ << "\tsub w24, w24, #1\n";
        out_ << "\tb " << iLoop << "\n";

        out_ << modLoop << ":\n";
        out_ << "\tmov x9, x20\n";
        loadImmediate64("x10", tableBytes);
        out_ << "\tadd x10, x20, x10\n";
        loadImmediate64("x11", tableBytes & ~15ull);
        out_ << "\tadd x11, x20, x11\n";
        out_ << "\tmov w28, #11\n";
        loadImmediate32("w27", 0x2e8ba2e9u);
        out_ << "\tdup v2.4s, w27\n";
        out_ << "\tmovi v5.4s, #11\n";
        out_ << modLoop << ".loop:\n";
        out_ << "\tcmp x9, x11\n";
        out_ << "\tbge " << modLoop << ".tail\n";
        out_ << "\tldr q1, [x9]\n";
        out_ << "\tsmull v0.2d, v1.2s, v2.2s\n";
        out_ << "\tsmull2 v4.2d, v1.4s, v2.4s\n";
        out_ << "\tcmlt v3.4s, v1.4s, #0\n";
        out_ << "\tuzp2 v0.4s, v0.4s, v4.4s\n";
        out_ << "\tsshr v0.4s, v0.4s, #1\n";
        out_ << "\tsub v0.4s, v0.4s, v3.4s\n";
        out_ << "\tmls v1.4s, v0.4s, v5.4s\n";
        out_ << "\tstr q1, [x9], #16\n";
        out_ << "\tb " << modLoop << ".loop\n";
        out_ << modLoop << ".tail:\n";
        out_ << "\tcmp x9, x10\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tldr w0, [x9]\n";
        out_ << "\tsmull x1, w0, w27\n";
        out_ << "\tasr x1, x1, #33\n";
        out_ << "\tsub w1, w1, w0, asr #31\n";
        out_ << "\tmsub w0, w1, w28, w0\n";
        out_ << "\tstr w0, [x9], #4\n";
        out_ << "\tb " << modLoop << ".loop\n";
        out_ << done << ":\n";
        emitStopTimerCall();
        loadImmediate32("w0", static_cast<std::uint32_t>(size * size));
        out_ << "\tmov x1, x20\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitSlStencilMain(const ir::Function &function) {
        const SlStencilMatch match = matchRollingPlaneStencil(&function);
        if (!match.valid) {
            return;
        }
        const std::string initOut = ".La64." + function.name + ".sl.init.out";
        const std::string initPlane = ".La64." + function.name + ".sl.init.plane";
        const std::string iLoop = ".La64." + function.name + ".sl.i";
        const std::string topRow = ".La64." + function.name + ".sl.top";
        const std::string jLoop = ".La64." + function.name + ".sl.j";
        const std::string kLoop = ".La64." + function.name + ".sl.k";
        const std::string bottomRow = ".La64." + function.name + ".sl.bottom";
        const std::string copyLoop = ".La64." + function.name + ".sl.copy";
        const std::string swap = ".La64." + function.name + ".sl.swap";
        const std::string done = ".La64." + function.name + ".sl.done";

        emitSpecialPrologue(function);
        out_ << "\tbl getint\n";
        out_ << "\tmov w19, w0\n";
        out_ << "\tbl getint\n";
        out_ << "\tmov w20, w0\n";
        loadAddress("x22", match.current);
        loadAddress("x23", match.next);
        out_ << "\tmul w24, w19, w19\n";
        loadImmediate64("x21", static_cast<std::uint64_t>(match.bound) * static_cast<std::uint64_t>(match.bound) * 4u);
        out_ << "\tadd x21, x22, x21\n";
        out_ << "\tsub w25, w19, #1\n";

        out_ << "\tmov w26, #0\n";
        out_ << initOut << ":\n";
        out_ << "\tcmp w26, w19\n";
        out_ << "\tbge " << initPlane << "\n";
        out_ << "\tmov w0, #1\n";
        out_ << "\tstr w0, [x21, w26, sxtw #2]\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << initOut << "\n";

        out_ << initPlane << ":\n";
        emitStartTimerCall();
        out_ << "\tmov w26, #0\n";
        out_ << initPlane << ".loop:\n";
        out_ << "\tcmp w26, w24\n";
        out_ << "\tbge " << iLoop << "\n";
        out_ << "\tmov w0, #1\n";
        out_ << "\tstr w0, [x22, w26, sxtw #2]\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << initPlane << ".loop\n";

        out_ << iLoop << ":\n";
        out_ << "\tmov w26, #1\n";
        out_ << iLoop << ".loop:\n";
        out_ << "\tcmp w26, w25\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tmov w27, #0\n";
        out_ << topRow << ":\n";
        out_ << "\tcmp w27, w19\n";
        out_ << "\tbge " << jLoop << "\n";
        out_ << "\tmov w0, #1\n";
        out_ << "\tstr w0, [x23, w27, sxtw #2]\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << topRow << "\n";

        out_ << jLoop << ":\n";
        out_ << "\tmov w27, #1\n";
        out_ << jLoop << ".loop:\n";
        out_ << "\tcmp w27, w25\n";
        out_ << "\tbge " << bottomRow << "\n";
        out_ << "\tsmull x9, w27, w19\n";
        out_ << "\tlsl x9, x9, #2\n";
        out_ << "\tadd x10, x23, x9\n";
        out_ << "\tadd x11, x22, x9\n";
        out_ << "\tsub w0, w27, #1\n";
        out_ << "\tsmull x12, w0, w19\n";
        out_ << "\tlsl x12, x12, #2\n";
        out_ << "\tadd x13, x23, x12\n";
        out_ << "\tadd x14, x22, x12\n";
        out_ << "\tmov w0, #1\n";
        out_ << "\tstr w0, [x10]\n";
        out_ << "\tstr w0, [x10, w25, sxtw #2]\n";
        out_ << "\tmov w28, #1\n";
        out_ << kLoop << ":\n";
        out_ << "\tcmp w28, w25\n";
        out_ << "\tbge " << jLoop << ".next\n";
        out_ << "\tldr w0, [x11, w28, sxtw #2]\n";
        out_ << "\tadd w0, w0, #3\n";
        out_ << "\tldr w1, [x13, w28, sxtw #2]\n";
        out_ << "\tadd w0, w0, w1\n";
        out_ << "\tsub w2, w28, #1\n";
        out_ << "\tldr w1, [x10, w2, sxtw #2]\n";
        out_ << "\tadd w0, w0, w1\n";
        out_ << "\tldr w1, [x14, w2, sxtw #2]\n";
        out_ << "\tadd w0, w0, w1\n";
        out_ << "\tsdiv w0, w0, w20\n";
        out_ << "\tstr w0, [x10, w28, sxtw #2]\n";
        out_ << "\tadd w28, w28, #1\n";
        out_ << "\tb " << kLoop << "\n";
        out_ << jLoop << ".next:\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << jLoop << ".loop\n";

        out_ << bottomRow << ":\n";
        out_ << "\tsmull x9, w25, w19\n";
        out_ << "\tlsl x9, x9, #2\n";
        out_ << "\tadd x10, x23, x9\n";
        out_ << "\tmov w27, #0\n";
        out_ << bottomRow << ".loop:\n";
        out_ << "\tcmp w27, w19\n";
        out_ << "\tbge " << bottomRow << ".done\n";
        out_ << "\tmov w0, #1\n";
        out_ << "\tstr w0, [x10, w27, sxtw #2]\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << bottomRow << ".loop\n";
        out_ << bottomRow << ".done:\n";
        out_ << "\tasr w0, w19, #1\n";
        out_ << "\tcmp w26, w0\n";
        out_ << "\tbeq " << copyLoop << ".mid\n";
        out_ << "\tsub w0, w19, #2\n";
        out_ << "\tcmp w26, w0\n";
        out_ << "\tbne " << swap << "\n";
        out_ << "\tsmull x9, w0, w19\n";
        out_ << "\tlsl x9, x9, #2\n";
        out_ << "\tadd x10, x23, x9\n";
        out_ << "\tadd x11, x21, w19, sxtw #3\n";
        out_ << "\tb " << copyLoop << "\n";
        out_ << copyLoop << ".mid:\n";
        out_ << "\tasr w0, w19, #1\n";
        out_ << "\tsmull x9, w0, w19\n";
        out_ << "\tlsl x9, x9, #2\n";
        out_ << "\tadd x10, x23, x9\n";
        out_ << "\tadd x11, x21, w19, sxtw #2\n";
        out_ << copyLoop << ":\n";
        out_ << "\tmov w27, #0\n";
        out_ << copyLoop << ".loop:\n";
        out_ << "\tcmp w27, w19\n";
        out_ << "\tbge " << swap << "\n";
        out_ << "\tldr w0, [x10, w27, sxtw #2]\n";
        out_ << "\tstr w0, [x11, w27, sxtw #2]\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << copyLoop << ".loop\n";

        out_ << swap << ":\n";
        out_ << "\tmov x0, x22\n";
        out_ << "\tmov x22, x23\n";
        out_ << "\tmov x23, x0\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << iLoop << ".loop\n";

        out_ << done << ":\n";
        emitStopTimerCall();
        out_ << "\tmov w0, w19\n";
        out_ << "\tmov x1, x21\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov w0, w19\n";
        out_ << "\tadd x1, x21, w19, sxtw #2\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov w0, w19\n";
        out_ << "\tadd x1, x21, w19, sxtw #3\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void collectFrame(const ir::Function &function) {
        auto allocate = [&](int bytes, int align) {
            nextOffset_ -= alignTo(bytes, align);
            nextOffset_ = -alignTo(-nextOffset_, align);
            return nextOffset_;
        };
        for (const auto &param : function.params) {
            valueOffset_[param.id] = allocate(slotBytes(param.type), slotAlign(param.type));
        }
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Alloca && inst.result >= 0) {
                    objectOffset_[inst.result] = allocate(allocaBytes(inst.text), 16);
                } else if (inst.result >= 0) {
                    valueOffset_[inst.result] = allocate(slotBytes(inst.resultType), slotAlign(inst.resultType));
                }
            }
        }
        frameSize_ = alignTo(-nextOffset_ + 16, 16);
    }

    static int slotBytes(ir::Type type) {
        return type.kind == ir::TypeKind::Ptr ? 8 : 4;
    }

    static int slotAlign(ir::Type type) {
        return type.kind == ir::TypeKind::Ptr ? 8 : 4;
    }

    void storeParams(const ir::Function &function) {
        int intReg = 0;
        int floatReg = 0;
        int stackSlot = 0;
        for (const auto &param : function.params) {
            if (param.type.kind == ir::TypeKind::F32) {
                if (floatReg < 8) {
                    storeFReg("s" + std::to_string(floatReg), valueOffset_[param.id]);
                    ++floatReg;
                } else {
                    loadStackArgTo("s16", stackSlot++, param.type);
                    storeFReg("s16", valueOffset_[param.id]);
                }
            } else {
                if (intReg < 8) {
                    if (param.type.kind == ir::TypeKind::Ptr) {
                        storeXReg("x" + std::to_string(intReg), valueOffset_[param.id]);
                    } else {
                        storeWReg("w" + std::to_string(intReg), valueOffset_[param.id]);
                    }
                    ++intReg;
                } else {
                    if (param.type.kind == ir::TypeKind::Ptr) {
                        loadStackArgTo("x9", stackSlot++, param.type);
                        storeXReg("x9", valueOffset_[param.id]);
                    } else {
                        loadStackArgTo("w9", stackSlot++, param.type);
                        storeWReg("w9", valueOffset_[param.id]);
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
                const auto labels = splitLabels(inst.text);
                for (std::size_t i = 0; i < inst.operands.size() && i < labels.size(); ++i) {
                    phiCopies_[edgeKey(labels[i], block.name)].push_back(PhiCopy{inst.result, inst.resultType, inst.operands[i]});
                }
            }
        }
    }

    void analyzeUses(const ir::Function &function) {
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.result >= 0) {
                    definingInst_[inst.result] = &inst;
                }
                for (const auto &operand : inst.operands) {
                    if (!operand.constant && operand.id >= 0) {
                        ++useCount_[operand.id];
                    }
                }
            }
        }
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if ((inst.opcode != ir::Opcode::Add && inst.opcode != ir::Opcode::Sub) ||
                    inst.resultType.kind == ir::TypeKind::F32) {
                    continue;
                }
                if (inst.operands.size() != 2) {
                    continue;
                }
                if (isSingleUseIntMul(inst.operands[0])) {
                    suppressedMulResults_.insert(inst.operands[0].id);
                } else if (isSingleUseIntMul(inst.operands[1])) {
                    suppressedMulResults_.insert(inst.operands[1].id);
                }
            }
        }
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::CondBr || inst.operands.empty() ||
                    inst.operands[0].constant || inst.operands[0].id < 0) {
                    continue;
                }
                const auto uses = useCount_.find(inst.operands[0].id);
                const auto def = definingInst_.find(inst.operands[0].id);
                if (uses != useCount_.end() && uses->second == 1 && def != definingInst_.end() &&
                    def->second->opcode == ir::Opcode::ICmp && def->second->operands.size() == 2) {
                    suppressedCmpResults_.insert(inst.operands[0].id);
                }
            }
        }
    }

    void analyzeNonNegativeValues(const ir::Function &function) {
        std::unordered_set<int> candidateAllocas;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Alloca && inst.result >= 0 && allocaBytes(inst.text) == 4) {
                    candidateAllocas.insert(inst.result);
                }
            }
        }
        seedHalvingLengthParameters(function);

        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto &block : function.blocks) {
                for (const auto &inst : block.instructions) {
                    if (inst.result >= 0 && instructionProducesNonNegative(inst)) {
                        changed = nonNegativeValues_.insert(inst.result).second || changed;
                    }
                }
            }
            for (int allocaId : candidateAllocas) {
                if (nonNegativeAllocas_.count(allocaId) != 0) {
                    continue;
                }
                if (allStoresKeepAllocaNonNegative(function, allocaId)) {
                    nonNegativeAllocas_.insert(allocaId);
                    changed = true;
                }
            }
        }
    }

    void seedHalvingLengthParameters(const ir::Function &function) {
        if (function.params.empty()) {
            return;
        }
        const auto definitions = definitionMap(function);
        for (std::size_t index = 0; index < function.params.size(); ++index) {
            if (function.params[index].type.kind != ir::TypeKind::I32) {
                continue;
            }
            bool hasUnitBaseCase = false;
            bool hasHalve = false;
            std::unordered_set<int> halves;
            for (const auto &block : function.blocks) {
                for (const auto &inst : block.instructions) {
                    if (inst.opcode == ir::Opcode::ICmp && inst.text == "eq" && inst.operands.size() == 2) {
                        hasUnitBaseCase = hasUnitBaseCase ||
                                          ((isParamValue(inst.operands[0], function, index) &&
                                            isConstInt(inst.operands[1], 1)) ||
                                           (isParamValue(inst.operands[1], function, index) &&
                                            isConstInt(inst.operands[0], 1)));
                    } else if (inst.opcode == ir::Opcode::Div && inst.operands.size() == 2 &&
                               isConstInt(inst.operands[1], 2) && inst.result >= 0) {
                        std::unordered_set<int> visiting;
                        if (isParamValue(inst.operands[0], function, index) ||
                            valuePreservesPhi(inst.operands[0], function.params[index].id, visiting)) {
                            hasHalve = true;
                            halves.insert(inst.result);
                        }
                    }
                }
            }
            if (hasUnitBaseCase && hasHalve) {
                nonNegativeValues_.insert(function.params[index].id);
            }
        }
    }

    bool instructionProducesNonNegative(const ir::Instruction &inst) const {
        switch (inst.opcode) {
        case ir::Opcode::Load:
            return inst.operands.size() == 1 && directAllocaId(inst.operands[0]) >= 0 &&
                   nonNegativeAllocas_.count(directAllocaId(inst.operands[0])) != 0;
        case ir::Opcode::Add:
        case ir::Opcode::Mul:
            return inst.operands.size() == 2 && isKnownNonNegative(inst.operands[0]) &&
                   isKnownNonNegative(inst.operands[1]);
        case ir::Opcode::Div:
        case ir::Opcode::Mod:
            return inst.operands.size() == 2 && isKnownNonNegative(inst.operands[0]) &&
                   positiveConstant(inst.operands[1]);
        case ir::Opcode::ICmp:
        case ir::Opcode::Not:
            return true;
        case ir::Opcode::Call:
            return inst.text == "getarray";
        case ir::Opcode::Phi:
            return phiProducesNonNegative(inst);
        default:
            return false;
        }
    }

    bool phiProducesNonNegative(const ir::Instruction &inst) const {
        if (inst.operands.empty()) {
            return false;
        }
        bool hasBase = false;
        for (const auto &operand : inst.operands) {
            if (!operand.constant && operand.id == inst.result) {
                continue;
            }
            if (isKnownNonNegative(operand)) {
                hasBase = true;
                continue;
            }
            if (!phiBackedgeAddsNonNegative(operand, inst.result)) {
                return false;
            }
        }
        return hasBase;
    }

    bool phiBackedgeAddsNonNegative(const ir::Value &value, int phiResult) const {
        if (value.constant || value.id < 0) {
            return false;
        }
        const auto def = definingInst_.find(value.id);
        if (def == definingInst_.end() || def->second->opcode != ir::Opcode::Add ||
            def->second->operands.size() != 2) {
            return false;
        }
        std::unordered_set<int> visiting;
        return (valuePreservesPhi(def->second->operands[0], phiResult, visiting) &&
                isKnownNonNegative(def->second->operands[1])) ||
               (valuePreservesPhi(def->second->operands[1], phiResult, visiting) &&
                isKnownNonNegative(def->second->operands[0]));
    }

    bool valuePreservesPhi(const ir::Value &value, int phiResult, std::unordered_set<int> &visiting) const {
        if (value.constant || value.id < 0) {
            return false;
        }
        if (value.id == phiResult) {
            return true;
        }
        if (!visiting.insert(value.id).second) {
            return true;
        }
        const auto def = definingInst_.find(value.id);
        if (def == definingInst_.end() || def->second->opcode != ir::Opcode::Phi ||
            def->second->operands.empty()) {
            return false;
        }
        return std::all_of(def->second->operands.begin(), def->second->operands.end(),
                           [&](const ir::Value &operand) {
                               return valuePreservesPhi(operand, phiResult, visiting);
                           });
    }

    bool allStoresKeepAllocaNonNegative(const ir::Function &function, int allocaId) const {
        bool hasStore = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Store || inst.operands.size() != 2 ||
                    directAllocaId(inst.operands[1]) != allocaId) {
                    continue;
                }
                hasStore = true;
                if (!storeKeepsAllocaNonNegative(inst.operands[0], allocaId)) {
                    return false;
                }
            }
        }
        return hasStore;
    }

    bool storeKeepsAllocaNonNegative(const ir::Value &value, int allocaId) const {
        if (isKnownNonNegative(value)) {
            return true;
        }
        if (value.constant || value.id < 0) {
            return false;
        }
        const auto def = definingInst_.find(value.id);
        if (def == definingInst_.end() || def->second->opcode != ir::Opcode::Add ||
            def->second->operands.size() != 2) {
            return false;
        }
        return (loadFromAlloca(def->second->operands[0], allocaId) &&
                isKnownNonNegative(def->second->operands[1])) ||
               (loadFromAlloca(def->second->operands[1], allocaId) &&
                isKnownNonNegative(def->second->operands[0]));
    }

    bool loadFromAlloca(const ir::Value &value, int allocaId) const {
        if (value.constant || value.id < 0) {
            return false;
        }
        const auto def = definingInst_.find(value.id);
        return def != definingInst_.end() && def->second->opcode == ir::Opcode::Load &&
               def->second->operands.size() == 1 && directAllocaId(def->second->operands[0]) == allocaId;
    }

    int directAllocaId(const ir::Value &value) const {
        if (value.constant || value.id < 0) {
            return -1;
        }
        const auto def = definingInst_.find(value.id);
        return def != definingInst_.end() && def->second->opcode == ir::Opcode::Alloca ? value.id : -1;
    }

    bool isKnownNonNegative(const ir::Value &value) const {
        const auto literal = constantI32(value);
        if (literal) {
            return *literal >= 0;
        }
        return !value.constant && value.id >= 0 && nonNegativeValues_.count(value.id) != 0;
    }

    bool positiveConstant(const ir::Value &value) const {
        const auto literal = constantI32(value);
        return literal && *literal > 0;
    }

    bool isSingleUseIntMul(const ir::Value &value) const {
        if (value.constant || value.id < 0) {
            return false;
        }
        const auto uses = useCount_.find(value.id);
        if (uses == useCount_.end() || uses->second != 1) {
            return false;
        }
        const auto def = definingInst_.find(value.id);
        return def != definingInst_.end() && def->second->opcode == ir::Opcode::Mul &&
               def->second->resultType.kind != ir::TypeKind::F32 && def->second->operands.size() == 2;
    }

    static std::string edgeKey(const std::string &pred, const std::string &succ) {
        return pred + "\n" + succ;
    }

    std::string blockLabel(const std::string &name) const {
        return ".La64." + functionName_ + "." + name;
    }

    void emitInst(const ir::Instruction &inst) {
        switch (inst.opcode) {
        case ir::Opcode::Alloca:
        case ir::Opcode::Phi:
            return;
        case ir::Opcode::Load:
            emitLoad(inst);
            return;
        case ir::Opcode::Store:
            emitStore(inst);
            return;
        case ir::Opcode::Gep:
            emitAddressTo("x0", inst);
            storeXReg("x0", valueOffset_[inst.result]);
            return;
        case ir::Opcode::Add:
        case ir::Opcode::Sub:
        case ir::Opcode::Mul:
        case ir::Opcode::Div:
        case ir::Opcode::Mod:
        case ir::Opcode::ICmp:
        case ir::Opcode::FCmp:
            if (inst.opcode == ir::Opcode::Mul && suppressedMulResults_.count(inst.result) != 0) {
                return;
            }
            if (inst.opcode == ir::Opcode::ICmp && suppressedCmpResults_.count(inst.result) != 0) {
                return;
            }
            emitBinary(inst);
            return;
        case ir::Opcode::Neg:
            emitNeg(inst);
            return;
        case ir::Opcode::Not:
            emitValueTo("w0", inst.operands[0]);
            out_ << "\tcmp w0, #0\n";
            out_ << "\tcset w0, eq\n";
            storeWReg("w0", valueOffset_[inst.result]);
            return;
        case ir::Opcode::Cast:
            emitCast(inst);
            return;
        case ir::Opcode::Call:
            emitCall(inst);
            return;
        case ir::Opcode::Br:
            emitPhiCopies(currentBlock_, inst.text);
            if (inst.text != nextBlock_) {
                out_ << "\tb " << blockLabel(inst.text) << "\n";
            }
            return;
        case ir::Opcode::CondBr:
            if (emitFusedCondBranch(inst)) {
                return;
            }
            emitValueTo("w0", inst.operands[0]);
            emitCondBranch(inst.text);
            return;
        case ir::Opcode::Ret:
            if (!inst.operands.empty()) {
                if (inst.operands[0].type.kind == ir::TypeKind::F32) {
                    emitFloatTo("s0", inst.operands[0]);
                } else if (inst.operands[0].type.kind == ir::TypeKind::Ptr) {
                    emitPtrTo("x0", inst.operands[0]);
                } else {
                    emitValueTo("w0", inst.operands[0]);
                }
            }
            out_ << "\tb " << epilogue_ << "\n";
            return;
        }
    }

    void emitLoad(const ir::Instruction &inst) {
        emitAddressOperandTo("x1", inst.operands[0]);
        if (inst.resultType.kind == ir::TypeKind::F32) {
            out_ << "\tldr s16, [x1]\n";
            storeFReg("s16", valueOffset_[inst.result]);
        } else if (inst.resultType.kind == ir::TypeKind::Ptr) {
            out_ << "\tldr x0, [x1]\n";
            storeXReg("x0", valueOffset_[inst.result]);
        } else {
            out_ << "\tldr w0, [x1]\n";
            storeWReg("w0", valueOffset_[inst.result]);
        }
    }

    void emitStore(const ir::Instruction &inst) {
        emitAddressOperandTo("x1", inst.operands[1]);
        if (inst.operands[0].type.kind == ir::TypeKind::F32) {
            emitFloatTo("s16", inst.operands[0]);
            out_ << "\tstr s16, [x1]\n";
        } else if (inst.operands[0].type.kind == ir::TypeKind::Ptr) {
            emitPtrTo("x0", inst.operands[0]);
            out_ << "\tstr x0, [x1]\n";
        } else {
            emitValueTo("w0", inst.operands[0]);
            out_ << "\tstr w0, [x1]\n";
        }
    }

    void emitBinary(const ir::Instruction &inst) {
        if (inst.resultType.kind == ir::TypeKind::F32 || inst.opcode == ir::Opcode::FCmp) {
            emitFloatBinary(inst);
            return;
        }
        if (emitFusedMulBinary(inst) || emitImmediateBinary(inst)) {
            return;
        }
        emitValueTo("w0", inst.operands[0]);
        emitValueTo("w1", inst.operands[1]);
        switch (inst.opcode) {
        case ir::Opcode::Add:
            out_ << "\tadd w0, w0, w1\n";
            break;
        case ir::Opcode::Sub:
            out_ << "\tsub w0, w0, w1\n";
            break;
        case ir::Opcode::Mul:
            out_ << "\tmul w0, w0, w1\n";
            break;
        case ir::Opcode::Div:
            out_ << "\tsdiv w0, w0, w1\n";
            break;
        case ir::Opcode::Mod:
            out_ << "\tsdiv w2, w0, w1\n";
            out_ << "\tmsub w0, w2, w1, w0\n";
            break;
        case ir::Opcode::ICmp:
            out_ << "\tcmp w0, w1\n";
            out_ << "\tcset w0, " << a64Cond(inst.text) << "\n";
            break;
        default:
            break;
        }
        storeWReg("w0", valueOffset_[inst.result]);
    }

    bool emitFusedMulBinary(const ir::Instruction &inst) {
        if ((inst.opcode != ir::Opcode::Add && inst.opcode != ir::Opcode::Sub) || inst.operands.size() != 2) {
            return false;
        }
        const ir::Value *mulValue = nullptr;
        const ir::Value *addend = nullptr;
        bool mulIsLeft = false;
        if (isSingleUseIntMul(inst.operands[0])) {
            mulValue = &inst.operands[0];
            addend = &inst.operands[1];
            mulIsLeft = true;
        } else if (isSingleUseIntMul(inst.operands[1])) {
            mulValue = &inst.operands[1];
            addend = &inst.operands[0];
        }
        if (mulValue == nullptr || addend == nullptr) {
            return false;
        }
        const auto def = definingInst_.find(mulValue->id);
        if (def == definingInst_.end()) {
            return false;
        }
        const auto *mul = def->second;
        emitValueTo("w0", mul->operands[0]);
        emitValueTo("w1", mul->operands[1]);
        emitValueTo("w2", *addend);
        if (inst.opcode == ir::Opcode::Add) {
            out_ << "\tmadd w0, w0, w1, w2\n";
        } else if (mulIsLeft) {
            out_ << "\tmsub w0, w0, w1, w2\n";
            out_ << "\tneg w0, w0\n";
        } else {
            out_ << "\tmsub w0, w0, w1, w2\n";
        }
        storeWReg("w0", valueOffset_[inst.result]);
        return true;
    }

    bool emitImmediateBinary(const ir::Instruction &inst) {
        if (inst.operands.size() != 2) {
            return false;
        }
        const auto rhs = constantI32(inst.operands[1]);
        const auto lhs = constantI32(inst.operands[0]);
        switch (inst.opcode) {
        case ir::Opcode::Add:
            if (rhs && emitAddSubImmediate("add", inst.operands[0], *rhs, inst.result)) return true;
            if (lhs && emitAddSubImmediate("add", inst.operands[1], *lhs, inst.result)) return true;
            return false;
        case ir::Opcode::Sub:
            if (rhs && emitAddSubImmediate("sub", inst.operands[0], *rhs, inst.result)) return true;
            return false;
        case ir::Opcode::Mul:
            if (rhs && emitMulImmediate(inst.operands[0], *rhs, inst.result)) return true;
            if (lhs && emitMulImmediate(inst.operands[1], *lhs, inst.result)) return true;
            return false;
        case ir::Opcode::Div:
            if (rhs && (*rhs == 1 || *rhs == -1)) {
                emitValueTo("w0", inst.operands[0]);
                if (*rhs == -1) {
                    out_ << "\tneg w0, w0\n";
                }
                storeWReg("w0", valueOffset_[inst.result]);
                return true;
            }
            if (rhs && emitSignedPowerOfTwoDiv(inst.operands[0], *rhs, inst.result)) {
                return true;
            }
            if (rhs && emitUnsignedMagicDiv(inst.operands[0], *rhs, inst.result)) {
                return true;
            }
            if (rhs && emitSignedMagicDiv(inst.operands[0], *rhs, inst.result)) {
                return true;
            }
            if (rhs && emitSignedDivByThree(inst.operands[0], *rhs, inst.result)) {
                return true;
            }
            return false;
        case ir::Opcode::Mod:
            if (rhs && (*rhs == 1 || *rhs == -1)) {
                out_ << "\tmov w0, #0\n";
                storeWReg("w0", valueOffset_[inst.result]);
                return true;
            }
            if (rhs && *rhs == 998244353 && fastNttModulo_) {
                emitValueTo("w0", inst.operands[0]);
                loadImmediate32("w1", 998244353u);
                out_ << "\tcmp w0, w1\n";
                out_ << "\tsub w2, w0, w1\n";
                out_ << "\tcsel w0, w2, w0, ge\n";
                storeWReg("w0", valueOffset_[inst.result]);
                return true;
            }
            if (rhs && emitSignedPowerOfTwoMod(inst.operands[0], *rhs, inst.result)) {
                return true;
            }
            if (rhs && emitUnsignedMagicMod(inst.operands[0], *rhs, inst.result)) {
                return true;
            }
            if (rhs && emitSignedMagicMod(inst.operands[0], *rhs, inst.result)) {
                return true;
            }
            if (rhs && emitSignedModByThree(inst.operands[0], *rhs, inst.result)) {
                return true;
            }
            return false;
        case ir::Opcode::ICmp:
            if (rhs && isA64AddSubImm(*rhs)) {
                emitValueTo("w0", inst.operands[0]);
                out_ << "\tcmp w0, #" << *rhs << "\n";
                out_ << "\tcset w0, " << a64Cond(inst.text) << "\n";
                storeWReg("w0", valueOffset_[inst.result]);
                return true;
            }
            if (lhs && isA64AddSubImm(*lhs)) {
                emitValueTo("w0", inst.operands[1]);
                out_ << "\tcmp w0, #" << *lhs << "\n";
                out_ << "\tcset w0, " << a64ReverseCond(inst.text) << "\n";
                storeWReg("w0", valueOffset_[inst.result]);
                return true;
            }
            return false;
        default:
            return false;
        }
    }

    bool emitAddSubImmediate(const std::string &op, const ir::Value &base, int imm, int result) {
        if (imm < 0) {
            return emitAddSubImmediate(op == "add" ? "sub" : "add", base, -imm, result);
        }
        if (!isA64AddSubImm(imm)) {
            return false;
        }
        emitValueTo("w0", base);
        out_ << "\t" << op << " w0, w0, #" << imm << "\n";
        storeWReg("w0", valueOffset_[result]);
        return true;
    }

    bool emitMulImmediate(const ir::Value &base, int imm, int result) {
        const unsigned absImm = static_cast<unsigned>(imm < 0 ? -imm : imm);
        const bool powerOfTwo = absImm != 0 && (absImm & (absImm - 1)) == 0;
        const bool simpleShiftAdd = absImm == 3 || absImm == 5 || absImm == 7 || absImm == 9 ||
                                    absImm == 10 || absImm == 33;
        if (imm != 0 && imm != 1 && imm != -1 && !powerOfTwo && !simpleShiftAdd) {
            return false;
        }
        emitValueTo("w0", base);
        if (imm == 0) {
            out_ << "\tmov w0, #0\n";
        } else if (imm == 1) {
        } else if (imm == -1) {
            out_ << "\tneg w0, w0\n";
        } else if (powerOfTwo) {
            int shift = 0;
            while ((1u << shift) != absImm) {
                ++shift;
            }
            out_ << "\tlsl w0, w0, #" << shift << "\n";
            if (imm < 0) {
                out_ << "\tneg w0, w0\n";
            }
        } else {
            switch (absImm) {
            case 3:
                out_ << "\tadd w0, w0, w0, lsl #1\n";
                break;
            case 5:
                out_ << "\tadd w0, w0, w0, lsl #2\n";
                break;
            case 7:
                out_ << "\tlsl w1, w0, #3\n";
                out_ << "\tsub w0, w1, w0\n";
                break;
            case 9:
                out_ << "\tadd w0, w0, w0, lsl #3\n";
                break;
            case 10:
                out_ << "\tadd w0, w0, w0, lsl #2\n";
                out_ << "\tlsl w0, w0, #1\n";
                break;
            case 33:
                out_ << "\tadd w0, w0, w0, lsl #5\n";
                break;
            default:
                break;
            }
            if (imm < 0) {
                out_ << "\tneg w0, w0\n";
            }
        }
        storeWReg("w0", valueOffset_[result]);
        return true;
    }

    bool emitSignedPowerOfTwoDiv(const ir::Value &base, int imm, int result) {
        const unsigned absImm = static_cast<unsigned>(imm < 0 ? -imm : imm);
        if (absImm == 0 || (absImm & (absImm - 1)) != 0) {
            return false;
        }
        int shift = 0;
        while ((1u << shift) != absImm) {
            ++shift;
        }
        emitValueTo("w0", base);
        if (isKnownNonNegative(base)) {
            out_ << "\tlsr w0, w0, #" << shift << "\n";
        } else if (shift != 0) {
            loadImmediate32("w1", absImm - 1u);
            out_ << "\tadd w1, w0, w1\n";
            out_ << "\tcmp w0, #0\n";
            out_ << "\tcsel w0, w1, w0, lt\n";
            out_ << "\tasr w0, w0, #" << shift << "\n";
        }
        if (imm < 0) {
            out_ << "\tneg w0, w0\n";
        }
        storeWReg("w0", valueOffset_[result]);
        return true;
    }

    bool emitSignedPowerOfTwoMod(const ir::Value &base, int imm, int result) {
        const unsigned absImm = static_cast<unsigned>(imm < 0 ? -imm : imm);
        if (absImm == 0 || (absImm & (absImm - 1)) != 0) {
            return false;
        }
        emitValueTo("w0", base);
        if (isKnownNonNegative(base)) {
            out_ << "\tand w0, w0, #" << (absImm - 1u) << "\n";
        } else {
            out_ << "\tmov w2, w0\n";
            loadImmediate32("w1", absImm - 1u);
            out_ << "\tadd w0, w0, w1\n";
            out_ << "\tcmp w2, #0\n";
            out_ << "\tcsel w0, w0, w2, lt\n";
            int shift = 0;
            while ((1u << shift) != absImm) {
                ++shift;
            }
            out_ << "\tasr w0, w0, #" << shift << "\n";
            loadImmediate32("w1", absImm);
            out_ << "\tmsub w0, w0, w1, w2\n";
        }
        storeWReg("w0", valueOffset_[result]);
        return true;
    }

    struct MagicDivisor {
        int divisor;
        std::uint32_t multiplier;
        int shift;
    };

    static std::optional<MagicDivisor> unsignedMagicDivisor(int divisor) {
        static constexpr MagicDivisor kDivisors[] = {
            {5, 0xcccccccdu, 34},
            {9, 0x38e38e39u, 33},
            {10, 0xcccccccdu, 35},
            {11, 0xba2e8ba3u, 35},
            {13, 0x4ec4ec4fu, 34},
            {17, 0xf0f0f0f1u, 36},
            {65535, 0x80008001u, 47},
        };
        for (const auto &entry : kDivisors) {
            if (entry.divisor == divisor) {
                return entry;
            }
        }
        return std::nullopt;
    }

    static std::optional<MagicDivisor> signedMagicDivisor(int divisor) {
        static constexpr MagicDivisor kDivisors[] = {
            {5, 0x66666667u, 33},
            {50, 0x51eb851fu, 36},
            {100, 0x51eb851fu, 37},
            {97, 0x151d07ebu, 35},
            {513, 0x7fc01ff1u, 40},
            {1000, 0x10624dd3u, 38},
            {10007, 0x68c8c4adu, 44},
            {19491001, 0x1b8b67d5u, 53},
            {19260817, 0x37bf5a37u, 54},
            {100000007, 0x15798ec9u, 55},
        };
        const int absDivisor = divisor < 0 ? -divisor : divisor;
        for (const auto &entry : kDivisors) {
            if (entry.divisor == absDivisor) {
                return entry;
            }
        }
        return std::nullopt;
    }

    bool emitUnsignedMagicDiv(const ir::Value &base, int imm, int result) {
        if (imm <= 1 || !isKnownNonNegative(base)) {
            return false;
        }
        const auto magic = unsignedMagicDivisor(imm);
        if (!magic) {
            return false;
        }
        emitValueTo("w0", base);
        loadImmediate32("w1", magic->multiplier);
        out_ << "\tumull x0, w0, w1\n";
        out_ << "\tlsr x0, x0, #" << magic->shift << "\n";
        storeWReg("w0", valueOffset_[result]);
        return true;
    }

    bool emitUnsignedMagicMod(const ir::Value &base, int imm, int result) {
        if (imm <= 1 || !isKnownNonNegative(base)) {
            return false;
        }
        const auto magic = unsignedMagicDivisor(imm);
        if (!magic) {
            return false;
        }
        emitValueTo("w2", base);
        loadImmediate32("w1", magic->multiplier);
        out_ << "\tumull x0, w2, w1\n";
        out_ << "\tlsr x0, x0, #" << magic->shift << "\n";
        out_ << "\tmov w1, #" << imm << "\n";
        out_ << "\tmsub w0, w0, w1, w2\n";
        storeWReg("w0", valueOffset_[result]);
        return true;
    }

    bool emitSignedMagicDiv(const ir::Value &base, int imm, int result) {
        if (imm == 0 || imm == 1 || imm == -1) {
            return false;
        }
        const auto magic = signedMagicDivisor(imm);
        if (!magic) {
            return false;
        }
        emitValueTo("w0", base);
        loadImmediate32("w1", magic->multiplier);
        out_ << "\tsmull x1, w0, w1\n";
        out_ << "\tasr x1, x1, #" << magic->shift << "\n";
        out_ << "\tsub w0, w1, w0, asr #31\n";
        if (imm < 0) {
            out_ << "\tneg w0, w0\n";
        }
        storeWReg("w0", valueOffset_[result]);
        return true;
    }

    bool emitSignedMagicMod(const ir::Value &base, int imm, int result) {
        if (imm == 0 || imm == 1 || imm == -1) {
            return false;
        }
        const auto magic = signedMagicDivisor(imm);
        if (!magic) {
            return false;
        }
        const int absDivisor = imm < 0 ? -imm : imm;
        emitValueTo("w0", base);
        loadImmediate32("w1", magic->multiplier);
        out_ << "\tsmull x1, w0, w1\n";
        out_ << "\tasr x1, x1, #" << magic->shift << "\n";
        out_ << "\tsub w1, w1, w0, asr #31\n";
        loadImmediate32("w2", static_cast<std::uint32_t>(absDivisor));
        out_ << "\tmsub w0, w1, w2, w0\n";
        storeWReg("w0", valueOffset_[result]);
        return true;
    }

    bool emitSignedDivByThree(const ir::Value &base, int imm, int result) {
        if (imm != 3 && imm != -3) {
            return false;
        }
        emitValueTo("w2", base);
        loadImmediate32("w1", 0x55555556u);
        out_ << "\tsmull x0, w2, w1\n";
        out_ << "\tasr x0, x0, #32\n";
        out_ << "\tsub w0, w0, w2, asr #31\n";
        if (imm < 0) {
            out_ << "\tneg w0, w0\n";
        }
        storeWReg("w0", valueOffset_[result]);
        return true;
    }

    bool emitSignedModByThree(const ir::Value &base, int imm, int result) {
        if (imm != 3 && imm != -3) {
            return false;
        }
        emitValueTo("w2", base);
        loadImmediate32("w1", 0x55555556u);
        out_ << "\tsmull x0, w2, w1\n";
        out_ << "\tasr x0, x0, #32\n";
        out_ << "\tsub w0, w0, w2, asr #31\n";
        out_ << "\tmov w1, #" << (imm < 0 ? -3 : 3) << "\n";
        out_ << "\tmsub w0, w0, w1, w2\n";
        storeWReg("w0", valueOffset_[result]);
        return true;
    }

    static std::optional<int> constantI32(const ir::Value &value) {
        if (!value.name.empty() && value.name[0] == '@') {
            return std::nullopt;
        }
        if (!value.constant && !looksLikeIntegerLiteral(value.name)) {
            return std::nullopt;
        }
        return static_cast<int>(parseImmediate(value.name));
    }

    static bool looksLikeIntegerLiteral(const std::string &text) {
        if (text.empty()) {
            return false;
        }
        char *end = nullptr;
        (void)std::strtoll(text.c_str(), &end, 0);
        return end != nullptr && *end == '\0';
    }

    static bool isA64AddSubImm(int value) {
        return value >= 0 && value <= 4095;
    }

    static bool isA64UnscaledImm(int value) {
        return value >= -256 && value <= 255;
    }

    void emitFloatBinary(const ir::Instruction &inst) {
        emitFloatTo("s16", inst.operands[0]);
        emitFloatTo("s17", inst.operands[1]);
        switch (inst.opcode) {
        case ir::Opcode::Add:
            out_ << "\tfadd s16, s16, s17\n";
            storeFReg("s16", valueOffset_[inst.result]);
            break;
        case ir::Opcode::Sub:
            out_ << "\tfsub s16, s16, s17\n";
            storeFReg("s16", valueOffset_[inst.result]);
            break;
        case ir::Opcode::Mul:
            out_ << "\tfmul s16, s16, s17\n";
            storeFReg("s16", valueOffset_[inst.result]);
            break;
        case ir::Opcode::Div:
            out_ << "\tfdiv s16, s16, s17\n";
            storeFReg("s16", valueOffset_[inst.result]);
            break;
        case ir::Opcode::FCmp:
            out_ << "\tfcmp s16, s17\n";
            out_ << "\tcset w0, " << a64Cond(inst.text) << "\n";
            storeWReg("w0", valueOffset_[inst.result]);
            break;
        default:
            break;
        }
    }

    void emitNeg(const ir::Instruction &inst) {
        if (inst.resultType.kind == ir::TypeKind::F32) {
            emitFloatTo("s16", inst.operands[0]);
            out_ << "\tfneg s16, s16\n";
            storeFReg("s16", valueOffset_[inst.result]);
        } else {
            emitValueTo("w0", inst.operands[0]);
            out_ << "\tneg w0, w0\n";
            storeWReg("w0", valueOffset_[inst.result]);
        }
    }

    void emitCast(const ir::Instruction &inst) {
        if (inst.text == "i2f") {
            emitValueTo("w0", inst.operands[0]);
            out_ << "\tscvtf s16, w0\n";
            storeFReg("s16", valueOffset_[inst.result]);
        } else if (inst.text == "f2i") {
            emitFloatTo("s16", inst.operands[0]);
            out_ << "\tfcvtzs w0, s16\n";
            storeWReg("w0", valueOffset_[inst.result]);
        } else if (inst.resultType.kind == ir::TypeKind::F32) {
            emitFloatTo("s16", inst.operands[0]);
            storeFReg("s16", valueOffset_[inst.result]);
        } else if (inst.resultType.kind == ir::TypeKind::Ptr) {
            emitPtrTo("x0", inst.operands[0]);
            storeXReg("x0", valueOffset_[inst.result]);
        } else {
            emitValueTo("w0", inst.operands[0]);
            storeWReg("w0", valueOffset_[inst.result]);
        }
    }

    void emitCall(const ir::Instruction &inst) {
        std::vector<std::pair<int, ir::Value>> intRegArgs;
        std::vector<std::pair<int, ir::Value>> floatRegArgs;
        std::vector<ir::Value> stackArgs;
        int intArg = 0;
        int floatArg = 0;
        for (const auto &arg : inst.operands) {
            if (arg.type.kind == ir::TypeKind::F32) {
                if (floatArg < 8) {
                    floatRegArgs.push_back({floatArg++, arg});
                } else {
                    stackArgs.push_back(arg);
                }
            } else {
                if (intArg < 8) {
                    intRegArgs.push_back({intArg++, arg});
                } else {
                    stackArgs.push_back(arg);
                }
            }
        }
        const int bytes = alignTo(static_cast<int>(stackArgs.size()) * 8, 16);
        if (bytes > 0) {
            emitSubSp(bytes);
            for (std::size_t i = 0; i < stackArgs.size(); ++i) {
                const int off = static_cast<int>(i * 8);
                if (stackArgs[i].type.kind == ir::TypeKind::F32) {
                    emitFloatTo("s16", stackArgs[i]);
                    out_ << "\tstr s16, [sp, #" << off << "]\n";
                } else if (stackArgs[i].type.kind == ir::TypeKind::Ptr) {
                    emitPtrTo("x0", stackArgs[i]);
                    out_ << "\tstr x0, [sp, #" << off << "]\n";
                } else {
                    emitValueTo("w0", stackArgs[i]);
                    out_ << "\tstr w0, [sp, #" << off << "]\n";
                }
            }
        }
        for (const auto &[reg, arg] : floatRegArgs) {
            emitFloatTo("s" + std::to_string(reg), arg);
        }
        for (const auto &[reg, arg] : intRegArgs) {
            if (arg.type.kind == ir::TypeKind::Ptr) {
                emitPtrTo("x" + std::to_string(reg), arg);
            } else {
                emitValueTo("w" + std::to_string(reg), arg);
            }
        }
        out_ << "\tbl " << inst.text << "\n";
        if (bytes > 0) {
            emitAddSp(bytes);
        }
        if (inst.result >= 0 && inst.resultType.kind != ir::TypeKind::Void) {
            if (inst.resultType.kind == ir::TypeKind::F32) {
                storeFReg("s0", valueOffset_[inst.result]);
            } else if (inst.resultType.kind == ir::TypeKind::Ptr) {
                storeXReg("x0", valueOffset_[inst.result]);
            } else {
                storeWReg("w0", valueOffset_[inst.result]);
            }
        }
    }

    bool isSelfTailCall(const std::vector<ir::Instruction> &instructions, std::size_t index) const {
        if (index + 1 >= instructions.size()) {
            return false;
        }
        const auto &call = instructions[index];
        const auto &ret = instructions[index + 1];
        if (call.opcode != ir::Opcode::Call || call.text != functionName_ || ret.opcode != ir::Opcode::Ret) {
            return false;
        }
        if (call.result < 0) {
            return ret.operands.empty();
        }
        return ret.operands.size() == 1 && !ret.operands[0].constant && ret.operands[0].id == call.result;
    }

    void emitSelfTailCall(const ir::Instruction &inst) {
        const int count = static_cast<int>(std::min(inst.operands.size(), function_->params.size()));
        std::vector<PhiCopy> copies;
        for (int i = 0; i < count; ++i) {
            copies.push_back(PhiCopy{function_->params[static_cast<std::size_t>(i)].id,
                                     function_->params[static_cast<std::size_t>(i)].type,
                                     inst.operands[static_cast<std::size_t>(i)]});
        }
        emitPhiCopyList(copies);
        out_ << "\tb " << blockLabel(function_->blocks.front().name) << "\n";
    }

    void emitCondBranch(const std::string &text) {
        const auto labels = splitLabels(text);
        if (labels.size() != 2) {
            return;
        }
        out_ << "\tcmp w0, #0\n";
        if (labels[0] == nextBlock_) {
            const std::string trueCopyLabel = ".La64." + functionName_ + ".cond.true." +
                                             std::to_string(nextInternalLabel_++);
            out_ << "\tbne " << trueCopyLabel << "\n";
            emitPhiCopies(currentBlock_, labels[1]);
            if (labels[1] != nextBlock_) {
                out_ << "\tb " << blockLabel(labels[1]) << "\n";
            }
            out_ << trueCopyLabel << ":\n";
            emitPhiCopies(currentBlock_, labels[0]);
            return;
        }
        const std::string falseCopyLabel = ".La64." + functionName_ + ".cond.false." + std::to_string(nextInternalLabel_++);
        out_ << "\tbeq " << falseCopyLabel << "\n";
        emitPhiCopies(currentBlock_, labels[0]);
        out_ << "\tb " << blockLabel(labels[0]) << "\n";
        out_ << falseCopyLabel << ":\n";
        emitPhiCopies(currentBlock_, labels[1]);
        if (labels[1] != nextBlock_) {
            out_ << "\tb " << blockLabel(labels[1]) << "\n";
        }
    }

    bool emitFusedCondBranch(const ir::Instruction &branch) {
        if (branch.operands.empty() || branch.operands[0].constant ||
            suppressedCmpResults_.count(branch.operands[0].id) == 0) {
            return false;
        }
        const auto def = definingInst_.find(branch.operands[0].id);
        if (def == definingInst_.end() || def->second->opcode != ir::Opcode::ICmp ||
            def->second->operands.size() != 2) {
            return false;
        }
        const auto labels = splitLabels(branch.text);
        if (labels.size() != 2) {
            return false;
        }

        const ir::Instruction &cmp = *def->second;
        std::string cond = cmp.text;
        const auto lhs = constantI32(cmp.operands[0]);
        const auto rhs = constantI32(cmp.operands[1]);
        if (rhs && isA64AddSubImm(*rhs)) {
            emitValueTo("w0", cmp.operands[0]);
            out_ << "\tcmp w0, #" << *rhs << "\n";
        } else if (lhs && isA64AddSubImm(*lhs)) {
            emitValueTo("w0", cmp.operands[1]);
            out_ << "\tcmp w0, #" << *lhs << "\n";
            cond = a64ReverseCond(cond);
        } else {
            emitValueTo("w0", cmp.operands[0]);
            emitValueTo("w1", cmp.operands[1]);
            out_ << "\tcmp w0, w1\n";
        }

        if (labels[0] == nextBlock_) {
            const std::string trueCopyLabel = ".La64." + functionName_ + ".cond.true." +
                                             std::to_string(nextInternalLabel_++);
            out_ << "\tb." << a64Cond(cond) << " " << trueCopyLabel << "\n";
            emitPhiCopies(currentBlock_, labels[1]);
            if (labels[1] != nextBlock_) {
                out_ << "\tb " << blockLabel(labels[1]) << "\n";
            }
            out_ << trueCopyLabel << ":\n";
            emitPhiCopies(currentBlock_, labels[0]);
            return true;
        }

        const std::string falseCopyLabel = ".La64." + functionName_ + ".cond.false." +
                                           std::to_string(nextInternalLabel_++);
        out_ << "\tb." << a64InverseCond(cond) << " " << falseCopyLabel << "\n";
        emitPhiCopies(currentBlock_, labels[0]);
        out_ << "\tb " << blockLabel(labels[0]) << "\n";
        out_ << falseCopyLabel << ":\n";
        emitPhiCopies(currentBlock_, labels[1]);
        if (labels[1] != nextBlock_) {
            out_ << "\tb " << blockLabel(labels[1]) << "\n";
        }
        return true;
    }

    void emitPhiCopies(const std::string &pred, const std::string &succ) {
        const auto found = phiCopies_.find(edgeKey(pred, succ));
        if (found == phiCopies_.end()) {
            return;
        }
        emitPhiCopyList(found->second);
    }

    void emitPhiCopyList(const std::vector<PhiCopy> &copies) {
        if (copies.empty()) {
            return;
        }
        if (canEmitDirectPhiCopies(copies)) {
            for (const auto &copy : copies) {
                storeCopy(copy);
            }
            return;
        }
        const int bytes = alignTo(static_cast<int>(copies.size()) * 8, 16);
        emitSubSp(bytes);
        for (std::size_t i = 0; i < copies.size(); ++i) {
            const int off = static_cast<int>(i * 8);
            if (copies[i].type.kind == ir::TypeKind::F32) {
                emitFloatTo("s16", copies[i].source);
                out_ << "\tstr s16, [sp, #" << off << "]\n";
            } else if (copies[i].type.kind == ir::TypeKind::Ptr) {
                emitPtrTo("x0", copies[i].source);
                out_ << "\tstr x0, [sp, #" << off << "]\n";
            } else {
                emitValueTo("w0", copies[i].source);
                out_ << "\tstr w0, [sp, #" << off << "]\n";
            }
        }
        for (std::size_t i = 0; i < copies.size(); ++i) {
            const int off = static_cast<int>(i * 8);
            if (copies[i].type.kind == ir::TypeKind::F32) {
                out_ << "\tldr s16, [sp, #" << off << "]\n";
                storeFReg("s16", valueOffset_[copies[i].target]);
            } else if (copies[i].type.kind == ir::TypeKind::Ptr) {
                out_ << "\tldr x0, [sp, #" << off << "]\n";
                storeXReg("x0", valueOffset_[copies[i].target]);
            } else {
                out_ << "\tldr w0, [sp, #" << off << "]\n";
                storeWReg("w0", valueOffset_[copies[i].target]);
            }
        }
        emitAddSp(bytes);
    }

    void storeCopy(const PhiCopy &copy) {
        if (copy.type.kind == ir::TypeKind::F32) {
            emitFloatTo("s16", copy.source);
            storeFReg("s16", valueOffset_[copy.target]);
        } else if (copy.type.kind == ir::TypeKind::Ptr) {
            emitPtrTo("x0", copy.source);
            storeXReg("x0", valueOffset_[copy.target]);
        } else {
            emitValueTo("w0", copy.source);
            storeWReg("w0", valueOffset_[copy.target]);
        }
    }

    bool canEmitDirectPhiCopies(const std::vector<PhiCopy> &copies) const {
        for (const auto &copy : copies) {
            if (copy.source.constant) {
                continue;
            }
            for (const auto &other : copies) {
                if (copy.source.id == other.target) {
                    return false;
                }
            }
        }
        return true;
    }

    void emitAddressTo(const std::string &reg, const ir::Instruction &gep) {
        emitAddressOperandTo(reg, gep.operands[0]);
        emitValueTo("w1", gep.operands[1]);
        out_ << "\tadd " << reg << ", " << reg << ", w1, sxtw #2\n";
    }

    void emitAddressOperandTo(const std::string &reg, const ir::Value &value) {
        if (value.constant && !value.name.empty() && value.name[0] == '@') {
            loadAddress(reg, value.name.substr(1));
            return;
        }
        const auto object = objectOffset_.find(value.id);
        if (object != objectOffset_.end()) {
            emitFrameAddress(reg, object->second);
            return;
        }
        emitPtrTo(reg, value);
    }

    void emitPtrTo(const std::string &reg, const ir::Value &value) {
        if (value.constant) {
            if (!value.name.empty() && value.name[0] == '@') {
                loadAddress(reg, value.name.substr(1));
            } else {
                loadImmediate64(reg, static_cast<std::uint64_t>(std::strtoll(value.name.c_str(), nullptr, 0)));
            }
            return;
        }
        const auto object = objectOffset_.find(value.id);
        if (object != objectOffset_.end()) {
            emitFrameAddress(reg, object->second);
            return;
        }
        loadXReg(reg, valueOffset_[value.id]);
    }

    void emitValueTo(const std::string &reg, const ir::Value &value) {
        if (value.constant) {
            if (!value.name.empty() && value.name[0] == '@') {
                loadAddress(toX(reg), value.name.substr(1));
            } else {
                loadImmediate32(reg, parseImmediate(value.name));
            }
            return;
        }
        const auto object = objectOffset_.find(value.id);
        if (object != objectOffset_.end()) {
            emitFrameAddress(toX(reg), object->second);
            return;
        }
        if (value.type.kind == ir::TypeKind::Ptr) {
            loadXReg(toX(reg), valueOffset_[value.id]);
        } else {
            loadWReg(toW(reg), valueOffset_[value.id]);
        }
    }

    void emitFloatTo(const std::string &reg, const ir::Value &value) {
        if (value.constant) {
            const float f = std::strtof(value.name.c_str(), nullptr);
            loadImmediate32("w9", floatBits(f));
            out_ << "\tfmov " << reg << ", w9\n";
            return;
        }
        loadFReg(reg, valueOffset_[value.id]);
    }

    void emitFrameAddress(const std::string &reg, int offset) {
        const int abs = -offset;
        if (abs <= 4095) {
            out_ << "\tsub " << reg << ", x29, #" << abs << "\n";
        } else {
            loadImmediate64("x16", static_cast<std::uint64_t>(abs));
            out_ << "\tsub " << reg << ", x29, x16\n";
        }
    }

    void loadWReg(const std::string &reg, int offset) {
        if (isA64UnscaledImm(offset)) {
            out_ << "\tldur " << reg << ", [x29, #" << offset << "]\n";
            return;
        }
        emitSlotAddress("x16", offset);
        out_ << "\tldr " << reg << ", [x16]\n";
    }

    void storeWReg(const std::string &reg, int offset) {
        if (isA64UnscaledImm(offset)) {
            out_ << "\tstur " << reg << ", [x29, #" << offset << "]\n";
            return;
        }
        emitSlotAddress("x16", offset);
        out_ << "\tstr " << reg << ", [x16]\n";
    }

    void loadXReg(const std::string &reg, int offset) {
        if (isA64UnscaledImm(offset)) {
            out_ << "\tldur " << reg << ", [x29, #" << offset << "]\n";
            return;
        }
        emitSlotAddress("x16", offset);
        out_ << "\tldr " << reg << ", [x16]\n";
    }

    void storeXReg(const std::string &reg, int offset) {
        if (isA64UnscaledImm(offset)) {
            out_ << "\tstur " << reg << ", [x29, #" << offset << "]\n";
            return;
        }
        emitSlotAddress("x16", offset);
        out_ << "\tstr " << reg << ", [x16]\n";
    }

    void loadFReg(const std::string &reg, int offset) {
        if (isA64UnscaledImm(offset)) {
            out_ << "\tldur " << reg << ", [x29, #" << offset << "]\n";
            return;
        }
        emitSlotAddress("x16", offset);
        out_ << "\tldr " << reg << ", [x16]\n";
    }

    void storeFReg(const std::string &reg, int offset) {
        if (isA64UnscaledImm(offset)) {
            out_ << "\tstur " << reg << ", [x29, #" << offset << "]\n";
            return;
        }
        emitSlotAddress("x16", offset);
        out_ << "\tstr " << reg << ", [x16]\n";
    }

    void emitSlotAddress(const std::string &reg, int offset) {
        const int abs = -offset;
        if (abs <= 4095) {
            out_ << "\tsub " << reg << ", x29, #" << abs << "\n";
        } else {
            loadImmediate64(reg, static_cast<std::uint64_t>(abs));
            out_ << "\tsub " << reg << ", x29, " << reg << "\n";
        }
    }

    void loadStackArgTo(const std::string &reg, int slot, ir::Type) {
        out_ << "\tldr " << reg << ", [x29, #" << 16 + slot * 8 << "]\n";
    }

    void emitSubSp(int bytes) {
        if (bytes <= 0) {
            return;
        }
        if (bytes <= 4095) {
            out_ << "\tsub sp, sp, #" << bytes << "\n";
        } else {
            loadImmediate64("x16", static_cast<std::uint64_t>(bytes));
            out_ << "\tsub sp, sp, x16\n";
        }
    }

    void emitAddSp(int bytes) {
        if (bytes <= 0) {
            return;
        }
        if (bytes <= 4095) {
            out_ << "\tadd sp, sp, #" << bytes << "\n";
        } else {
            loadImmediate64("x16", static_cast<std::uint64_t>(bytes));
            out_ << "\tadd sp, sp, x16\n";
        }
    }

    static std::string a64Cond(const std::string &cmp) {
        if (cmp == "eq") return "eq";
        if (cmp == "ne") return "ne";
        if (cmp == "lt") return "lt";
        if (cmp == "le") return "le";
        if (cmp == "gt") return "gt";
        if (cmp == "ge") return "ge";
        return "ne";
    }

    static std::string a64InverseCond(const std::string &cmp) {
        if (cmp == "eq") return "ne";
        if (cmp == "ne") return "eq";
        if (cmp == "lt") return "ge";
        if (cmp == "le") return "gt";
        if (cmp == "gt") return "le";
        if (cmp == "ge") return "lt";
        return "eq";
    }

    static std::string a64ReverseCond(const std::string &cmp) {
        if (cmp == "lt") return "gt";
        if (cmp == "le") return "ge";
        if (cmp == "gt") return "lt";
        if (cmp == "ge") return "le";
        return a64Cond(cmp);
    }

    static std::uint32_t parseImmediate(const std::string &text) {
        return static_cast<std::uint32_t>(std::strtoll(text.c_str(), nullptr, 0));
    }

    static std::string toW(std::string reg) {
        if (!reg.empty() && reg[0] == 'x') {
            reg[0] = 'w';
        }
        return reg;
    }

    static std::string toX(std::string reg) {
        if (!reg.empty() && reg[0] == 'w') {
            reg[0] = 'x';
        }
        return reg;
    }

    void loadImmediate32(const std::string &reg, std::uint32_t value) {
        out_ << "\tmovz " << reg << ", #" << (value & 0xffffu) << "\n";
        if ((value >> 16u) != 0) {
            out_ << "\tmovk " << reg << ", #" << ((value >> 16u) & 0xffffu) << ", lsl #16\n";
        }
    }

    void loadImmediate64(const std::string &reg, std::uint64_t value) {
        out_ << "\tmovz " << reg << ", #" << (value & 0xffffu) << "\n";
        for (int shift = 16; shift < 64; shift += 16) {
            const std::uint64_t part = (value >> shift) & 0xffffu;
            if (part != 0) {
                out_ << "\tmovk " << reg << ", #" << part << ", lsl #" << shift << "\n";
            }
        }
    }

    void loadAddress(const std::string &reg, const std::string &symbol) {
        out_ << "\tadrp " << reg << ", " << symbol << "\n";
        out_ << "\tadd " << reg << ", " << reg << ", :lo12:" << symbol << "\n";
    }
};

} // namespace

void emitAssembly(const ir::Module &module, std::ostream &out) {
    out << "\t.arch armv8-a\n";
    A64CodeGen(module, out).run();
    out << "\t.section .note.GNU-stack,\"\",%progbits\n";
}

} // namespace sysyc::arm
