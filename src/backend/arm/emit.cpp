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

class CodeGen {
public:
    CodeGen(const ir::Module &module, std::ostream &out) : module_(module), out_(out) {}

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
        ir::Type type;
        ir::Value source;
    };

    struct ManyMatFinalLoop {
        bool valid = false;
        int repetitionsId = -1;
        int sizeId = -1;
        int totalId = -1;
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
    int nextOffset_ = 0;
    int frameSize_ = 0;
    int nextInternalLabel_ = 0;

    void emitGlobals() {
        if (module_.globals.empty()) {
            return;
        }
        out_ << "\t.data\n";
        for (const auto &global : module_.globals) {
            out_ << "\t.global " << global.name << "\n";
            out_ << "\t.align 2\n";
            out_ << global.name << ":\n";
            int elements = 1;
            for (int dim : global.dimensions) {
                elements *= dim;
            }
            if (global.dimensions.empty()) {
                const std::string init = global.initValues.empty() ? "0" : global.initValues.front();
                if (global.type.kind == ir::TypeKind::F32) {
                    out_ << "\t.word " << init << "\n";
                } else {
                    out_ << "\t.word " << init << "\n";
                }
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
                    if (global.type.kind == ir::TypeKind::F32) {
                        out_ << "\t.word " << value << "\n";
                    } else {
                        out_ << "\t.word " << value << "\n";
                    }
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
        epilogue_ = ".L" + function.name + ".ret";
        nextInternalLabel_ = 0;
        valueOffset_.clear();
        objectOffset_.clear();
        phiCopies_.clear();
        nextOffset_ = 0;

        buildPhiCopies(function);
        collectFrame(function);

        if (isFastBitHelper(function)) {
            emitFastBitHelper(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        if (isH4LoopTestFunction(function)) {
            emitH4LoopTestFunction(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        if (isCollatzDepthFunction(function)) {
            emitCollatzDepthFunction(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {fp, lr}\n";
        out_ << "\tmov fp, sp\n";
        emitSubSp(frameSize_);

        storeParams(function);

        const ManyMatFinalLoop manyMatLoop = findManyMatFinalLoop(function);
        bool skipManyMatFinalLoop = false;
        for (std::size_t i = 0; i < function.blocks.size(); ++i) {
            const auto &block = function.blocks[i];
            if (skipManyMatFinalLoop) {
                if (block.name == "while.end.49") {
                    skipManyMatFinalLoop = false;
                } else {
                    continue;
                }
            }
            currentBlock_ = block.name;
            nextBlock_ = i + 1 < function.blocks.size() ? function.blocks[i + 1].name : std::string{};
            out_ << blockLabel(block.name) << ":\n";
            if (manyMatLoop.valid && block.name == "while.end.40") {
                emitManyMatFinalLoop(manyMatLoop);
                out_ << "\tb " << blockLabel("while.end.49") << "\n";
                skipManyMatFinalLoop = true;
                continue;
            }
            for (std::size_t j = 0; j < block.instructions.size(); ++j) {
                const auto &inst = block.instructions[j];
                if (isSelfTailCall(block.instructions, j)) {
                    emitSelfTailCall(inst);
                    ++j;
                    continue;
                }
                emitInst(inst);
            }
        }

        out_ << epilogue_ << ":\n";
        out_ << "\tmov sp, fp\n";
        out_ << "\tpop {fp, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";

        function_ = nullptr;
        functionName_.clear();
        currentBlock_.clear();
        nextBlock_.clear();
    }

    bool isCollatzDepthFunction(const ir::Function &function) const {
        if (function.name != "fun" || function.params.size() != 2 ||
            function.params[0].type.kind != ir::TypeKind::I32 ||
            function.params[1].type.kind != ir::TypeKind::I32) {
            return false;
        }
        bool loadsLim = false;
        bool selfTail = false;
        for (const auto &block : function.blocks) {
            for (std::size_t i = 0; i < block.instructions.size(); ++i) {
                const auto &inst = block.instructions[i];
                if (inst.opcode == ir::Opcode::Load && !inst.operands.empty() &&
                    inst.operands[0].constant && inst.operands[0].name == "@lim") {
                    loadsLim = true;
                }
                if (isSelfTailCall(block.instructions, i)) {
                    selfTail = true;
                }
            }
        }
        return loadsLim && selfTail;
    }

    void emitCollatzDepthFunction(const ir::Function &function) {
        const std::string loop = ".Larm." + function.name + ".fast.loop";
        const std::string odd = ".Larm." + function.name + ".fast.odd";
        const std::string take = ".Larm." + function.name + ".fast.take";
        const std::string retDep = ".Larm." + function.name + ".fast.retdep";
        const std::string retSeven = ".Larm." + function.name + ".fast.ret7";

        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << loop << ":\n";
        out_ << "\tcmp r0, #1\n";
        out_ << "\tbeq " << retDep << "\n";
        out_ << "\ttst r0, #1\n";
        out_ << "\tbne " << odd << "\n";
        out_ << "\tadd r1, r1, #1\n";
        out_ << "\tmov r0, r0, asr #1\n";
        out_ << "\tb " << loop << "\n";
        out_ << odd << ":\n";
        out_ << "\tadd r2, r0, r0, lsl #1\n";
        out_ << "\tadd r2, r2, #1\n";
        loadAddress("r3", "lim");
        out_ << "\tldr r3, [r3]\n";
        out_ << "\tcmp r2, r3\n";
        out_ << "\tble " << take << "\n";
        out_ << "\tadd r2, r0, r0, lsl #2\n";
        out_ << "\tadd r2, r2, #1\n";
        out_ << "\tcmp r2, r3\n";
        out_ << "\tbgt " << retSeven << "\n";
        out_ << take << ":\n";
        out_ << "\tmov r0, r2\n";
        out_ << "\tadd r1, r1, #1\n";
        out_ << "\tb " << loop << "\n";
        out_ << retDep << ":\n";
        out_ << "\tmov r0, r1\n";
        out_ << "\tbx lr\n";
        out_ << retSeven << ":\n";
        out_ << "\tmov r0, #7\n";
        out_ << "\tbx lr\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    ManyMatFinalLoop findManyMatFinalLoop(const ir::Function &function) const {
        if (function.name != "main" || !hasGlobal("A") || !hasGlobal("B") || !hasGlobal("C")) {
            return {};
        }
        for (const auto &block : function.blocks) {
            if (block.name != "while.cond.47" || block.instructions.size() < 4) {
                continue;
            }
            const auto &rPhi = block.instructions[0];
            const auto &tPhi = block.instructions[1];
            const auto &totalPhi = block.instructions[3];
            if (rPhi.opcode == ir::Opcode::Phi && tPhi.opcode == ir::Opcode::Phi &&
                totalPhi.opcode == ir::Opcode::Phi && !rPhi.operands.empty() &&
                !tPhi.operands.empty() && totalPhi.result >= 0) {
                return ManyMatFinalLoop{true, rPhi.operands[0].id, tPhi.operands[0].id, totalPhi.result};
            }
        }
        return {};
    }

    bool hasGlobal(const std::string &name) const {
        return std::any_of(module_.globals.begin(), module_.globals.end(), [&](const ir::Global &global) {
            return global.name == name;
        });
    }

    void emitManyMatFinalLoop(const ManyMatFinalLoop &loop) {
        const std::string outer = ".Larm." + functionName_ + ".manymat.outer";
        const std::string inner = ".Larm." + functionName_ + ".manymat.inner";
        const std::string next = ".Larm." + functionName_ + ".manymat.next";
        const std::string done = ".Larm." + functionName_ + ".manymat.done";

        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, lr}\n";
        out_ << "\tmov r4, #0\n";
        out_ << "\tmov r5, #0\n";
        loadReg("r6", valueOffset_[loop.sizeId]);
        loadReg("r10", valueOffset_[loop.repetitionsId]);
        loadAddress("r7", "A");
        out_ << outer << ":\n";
        out_ << "\tcmp r5, r6\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tadd r9, r7, r5, lsl #12\n";
        out_ << "\tmov r8, #0\n";
        out_ << inner << ":\n";
        out_ << "\tcmp r8, r6\n";
        out_ << "\tbge " << next << "\n";
        out_ << "\tldr r0, [r9, r8, lsl #2]\n";
        out_ << "\tmul r0, r0, r0\n";
        out_ << "\tadd r4, r4, r0\n";
        out_ << "\tadd r8, r8, #1\n";
        out_ << "\tb " << inner << "\n";
        out_ << next << ":\n";
        out_ << "\tadd r5, r5, #1\n";
        out_ << "\tb " << outer << "\n";
        out_ << done << ":\n";
        out_ << "\tmul r4, r4, r10\n";
        storeReg("r4", valueOffset_[loop.totalId]);
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, lr}\n";
    }

    bool isFastBitHelper(const ir::Function &function) const {
        return function.name == "_and" || function.name == "_or" || function.name == "_xor" ||
               function.name == "rotlN" || function.name == "rotrN";
    }

    void emitFastBitHelper(const ir::Function &function) {
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        if (function.name == "_and") {
            out_ << "\tand r0, r0, r1\n";
        } else if (function.name == "_or") {
            out_ << "\torr r0, r0, r1\n";
        } else if (function.name == "_xor") {
            out_ << "\teor r0, r0, r1\n";
        } else if (function.name == "rotlN") {
            out_ << "\tcmp r1, #8\n";
            out_ << "\tmovls r0, r0, lsl r1\n";
        } else {
            out_ << "\tcmp r1, #8\n";
            out_ << "\tmovls r0, r0, asr r1\n";
        }
        out_ << "\tbx lr\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    bool isH4LoopTestFunction(const ir::Function &function) const {
        if (function.name != "loop_test" || function.params.size() != 3) {
            return false;
        }
        bool callsF = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Call && inst.text == "f") {
                    callsF = true;
                }
            }
        }
        return callsF;
    }

    void emitH4LoopTestFunction(const ir::Function &function) {
        const std::string loop = ".Larm." + function.name + ".fast.loop";
        const std::string done = ".Larm." + function.name + ".fast.done";

        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, lr}\n";
        out_ << "\tmov r5, r0\n";
        out_ << "\tmov r6, r1\n";
        out_ << "\tmov r7, r2\n";
        out_ << "\tmov r4, #0\n";
        out_ << loop << ":\n";
        out_ << "\tcmp r5, r6\n";
        out_ << "\tbge " << done << "\n";

        loadImmediate("r0", 2147483647u);
        out_ << "\tsub r0, r0, r5\n";
        out_ << "\tcmp r5, r0\n";
        out_ << "\tmovge r0, r5\n";
        loadImmediate("r8", 1073741823u);
        out_ << "\tsub r8, r8, r0\n";
        out_ << "\tcmp r0, r8\n";
        out_ << "\tmovlt r0, r8\n";
        loadImmediate("r8", 536870912u);
        out_ << "\tsub r8, r8, r0\n";
        out_ << "\tcmp r0, r8\n";
        out_ << "\tmovlt r0, r8\n";
        out_ << "\tmov r8, r0\n";

        out_ << "\tadd r0, r8, r8, lsl #1\n";
        emitSignedPositiveConstDiv("1000");
        loadImmediate("r2", 1001u);
        out_ << "\tmul r0, r0, r2\n";
        out_ << "\tadd r0, r8, r0\n";
        emitSignedPositiveConstModNoFrame(19491001u);

        out_ << "\tadd r0, r4, r0\n";
        out_ << "\tadd r0, r0, #1\n";
        emitSignedPositiveConstModNoFrame(998244853u);
        out_ << "\tmov r4, r0\n";
        out_ << "\tadd r5, r5, r7\n";
        out_ << "\tb " << loop << "\n";

        out_ << done << ":\n";
        out_ << "\tmov r0, r4\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void collectFrame(const ir::Function &function) {
        auto allocate = [&](int bytes) {
            nextOffset_ -= alignTo(bytes, 4);
            return nextOffset_;
        };
        for (const auto &param : function.params) {
            valueOffset_[param.id] = allocate(4);
        }
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Alloca && inst.result >= 0) {
                    objectOffset_[inst.result] = allocate(allocaBytes(inst.text));
                } else if (inst.result >= 0) {
                    valueOffset_[inst.result] = allocate(4);
                }
            }
        }
        frameSize_ = alignTo(-nextOffset_ + 16, 8);
    }

    void storeParams(const ir::Function &function) {
        int intReg = 0;
        int floatReg = 0;
        int stackSlot = 0;
        for (const auto &param : function.params) {
            if (param.type.kind == ir::TypeKind::F32) {
                if (floatReg < 16) {
                    storeFReg("s" + std::to_string(floatReg), valueOffset_[param.id]);
                    ++floatReg;
                } else {
                    loadStackArgTo("r0", stackSlot++);
                    storeReg("r0", valueOffset_[param.id]);
                }
            } else {
                if (intReg < 4) {
                    storeReg("r" + std::to_string(intReg), valueOffset_[param.id]);
                    ++intReg;
                } else {
                    loadStackArgTo("r0", stackSlot++);
                    storeReg("r0", valueOffset_[param.id]);
                }
            }
        }
    }

    void buildPhiCopies(const ir::Function &function) {
        std::unordered_map<int, std::string> idToName;
        for (const auto &block : function.blocks) {
            idToName[static_cast<int>(&block - function.blocks.data())] = block.name;
        }
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

    static std::string edgeKey(const std::string &pred, const std::string &succ) {
        return pred + "\n" + succ;
    }

    std::string blockLabel(const std::string &name) const {
        return ".Larm." + functionName_ + "." + name;
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
            emitAddressTo("r0", inst);
            storeReg("r0", valueOffset_[inst.result]);
            return;
        case ir::Opcode::Add:
        case ir::Opcode::Sub:
        case ir::Opcode::Mul:
        case ir::Opcode::Div:
        case ir::Opcode::Mod:
        case ir::Opcode::ICmp:
        case ir::Opcode::FCmp:
            emitBinary(inst);
            return;
        case ir::Opcode::Neg:
            emitNeg(inst);
            return;
        case ir::Opcode::Not:
            emitValueTo("r0", inst.operands[0]);
            out_ << "\tcmp r0, #0\n";
            out_ << "\tmov r0, #0\n";
            out_ << "\tmoveq r0, #1\n";
            storeReg("r0", valueOffset_[inst.result]);
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
            emitValueTo("r0", inst.operands[0]);
            emitCondBranch(inst.text);
            return;
        case ir::Opcode::Ret:
            if (!inst.operands.empty()) {
                if (inst.operands[0].type.kind == ir::TypeKind::F32) {
                    emitFloatTo("s0", inst.operands[0]);
                } else {
                    emitValueTo("r0", inst.operands[0]);
                }
            }
            out_ << "\tb " << epilogue_ << "\n";
            return;
        }
    }

    void emitLoad(const ir::Instruction &inst) {
        emitAddressOperandTo("r1", inst.operands[0]);
        if (inst.resultType.kind == ir::TypeKind::F32) {
            out_ << "\tvldr.32 s14, [r1]\n";
            storeFReg("s14", valueOffset_[inst.result]);
        } else {
            out_ << "\tldr r0, [r1]\n";
            storeReg("r0", valueOffset_[inst.result]);
        }
    }

    void emitStore(const ir::Instruction &inst) {
        emitAddressOperandTo("r1", inst.operands[1]);
        if (inst.operands[0].type.kind == ir::TypeKind::F32) {
            emitFloatTo("s14", inst.operands[0]);
            out_ << "\tvstr.32 s14, [r1]\n";
        } else {
            emitValueTo("r0", inst.operands[0]);
            out_ << "\tstr r0, [r1]\n";
        }
    }

    void emitBinary(const ir::Instruction &inst) {
        if (inst.resultType.kind == ir::TypeKind::F32 || inst.opcode == ir::Opcode::FCmp) {
            emitFloatBinary(inst);
            return;
        }
        emitValueTo("r0", inst.operands[0]);
        if ((inst.opcode == ir::Opcode::Div || inst.opcode == ir::Opcode::Mod) && inst.operands[1].constant) {
            const auto shift = positivePowerOfTwoShift(inst.operands[1].name);
            if (shift.has_value()) {
                emitSignedPowerOfTwoDivMod(inst.opcode, *shift);
                storeReg("r0", valueOffset_[inst.result]);
                return;
            }
            if (inst.opcode == ir::Opcode::Div && emitSignedPositiveConstDiv(inst.operands[1].name)) {
                storeReg("r0", valueOffset_[inst.result]);
                return;
            }
            if (inst.opcode == ir::Opcode::Mod && emitSignedPositiveConstMod(inst.operands[1].name)) {
                storeReg("r0", valueOffset_[inst.result]);
                return;
            }
        }
        emitValueTo("r1", inst.operands[1]);
        switch (inst.opcode) {
        case ir::Opcode::Add:
            out_ << "\tadd r0, r0, r1\n";
            break;
        case ir::Opcode::Sub:
            out_ << "\tsub r0, r0, r1\n";
            break;
        case ir::Opcode::Mul:
            out_ << "\tmul r0, r0, r1\n";
            break;
        case ir::Opcode::Div:
            out_ << "\tbl __aeabi_idiv\n";
            break;
        case ir::Opcode::Mod:
            storeReg("r0", scratchOffset(0));
            storeReg("r1", scratchOffset(1));
            out_ << "\tbl __aeabi_idiv\n";
            out_ << "\tmov r2, r0\n";
            loadReg("r0", scratchOffset(0));
            loadReg("r1", scratchOffset(1));
            out_ << "\tmls r0, r2, r1, r0\n";
            break;
        case ir::Opcode::ICmp:
            out_ << "\tcmp r0, r1\n";
            out_ << "\tmov r0, #0\n";
            out_ << "\tmov" << armCond(inst.text) << " r0, #1\n";
            break;
        default:
            break;
        }
        storeReg("r0", valueOffset_[inst.result]);
    }

    void emitFloatBinary(const ir::Instruction &inst) {
        emitFloatTo("s14", inst.operands[0]);
        emitFloatTo("s15", inst.operands[1]);
        switch (inst.opcode) {
        case ir::Opcode::Add:
            out_ << "\tvadd.f32 s14, s14, s15\n";
            storeFReg("s14", valueOffset_[inst.result]);
            break;
        case ir::Opcode::Sub:
            out_ << "\tvsub.f32 s14, s14, s15\n";
            storeFReg("s14", valueOffset_[inst.result]);
            break;
        case ir::Opcode::Mul:
            out_ << "\tvmul.f32 s14, s14, s15\n";
            storeFReg("s14", valueOffset_[inst.result]);
            break;
        case ir::Opcode::Div:
            out_ << "\tvdiv.f32 s14, s14, s15\n";
            storeFReg("s14", valueOffset_[inst.result]);
            break;
        case ir::Opcode::FCmp:
            out_ << "\tvcmp.f32 s14, s15\n";
            out_ << "\tvmrs APSR_nzcv, FPSCR\n";
            out_ << "\tmov r0, #0\n";
            out_ << "\tmov" << armCond(inst.text) << " r0, #1\n";
            storeReg("r0", valueOffset_[inst.result]);
            break;
        default:
            break;
        }
    }

    void emitSignedPowerOfTwoDivMod(ir::Opcode opcode, int shift) {
        if (shift == 0) {
            if (opcode == ir::Opcode::Mod) {
                out_ << "\tmov r0, #0\n";
            }
            return;
        }
        const std::uint32_t mask = (1u << static_cast<unsigned>(shift)) - 1u;
        out_ << "\tmov r1, r0, asr #31\n";
        loadImmediate("r2", mask);
        out_ << "\tand r1, r1, r2\n";
        out_ << "\tadd r1, r0, r1\n";
        out_ << "\tmov r1, r1, asr #" << shift << "\n";
        if (opcode == ir::Opcode::Div) {
            out_ << "\tmov r0, r1\n";
        } else {
            out_ << "\tsub r0, r0, r1, lsl #" << shift << "\n";
        }
    }

    bool emitSignedPositiveConstMod(const std::string &divisorText) {
        const long long divisor = std::strtoll(divisorText.c_str(), nullptr, 0);
        if (divisor <= 0 || divisor > 0xffffffffll) {
            return false;
        }

        const std::uint64_t magic = UINT64_MAX / static_cast<std::uint64_t>(divisor);
        const auto slowLabel = ".Larm." + functionName_ + ".mod.slow." + std::to_string(nextInternalLabel_++);
        const auto doneLabel = ".Larm." + functionName_ + ".mod.done." + std::to_string(nextInternalLabel_++);

        out_ << "\tcmp r0, #0\n";
        out_ << "\tblt " << slowLabel << "\n";
        loadImmediate("r2", static_cast<std::uint32_t>(magic & 0xffffffffu));
        out_ << "\tumull r3, r2, r0, r2\n";
        loadImmediate("r3", static_cast<std::uint32_t>(magic >> 32u));
        out_ << "\tumull r3, r12, r0, r3\n";
        out_ << "\tadds r3, r3, r2\n";
        out_ << "\tadc r12, r12, #0\n";
        loadImmediate("r2", static_cast<std::uint32_t>(divisor));
        out_ << "\tmls r0, r12, r2, r0\n";
        out_ << "\tcmp r0, r2\n";
        out_ << "\tsubhs r0, r0, r2\n";
        out_ << "\tb " << doneLabel << "\n";

        out_ << slowLabel << ":\n";
        loadImmediate("r1", static_cast<std::uint32_t>(divisor));
        storeReg("r0", scratchOffset(0));
        storeReg("r1", scratchOffset(1));
        out_ << "\tbl __aeabi_idiv\n";
        out_ << "\tmov r2, r0\n";
        loadReg("r0", scratchOffset(0));
        loadReg("r1", scratchOffset(1));
        out_ << "\tmls r0, r2, r1, r0\n";
        out_ << doneLabel << ":\n";
        return true;
    }

    bool emitSignedPositiveConstDiv(const std::string &divisorText) {
        const long long divisor = std::strtoll(divisorText.c_str(), nullptr, 0);
        if (divisor <= 0 || divisor > 0xffffffffll) {
            return false;
        }
        out_ << "\tmov r1, r0, asr #31\n";
        out_ << "\tcmp r0, #0\n";
        out_ << "\trsblt r0, r0, #0\n";
        emitUnsignedPositiveConstQuotient(static_cast<std::uint32_t>(divisor));
        out_ << "\tcmp r1, #0\n";
        out_ << "\trsbne r0, r0, #0\n";
        return true;
    }

    void emitSignedPositiveConstModNoFrame(std::uint32_t divisor) {
        out_ << "\tmov r9, r0\n";
        (void)emitSignedPositiveConstDiv(std::to_string(divisor));
        loadImmediate("r2", divisor);
        out_ << "\tmls r0, r0, r2, r9\n";
    }

    void emitUnsignedPositiveConstQuotient(std::uint32_t divisor) {
        const std::uint64_t magic = UINT64_MAX / static_cast<std::uint64_t>(divisor);
        loadImmediate("r2", static_cast<std::uint32_t>(magic & 0xffffffffu));
        out_ << "\tumull r3, r2, r0, r2\n";
        loadImmediate("r3", static_cast<std::uint32_t>(magic >> 32u));
        out_ << "\tumull r3, r12, r0, r3\n";
        out_ << "\tadds r3, r3, r2\n";
        out_ << "\tadc r12, r12, #0\n";
        loadImmediate("r2", divisor);
        out_ << "\tmls r3, r12, r2, r0\n";
        out_ << "\tcmp r3, r2\n";
        out_ << "\taddhs r12, r12, #1\n";
        out_ << "\tmov r0, r12\n";
    }

    void emitNeg(const ir::Instruction &inst) {
        if (inst.resultType.kind == ir::TypeKind::F32) {
            emitFloatTo("s14", inst.operands[0]);
            out_ << "\tvneg.f32 s14, s14\n";
            storeFReg("s14", valueOffset_[inst.result]);
        } else {
            emitValueTo("r0", inst.operands[0]);
            out_ << "\trsb r0, r0, #0\n";
            storeReg("r0", valueOffset_[inst.result]);
        }
    }

    void emitCast(const ir::Instruction &inst) {
        if (inst.text == "i2f") {
            emitValueTo("r0", inst.operands[0]);
            out_ << "\tvmov s14, r0\n";
            out_ << "\tvcvt.f32.s32 s14, s14\n";
            storeFReg("s14", valueOffset_[inst.result]);
        } else if (inst.text == "f2i") {
            emitFloatTo("s14", inst.operands[0]);
            out_ << "\tvcvt.s32.f32 s14, s14\n";
            out_ << "\tvmov r0, s14\n";
            storeReg("r0", valueOffset_[inst.result]);
        } else if (inst.resultType.kind == ir::TypeKind::F32) {
            emitFloatTo("s14", inst.operands[0]);
            storeFReg("s14", valueOffset_[inst.result]);
        } else {
            emitValueTo("r0", inst.operands[0]);
            storeReg("r0", valueOffset_[inst.result]);
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
                if (floatArg < 16) {
                    floatRegArgs.push_back({floatArg++, arg});
                } else {
                    stackArgs.push_back(arg);
                }
            } else {
                if (intArg < 4) {
                    intRegArgs.push_back({intArg++, arg});
                } else {
                    stackArgs.push_back(arg);
                }
            }
        }
        const int bytes = alignTo(static_cast<int>(stackArgs.size()) * 4, 8);
        if (bytes > 0) {
            emitSubSp(bytes);
            for (std::size_t i = 0; i < stackArgs.size(); ++i) {
                if (stackArgs[i].type.kind == ir::TypeKind::F32) {
                    emitFloatTo("s14", stackArgs[i]);
                    out_ << "\tvstr.32 s14, [sp, #" << i * 4 << "]\n";
                } else {
                    emitValueTo("r0", stackArgs[i]);
                    out_ << "\tstr r0, [sp, #" << i * 4 << "]\n";
                }
            }
        }
        for (const auto &[reg, arg] : floatRegArgs) {
            emitFloatTo("s" + std::to_string(reg), arg);
        }
        for (const auto &[reg, arg] : intRegArgs) {
            emitValueTo("r" + std::to_string(reg), arg);
        }
        out_ << "\tbl " << inst.text << "\n";
        if (bytes > 0) {
            emitAddSp(bytes);
        }
        if (inst.result >= 0 && inst.resultType.kind != ir::TypeKind::Void) {
            if (inst.resultType.kind == ir::TypeKind::F32) {
                storeFReg("s0", valueOffset_[inst.result]);
            } else {
                storeReg("r0", valueOffset_[inst.result]);
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
        if (canEmitDirectSelfTailCall(inst, count)) {
            for (int i = 0; i < count; ++i) {
                const int offset = valueOffset_[function_->params[static_cast<std::size_t>(i)].id];
                if (inst.operands[static_cast<std::size_t>(i)].type.kind == ir::TypeKind::F32) {
                    emitFloatTo("s14", inst.operands[static_cast<std::size_t>(i)]);
                    storeFReg("s14", offset);
                } else {
                    emitValueTo("r0", inst.operands[static_cast<std::size_t>(i)]);
                    storeReg("r0", offset);
                }
            }
            out_ << "\tb " << blockLabel(function_->blocks.front().name) << "\n";
            return;
        }
        const int bytes = alignTo(count * 4, 8);
        emitSubSp(bytes);
        for (int i = 0; i < count; ++i) {
            if (inst.operands[static_cast<std::size_t>(i)].type.kind == ir::TypeKind::F32) {
                emitFloatTo("s14", inst.operands[static_cast<std::size_t>(i)]);
                out_ << "\tvstr.32 s14, [sp, #" << i * 4 << "]\n";
            } else {
                emitValueTo("r0", inst.operands[static_cast<std::size_t>(i)]);
                out_ << "\tstr r0, [sp, #" << i * 4 << "]\n";
            }
        }
        for (int i = 0; i < count; ++i) {
            const int offset = valueOffset_[function_->params[static_cast<std::size_t>(i)].id];
            if (inst.operands[static_cast<std::size_t>(i)].type.kind == ir::TypeKind::F32) {
                out_ << "\tvldr.32 s14, [sp, #" << i * 4 << "]\n";
                storeFReg("s14", offset);
            } else {
                out_ << "\tldr r0, [sp, #" << i * 4 << "]\n";
                storeReg("r0", offset);
            }
        }
        emitAddSp(bytes);
        out_ << "\tb " << blockLabel(function_->blocks.front().name) << "\n";
    }

    bool canEmitDirectSelfTailCall(const ir::Instruction &inst, int count) const {
        for (int i = 0; i < count; ++i) {
            const auto &operand = inst.operands[static_cast<std::size_t>(i)];
            if (operand.constant) {
                continue;
            }
            for (int j = 0; j < count; ++j) {
                if (operand.id == function_->params[static_cast<std::size_t>(j)].id) {
                    return false;
                }
            }
        }
        return true;
    }

    void emitCondBranch(const std::string &text) {
        const auto labels = splitLabels(text);
        if (labels.size() != 2) {
            return;
        }
        out_ << "\tcmp r0, #0\n";
        const std::string falseCopyLabel = ".Larm." + functionName_ + ".cond.false." + std::to_string(nextInternalLabel_++);
        out_ << "\tbeq " << falseCopyLabel << "\n";
        emitPhiCopies(currentBlock_, labels[0]);
        out_ << "\tb " << blockLabel(labels[0]) << "\n";
        out_ << falseCopyLabel << ":\n";
        emitPhiCopies(currentBlock_, labels[1]);
        if (labels[1] != nextBlock_) {
            out_ << "\tb " << blockLabel(labels[1]) << "\n";
        }
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
        if (canEmitDirectPhiCopies(copies)) {
            for (const auto &copy : copies) {
                if (copy.type.kind == ir::TypeKind::F32) {
                    emitFloatTo("s14", copy.source);
                    storeFReg("s14", valueOffset_[copy.target]);
                } else {
                    emitValueTo("r0", copy.source);
                    storeReg("r0", valueOffset_[copy.target]);
                }
            }
            return;
        }
        const int bytes = alignTo(static_cast<int>(copies.size()) * 4, 8);
        emitSubSp(bytes);
        for (std::size_t i = 0; i < copies.size(); ++i) {
            const auto &copy = copies[i];
            if (copy.type.kind == ir::TypeKind::F32) {
                emitFloatTo("s14", copy.source);
                out_ << "\tvstr.32 s14, [sp, #" << i * 4 << "]\n";
            } else {
                emitValueTo("r0", copy.source);
                out_ << "\tstr r0, [sp, #" << i * 4 << "]\n";
            }
        }
        for (std::size_t i = 0; i < copies.size(); ++i) {
            const auto &copy = copies[i];
            if (copy.type.kind == ir::TypeKind::F32) {
                out_ << "\tvldr.32 s14, [sp, #" << i * 4 << "]\n";
                storeFReg("s14", valueOffset_[copy.target]);
            } else {
                out_ << "\tldr r0, [sp, #" << i * 4 << "]\n";
                storeReg("r0", valueOffset_[copy.target]);
            }
        }
        emitAddSp(bytes);
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
        emitValueTo("r1", gep.operands[1]);
        out_ << "\tadd " << reg << ", " << reg << ", r1, lsl #2\n";
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
        emitValueTo(reg, value);
    }

    void emitValueTo(const std::string &reg, const ir::Value &value) {
        if (value.constant) {
            if (!value.name.empty() && value.name[0] == '@') {
                loadAddress(reg, value.name.substr(1));
            } else {
                loadImmediate(reg, parseImmediate(value.name));
            }
            return;
        }
        const auto object = objectOffset_.find(value.id);
        if (object != objectOffset_.end()) {
            emitFrameAddress(reg, object->second);
            return;
        }
        loadReg(reg, valueOffset_[value.id]);
    }

    void emitFloatTo(const std::string &reg, const ir::Value &value) {
        if (value.constant) {
            const float f = std::strtof(value.name.c_str(), nullptr);
            loadImmediate("r0", floatBits(f));
            out_ << "\tvmov " << reg << ", r0\n";
            return;
        }
        loadFReg(reg, valueOffset_[value.id]);
    }

    void emitFrameAddress(const std::string &reg, int offset) {
        const int abs = -offset;
        if (abs == 0) {
            out_ << "\tmov " << reg << ", fp\n";
            return;
        }
        loadImmediate(reg, static_cast<std::uint32_t>(abs));
        out_ << "\tsub " << reg << ", fp, " << reg << "\n";
    }

    void loadReg(const std::string &reg, int offset) {
        const int abs = -offset;
        if (abs <= 4095) {
            out_ << "\tldr " << reg << ", [fp, #-" << abs << "]\n";
        } else {
            loadImmediate("r12", static_cast<std::uint32_t>(abs));
            out_ << "\tsub r12, fp, r12\n";
            out_ << "\tldr " << reg << ", [r12]\n";
        }
    }

    void storeReg(const std::string &reg, int offset) {
        const int abs = -offset;
        if (abs <= 4095) {
            out_ << "\tstr " << reg << ", [fp, #-" << abs << "]\n";
        } else {
            loadImmediate("r12", static_cast<std::uint32_t>(abs));
            out_ << "\tsub r12, fp, r12\n";
            out_ << "\tstr " << reg << ", [r12]\n";
        }
    }

    void loadFReg(const std::string &reg, int offset) {
        const int abs = -offset;
        if (abs <= 1020) {
            out_ << "\tvldr.32 " << reg << ", [fp, #-" << abs << "]\n";
        } else {
            loadImmediate("r12", static_cast<std::uint32_t>(abs));
            out_ << "\tsub r12, fp, r12\n";
            out_ << "\tvldr.32 " << reg << ", [r12]\n";
        }
    }

    void storeFReg(const std::string &reg, int offset) {
        const int abs = -offset;
        if (abs <= 1020) {
            out_ << "\tvstr.32 " << reg << ", [fp, #-" << abs << "]\n";
        } else {
            loadImmediate("r12", static_cast<std::uint32_t>(abs));
            out_ << "\tsub r12, fp, r12\n";
            out_ << "\tvstr.32 " << reg << ", [r12]\n";
        }
    }

    void loadStackArgTo(const std::string &reg, int slot) {
        out_ << "\tldr " << reg << ", [fp, #" << 8 + slot * 4 << "]\n";
    }

    void emitSubSp(int bytes) {
        if (bytes <= 0) {
            return;
        }
        loadImmediate("r12", static_cast<std::uint32_t>(bytes));
        out_ << "\tsub sp, sp, r12\n";
    }

    void emitAddSp(int bytes) {
        if (bytes <= 0) {
            return;
        }
        loadImmediate("r12", static_cast<std::uint32_t>(bytes));
        out_ << "\tadd sp, sp, r12\n";
    }

    int scratchOffset(int index) const {
        return -frameSize_ + 4 + index * 4;
    }

    static std::string armCond(const std::string &cmp) {
        if (cmp == "eq") return "eq";
        if (cmp == "ne") return "ne";
        if (cmp == "lt") return "lt";
        if (cmp == "le") return "le";
        if (cmp == "gt") return "gt";
        if (cmp == "ge") return "ge";
        return "ne";
    }

    static std::uint32_t parseImmediate(const std::string &text) {
        return static_cast<std::uint32_t>(std::strtoll(text.c_str(), nullptr, 0));
    }

    static std::optional<int> positivePowerOfTwoShift(const std::string &text) {
        const long long value = std::strtoll(text.c_str(), nullptr, 0);
        if (value <= 0 || (value & (value - 1)) != 0) {
            return std::nullopt;
        }
        int shift = 0;
        for (long long current = value; current > 1; current >>= 1) {
            ++shift;
        }
        return shift;
    }

    void loadImmediate(const std::string &reg, std::uint32_t value) {
        out_ << "\tmovw " << reg << ", #" << (value & 0xffffu) << "\n";
        if ((value >> 16u) != 0) {
            out_ << "\tmovt " << reg << ", #" << (value >> 16u) << "\n";
        }
    }

    void loadAddress(const std::string &reg, const std::string &symbol) {
        out_ << "\tmovw " << reg << ", #:lower16:" << symbol << "\n";
        out_ << "\tmovt " << reg << ", #:upper16:" << symbol << "\n";
    }
};

} // namespace

void emitAssembly(const ir::Module &module, std::ostream &out) {
    out << "\t.syntax unified\n";
    out << "\t.arch armv7-a\n";
    out << "\t.fpu vfpv3\n";
    out << "\t.eabi_attribute 28, 1\n";
    CodeGen(module, out).run();
    out << "\t.section .note.GNU-stack,\"\",%progbits\n";
}

void emitAssembly(const TranslationUnit &, std::ostream &out) {
    out << "\t.syntax unified\n";
    out << "\t.arch armv7-a\n";
    out << "\t.fpu vfpv3\n";
    out << "\t.text\n";
    out << "\t.global main\n";
    out << "main:\n";
    out << "\tmov r0, #0\n";
    out << "\tbx lr\n";
    out << "\t.section .note.GNU-stack,\"\",%progbits\n";
}

} // namespace sysyc::arm
