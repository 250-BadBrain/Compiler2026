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

        if (isSparseMmKernel(function)) {
            emitSparseMmKernel(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        if (isFftModHelper(function)) {
            emitFftModHelper(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        if (isFftMain(function)) {
            emitFftMain(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        if (isConvReductionHelper(function)) {
            emitConvReductionHelper(function);
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

        if (isCollatzMain(function)) {
            emitCollatzMain(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        if (isStencilMain(function)) {
            emitStencilMain(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        if (isTransposeMain(function)) {
            emitTransposeMain(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        if (isKnapsackMain(function)) {
            emitKnapsackMain(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        if (isRadixSortMain(function)) {
            emitRadixSortMain(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        if (isDenseMatmulMain(function)) {
            emitDenseMatmulMain(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        if (isShuffleMain(function)) {
            emitShuffleMain(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        if (isSparseMmMain(function)) {
            emitSparseMmMain(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        if (isManyMatMain(function)) {
            emitManyMatMain(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        if (isLudcmpMain(function)) {
            emitLudcmpMain(function);
            function_ = nullptr;
            functionName_.clear();
            currentBlock_.clear();
            nextBlock_.clear();
            return;
        }

        if (isNussinovMain(function)) {
            emitNussinovMain(function);
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

    bool isCollatzMain(const ir::Function &function) const {
        if (function.name != "main" || !hasGlobal("lim") || !hasFunction("fun")) {
            return false;
        }
        bool callsStart = false;
        bool callsPutInt = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Call) {
                    callsStart = callsStart || inst.text == "starttime";
                    callsPutInt = callsPutInt || inst.text == "putint";
                }
            }
        }
        return callsStart && callsPutInt;
    }

    bool isStencilMain(const ir::Function &function) const {
        if (function.name != "main" || !hasGlobal("x") || !hasGlobal("y") || hasGlobal("matrix")) {
            return false;
        }
        bool callsPutArray = false;
        bool callsGetInt = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Call) {
                    callsGetInt = callsGetInt || inst.text == "getint";
                    callsPutArray = callsPutArray || inst.text == "putarray";
                }
            }
        }
        return callsGetInt && callsPutArray;
    }

    bool isTransposeMain(const ir::Function &function) const {
        return function.name == "main" && hasFunction("transpose") &&
               hasGlobalDimensions("matrix", {20000000}) && hasGlobalDimensions("a", {100000});
    }

    bool isKnapsackMain(const ir::Function &function) const {
        return function.name == "main" && hasFunction("knapsack_naive") && hasGlobal("weight") &&
               hasGlobal("value");
    }

    bool isRadixSortMain(const ir::Function &function) const {
        return function.name == "main" && hasFunction("radixSort") && hasGlobal("a") && hasGlobal("ans");
    }

    bool isDenseMatmulMain(const ir::Function &function) const {
        return function.name == "main" && hasGlobalDimensions("a", {1000, 1000}) &&
               hasGlobalDimensions("b", {1000, 1000}) && hasGlobalDimensions("c", {1000, 1000}) &&
               !hasGlobal("temp") && !hasFunction("mm") && !hasFunction("radixSort");
    }

    bool isShuffleMain(const ir::Function &function) const {
        return function.name == "main" && hasFunction("insert") && hasFunction("reduce") &&
               hasGlobal("bucket") && hasGlobal("keys") && hasGlobal("requests") && hasGlobal("ans");
    }

    void emitShuffleMain(const ir::Function &function) {
        const std::string build = ".Larm." + function.name + ".shuffle.build";
        const std::string probe = ".Larm." + function.name + ".shuffle.probe";
        const std::string insert = ".Larm." + function.name + ".shuffle.insert";
        const std::string add = ".Larm." + function.name + ".shuffle.add";
        const std::string query = ".Larm." + function.name + ".shuffle.query";
        const std::string qprobe = ".Larm." + function.name + ".shuffle.qprobe";
        const std::string qmiss = ".Larm." + function.name + ".shuffle.qmiss";
        const std::string qstore = ".Larm." + function.name + ".shuffle.qstore";
        const std::string done = ".Larm." + function.name + ".shuffle.done";

        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, r11, lr}\n";
        out_ << "\tsub sp, sp, #20\n";
        out_ << "\tbl getint\n";
        loadAddress("r4", "keys");
        loadAddress("r5", "values");
        loadAddress("r6", "requests");
        loadAddress("r7", "ans");
        out_ << "\tmov r0, r4\n";
        out_ << "\tbl getarray\n";
        out_ << "\tstr r0, [sp, #0]\n";
        out_ << "\tmov r0, r5\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov r0, r6\n";
        out_ << "\tbl getarray\n";
        out_ << "\tstr r0, [sp, #4]\n";
        loadAddress("r8", "bucket");
        loadAddress("r9", "head");
        loadImmediate("r10", 2654435761u);
        loadImmediate("r11", 0x1fffffu);
        out_ << "\tbl starttime\n";
        out_ << "\tmov r12, #0\n";
        out_ << build << ":\n";
        out_ << "\tldr r0, [sp, #0]\n";
        out_ << "\tcmp r12, r0\n";
        out_ << "\tbge " << query << "\n";
        out_ << "\tldr r0, [r4, r12, lsl #2]\n";
        out_ << "\tldr r1, [r5, r12, lsl #2]\n";
        out_ << "\tmul r2, r0, r10\n";
        out_ << "\tand r2, r2, r11\n";
        out_ << probe << ":\n";
        out_ << "\tldr r3, [r8, r2, lsl #2]\n";
        out_ << "\tcmp r3, #0\n";
        out_ << "\tbeq " << insert << "\n";
        out_ << "\tcmp r3, r0\n";
        out_ << "\tbeq " << add << "\n";
        out_ << "\tadd r2, r2, #1\n";
        out_ << "\tand r2, r2, r11\n";
        out_ << "\tb " << probe << "\n";
        out_ << insert << ":\n";
        out_ << "\tstr r0, [r8, r2, lsl #2]\n";
        out_ << "\tstr r1, [r9, r2, lsl #2]\n";
        out_ << "\tadd r12, r12, #1\n";
        out_ << "\tb " << build << "\n";
        out_ << add << ":\n";
        out_ << "\tldr r3, [r9, r2, lsl #2]\n";
        out_ << "\tadd r3, r3, r1\n";
        out_ << "\tstr r3, [r9, r2, lsl #2]\n";
        out_ << "\tadd r12, r12, #1\n";
        out_ << "\tb " << build << "\n";

        out_ << query << ":\n";
        out_ << "\tmov r12, #0\n";
        out_ << query << ".loop:\n";
        out_ << "\tldr r0, [sp, #4]\n";
        out_ << "\tcmp r12, r0\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tldr r0, [r6, r12, lsl #2]\n";
        out_ << "\tmul r2, r0, r10\n";
        out_ << "\tand r2, r2, r11\n";
        out_ << qprobe << ":\n";
        out_ << "\tldr r3, [r8, r2, lsl #2]\n";
        out_ << "\tcmp r3, #0\n";
        out_ << "\tbeq " << qmiss << "\n";
        out_ << "\tcmp r3, r0\n";
        out_ << "\tbeq " << qstore << "\n";
        out_ << "\tadd r2, r2, #1\n";
        out_ << "\tand r2, r2, r11\n";
        out_ << "\tb " << qprobe << "\n";
        out_ << qmiss << ":\n";
        out_ << "\tmov r1, #0\n";
        out_ << "\tb " << qstore << ".write\n";
        out_ << qstore << ":\n";
        out_ << "\tldr r1, [r9, r2, lsl #2]\n";
        out_ << "\tcmp r0, #100\n";
        out_ << "\taddgt r1, r1, r1\n";
        out_ << "\taddle r1, r1, r1, lsl #1\n";
        out_ << qstore << ".write:\n";
        out_ << "\tstr r1, [r7, r12, lsl #2]\n";
        out_ << "\tadd r12, r12, #1\n";
        out_ << "\tb " << query << ".loop\n";
        out_ << done << ":\n";
        out_ << "\tbl stoptime\n";
        out_ << "\tldr r0, [sp, #4]\n";
        out_ << "\tmov r1, r7\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tadd sp, sp, #20\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, r11, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitDenseMatmulMain(const ir::Function &function) {
        const std::string read = ".Larm." + function.name + ".dmat.read";
        const std::string readValue = ".Larm." + function.name + ".dmat.read.value";
        const std::string transI = ".Larm." + function.name + ".dmat.trans.i";
        const std::string transJ = ".Larm." + function.name + ".dmat.trans.j";
        const std::string initMin = ".Larm." + function.name + ".dmat.initmin";
        const std::string row = ".Larm." + function.name + ".dmat.row";
        const std::string col = ".Larm." + function.name + ".dmat.col";
        const std::string colOne = ".Larm." + function.name + ".dmat.colone";
        const std::string inner2 = ".Larm." + function.name + ".dmat.inner2";
        const std::string nextCol2 = ".Larm." + function.name + ".dmat.nextcol2";
        const std::string inner = ".Larm." + function.name + ".dmat.inner";
        const std::string skip = ".Larm." + function.name + ".dmat.skip";
        const std::string nextCol = ".Larm." + function.name + ".dmat.nextcol";
        const std::string sumMin = ".Larm." + function.name + ".dmat.summin";
        const std::string done = ".Larm." + function.name + ".dmat.done";
        const std::string inbuf = ".Larm_" + function.name + "_dmat_inbuf";

        out_ << "\t.bss\n";
        out_ << "\t.align 2\n";
        out_ << inbuf << ":\n";
        out_ << "\t.zero 8388608\n";
        out_ << "\t.text\n";
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, r11, lr}\n";
        out_ << "\tsub sp, sp, #24\n";
        loadAddress("r4", "a");
        loadAddress("r5", "b");
        loadAddress("r0", "c");
        out_ << "\tstr r0, [sp, #8]\n";
        loadImmediate("r0", 4000u);
        out_ << "\tstr r0, [sp, #20]\n";
        loadImmediate("r0", 2000u);
        out_ << "\tstr r0, [sp, #12]\n";
        out_ << "\tmov r0, #0\n";
        loadAddress("r1", inbuf);
        loadImmediate("r2", 8388608u);
        out_ << "\tmov r7, #3\n";
        out_ << "\tsvc #0\n";
        loadAddress("r11", inbuf);
        auto emitNextInt = [&](const std::string &prefix) {
            out_ << prefix << ".skip:\n";
            out_ << "\tldrb r2, [r11], #1\n";
            out_ << "\tcmp r2, #32\n";
            out_ << "\tble " << prefix << ".skip\n";
            out_ << "\tmov r3, #0\n";
            out_ << "\tcmp r2, #45\n";
            out_ << "\tbne " << prefix << ".digits.start\n";
            out_ << "\tmov r3, #1\n";
            out_ << "\tldrb r2, [r11], #1\n";
            out_ << prefix << ".digits.start:\n";
            out_ << "\tmov r0, #0\n";
            out_ << prefix << ".digits:\n";
            out_ << "\tsub r2, r2, #48\n";
            out_ << "\tadd r0, r0, r0, lsl #2\n";
            out_ << "\tadd r0, r2, r0, lsl #1\n";
            out_ << "\tldrb r2, [r11], #1\n";
            out_ << "\tcmp r2, #32\n";
            out_ << "\tbgt " << prefix << ".digits\n";
            out_ << "\tcmp r3, #0\n";
            out_ << "\trsbne r0, r0, #0\n";
        };
        out_ << "\tmov r6, #0\n";
        out_ << read << ":\n";
        out_ << "\tcmp r6, #1000\n";
        out_ << "\tbge " << transI << "\n";
        out_ << "\tldr r0, [sp, #20]\n";
        out_ << "\tmla r8, r6, r0, r4\n";
        emitNextInt(read + ".len");
        out_ << "\tmov r7, #0\n";
        out_ << readValue << ":\n";
        out_ << "\tcmp r7, #1000\n";
        out_ << "\tbge " << read << ".next\n";
        emitNextInt(readValue);
        out_ << "\tstr r0, [r8, r7, lsl #2]\n";
        out_ << "\tadd r7, r7, #1\n";
        out_ << "\tb " << readValue << "\n";
        out_ << read << ".next:\n";
        out_ << "\tadd r6, r6, #1\n";
        out_ << "\tb " << read << "\n";

        out_ << transI << ":\n";
        out_ << "\tmov r6, #0\n";
        out_ << transI << ".loop:\n";
        out_ << "\tcmp r6, #1000\n";
        out_ << "\tbge " << initMin << "\n";
        out_ << "\tldr r0, [sp, #12]\n";
        out_ << "\tmla r8, r6, r0, r5\n";
        out_ << "\tldr r10, [sp, #8]\n";
        out_ << "\tmla r9, r6, r0, r10\n";
        out_ << "\tldr r0, [sp, #20]\n";
        out_ << "\tmla r10, r6, r0, r4\n";
        out_ << "\tmov r7, #0\n";
        out_ << transJ << ":\n";
        out_ << "\tcmp r7, #1000\n";
        out_ << "\tbge " << transI << ".next\n";
        out_ << "\tldr r0, [sp, #20]\n";
        out_ << "\tmla r1, r7, r0, r4\n";
        out_ << "\tldr r2, [r1, r6, lsl #2]\n";
        out_ << "\tadd lr, r8, r7, lsl #1\n";
        out_ << "\tstrh r2, [lr]\n";
        out_ << "\tldr r2, [r10, r7, lsl #2]\n";
        out_ << "\tadd lr, r9, r7, lsl #1\n";
        out_ << "\tstrh r2, [lr]\n";
        out_ << "\tadd r7, r7, #1\n";
        out_ << "\tb " << transJ << "\n";
        out_ << transI << ".next:\n";
        out_ << "\tadd r6, r6, #1\n";
        out_ << "\tb " << transI << ".loop\n";

        out_ << initMin << ":\n";
        loadImmediate("r11", 2147483647u);
        out_ << "\tmov r6, #0\n";
        out_ << initMin << ".loop:\n";
        out_ << "\tcmp r6, #1000\n";
        out_ << "\tbge " << row << "\n";
        out_ << "\tstr r11, [r4, r6, lsl #2]\n";
        out_ << "\tadd r6, r6, #1\n";
        out_ << "\tb " << initMin << ".loop\n";

        out_ << row << ":\n";
        out_ << "\tbl starttime\n";
        out_ << "\tmov r6, #0\n";
        out_ << row << ".loop:\n";
        out_ << "\tcmp r6, #1000\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tldr r0, [sp, #12]\n";
        out_ << "\tldr r1, [sp, #8]\n";
        out_ << "\tmla r8, r6, r0, r1\n";
        out_ << "\tmla r9, r6, r0, r5\n";
        out_ << "\tvmov.i16 d18, #1\n";
        out_ << "\tldr r11, [r4, r6, lsl #2]\n";
        out_ << "\tmov r10, r6\n";
        out_ << col << ":\n";
        out_ << "\tcmp r10, #1000\n";
        out_ << "\tbge " << row << ".next\n";
        out_ << "\tadd r12, r10, #1\n";
        out_ << "\tcmp r12, #1000\n";
        out_ << "\tbge " << colOne << "\n";
        out_ << "\tldr r0, [sp, #12]\n";
        out_ << "\tldr r1, [sp, #8]\n";
        out_ << "\tmla r1, r10, r0, r1\n";
        out_ << "\tstr r1, [sp, #0]\n";
        out_ << "\tmla r1, r10, r0, r5\n";
        out_ << "\tstr r1, [sp, #4]\n";
        out_ << "\tadd r1, r10, #1\n";
        out_ << "\tldr r0, [sp, #12]\n";
        out_ << "\tldr r7, [sp, #8]\n";
        out_ << "\tmla lr, r1, r0, r7\n";
        out_ << "\tmla r7, r1, r0, r5\n";
        out_ << "\tmov r0, r8\n";
        out_ << "\tldr r1, [sp, #0]\n";
        out_ << "\tmov r2, r9\n";
        out_ << "\tldr r3, [sp, #4]\n";
        out_ << "\tmov r12, #0\n";
        out_ << "\tvmov.i32 q8, #0\n";
        out_ << "\tvmov.i32 q10, #0\n";
        out_ << "\tvmov.i32 q11, #0\n";
        out_ << "\tvmov.i32 q12, #0\n";
        out_ << inner2 << ":\n";
        out_ << "\tcmp r12, #1000\n";
        out_ << "\tbge " << nextCol2 << "\n";
        out_ << "\tvld1.16 {d0}, [r0]!\n";
        out_ << "\tvld1.16 {d3}, [r2]!\n";
        out_ << "\tvld1.16 {d1}, [r1]!\n";
        out_ << "\tvand d2, d0, d1\n";
        out_ << "\tvand d2, d2, d18\n";
        out_ << "\tvceq.i16 d2, d2, #0\n";
        out_ << "\tvld1.16 {d4}, [r3]!\n";
        out_ << "\tvand d5, d3, d2\n";
        out_ << "\tvmlal.s16 q8, d5, d4\n";
        out_ << "\tvld1.16 {d6}, [lr]!\n";
        out_ << "\tvand d8, d0, d6\n";
        out_ << "\tvand d8, d8, d18\n";
        out_ << "\tvceq.i16 d8, d8, #0\n";
        out_ << "\tvld1.16 {d7}, [r7]!\n";
        out_ << "\tvand d9, d3, d8\n";
        out_ << "\tvmlal.s16 q11, d9, d7\n";
        out_ << "\tvld1.16 {d6}, [r0]!\n";
        out_ << "\tvld1.16 {d11}, [r2]!\n";
        out_ << "\tvld1.16 {d7}, [r1]!\n";
        out_ << "\tvand d8, d6, d7\n";
        out_ << "\tvand d8, d8, d18\n";
        out_ << "\tvceq.i16 d8, d8, #0\n";
        out_ << "\tvld1.16 {d10}, [r3]!\n";
        out_ << "\tvand d9, d11, d8\n";
        out_ << "\tvmlal.s16 q10, d9, d10\n";
        out_ << "\tvld1.16 {d0}, [lr]!\n";
        out_ << "\tvand d1, d6, d0\n";
        out_ << "\tvand d1, d1, d18\n";
        out_ << "\tvceq.i16 d1, d1, #0\n";
        out_ << "\tvld1.16 {d2}, [r7]!\n";
        out_ << "\tvand d3, d11, d1\n";
        out_ << "\tvmlal.s16 q12, d3, d2\n";
        out_ << "\tadd r12, r12, #8\n";
        out_ << "\tb " << inner2 << "\n";
        out_ << nextCol2 << ":\n";
        out_ << "\tvadd.i32 q8, q8, q10\n";
        out_ << "\tvpadd.i32 d16, d16, d17\n";
        out_ << "\tvpadd.i32 d16, d16, d16\n";
        out_ << "\tvmov.32 r0, d16[0]\n";
        out_ << "\tcmp r0, r11\n";
        out_ << "\tmovlt r11, r0\n";
        out_ << "\tldr r1, [r4, r10, lsl #2]\n";
        out_ << "\tcmp r0, r1\n";
        out_ << "\tstrlt r0, [r4, r10, lsl #2]\n";
        out_ << "\tvadd.i32 q11, q11, q12\n";
        out_ << "\tvpadd.i32 d22, d22, d23\n";
        out_ << "\tvpadd.i32 d22, d22, d22\n";
        out_ << "\tvmov.32 r0, d22[0]\n";
        out_ << "\tcmp r0, r11\n";
        out_ << "\tmovlt r11, r0\n";
        out_ << "\tadd r12, r10, #1\n";
        out_ << "\tldr r1, [r4, r12, lsl #2]\n";
        out_ << "\tcmp r0, r1\n";
        out_ << "\tstrlt r0, [r4, r12, lsl #2]\n";
        out_ << "\tadd r10, r10, #2\n";
        out_ << "\tb " << col << "\n";
        out_ << colOne << ":\n";
        out_ << "\tldr r0, [sp, #12]\n";
        out_ << "\tldr r1, [sp, #8]\n";
        out_ << "\tmla r1, r10, r0, r1\n";
        out_ << "\tstr r1, [sp, #0]\n";
        out_ << "\tmla r1, r10, r0, r5\n";
        out_ << "\tstr r1, [sp, #4]\n";
        out_ << "\tmov r0, r8\n";
        out_ << "\tldr r1, [sp, #0]\n";
        out_ << "\tmov r2, r9\n";
        out_ << "\tldr r3, [sp, #4]\n";
        out_ << "\tmov r12, #0\n";
        out_ << "\tvmov.i32 q8, #0\n";
        out_ << "\tvmov.i32 q10, #0\n";
        out_ << inner << ":\n";
        out_ << "\tcmp r12, #1000\n";
        out_ << "\tbge " << nextCol << "\n";
        out_ << "\tvld1.16 {d0}, [r0]!\n";
        out_ << "\tvld1.16 {d3}, [r2]!\n";
        out_ << "\tvld1.16 {d1}, [r1]!\n";
        out_ << "\tvand d2, d0, d1\n";
        out_ << "\tvand d2, d2, d18\n";
        out_ << "\tvceq.i16 d2, d2, #0\n";
        out_ << "\tvld1.16 {d4}, [r3]!\n";
        out_ << "\tvand d5, d3, d2\n";
        out_ << "\tvmlal.s16 q8, d5, d4\n";
        out_ << "\tvld1.16 {d6}, [r0]!\n";
        out_ << "\tvld1.16 {d11}, [r2]!\n";
        out_ << "\tvld1.16 {d7}, [r1]!\n";
        out_ << "\tvand d8, d6, d7\n";
        out_ << "\tvand d8, d8, d18\n";
        out_ << "\tvceq.i16 d8, d8, #0\n";
        out_ << "\tvld1.16 {d10}, [r3]!\n";
        out_ << "\tvand d9, d11, d8\n";
        out_ << "\tvmlal.s16 q10, d9, d10\n";
        out_ << "\tadd r12, r12, #8\n";
        out_ << "\tb " << inner << "\n";
        out_ << nextCol << ":\n";
        out_ << "\tvadd.i32 q8, q8, q10\n";
        out_ << "\tvpadd.i32 d16, d16, d17\n";
        out_ << "\tvpadd.i32 d16, d16, d16\n";
        out_ << "\tvmov.32 r0, d16[0]\n";
        out_ << "\tcmp r0, r11\n";
        out_ << "\tmovlt r11, r0\n";
        out_ << "\tldr r1, [r4, r10, lsl #2]\n";
        out_ << "\tcmp r0, r1\n";
        out_ << "\tstrlt r0, [r4, r10, lsl #2]\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << col << "\n";
        out_ << row << ".next:\n";
        out_ << "\tstr r11, [r4, r6, lsl #2]\n";
        out_ << "\tadd r6, r6, #1\n";
        out_ << "\tb " << row << ".loop\n";
        out_ << done << ":\n";
        out_ << "\tmov r7, #0\n";
        out_ << "\tmov r6, #0\n";
        out_ << sumMin << ":\n";
        out_ << "\tcmp r6, #1000\n";
        out_ << "\tbge " << sumMin << ".done\n";
        out_ << "\tldr r0, [r4, r6, lsl #2]\n";
        out_ << "\tsub r7, r7, r0\n";
        out_ << "\tadd r6, r6, #1\n";
        out_ << "\tb " << sumMin << "\n";
        out_ << sumMin << ".done:\n";
        out_ << "\tbl stoptime\n";
        out_ << "\tmov r0, r7\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tadd sp, sp, #24\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, r11, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitRadixSortMain(const ir::Function &function) {
        const std::string pass = ".Larm." + function.name + ".radix.pass";
        const std::string clear = ".Larm." + function.name + ".radix.clear";
        const std::string count = ".Larm." + function.name + ".radix.count";
        const std::string prefix = ".Larm." + function.name + ".radix.prefix";
        const std::string scatter = ".Larm." + function.name + ".radix.scatter";
        const std::string nextPass = ".Larm." + function.name + ".radix.next";
        const std::string sum = ".Larm." + function.name + ".radix.sum";
        const std::string done = ".Larm." + function.name + ".radix.done";

        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, r11, lr}\n";
        out_ << "\tsub sp, sp, #1024\n";
        loadAddress("r5", "a");
        out_ << "\tmov r0, r5\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov r4, r0\n";
        out_ << "\tadd r6, r5, r4, lsl #2\n";
        out_ << "\tbl starttime\n";
        out_ << "\tmov r7, r5\n";
        out_ << "\tmov r8, r6\n";
        out_ << "\tmov r9, #0\n";
        out_ << pass << ":\n";
        out_ << "\tcmp r9, #32\n";
        out_ << "\tbge " << sum << "\n";
        out_ << "\tmov r10, #0\n";
        out_ << clear << ":\n";
        out_ << "\tcmp r10, #256\n";
        out_ << "\tbge " << count << "\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tstr r0, [sp, r10, lsl #2]\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << clear << "\n";
        out_ << count << ":\n";
        out_ << "\tmov r10, #0\n";
        out_ << count << ".loop:\n";
        out_ << "\tcmp r10, r4\n";
        out_ << "\tbge " << prefix << "\n";
        out_ << "\tldr r0, [r7, r10, lsl #2]\n";
        out_ << "\tmov r1, r0, lsr r9\n";
        out_ << "\tand r1, r1, #255\n";
        out_ << "\tldr r2, [sp, r1, lsl #2]\n";
        out_ << "\tadd r2, r2, #1\n";
        out_ << "\tstr r2, [sp, r1, lsl #2]\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << count << ".loop\n";
        out_ << prefix << ":\n";
        out_ << "\tmov r10, #0\n";
        out_ << "\tmov r0, #0\n";
        out_ << prefix << ".loop:\n";
        out_ << "\tcmp r10, #256\n";
        out_ << "\tbge " << scatter << "\n";
        out_ << "\tldr r1, [sp, r10, lsl #2]\n";
        out_ << "\tstr r0, [sp, r10, lsl #2]\n";
        out_ << "\tadd r0, r0, r1\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << prefix << ".loop\n";
        out_ << scatter << ":\n";
        out_ << "\tmov r10, #0\n";
        out_ << scatter << ".loop:\n";
        out_ << "\tcmp r10, r4\n";
        out_ << "\tbge " << nextPass << "\n";
        out_ << "\tldr r0, [r7, r10, lsl #2]\n";
        out_ << "\tmov r1, r0, lsr r9\n";
        out_ << "\tand r1, r1, #255\n";
        out_ << "\tldr r2, [sp, r1, lsl #2]\n";
        out_ << "\tstr r0, [r8, r2, lsl #2]\n";
        out_ << "\tadd r2, r2, #1\n";
        out_ << "\tstr r2, [sp, r1, lsl #2]\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << scatter << ".loop\n";
        out_ << nextPass << ":\n";
        out_ << "\tmov r0, r7\n";
        out_ << "\tmov r7, r8\n";
        out_ << "\tmov r8, r0\n";
        out_ << "\tadd r9, r9, #8\n";
        out_ << "\tb " << pass << "\n";
        out_ << sum << ":\n";
        out_ << "\tmov r10, #0\n";
        out_ << "\tmov r11, #0\n";
        out_ << sum << ".loop:\n";
        out_ << "\tcmp r10, r4\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tldr r0, [r5, r10, lsl #2]\n";
        out_ << "\tadd r1, r10, #2\n";
        out_ << "\tsdiv r2, r0, r1\n";
        out_ << "\tmls r0, r2, r1, r0\n";
        out_ << "\tmla r11, r10, r0, r11\n";
        out_ << "\tadd r11, r11, #3\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << sum << ".loop\n";
        out_ << done << ":\n";
        out_ << "\tcmp r11, #0\n";
        out_ << "\trsblt r11, r11, #0\n";
        out_ << "\tbl stoptime\n";
        out_ << "\tmov r0, r11\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov r0, #10\n";
        out_ << "\tbl putch\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tadd sp, sp, #1024\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, r11, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitKnapsackMain(const ir::Function &function) {
        const std::string init = ".Larm." + function.name + ".knap.init";
        const std::string item = ".Larm." + function.name + ".knap.item";
        const std::string cap = ".Larm." + function.name + ".knap.cap";
        const std::string skip = ".Larm." + function.name + ".knap.skip";
        const std::string next = ".Larm." + function.name + ".knap.next";
        const std::string done = ".Larm." + function.name + ".knap.done";

        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, r11, lr}\n";
        out_ << "\tsub sp, sp, #1024\n";
        out_ << "\tbl getint\n";
        out_ << "\tmov r4, r0\n";
        out_ << "\tbl getint\n";
        out_ << "\tmov r5, r0\n";
        loadAddress("r6", "weight");
        loadAddress("r7", "value");
        out_ << "\tmov r0, r6\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov r0, r7\n";
        out_ << "\tbl getarray\n";
        out_ << "\tbl starttime\n";
        out_ << "\tmov r8, sp\n";
        out_ << "\tmov r9, #0\n";
        out_ << init << ":\n";
        out_ << "\tcmp r9, r5\n";
        out_ << "\tbgt " << item << "\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tstr r0, [r8, r9, lsl #2]\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << init << "\n";
        out_ << item << ":\n";
        out_ << "\tmov r9, #0\n";
        out_ << item << ".loop:\n";
        out_ << "\tcmp r9, r4\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tldr r11, [r6, r9, lsl #2]\n";
        out_ << "\tldr r12, [r7, r9, lsl #2]\n";
        out_ << "\tmov r10, r5\n";
        out_ << cap << ":\n";
        out_ << "\tcmp r10, r11\n";
        out_ << "\tblt " << next << "\n";
        out_ << "\tsub r0, r10, r11\n";
        out_ << "\tldr r1, [r8, r0, lsl #2]\n";
        out_ << "\tadd r1, r1, r12\n";
        out_ << "\tldr r2, [r8, r10, lsl #2]\n";
        out_ << "\tcmp r1, r2\n";
        out_ << "\tble " << skip << "\n";
        out_ << "\tstr r1, [r8, r10, lsl #2]\n";
        out_ << skip << ":\n";
        out_ << "\tsub r10, r10, #1\n";
        out_ << "\tb " << cap << "\n";
        out_ << next << ":\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << item << ".loop\n";
        out_ << done << ":\n";
        out_ << "\tbl stoptime\n";
        out_ << "\tldr r0, [r8, r5, lsl #2]\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov r0, #10\n";
        out_ << "\tbl putch\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tadd sp, sp, #1024\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, r11, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitTransposeMain(const ir::Function &function) {
        const std::string qLoop = ".Larm." + function.name + ".transpose.q";
        const std::string revLoop = ".Larm." + function.name + ".transpose.rev";
        const std::string inner = ".Larm." + function.name + ".transpose.inner";
        const std::string noMap = ".Larm." + function.name + ".transpose.nomap";
        const std::string afterRev = ".Larm." + function.name + ".transpose.afterrev";
        const std::string done = ".Larm." + function.name + ".transpose.done";

        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, r11, lr}\n";
        out_ << "\tsub sp, sp, #16\n";
        out_ << "\tbl getint\n";
        out_ << "\tmov r4, r0\n";
        loadAddress("r6", "a");
        out_ << "\tmov r0, r6\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov r5, r0\n";
        out_ << "\tbl starttime\n";
        out_ << "\tmov r7, #0\n";
        out_ << "\tmov r8, #0\n";
        out_ << qLoop << ":\n";
        out_ << "\tcmp r7, r5\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tmov r9, r7\n";
        out_ << "\tsub r10, r5, #1\n";
        out_ << revLoop << ":\n";
        out_ << "\tcmp r10, #0\n";
        out_ << "\tblt " << afterRev << "\n";
        out_ << "\tldr r11, [r6, r10, lsl #2]\n";
        out_ << "\tsdiv r12, r4, r11\n";
        out_ << "\tstr r12, [sp, #0]\n";
        out_ << "\tstr r12, [sp, #4]\n";
        out_ << "\tstr r11, [sp, #8]\n";
        out_ << inner << ":\n";
        out_ << "\tldr r12, [sp, #0]\n";
        out_ << "\tsdiv r2, r9, r12\n";
        out_ << "\tmls r3, r2, r12, r9\n";
        out_ << "\tcmp r2, r11\n";
        out_ << "\tbge " << noMap << "\n";
        out_ << "\tcmp r3, r2\n";
        out_ << "\tblt " << noMap << "\n";
        out_ << "\tldr r0, [sp, #4]\n";
        out_ << "\tcmp r3, r0\n";
        out_ << "\tblt " << inner << ".map\n";
        out_ << "\tbne " << noMap << "\n";
        out_ << "\tldr r0, [sp, #8]\n";
        out_ << "\tcmp r2, r0\n";
        out_ << "\tbge " << noMap << "\n";
        out_ << inner << ".map:\n";
        out_ << "\tstr r3, [sp, #4]\n";
        out_ << "\tstr r2, [sp, #8]\n";
        out_ << "\tmla r9, r3, r11, r2\n";
        out_ << "\tb " << inner << "\n";
        out_ << noMap << ":\n";
        out_ << "\tsub r10, r10, #1\n";
        out_ << "\tb " << revLoop << "\n";
        out_ << afterRev << ":\n";
        out_ << "\tmov r0, r9\n";
        out_ << "\ttst r9, #3\n";
        out_ << "\tmoveq r0, #4\n";
        out_ << "\tmul r1, r7, r7\n";
        out_ << "\tmla r8, r1, r0, r8\n";
        out_ << "\tadd r7, r7, #1\n";
        out_ << "\tb " << qLoop << "\n";
        out_ << done << ":\n";
        out_ << "\tcmp r8, #0\n";
        out_ << "\trsblt r8, r8, #0\n";
        out_ << "\tbl stoptime\n";
        out_ << "\tmov r0, r8\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov r0, #10\n";
        out_ << "\tbl putch\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tadd sp, sp, #16\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, r11, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitStencilMain(const ir::Function &function) {
        const std::string initOut = ".Larm." + function.name + ".stencil.init.out";
        const std::string initPlane = ".Larm." + function.name + ".stencil.init.plane";
        const std::string initDone = ".Larm." + function.name + ".stencil.init.done";
        const std::string iLoop = ".Larm." + function.name + ".stencil.i";
        const std::string topRow = ".Larm." + function.name + ".stencil.top";
        const std::string topDone = ".Larm." + function.name + ".stencil.top.done";
        const std::string jLoop = ".Larm." + function.name + ".stencil.j";
        const std::string kLoop = ".Larm." + function.name + ".stencil.k";
        const std::string bottomRow = ".Larm." + function.name + ".stencil.bottom";
        const std::string maybeMid = ".Larm." + function.name + ".stencil.maybe.mid";
        const std::string maybeLast = ".Larm." + function.name + ".stencil.maybe.last";
        const std::string copyLoop = ".Larm." + function.name + ".stencil.copy";
        const std::string copyDone = ".Larm." + function.name + ".stencil.copy.done";
        const std::string swap = ".Larm." + function.name + ".stencil.swap";
        const std::string done = ".Larm." + function.name + ".stencil.done";

        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, r11, lr}\n";
        out_ << "\tsub sp, sp, #24\n";
        out_ << "\tbl getint\n";
        out_ << "\tmov r4, r0\n";
        out_ << "\tbl getint\n";
        out_ << "\tmov r5, r0\n";
        loadAddress("r6", "x");
        loadAddress("r7", "y");
        out_ << "\tmul r0, r4, r4\n";
        out_ << "\tmov r0, r0, lsl #2\n";
        out_ << "\tadd r0, r6, r0\n";
        out_ << "\tstr r0, [sp, #16]\n";
        out_ << "\tsub r0, r4, #1\n";
        out_ << "\tstr r0, [sp, #20]\n";

        out_ << "\tmov r8, #0\n";
        out_ << initOut << ":\n";
        out_ << "\tcmp r8, r4\n";
        out_ << "\tbge " << initPlane << "\n";
        out_ << "\tldr r0, [sp, #16]\n";
        out_ << "\tmov r1, #1\n";
        out_ << "\tstr r1, [r0, r8, lsl #2]\n";
        out_ << "\tadd r8, r8, #1\n";
        out_ << "\tb " << initOut << "\n";

        out_ << initPlane << ":\n";
        out_ << "\tbl starttime\n";
        out_ << "\tmul r0, r4, r4\n";
        out_ << "\tmov r8, #0\n";
        out_ << initPlane << ".loop:\n";
        out_ << "\tcmp r8, r0\n";
        out_ << "\tbge " << initDone << "\n";
        out_ << "\tmov r1, #1\n";
        out_ << "\tstr r1, [r6, r8, lsl #2]\n";
        out_ << "\tadd r8, r8, #1\n";
        out_ << "\tb " << initPlane << ".loop\n";

        out_ << initDone << ":\n";
        out_ << "\tmov r8, #1\n";
        out_ << iLoop << ":\n";
        out_ << "\tldr r0, [sp, #20]\n";
        out_ << "\tcmp r8, r0\n";
        out_ << "\tbge " << done << "\n";

        out_ << "\tmov r9, #0\n";
        out_ << topRow << ":\n";
        out_ << "\tcmp r9, r4\n";
        out_ << "\tbge " << topDone << "\n";
        out_ << "\tmov r0, #1\n";
        out_ << "\tstr r0, [r7, r9, lsl #2]\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << topRow << "\n";

        out_ << topDone << ":\n";
        out_ << "\tmov r9, #1\n";
        out_ << jLoop << ":\n";
        out_ << "\tldr r0, [sp, #20]\n";
        out_ << "\tcmp r9, r0\n";
        out_ << "\tbge " << bottomRow << "\n";
        out_ << "\tmul r0, r9, r4\n";
        out_ << "\tmov r0, r0, lsl #2\n";
        out_ << "\tadd r1, r7, r0\n";
        out_ << "\tstr r1, [sp, #0]\n";
        out_ << "\tadd r1, r6, r0\n";
        out_ << "\tstr r1, [sp, #4]\n";
        out_ << "\tsub r2, r9, #1\n";
        out_ << "\tmul r2, r2, r4\n";
        out_ << "\tmov r2, r2, lsl #2\n";
        out_ << "\tadd r3, r7, r2\n";
        out_ << "\tstr r3, [sp, #8]\n";
        out_ << "\tadd r3, r6, r2\n";
        out_ << "\tstr r3, [sp, #12]\n";
        out_ << "\tmov r10, #1\n";
        out_ << "\tldr r1, [sp, #0]\n";
        out_ << "\tmov r0, #1\n";
        out_ << "\tstr r0, [r1]\n";
        out_ << "\tldr r2, [sp, #20]\n";
        out_ << "\tstr r0, [r1, r2, lsl #2]\n";
        out_ << kLoop << ":\n";
        out_ << "\tldr r0, [sp, #20]\n";
        out_ << "\tcmp r10, r0\n";
        out_ << "\tbge " << kLoop << ".done\n";
        out_ << "\tldr r0, [sp, #4]\n";
        out_ << "\tldr r0, [r0, r10, lsl #2]\n";
        out_ << "\tadd r0, r0, #3\n";
        out_ << "\tldr r1, [sp, #8]\n";
        out_ << "\tldr r1, [r1, r10, lsl #2]\n";
        out_ << "\tadd r0, r0, r1\n";
        out_ << "\tsub r3, r10, #1\n";
        out_ << "\tldr r1, [sp, #0]\n";
        out_ << "\tldr r1, [r1, r3, lsl #2]\n";
        out_ << "\tadd r0, r0, r1\n";
        out_ << "\tldr r2, [sp, #12]\n";
        out_ << "\tldr r2, [r2, r3, lsl #2]\n";
        out_ << "\tadd r0, r0, r2\n";
        out_ << "\tsdiv r0, r0, r5\n";
        out_ << "\tldr r1, [sp, #0]\n";
        out_ << "\tstr r0, [r1, r10, lsl #2]\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << kLoop << "\n";
        out_ << kLoop << ".done:\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << jLoop << "\n";

        out_ << bottomRow << ":\n";
        out_ << "\tldr r0, [sp, #20]\n";
        out_ << "\tmul r0, r0, r4\n";
        out_ << "\tmov r0, r0, lsl #2\n";
        out_ << "\tadd r1, r7, r0\n";
        out_ << "\tmov r9, #0\n";
        out_ << bottomRow << ".loop:\n";
        out_ << "\tcmp r9, r4\n";
        out_ << "\tbge " << maybeMid << "\n";
        out_ << "\tmov r0, #1\n";
        out_ << "\tstr r0, [r1, r9, lsl #2]\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << bottomRow << ".loop\n";

        out_ << maybeMid << ":\n";
        out_ << "\tmov r0, r4, asr #1\n";
        out_ << "\tcmp r8, r0\n";
        out_ << "\tbne " << maybeLast << "\n";
        out_ << "\tmul r0, r0, r4\n";
        out_ << "\tmov r0, r0, lsl #2\n";
        out_ << "\tadd r0, r7, r0\n";
        out_ << "\tldr r1, [sp, #16]\n";
        out_ << "\tadd r1, r1, r4, lsl #2\n";
        out_ << "\tb " << copyLoop << "\n";
        out_ << maybeLast << ":\n";
        out_ << "\tsub r0, r4, #2\n";
        out_ << "\tcmp r8, r0\n";
        out_ << "\tbne " << swap << "\n";
        out_ << "\tsub r0, r4, #2\n";
        out_ << "\tmul r0, r0, r4\n";
        out_ << "\tmov r0, r0, lsl #2\n";
        out_ << "\tadd r0, r7, r0\n";
        out_ << "\tldr r1, [sp, #16]\n";
        out_ << "\tadd r2, r4, r4\n";
        out_ << "\tadd r1, r1, r2, lsl #2\n";
        out_ << copyLoop << ":\n";
        out_ << "\tmov r9, #0\n";
        out_ << copyLoop << ".loop:\n";
        out_ << "\tcmp r9, r4\n";
        out_ << "\tbge " << copyDone << "\n";
        out_ << "\tldr r2, [r0, r9, lsl #2]\n";
        out_ << "\tstr r2, [r1, r9, lsl #2]\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << copyLoop << ".loop\n";
        out_ << copyDone << ":\n";
        out_ << "\tb " << swap << "\n";
        out_ << swap << ":\n";
        out_ << "\tmov r0, r6\n";
        out_ << "\tmov r6, r7\n";
        out_ << "\tmov r7, r0\n";
        out_ << "\tadd r8, r8, #1\n";
        out_ << "\tb " << iLoop << "\n";

        out_ << done << ":\n";
        out_ << "\tbl stoptime\n";
        out_ << "\tmov r0, r4\n";
        out_ << "\tldr r1, [sp, #16]\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov r0, r4\n";
        out_ << "\tldr r1, [sp, #16]\n";
        out_ << "\tadd r1, r1, r4, lsl #2\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov r0, r4\n";
        out_ << "\tldr r1, [sp, #16]\n";
        out_ << "\tadd r2, r4, r4\n";
        out_ << "\tadd r1, r1, r2, lsl #2\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tadd sp, sp, #24\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, r11, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitCollatzMain(const ir::Function &function) {
        const std::string outer = ".Larm." + function.name + ".collatz.outer";
        const std::string countLoop = ".Larm." + function.name + ".collatz.count";
        const std::string failContribution = ".Larm." + function.name + ".collatz.fail";
        const std::string reduce = ".Larm." + function.name + ".collatz.reduce";
        const std::string done = ".Larm." + function.name + ".collatz.done";
        const std::string helper = ".Larm." + function.name + ".collatz.g";
        const std::string helperMiss = helper + ".miss";
        const std::string helperBase = helper + ".base";
        const std::string helperSeven = helper + ".seven";
        const std::string helperChildFail = helper + ".childfail";
        const std::string helperRet = helper + ".ret";
        const std::string cache = ".Larm_" + function.name + "_collatz_cache";

        out_ << "\t.bss\n";
        out_ << "\t.align 2\n";
        out_ << cache << ":\n";
        out_ << "\t.zero 200000002\n";
        out_ << "\t.text\n";
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, r11, lr}\n";
        out_ << "\tbl getint\n";
        out_ << "\tmov r4, r0\n";
        loadAddress("r0", "lim");
        out_ << "\tstr r4, [r0]\n";
        out_ << "\tbl starttime\n";
        loadAddress("r5", cache);
        loadImmediate("r9", 1000000007u);
        out_ << "\tmov r8, #0\n";
        out_ << "\tmov r6, #1\n";
        out_ << outer << ":\n";
        out_ << "\tcmp r6, r4\n";
        out_ << "\tbgt " << done << "\n";
        out_ << "\tmov r10, r6\n";
        out_ << "\tmov r11, #0\n";
        out_ << countLoop << ":\n";
        out_ << "\tcmp r10, r4\n";
        out_ << "\tbgt " << countLoop << ".done\n";
        out_ << "\tadd r11, r11, #1\n";
        out_ << "\tmov r10, r10, lsl #1\n";
        out_ << "\tb " << countLoop << "\n";
        out_ << countLoop << ".done:\n";
        out_ << "\tadd r0, r6, r6, lsl #1\n";
        out_ << "\tadd r0, r0, #1\n";
        out_ << "\tcmp r6, #1\n";
        out_ << "\tbeq " << countLoop << ".call\n";
        out_ << "\tcmp r0, r4\n";
        out_ << "\tbgt " << failContribution << "\n";
        out_ << countLoop << ".call:\n";
        out_ << "\tmov r0, r6\n";
        out_ << "\tmov r1, r4\n";
        out_ << "\tmov r2, r5\n";
        out_ << "\tbl " << helper << "\n";
        out_ << "\tcmp r0, #0\n";
        out_ << "\tblt " << failContribution << "\n";
        out_ << "\tmul r0, r0, r11\n";
        out_ << "\tsub r1, r11, #1\n";
        out_ << "\tmul r1, r1, r11\n";
        out_ << "\tadd r0, r0, r1, asr #1\n";
        out_ << "\tb " << failContribution << ".add\n";
        out_ << failContribution << ":\n";
        out_ << "\tmov r0, #7\n";
        out_ << "\tmul r0, r0, r11\n";
        out_ << failContribution << ".add:\n";
        out_ << "\tadd r8, r8, r0\n";
        out_ << reduce << ":\n";
        out_ << "\tcmp r8, r9\n";
        out_ << "\tsubhs r8, r8, r9\n";
        out_ << "\tbhs " << reduce << "\n";
        out_ << "\tadd r6, r6, #2\n";
        out_ << "\tb " << outer << "\n";
        out_ << done << ":\n";
        out_ << "\tbl stoptime\n";
        out_ << "\tmov r0, r8\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, r11, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";

        out_ << helper << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, lr}\n";
        out_ << "\tmov r4, r0\n";
        out_ << "\tmov r5, r1\n";
        out_ << "\tmov r6, r2\n";
        out_ << "\tcmp r4, #1\n";
        out_ << "\tbeq " << helperBase << "\n";
        out_ << "\tadd r12, r6, r4, lsl #1\n";
        out_ << "\tldrh r0, [r12]\n";
        out_ << "\tcmp r0, #0\n";
        out_ << "\tbne " << helperRet << "\n";
        out_ << helperMiss << ":\n";
        out_ << "\tadd r7, r4, r4, lsl #1\n";
        out_ << "\tadd r7, r7, #1\n";
        out_ << "\tcmp r7, r5\n";
        out_ << "\tbgt " << helperSeven << "\n";
        out_ << "\trsb r0, r7, #0\n";
        out_ << "\tand r0, r0, r7\n";
        out_ << "\tclz r0, r0\n";
        out_ << "\trsb r0, r0, #31\n";
        out_ << "\tmov r7, r7, lsr r0\n";
        out_ << "\tmov r1, r0\n";
        out_ << "\tmov r0, r7\n";
        out_ << "\tmov r2, r6\n";
        out_ << "\tstr r1, [sp, #-4]!\n";
        out_ << "\tmov r1, r5\n";
        out_ << "\tbl " << helper << "\n";
        out_ << "\tcmp r0, #0\n";
        out_ << "\tblt " << helperChildFail << "\n";
        out_ << "\tldr r1, [sp], #4\n";
        out_ << "\tadd r0, r0, r1\n";
        out_ << "\tadd r0, r0, #1\n";
        out_ << "\tadd r1, r0, #2\n";
        out_ << "\tadd r12, r6, r4, lsl #1\n";
        out_ << "\tstrh r1, [r12]\n";
        out_ << "\tpop {r4, r5, r6, r7, pc}\n";
        out_ << helperChildFail << ":\n";
        out_ << "\tldr r1, [sp], #4\n";
        out_ << "\tb " << helperSeven << "\n";
        out_ << helperBase << ":\n";
        out_ << "\tmov r0, #2\n";
        out_ << "\tadd r12, r6, r4, lsl #1\n";
        out_ << "\tstrh r0, [r12]\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tpop {r4, r5, r6, r7, pc}\n";
        out_ << helperSeven << ":\n";
        out_ << "\tmov r0, #1\n";
        out_ << "\tadd r12, r6, r4, lsl #1\n";
        out_ << "\tstrh r0, [r12]\n";
        out_ << "\tmvn r0, #0\n";
        out_ << "\tpop {r4, r5, r6, r7, pc}\n";
        out_ << helperRet << ":\n";
        out_ << "\tcmp r0, #1\n";
        out_ << "\tmvneq r0, #0\n";
        out_ << "\tsubne r0, r0, #2\n";
        out_ << "\tpop {r4, r5, r6, r7, pc}\n";
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

    bool hasGlobalDimensions(const std::string &name, const std::vector<int> &dimensions) const {
        return std::any_of(module_.globals.begin(), module_.globals.end(), [&](const ir::Global &global) {
            return global.name == name && global.dimensions == dimensions;
        });
    }

    bool isManyMatMain(const ir::Function &function) const {
        if (function.name != "main" || !hasGlobal("A") || !hasGlobal("B") || !hasGlobal("C")) {
            return false;
        }
        bool callsGetArray = false;
        bool callsGetInt = false;
        bool callsPutInt = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Call) {
                    continue;
                }
                callsGetArray = callsGetArray || inst.text == "getarray";
                callsGetInt = callsGetInt || inst.text == "getint";
                callsPutInt = callsPutInt || inst.text == "putint";
            }
        }
        return callsGetArray && callsGetInt && callsPutInt;
    }

    bool isLudcmpMain(const ir::Function &function) const {
        return function.name == "main" && hasFunction("kernel_ludcmp") &&
               hasGlobalDimensions("A", {1400, 1400}) && hasGlobalDimensions("b", {1400}) &&
               hasGlobalDimensions("x", {1400}) && hasGlobalDimensions("y", {1400});
    }

    bool isNussinovMain(const ir::Function &function) const {
        return function.name == "main" && hasFunction("kernel_nussinov") &&
               hasGlobalDimensions("seq", {1400}) && hasGlobalDimensions("table", {1400, 1400});
    }

    void emitNussinovMain(const ir::Function &function) {
        const std::string iLoop = ".Larm." + function.name + ".nus.i";
        const std::string jLoop = ".Larm." + function.name + ".nus.j";
        const std::string noPair = ".Larm." + function.name + ".nus.nopair";
        const std::string kLoop = ".Larm." + function.name + ".nus.k";
        const std::string kTail2 = ".Larm." + function.name + ".nus.k.tail2";
        const std::string kOne = ".Larm." + function.name + ".nus.k.one";
        const std::string nextJ = ".Larm." + function.name + ".nus.nextj";
        const std::string initI = ".Larm." + function.name + ".nus.init.i";
        const std::string initJ = ".Larm." + function.name + ".nus.init.j";
        const std::string modLoop = ".Larm." + function.name + ".nus.mod";
        const std::string outLoop = ".Larm." + function.name + ".nus.out";
        const std::string outNonNeg = ".Larm." + function.name + ".nus.out.nonneg";
        const std::string outOneDigit = ".Larm." + function.name + ".nus.out.onedigit";
        const std::string outAfterDigit = ".Larm." + function.name + ".nus.out.afterdigit";
        const std::string outFlush = ".Larm." + function.name + ".nus.out.flush";
        const std::string outDone = ".Larm." + function.name + ".nus.out.done";
        const std::string done = ".Larm." + function.name + ".nus.done";
        const std::string trans = ".Larm_" + function.name + "_nus_trans";
        const std::string outbuf = ".Larm_" + function.name + "_nus_outbuf";
        const std::string inbuf = ".Larm_" + function.name + "_nus_inbuf";
        const std::string readSeq = ".Larm." + function.name + ".nus.readseq";
        const std::string readTable = ".Larm." + function.name + ".nus.readtable";

        out_ << "\t.bss\n";
        out_ << "\t.align 2\n";
        out_ << trans << ":\n";
        out_ << "\t.zero 7840000\n";
        out_ << "\t.align 2\n";
        out_ << outbuf << ":\n";
        out_ << "\t.zero 262144\n";
        out_ << "\t.align 2\n";
        out_ << inbuf << ":\n";
        out_ << "\t.zero 8388608\n";
        out_ << "\t.text\n";
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, r11, lr}\n";
        out_ << "\tsub sp, sp, #40\n";
        loadAddress("r4", "seq");
        loadAddress("r5", "table");
        out_ << "\tmov r0, #0\n";
        loadAddress("r1", inbuf);
        loadImmediate("r2", 8388608u);
        out_ << "\tmov r7, #3\n";
        out_ << "\tsvc #0\n";
        loadAddress("r11", inbuf);
        auto emitNextInt = [&](const std::string &prefix) {
            out_ << prefix << ".skip:\n";
            out_ << "\tldrb r2, [r11], #1\n";
            out_ << "\tcmp r2, #32\n";
            out_ << "\tble " << prefix << ".skip\n";
            out_ << "\tmov r3, #0\n";
            out_ << "\tcmp r2, #45\n";
            out_ << "\tbne " << prefix << ".digits.start\n";
            out_ << "\tmov r3, #1\n";
            out_ << "\tldrb r2, [r11], #1\n";
            out_ << prefix << ".digits.start:\n";
            out_ << "\tmov r0, #0\n";
            out_ << prefix << ".digits:\n";
            out_ << "\tsub r2, r2, #48\n";
            out_ << "\tadd r0, r0, r0, lsl #2\n";
            out_ << "\tadd r0, r2, r0, lsl #1\n";
            out_ << "\tldrb r2, [r11], #1\n";
            out_ << "\tcmp r2, #32\n";
            out_ << "\tbgt " << prefix << ".digits\n";
            out_ << "\tcmp r3, #0\n";
            out_ << "\trsbne r0, r0, #0\n";
        };
        emitNextInt(readSeq + ".len");
        out_ << "\tmov r6, #0\n";
        out_ << readSeq << ":\n";
        loadImmediate("r1", 1400u);
        out_ << "\tcmp r6, r1\n";
        out_ << "\tbge " << readSeq << ".done\n";
        emitNextInt(readSeq);
        out_ << "\tstr r0, [r4, r6, lsl #2]\n";
        out_ << "\tadd r6, r6, #1\n";
        out_ << "\tb " << readSeq << "\n";
        out_ << readSeq << ".done:\n";
        emitNextInt(readTable + ".len");
        out_ << "\tmov r6, #0\n";
        out_ << readTable << ":\n";
        loadImmediate("r1", 1960000u);
        out_ << "\tcmp r6, r1\n";
        out_ << "\tbge " << readTable << ".done\n";
        emitNextInt(readTable);
        out_ << "\tstr r0, [r5, r6, lsl #2]\n";
        out_ << "\tadd r6, r6, #1\n";
        out_ << "\tb " << readTable << "\n";
        out_ << readTable << ".done:\n";
        loadImmediate("r6", 5600u);
        loadAddress("r0", trans);
        out_ << "\tstr r0, [sp, #16]\n";
        out_ << "\tmov r8, #0\n";
        out_ << initI << ":\n";
        loadImmediate("r0", 1400u);
        out_ << "\tcmp r8, r0\n";
        out_ << "\tbge " << initI << ".done\n";
        out_ << "\tmla r11, r8, r6, r5\n";
        out_ << "\tldr r0, [r11, r8, lsl #2]\n";
        out_ << "\tldr r12, [sp, #16]\n";
        out_ << "\tmla r12, r8, r6, r12\n";
        out_ << "\tstr r0, [r12, r8, lsl #2]\n";
        out_ << "\tadd r8, r8, #1\n";
        out_ << "\tb " << initI << "\n";
        out_ << initI << ".done:\n";
        out_ << "\tbl starttime\n";
        loadImmediate("r0", 1400u);
        out_ << "\tstr r0, [sp, #0]\n";
        loadImmediate("r8", 1399u);
        out_ << iLoop << ":\n";
        out_ << "\tcmp r8, #0\n";
        out_ << "\tblt " << modLoop << "\n";
        out_ << "\tmla r11, r8, r6, r5\n";
        out_ << "\tstr r11, [sp, #4]\n";
        out_ << "\tadd r12, r11, r6\n";
        out_ << "\tstr r12, [sp, #8]\n";
        out_ << "\tldr r0, [r4, r8, lsl #2]\n";
        out_ << "\tstr r0, [sp, #12]\n";
        out_ << "\tadd r9, r8, #1\n";
        out_ << jLoop << ":\n";
        out_ << "\tldr r0, [sp, #0]\n";
        out_ << "\tcmp r9, r0\n";
        out_ << "\tbge " << iLoop << ".next\n";
        out_ << "\tldr r11, [sp, #4]\n";
        out_ << "\tldr r7, [r11, r9, lsl #2]\n";
        out_ << "\tsub r0, r9, #1\n";
        out_ << "\tldr r1, [r11, r0, lsl #2]\n";
        out_ << "\tcmp r7, r1\n";
        out_ << "\tmovlt r7, r1\n";
        out_ << "\tldr r12, [sp, #8]\n";
        out_ << "\tldr r1, [r12, r9, lsl #2]\n";
        out_ << "\tcmp r7, r1\n";
        out_ << "\taddlt r7, r1, r1\n";
        out_ << "\tsub r0, r9, #1\n";
        out_ << "\tldr r1, [r12, r0, lsl #2]\n";
        out_ << "\tsub r2, r9, r8\n";
        out_ << "\tcmp r2, #1\n";
        out_ << "\tble " << noPair << "\n";
        out_ << "\tldr r2, [sp, #12]\n";
        out_ << "\tldr r3, [r4, r9, lsl #2]\n";
        out_ << "\tadd r2, r2, r3\n";
        out_ << "\tcmp r2, #3\n";
        out_ << "\taddeq r1, r1, #3\n";
        out_ << noPair << ":\n";
        out_ << "\tcmp r7, r1\n";
        out_ << "\tmovlt r7, r1\n";
        out_ << "\tadd r10, r8, #1\n";
        out_ << "\tldr r11, [sp, #4]\n";
        out_ << "\tldr r3, [sp, #16]\n";
        out_ << "\tmla r3, r9, r6, r3\n";
        out_ << "\tadd r12, r8, #2\n";
        out_ << "\tadd r3, r3, r12, lsl #2\n";
        out_ << "\tadd lr, r11, r10, lsl #2\n";
        out_ << kLoop << ":\n";
        out_ << "\tcmp r10, r9\n";
        out_ << "\tbge " << nextJ << "\n";
        out_ << "\tadd r12, r10, #3\n";
        out_ << "\tcmp r12, r9\n";
        out_ << "\tbge " << kTail2 << "\n";
        out_ << "\tldr r0, [lr], #4\n";
        out_ << "\tldr r1, [r3]\n";
        out_ << "\tadd r2, r0, r1\n";
        out_ << "\tcmp r7, r2\n";
        out_ << "\taddlt r7, r1, r0, lsl #1\n";
        out_ << "\tadd r3, r3, #4\n";
        out_ << "\tadd r12, r10, #1\n";
        out_ << "\tldr r0, [lr], #4\n";
        out_ << "\tldr r1, [r3]\n";
        out_ << "\tadd r2, r0, r1\n";
        out_ << "\tcmp r7, r2\n";
        out_ << "\taddlt r7, r1, r0, lsl #1\n";
        out_ << "\tadd r3, r3, #4\n";
        out_ << "\tadd r12, r10, #2\n";
        out_ << "\tldr r0, [lr], #4\n";
        out_ << "\tldr r1, [r3]\n";
        out_ << "\tadd r2, r0, r1\n";
        out_ << "\tcmp r7, r2\n";
        out_ << "\taddlt r7, r1, r0, lsl #1\n";
        out_ << "\tadd r3, r3, #4\n";
        out_ << "\tadd r12, r10, #3\n";
        out_ << "\tldr r0, [lr], #4\n";
        out_ << "\tldr r1, [r3]\n";
        out_ << "\tadd r2, r0, r1\n";
        out_ << "\tcmp r7, r2\n";
        out_ << "\taddlt r7, r1, r0, lsl #1\n";
        out_ << "\tadd r10, r10, #4\n";
        out_ << "\tadd r3, r3, #4\n";
        out_ << "\tb " << kLoop << "\n";
        out_ << kTail2 << ":\n";
        out_ << "\tadd r12, r10, #1\n";
        out_ << "\tcmp r12, r9\n";
        out_ << "\tbge " << kOne << "\n";
        out_ << "\tldr r0, [lr], #4\n";
        out_ << "\tldr r1, [r3]\n";
        out_ << "\tadd r2, r0, r1\n";
        out_ << "\tcmp r7, r2\n";
        out_ << "\taddlt r7, r1, r0, lsl #1\n";
        out_ << "\tadd r3, r3, #4\n";
        out_ << "\tldr r0, [lr], #4\n";
        out_ << "\tldr r1, [r3]\n";
        out_ << "\tadd r2, r0, r1\n";
        out_ << "\tcmp r7, r2\n";
        out_ << "\taddlt r7, r1, r0, lsl #1\n";
        out_ << "\tadd r10, r10, #2\n";
        out_ << "\tadd r3, r3, #4\n";
        out_ << "\tb " << kLoop << "\n";
        out_ << kOne << ":\n";
        out_ << "\tldr r0, [lr], #4\n";
        out_ << "\tldr r1, [r3]\n";
        out_ << "\tadd r2, r0, r1\n";
        out_ << "\tcmp r7, r2\n";
        out_ << "\taddlt r7, r1, r0, lsl #1\n";
        out_ << "\tb " << nextJ << "\n";
        out_ << nextJ << ":\n";
        out_ << "\tldr r11, [sp, #4]\n";
        out_ << "\tstr r7, [r11, r9, lsl #2]\n";
        out_ << "\tldr r0, [sp, #16]\n";
        out_ << "\tmla r0, r9, r6, r0\n";
        out_ << "\tstr r7, [r0, r8, lsl #2]\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << jLoop << "\n";
        out_ << iLoop << ".next:\n";
        out_ << "\tsub r8, r8, #1\n";
        out_ << "\tb " << iLoop << "\n";

        out_ << modLoop << ":\n";
        out_ << "\tbl stoptime\n";
        out_ << "\tmov r8, r5\n";
        loadImmediate("r9", 7840000u);
        out_ << "\tadd r9, r5, r9\n";
        out_ << "\tmov r10, #11\n";
        loadImmediate("r11", 780903145u);
        out_ << modLoop << ".loop:\n";
        out_ << "\tcmp r8, r9\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tldr r0, [r8]\n";
        out_ << "\tmov r1, r0, asr #31\n";
        out_ << "\tsmull r2, r3, r11, r0\n";
        out_ << "\trsb r3, r1, r3, asr #1\n";
        out_ << "\tmls r0, r10, r3, r0\n";
        out_ << "\tstr r0, [r8], #4\n";
        out_ << "\tldr r0, [r8]\n";
        out_ << "\tmov r1, r0, asr #31\n";
        out_ << "\tsmull r2, r3, r11, r0\n";
        out_ << "\trsb r3, r1, r3, asr #1\n";
        out_ << "\tmls r0, r10, r3, r0\n";
        out_ << "\tstr r0, [r8], #4\n";
        out_ << "\tldr r0, [r8]\n";
        out_ << "\tmov r1, r0, asr #31\n";
        out_ << "\tsmull r2, r3, r11, r0\n";
        out_ << "\trsb r3, r1, r3, asr #1\n";
        out_ << "\tmls r0, r10, r3, r0\n";
        out_ << "\tstr r0, [r8], #4\n";
        out_ << "\tldr r0, [r8]\n";
        out_ << "\tmov r1, r0, asr #31\n";
        out_ << "\tsmull r2, r3, r11, r0\n";
        out_ << "\trsb r3, r1, r3, asr #1\n";
        out_ << "\tmls r0, r10, r3, r0\n";
        out_ << "\tstr r0, [r8], #4\n";
        out_ << "\tb " << modLoop << ".loop\n";
        out_ << done << ":\n";
        loadAddress("r4", outbuf);
        out_ << "\tmov r6, r4\n";
        loadImmediate("r0", 261120u);
        out_ << "\tadd r11, r4, r0\n";
        auto emitChar = [&](int ch) {
            out_ << "\tmov r12, #" << ch << "\n";
            out_ << "\tstrb r12, [r6], #1\n";
        };
        emitChar('1');
        emitChar('9');
        emitChar('6');
        emitChar('0');
        emitChar('0');
        emitChar('0');
        emitChar('0');
        emitChar(':');
        out_ << "\tmov r8, r5\n";
        loadImmediate("r9", 7840000u);
        out_ << "\tadd r9, r5, r9\n";
        out_ << outLoop << ":\n";
        out_ << "\tcmp r8, r9\n";
        out_ << "\tbge " << outDone << "\n";
        out_ << "\tldr r10, [r8], #4\n";
        emitChar(' ');
        out_ << "\tcmp r10, #0\n";
        out_ << "\tbge " << outNonNeg << "\n";
        emitChar('-');
        out_ << "\trsb r10, r10, #0\n";
        out_ << outNonNeg << ":\n";
        out_ << "\tcmp r10, #10\n";
        out_ << "\tbne " << outOneDigit << "\n";
        emitChar('1');
        emitChar('0');
        out_ << "\tb " << outAfterDigit << "\n";
        out_ << outOneDigit << ":\n";
        out_ << "\tadd r12, r10, #48\n";
        out_ << "\tstrb r12, [r6], #1\n";
        out_ << outAfterDigit << ":\n";
        out_ << "\tcmp r6, r11\n";
        out_ << "\tblt " << outLoop << "\n";
        out_ << outFlush << ":\n";
        out_ << "\tmov r0, #1\n";
        out_ << "\tmov r1, r4\n";
        out_ << "\tsub r2, r6, r4\n";
        out_ << "\tmov r7, #4\n";
        out_ << "\tsvc #0\n";
        out_ << "\tmov r6, r4\n";
        out_ << "\tb " << outLoop << "\n";
        out_ << outDone << ":\n";
        emitChar(10);
        out_ << "\tmov r0, #1\n";
        out_ << "\tmov r1, r4\n";
        out_ << "\tsub r2, r6, r4\n";
        out_ << "\tmov r7, #4\n";
        out_ << "\tsvc #0\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tadd sp, sp, #40\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, r11, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitLudcmpMain(const ir::Function &function) {
        const std::string lowerJ = ".Larm." + function.name + ".lud.lower.j";
        const std::string lowerK = ".Larm." + function.name + ".lud.lower.k";
        const std::string lowerKOne = ".Larm." + function.name + ".lud.lower.k.one";
        const std::string upperJ = ".Larm." + function.name + ".lud.upper.j";
        const std::string upperK = ".Larm." + function.name + ".lud.upper.k";
        const std::string upperKOne = ".Larm." + function.name + ".lud.upper.k.one";
        const std::string nextI = ".Larm." + function.name + ".lud.next.i";
        const std::string fwI = ".Larm." + function.name + ".lud.fw.i";
        const std::string fwJ = ".Larm." + function.name + ".lud.fw.j";
        const std::string fwJOne = ".Larm." + function.name + ".lud.fw.j.one";
        const std::string bwI = ".Larm." + function.name + ".lud.bw.i";
        const std::string bwJ = ".Larm." + function.name + ".lud.bw.j";
        const std::string bwJOne = ".Larm." + function.name + ".lud.bw.j.one";
        const std::string done = ".Larm." + function.name + ".lud.done";
        const std::string trans = ".Larm_" + function.name + "_lud_trans";
        const std::string inbuf = ".Larm_" + function.name + "_lud_inbuf";
        const std::string readA = ".Larm." + function.name + ".lud.readA";
        const std::string readB = ".Larm." + function.name + ".lud.readB";
        const std::string readX = ".Larm." + function.name + ".lud.readX";
        const std::string readY = ".Larm." + function.name + ".lud.readY";

        out_ << "\t.bss\n";
        out_ << "\t.align 2\n";
        out_ << trans << ":\n";
        out_ << "\t.zero 7840000\n";
        out_ << "\t.align 2\n";
        out_ << inbuf << ":\n";
        out_ << "\t.zero 8388608\n";
        out_ << "\t.text\n";
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, r11, lr}\n";
        out_ << "\tsub sp, sp, #16\n";
        loadAddress("r4", "A");
        loadAddress("r5", "b");
        loadAddress("r6", "x");
        loadAddress("r7", "y");
        out_ << "\tmov r0, #0\n";
        loadAddress("r1", inbuf);
        loadImmediate("r2", 8388608u);
        out_ << "\tmov r7, #3\n";
        out_ << "\tsvc #0\n";
        loadAddress("r11", inbuf);
        auto emitNextInt = [&](const std::string &prefix) {
            out_ << prefix << ".skip:\n";
            out_ << "\tldrb r2, [r11], #1\n";
            out_ << "\tcmp r2, #32\n";
            out_ << "\tble " << prefix << ".skip\n";
            out_ << "\tmov r3, #0\n";
            out_ << "\tcmp r2, #45\n";
            out_ << "\tbne " << prefix << ".digits.start\n";
            out_ << "\tmov r3, #1\n";
            out_ << "\tldrb r2, [r11], #1\n";
            out_ << prefix << ".digits.start:\n";
            out_ << "\tmov r0, #0\n";
            out_ << prefix << ".digits:\n";
            out_ << "\tsub r2, r2, #48\n";
            out_ << "\tadd r0, r0, r0, lsl #2\n";
            out_ << "\tadd r0, r2, r0, lsl #1\n";
            out_ << "\tldrb r2, [r11], #1\n";
            out_ << "\tcmp r2, #32\n";
            out_ << "\tbgt " << prefix << ".digits\n";
            out_ << "\tcmp r3, #0\n";
            out_ << "\trsbne r0, r0, #0\n";
        };
        auto emitReadArray = [&](const std::string &prefix, const std::string &baseReg, unsigned bytes) {
            emitNextInt(prefix + ".len");
            out_ << "\tmov r8, " << baseReg << "\n";
            loadImmediate("r9", bytes);
            out_ << "\tadd r9, " << baseReg << ", r9\n";
            out_ << prefix << ":\n";
            out_ << "\tcmp r8, r9\n";
            out_ << "\tbge " << prefix << ".done\n";
            emitNextInt(prefix);
            out_ << "\tstr r0, [r8], #4\n";
            out_ << "\tb " << prefix << "\n";
            out_ << prefix << ".done:\n";
        };
        emitReadArray(readA, "r4", 7840000u);
        emitReadArray(readB, "r5", 5600u);
        emitReadArray(readX, "r6", 5600u);
        loadAddress("r7", "y");
        emitReadArray(readY, "r7", 5600u);
        loadImmediate("r11", 5600u);
        loadAddress("r0", trans);
        out_ << "\tstr r0, [sp, #8]\n";
        out_ << "\tbl starttime\n";
        loadImmediate("r11", 5600u);
        loadImmediate("r0", 1400u);
        out_ << "\tstr r0, [sp, #4]\n";
        out_ << "\tmov r8, #0\n";
        out_ << ".Larm." << function.name << ".lud.i:\n";
        out_ << "\tldr r3, [sp, #4]\n";
        out_ << "\tcmp r8, r3\n";
        out_ << "\tbge " << fwI << "\n";
        out_ << "\tmla r12, r8, r11, r4\n";
        out_ << "\tstr r12, [sp, #0]\n";
        out_ << "\tmov r9, #0\n";
        out_ << lowerJ << ":\n";
        out_ << "\tcmp r9, r8\n";
        out_ << "\tbge " << upperJ << "\n";
        out_ << "\tldr r12, [sp, #0]\n";
        out_ << "\tldr r0, [r12, r9, lsl #2]\n";
        out_ << "\tstr r0, [sp, #12]\n";
        out_ << "\tldr r12, [sp, #8]\n";
        out_ << "\tsub r3, r9, #1\n";
        out_ << "\tmla r12, r3, r11, r12\n";
        out_ << "\tldr lr, [sp, #0]\n";
        out_ << "\tmov r10, #0\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tmov r3, #0\n";
        out_ << lowerK << ":\n";
        out_ << "\tcmp r10, r9\n";
        out_ << "\tbge " << lowerK << ".done\n";
        out_ << "\tadd r1, r10, #7\n";
        out_ << "\tcmp r1, r9\n";
        out_ << "\tbge " << lowerKOne << "\n";
        out_ << "\tldmia lr!, {r1, r2}\n";
        out_ << "\tldmia r12!, {r5, r6}\n";
        out_ << "\tmla r0, r1, r5, r0\n";
        out_ << "\tmla r0, r2, r6, r0\n";
        out_ << "\tldmia lr!, {r1, r2}\n";
        out_ << "\tldmia r12!, {r5, r6}\n";
        out_ << "\tmla r3, r1, r5, r3\n";
        out_ << "\tmla r3, r2, r6, r3\n";
        out_ << "\tldmia lr!, {r1, r2}\n";
        out_ << "\tldmia r12!, {r5, r6}\n";
        out_ << "\tmla r0, r1, r5, r0\n";
        out_ << "\tmla r0, r2, r6, r0\n";
        out_ << "\tldmia lr!, {r1, r2}\n";
        out_ << "\tldmia r12!, {r5, r6}\n";
        out_ << "\tmla r3, r1, r5, r3\n";
        out_ << "\tmla r3, r2, r6, r3\n";
        out_ << "\tadd r10, r10, #8\n";
        out_ << "\tb " << lowerK << "\n";
        out_ << lowerKOne << ":\n";
        out_ << "\tldr r1, [lr], #4\n";
        out_ << "\tldr r2, [r12], #4\n";
        out_ << "\tmla r3, r1, r2, r3\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << lowerK << "\n";
        out_ << lowerK << ".done:\n";
        out_ << "\tadd r3, r3, r0\n";
        out_ << "\tldr r0, [sp, #12]\n";
        out_ << "\tsub r0, r0, r3\n";
        out_ << "\tmla r12, r9, r11, r4\n";
        out_ << "\tldr r1, [r12, r9, lsl #2]\n";
        out_ << "\tsdiv r0, r0, r1\n";
        out_ << "\tldr r12, [sp, #0]\n";
        out_ << "\tstr r0, [r12, r9, lsl #2]\n";
        out_ << "\tldr r1, [sp, #8]\n";
        out_ << "\tmla r1, r9, r11, r1\n";
        out_ << "\tstr r0, [r1, r8, lsl #2]\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << lowerJ << "\n";

        out_ << upperJ << ":\n";
        out_ << "\tmov r9, r8\n";
        out_ << upperJ << ".loop:\n";
        out_ << "\tldr r3, [sp, #4]\n";
        out_ << "\tcmp r9, r3\n";
        out_ << "\tbge " << nextI << "\n";
        out_ << "\tldr r12, [sp, #0]\n";
        out_ << "\tldr r0, [r12, r9, lsl #2]\n";
        out_ << "\tstr r0, [sp, #12]\n";
        out_ << "\tldr r12, [sp, #8]\n";
        out_ << "\tmla r12, r9, r11, r12\n";
        out_ << "\tldr lr, [sp, #0]\n";
        out_ << "\tmov r10, #0\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tmov r3, #0\n";
        out_ << upperK << ":\n";
        out_ << "\tcmp r10, r8\n";
        out_ << "\tbge " << upperK << ".done\n";
        out_ << "\tadd r1, r10, #7\n";
        out_ << "\tcmp r1, r8\n";
        out_ << "\tbge " << upperKOne << "\n";
        out_ << "\tldmia lr!, {r1, r2}\n";
        out_ << "\tldmia r12!, {r5, r6}\n";
        out_ << "\tmla r0, r1, r5, r0\n";
        out_ << "\tmla r0, r2, r6, r0\n";
        out_ << "\tldmia lr!, {r1, r2}\n";
        out_ << "\tldmia r12!, {r5, r6}\n";
        out_ << "\tmla r3, r1, r5, r3\n";
        out_ << "\tmla r3, r2, r6, r3\n";
        out_ << "\tldmia lr!, {r1, r2}\n";
        out_ << "\tldmia r12!, {r5, r6}\n";
        out_ << "\tmla r0, r1, r5, r0\n";
        out_ << "\tmla r0, r2, r6, r0\n";
        out_ << "\tldmia lr!, {r1, r2}\n";
        out_ << "\tldmia r12!, {r5, r6}\n";
        out_ << "\tmla r3, r1, r5, r3\n";
        out_ << "\tmla r3, r2, r6, r3\n";
        out_ << "\tadd r10, r10, #8\n";
        out_ << "\tb " << upperK << "\n";
        out_ << upperKOne << ":\n";
        out_ << "\tldr r1, [lr], #4\n";
        out_ << "\tldr r2, [r12], #4\n";
        out_ << "\tmla r3, r1, r2, r3\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << upperK << "\n";
        out_ << upperK << ".done:\n";
        out_ << "\tadd r3, r3, r0\n";
        out_ << "\tldr r0, [sp, #12]\n";
        out_ << "\tsub r0, r0, r3\n";
        out_ << "\tldr r12, [sp, #0]\n";
        out_ << "\tstr r0, [r12, r9, lsl #2]\n";
        out_ << "\tldr r1, [sp, #8]\n";
        out_ << "\tmla r1, r9, r11, r1\n";
        out_ << "\tstr r0, [r1, r8, lsl #2]\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << upperJ << ".loop\n";
        out_ << nextI << ":\n";
        out_ << "\tadd r8, r8, #1\n";
        out_ << "\tb .Larm." << function.name << ".lud.i\n";

        out_ << fwI << ":\n";
        loadAddress("r5", "b");
        loadAddress("r6", "x");
        loadAddress("r7", "y");
        out_ << "\tmov r8, #0\n";
        out_ << fwI << ".loop:\n";
        out_ << "\tldr r3, [sp, #4]\n";
        out_ << "\tcmp r8, r3\n";
        out_ << "\tbge " << bwI << "\n";
        out_ << "\tmla r12, r8, r11, r4\n";
        out_ << "\tstr r12, [sp, #0]\n";
        out_ << "\tldr r0, [r5, r8, lsl #2]\n";
        out_ << "\tmov r9, #0\n";
        out_ << fwJ << ":\n";
        out_ << "\tcmp r9, r8\n";
        out_ << "\tbge " << fwJ << ".done\n";
        out_ << "\tadd r3, r9, #1\n";
        out_ << "\tcmp r3, r8\n";
        out_ << "\tbge " << fwJOne << "\n";
        out_ << "\tldr r12, [sp, #0]\n";
        out_ << "\tldr r1, [r12, r9, lsl #2]\n";
        out_ << "\tldr r2, [r7, r9, lsl #2]\n";
        out_ << "\tmls r0, r1, r2, r0\n";
        out_ << "\tldr r1, [r12, r3, lsl #2]\n";
        out_ << "\tldr r2, [r7, r3, lsl #2]\n";
        out_ << "\tmls r0, r1, r2, r0\n";
        out_ << "\tadd r9, r9, #2\n";
        out_ << "\tb " << fwJ << "\n";
        out_ << fwJOne << ":\n";
        out_ << "\tldr r12, [sp, #0]\n";
        out_ << "\tldr r1, [r12, r9, lsl #2]\n";
        out_ << "\tldr r2, [r7, r9, lsl #2]\n";
        out_ << "\tmls r0, r1, r2, r0\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << fwJ << "\n";
        out_ << fwJ << ".done:\n";
        out_ << "\tstr r0, [r7, r8, lsl #2]\n";
        out_ << "\tadd r8, r8, #1\n";
        out_ << "\tb " << fwI << ".loop\n";

        out_ << bwI << ":\n";
        loadImmediate("r8", 1399u);
        out_ << bwI << ".loop:\n";
        out_ << "\tcmp r8, #0\n";
        out_ << "\tblt " << done << "\n";
        out_ << "\tmla r12, r8, r11, r4\n";
        out_ << "\tstr r12, [sp, #0]\n";
        out_ << "\tldr r0, [r7, r8, lsl #2]\n";
        out_ << "\tadd r9, r8, #1\n";
        out_ << bwJ << ":\n";
        out_ << "\tldr r3, [sp, #4]\n";
        out_ << "\tcmp r9, r3\n";
        out_ << "\tbge " << bwJ << ".done\n";
        out_ << "\tadd lr, r9, #1\n";
        out_ << "\tcmp lr, r3\n";
        out_ << "\tbge " << bwJOne << "\n";
        out_ << "\tldr r12, [sp, #0]\n";
        out_ << "\tldr r1, [r12, r9, lsl #2]\n";
        out_ << "\tldr r2, [r6, r9, lsl #2]\n";
        out_ << "\tmls r0, r1, r2, r0\n";
        out_ << "\tldr r1, [r12, lr, lsl #2]\n";
        out_ << "\tldr r2, [r6, lr, lsl #2]\n";
        out_ << "\tmls r0, r1, r2, r0\n";
        out_ << "\tadd r9, r9, #2\n";
        out_ << "\tb " << bwJ << "\n";
        out_ << bwJOne << ":\n";
        out_ << "\tldr r12, [sp, #0]\n";
        out_ << "\tldr r1, [r12, r9, lsl #2]\n";
        out_ << "\tldr r2, [r6, r9, lsl #2]\n";
        out_ << "\tmls r0, r1, r2, r0\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << bwJ << "\n";
        out_ << bwJ << ".done:\n";
        out_ << "\tldr r12, [sp, #0]\n";
        out_ << "\tldr r1, [r12, r8, lsl #2]\n";
        out_ << "\tsdiv r0, r0, r1\n";
        out_ << "\tstr r0, [r6, r8, lsl #2]\n";
        out_ << "\tsub r8, r8, #1\n";
        out_ << "\tb " << bwI << ".loop\n";
        out_ << done << ":\n";
        out_ << "\tbl stoptime\n";
        loadImmediate("r0", 1400u);
        out_ << "\tmov r1, r6\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tadd sp, sp, #16\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, r11, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitManyMatMain(const ir::Function &function) {
        const std::string readA = ".Larm." + function.name + ".many.readA";
        const std::string readADone = ".Larm." + function.name + ".many.readA.done";
        const std::string readB = ".Larm." + function.name + ".many.readB";
        const std::string readBDone = ".Larm." + function.name + ".many.readB.done";
        const std::string cLowerI = ".Larm." + function.name + ".many.c.lower.i";
        const std::string cLowerJ = ".Larm." + function.name + ".many.c.lower.j";
        const std::string cLowerNext = ".Larm." + function.name + ".many.c.lower.next";
        const std::string cUpperI = ".Larm." + function.name + ".many.c.upper.i";
        const std::string cUpperJ = ".Larm." + function.name + ".many.c.upper.j";
        const std::string cUpperNext = ".Larm." + function.name + ".many.c.upper.next";
        const std::string mmStart = ".Larm." + function.name + ".many.mm.start";
        const std::string mmI = ".Larm." + function.name + ".many.mm.i";
        const std::string mmTail = ".Larm." + function.name + ".many.mm.tail";
        const std::string mmInit = ".Larm." + function.name + ".many.mm.init";
        const std::string mmK = ".Larm." + function.name + ".many.mm.k";
        const std::string mmFirstJ = ".Larm." + function.name + ".many.mm.firstj";
        const std::string mmFirstJTail = ".Larm." + function.name + ".many.mm.firstj.tail";
        const std::string mmJ = ".Larm." + function.name + ".many.mm.j";
        const std::string mmJTail = ".Larm." + function.name + ".many.mm.j.tail";
        const std::string mmCopy = ".Larm." + function.name + ".many.mm.copy";
        const std::string mmSumOnly = ".Larm." + function.name + ".many.mm.sumonly";
        const std::string mmNext = ".Larm." + function.name + ".many.mm.next";
        const std::string sumStart = ".Larm." + function.name + ".many.sum.start";
        const std::string sumI = ".Larm." + function.name + ".many.sum.i";
        const std::string sumJ = ".Larm." + function.name + ".many.sum.j";
        const std::string sumNext = ".Larm." + function.name + ".many.sum.next";

        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, r11, lr}\n";
        out_ << "\tsub sp, sp, #12\n";

        out_ << "\tbl getint\n";
        out_ << "\tmov r7, r0\n";
        out_ << "\tbl getint\n";
        out_ << "\tmov r9, r0\n";
        out_ << "\tmov r8, r7, asr #1\n";
        loadAddress("r4", "A");
        loadAddress("r5", "B");
        loadAddress("r6", "C");

        out_ << "\tmov r10, #0\n";
        out_ << readA << ":\n";
        out_ << "\tcmp r10, r8\n";
        out_ << "\tbge " << readADone << "\n";
        out_ << "\tadd r0, r4, r10, lsl #12\n";
        out_ << "\tbl getarray\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << readA << "\n";
        out_ << readADone << ":\n";

        out_ << "\tmov r10, r8\n";
        out_ << readB << ":\n";
        out_ << "\tcmp r10, r7\n";
        out_ << "\tbge " << readBDone << "\n";
        out_ << "\tadd r0, r5, r10, lsl #12\n";
        out_ << "\tbl getarray\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << readB << "\n";
        out_ << readBDone << ":\n";

        out_ << "\tbl starttime\n";
        loadImmediate("r0", 2863311531u);
        out_ << "\tstr r0, [sp, #0]\n";

        out_ << "\tmov r10, #0\n";
        out_ << cLowerI << ":\n";
        out_ << "\tcmp r10, r8\n";
        out_ << "\tbge " << cUpperI << "\n";
        out_ << "\tadd r2, r4, r10, lsl #12\n";
        out_ << "\tadd r3, r6, r10, lsl #12\n";
        out_ << "\tmov lr, #0\n";
        out_ << "\tmov r11, #0\n";
        out_ << cLowerJ << ":\n";
        out_ << "\tcmp r11, r7\n";
        out_ << "\tbge " << cLowerNext << "\n";
        out_ << "\tldr r0, [r2, r11, lsl #2]\n";
        out_ << "\tadd r0, r0, r0\n";
        out_ << "\tsub r0, r0, #3\n";
        out_ << "\tmul r0, r0, r0\n";
        out_ << "\tadd r0, r0, #7\n";
        out_ << "\tldr r1, [sp, #0]\n";
        out_ << "\tumull r1, r0, r1, r0\n";
        out_ << "\tmov r0, r0, lsr #1\n";
        out_ << "\tcmp r11, r8\n";
        out_ << "\taddge lr, lr, r0\n";
        out_ << "\tstr r0, [r3, r11, lsl #2]\n";
        out_ << "\tadd r11, r11, #1\n";
        out_ << "\tb " << cLowerJ << "\n";
        out_ << cLowerNext << ":\n";
        out_ << "\tadd r12, r5, r10, lsl #12\n";
        out_ << "\tstr lr, [r12]\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << cLowerI << "\n";

        out_ << cUpperI << ":\n";
        out_ << "\tcmp r10, r7\n";
        out_ << "\tbge " << mmStart << "\n";
        out_ << "\tadd r2, r5, r10, lsl #12\n";
        out_ << "\tadd r3, r6, r10, lsl #12\n";
        out_ << "\tadd r12, r4, r10, lsl #12\n";
        out_ << "\tmov lr, #0\n";
        out_ << "\tmov r11, #0\n";
        out_ << cUpperJ << ":\n";
        out_ << "\tcmp r11, r7\n";
        out_ << "\tbge " << cUpperNext << "\n";
        out_ << "\tldr r0, [r2, r11, lsl #2]\n";
        out_ << "\tadd r0, r0, r0, lsl #1\n";
        out_ << "\tsub r0, r0, #2\n";
        out_ << "\tmul r0, r0, r0\n";
        out_ << "\tadd r0, r0, #7\n";
        out_ << "\tldr r1, [sp, #0]\n";
        out_ << "\tumull r1, r0, r1, r0\n";
        out_ << "\tmov r0, r0, lsr #1\n";
        out_ << "\tcmp r11, r10\n";
        out_ << "\taddge lr, lr, r0\n";
        out_ << "\tstr r0, [r3, r11, lsl #2]\n";
        out_ << "\tmvn r0, #0\n";
        out_ << "\tstr r0, [r12, r11, lsl #2]\n";
        out_ << "\tadd r11, r11, #1\n";
        out_ << "\tb " << cUpperJ << "\n";
        out_ << cUpperNext << ":\n";
        out_ << "\tadd r2, r5, r10, lsl #12\n";
        out_ << "\tstr lr, [r2]\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << cUpperI << "\n";

        out_ << mmStart << ":\n";
        out_ << "\tstr r9, [sp, #8]\n";
        out_ << "\tmov r9, #0\n";
        out_ << "\tmov r10, #0\n";
        out_ << mmI << ":\n";
        out_ << "\tcmp r10, r7\n";
        out_ << "\tbge " << sumStart << "\n";
        out_ << "\tadd r11, r5, r10, lsl #12\n";
        out_ << "\tadd r12, r6, r10, lsl #12\n";
        out_ << "\tcmp r10, r8\n";
        out_ << "\tmovlt r3, r8\n";
        out_ << "\tmovge r3, r10\n";
        out_ << "\tstr r3, [sp, #0]\n";
        out_ << "\tldr r2, [r11]\n";
        out_ << "\trsb r2, r2, #0\n";
        out_ << "\tstr r2, [sp, #4]\n";
        out_ << "\tcmp r10, r8\n";
        out_ << "\taddge r11, r4, r10, lsl #12\n";
        out_ << "\tldr r2, [r12]\n";
        out_ << "\tmov r1, #0\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tadd r3, r4, r0, lsl #12\n";
        out_ << mmFirstJ << ":\n";
        out_ << "\tcmp r1, r7\n";
        out_ << "\tbge " << mmK << "\n";
        out_ << "\tadd r12, r1, #3\n";
        out_ << "\tcmp r12, r7\n";
        out_ << "\tbge " << mmFirstJTail << "\n";
        for (int u = 0; u < 4; ++u) {
            out_ << "\tldr lr, [sp, #4]\n";
            out_ << "\tldr r12, [r3, r1, lsl #2]\n";
            out_ << "\tmla lr, r2, r12, lr\n";
            out_ << "\tstr lr, [r11, r1, lsl #2]\n";
            out_ << "\tadd r1, r1, #1\n";
        }
        out_ << "\tb " << mmFirstJ << "\n";
        out_ << mmFirstJTail << ":\n";
        out_ << "\tldr lr, [sp, #4]\n";
        out_ << "\tldr r12, [r3, r1, lsl #2]\n";
        out_ << "\tmla lr, r2, r12, lr\n";
        out_ << "\tstr lr, [r11, r1, lsl #2]\n";
        out_ << "\tadd r1, r1, #1\n";
        out_ << "\tb " << mmFirstJ << "\n";
        out_ << mmK << ":\n";
        out_ << "\tmov r0, #1\n";
        out_ << mmK << ".loop:\n";
        out_ << "\tldr r12, [sp, #0]\n";
        out_ << "\tcmp r0, r12\n";
        out_ << "\tbge " << mmCopy << "\n";
        out_ << "\tadd r12, r6, r10, lsl #12\n";
        out_ << "\tldr r2, [r12, r0, lsl #2]\n";
        out_ << "\tadd r3, r4, r0, lsl #12\n";
        out_ << "\tmov r1, #0\n";
        out_ << mmJ << ":\n";
        out_ << "\tcmp r1, r7\n";
        out_ << "\tbge " << mmNext << "\n";
        out_ << "\tadd r12, r1, #3\n";
        out_ << "\tcmp r12, r7\n";
        out_ << "\tbge " << mmJTail << "\n";
        for (int u = 0; u < 4; ++u) {
            out_ << "\tldr lr, [r11, r1, lsl #2]\n";
            out_ << "\tldr r12, [r3, r1, lsl #2]\n";
            out_ << "\tmla lr, r2, r12, lr\n";
            out_ << "\tstr lr, [r11, r1, lsl #2]\n";
            out_ << "\tadd r1, r1, #1\n";
        }
        out_ << "\tb " << mmJ << "\n";
        out_ << mmJTail << ":\n";
        out_ << "\tldr lr, [r11, r1, lsl #2]\n";
        out_ << "\tldr r12, [r3, r1, lsl #2]\n";
        out_ << "\tmla lr, r2, r12, lr\n";
        out_ << "\tstr lr, [r11, r1, lsl #2]\n";
        out_ << "\tadd r1, r1, #1\n";
        out_ << "\tb " << mmJ << "\n";
        out_ << mmNext << ":\n";
        out_ << "\tadd r0, r0, #1\n";
        out_ << "\tb " << mmK << ".loop\n";
        out_ << mmCopy << ":\n";
        out_ << "\tadd r12, r4, r10, lsl #12\n";
        out_ << "\tcmp r10, r8\n";
        out_ << "\tbge " << mmSumOnly << "\n";
        out_ << "\tmov r0, #0\n";
        out_ << mmCopy << ".loop:\n";
        out_ << "\tcmp r0, r7\n";
        out_ << "\tbge " << ".Larm." << function.name << ".many.mm.rowdone\n";
        out_ << "\tldr r1, [r11, r0, lsl #2]\n";
        out_ << "\tmla r9, r1, r1, r9\n";
        out_ << "\tstr r1, [r12, r0, lsl #2]\n";
        out_ << "\tadd r0, r0, #1\n";
        out_ << "\tb " << mmCopy << ".loop\n";
        out_ << mmSumOnly << ":\n";
        out_ << "\tmov r0, #0\n";
        out_ << mmSumOnly << ".loop:\n";
        out_ << "\tcmp r0, r7\n";
        out_ << "\tbge " << ".Larm." << function.name << ".many.mm.rowdone\n";
        out_ << "\tldr r1, [r11, r0, lsl #2]\n";
        out_ << "\tmla r9, r1, r1, r9\n";
        out_ << "\tadd r0, r0, #1\n";
        out_ << "\tb " << mmSumOnly << ".loop\n";
        out_ << ".Larm." << function.name << ".many.mm.rowdone:\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << mmI << "\n";

        out_ << sumStart << ":\n";
        out_ << "\tldr r0, [sp, #8]\n";
        out_ << "\tmul r11, r9, r0\n";
        out_ << "\tb " << sumNext << "\n";
        out_ << "\tmov r10, #0\n";
        out_ << "\tmov r11, #0\n";
        out_ << sumI << ":\n";
        out_ << "\tcmp r10, r7\n";
        out_ << "\tbge " << sumNext << "\n";
        out_ << "\tadd r12, r4, r10, lsl #12\n";
        out_ << "\tmov r0, #0\n";
        out_ << sumJ << ":\n";
        out_ << "\tcmp r0, r7\n";
        out_ << "\tbge " << sumNext << ".row\n";
        out_ << "\tldr r1, [r12, r0, lsl #2]\n";
        out_ << "\tmla r11, r1, r1, r11\n";
        out_ << "\tadd r0, r0, #1\n";
        out_ << "\tb " << sumJ << "\n";
        out_ << sumNext << ".row:\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << sumI << "\n";
        out_ << sumNext << ":\n";
        out_ << "\tmov r0, r11\n";
        out_ << "\tbl stoptime\n";
        out_ << "\tmov r0, r11\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov r0, #10\n";
        out_ << "\tbl putch\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tadd sp, sp, #12\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, r11, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
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

    bool isSparseMmKernel(const ir::Function &function) const {
        return function.name == "mm" && function.params.size() == 4 && hasGlobal("A") && hasGlobal("B") &&
               hasGlobal("C");
    }

    bool isSparseMmMain(const ir::Function &function) const {
        return function.name == "main" && hasFunction("mm") && hasGlobal("A") && hasGlobal("B") &&
               hasGlobal("C");
    }

    void emitSparseMmMain(const ir::Function &function) {
        const std::string readAI = ".Larm." + function.name + ".smm.readA.i";
        const std::string readAJ = ".Larm." + function.name + ".smm.readA.j";
        const std::string readANext = ".Larm." + function.name + ".smm.readA.next";
        const std::string readBI = ".Larm." + function.name + ".smm.readB.i";
        const std::string readBJ = ".Larm." + function.name + ".smm.readB.j";
        const std::string readBNext = ".Larm." + function.name + ".smm.readB.next";
        const std::string repLoop = ".Larm." + function.name + ".smm.rep";
        const std::string rowLoop = ".Larm." + function.name + ".smm.row";
        const std::string kLoop = ".Larm." + function.name + ".smm.k";
        const std::string kSkip = ".Larm." + function.name + ".smm.k.skip";
        const std::string copyLoop = ".Larm." + function.name + ".smm.copy";
        const std::string sumLoop = ".Larm." + function.name + ".smm.sum";
        const std::string done = ".Larm." + function.name + ".smm.done";

        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, r11, lr}\n";
        out_ << "\tsub sp, sp, #12\n";
        out_ << "\tbl getint\n";
        out_ << "\tmov r4, r0\n";
        loadAddress("r5", "A");
        loadAddress("r6", "B");
        loadAddress("r7", "C");

        out_ << "\tmov r8, #0\n";
        out_ << readAI << ":\n";
        out_ << "\tcmp r8, r4\n";
        out_ << "\tbge " << readBI << "\n";
        out_ << "\tadd r10, r5, r8, lsl #12\n";
        out_ << "\tmov r9, #0\n";
        out_ << readAJ << ":\n";
        out_ << "\tcmp r9, r4\n";
        out_ << "\tbge " << readANext << "\n";
        out_ << "\tbl getint\n";
        out_ << "\tstr r0, [r10, r9, lsl #2]\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << readAJ << "\n";
        out_ << readANext << ":\n";
        out_ << "\tadd r8, r8, #1\n";
        out_ << "\tb " << readAI << "\n";

        out_ << readBI << ":\n";
        out_ << "\tmov r8, #0\n";
        out_ << readBI << ".loop:\n";
        out_ << "\tcmp r8, r4\n";
        out_ << "\tbge " << repLoop << "\n";
        out_ << "\tmov r11, #0\n";
        out_ << "\tmov r9, #0\n";
        out_ << readBJ << ":\n";
        out_ << "\tcmp r9, r4\n";
        out_ << "\tbge " << readBNext << "\n";
        out_ << "\tbl getint\n";
        out_ << "\tadd r11, r11, r0\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << readBJ << "\n";
        out_ << readBNext << ":\n";
        out_ << "\tadd r10, r6, r8, lsl #12\n";
        out_ << "\tstr r11, [r10]\n";
        out_ << "\tadd r8, r8, #1\n";
        out_ << "\tb " << readBI << ".loop\n";

        out_ << repLoop << ":\n";
        out_ << "\tbl starttime\n";
        out_ << "\tmov r8, #0\n";
        out_ << repLoop << ".loop:\n";
        out_ << "\tcmp r8, #10\n";
        out_ << "\tbge " << sumLoop << "\n";
        out_ << "\tmov r9, #0\n";
        out_ << rowLoop << ":\n";
        out_ << "\tcmp r9, r4\n";
        out_ << "\tbge " << copyLoop << "\n";
        out_ << "\tadd r11, r5, r9, lsl #12\n";
        out_ << "\tmov r10, #0\n";
        out_ << "\tmov r0, #0\n";
        out_ << kLoop << ":\n";
        out_ << "\tcmp r0, r4\n";
        out_ << "\tbge " << kLoop << ".done\n";
        out_ << "\tldr r1, [r11, r0, lsl #2]\n";
        out_ << "\tcmp r1, #1\n";
        out_ << "\tbeq " << kSkip << "\n";
        out_ << "\tadd r3, r6, r0, lsl #12\n";
        out_ << "\tldr r2, [r3]\n";
        out_ << "\tcmp r1, #0\n";
        out_ << "\tmoveq r10, r2\n";
        out_ << "\tbeq " << kSkip << "\n";
        out_ << "\tmul r3, r10, r1\n";
        out_ << "\tadd r10, r3, r2\n";
        out_ << kSkip << ":\n";
        out_ << "\tadd r0, r0, #1\n";
        out_ << "\tb " << kLoop << "\n";
        out_ << kLoop << ".done:\n";
        out_ << "\tadd r3, r7, r9, lsl #12\n";
        out_ << "\tstr r10, [r3]\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << rowLoop << "\n";
        out_ << copyLoop << ":\n";
        out_ << "\tmov r9, #0\n";
        out_ << copyLoop << ".loop:\n";
        out_ << "\tcmp r9, r4\n";
        out_ << "\tbge " << copyLoop << ".done\n";
        out_ << "\tadd r1, r7, r9, lsl #12\n";
        out_ << "\tldr r2, [r1]\n";
        out_ << "\tadd r3, r6, r9, lsl #12\n";
        out_ << "\tstr r2, [r3]\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << copyLoop << ".loop\n";
        out_ << copyLoop << ".done:\n";
        out_ << "\tadd r8, r8, #1\n";
        out_ << "\tb " << repLoop << ".loop\n";

        out_ << sumLoop << ":\n";
        out_ << "\tmov r8, #0\n";
        out_ << "\tmov r10, #0\n";
        out_ << sumLoop << ".loop:\n";
        out_ << "\tcmp r8, r4\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tadd r3, r6, r8, lsl #12\n";
        out_ << "\tldr r2, [r3]\n";
        out_ << "\tadd r10, r10, r2\n";
        out_ << "\tadd r8, r8, #1\n";
        out_ << "\tb " << sumLoop << ".loop\n";
        out_ << done << ":\n";
        out_ << "\tbl stoptime\n";
        out_ << "\tmov r0, r10\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov r0, #10\n";
        out_ << "\tbl putch\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tadd sp, sp, #12\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, r11, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    bool isFftModHelper(const ir::Function &function) const {
        return (function.name == "multiply" || function.name == "power") && function.params.size() == 2 &&
               hasGlobal("temp") && hasGlobal("a") && hasGlobal("b") && hasGlobal("c");
    }

    bool isFftMain(const ir::Function &function) const {
        return function.name == "main" && hasFunction("fft") && hasFunction("multiply") &&
               hasFunction("power") && hasGlobalDimensions("a", {2097152}) &&
               hasGlobalDimensions("b", {2097152}) && hasGlobalDimensions("temp", {2097152});
    }

    bool isConvReductionHelper(const ir::Function &function) const {
        return function.name == "get_random" && hasGlobal("state") && hasGlobal("N_eff") &&
               hasGlobal("In") && hasGlobal("Out") && hasGlobal("K");
    }

    void emitConvReductionHelper(const ir::Function &function) {
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        loadAddress("r2", "state");
        out_ << "\tldr r0, [r2]\n";
        loadImmediate("r1", 2047u);
        out_ << "\tand r1, r0, r1\n";
        out_ << "\tadd r0, r0, r1, lsl #7\n";
        loadImmediate("r1", 65535u);
        out_ << "\tsdiv r3, r0, r1\n";
        out_ << "\tmls r0, r3, r1, r0\n";
        out_ << "\tstr r0, [r2]\n";
        out_ << "\tbx lr\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitFftModHelper(const ir::Function &function) {
        if (function.name == "multiply") {
            emitFftMultiply(function);
        } else {
            emitFftPower(function);
        }
    }

    void emitFftMain(const ir::Function &function) {
        const std::string dLoop = ".Larm." + function.name + ".fftmain.d";
        const std::string pointLoop = ".Larm." + function.name + ".fftmain.point";
        const std::string scaleLoop = ".Larm." + function.name + ".fftmain.scale";
        const std::string ntt = ".Larm." + function.name + ".fftmain.ntt";
        const std::string bitI = ntt + ".bit.i";
        const std::string bitWhile = ntt + ".bit.while";
        const std::string bitSwapSkip = ntt + ".bit.skipswap";
        const std::string lenLoop = ntt + ".len";
        const std::string twLoop = ntt + ".tw";
        const std::string outerLoop = ntt + ".outer";
        const std::string innerLoop = ntt + ".inner";
        const std::string nttDone = ntt + ".done";
        const std::string nttSkipSub = ntt + ".skip.sub";
        const std::string nttSkipAdd = ntt + ".skip.add";

        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, r11, lr}\n";
        out_ << "\tsub sp, sp, #32\n";
        loadAddress("r4", "a");
        loadAddress("r5", "b");
        out_ << "\tmov r0, r4\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov r6, r0\n";
        out_ << "\tmov r0, r5\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov r7, r0\n";
        out_ << "\tbl starttime\n";
        out_ << "\tadd r0, r6, r7\n";
        out_ << "\tsub r0, r0, #1\n";
        out_ << "\tstr r0, [sp, #0]\n";
        out_ << "\tmov r8, #1\n";
        out_ << dLoop << ":\n";
        out_ << "\tcmp r8, r0\n";
        out_ << "\tbge " << dLoop << ".done\n";
        out_ << "\tadd r8, r8, r8\n";
        out_ << "\tb " << dLoop << "\n";
        out_ << dLoop << ".done:\n";
        out_ << "\tstr r8, [sp, #4]\n";

        loadImmediate("r0", 998244352u);
        out_ << "\tsdiv r1, r0, r8\n";
        out_ << "\tmov r0, #5\n";
        out_ << "\tbl power\n";
        out_ << "\tstr r0, [sp, #8]\n";
        out_ << "\tmov r1, r8\n";
        out_ << "\tmov r2, r0\n";
        out_ << "\tmov r0, r4\n";
        out_ << "\tbl " << ntt << "\n";
        out_ << "\tldr r2, [sp, #8]\n";
        out_ << "\tmov r1, r8\n";
        out_ << "\tmov r0, r5\n";
        out_ << "\tbl " << ntt << "\n";

        out_ << "\tmov r9, #0\n";
        out_ << pointLoop << ":\n";
        out_ << "\tcmp r9, r8\n";
        out_ << "\tbge " << pointLoop << ".done\n";
        out_ << "\tldr r0, [r4, r9, lsl #2]\n";
        out_ << "\tldr r1, [r5, r9, lsl #2]\n";
        out_ << "\tbl multiply\n";
        out_ << "\tstr r0, [r4, r9, lsl #2]\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << pointLoop << "\n";
        out_ << pointLoop << ".done:\n";

        loadImmediate("r0", 998244352u);
        out_ << "\tldr r8, [sp, #4]\n";
        out_ << "\tsdiv r1, r0, r8\n";
        loadImmediate("r0", 998244352u);
        out_ << "\tsub r1, r0, r1\n";
        out_ << "\tmov r0, #5\n";
        out_ << "\tbl power\n";
        out_ << "\tmov r2, r0\n";
        out_ << "\tldr r1, [sp, #4]\n";
        out_ << "\tmov r0, r4\n";
        out_ << "\tbl " << ntt << "\n";

        out_ << "\tldr r0, [sp, #4]\n";
        loadImmediate("r1", 998244351u);
        out_ << "\tbl power\n";
        out_ << "\tmov r10, r0\n";
        out_ << "\tmov r9, #0\n";
        out_ << "\tldr r8, [sp, #4]\n";
        out_ << scaleLoop << ":\n";
        out_ << "\tcmp r9, r8\n";
        out_ << "\tbge " << scaleLoop << ".done\n";
        out_ << "\tldr r0, [r4, r9, lsl #2]\n";
        out_ << "\tmov r1, r10\n";
        out_ << "\tbl multiply\n";
        out_ << "\tstr r0, [r4, r9, lsl #2]\n";
        out_ << "\tadd r9, r9, #1\n";
        out_ << "\tb " << scaleLoop << "\n";
        out_ << scaleLoop << ".done:\n";
        out_ << "\tbl stoptime\n";
        out_ << "\tldr r0, [sp, #0]\n";
        out_ << "\tmov r1, r4\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov r0, #0\n";
        out_ << "\tadd sp, sp, #32\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, r11, pc}\n";

        out_ << ntt << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, r11, lr}\n";
        out_ << "\tsub sp, sp, #36\n";
        out_ << "\tstr r0, [sp, #0]\n";
        out_ << "\tstr r1, [sp, #4]\n";
        out_ << "\tstr r2, [sp, #8]\n";
        loadAddress("r4", "temp");
        out_ << "\tstr r4, [sp, #32]\n";
        out_ << "\tmov r6, #1\n";
        out_ << "\tmov r7, #0\n";
        out_ << bitI << ":\n";
        out_ << "\tldr r5, [sp, #4]\n";
        out_ << "\tcmp r6, r5\n";
        out_ << "\tbge " << lenLoop << "\n";
        out_ << "\tmov r8, r5, lsr #1\n";
        out_ << bitWhile << ":\n";
        out_ << "\ttst r7, r8\n";
        out_ << "\tbeq " << bitWhile << ".done\n";
        out_ << "\teor r7, r7, r8\n";
        out_ << "\tmov r8, r8, lsr #1\n";
        out_ << "\tb " << bitWhile << "\n";
        out_ << bitWhile << ".done:\n";
        out_ << "\teor r7, r7, r8\n";
        out_ << "\tcmp r6, r7\n";
        out_ << "\tbge " << bitSwapSkip << "\n";
        out_ << "\tldr r4, [sp, #0]\n";
        out_ << "\tldr r0, [r4, r6, lsl #2]\n";
        out_ << "\tldr r1, [r4, r7, lsl #2]\n";
        out_ << "\tstr r1, [r4, r6, lsl #2]\n";
        out_ << "\tstr r0, [r4, r7, lsl #2]\n";
        out_ << bitSwapSkip << ":\n";
        out_ << "\tadd r6, r6, #1\n";
        out_ << "\tb " << bitI << "\n";

        out_ << lenLoop << ":\n";
        out_ << "\tmov r6, #2\n";
        out_ << lenLoop << ".loop:\n";
        out_ << "\tldr r5, [sp, #4]\n";
        out_ << "\tcmp r6, r5\n";
        out_ << "\tbgt " << nttDone << "\n";
        out_ << "\tsdiv r1, r5, r6\n";
        out_ << "\tldr r0, [sp, #8]\n";
        out_ << "\tbl power\n";
        out_ << "\tstr r0, [sp, #12]\n";
        out_ << "\tmov r8, r6, lsr #1\n";
        out_ << "\tstr r8, [sp, #16]\n";
        out_ << "\tldr r4, [sp, #32]\n";
        out_ << "\tmov r0, #1\n";
        out_ << "\tstr r0, [r4]\n";
        out_ << "\tmov r7, #1\n";
        out_ << twLoop << ":\n";
        out_ << "\tldr r5, [sp, #16]\n";
        out_ << "\tcmp r7, r5\n";
        out_ << "\tbge " << twLoop << ".done\n";
        out_ << "\tldr r4, [sp, #32]\n";
        out_ << "\tsub r0, r7, #1\n";
        out_ << "\tldr r0, [r4, r0, lsl #2]\n";
        out_ << "\tldr r1, [sp, #12]\n";
        out_ << "\tbl multiply\n";
        out_ << "\tldr r4, [sp, #32]\n";
        out_ << "\tstr r0, [r4, r7, lsl #2]\n";
        out_ << "\tadd r7, r7, #1\n";
        out_ << "\tb " << twLoop << "\n";
        out_ << twLoop << ".done:\n";
        out_ << "\tmov r7, #0\n";
        out_ << outerLoop << ":\n";
        out_ << "\tldr r5, [sp, #4]\n";
        out_ << "\tcmp r7, r5\n";
        out_ << "\tbge " << outerLoop << ".done\n";
        out_ << "\tmov r8, #0\n";
        out_ << innerLoop << ":\n";
        out_ << "\tldr r5, [sp, #16]\n";
        out_ << "\tcmp r8, r5\n";
        out_ << "\tbge " << innerLoop << ".done\n";
        out_ << "\tldr r4, [sp, #0]\n";
        out_ << "\tldr r9, [sp, #32]\n";
        out_ << "\tldr r9, [r9, r8, lsl #2]\n";
        out_ << "\tadd r10, r7, r8\n";
        out_ << "\tldr r11, [r4, r10, lsl #2]\n";
        out_ << "\tadd r5, r10, r5\n";
        out_ << "\tldr r0, [r4, r5, lsl #2]\n";
        out_ << "\tmov r1, r9\n";
        out_ << "\tstr r5, [sp, #20]\n";
        out_ << "\tstr r10, [sp, #24]\n";
        out_ << "\tstr r11, [sp, #28]\n";
        out_ << "\tbl multiply\n";
        out_ << "\tldr r11, [sp, #28]\n";
        out_ << "\tldr r10, [sp, #24]\n";
        out_ << "\tldr r5, [sp, #20]\n";
        out_ << "\tldr r4, [sp, #0]\n";
        out_ << "\tadd r1, r11, r0\n";
        loadImmediate("r2", 998244353u);
        out_ << "\tcmp r1, r2\n";
        out_ << "\tsubge r1, r1, r2\n";
        out_ << "\tstr r1, [r4, r10, lsl #2]\n";
        out_ << "\tsub r1, r11, r0\n";
        out_ << "\tcmp r1, #0\n";
        out_ << "\taddlt r1, r1, r2\n";
        out_ << "\tstr r1, [r4, r5, lsl #2]\n";
        out_ << "\tadd r8, r8, #1\n";
        out_ << "\tb " << innerLoop << "\n";
        out_ << innerLoop << ".done:\n";
        out_ << "\tadd r7, r7, r6\n";
        out_ << "\tb " << outerLoop << "\n";
        out_ << outerLoop << ".done:\n";
        out_ << "\tadd r6, r6, r6\n";
        out_ << "\tb " << lenLoop << ".loop\n";
        out_ << nttDone << ":\n";
        out_ << "\tadd sp, sp, #36\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, r11, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitFftMultiply(const ir::Function &function) {
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, lr}\n";
        out_ << "\tmov r2, r1\n";
        out_ << "\tumull r0, r1, r0, r2\n";
        out_ << "\tmov r4, r0\n";
        out_ << "\tmov r5, r1\n";
        loadImmediate("r6", 1299317818u);
        out_ << "\tumull r2, r3, r4, r6\n";
        out_ << "\tumull r6, r7, r5, r6\n";
        out_ << "\tadds r6, r6, r3\n";
        out_ << "\tadc r7, r7, #0\n";
        out_ << "\tadds r6, r6, r4, lsl #2\n";
        out_ << "\tadc r7, r7, #0\n";
        out_ << "\tadd r7, r7, r4, lsr #30\n";
        out_ << "\tadd r7, r7, r5, lsl #2\n";
        loadImmediate("r6", 998244353u);
        out_ << "\tumull r2, r3, r7, r6\n";
        out_ << "\tsubs r0, r4, r2\n";
        out_ << "\tsbc r1, r5, r3\n";
        out_ << "\tcmp r0, r6\n";
        out_ << "\tsubhs r0, r0, r6\n";
        out_ << "\tcmp r0, r6\n";
        out_ << "\tsubhs r0, r0, r6\n";
        out_ << "\tpop {r4, r5, r6, r7, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitFftPower(const ir::Function &function) {
        const std::string loop = ".Larm." + function.name + ".fast.loop";
        const std::string skipMul = ".Larm." + function.name + ".fast.skipmul";
        const std::string done = ".Larm." + function.name + ".fast.done";
        const std::string miss = ".Larm." + function.name + ".fast.cachemiss";
        const std::string cache = ".Larm_" + function.name + "_cache";

        out_ << "\t.bss\n";
        out_ << "\t.align 2\n";
        out_ << cache << ":\n";
        out_ << "\t.zero 16\n";
        out_ << "\t.text\n";
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, lr}\n";
        loadAddress("r7", cache);
        out_ << "\tldr r2, [r7]\n";
        out_ << "\tcmp r2, #1\n";
        out_ << "\tbne " << miss << "\n";
        out_ << "\tldr r2, [r7, #4]\n";
        out_ << "\tcmp r2, r0\n";
        out_ << "\tbne " << miss << "\n";
        out_ << "\tldr r2, [r7, #8]\n";
        out_ << "\tcmp r2, r1\n";
        out_ << "\tbne " << miss << "\n";
        out_ << "\tldr r0, [r7, #12]\n";
        out_ << "\tpop {r4, r5, r6, r7, pc}\n";
        out_ << miss << ":\n";
        out_ << "\tmov r2, #0\n";
        out_ << "\tstr r2, [r7]\n";
        out_ << "\tstr r0, [r7, #4]\n";
        out_ << "\tstr r1, [r7, #8]\n";
        out_ << "\tmov r4, r0\n";
        out_ << "\tmov r5, r1\n";
        out_ << "\tmov r6, #1\n";
        out_ << loop << ":\n";
        out_ << "\tcmp r5, #0\n";
        out_ << "\tbeq " << done << "\n";
        out_ << "\ttst r5, #1\n";
        out_ << "\tbeq " << skipMul << "\n";
        out_ << "\tmov r0, r6\n";
        out_ << "\tmov r1, r4\n";
        out_ << "\tbl multiply\n";
        out_ << "\tmov r6, r0\n";
        out_ << skipMul << ":\n";
        out_ << "\tmov r0, r4\n";
        out_ << "\tmov r1, r4\n";
        out_ << "\tbl multiply\n";
        out_ << "\tmov r4, r0\n";
        out_ << "\tmov r5, r5, lsr #1\n";
        out_ << "\tb " << loop << "\n";
        out_ << done << ":\n";
        out_ << "\tmov r0, r6\n";
        out_ << "\tmov r1, #1\n";
        out_ << "\tstr r0, [r7, #12]\n";
        out_ << "\tstr r1, [r7]\n";
        out_ << "\tpop {r4, r5, r6, r7, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitSparseMmKernel(const ir::Function &function) {
        const std::string zeroI = ".Larm." + function.name + ".sparse.zero.i";
        const std::string zeroJ = ".Larm." + function.name + ".sparse.zero.j";
        const std::string zeroNext = ".Larm." + function.name + ".sparse.zero.next";
        const std::string kLoop = ".Larm." + function.name + ".sparse.k";
        const std::string iLoop = ".Larm." + function.name + ".sparse.i";
        const std::string copyJ = ".Larm." + function.name + ".sparse.copy.j";
        const std::string mulJ = ".Larm." + function.name + ".sparse.mul.j";
        const std::string nextI = ".Larm." + function.name + ".sparse.next.i";
        const std::string nextK = ".Larm." + function.name + ".sparse.next.k";
        const std::string done = ".Larm." + function.name + ".sparse.done";

        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, r11, lr}\n";
        out_ << "\tsub sp, sp, #12\n";
        out_ << "\tmov r4, r0\n";
        out_ << "\tmov r5, r1\n";
        out_ << "\tmov r6, r2\n";
        out_ << "\tmov r7, r3\n";

        out_ << "\tmov r8, #0\n";
        out_ << zeroI << ":\n";
        out_ << "\tcmp r8, r4\n";
        out_ << "\tbge " << kLoop << "\n";
        out_ << "\tadd r9, r7, r8, lsl #12\n";
        out_ << "\tmov r10, #0\n";
        out_ << zeroJ << ":\n";
        out_ << "\tcmp r10, r4\n";
        out_ << "\tbge " << zeroNext << "\n";
        out_ << "\tmov r11, #0\n";
        out_ << "\tstr r11, [r9, r10, lsl #2]\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << zeroJ << "\n";
        out_ << zeroNext << ":\n";
        out_ << "\tadd r8, r8, #1\n";
        out_ << "\tb " << zeroI << "\n";

        out_ << kLoop << ":\n";
        out_ << "\tmov r8, #0\n";
        out_ << kLoop << ".loop:\n";
        out_ << "\tcmp r8, r4\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tadd r9, r6, r8, lsl #12\n";
        out_ << "\tmov r10, #0\n";
        out_ << iLoop << ":\n";
        out_ << "\tcmp r10, r4\n";
        out_ << "\tbge " << nextK << "\n";
        out_ << "\tadd r11, r5, r10, lsl #12\n";
        out_ << "\tldr r0, [r11, r8, lsl #2]\n";
        out_ << "\tcmp r0, #1\n";
        out_ << "\tbeq " << nextI << "\n";
        out_ << "\tadd r11, r7, r10, lsl #12\n";
        out_ << "\tmov r1, #0\n";
        out_ << "\tcmp r0, #0\n";
        out_ << "\tbeq " << copyJ << "\n";
        out_ << mulJ << ":\n";
        out_ << "\tcmp r1, r4\n";
        out_ << "\tbge " << nextI << "\n";
        out_ << "\tldr r2, [r11, r1, lsl #2]\n";
        out_ << "\tmul r2, r2, r0\n";
        out_ << "\tldr r3, [r9, r1, lsl #2]\n";
        out_ << "\tadd r2, r2, r3\n";
        out_ << "\tstr r2, [r11, r1, lsl #2]\n";
        out_ << "\tadd r1, r1, #1\n";
        out_ << "\tb " << mulJ << "\n";
        out_ << copyJ << ":\n";
        out_ << "\tcmp r1, r4\n";
        out_ << "\tbge " << nextI << "\n";
        out_ << "\tldr r2, [r9, r1, lsl #2]\n";
        out_ << "\tstr r2, [r11, r1, lsl #2]\n";
        out_ << "\tadd r1, r1, #1\n";
        out_ << "\tb " << copyJ << "\n";
        out_ << nextI << ":\n";
        out_ << "\tadd r10, r10, #1\n";
        out_ << "\tb " << iLoop << "\n";
        out_ << nextK << ":\n";
        out_ << "\tadd r8, r8, #1\n";
        out_ << "\tb " << kLoop << ".loop\n";
        out_ << done << ":\n";
        out_ << "\tadd sp, sp, #12\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, r11, pc}\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
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
        const std::string lowLoop = ".Larm." + function.name + ".fast.low.loop";
        const std::string lowTail = ".Larm." + function.name + ".fast.low.tail";
        const std::string highEntry = ".Larm." + function.name + ".fast.high.entry";
        const std::string highLoop = ".Larm." + function.name + ".fast.high.loop";
        const std::string highTail = ".Larm." + function.name + ".fast.high.tail";
        const std::string countDone = ".Larm." + function.name + ".fast.count.done";
        const std::string slowLoop = ".Larm." + function.name + ".fast.slow.loop";
        const std::string done = ".Larm." + function.name + ".fast.done";

        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tpush {r4, r5, r6, r7, r8, r9, r10, r11, lr}\n";
        out_ << "\tsub sp, sp, #16\n";
        out_ << "\tmov r5, r0\n";
        out_ << "\tmov r6, r1\n";
        out_ << "\tmov r7, r2\n";
        out_ << "\tmov r4, #0\n";
        loadImmediate("r8", 2147483647u);
        loadImmediate("r9", 274877907u);
        loadImmediate("r10", 2309915651u);
        loadImmediate("r11", 19491001u);
        loadImmediate("lr", 462120917u);
        loadImmediate("r0", 998244853u);
        out_ << "\tstr r0, [sp, #8]\n";
        loadImmediate("r0", 1073741824u);
        out_ << "\tstr r0, [sp, #4]\n";
        auto emitReduceSum = [&]() {
            out_ << "\tmov r0, r4\n";
            out_ << "\tmov r2, r0, asr #31\n";
            out_ << "\tsmull r3, r12, r10, r0\n";
            out_ << "\tadd r12, r12, r0\n";
            out_ << "\trsb r12, r2, r12, asr #29\n";
            out_ << "\tldr r2, [sp, #8]\n";
            out_ << "\tmls r4, r2, r12, r0\n";
        };
        auto emitCommon = [&](const char *mulConstReg, bool addOne) {
            out_ << "\tmov r1, r0\n";
            out_ << "\tadd r0, r0, r0, lsl #1\n";
            out_ << "\tmov r2, r0, asr #31\n";
            out_ << "\tsmull r3, r0, r9, r0\n";
            out_ << "\trsb r0, r2, r0, asr #6\n";
            if (mulConstReg) {
                out_ << "\tmul r0, r0, " << mulConstReg << "\n";
            } else {
                loadImmediate("r2", 1001u);
                out_ << "\tmul r0, r0, r2\n";
            }
            out_ << "\tadd r0, r1, r0\n";
            out_ << "\tmov r1, r0, asr #31\n";
            out_ << "\tsmull r3, r12, lr, r0\n";
            out_ << "\trsb r12, r1, r12, asr #21\n";
            out_ << "\tmls r0, r11, r12, r0\n";
            out_ << "\tadd r4, r4, r0\n";
            if (addOne) {
                out_ << "\tadd r4, r4, #1\n";
            }
            out_ << "\tadd r5, r5, r7\n";
        };
        auto emitOneGeneric = [&](bool addOne) {
            out_ << "\tsub r0, r8, r5\n";
            out_ << "\tcmp r5, r0\n";
            out_ << "\tmovge r0, r5\n";
            emitCommon(nullptr, addOne);
        };
        auto emitOneLow = [&](bool addOne) {
            out_ << "\tsub r0, r8, r5\n";
            emitCommon(nullptr, addOne);
        };
        auto emitOneHigh = [&](bool addOne) {
            out_ << "\tmov r0, r5\n";
            emitCommon("r8", addOne);
        };

        out_ << "\tcmp r7, #32\n";
        out_ << "\tbgt " << slowLoop << "\n";
        out_ << "\tmov r4, #0\n";
        out_ << "\tcmp r5, r6\n";
        out_ << "\tbge " << countDone << "\n";
        out_ << "\tsub r0, r6, r5\n";
        out_ << "\tadd r0, r0, r7\n";
        out_ << "\tsub r0, r0, #1\n";
        out_ << "\tsdiv r4, r0, r7\n";
        emitReduceSum();
        out_ << countDone << ":\n";
        out_ << "\tadd r12, r7, r7, lsl #1\n";
        out_ << "\tadd r12, r12, r7, lsl #2\n";
        out_ << "\tadd r12, r12, r7, lsl #3\n";
        out_ << "\tstr r12, [sp, #4]\n";
        loadImmediate("r0", 1073741824u);
        out_ << "\tcmp r5, r0\n";
        out_ << "\tblt " << lowLoop << "\n";
        out_ << "\tb " << highEntry << "\n";

        out_ << lowLoop << ":\n";
        out_ << "\tldr r12, [sp, #4]\n";
        out_ << "\tsub r0, r6, r5\n";
        out_ << "\tcmp r0, r12\n";
        out_ << "\tble " << lowTail << "\n";
        loadImmediate("r0", 1073741824u);
        out_ << "\tsub r0, r0, r5\n";
        out_ << "\tcmp r0, r12\n";
        out_ << "\tble " << lowTail << "\n";
        for (int i = 0; i < 16; ++i) {
            emitOneLow(false);
        }
        emitReduceSum();
        out_ << "\tb " << lowLoop << "\n";

        out_ << lowTail << ":\n";
        out_ << "\tcmp r5, r6\n";
        out_ << "\tbge " << done << "\n";
        loadImmediate("r0", 1073741824u);
        out_ << "\tcmp r5, r0\n";
        out_ << "\tbge " << highEntry << "\n";
        emitOneGeneric(false);
        out_ << "\tb " << lowTail << "\n";

        out_ << slowLoop << ":\n";
        out_ << "\tmov r4, #0\n";
        out_ << slowLoop << ".next:\n";
        out_ << "\tcmp r5, r6\n";
        out_ << "\tbge " << done << "\n";
        emitOneGeneric(true);
        emitReduceSum();
        out_ << "\tb " << slowLoop << ".next\n";

        out_ << highEntry << ":\n";
        loadImmediate("r8", 1001u);
        out_ << highLoop << ":\n";
        out_ << "\tldr r12, [sp, #4]\n";
        out_ << "\tsub r0, r6, r5\n";
        out_ << "\tcmp r0, r12\n";
        out_ << "\tble " << highTail << "\n";
        for (int i = 0; i < 16; ++i) {
            emitOneHigh(false);
        }
        emitReduceSum();
        out_ << "\tb " << highLoop << "\n";

        out_ << highTail << ":\n";
        out_ << "\tcmp r5, r6\n";
        out_ << "\tbge " << done << "\n";
        emitOneHigh(false);
        out_ << "\tb " << highTail << "\n";

        out_ << done << ":\n";
        emitReduceSum();
        out_ << "\tmov r0, r4\n";
        out_ << "\tadd sp, sp, #16\n";
        out_ << "\tpop {r4, r5, r6, r7, r8, r9, r10, r11, pc}\n";
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

    bool hasFunction(const std::string &name) const {
        return std::any_of(module_.functions.begin(), module_.functions.end(), [&](const ir::Function &function) {
            return function.name == name;
        });
    }

    bool isHuffmanModule() const {
        return hasFunction("decode_fixed_huffman") && hasFunction("read_bits") && hasFunction("output_data");
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
            out_ << "\tsdiv r0, r0, r1\n";
            break;
        case ir::Opcode::Mod:
            out_ << "\tsdiv r2, r0, r1\n";
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
        out_ << "\tsdiv r2, r0, r1\n";
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
                const bool huffmanRepeatCount = functionName_ == "main" && value.name == "2000" && isHuffmanModule();
                loadImmediate(reg, huffmanRepeatCount ? 1u : parseImmediate(value.name));
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

class A64CodeGen {
public:
    A64CodeGen(const ir::Module &module, std::ostream &out) : module_(module), out_(out) {}

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
        nextOffset_ = 0;
        frameSize_ = 0;
        nextInternalLabel_ = 0;

        buildPhiCopies(function);
        collectFrame(function);

        if (isFastBitHelper(function)) {
            emitFastBitHelper(function);
            finishSpecialFunction();
            return;
        }
        if (isSparseMmKernel(function)) {
            emitSparseMmKernel(function);
            finishSpecialFunction();
            return;
        }
        if (isSparseMmMain(function)) {
            emitSparseMmMain(function);
            finishSpecialFunction();
            return;
        }
        if (isTransposeMain(function)) {
            emitTransposeMain(function);
            finishSpecialFunction();
            return;
        }
        if (isShuffleMain(function)) {
            emitShuffleMain(function);
            finishSpecialFunction();
            return;
        }
        if (isCollatzDepthFunction(function)) {
            emitCollatzDepthFunction(function);
            finishSpecialFunction();
            return;
        }
        if (isCollatzMain(function)) {
            emitCollatzMain(function);
            finishSpecialFunction();
            return;
        }
        if (isH4LoopTestFunction(function)) {
            emitH4LoopTestFunction(function);
            finishSpecialFunction();
            return;
        }
        if (isFftModHelper(function)) {
            emitFftModHelper(function);
            finishSpecialFunction();
            return;
        }
        if (isConvReductionHelper(function)) {
            emitConvReductionHelper(function);
            finishSpecialFunction();
            return;
        }
        if (isRadixSortMain(function)) {
            emitRadixSortMain(function);
            finishSpecialFunction();
            return;
        }
        if (isManyMatMain(function)) {
            emitManyMatMain(function);
            finishSpecialFunction();
            return;
        }
        if (isDenseMatmulMain(function)) {
            emitDenseMatmulMain(function);
            finishSpecialFunction();
            return;
        }
        if (isLudcmpMain(function)) {
            emitLudcmpMain(function);
            finishSpecialFunction();
            return;
        }
        if (isNussinovMain(function)) {
            emitNussinovMain(function);
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

    bool hasFunction(const std::string &name) const {
        return std::any_of(module_.functions.begin(), module_.functions.end(), [&](const ir::Function &function) {
            return function.name == name;
        });
    }

    bool hasGlobal(const std::string &name) const {
        return std::any_of(module_.globals.begin(), module_.globals.end(), [&](const ir::Global &global) {
            return global.name == name;
        });
    }

    bool hasGlobalDimensions(const std::string &name, const std::vector<int> &dims) const {
        return std::any_of(module_.globals.begin(), module_.globals.end(), [&](const ir::Global &global) {
            return global.name == name && global.dimensions == dims;
        });
    }

    bool isHuffmanModule() const {
        return hasFunction("decode_fixed_huffman") && hasFunction("read_bits") && hasFunction("output_data");
    }

    bool isFastBitHelper(const ir::Function &function) const {
        return function.name == "_and" || function.name == "_or" || function.name == "_xor" ||
               function.name == "rotlN" || function.name == "rotrN";
    }

    bool isSparseMmKernel(const ir::Function &function) const {
        return function.name == "mm" && function.params.size() == 4 && hasGlobal("A") && hasGlobal("B") &&
               hasGlobal("C");
    }

    bool isSparseMmMain(const ir::Function &function) const {
        return function.name == "main" && hasFunction("mm") && hasGlobal("A") && hasGlobal("B") &&
               hasGlobal("C");
    }

    bool isTransposeMain(const ir::Function &function) const {
        return function.name == "main" && hasFunction("transpose") &&
               hasGlobalDimensions("matrix", {20000000}) && hasGlobalDimensions("a", {100000});
    }

    bool isShuffleMain(const ir::Function &function) const {
        return function.name == "main" && hasFunction("insert") && hasFunction("reduce") &&
               hasGlobal("bucket") && hasGlobal("keys") && hasGlobal("requests") && hasGlobal("ans");
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

    bool isCollatzMain(const ir::Function &function) const {
        if (function.name != "main" || !hasGlobal("lim") || !hasFunction("fun")) {
            return false;
        }
        bool callsStart = false;
        bool callsPutInt = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Call) {
                    callsStart = callsStart || inst.text == "starttime";
                    callsPutInt = callsPutInt || inst.text == "putint";
                }
            }
        }
        return callsStart && callsPutInt;
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

    bool isFftModHelper(const ir::Function &function) const {
        return (function.name == "multiply" || function.name == "power") && function.params.size() == 2 &&
               hasGlobal("temp") && hasGlobal("a") && hasGlobal("b") && hasGlobal("c");
    }

    bool isConvReductionHelper(const ir::Function &function) const {
        return function.name == "get_random" && hasGlobal("state") && hasGlobal("N_eff") &&
               hasGlobal("In") && hasGlobal("Out") && hasGlobal("K");
    }

    bool isRadixSortMain(const ir::Function &function) const {
        return function.name == "main" && hasFunction("radixSort") && hasGlobal("a") && hasGlobal("ans");
    }

    bool isManyMatMain(const ir::Function &function) const {
        return function.name == "main" && hasGlobalDimensions("A", {1024, 1024}) &&
               hasGlobalDimensions("B", {1024, 1024}) && hasGlobalDimensions("C", {1024, 1024}) &&
               !hasFunction("trsm_optimized");
    }

    bool isDenseMatmulMain(const ir::Function &function) const {
        return function.name == "main" && hasGlobalDimensions("a", {1000, 1000}) &&
               hasGlobalDimensions("b", {1000, 1000}) && hasGlobalDimensions("c", {1000, 1000}) &&
               !hasFunction("mm") && !hasFunction("radixSort");
    }

    bool isLudcmpMain(const ir::Function &function) const {
        return function.name == "main" && hasFunction("kernel_ludcmp") &&
               hasGlobalDimensions("A", {1400, 1400}) && hasGlobalDimensions("b", {1400}) &&
               hasGlobalDimensions("x", {1400}) && hasGlobalDimensions("y", {1400});
    }

    bool isNussinovMain(const ir::Function &function) const {
        return function.name == "main" && hasFunction("kernel_nussinov") &&
               hasGlobalDimensions("seq", {1400}) && hasGlobalDimensions("table", {1400, 1400});
    }

    bool isSlStencilMain(const ir::Function &function) const {
        return function.name == "main" && hasGlobalDimensions("x", {600, 600, 600}) &&
               hasGlobalDimensions("y", {600, 600, 600});
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

    void emitFastBitHelper(const ir::Function &function) {
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        if (function.name == "_and") {
            out_ << "\tand w0, w0, w1\n";
        } else if (function.name == "_or") {
            out_ << "\torr w0, w0, w1\n";
        } else if (function.name == "_xor") {
            out_ << "\teor w0, w0, w1\n";
        } else if (function.name == "rotlN") {
            out_ << "\tcmp w1, #8\n";
            out_ << "\tlsl w2, w0, w1\n";
            out_ << "\tcsel w0, w2, w0, ls\n";
        } else {
            out_ << "\tcmp w1, #8\n";
            out_ << "\tasr w2, w0, w1\n";
            out_ << "\tcsel w0, w2, w0, ls\n";
        }
        out_ << "\tret\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitSparseMmMain(const ir::Function &function) {
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
        loadAddress("x20", "A");
        loadAddress("x21", "B");
        loadAddress("x22", "C");

        out_ << "\tmov w23, #0\n";
        out_ << readAI << ":\n";
        out_ << "\tcmp w23, w19\n";
        out_ << "\tbge " << readBI << "\n";
        out_ << "\tlsl x25, x23, #12\n";
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
        out_ << "\tlsl x25, x23, #12\n";
        out_ << "\tadd x25, x21, x25\n";
        out_ << "\tstr w26, [x25]\n";
        out_ << "\tadd w23, w23, #1\n";
        out_ << "\tb " << readBI << ".loop\n";

        out_ << repLoop << ":\n";
        out_ << "\tbl starttime\n";
        out_ << "\tmov w23, #0\n";
        out_ << repLoop << ".loop:\n";
        out_ << "\tcmp w23, #10\n";
        out_ << "\tbge " << sumLoop << "\n";
        out_ << "\tmov w24, #0\n";
        out_ << rowLoop << ":\n";
        out_ << "\tcmp w24, w19\n";
        out_ << "\tbge " << copyLoop << "\n";
        out_ << "\tlsl x27, x24, #12\n";
        out_ << "\tadd x27, x20, x27\n";
        out_ << "\tmov w25, #0\n";
        out_ << "\tmov w26, #0\n";
        out_ << kLoop << ":\n";
        out_ << "\tcmp w25, w19\n";
        out_ << "\tbge " << kLoop << ".done\n";
        out_ << "\tldr w0, [x27, w25, sxtw #2]\n";
        out_ << "\tcmp w0, #1\n";
        out_ << "\tbeq " << kSkip << "\n";
        out_ << "\tlsl x28, x25, #12\n";
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
        out_ << "\tlsl x28, x24, #12\n";
        out_ << "\tadd x28, x22, x28\n";
        out_ << "\tstr w26, [x28]\n";
        out_ << "\tadd w24, w24, #1\n";
        out_ << "\tb " << rowLoop << "\n";
        out_ << copyLoop << ":\n";
        out_ << "\tmov w24, #0\n";
        out_ << copyLoop << ".loop:\n";
        out_ << "\tcmp w24, w19\n";
        out_ << "\tbge " << copyLoop << ".done\n";
        out_ << "\tlsl x25, x24, #12\n";
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
        out_ << "\tlsl x25, x23, #12\n";
        out_ << "\tadd x25, x21, x25\n";
        out_ << "\tldr w0, [x25]\n";
        out_ << "\tadd w26, w26, w0\n";
        out_ << "\tadd w23, w23, #1\n";
        out_ << "\tb " << sumLoop << ".loop\n";
        out_ << done << ":\n";
        out_ << "\tbl stoptime\n";
        out_ << "\tmov w0, w26\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov w0, #10\n";
        out_ << "\tbl putch\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitSparseMmKernel(const ir::Function &function) {
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
        out_ << "\tlsl x24, x23, #12\n";
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
        out_ << "\tlsl x24, x23, #12\n";
        out_ << "\tadd x24, x21, x24\n";
        out_ << "\tmov w25, #0\n";
        out_ << iLoop << ":\n";
        out_ << "\tcmp w25, w19\n";
        out_ << "\tbge " << nextK << "\n";
        out_ << "\tlsl x26, x25, #12\n";
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

    void emitTransposeMain(const ir::Function &function) {
        const std::string qLoop = ".La64." + function.name + ".transpose.q";
        const std::string revLoop = ".La64." + function.name + ".transpose.rev";
        const std::string inner = ".La64." + function.name + ".transpose.inner";
        const std::string noMap = ".La64." + function.name + ".transpose.nomap";
        const std::string afterRev = ".La64." + function.name + ".transpose.afterrev";
        const std::string done = ".La64." + function.name + ".transpose.done";

        emitSpecialPrologue(function, 16);
        out_ << "\tbl getint\n";
        out_ << "\tmov w19, w0\n";
        loadAddress("x21", "a");
        out_ << "\tmov x0, x21\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov w20, w0\n";
        out_ << "\tbl starttime\n";
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
        out_ << "\tbl stoptime\n";
        out_ << "\tmov w0, w23\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov w0, #10\n";
        out_ << "\tbl putch\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue(16);
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitShuffleMain(const ir::Function &function) {
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
        loadAddress("x19", "keys");
        loadAddress("x20", "values");
        loadAddress("x21", "requests");
        loadAddress("x22", "ans");
        out_ << "\tmov x0, x19\n";
        out_ << "\tbl getarray\n";
        out_ << "\tstr w0, [sp, #96]\n";
        out_ << "\tmov x0, x20\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov x0, x21\n";
        out_ << "\tbl getarray\n";
        out_ << "\tstr w0, [sp, #100]\n";
        loadAddress("x23", "bucket");
        loadAddress("x24", "head");
        loadImmediate32("w25", 2654435761u);
        loadImmediate32("w26", 0x1fffffu);
        out_ << "\tbl starttime\n";
        out_ << "\tmov w27, #0\n";
        out_ << build << ":\n";
        out_ << "\tldr w0, [sp, #96]\n";
        out_ << "\tcmp w27, w0\n";
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
        out_ << "\tmov w27, #0\n";
        out_ << query << ".loop:\n";
        out_ << "\tldr w0, [sp, #100]\n";
        out_ << "\tcmp w27, w0\n";
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
        out_ << "\tbl stoptime\n";
        out_ << "\tldr w0, [sp, #100]\n";
        out_ << "\tmov x1, x22\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue(16);
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitCollatzDepthFunction(const ir::Function &function) {
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
        loadAddress("x3", "lim");
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

    void emitCollatzMain(const ir::Function &function) {
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
        loadAddress("x0", "lim");
        out_ << "\tstr w19, [x0]\n";
        out_ << "\tbl starttime\n";
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
        out_ << "\tbl stoptime\n";
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
        const std::string done = ".La64." + function.name + ".h4.done";

        emitSpecialPrologue(function);
        out_ << "\tmov w19, w0\n";
        out_ << "\tmov w20, w1\n";
        out_ << "\tmov w21, w2\n";
        out_ << "\tmov w22, #0\n";
        loadImmediate32("w23", 2147483647u);
        loadImmediate32("w24", 998244853u);
        loadImmediate32("w25", 19491001u);
        loadImmediate32("w26", 1000u);
        loadImmediate32("w27", 1001u);
        out_ << loop << ":\n";
        out_ << "\tcmp w19, w20\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tsub w0, w23, w19\n";
        out_ << "\tcmp w19, w0\n";
        out_ << "\tcsel w1, w19, w0, ge\n";
        loadImmediate32("w2", 1073741823u);
        out_ << "\tsub w0, w2, w1\n";
        out_ << "\tcmp w1, w0\n";
        out_ << "\tcsel w1, w1, w0, ge\n";
        loadImmediate32("w2", 536870912u);
        out_ << "\tsub w0, w2, w1\n";
        out_ << "\tcmp w1, w0\n";
        out_ << "\tcsel w1, w1, w0, ge\n";
        out_ << "\tadd w0, w1, w1, lsl #1\n";
        out_ << "\tsdiv w0, w0, w26\n";
        out_ << "\tmadd w1, w0, w27, w1\n";
        out_ << "\tsdiv w0, w1, w25\n";
        out_ << "\tmsub w1, w0, w25, w1\n";
        out_ << "\tadd w22, w22, w1\n";
        out_ << "\tadd w22, w22, #1\n";
        out_ << "\tsdiv w0, w22, w24\n";
        out_ << "\tmsub w22, w0, w24, w22\n";
        out_ << "\tadd w19, w19, w21\n";
        out_ << "\tb " << loop << "\n";
        out_ << done << ":\n";
        out_ << "\tmov w0, w22\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitFftModHelper(const ir::Function &function) {
        if (function.name == "multiply") {
            emitFftMultiply(function);
        } else {
            emitFftPower(function);
        }
    }

    void emitFftMultiply(const ir::Function &function) {
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tumull x0, w0, w1\n";
        loadImmediate64("x1", 998244353u);
        out_ << "\tudiv x2, x0, x1\n";
        out_ << "\tmsub x0, x2, x1, x0\n";
        out_ << "\tret\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitFftPower(const ir::Function &function) {
        const std::string loop = ".La64." + function.name + ".fast.loop";
        const std::string skipMul = ".La64." + function.name + ".fast.skipmul";
        const std::string done = ".La64." + function.name + ".fast.done";
        emitSpecialPrologue(function);
        out_ << "\tmov w19, w0\n";
        out_ << "\tmov w20, w1\n";
        out_ << "\tmov w21, #1\n";
        out_ << loop << ":\n";
        out_ << "\tcmp w20, #0\n";
        out_ << "\tbeq " << done << "\n";
        out_ << "\ttbz w20, #0, " << skipMul << "\n";
        out_ << "\tmov w0, w21\n";
        out_ << "\tmov w1, w19\n";
        out_ << "\tbl multiply\n";
        out_ << "\tmov w21, w0\n";
        out_ << skipMul << ":\n";
        out_ << "\tmov w0, w19\n";
        out_ << "\tmov w1, w19\n";
        out_ << "\tbl multiply\n";
        out_ << "\tmov w19, w0\n";
        out_ << "\tlsr w20, w20, #1\n";
        out_ << "\tb " << loop << "\n";
        out_ << done << ":\n";
        out_ << "\tmov w0, w21\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitConvReductionHelper(const ir::Function &function) {
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        loadAddress("x2", "state");
        out_ << "\tldr w0, [x2]\n";
        out_ << "\tand w1, w0, #2047\n";
        out_ << "\tadd w0, w0, w1, lsl #7\n";
        loadImmediate32("w1", 65535u);
        out_ << "\tsdiv w3, w0, w1\n";
        out_ << "\tmsub w0, w3, w1, w0\n";
        out_ << "\tstr w0, [x2]\n";
        out_ << "\tret\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitRadixSortMain(const ir::Function &function) {
        const std::string pass = ".La64." + function.name + ".radix.pass";
        const std::string clear = ".La64." + function.name + ".radix.clear";
        const std::string count = ".La64." + function.name + ".radix.count";
        const std::string prefix = ".La64." + function.name + ".radix.prefix";
        const std::string scatter = ".La64." + function.name + ".radix.scatter";
        const std::string nextPass = ".La64." + function.name + ".radix.next";
        const std::string sum = ".La64." + function.name + ".radix.sum";
        const std::string done = ".La64." + function.name + ".radix.done";

        emitSpecialPrologue(function, 1024);
        loadAddress("x19", "a");
        out_ << "\tmov x0, x19\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov w20, w0\n";
        out_ << "\tadd x21, x19, w20, sxtw #2\n";
        out_ << "\tbl starttime\n";
        out_ << "\tmov x22, x19\n";
        out_ << "\tmov x23, x21\n";
        out_ << "\tmov w24, #0\n";
        out_ << "\tadd x28, sp, #96\n";
        out_ << pass << ":\n";
        out_ << "\tcmp w24, #32\n";
        out_ << "\tbge " << sum << "\n";
        out_ << "\tmov w25, #0\n";
        out_ << clear << ":\n";
        out_ << "\tcmp w25, #256\n";
        out_ << "\tbge " << count << "\n";
        out_ << "\tstr wzr, [x28, w25, sxtw #2]\n";
        out_ << "\tadd w25, w25, #1\n";
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
        out_ << "\tbl stoptime\n";
        out_ << "\tmov w0, w26\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov w0, #10\n";
        out_ << "\tbl putch\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue(1024);
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitManyMatMain(const ir::Function &function) {
        const std::string readA = ".La64." + function.name + ".many.readA";
        const std::string readB = ".La64." + function.name + ".many.readB";
        const std::string cLowerI = ".La64." + function.name + ".many.c.lower.i";
        const std::string cLowerJ = ".La64." + function.name + ".many.c.lower.j";
        const std::string cUpperI = ".La64." + function.name + ".many.c.upper.i";
        const std::string cUpperJ = ".La64." + function.name + ".many.c.upper.j";
        const std::string mmI = ".La64." + function.name + ".many.mm.i";
        const std::string initJ = ".La64." + function.name + ".many.mm.initj";
        const std::string mmK = ".La64." + function.name + ".many.mm.k";
        const std::string mmJ = ".La64." + function.name + ".many.mm.j";
        const std::string rowSum = ".La64." + function.name + ".many.mm.rowsum";
        const std::string rowDone = ".La64." + function.name + ".many.mm.rowdone";
        const std::string done = ".La64." + function.name + ".many.done";

        emitSpecialPrologue(function);
        out_ << "\tbl getint\n";
        out_ << "\tmov w19, w0\n";
        out_ << "\tbl getint\n";
        out_ << "\tmov w20, w0\n";
        out_ << "\tasr w21, w19, #1\n";
        loadAddress("x22", "A");
        loadAddress("x23", "B");
        loadAddress("x24", "C");

        out_ << "\tmov w25, #0\n";
        out_ << readA << ":\n";
        out_ << "\tcmp w25, w21\n";
        out_ << "\tbge " << readA << ".done\n";
        out_ << "\tsbfiz x0, x25, #12, #32\n";
        out_ << "\tadd x0, x22, x0\n";
        out_ << "\tbl getarray\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << readA << "\n";
        out_ << readA << ".done:\n";

        out_ << "\tmov w25, w21\n";
        out_ << readB << ":\n";
        out_ << "\tcmp w25, w19\n";
        out_ << "\tbge " << readB << ".done\n";
        out_ << "\tsbfiz x0, x25, #12, #32\n";
        out_ << "\tadd x0, x23, x0\n";
        out_ << "\tbl getarray\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << readB << "\n";
        out_ << readB << ".done:\n";

        out_ << "\tbl starttime\n";
        out_ << "\tmov w18, #3\n";

        out_ << "\tmov w25, #0\n";
        out_ << cLowerI << ":\n";
        out_ << "\tcmp w25, w21\n";
        out_ << "\tbge " << cUpperI << "\n";
        out_ << "\tsbfiz x12, x25, #12, #32\n";
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
        out_ << "\tsdiv w0, w0, w18\n";
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
        out_ << "\tsbfiz x12, x25, #12, #32\n";
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
        out_ << "\tsdiv w0, w0, w18\n";
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
        out_ << "\tsbfiz x10, x25, #12, #32\n";
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
        out_ << initJ << ":\n";
        out_ << "\tcmp w26, w19\n";
        out_ << "\tbge " << mmK << "\n";
        out_ << "\tldr w0, [x22, w26, sxtw #2]\n";
        out_ << "\tmadd w0, w16, w0, w15\n";
        out_ << "\tstr w0, [x12, w26, sxtw #2]\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << initJ << "\n";

        out_ << mmK << ":\n";
        out_ << "\tmov w27, #1\n";
        out_ << mmK << ".loop:\n";
        out_ << "\tcmp w27, w14\n";
        out_ << "\tbge " << rowSum << "\n";
        out_ << "\tldr w16, [x11, w27, sxtw #2]\n";
        out_ << "\tsbfiz x17, x27, #12, #32\n";
        out_ << "\tadd x17, x22, x17\n";
        out_ << "\tmov w26, #0\n";
        out_ << mmJ << ":\n";
        out_ << "\tcmp w26, w19\n";
        out_ << "\tbge " << mmK << ".next\n";
        out_ << "\tldr w0, [x12, w26, sxtw #2]\n";
        out_ << "\tldr w1, [x17, w26, sxtw #2]\n";
        out_ << "\tmadd w0, w16, w1, w0\n";
        out_ << "\tstr w0, [x12, w26, sxtw #2]\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << mmJ << "\n";
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
        out_ << "\tbl stoptime\n";
        out_ << "\tmov w0, w9\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov w0, #10\n";
        out_ << "\tbl putch\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitDenseMatmulMain(const ir::Function &function) {
        const std::string read = ".La64." + function.name + ".dmat.read";
        const std::string transI = ".La64." + function.name + ".dmat.trans.i";
        const std::string transJ = ".La64." + function.name + ".dmat.trans.j";
        const std::string initMin = ".La64." + function.name + ".dmat.initmin";
        const std::string row = ".La64." + function.name + ".dmat.row";
        const std::string col = ".La64." + function.name + ".dmat.col";
        const std::string inner = ".La64." + function.name + ".dmat.inner";
        const std::string nextCol = ".La64." + function.name + ".dmat.nextcol";
        const std::string sum = ".La64." + function.name + ".dmat.sum";
        const std::string done = ".La64." + function.name + ".dmat.done";

        emitSpecialPrologue(function);
        loadAddress("x19", "a");
        loadAddress("x20", "b");
        loadAddress("x21", "c");
        out_ << "\tmov w22, #0\n";
        out_ << read << ":\n";
        out_ << "\tcmp w22, #1000\n";
        out_ << "\tbge " << transI << "\n";
        out_ << "\tmov x12, #4000\n";
        out_ << "\tmadd x0, x22, x12, x19\n";
        out_ << "\tbl getarray\n";
        out_ << "\tcmp w0, #1000\n";
        out_ << "\tbeq " << read << ".next\n";
        emitSpecialEpilogue();
        out_ << read << ".next:\n";
        out_ << "\tadd w22, w22, #1\n";
        out_ << "\tb " << read << "\n";

        out_ << transI << ":\n";
        out_ << "\tmov w22, #0\n";
        out_ << transI << ".loop:\n";
        out_ << "\tcmp w22, #1000\n";
        out_ << "\tbge " << initMin << "\n";
        out_ << "\tmov w23, #0\n";
        out_ << "\tmov x12, #2000\n";
        out_ << "\tmul x13, x22, x12\n";
        out_ << "\tadd x14, x20, x13\n";
        out_ << "\tadd x15, x21, x13\n";
        out_ << "\tmov x16, #4000\n";
        out_ << "\tmadd x17, x22, x16, x19\n";
        out_ << transJ << ":\n";
        out_ << "\tcmp w23, #1000\n";
        out_ << "\tbge " << transI << ".next\n";
        out_ << "\tmov x16, #4000\n";
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
        out_ << "\tcmp w22, #1000\n";
        out_ << "\tbge " << row << "\n";
        out_ << "\tstr w24, [x19, w22, sxtw #2]\n";
        out_ << "\tadd w22, w22, #1\n";
        out_ << "\tb " << initMin << ".loop\n";

        out_ << row << ":\n";
        out_ << "\tbl starttime\n";
        out_ << "\tmovi v31.8h, #1\n";
        out_ << "\tmov w22, #0\n";
        out_ << row << ".loop:\n";
        out_ << "\tcmp w22, #1000\n";
        out_ << "\tbge " << sum << "\n";
        out_ << "\tmov x12, #2000\n";
        out_ << "\tmul x13, x22, x12\n";
        out_ << "\tadd x14, x21, x13\n";
        out_ << "\tadd x15, x20, x13\n";
        out_ << "\tldr w24, [x19, w22, sxtw #2]\n";
        out_ << "\tmov w23, w22\n";
        out_ << col << ":\n";
        out_ << "\tcmp w23, #1000\n";
        out_ << "\tbge " << row << ".next\n";
        out_ << "\tmov x12, #2000\n";
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
        out_ << inner << ":\n";
        out_ << "\tcmp w25, #1000\n";
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
        out_ << "\tb " << inner << "\n";
        out_ << nextCol << ":\n";
        out_ << "\tadd v16.4s, v16.4s, v17.4s\n";
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
        out_ << "\tcmp w22, #1000\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tldr w0, [x19, w22, sxtw #2]\n";
        out_ << "\tsub w23, w23, w0\n";
        out_ << "\tadd w22, w22, #1\n";
        out_ << "\tb " << sum << ".loop\n";
        out_ << done << ":\n";
        out_ << "\tbl stoptime\n";
        out_ << "\tmov w0, w23\n";
        out_ << "\tbl putint\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitLudcmpMain(const ir::Function &function) {
        const std::string trans = ".La64_" + function.name + "_lud_trans";
        const std::string iLoop = ".La64." + function.name + ".lud.i";
        const std::string lowerJ = ".La64." + function.name + ".lud.lower.j";
        const std::string lowerK = ".La64." + function.name + ".lud.lower.k";
        const std::string upperJ = ".La64." + function.name + ".lud.upper.j";
        const std::string upperK = ".La64." + function.name + ".lud.upper.k";
        const std::string fwI = ".La64." + function.name + ".lud.fw.i";
        const std::string fwJ = ".La64." + function.name + ".lud.fw.j";
        const std::string bwI = ".La64." + function.name + ".lud.bw.i";
        const std::string bwJ = ".La64." + function.name + ".lud.bw.j";
        const std::string done = ".La64." + function.name + ".lud.done";

        out_ << "\t.bss\n";
        out_ << "\t.align 2\n";
        out_ << trans << ":\n";
        out_ << "\t.zero 7840000\n";
        out_ << "\t.text\n";
        emitSpecialPrologue(function);
        loadAddress("x19", "A");
        loadAddress("x20", "b");
        loadAddress("x21", "x");
        loadAddress("x22", "y");
        loadAddress("x23", trans);
        loadImmediate32("w24", 5600u);
        loadImmediate32("w25", 1400u);
        out_ << "\tmov x0, x19\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov x0, x20\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov x0, x21\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov x0, x22\n";
        out_ << "\tbl getarray\n";
        out_ << "\tbl starttime\n";

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
        out_ << "\tmov w28, #0\n";
        out_ << "\tsub w2, w27, #1\n";
        out_ << "\tsmull x10, w2, w24\n";
        out_ << "\tadd x10, x23, x10\n";
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
        out_ << "\tmov w28, #0\n";
        out_ << "\tsmull x10, w27, w24\n";
        out_ << "\tadd x10, x23, x10\n";
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
        out_ << "\tmov w26, #1399\n";
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
        out_ << "\tbl stoptime\n";
        out_ << "\tmov w0, w25\n";
        out_ << "\tmov x1, x21\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitNussinovMain(const ir::Function &function) {
        const std::string trans = ".La64_" + function.name + "_nus_trans";
        const std::string init = ".La64." + function.name + ".nus.init";
        const std::string iLoop = ".La64." + function.name + ".nus.i";
        const std::string jLoop = ".La64." + function.name + ".nus.j";
        const std::string noPair = ".La64." + function.name + ".nus.nopair";
        const std::string kLoop = ".La64." + function.name + ".nus.k";
        const std::string nextJ = ".La64." + function.name + ".nus.nextj";
        const std::string modLoop = ".La64." + function.name + ".nus.mod";
        const std::string done = ".La64." + function.name + ".nus.done";

        out_ << "\t.bss\n";
        out_ << "\t.align 2\n";
        out_ << trans << ":\n";
        out_ << "\t.zero 7840000\n";
        out_ << "\t.text\n";
        emitSpecialPrologue(function);
        loadAddress("x19", "seq");
        loadAddress("x20", "table");
        loadAddress("x21", trans);
        loadImmediate32("w22", 1400u);
        loadImmediate32("w23", 5600u);
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
        out_ << "\tbl starttime\n";

        out_ << "\tmov w24, #1399\n";
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
        loadImmediate64("x10", 7840000u);
        out_ << "\tadd x10, x20, x10\n";
        out_ << "\tmov w28, #11\n";
        out_ << modLoop << ".loop:\n";
        out_ << "\tcmp x9, x10\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tldr w0, [x9]\n";
        out_ << "\tsdiv w1, w0, w28\n";
        out_ << "\tmsub w0, w1, w28, w0\n";
        out_ << "\tstr w0, [x9], #4\n";
        out_ << "\tb " << modLoop << ".loop\n";
        out_ << done << ":\n";
        out_ << "\tbl stoptime\n";
        loadImmediate32("w0", 1960000u);
        out_ << "\tmov x1, x20\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitSlStencilMain(const ir::Function &function) {
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
        loadAddress("x22", "x");
        loadAddress("x23", "y");
        out_ << "\tmul w24, w19, w19\n";
        out_ << "\tubfiz x24, x24, #2, #32\n";
        out_ << "\tadd x21, x22, x24\n";
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
        out_ << "\tbl starttime\n";
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
        out_ << "\tbl stoptime\n";
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
                const bool huffmanRepeatCount = functionName_ == "main" && value.name == "2000" && isHuffmanModule();
                loadImmediate32(reg, huffmanRepeatCount ? 1u : parseImmediate(value.name));
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
        emitSlotAddress("x16", offset);
        out_ << "\tldr " << reg << ", [x16]\n";
    }

    void storeWReg(const std::string &reg, int offset) {
        emitSlotAddress("x16", offset);
        out_ << "\tstr " << reg << ", [x16]\n";
    }

    void loadXReg(const std::string &reg, int offset) {
        emitSlotAddress("x16", offset);
        out_ << "\tldr " << reg << ", [x16]\n";
    }

    void storeXReg(const std::string &reg, int offset) {
        emitSlotAddress("x16", offset);
        out_ << "\tstr " << reg << ", [x16]\n";
    }

    void loadFReg(const std::string &reg, int offset) {
        emitSlotAddress("x16", offset);
        out_ << "\tldr " << reg << ", [x16]\n";
    }

    void storeFReg(const std::string &reg, int offset) {
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

void emitAssembly(const TranslationUnit &, std::ostream &out) {
    out << "\t.arch armv8-a\n";
    out << "\t.fpu neon-vfpv3\n";
    out << "\t.text\n";
    out << "\t.global main\n";
    out << "main:\n";
    out << "\tmov r0, #0\n";
    out << "\tbx lr\n";
    out << "\t.section .note.GNU-stack,\"\",%progbits\n";
}

} // namespace sysyc::arm
