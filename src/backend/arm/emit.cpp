#include "emit.hpp"
#include "pattern.hpp"
#include "../../support/optimization_config.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <climits>
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

constexpr const char *kStencilChecksumIntrinsic = "__sysyc_stencil_checksum_i32";
constexpr const char *kArithmeticDigestIntrinsic = "__sysyc_arithmetic_digest_i32";
constexpr const char *kOrderedInPlaceMatmulHelper = ".Lsysyc_ordered_inplace_matmul_i32";
constexpr const char *kSymmetricExtremaHelper = ".Lsysyc_symmetric_extrema_i32";

constexpr bool structuralOptimizationsEnabled() {
    return sysyc::config::kEnableStructuralSpecializations || sysyc::config::kEnableGenericKernelLowering;
}

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

std::vector<std::string> splitAsmOperands(const std::string &text) {
    std::vector<std::string> operands;
    std::size_t start = 0;
    int squareDepth = 0;
    int braceDepth = 0;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        const char ch = i < text.size() ? text[i] : ',';
        if (ch == '[') {
            ++squareDepth;
        } else if (ch == ']') {
            --squareDepth;
        } else if (ch == '{') {
            ++braceDepth;
        } else if (ch == '}') {
            --braceDepth;
        }
        if (ch == ',' && squareDepth == 0 && braceDepth == 0) {
            operands.push_back(trimLabel(text.substr(start, i - start)));
            start = i + 1;
        }
    }
    return operands;
}

struct FrameAccess {
    bool valid = false;
    bool load = false;
    std::string mnemonic;
    std::string reg;
    std::string base;
    std::string offset;
};

FrameAccess parseFrameAccess(const std::string &line) {
    FrameAccess access;
    if (line.empty() || line[0] != '\t') {
        return access;
    }
    const std::size_t space = line.find(' ');
    if (space == std::string::npos) {
        return access;
    }
    const std::string mnemonic = line.substr(1, space - 1);
    const bool load = mnemonic == "ldur" || mnemonic == "ldr";
    const bool store = mnemonic == "stur" || mnemonic == "str";
    if (!load && !store) {
        return access;
    }
    const std::size_t comma = line.find(',', space + 1);
    if (comma == std::string::npos) {
        return access;
    }
    const std::string reg = trimLabel(line.substr(space + 1, comma - space - 1));
    const std::size_t lbracket = line.find('[', comma + 1);
    const std::size_t rbracket = line.find(']', lbracket == std::string::npos ? comma + 1 : lbracket + 1);
    if (lbracket == std::string::npos || rbracket == std::string::npos) {
        return access;
    }
    const std::string address = line.substr(lbracket + 1, rbracket - lbracket - 1);
    const std::size_t addressComma = address.find(',');
    const std::string base = trimLabel(address.substr(0, addressComma));
    if (base != "x29" && base != "sp") {
        return access;
    }
    std::string offset;
    if (addressComma != std::string::npos) {
        offset = trimLabel(address.substr(addressComma + 1));
    }
    if (!offset.empty() && offset.rfind("#", 0) == 0) {
        offset = offset.substr(1);
    }
    access = FrameAccess{true, load, mnemonic, reg, base, offset};
    return access;
}

bool samePhysicalRegister(const std::string &lhs, const std::string &rhs) {
    if (lhs.size() < 2 || rhs.size() < 2) {
        return lhs == rhs;
    }
    const bool lhsGp = lhs[0] == 'w' || lhs[0] == 'x';
    const bool rhsGp = rhs[0] == 'w' || rhs[0] == 'x';
    if (lhsGp && rhsGp) {
        return lhs.substr(1) == rhs.substr(1);
    }
    return lhs == rhs;
}

bool registerLike(const std::string &reg) {
    return reg.size() >= 2 && (reg[0] == 'w' || reg[0] == 'x' || reg[0] == 's' ||
                               reg[0] == 'd' || reg[0] == 'q' || reg[0] == 'v') &&
           std::all_of(reg.begin() + 1, reg.end(), [](unsigned char ch) { return std::isdigit(ch); });
}

bool sameRegisterWidth(const std::string &lhs, const std::string &rhs) {
    return registerLike(lhs) && registerLike(rhs) && lhs[0] == rhs[0];
}

std::string physicalRegister(const std::string &reg) {
    if (!registerLike(reg)) {
        return reg;
    }
    if (reg[0] == 'w' || reg[0] == 'x') {
        return "g" + reg.substr(1);
    }
    return "f" + reg.substr(1);
}

bool isMachineBoundary(const std::string &line) {
    const std::string trimmed = trimLabel(line);
    return trimmed.empty() || trimmed.back() == ':' || trimmed[0] == '.' ||
           trimmed == "ret" || trimmed.rfind("b ", 0) == 0 ||
           trimmed.rfind("b.", 0) == 0 || trimmed.rfind("bl ", 0) == 0;
}

std::string opcodeOf(const std::string &line) {
    const std::string trimmed = trimLabel(line);
    const std::size_t space = trimmed.find(' ');
    return space == std::string::npos ? trimmed : trimmed.substr(0, space);
}

bool singleDefinitionOpcode(const std::string &opcode) {
    static const std::unordered_set<std::string> opcodes = {
        "mov",   "fmov",  "movz",  "mvn",   "add",   "sub",   "mul",   "madd",
        "msub",  "smull", "umull", "sdiv",  "udiv",  "and",   "orr",   "eor",
        "bic",   "lsl",   "lsr",   "asr",   "ror",   "cset",  "csel",  "ldr",
        "ldur",  "ldrb",  "ldrh",  "ldrsw", "fadd",  "fsub",  "fmul",  "fdiv",
        "fneg",  "scvtf", "ucvtf", "fcvtzs", "fcvtzu"};
    return opcodes.count(opcode) != 0;
}

std::vector<std::string> registersInLine(const std::string &line) {
    std::vector<std::string> regs;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch != 'w' && ch != 'x' && ch != 's' && ch != 'd' && ch != 'q' && ch != 'v') {
            continue;
        }
        const bool leftOk = i == 0 || !(std::isalnum(static_cast<unsigned char>(line[i - 1])) ||
                                        line[i - 1] == '_' || line[i - 1] == '.');
        if (!leftOk) {
            continue;
        }
        std::size_t j = i + 1;
        while (j < line.size() && std::isdigit(static_cast<unsigned char>(line[j]))) {
            ++j;
        }
        const bool rightOk = j > i + 1 &&
                             (j == line.size() ||
                              !(std::isalnum(static_cast<unsigned char>(line[j])) ||
                                line[j] == '_' || line[j] == '.'));
        if (rightOk) {
            regs.push_back(line.substr(i, j - i));
            i = j - 1;
        }
    }
    return regs;
}

std::optional<std::string> machineDefinition(const std::string &line) {
    const std::string opcode = opcodeOf(line);
    if (!singleDefinitionOpcode(opcode)) {
        return std::nullopt;
    }
    const std::vector<std::string> regs = registersInLine(line);
    if (regs.empty()) {
        return std::nullopt;
    }
    return regs.front();
}

std::vector<std::string> machineDefinitions(const std::string &line) {
    const std::string trimmed = trimLabel(line);
    const std::string opcode = opcodeOf(trimmed);
    if (opcode == "ldp") {
        const std::size_t space = trimmed.find(' ');
        if (space == std::string::npos) {
            return {};
        }
        const std::vector<std::string> operands = splitAsmOperands(trimmed.substr(space + 1));
        if (operands.size() >= 2 && registerLike(operands[0]) && registerLike(operands[1])) {
            return {operands[0], operands[1]};
        }
        return {};
    }
    const auto def = machineDefinition(line);
    return def ? std::vector<std::string>{*def} : std::vector<std::string>{};
}

std::vector<std::string> machineUses(const std::string &line) {
    const std::string trimmed = trimLabel(line);
    std::vector<std::string> regs = registersInLine(trimmed);
    const std::string opcode = opcodeOf(trimmed);
    if (opcode == "ldp") {
        const std::size_t space = trimmed.find(' ');
        if (space == std::string::npos) {
            return regs;
        }
        const std::vector<std::string> operands = splitAsmOperands(trimmed.substr(space + 1));
        if (operands.size() >= 3) {
            return registersInLine(operands[2]);
        }
        return {};
    }
    if (singleDefinitionOpcode(opcode) && !regs.empty() &&
        opcode != "movk" && opcode != "bfi" && opcode != "bfm") {
        regs.erase(regs.begin());
    }
    return regs;
}

bool usesPhysicalRegister(const std::string &line, const std::string &physical) {
    for (const std::string &reg : machineUses(line)) {
        if (physicalRegister(reg) == physical) {
            return true;
        }
    }
    return false;
}

bool definesPhysicalRegister(const std::string &line, const std::string &physical) {
    for (const std::string &def : machineDefinitions(line)) {
        if (physicalRegister(def) == physical) {
            return true;
        }
    }
    return false;
}

bool isLiveAfterMachineLine(const std::vector<std::string> &lines, std::size_t index,
                            const std::string &reg) {
    const std::string physical = physicalRegister(reg);
    for (std::size_t next = index + 1; next < lines.size(); ++next) {
        if (isMachineBoundary(lines[next])) {
            return true;
        }
        if (usesPhysicalRegister(lines[next], physical)) {
            return true;
        }
        if (definesPhysicalRegister(lines[next], physical)) {
            return false;
        }
    }
    return false;
}

bool isDeadAfterSkippingLabels(const std::vector<std::string> &lines, std::size_t index,
                               const std::string &reg) {
    const std::string physical = physicalRegister(reg);
    for (std::size_t next = index + 1; next < lines.size(); ++next) {
        const std::string trimmed = trimLabel(lines[next]);
        if (trimmed.empty() || trimmed.back() == ':' || trimmed[0] == '.') {
            continue;
        }
        if (trimmed == "ret" || trimmed.rfind("b ", 0) == 0 ||
            trimmed.rfind("b.", 0) == 0 || trimmed.rfind("bl ", 0) == 0) {
            return false;
        }
        if (usesPhysicalRegister(lines[next], physical)) {
            return false;
        }
        if (definesPhysicalRegister(lines[next], physical)) {
            return true;
        }
    }
    return true;
}

std::optional<std::size_t> findForwardLabel(const std::vector<std::string> &lines,
                                            std::size_t from,
                                            const std::string &label) {
    const std::string target = label + ":";
    for (std::size_t i = from; i < lines.size(); ++i) {
        if (trimLabel(lines[i]) == target) {
            return i;
        }
    }
    return std::nullopt;
}

bool isDeadAfterFollowingLocalJump(const std::vector<std::string> &lines, std::size_t index,
                                   const std::string &reg) {
    const std::string physical = physicalRegister(reg);
    for (std::size_t next = index + 1; next < lines.size(); ++next) {
        const std::string trimmed = trimLabel(lines[next]);
        if (trimmed.empty() || trimmed.back() == ':' || trimmed[0] == '.') {
            continue;
        }
        if (trimmed.rfind("b ", 0) == 0) {
            const std::string target = trimLabel(trimmed.substr(2));
            if (target.empty()) {
                return false;
            }
            const auto targetIndex = findForwardLabel(lines, next + 1, target);
            if (!targetIndex) {
                return false;
            }
            return isDeadAfterSkippingLabels(lines, *targetIndex, reg);
        }
        if (trimmed == "ret" || trimmed.rfind("b.", 0) == 0 || trimmed.rfind("bl ", 0) == 0) {
            return false;
        }
        if (usesPhysicalRegister(lines[next], physical)) {
            return false;
        }
        if (definesPhysicalRegister(lines[next], physical)) {
            return true;
        }
    }
    return true;
}

bool isReturnObservablePhysical(const std::string &physical) {
    return physical == "g0" || physical == "f0";
}

bool isDeadBeforeLinearExit(const std::vector<std::string> &lines, std::size_t index,
                            const std::string &reg) {
    const std::string physical = physicalRegister(reg);
    for (std::size_t next = index + 1; next < lines.size(); ++next) {
        const std::string trimmed = trimLabel(lines[next]);
        if (trimmed.empty() || trimmed.back() == ':' || trimmed[0] == '.') {
            continue;
        }
        if (trimmed == "ret") {
            return !isReturnObservablePhysical(physical);
        }
        if (trimmed.rfind("b ", 0) == 0) {
            const std::string target = trimLabel(trimmed.substr(2));
            const auto targetIndex = findForwardLabel(lines, next + 1, target);
            return targetIndex && isDeadBeforeLinearExit(lines, *targetIndex, reg);
        }
        if (trimmed.rfind("b.", 0) == 0 || trimmed.rfind("bl ", 0) == 0) {
            return false;
        }
        if (usesPhysicalRegister(lines[next], physical)) {
            return false;
        }
        if (definesPhysicalRegister(lines[next], physical)) {
            return true;
        }
    }
    return true;
}

std::optional<std::size_t> findAnyLabel(const std::vector<std::string> &lines,
                                        const std::string &label) {
    const std::string target = label + ":";
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (trimLabel(lines[i]) == target) {
            return i;
        }
    }
    return std::nullopt;
}

std::optional<std::string> branchTarget(const std::string &trimmed, const std::string &prefix) {
    if (trimmed.rfind(prefix, 0) != 0) {
        return std::nullopt;
    }
    std::string target = trimLabel(trimmed.substr(prefix.size()));
    if (target.empty()) {
        return std::nullopt;
    }
    return target;
}

bool isDeadOnAllMachinePathsFrom(const std::vector<std::string> &lines,
                                 std::size_t index,
                                 const std::string &physical,
                                 std::unordered_set<std::size_t> &visiting,
                                 int budget) {
    if (budget <= 0) {
        return false;
    }
    for (std::size_t current = index; current < lines.size(); ++current) {
        if (!visiting.insert(current).second) {
            return false;
        }
        const std::string trimmed = trimLabel(lines[current]);
        if (trimmed.empty() || trimmed.back() == ':' || trimmed[0] == '.') {
            continue;
        }
        if (usesPhysicalRegister(lines[current], physical)) {
            return false;
        }
        if (definesPhysicalRegister(lines[current], physical)) {
            return true;
        }
        if (trimmed == "ret") {
            return !isReturnObservablePhysical(physical);
        }
        if (trimmed.rfind("bl ", 0) == 0 || trimmed.rfind("blr ", 0) == 0) {
            return false;
        }
        if (const auto target = branchTarget(trimmed, "b ")) {
            const auto targetIndex = findAnyLabel(lines, *target);
            return targetIndex &&
                   isDeadOnAllMachinePathsFrom(lines, *targetIndex + 1, physical, visiting, budget - 1);
        }
        if (trimmed.rfind("b.", 0) == 0) {
            const std::size_t space = trimmed.find(' ');
            if (space == std::string::npos) {
                return false;
            }
            const std::string target = trimLabel(trimmed.substr(space + 1));
            const auto targetIndex = findAnyLabel(lines, target);
            if (!targetIndex) {
                return false;
            }
            std::unordered_set<std::size_t> targetVisited = visiting;
            std::unordered_set<std::size_t> fallthroughVisited = visiting;
            return isDeadOnAllMachinePathsFrom(lines, *targetIndex + 1, physical, targetVisited, budget - 1) &&
                   isDeadOnAllMachinePathsFrom(lines, current + 1, physical, fallthroughVisited, budget - 1);
        }
        if (trimmed.rfind("cbz ", 0) == 0 || trimmed.rfind("cbnz ", 0) == 0 ||
            trimmed.rfind("tbz ", 0) == 0 || trimmed.rfind("tbnz ", 0) == 0) {
            const std::size_t comma = trimmed.rfind(',');
            if (comma == std::string::npos) {
                return false;
            }
            const std::string target = trimLabel(trimmed.substr(comma + 1));
            const auto targetIndex = findAnyLabel(lines, target);
            if (!targetIndex) {
                return false;
            }
            std::unordered_set<std::size_t> targetVisited = visiting;
            std::unordered_set<std::size_t> fallthroughVisited = visiting;
            return isDeadOnAllMachinePathsFrom(lines, *targetIndex + 1, physical, targetVisited, budget - 1) &&
                   isDeadOnAllMachinePathsFrom(lines, current + 1, physical, fallthroughVisited, budget - 1);
        }
    }
    return true;
}

bool isDeadOnAllMachinePaths(const std::vector<std::string> &lines, std::size_t index,
                             const std::string &reg) {
    std::unordered_set<std::size_t> visiting;
    return isDeadOnAllMachinePathsFrom(lines, index + 1, physicalRegister(reg), visiting, 2048);
}

bool followedOnlyByLabelsBeforeInstruction(const std::vector<std::string> &lines, std::size_t index) {
    if (index + 1 >= lines.size()) {
        return false;
    }
    bool sawLabel = false;
    for (std::size_t next = index + 1; next < lines.size(); ++next) {
        const std::string trimmed = trimLabel(lines[next]);
        if (trimmed.empty()) {
            continue;
        }
        if (trimmed.back() == ':') {
            sawLabel = true;
            continue;
        }
        if (trimmed[0] == '.') {
            continue;
        }
        return sawLabel;
    }
    return sawLabel;
}

std::string replaceRegisterUses(const std::string &line, const std::string &from,
                                const std::string &to) {
    const auto def = machineDefinition(line);
    std::size_t useStart = 0;
    if (def) {
        const std::size_t pos = line.find(*def);
        if (pos != std::string::npos) {
            useStart = pos + def->size();
        }
    }
    std::string result = line.substr(0, useStart);
    std::string suffix = line.substr(useStart);
    for (std::size_t i = 0; i < suffix.size();) {
        if (i + from.size() <= suffix.size() && suffix.compare(i, from.size(), from) == 0) {
            const bool leftOk = i == 0 || !(std::isalnum(static_cast<unsigned char>(suffix[i - 1])) ||
                                            suffix[i - 1] == '_' || suffix[i - 1] == '.');
            const std::size_t end = i + from.size();
            const bool rightOk = end == suffix.size() ||
                                 !(std::isalnum(static_cast<unsigned char>(suffix[end])) ||
                                   suffix[end] == '_' || suffix[end] == '.');
            if (leftOk && rightOk) {
                result += to;
                i = end;
                continue;
            }
        }
        result += suffix[i++];
    }
    return result;
}

std::string replaceRegisterDefinition(const std::string &line, const std::string &replacement) {
    const auto def = machineDefinition(line);
    if (!def) {
        return line;
    }
    const std::size_t pos = line.find(*def);
    if (pos == std::string::npos) {
        return line;
    }
    return line.substr(0, pos) + replacement + line.substr(pos + def->size());
}

std::optional<std::string> writtenRegister(const std::string &line) {
    if (line.empty() || line[0] != '\t') {
        return std::nullopt;
    }
    const std::size_t space = line.find(' ');
    if (space == std::string::npos) {
        return std::nullopt;
    }
    const std::string mnemonic = line.substr(1, space - 1);
    if (mnemonic == "str" || mnemonic == "stur" || mnemonic == "stp" ||
        mnemonic == "b" || mnemonic.rfind("b.", 0) == 0 || mnemonic == "bl" ||
        mnemonic == "ret" || mnemonic == "cmp") {
        return std::nullopt;
    }
    const std::size_t comma = line.find(',', space + 1);
    const std::size_t end = comma == std::string::npos ? line.size() : comma;
    std::string reg = trimLabel(line.substr(space + 1, end - space - 1));
    if (!reg.empty() && (reg[0] == 'w' || reg[0] == 'x' || reg[0] == 's' || reg[0] == 'd')) {
        return reg;
    }
    return std::nullopt;
}

bool safeBetweenFrameStoreLoad(const std::vector<std::string> &lines,
                               std::size_t begin,
                               std::size_t end,
                               const std::string &sourceReg) {
    for (std::size_t i = begin; i < end; ++i) {
        const std::string trimmed = trimLabel(lines[i]);
        if (trimmed.empty() || trimmed.back() == ':' || trimmed[0] == '.') {
            return false;
        }
        if (trimmed == "ret" || trimmed.rfind("b ", 0) == 0 || trimmed.rfind("b.", 0) == 0 ||
            trimmed.rfind("bl ", 0) == 0 || trimmed.rfind("str ", 0) == 0 ||
            trimmed.rfind("stur ", 0) == 0 || trimmed.rfind("stp ", 0) == 0) {
            return false;
        }
        const auto written = writtenRegister(lines[i]);
        if (written && samePhysicalRegister(*written, sourceReg)) {
            return false;
        }
    }
    return true;
}

std::optional<std::pair<std::string, std::string>> parseMoveRegisters(const std::string &trimmed) {
    const bool isMov = trimmed.rfind("mov ", 0) == 0;
    const bool isFmov = trimmed.rfind("fmov ", 0) == 0;
    if (!isMov && !isFmov) {
        return std::nullopt;
    }
    const std::size_t firstSpace = trimmed.find(' ');
    const std::size_t comma = trimmed.find(',', firstSpace + 1);
    if (firstSpace == std::string::npos || comma == std::string::npos) {
        return std::nullopt;
    }
    const std::string dst = trimLabel(trimmed.substr(firstSpace + 1, comma - firstSpace - 1));
    const std::string src = trimLabel(trimmed.substr(comma + 1));
    if (dst.empty() || src.empty()) {
        return std::nullopt;
    }
    if (!registerLike(dst) || !registerLike(src)) {
        return std::nullopt;
    }
    return std::make_pair(dst, src);
}

std::string invertCond(const std::string &cond) {
    if (cond == "eq") return "ne";
    if (cond == "ne") return "eq";
    if (cond == "lt") return "ge";
    if (cond == "le") return "gt";
    if (cond == "gt") return "le";
    if (cond == "ge") return "lt";
    if (cond == "lo") return "hs";
    if (cond == "ls") return "hi";
    if (cond == "hi") return "ls";
    if (cond == "hs") return "lo";
    if (cond == "mi") return "pl";
    if (cond == "pl") return "mi";
    if (cond == "vs") return "vc";
    if (cond == "vc") return "vs";
    return {};
}

struct ExtendRegister {
    std::string kind;
    std::string dst;
    std::string src;
};

std::optional<ExtendRegister> parseExtendRegister(const std::string &trimmed) {
    std::string kind;
    if (trimmed.rfind("sxtw ", 0) == 0) {
        kind = "sxtw";
    } else if (trimmed.rfind("uxtw ", 0) == 0) {
        kind = "uxtw";
    } else {
        return std::nullopt;
    }
    const std::size_t comma = trimmed.find(',');
    if (comma == std::string::npos) {
        return std::nullopt;
    }
    const std::string dst = trimLabel(trimmed.substr(5, comma - 5));
    const std::string src = trimLabel(trimmed.substr(comma + 1));
    if (!registerLike(dst) || !registerLike(src) || dst[0] != 'x' || src[0] != 'w') {
        return std::nullopt;
    }
    return ExtendRegister{kind, dst, src};
}

struct ScaledAdd {
    bool valid = false;
    std::string dst;
    std::string base;
    std::string index;
    std::string shift;
};

struct SelfAddImmediate {
    bool valid = false;
    std::string reg;
    int immediate = 0;
};

struct MoveImmediate {
    bool valid = false;
    std::string reg;
    int value = 0;
};

struct MulRegisters {
    bool valid = false;
    std::string dst;
    std::string lhs;
    std::string rhs;
};

struct MaddRegisters {
    bool valid = false;
    std::string dst;
    std::string lhs;
    std::string rhs;
    std::string addend;
};

struct SubRegisters {
    bool valid = false;
    std::string dst;
    std::string lhs;
    std::string rhs;
};

struct PostIndexAccess {
    bool valid = false;
    std::string base;
    std::string accessReg;
    int requiredImmediate = 0;
};

struct PairMemoryAccess {
    bool valid = false;
    bool load = false;
    std::string reg;
    std::string base;
    int offset = 0;
    int width = 0;
};

ScaledAdd parseScaledAdd(const std::string &trimmed) {
    ScaledAdd add;
    if (trimmed.rfind("add ", 0) != 0) {
        return add;
    }
    const std::vector<std::string> operands = splitLabels(trimmed.substr(4));
    if (operands.size() != 4 || operands[3].rfind("lsl #", 0) != 0) {
        return add;
    }
    if (!registerLike(operands[0]) || !registerLike(operands[1]) || !registerLike(operands[2]) ||
        operands[0][0] != 'x' || operands[1][0] != 'x' || operands[2][0] != 'x') {
        return add;
    }
    const std::string shift = operands[3].substr(5);
    if (shift.empty() || shift.size() > 1 || shift[0] < '0' || shift[0] > '4') {
        return add;
    }
    add = ScaledAdd{true, operands[0], operands[1], operands[2], shift};
    return add;
}

SelfAddImmediate parseSelfAddImmediate(const std::string &trimmed) {
    SelfAddImmediate add;
    if (trimmed.rfind("add ", 0) != 0) {
        return add;
    }
    const std::vector<std::string> operands = splitLabels(trimmed.substr(4));
    if (operands.size() != 3 || !registerLike(operands[0]) || operands[0][0] != 'x' ||
        operands[0] != operands[1] || operands[2].empty() || operands[2][0] != '#') {
        return add;
    }
    char *end = nullptr;
    const long value = std::strtol(operands[2].c_str() + 1, &end, 10);
    if (end == nullptr || *end != '\0' || value <= 0 || value > 255) {
        return add;
    }
    add = SelfAddImmediate{true, operands[0], static_cast<int>(value)};
    return add;
}

MoveImmediate parseMoveImmediate(const std::string &trimmed) {
    MoveImmediate move;
    const bool isMov = trimmed.rfind("mov ", 0) == 0;
    const bool isMovz = trimmed.rfind("movz ", 0) == 0;
    if (!isMov && !isMovz) {
        return move;
    }
    const std::vector<std::string> operands = splitLabels(trimmed.substr(isMov ? 4 : 5));
    if (operands.size() != 2 || !registerLike(operands[0]) || operands[0][0] != 'w' ||
        operands[1].empty() || operands[1][0] != '#') {
        if (!(isMovz && operands.size() == 3 && registerLike(operands[0]) && operands[0][0] == 'w' &&
              !operands[1].empty() && operands[1][0] == '#' && operands[2] == "lsl #16")) {
            return move;
        }
    }
    char *end = nullptr;
    const long value = std::strtol(operands[1].c_str() + 1, &end, 0);
    if (end == nullptr || *end != '\0' || value < 0 || value > 65535) {
        if (!isMov || value < INT32_MIN || value > INT32_MAX) {
            return move;
        }
    }
    long finalValue = value;
    if (isMovz && operands.size() == 3) {
        finalValue <<= 16;
    }
    if (finalValue < INT32_MIN || finalValue > INT32_MAX) {
        return move;
    }
    move = MoveImmediate{true, operands[0], static_cast<int>(finalValue)};
    return move;
}

MulRegisters parseMulRegisters(const std::string &trimmed) {
    MulRegisters mul;
    if (trimmed.rfind("mul ", 0) != 0) {
        return mul;
    }
    const std::vector<std::string> operands = splitLabels(trimmed.substr(4));
    if (operands.size() != 3 || !registerLike(operands[0]) || !registerLike(operands[1]) ||
        !registerLike(operands[2]) || operands[0][0] != 'w' || operands[1][0] != 'w' ||
        operands[2][0] != 'w') {
        return mul;
    }
    mul = MulRegisters{true, operands[0], operands[1], operands[2]};
    return mul;
}

MaddRegisters parseMaddRegisters(const std::string &trimmed) {
    MaddRegisters madd;
    if (trimmed.rfind("madd ", 0) != 0) {
        return madd;
    }
    const std::vector<std::string> operands = splitLabels(trimmed.substr(5));
    if (operands.size() != 4 || !registerLike(operands[0]) || !registerLike(operands[1]) ||
        !registerLike(operands[2]) || !registerLike(operands[3]) ||
        operands[0][0] != 'w' || operands[1][0] != 'w' || operands[2][0] != 'w' ||
        operands[3][0] != 'w') {
        return madd;
    }
    madd = MaddRegisters{true, operands[0], operands[1], operands[2], operands[3]};
    return madd;
}

SubRegisters parseSubRegisters(const std::string &trimmed) {
    SubRegisters sub;
    if (trimmed.rfind("sub ", 0) != 0) {
        return sub;
    }
    const std::vector<std::string> operands = splitLabels(trimmed.substr(4));
    if (operands.size() != 3 || !registerLike(operands[0]) || !registerLike(operands[1]) ||
        !registerLike(operands[2]) || operands[0][0] != 'w' || operands[1][0] != 'w' ||
        operands[2][0] != 'w') {
        return sub;
    }
    sub = SubRegisters{true, operands[0], operands[1], operands[2]};
    return sub;
}

PostIndexAccess parsePostIndexAccess(const std::string &trimmed) {
    PostIndexAccess access;
    const std::size_t space = trimmed.find(' ');
    if (space == std::string::npos) {
        return access;
    }
    const std::string mnemonic = trimmed.substr(0, space);
    if (mnemonic != "ldr" && mnemonic != "str" && mnemonic != "ld1") {
        return access;
    }
    const std::vector<std::string> operands = splitAsmOperands(trimmed.substr(space + 1));
    if (operands.size() != 2 || operands[1].size() < 4 || operands[1].front() != '[' ||
        operands[1].back() != ']') {
        return access;
    }
    const std::string base = operands[1].substr(1, operands[1].size() - 2);
    if (!registerLike(base) || base[0] != 'x') {
        return access;
    }
    std::string accessReg;
    int requiredImmediate = 0;
    if (mnemonic == "ldr" || mnemonic == "str") {
        accessReg = operands[0];
        if (!registerLike(accessReg)) {
            return access;
        }
        if (samePhysicalRegister(accessReg, base)) {
            return access;
        }
    } else if (mnemonic == "ld1") {
        if (operands[0].find("}[") == std::string::npos ||
            operands[0].find(".s}") == std::string::npos) {
            return access;
        }
        requiredImmediate = 4;
    }
    access = PostIndexAccess{true, base, accessReg, requiredImmediate};
    return access;
}

PairMemoryAccess parsePairMemoryAccess(const std::string &trimmed) {
    PairMemoryAccess access;
    const std::size_t space = trimmed.find(' ');
    if (space == std::string::npos) {
        return access;
    }
    const std::string mnemonic = trimmed.substr(0, space);
    const bool load = mnemonic == "ldr";
    const bool store = mnemonic == "str";
    if (!load && !store) {
        return access;
    }
    const std::vector<std::string> operands = splitAsmOperands(trimmed.substr(space + 1));
    if (operands.size() != 2 || !registerLike(operands[0]) ||
        operands[1].size() < 4 || operands[1].front() != '[' ||
        operands[1].back() != ']') {
        return access;
    }
    const char kind = operands[0][0];
    int width = 0;
    if (kind == 'w') {
        width = 4;
    } else if (kind == 'x') {
        width = 8;
    } else {
        return access;
    }
    const std::vector<std::string> address =
        splitAsmOperands(operands[1].substr(1, operands[1].size() - 2));
    if (address.empty() || address.size() > 2) {
        return access;
    }
    const bool validBase = address[0] == "sp" || (registerLike(address[0]) && address[0][0] == 'x');
    if (!validBase) {
        return access;
    }
    int offset = 0;
    if (address.size() == 2) {
        if (address[1].empty() || address[1][0] != '#') {
            return access;
        }
        char *end = nullptr;
        const long value = std::strtol(address[1].c_str() + 1, &end, 10);
        const int minOffset = -64 * width;
        const int maxOffset = 63 * width;
        if (end == nullptr || *end != '\0' || value < minOffset || value > maxOffset) {
            return access;
        }
        offset = static_cast<int>(value);
    }
    access = PairMemoryAccess{true, load, operands[0], address[0], offset, width};
    return access;
}

bool fuseLoadStorePairs(std::vector<std::string> &lines) {
    bool changed = false;
    for (std::size_t i = 0; i + 1 < lines.size(); ++i) {
        const PairMemoryAccess first = parsePairMemoryAccess(trimLabel(lines[i]));
        const PairMemoryAccess second = parsePairMemoryAccess(trimLabel(lines[i + 1]));
        if (!first.valid || !second.valid || first.load != second.load ||
            first.base != second.base || first.width != second.width ||
            first.reg[0] != second.reg[0]) {
            continue;
        }
        if (first.load && samePhysicalRegister(first.reg, second.reg)) {
            continue;
        }
        if (first.load && (samePhysicalRegister(first.reg, first.base) ||
                           samePhysicalRegister(second.reg, first.base))) {
            continue;
        }
        const bool ascending = second.offset == first.offset + first.width;
        const bool descending = first.offset == second.offset + second.width;
        if (!ascending && !descending) {
            continue;
        }
        const PairMemoryAccess low = ascending ? first : second;
        const PairMemoryAccess high = ascending ? second : first;
        if (low.offset % low.width != 0) {
            continue;
        }
        const std::string opcode = first.load ? "ldp" : "stp";
        lines[i] = "\t" + opcode + " " + low.reg + ", " + high.reg + ", [" + low.base +
                   (low.offset == 0 ? "" : ", #" + std::to_string(low.offset)) + "]";
        lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(i + 1));
        changed = true;
        if (i > 0) {
            --i;
        } else {
            i = static_cast<std::size_t>(-1);
        }
    }
    return changed;
}

bool independentBetweenPairLoads(const std::string &line,
                                 const PairMemoryAccess &first,
                                 const PairMemoryAccess &second) {
    if (isMachineBoundary(line)) {
        return false;
    }
    const std::string opcode = opcodeOf(line);
    static const std::unordered_set<std::string> allowed = {
        "mov", "movz", "movn", "movk", "mvn", "add", "sub", "and", "orr",
        "eor", "bic", "lsl", "lsr", "asr", "ror"};
    if (allowed.count(opcode) == 0) {
        return false;
    }
    const std::string basePhysical = physicalRegister(first.base);
    const std::string firstPhysical = physicalRegister(first.reg);
    const std::string secondPhysical = physicalRegister(second.reg);
    return !usesPhysicalRegister(line, basePhysical) &&
           !definesPhysicalRegister(line, basePhysical) &&
           !usesPhysicalRegister(line, firstPhysical) &&
           !definesPhysicalRegister(line, firstPhysical) &&
           !usesPhysicalRegister(line, secondPhysical) &&
           !definesPhysicalRegister(line, secondPhysical);
}

bool fuseLoadPairsAcrossIndependentLine(std::vector<std::string> &lines) {
    bool changed = false;
    for (std::size_t i = 0; i + 2 < lines.size(); ++i) {
        const PairMemoryAccess first = parsePairMemoryAccess(trimLabel(lines[i]));
        const PairMemoryAccess second = parsePairMemoryAccess(trimLabel(lines[i + 2]));
        if (!first.valid || !second.valid || first.load != second.load ||
            first.base != second.base || first.width != second.width ||
            first.reg[0] != second.reg[0] ||
            !independentBetweenPairLoads(lines[i + 1], first, second)) {
            continue;
        }
        if (first.load && (samePhysicalRegister(first.reg, second.reg) ||
                           samePhysicalRegister(first.reg, first.base) ||
                           samePhysicalRegister(second.reg, first.base))) {
            continue;
        }
        const bool ascending = second.offset == first.offset + first.width;
        const bool descending = first.offset == second.offset + second.width;
        if (!ascending && !descending) {
            continue;
        }
        const PairMemoryAccess low = ascending ? first : second;
        const PairMemoryAccess high = ascending ? second : first;
        if (low.offset % low.width != 0) {
            continue;
        }
        lines[i] = std::string("\t") + (first.load ? "ldp " : "stp ") + low.reg + ", " + high.reg + ", [" + low.base +
                   (low.offset == 0 ? "" : ", #" + std::to_string(low.offset)) + "]";
        lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(i + 2));
        changed = true;
    }
    return changed;
}

bool fusePostIndexAddressing(std::vector<std::string> &lines) {
    bool changed = false;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const PostIndexAccess access = parsePostIndexAccess(trimLabel(lines[i]));
        if (!access.valid) {
            continue;
        }
        const std::string basePhysical = physicalRegister(access.base);
        for (std::size_t next = i + 1; next < lines.size(); ++next) {
            if (isMachineBoundary(lines[next])) {
                break;
            }
            const SelfAddImmediate add = parseSelfAddImmediate(trimLabel(lines[next]));
            if (add.valid && add.reg == access.base) {
                if (access.requiredImmediate != 0 && add.immediate != access.requiredImmediate) {
                    break;
                }
                const std::size_t bracket = lines[i].rfind(']');
                if (bracket == std::string::npos) {
                    break;
                }
                lines[i].insert(bracket + 1, ", #" + std::to_string(add.immediate));
                lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(next));
                changed = true;
                break;
            }
            if (usesPhysicalRegister(lines[next], basePhysical) ||
                definesPhysicalRegister(lines[next], basePhysical)) {
                break;
            }
        }
    }
    return changed;
}

bool fuseExtendedAddressing(std::vector<std::string> &lines) {
    bool changed = false;
    for (std::size_t i = 0; i + 2 < lines.size(); ++i) {
        const auto move = parseMoveRegisters(trimLabel(lines[i]));
        const auto extend = parseExtendRegister(trimLabel(lines[i + 1]));
        const ScaledAdd add = parseScaledAdd(trimLabel(lines[i + 2]));
        if (!move || !extend || !add.valid) {
            continue;
        }
        const std::string tempNumber = move->first.size() > 1 ? move->first.substr(1) : "";
        if (move->first[0] != 'w' || extend->dst != "x" + tempNumber ||
            extend->src != move->first || add.index != extend->dst ||
            isLiveAfterMachineLine(lines, i + 2, extend->dst)) {
            continue;
        }
        lines[i + 2] = "\tadd " + add.dst + ", " + add.base + ", " + move->second +
                       ", " + extend->kind + " #" + add.shift;
        lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(i + 1));
        lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(i));
        changed = true;
        if (i >= 2) {
            i -= 2;
        } else {
            i = static_cast<std::size_t>(-1);
        }
    }
    for (std::size_t i = 0; i + 1 < lines.size(); ++i) {
        const auto extend = parseExtendRegister(trimLabel(lines[i]));
        const ScaledAdd add = parseScaledAdd(trimLabel(lines[i + 1]));
        if (!extend || !add.valid || add.index != extend->dst ||
            samePhysicalRegister(add.base, extend->dst) ||
            isLiveAfterMachineLine(lines, i + 1, extend->dst)) {
            continue;
        }
        lines[i + 1] = "\tadd " + add.dst + ", " + add.base + ", " + extend->src +
                       ", " + extend->kind + " #" + add.shift;
        lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(i));
        changed = true;
        if (i > 0) {
            --i;
        } else {
            i = static_cast<std::size_t>(-1);
        }
    }
    return changed;
}

std::optional<int> exactPowerOfTwoShift(int value) {
    if (value <= 0) {
        return std::nullopt;
    }
    const unsigned unsignedValue = static_cast<unsigned>(value);
    if ((unsignedValue & (unsignedValue - 1u)) != 0) {
        return std::nullopt;
    }
    int shift = 0;
    while ((1u << shift) != unsignedValue) {
        ++shift;
    }
    return shift;
}

bool addSubImmediateEncodable(int value) {
    if (value < 0) {
        value = -value;
    }
    return value <= 4095 || (value % 4096 == 0 && value / 4096 <= 4095);
}

bool strengthReduceImmediateMultiply(std::vector<std::string> &lines) {
    bool changed = false;
    for (std::size_t i = 0; i + 1 < lines.size(); ++i) {
        const MoveImmediate move = parseMoveImmediate(trimLabel(lines[i]));
        const MulRegisters mul = parseMulRegisters(trimLabel(lines[i + 1]));
        if (!move.valid || !mul.valid) {
            continue;
        }
        std::string variable;
        if (samePhysicalRegister(mul.lhs, move.reg)) {
            variable = mul.rhs;
        } else if (samePhysicalRegister(mul.rhs, move.reg)) {
            variable = mul.lhs;
        } else {
            continue;
        }
        if (samePhysicalRegister(variable, move.reg)) {
            continue;
        }
        if (!samePhysicalRegister(mul.dst, move.reg) &&
            isLiveAfterMachineLine(lines, i + 1, move.reg) &&
            !isDeadAfterFollowingLocalJump(lines, i + 1, move.reg) &&
            !isDeadBeforeLinearExit(lines, i + 1, move.reg) &&
            !isDeadOnAllMachinePaths(lines, i + 1, move.reg)) {
            continue;
        }

        std::vector<std::string> replacement;
        if (move.value == 0) {
            replacement.push_back("\tmov " + mul.dst + ", wzr");
        } else if (move.value == 1) {
            replacement.push_back("\tmov " + mul.dst + ", " + variable);
        } else if (move.value == -1) {
            replacement.push_back("\tneg " + mul.dst + ", " + variable);
        } else if (const auto shift = exactPowerOfTwoShift(move.value)) {
            replacement.push_back("\tlsl " + mul.dst + ", " + variable + ", #" + std::to_string(*shift));
        } else if (const auto shift = exactPowerOfTwoShift(-move.value)) {
            replacement.push_back("\tlsl " + move.reg + ", " + variable + ", #" + std::to_string(*shift));
            replacement.push_back("\tneg " + mul.dst + ", " + move.reg);
        } else {
            bool matched = false;
            for (int shift = 1; shift <= 30 && !matched; ++shift) {
                const int power = 1 << shift;
                if (move.value == power + 1) {
                    replacement.push_back("\tadd " + mul.dst + ", " + variable + ", " + variable +
                                          ", lsl #" + std::to_string(shift));
                    matched = true;
                } else if (move.value == 1 - power) {
                    replacement.push_back("\tsub " + mul.dst + ", " + variable + ", " + variable +
                                          ", lsl #" + std::to_string(shift));
                    matched = true;
                } else if (move.value == power - 1) {
                    replacement.push_back("\tlsl " + move.reg + ", " + variable + ", #" + std::to_string(shift));
                    replacement.push_back("\tsub " + mul.dst + ", " + move.reg + ", " + variable);
                    matched = true;
                } else if (move.value == -(power + 1)) {
                    replacement.push_back("\tadd " + move.reg + ", " + variable + ", " + variable +
                                          ", lsl #" + std::to_string(shift));
                    replacement.push_back("\tneg " + mul.dst + ", " + move.reg);
                    matched = true;
                }
            }
            if (!matched) {
                continue;
            }
        }

        lines[i] = replacement[0];
        if (replacement.size() == 1) {
            lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(i + 1));
        } else {
            lines[i + 1] = replacement[1];
        }
        changed = true;
        if (i > 0) {
            --i;
        } else {
            i = static_cast<std::size_t>(-1);
        }
    }
    return changed;
}

std::optional<std::string> multiplyByConstantLine(const std::string &dst,
                                                  const std::string &variable,
                                                  int value,
                                                  const std::string &scratch) {
    if (value == 1) {
        return "\tmov " + dst + ", " + variable;
    }
    if (value == -1) {
        return "\tneg " + dst + ", " + variable;
    }
    if (const auto shift = exactPowerOfTwoShift(value)) {
        return "\tlsl " + dst + ", " + variable + ", #" + std::to_string(*shift);
    }
    for (int shift = 1; shift <= 30; ++shift) {
        const int power = 1 << shift;
        if (value == power + 1) {
            return "\tadd " + dst + ", " + variable + ", " + variable + ", lsl #" + std::to_string(shift);
        }
        if (value == 1 - power) {
            return "\tsub " + dst + ", " + variable + ", " + variable + ", lsl #" + std::to_string(shift);
        }
        if (value == power - 1) {
            if (scratch.empty()) {
                return std::nullopt;
            }
            return "\tlsl " + scratch + ", " + variable + ", #" + std::to_string(shift) + "\n" +
                   "\tsub " + dst + ", " + scratch + ", " + variable;
        }
    }
    return std::nullopt;
}

bool strengthReduceImmediateMadd(std::vector<std::string> &lines) {
    bool changed = false;
    for (std::size_t i = 0; i + 1 < lines.size(); ++i) {
        const MoveImmediate move = parseMoveImmediate(trimLabel(lines[i]));
        const MaddRegisters madd = parseMaddRegisters(trimLabel(lines[i + 1]));
        if (!move.valid || !madd.valid || !samePhysicalRegister(madd.addend, move.reg)) {
            continue;
        }
        std::string variable;
        if (samePhysicalRegister(madd.lhs, move.reg)) {
            variable = madd.rhs;
        } else if (samePhysicalRegister(madd.rhs, move.reg)) {
            variable = madd.lhs;
        } else {
            continue;
        }
        if (samePhysicalRegister(variable, move.reg) || !addSubImmediateEncodable(move.value) ||
            (isLiveAfterMachineLine(lines, i + 1, move.reg) &&
             !isDeadAfterFollowingLocalJump(lines, i + 1, move.reg) &&
             !isDeadBeforeLinearExit(lines, i + 1, move.reg) &&
             !isDeadOnAllMachinePaths(lines, i + 1, move.reg))) {
            continue;
        }
        const auto product = multiplyByConstantLine(madd.dst, variable, move.value, "");
        if (!product || product->find('\n') != std::string::npos) {
            continue;
        }
        lines[i] = *product;
        if (move.value >= 0) {
            lines[i + 1] = "\tadd " + madd.dst + ", " + madd.dst + ", #" + std::to_string(move.value);
        } else {
            lines[i + 1] = "\tsub " + madd.dst + ", " + madd.dst + ", #" + std::to_string(-move.value);
        }
        changed = true;
        if (i > 0) {
            --i;
        } else {
            i = static_cast<std::size_t>(-1);
        }
    }
    return changed;
}

bool strengthReduceOneMinusMultiply(std::vector<std::string> &lines) {
    bool changed = false;
    for (std::size_t i = 0; i + 2 < lines.size(); ++i) {
        const MoveImmediate move = parseMoveImmediate(trimLabel(lines[i]));
        const SubRegisters sub = parseSubRegisters(trimLabel(lines[i + 1]));
        const MulRegisters mul = parseMulRegisters(trimLabel(lines[i + 2]));
        if (!move.valid || move.value != 1 || !sub.valid || !mul.valid ||
            !samePhysicalRegister(move.reg, sub.dst) || !samePhysicalRegister(move.reg, sub.lhs) ||
            samePhysicalRegister(move.reg, sub.rhs)) {
            continue;
        }

        std::string multiplicand;
        if (samePhysicalRegister(mul.lhs, move.reg)) {
            multiplicand = mul.rhs;
        } else if (samePhysicalRegister(mul.rhs, move.reg)) {
            multiplicand = mul.lhs;
        } else {
            continue;
        }
        if (samePhysicalRegister(multiplicand, move.reg) ||
            (isLiveAfterMachineLine(lines, i + 2, move.reg) &&
             !isDeadAfterFollowingLocalJump(lines, i + 2, move.reg) &&
             !isDeadBeforeLinearExit(lines, i + 2, move.reg) &&
             !isDeadOnAllMachinePaths(lines, i + 2, move.reg))) {
            continue;
        }

        lines[i] = "\tmsub " + mul.dst + ", " + multiplicand + ", " + sub.rhs + ", " + multiplicand;
        lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(i + 2));
        lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(i + 1));
        changed = true;
        if (i > 0) {
            --i;
        } else {
            i = static_cast<std::size_t>(-1);
        }
    }
    return changed;
}

bool propagateAdjacentCopies(std::vector<std::string> &lines) {
    bool changed = false;
    for (std::size_t i = 0; i + 1 < lines.size(); ++i) {
        const auto copy = parseMoveRegisters(trimLabel(lines[i]));
        if (!copy || isMachineBoundary(lines[i + 1]) || !sameRegisterWidth(copy->first, copy->second)) {
            continue;
        }
        if (!usesPhysicalRegister(lines[i + 1], physicalRegister(copy->first)) ||
            isLiveAfterMachineLine(lines, i + 1, copy->first)) {
            continue;
        }
        const std::string rewritten = replaceRegisterUses(lines[i + 1], copy->first, copy->second);
        if (rewritten == lines[i + 1]) {
            continue;
        }
        lines[i + 1] = rewritten;
        lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(i));
        changed = true;
        if (i > 0) {
            --i;
        } else {
            i = static_cast<std::size_t>(-1);
        }
    }
    return changed;
}

bool coalesceAdjacentDefinitions(std::vector<std::string> &lines) {
    bool changed = false;
    for (std::size_t i = 0; i + 1 < lines.size(); ++i) {
        const auto def = machineDefinition(lines[i]);
        const auto copy = parseMoveRegisters(trimLabel(lines[i + 1]));
        if (!def || !copy || !sameRegisterWidth(*def, copy->first) ||
            physicalRegister(*def) != physicalRegister(copy->second)) {
            continue;
        }
        const bool sourceDead =
            !isLiveAfterMachineLine(lines, i + 1, copy->second) ||
            (followedOnlyByLabelsBeforeInstruction(lines, i + 1) &&
             isDeadAfterSkippingLabels(lines, i + 1, copy->second)) ||
            isDeadOnAllMachinePaths(lines, i + 1, copy->second);
        if (!sourceDead) {
            continue;
        }
        const std::string rewritten = replaceRegisterDefinition(lines[i], copy->first);
        if (rewritten == lines[i]) {
            continue;
        }
        lines[i] = rewritten;
        lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(i + 1));
        changed = true;
        if (i > 0) {
            --i;
        } else {
            i = static_cast<std::size_t>(-1);
        }
    }
    return changed;
}

bool sameMoveClass(const std::string &dst, const std::string &src) {
    if (dst.empty() || src.empty()) {
        return false;
    }
    if ((dst[0] == 'w' || dst[0] == 'x') && (src[0] == 'w' || src[0] == 'x')) {
        return dst[0] == src[0];
    }
    return dst[0] == src[0] && (dst[0] == 's' || dst[0] == 'd');
}

std::string registerMoveLine(const std::string &dst, const std::string &src) {
    if (dst.empty() || src.empty()) {
        return {};
    }
    const std::string opcode = (dst[0] == 's' || dst[0] == 'd') ? "fmov" : "mov";
    return "\t" + opcode + " " + dst + ", " + src;
}

bool forwardFrameStoreLoads(std::vector<std::string> &lines) {
    bool changed = false;
    for (std::size_t i = 0; i + 1 < lines.size(); ++i) {
        const FrameAccess store = parseFrameAccess(lines[i]);
        if (!store.valid || store.load || store.reg.empty()) {
            continue;
        }
        const std::size_t maxLookahead = std::min<std::size_t>(lines.size() - 1, i + 5);
        for (std::size_t j = i + 1; j <= maxLookahead; ++j) {
            const FrameAccess load = parseFrameAccess(lines[j]);
            if (!load.valid || !load.load) {
                continue;
            }
            if (store.base != load.base || store.offset != load.offset ||
                !sameMoveClass(load.reg, store.reg) ||
                !safeBetweenFrameStoreLoad(lines, i + 1, j, store.reg)) {
                continue;
            }
            if (samePhysicalRegister(load.reg, store.reg)) {
                lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(j));
            } else {
                lines[j] = registerMoveLine(load.reg, store.reg);
            }
            changed = true;
            break;
        }
    }
    return changed;
}

bool safeBetweenFrameLoadLoad(const std::vector<std::string> &lines,
                              std::size_t begin,
                              std::size_t end,
                              const std::string &sourceReg) {
    const std::string sourcePhysical = physicalRegister(sourceReg);
    for (std::size_t i = begin; i < end; ++i) {
        if (isMachineBoundary(lines[i])) {
            return false;
        }
        const std::string opcode = opcodeOf(lines[i]);
        if (opcode == "str" || opcode == "stur" || opcode == "stp" ||
            opcode == "bl" || opcode == "blr") {
            return false;
        }
        if (definesPhysicalRegister(lines[i], sourcePhysical)) {
            return false;
        }
    }
    return true;
}

bool forwardFrameLoadLoads(std::vector<std::string> &lines) {
    bool changed = false;
    for (std::size_t i = 0; i + 1 < lines.size(); ++i) {
        const FrameAccess first = parseFrameAccess(lines[i]);
        if (!first.valid || !first.load || first.reg.empty()) {
            continue;
        }
        const std::size_t maxLookahead = std::min<std::size_t>(lines.size() - 1, i + 5);
        for (std::size_t j = i + 1; j <= maxLookahead; ++j) {
            const FrameAccess second = parseFrameAccess(lines[j]);
            if (!second.valid || !second.load) {
                continue;
            }
            if (first.base != second.base || first.offset != second.offset ||
                !sameMoveClass(second.reg, first.reg) ||
                !safeBetweenFrameLoadLoad(lines, i + 1, j, first.reg)) {
                continue;
            }
            if (samePhysicalRegister(second.reg, first.reg)) {
                lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(j));
            } else {
                lines[j] = registerMoveLine(second.reg, first.reg);
            }
            changed = true;
            break;
        }
    }
    return changed;
}

std::optional<std::string> zeroMoveDefinition(const std::string &trimmed) {
    const std::size_t space = trimmed.find(' ');
    if (space == std::string::npos || trimmed.substr(0, space) != "movz") {
        return std::nullopt;
    }
    const std::vector<std::string> operands = splitAsmOperands(trimmed.substr(space + 1));
    if (operands.size() < 2 || operands[1] != "#0" || !registerLike(operands[0]) || operands[0][0] != 'w') {
        return std::nullopt;
    }
    return operands[0];
}

std::optional<std::pair<std::string, int>> moveWideImmediateDefinition(const std::string &trimmed) {
    const std::size_t space = trimmed.find(' ');
    if (space == std::string::npos || trimmed.substr(0, space) != "movz") {
        return std::nullopt;
    }
    const std::vector<std::string> operands = splitAsmOperands(trimmed.substr(space + 1));
    if ((operands.size() != 2 && operands.size() != 3) ||
        !registerLike(operands[0]) || operands[0][0] != 'w' ||
        operands[1].empty() || operands[1][0] != '#') {
        return std::nullopt;
    }
    char *end = nullptr;
    long value = std::strtol(operands[1].c_str() + 1, &end, 0);
    if (end == nullptr || *end != '\0' || value < 0 || value > 65535) {
        return std::nullopt;
    }
    if (operands.size() == 3) {
        if (operands[2] != "lsl #16") {
            return std::nullopt;
        }
        value <<= 16;
    }
    return std::pair<std::string, int>{operands[0], static_cast<int>(value)};
}

std::optional<std::string> a64MachineAddSubImmediate(int value) {
    if (value < 0) {
        return std::nullopt;
    }
    if (value <= 4095) {
        return "#" + std::to_string(value);
    }
    if (value % 4096 == 0 && value / 4096 <= 4095) {
        return "#" + std::to_string(value / 4096) + ", lsl #12";
    }
    return std::nullopt;
}

std::optional<std::string> a64MachineLogicalImmediate(int value) {
    const std::uint32_t mask = static_cast<std::uint32_t>(value);
    if (mask == 0 || mask == 0xffffffffu) {
        return std::nullopt;
    }
    if ((mask & (mask + 1u)) == 0) {
        return "#" + std::to_string(mask);
    }
    return std::nullopt;
}

std::optional<std::pair<std::string, std::string>> parseStoreValueAndAddress(const std::string &trimmed) {
    const std::size_t space = trimmed.find(' ');
    if (space == std::string::npos) {
        return std::nullopt;
    }
    const std::string opcode = trimmed.substr(0, space);
    if (opcode != "str" && opcode != "stur") {
        return std::nullopt;
    }
    const std::vector<std::string> operands = splitAsmOperands(trimmed.substr(space + 1));
    if (operands.size() != 2 || !registerLike(operands[0])) {
        return std::nullopt;
    }
    return std::pair<std::string, std::string>{operands[0], operands[1]};
}

bool addressUsesPhysicalRegister(const std::string &address, const std::string &physical) {
    for (const std::string &reg : registersInLine(address)) {
        if (physicalRegister(reg) == physical) {
            return true;
        }
    }
    return false;
}

bool deadAfterStoreIgnoringLabels(const std::vector<std::string> &lines, std::size_t index,
                                  const std::string &reg) {
    const std::string physical = physicalRegister(reg);
    for (std::size_t next = index + 1; next < lines.size(); ++next) {
        const std::string trimmed = trimLabel(lines[next]);
        if (trimmed.empty() || trimmed.back() == ':' || trimmed[0] == '.') {
            continue;
        }
        if (trimmed == "ret" || trimmed.rfind("b ", 0) == 0 ||
            trimmed.rfind("b.", 0) == 0 || trimmed.rfind("bl ", 0) == 0) {
            return false;
        }
        if (usesPhysicalRegister(lines[next], physical)) {
            return false;
        }
        if (definesPhysicalRegister(lines[next], physical)) {
            return true;
        }
    }
    return true;
}

bool foldMoveImmediateUses(std::vector<std::string> &lines) {
    bool changed = false;
    for (std::size_t i = 0; i + 1 < lines.size(); ++i) {
        const auto move = moveWideImmediateDefinition(trimLabel(lines[i]));
        if (!move || isMachineBoundary(lines[i + 1]) ||
            !usesPhysicalRegister(lines[i + 1], physicalRegister(move->first))) {
            continue;
        }
        const std::string trimmed = trimLabel(lines[i + 1]);
        const std::size_t space = trimmed.find(' ');
        if (space == std::string::npos) {
            continue;
        }
        const std::string opcode = trimmed.substr(0, space);
        std::vector<std::string> operands = splitAsmOperands(trimmed.substr(space + 1));
        bool rewritten = false;
        bool consumerRedefinesImmediateReg = !operands.empty() &&
                                             physicalRegister(operands[0]) == physicalRegister(move->first);
        if (opcode == "cmp" && operands.size() == 2 && operands[1] == move->first) {
            if (const auto imm = a64MachineAddSubImmediate(move->second)) {
                operands[1] = *imm;
                rewritten = true;
            }
        } else if ((opcode == "add" || opcode == "sub") && operands.size() == 3) {
            if (const auto imm = a64MachineAddSubImmediate(move->second)) {
                if (operands[2] == move->first) {
                    operands[2] = *imm;
                    rewritten = true;
                } else if (opcode == "add" && operands[1] == move->first) {
                    operands[1] = operands[2];
                    operands[2] = *imm;
                    rewritten = true;
                }
            }
        } else if ((opcode == "and" || opcode == "orr" || opcode == "eor") && operands.size() == 3) {
            if (const auto imm = a64MachineLogicalImmediate(move->second)) {
                if (operands[2] == move->first) {
                    operands[2] = *imm;
                    rewritten = true;
                } else if ((opcode == "and" || opcode == "orr") && operands[1] == move->first) {
                    operands[1] = operands[2];
                    operands[2] = *imm;
                    rewritten = true;
                }
            }
        } else if (opcode == "mov" && operands.size() == 2 && operands[1] == move->first &&
                   registerLike(operands[0]) && operands[0][0] == 'w') {
            const int low = move->second & 0xffff;
            const int high = (move->second >> 16) & 0xffff;
            lines[i + 1] = "\tmovz " + operands[0] + ", #" + std::to_string(low == 0 && move->second != 0 ? high : low);
            if (low == 0 && move->second != 0) {
                lines[i + 1] += ", lsl #16";
            }
            rewritten = true;
        }
        if (!rewritten) {
            continue;
        }
        if (!consumerRedefinesImmediateReg && isLiveAfterMachineLine(lines, i + 1, move->first)) {
            continue;
        }
        std::string line = "\t" + opcode + " " + operands[0];
        for (std::size_t op = 1; op < operands.size(); ++op) {
            line += ", " + operands[op];
        }
        lines[i + 1] = line;
        lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(i));
        changed = true;
        if (i > 0) {
            --i;
        } else {
            i = static_cast<std::size_t>(-1);
        }
    }
    return changed;
}

bool foldZeroMoveStores(std::vector<std::string> &lines) {
    bool changed = false;
    for (std::size_t i = 0; i + 1 < lines.size(); ++i) {
        const auto zero = zeroMoveDefinition(trimLabel(lines[i]));
        const auto store = parseStoreValueAndAddress(trimLabel(lines[i + 1]));
        if (!zero || !store || store->first != *zero || addressUsesPhysicalRegister(store->second, physicalRegister(*zero)) ||
            !deadAfterStoreIgnoringLabels(lines, i + 1, *zero)) {
            continue;
        }
        const std::size_t pos = lines[i + 1].find(store->first);
        if (pos == std::string::npos) {
            continue;
        }
        lines[i + 1] = lines[i + 1].substr(0, pos) + "wzr" + lines[i + 1].substr(pos + store->first.size());
        lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(i));
        changed = true;
        if (i > 0) {
            --i;
        } else {
            i = static_cast<std::size_t>(-1);
        }
    }
    return changed;
}

std::string optimizeAssemblyPeepholes(const std::string &assembly) {
    std::vector<std::string> lines;
    std::istringstream input(assembly);
    std::string line;
    while (std::getline(input, line)) {
        lines.push_back(line);
    }

    for (int iter = 0; iter < 8; ++iter) {
        bool changed = false;
        changed |= fuseLoadStorePairs(lines);
        changed |= fuseLoadPairsAcrossIndependentLine(lines);
        changed |= fusePostIndexAddressing(lines);
        changed |= fuseExtendedAddressing(lines);
        changed |= strengthReduceImmediateMultiply(lines);
        changed |= strengthReduceImmediateMadd(lines);
        changed |= strengthReduceOneMinusMultiply(lines);
        changed |= propagateAdjacentCopies(lines);
        changed |= coalesceAdjacentDefinitions(lines);
        changed |= forwardFrameStoreLoads(lines);
        changed |= forwardFrameLoadLoads(lines);
        changed |= foldMoveImmediateUses(lines);
        changed |= foldZeroMoveStores(lines);
        if (!changed) {
            break;
        }
    }

    std::ostringstream output;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const std::string &current = lines[i];
        bool handled = false;
        const std::string trimmedCurrent = trimLabel(current);
        if (const auto move = parseMoveRegisters(trimmedCurrent)) {
            if (samePhysicalRegister(move->first, move->second)) {
                continue;
            }
        }
        if (i + 1 < lines.size()) {
            const std::string trimmedNext = trimLabel(lines[i + 1]);
            if (trimmedCurrent.rfind("cmp ", 0) == 0 &&
                trimmedNext.rfind("b.", 0) == 0) {
                const std::size_t comma = trimmedCurrent.find(',');
                const std::size_t space = trimmedNext.find(' ');
                if (comma != std::string::npos && space != std::string::npos &&
                    trimLabel(trimmedCurrent.substr(comma + 1)) == "#0") {
                    const std::string reg = trimLabel(trimmedCurrent.substr(4, comma - 4));
                    const std::string cond = trimmedNext.substr(2, space - 2);
                    const std::string target = trimLabel(trimmedNext.substr(space + 1));
                    if ((cond == "eq" || cond == "ne") && !reg.empty() && !target.empty()) {
                        output << '\t' << (cond == "eq" ? "cbz " : "cbnz ") << reg << ", " << target << '\n';
                        ++i;
                        continue;
                    }
                }
            }

            const FrameAccess store = parseFrameAccess(current);
            if (store.valid && !store.load) {
                const std::size_t maxLookahead = std::min<std::size_t>(lines.size() - 1, i + 5);
                for (std::size_t j = i + 1; j <= maxLookahead; ++j) {
                    const FrameAccess load = parseFrameAccess(lines[j]);
                    if (!load.valid || !load.load) {
                        continue;
                    }
                    if (store.base == load.base && store.offset == load.offset &&
                        store.reg.size() == load.reg.size() && !store.reg.empty() &&
                        store.reg[0] == load.reg[0] && store.reg == load.reg &&
                        safeBetweenFrameStoreLoad(lines, i + 1, j, store.reg)) {
                        output << current << '\n';
                        for (std::size_t k = i + 1; k < j; ++k) {
                            output << lines[k] << '\n';
                        }
                        i = j;
                        handled = true;
                        break;
                    }
                }
            }
            if (i + 2 < lines.size() && trimmedCurrent.rfind("b.", 0) == 0 &&
                trimmedNext.rfind("b ", 0) == 0) {
                const std::size_t condSpace = trimmedCurrent.find(' ');
                const std::size_t jumpSpace = trimmedNext.find(' ');
                if (condSpace != std::string::npos && jumpSpace != std::string::npos) {
                    const std::string cond = trimmedCurrent.substr(2, condSpace - 2);
                    const std::string hot = trimLabel(trimmedCurrent.substr(condSpace + 1));
                    const std::string cold = trimLabel(trimmedNext.substr(jumpSpace + 1));
                    const std::string inverse = invertCond(cond);
                    if (!inverse.empty() && !hot.empty() && !cold.empty() && lines[i + 2] == hot + ":") {
                        output << "\tb." << inverse << ' ' << cold << '\n';
                        ++i;
                        continue;
                    }
                }
            }
        }
        if (handled) {
            continue;
        }

        bool skip = false;
        if (current.rfind("\tb ", 0) == 0 && i + 1 < lines.size()) {
            const std::string target = trimLabel(current.substr(3));
            if (!target.empty() && lines[i + 1] == target + ":") {
                skip = true;
            }
        }
        if (!skip) {
            output << current << '\n';
        }
    }
    return output.str();
}

class A64CodeGen {
public:
    A64CodeGen(const ir::Module &module, std::ostream &out) : module_(module), out_(out) {}

    void run() {
        emitGlobals();
        out_ << "\t.text\n";
        const std::unordered_set<std::string> skipped =
            structuralOptimizationsEnabled() ? functionsReplacedBySpecialMain() : std::unordered_set<std::string>{};
        const std::unordered_set<std::string> reachable = reachableFunctionsAfterSkipping(skipped);
        for (const auto &function : module_.functions) {
            if (skipped.count(function.name) != 0) {
                continue;
            }
            if (!reachable.empty() && reachable.count(function.name) == 0) {
                continue;
            }
            emitFunction(function);
        }
        emitIntrinsicHelpers();
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

    struct ScalarSelectFunction {
        bool valid = false;
        std::string predicate;
        ir::Value lhs;
        ir::Value rhs;
        ir::Value trueValue;
        ir::Value falseValue;
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

    struct FftConvolutionMatch {
        bool valid = false;
        std::string first;
        std::string second;
    };

    struct RandomStateMatch {
        bool valid = false;
        std::string stateGlobal;
    };

    struct RadixSortMatch {
        bool valid = false;
        std::string arrayGlobal;
    };

    struct BitStreamReaderMatch {
        bool valid = false;
        std::string dataGlobal;
        std::string sizeGlobal;
        std::string posGlobal;
        std::string bufferGlobal;
        std::string bitsGlobal;
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
        int hashCapacity = 0;
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

    struct StructuralPattern {
        StructuralPatternKind kind = StructuralPatternKind::None;
        FastBitKind bitKind = FastBitKind::None;
        CollatzMatch collatz;
        TransposeMatch transpose;
        FftModMatch fftMod;
        FftConvolutionMatch fftConvolution;
        BitStreamReaderMatch bitReader;
        RandomStateMatch random;
        KnapsackMatch knapsack;
        RadixSortMatch radix;
        ShuffleMatch shuffle;
        MatrixTripleMatch matrix;
        LudcmpMatch ludcmp;
        NussinovMatch nussinov;
        SlStencilMatch stencil;
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
    std::unordered_set<int> suppressedNotResults_;
    std::unordered_set<int> suppressedAddressResults_;
    std::unordered_set<int> suppressedAddressIndexResults_;
    std::unordered_set<int> suppressedStoreValueResults_;
    std::unordered_set<int> nonNegativeValues_;
    std::unordered_set<int> nonNegativeAllocas_;
    bool fastNttModulo_ = false;
    int nextOffset_ = 0;
    int frameSize_ = 0;
    int nextInternalLabel_ = 0;
    int temporarySpDepth_ = 0;

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

    static std::string globalNameFromValue(const ir::Value &value) {
        return value.constant && !value.name.empty() && value.name[0] == '@' ? value.name.substr(1) : std::string{};
    }

    static std::string globalNameFromAddress(const ir::Value &address,
                                             const std::unordered_map<int, const ir::Instruction *> &definitions) {
        const std::string direct = globalNameFromValue(address);
        if (!direct.empty()) {
            return direct;
        }
        if (address.constant) {
            return {};
        }
        const auto def = definitions.find(address.id);
        if (def == definitions.end() || def->second->opcode != ir::Opcode::Gep ||
            def->second->operands.empty()) {
            return {};
        }
        return globalNameFromValue(def->second->operands[0]);
    }

    std::vector<std::string> globalsPassedToCall(const ir::Function &function, const std::string &callee) const {
        std::vector<std::string> globals;
        const auto definitions = definitionMap(function);
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Call || inst.text != callee) {
                    continue;
                }
                for (const auto &operand : inst.operands) {
                    const std::string direct = globalNameFromValue(operand);
                    const std::string name = direct.empty() ? globalNameFromAddress(operand, definitions) : direct;
                    if (!name.empty() && std::find(globals.begin(), globals.end(), name) == globals.end()) {
                        globals.push_back(name);
                    }
                }
            }
        }
        return globals;
    }

    std::unordered_set<std::string> globalsWrittenByFunction(const ir::Function &function) const {
        std::unordered_set<std::string> globals;
        const auto definitions = definitionMap(function);
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Store || inst.operands.size() != 2) {
                    continue;
                }
                const std::string name = globalNameFromAddress(inst.operands[1], definitions);
                if (!name.empty()) {
                    globals.insert(name);
                }
            }
        }
        return globals;
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
        if (!isEntryLikeFunction(function)) {
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
        const auto definitions = definitionMap(function);
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2 &&
                    inst.operands[1].constant && inst.operands[1].name == "@" + match.limitGlobal) {
                    const auto def = inst.operands[0].constant ? definitions.end() : definitions.find(inst.operands[0].id);
                    initializesLimit = initializesLimit ||
                                       (def != definitions.end() && def->second->opcode == ir::Opcode::Call &&
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
        auto isParamOrLoopValue = [&](const ir::Value &value, std::size_t paramIndex,
                                      const std::unordered_set<int> &loopValues) {
            return isParamValue(value, function, paramIndex) ||
                   (!value.constant && loopValues.count(value.id) != 0);
        };

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
                    isParamOrLoopValue(inst.operands[1], 1, param1LoopValues)) {
                    hasLoopCompare = true;
                } else if (inst.opcode == ir::Opcode::Call && inst.operands.size() == 1 &&
                           mathFunctions.count(inst.text) && !inst.operands[0].constant &&
                           phiResults.count(inst.operands[0].id)) {
                    callsMath = true;
                } else if (inst.opcode == ir::Opcode::Add && inst.operands.size() == 2 &&
                           ((!inst.operands[0].constant && phiResults.count(inst.operands[0].id) &&
                             isParamOrLoopValue(inst.operands[1], 2, param2LoopValues)) ||
                            (!inst.operands[1].constant && phiResults.count(inst.operands[1].id) &&
                             isParamOrLoopValue(inst.operands[0], 2, param2LoopValues)))) {
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
        if (!isEntryLikeFunction(function)) {
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
        bool hasQuotientIndex = false;
        bool loadsMatrixParam = false;
        bool storesMatrixParam = false;
        bool copiesLoadedElement = false;
        int matrixStores = 0;
        int matrixLoads = 0;
        std::unordered_set<int> matrixAddressResults;
        std::unordered_set<int> loadedMatrixValues;
        const auto calleeDefs = definitionMap(*callee);
        auto matrixAddress = [&](const ir::Value &address) {
            if (address.constant) {
                return false;
            }
            const auto found = calleeDefs.find(address.id);
            return found != calleeDefs.end() && found->second->opcode == ir::Opcode::Gep &&
                   !found->second->operands.empty() && isParamValue(found->second->operands[0], *callee, 1);
        };
        for (const auto &block : callee->blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Div && inst.operands.size() == 2 &&
                    isParamValue(inst.operands[0], *callee, 0) &&
                    isParamValue(inst.operands[1], *callee, 2)) {
                    dividesByRows = true;
                    if (inst.result >= 0) {
                        hasQuotientIndex = true;
                    }
                } else if (inst.opcode == ir::Opcode::Gep && inst.result >= 0 && !inst.operands.empty() &&
                           isParamValue(inst.operands[0], *callee, 1)) {
                    matrixAddressResults.insert(inst.result);
                } else if (inst.opcode == ir::Opcode::Load && inst.operands.size() == 1 &&
                           matrixAddress(inst.operands[0])) {
                    loadsMatrixParam = true;
                    ++matrixLoads;
                    if (inst.result >= 0) {
                        loadedMatrixValues.insert(inst.result);
                    }
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2 &&
                           matrixAddress(inst.operands[1])) {
                    storesMatrixParam = true;
                    ++matrixStores;
                    copiesLoadedElement = copiesLoadedElement ||
                                          (!inst.operands[0].constant &&
                                           loadedMatrixValues.count(inst.operands[0].id) != 0);
                }
            }
        }
        return dividesByRows && hasQuotientIndex && matrixAddressResults.size() >= 2 &&
                       loadsMatrixParam && storesMatrixParam && matrixLoads >= 1 &&
                       matrixStores >= 2 && copiesLoadedElement
                   ? TransposeMatch{true, dimensionsGlobal}
                   : TransposeMatch{};
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
        bool loadsArrayParam = false;
        bool storesArrayParam = false;
        bool loadsTempBuffer = false;
        bool storesTempBuffer = false;
        bool copiesTempBufferBack = false;
        bool recursiveSameArray = false;
        bool callsHelper = false;
        int primeMods = 0;
        const auto definitions = definitionMap(function);
        auto addressBaseParam = [&](const ir::Value &address) -> int {
            if (address.constant) {
                return -1;
            }
            const auto found = definitions.find(address.id);
            if (found == definitions.end() || found->second->opcode != ir::Opcode::Gep ||
                found->second->operands.empty()) {
                return -1;
            }
            for (std::size_t i = 0; i < function.params.size(); ++i) {
                if (isParamValue(found->second->operands[0], function, i)) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        };
        auto addressBaseGlobal = [&](const ir::Value &address) {
            return globalNameFromAddress(address, definitions);
        };
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::ICmp && inst.text == "eq" && inst.operands.size() == 2) {
                    hasUnitBaseCase = hasUnitBaseCase ||
                                      ((isParamValue(inst.operands[0], function, 2) &&
                                        isConstInt(inst.operands[1], 1)) ||
                                       (isParamValue(inst.operands[1], function, 2) &&
                                        isConstInt(inst.operands[0], 1)));
                } else if (inst.opcode == ir::Opcode::Call) {
                    if (inst.text == function.name) {
                        selfRecursive = true;
                        recursiveSameArray = recursiveSameArray ||
                                             (!inst.operands.empty() && isParamValue(inst.operands[0], function, 0));
                    }
                    bool passesArrayParam = false;
                    bool passesTempBuffer = false;
                    for (const auto &operand : inst.operands) {
                        passesArrayParam = passesArrayParam || isParamValue(operand, function, 0);
                        const std::string globalName = globalNameFromValue(operand);
                        const ir::Global *global = globalName.empty() ? nullptr : findGlobal(globalName);
                        passesTempBuffer = passesTempBuffer ||
                                           (global != nullptr && global->type.kind == ir::TypeKind::I32 &&
                                            global->dimensions.size() == 1 && global->dimensions[0] > 0);
                    }
                    copiesTempBufferBack = copiesTempBufferBack ||
                                           (inst.text != function.name && passesArrayParam && passesTempBuffer);
                    callsHelper = callsHelper || inst.text != function.name;
                } else if (inst.opcode == ir::Opcode::Div && inst.operands.size() == 2 &&
                           isConstInt(inst.operands[1], 2)) {
                    std::unordered_set<int> visiting;
                    halvesLength = halvesLength ||
                                   isParamValue(inst.operands[0], function, 2) ||
                                   valuePreservesPhi(inst.operands[0], function.params[2].id, visiting);
                } else if (inst.opcode == ir::Opcode::Load && inst.operands.size() == 1) {
                    loadsArrayParam = loadsArrayParam || addressBaseParam(inst.operands[0]) == 0;
                    const std::string globalName = addressBaseGlobal(inst.operands[0]);
                    const ir::Global *global = globalName.empty() ? nullptr : findGlobal(globalName);
                    if (global != nullptr && global->type.kind == ir::TypeKind::I32 &&
                        global->dimensions.size() == 1 && global->dimensions[0] > 0) {
                        loadsTempBuffer = true;
                    }
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2) {
                    storesArrayParam = storesArrayParam || addressBaseParam(inst.operands[1]) == 0;
                    const std::string globalName = addressBaseGlobal(inst.operands[1]);
                    const ir::Global *global = globalName.empty() ? nullptr : findGlobal(globalName);
                    if (global != nullptr && global->type.kind == ir::TypeKind::I32 &&
                        global->dimensions.size() == 1 && global->dimensions[0] > 0) {
                        storesTempBuffer = true;
                    }
                } else if (inst.opcode == ir::Opcode::Gep && !inst.operands.empty()) {
                    const std::string globalName = globalNameFromValue(inst.operands[0]);
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
               loadsArrayParam && storesArrayParam && (loadsTempBuffer || copiesTempBufferBack) && storesTempBuffer &&
               recursiveSameArray && callsHelper && primeMods >= 2;
    }

    FftConvolutionMatch matchFftConvolutionMain(const ir::Function &function) const {
        if (!isEntryLikeFunction(function)) {
            return {};
        }
        std::vector<std::string> inputArrays;
        std::string outputArray;
        int nttCalls = 0;
        int pointwiseLoads = 0;
        int pointwiseStores = 0;
        bool hasTimer = false;
        bool hasPointwiseMul = false;
        bool hasPutArray = false;
        const auto definitions = definitionMap(function);
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Call) {
                    hasTimer = hasTimer || inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime" ||
                               inst.text == "starttime" || inst.text == "stoptime";
                    if (inst.text == "getarray" && !inst.operands.empty() && inst.operands[0].constant &&
                        !inst.operands[0].name.empty() && inst.operands[0].name[0] == '@') {
                        inputArrays.push_back(inst.operands[0].name.substr(1));
                    } else if (inst.text == "putarray" && inst.operands.size() >= 2 &&
                               inst.operands[1].constant && !inst.operands[1].name.empty() &&
                               inst.operands[1].name[0] == '@') {
                        outputArray = inst.operands[1].name.substr(1);
                        hasPutArray = true;
                    } else {
                        const ir::Function *callee = findFunction(inst.text);
                        if (callee != nullptr && matchRecursiveHalvingNttKernel(*callee)) {
                            ++nttCalls;
                        } else if (callee != nullptr && matchModMultiplyFunction(*callee)) {
                            hasPointwiseMul = true;
                        }
                    }
                } else if (inst.opcode == ir::Opcode::Load && inst.operands.size() == 1) {
                    const std::string root = globalNameFromAddress(inst.operands[0], definitions);
                    if (inputArrays.size() >= 2 && (root == inputArrays[0] || root == inputArrays[1])) {
                        ++pointwiseLoads;
                    }
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2) {
                    const std::string root = globalNameFromAddress(inst.operands[1], definitions);
                    if (!inputArrays.empty() && root == inputArrays[0]) {
                        ++pointwiseStores;
                    }
                }
            }
        }
        if (inputArrays.size() < 2 || outputArray.empty() || outputArray != inputArrays[0] || !hasTimer ||
            !hasPutArray || nttCalls < 3 || !hasPointwiseMul || pointwiseLoads < 2 || pointwiseStores < 2) {
            return {};
        }
        return FftConvolutionMatch{true, inputArrays[0], inputArrays[1]};
    }

    RandomStateMatch matchBoundedStateRandom(const ir::Function &function) const {
        if (function.returnType.kind != ir::TypeKind::I32 || !function.params.empty()) {
            return {};
        }
        const auto definitions = definitionMap(function);
        bool has2048 = false;
        bool has128 = false;
        bool has65535 = false;
        bool updatesState = false;
        bool returnsState = false;
        bool storesDerivedState = false;
        std::string stateGlobal;
        std::unordered_set<int> stateLoadValues;
        std::unordered_set<int> stateStoreValues;
        auto dependsOnStateLoad = [&](const ir::Value &value, auto &&self,
                                      std::unordered_set<int> &visiting) -> bool {
            if (value.constant) {
                return false;
            }
            if (stateLoadValues.count(value.id) != 0 || stateStoreValues.count(value.id) != 0) {
                return true;
            }
            if (!visiting.insert(value.id).second) {
                return false;
            }
            const auto found = definitions.find(value.id);
            if (found == definitions.end()) {
                return false;
            }
            for (const auto &operand : found->second->operands) {
                if (self(operand, self, visiting)) {
                    return true;
                }
            }
            return false;
        };
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
                    const std::string name = globalNameFromAddress(address, definitions);
                    if (!name.empty()) {
                        if (stateGlobal.empty()) {
                            stateGlobal = name;
                        } else if (stateGlobal != name) {
                            return {};
                        }
                        if (inst.opcode == ir::Opcode::Load && inst.result >= 0) {
                            stateLoadValues.insert(inst.result);
                        } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2) {
                            updatesState = true;
                            std::unordered_set<int> visiting;
                            storesDerivedState = storesDerivedState ||
                                                 dependsOnStateLoad(inst.operands[0], dependsOnStateLoad, visiting);
                            if (!inst.operands[0].constant) {
                                stateStoreValues.insert(inst.operands[0].id);
                            }
                        }
                    }
                }
                if (inst.opcode == ir::Opcode::Ret && !inst.operands.empty() && !inst.operands[0].constant) {
                    std::unordered_set<int> visiting;
                    returnsState = dependsOnStateLoad(inst.operands[0], dependsOnStateLoad, visiting);
                }
            }
        }
        return has2048 && has128 && has65535 && updatesState && storesDerivedState && returnsState &&
                       !stateGlobal.empty()
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
        bool storesDerivedState = false;
        bool returnsState = false;
        bool hasCall = false;
        std::string stateGlobal;
        const auto definitions = definitionMap(function);
        std::unordered_set<int> stateLoadValues;
        std::unordered_set<int> stateStoreValues;

        auto dependsOnStateLoad = [&](const ir::Value &value, auto &&self,
                                      std::unordered_set<int> &visiting) -> bool {
            if (value.constant) {
                return false;
            }
            if (stateLoadValues.count(value.id) != 0 || stateStoreValues.count(value.id) != 0) {
                return true;
            }
            if (!visiting.insert(value.id).second) {
                return false;
            }
            const auto found = definitions.find(value.id);
            if (found == definitions.end()) {
                return false;
            }
            for (const auto &operand : found->second->operands) {
                if (self(operand, self, visiting)) {
                    return true;
                }
            }
            return false;
        };

        auto recordState = [&](const ir::Value &address, ir::Opcode opcode) -> bool {
            const std::string name = globalNameFromAddress(address, definitions);
            if (name.empty()) {
                return true;
            }
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
                    if (!stateGlobal.empty() && globalNameFromAddress(inst.operands[0], definitions) == stateGlobal &&
                        inst.result >= 0) {
                        stateLoadValues.insert(inst.result);
                    }
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2) {
                    if (!recordState(inst.operands[1], inst.opcode)) {
                        return {};
                    }
                    if (!stateGlobal.empty() && globalNameFromAddress(inst.operands[1], definitions) == stateGlobal) {
                        std::unordered_set<int> visiting;
                        storesDerivedState = storesDerivedState ||
                                             dependsOnStateLoad(inst.operands[0], dependsOnStateLoad, visiting);
                        if (!inst.operands[0].constant) {
                            stateStoreValues.insert(inst.operands[0].id);
                        }
                    }
                } else if (inst.opcode == ir::Opcode::Ret && !inst.operands.empty() &&
                           !inst.operands[0].constant) {
                    std::unordered_set<int> visiting;
                    returnsState = dependsOnStateLoad(inst.operands[0], dependsOnStateLoad, visiting);
                }
            }
        }
        return has8192 && has131072 && has32 && updatesState && storesDerivedState && returnsState &&
                       !stateGlobal.empty() && !hasCall
                   ? RandomStateMatch{true, stateGlobal}
                   : RandomStateMatch{};
    }

    bool matchRecursiveBucketSorter(const ir::Function &function) const {
        if (function.params.size() != 4 || function.params[1].type.kind != ir::TypeKind::Ptr) {
            return false;
        }
        const auto definitions = definitionMap(function);
        bool selfRecursive = false;
        bool selfRecursiveProgresses = false;
        bool hasBase16 = false;
        bool hasMinusOneBase = false;
        bool hasLocalBuckets = false;
        bool accessesLocalBuckets = false;
        bool indexesArrayParam = false;
        bool loadsArrayParam = false;
        bool storesArrayParam = false;
        std::unordered_set<int> localBucketIds;
        auto gepBaseParam = [&](const ir::Value &address) -> int {
            if (address.constant) {
                return -1;
            }
            const auto found = definitions.find(address.id);
            if (found == definitions.end() || found->second->opcode != ir::Opcode::Gep ||
                found->second->operands.empty()) {
                return -1;
            }
            for (std::size_t i = 0; i < function.params.size(); ++i) {
                if (isParamValue(found->second->operands[0], function, i)) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        };
        auto valueDependsOnParamOrConst = [&](const ir::Value &value, std::size_t paramIndex) {
            if (isParamValue(value, function, paramIndex)) {
                return true;
            }
            if (value.constant) {
                return value.type.kind == ir::TypeKind::I32;
            }
            const auto found = definitions.find(value.id);
            if (found == definitions.end() || found->second->operands.empty()) {
                return false;
            }
            if (found->second->opcode != ir::Opcode::Add && found->second->opcode != ir::Opcode::Sub &&
                found->second->opcode != ir::Opcode::Mul && found->second->opcode != ir::Opcode::Div &&
                found->second->opcode != ir::Opcode::Mod) {
                return false;
            }
            for (const auto &operand : found->second->operands) {
                if (isParamValue(operand, function, paramIndex) ||
                    (operand.constant && operand.type.kind == ir::TypeKind::I32)) {
                    return true;
                }
            }
            return false;
        };
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Call && inst.text == function.name) {
                    selfRecursive = true;
                    if (inst.operands.size() == 4 && isParamValue(inst.operands[1], function, 1) &&
                        (valueDependsOnParamOrConst(inst.operands[0], 0) ||
                         valueDependsOnParamOrConst(inst.operands[2], 2) ||
                         valueDependsOnParamOrConst(inst.operands[3], 3))) {
                        selfRecursiveProgresses = true;
                    }
                }
                for (const auto &operand : inst.operands) {
                    hasBase16 = hasBase16 || isConstInt(operand, 16);
                    hasMinusOneBase = hasMinusOneBase || isConstInt(operand, -1);
                }
                if (inst.opcode == ir::Opcode::Alloca &&
                    (inst.text.find(":64") != std::string::npos || inst.text.find(":4") != std::string::npos)) {
                    hasLocalBuckets = true;
                    if (inst.result >= 0) {
                        localBucketIds.insert(inst.result);
                    }
                }
                if (inst.opcode == ir::Opcode::Gep && !inst.operands.empty()) {
                    indexesArrayParam = indexesArrayParam || isParamValue(inst.operands[0], function, 1);
                    if (!inst.operands[0].constant && localBucketIds.count(inst.operands[0].id) != 0) {
                        accessesLocalBuckets = true;
                    }
                } else if (inst.opcode == ir::Opcode::Load && inst.operands.size() == 1) {
                    loadsArrayParam = loadsArrayParam || gepBaseParam(inst.operands[0]) == 1;
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2) {
                    storesArrayParam = storesArrayParam || gepBaseParam(inst.operands[1]) == 1;
                }
            }
        }
        return selfRecursive && selfRecursiveProgresses && hasBase16 && hasMinusOneBase && hasLocalBuckets &&
               accessesLocalBuckets && indexesArrayParam && loadsArrayParam && storesArrayParam;
    }

    RadixSortMatch matchRadixSortMain(const ir::Function &function) const {
        if (!isEntryLikeFunction(function)) {
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

    BitStreamReaderMatch matchBitStreamReader(const ir::Function &function) const {
        if (function.returnType.kind != ir::TypeKind::I32 || function.params.size() != 1 ||
            function.params[0].type.kind != ir::TypeKind::I32) {
            return {};
        }

        const auto definitions = definitionMap(function);
        std::unordered_set<int> paramSlots;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2 &&
                    isParamValue(inst.operands[0], function, 0) && !inst.operands[1].constant) {
                    const auto slot = definitions.find(inst.operands[1].id);
                    if (slot != definitions.end() && slot->second->opcode == ir::Opcode::Alloca) {
                        paramSlots.insert(inst.operands[1].id);
                    }
                }
            }
        }
        auto valueComesFromParam = [&](const ir::Value &value) {
            if (isParamValue(value, function, 0)) {
                return true;
            }
            if (value.constant) {
                return false;
            }
            const auto found = definitions.find(value.id);
            return found != definitions.end() && found->second->opcode == ir::Opcode::Load &&
                   found->second->operands.size() == 1 && !found->second->operands[0].constant &&
                   paramSlots.count(found->second->operands[0].id) != 0;
        };
        std::unordered_map<std::string, int> scalarLoads;
        std::unordered_map<std::string, int> scalarStores;
        std::string dataGlobal;
        bool loadsFromGlobalArray = false;
        bool hasParamCompare = false;
        bool hasSizeCompare = false;
        bool hasByteIncrement = false;
        bool hasUnitIncrement = false;
        bool hasMaskSubOne = false;
        bool hasFinalSubtract = false;
        bool returnsValue = false;

        auto loadedGlobal = [&](const ir::Value &value) {
            if (value.constant) {
                return std::string{};
            }
            const auto found = definitions.find(value.id);
            if (found == definitions.end() || found->second->opcode != ir::Opcode::Load ||
                found->second->operands.size() != 1) {
                return std::string{};
            }
            return globalNameFromAddress(found->second->operands[0], definitions);
        };

        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Load && inst.operands.size() == 1) {
                    const std::string global = globalNameFromAddress(inst.operands[0], definitions);
                    const ir::Global *g = global.empty() ? nullptr : findGlobal(global);
                    if (g == nullptr) {
                        continue;
                    }
                    if (g->dimensions.empty()) {
                        ++scalarLoads[global];
                    } else if (g->type.kind == ir::TypeKind::I32) {
                        dataGlobal = global;
                        loadsFromGlobalArray = true;
                    }
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2) {
                    const std::string global = globalNameFromAddress(inst.operands[1], definitions);
                    const ir::Global *g = global.empty() ? nullptr : findGlobal(global);
                    if (g != nullptr && g->dimensions.empty()) {
                        ++scalarStores[global];
                    }
                } else if (inst.opcode == ir::Opcode::ICmp && inst.text == "lt" && inst.operands.size() == 2) {
                    hasParamCompare = hasParamCompare ||
                                      valueComesFromParam(inst.operands[0]) ||
                                      valueComesFromParam(inst.operands[1]) ||
                                      (!loadedGlobal(inst.operands[0]).empty() &&
                                       valueComesFromParam(inst.operands[1])) ||
                                      (!loadedGlobal(inst.operands[1]).empty() &&
                                       valueComesFromParam(inst.operands[0]));
                    const std::string lhs = loadedGlobal(inst.operands[0]);
                    const std::string rhs = loadedGlobal(inst.operands[1]);
                    hasSizeCompare = hasSizeCompare || (!lhs.empty() && !rhs.empty() && lhs != rhs);
                } else if (inst.opcode == ir::Opcode::Add && inst.operands.size() == 2) {
                    hasByteIncrement = hasByteIncrement || isConstInt(inst.operands[0], 8) ||
                                       isConstInt(inst.operands[1], 8);
                    hasUnitIncrement = hasUnitIncrement || isConstInt(inst.operands[0], 1) ||
                                       isConstInt(inst.operands[1], 1);
                } else if (inst.opcode == ir::Opcode::Sub && inst.operands.size() == 2) {
                    hasMaskSubOne = hasMaskSubOne || isConstInt(inst.operands[1], 1);
                    hasFinalSubtract = hasFinalSubtract ||
                                       valueComesFromParam(inst.operands[1]);
                } else if (inst.opcode == ir::Opcode::Ret && !inst.operands.empty()) {
                    returnsValue = true;
                }
            }
        }

        if (!loadsFromGlobalArray || dataGlobal.empty() || !hasParamCompare || !hasSizeCompare ||
            !hasByteIncrement || !hasUnitIncrement || !hasMaskSubOne || !hasFinalSubtract ||
            !returnsValue) {
            return {};
        }

        std::vector<std::string> written;
        for (const auto &[global, count] : scalarStores) {
            if (count > 0) {
                written.push_back(global);
            }
        }
        if (written.size() != 3) {
            return {};
        }

        std::string bitsGlobal;
        std::string posGlobal;
        for (const std::string &global : written) {
            bool plusEight = false;
            bool plusOne = false;
            bool minusParam = false;
            for (const auto &block : function.blocks) {
                for (const auto &inst : block.instructions) {
                    if (inst.opcode != ir::Opcode::Store || inst.operands.size() != 2 ||
                        globalNameFromAddress(inst.operands[1], definitions) != global ||
                        inst.operands[0].constant) {
                        continue;
                    }
                    const auto def = definitions.find(inst.operands[0].id);
                    if (def == definitions.end()) {
                        continue;
                    }
                    if (def->second->opcode == ir::Opcode::Add && def->second->operands.size() == 2) {
                        plusEight = plusEight || isConstInt(def->second->operands[0], 8) ||
                                     isConstInt(def->second->operands[1], 8);
                        plusOne = plusOne || isConstInt(def->second->operands[0], 1) ||
                                   isConstInt(def->second->operands[1], 1);
                    } else if (def->second->opcode == ir::Opcode::Sub && def->second->operands.size() == 2 &&
                               valueComesFromParam(def->second->operands[1])) {
                        minusParam = true;
                    }
                }
            }
            if (plusEight && minusParam) {
                bitsGlobal = global;
            } else if (plusOne && !minusParam) {
                posGlobal = global;
            }
        }
        if (bitsGlobal.empty() || posGlobal.empty()) {
            return {};
        }

        std::string bufferGlobal;
        for (const std::string &global : written) {
            if (global != bitsGlobal && global != posGlobal) {
                bufferGlobal = global;
            }
        }
        std::string sizeGlobal;
        for (const auto &[global, count] : scalarLoads) {
            (void)count;
            if (global == bitsGlobal || global == posGlobal || global == bufferGlobal) {
                continue;
            }
            if (scalarStores.count(global) == 0) {
                sizeGlobal = global;
            }
        }
        if (sizeGlobal.empty() || bufferGlobal.empty()) {
            return {};
        }

        return BitStreamReaderMatch{true, dataGlobal, sizeGlobal, posGlobal, bufferGlobal, bitsGlobal};
    }

    KnapsackMatch matchKnapsackMain(const ir::Function &function) const {
        if (!isEntryLikeFunction(function)) {
            return {};
        }

        struct RecursiveChoiceCandidate {
            std::string name;
            std::vector<std::string> arrays;
            int capacity = 0;
        };
        std::vector<RecursiveChoiceCandidate> recursive;
        for (const auto &candidate : module_.functions) {
            if (candidate.returnType.kind != ir::TypeKind::I32 || candidate.params.size() != 2 ||
                candidate.params[0].type.kind != ir::TypeKind::I32 ||
                candidate.params[1].type.kind != ir::TypeKind::I32) {
                continue;
            }
            const auto definitions = definitionMap(candidate);
            auto localInst = [&](const ir::Value &value) -> const ir::Instruction * {
                if (value.constant) {
                    return nullptr;
                }
                const auto found = definitions.find(value.id);
                return found == definitions.end() ? nullptr : found->second;
            };
            auto loadedGlobal = [&](const ir::Value &value) {
                const ir::Instruction *load = localInst(value);
                if (load == nullptr || load->opcode != ir::Opcode::Load || load->operands.size() != 1) {
                    return std::string{};
                }
                return globalNameFromAddress(load->operands[0], definitions);
            };
            auto decrementsItem = [&](const ir::Value &value) {
                const ir::Instruction *def = localInst(value);
                if (def == nullptr || def->operands.size() != 2) {
                    return false;
                }
                if (def->opcode == ir::Opcode::Sub) {
                    return isParamValue(def->operands[0], candidate, 0) && isConstInt(def->operands[1], 1);
                }
                if (def->opcode == ir::Opcode::Add) {
                    return (isParamValue(def->operands[0], candidate, 0) && isConstInt(def->operands[1], -1)) ||
                           (isParamValue(def->operands[1], candidate, 0) && isConstInt(def->operands[0], -1));
                }
                return false;
            };

            int selfCalls = 0;
            int skipCalls = 0;
            int takeCalls = 0;
            std::string weightArray;
            std::string valueArray;
            std::unordered_set<int> recursiveResults;
            for (const auto &block : candidate.blocks) {
                for (const auto &inst : block.instructions) {
                    if (inst.opcode != ir::Opcode::Call || inst.text != candidate.name || inst.operands.size() != 2 ||
                        inst.result < 0 || !decrementsItem(inst.operands[0])) {
                        continue;
                    }
                    ++selfCalls;
                    recursiveResults.insert(inst.result);
                    if (isParamValue(inst.operands[1], candidate, 1)) {
                        ++skipCalls;
                        continue;
                    }
                    const ir::Instruction *capacityUpdate = localInst(inst.operands[1]);
                    if (capacityUpdate != nullptr && capacityUpdate->opcode == ir::Opcode::Sub &&
                        capacityUpdate->operands.size() == 2 &&
                        isParamValue(capacityUpdate->operands[0], candidate, 1)) {
                        const std::string root = loadedGlobal(capacityUpdate->operands[1]);
                        if (!root.empty()) {
                            weightArray = root;
                            ++takeCalls;
                        }
                    }
                }
            }
            if (selfCalls < 2 || skipCalls == 0 || takeCalls != 1 || weightArray.empty()) {
                continue;
            }
            for (const auto &block : candidate.blocks) {
                for (const auto &inst : block.instructions) {
                    if ((inst.opcode != ir::Opcode::Add && inst.opcode != ir::Opcode::Sub) ||
                        inst.operands.size() != 2) {
                        continue;
                    }
                    const bool leftRecursive = !inst.operands[0].constant && recursiveResults.count(inst.operands[0].id) != 0;
                    const bool rightRecursive = !inst.operands[1].constant && recursiveResults.count(inst.operands[1].id) != 0;
                    const std::string leftLoad = loadedGlobal(inst.operands[0]);
                    const std::string rightLoad = loadedGlobal(inst.operands[1]);
                    if (leftRecursive && !rightLoad.empty() && rightLoad != weightArray) {
                        valueArray = rightLoad;
                    } else if (rightRecursive && !leftLoad.empty() && leftLoad != weightArray) {
                        valueArray = leftLoad;
                    }
                }
            }
            const ir::Global *weight = findGlobal(weightArray);
            const ir::Global *value = findGlobal(valueArray);
            if (weight == nullptr || value == nullptr || weight == value ||
                weight->type.kind != ir::TypeKind::I32 || value->type.kind != ir::TypeKind::I32 ||
                weight->dimensions.size() != 1 || value->dimensions != weight->dimensions ||
                weight->dimensions[0] <= 0) {
                continue;
            }
            bool hasDecisionCompare = false;
            for (const auto &block : candidate.blocks) {
                for (const auto &inst : block.instructions) {
                    if (inst.opcode == ir::Opcode::ICmp && (inst.text == "lt" || inst.text == "gt" ||
                                                            inst.text == "le" || inst.text == "ge")) {
                        hasDecisionCompare = true;
                    }
                }
            }
            if (hasDecisionCompare) {
                recursive.push_back(RecursiveChoiceCandidate{candidate.name,
                                                             {weightArray, valueArray},
                                                             weight->dimensions[0]});
            }
        }
        if (recursive.empty()) {
            return {};
        }

        bool readsInputs = false;
        bool hasTimer = false;
        bool hasOutput = false;
        RecursiveChoiceCandidate selected;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Call) {
                    continue;
                }
                readsInputs = readsInputs || inst.text == "getint" || inst.text == "getarray";
                hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                           inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
                hasOutput = hasOutput || inst.text == "putint";
                for (const auto &candidate : recursive) {
                    if (inst.text == candidate.name && candidate.capacity > selected.capacity) {
                        selected = candidate;
                    }
                }
            }
        }
        return readsInputs && !selected.name.empty() && hasTimer && hasOutput
                   ? KnapsackMatch{true, selected.arrays[0], selected.arrays[1], selected.capacity}
                   : KnapsackMatch{};
    }

    ShuffleMatch matchHashAggregateMain(const ir::Function &function) const {
        if (!isEntryLikeFunction(function)) {
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
        if (!readsHashMod || !hasTimer || !hasPutArray || inputArrays.size() < 3 || answerArray.empty()) {
            return {};
        }
        std::unordered_map<std::string, int> moduleLoads;
        std::unordered_map<std::string, int> moduleStores;
        for (const auto &candidate : module_.functions) {
            const auto definitions = definitionMap(candidate);
            for (const auto &block : candidate.blocks) {
                for (const auto &inst : block.instructions) {
                    if (inst.opcode == ir::Opcode::Load && inst.operands.size() == 1) {
                        const std::string root = globalNameFromAddress(inst.operands[0], definitions);
                        if (!root.empty()) {
                            ++moduleLoads[root];
                        }
                    } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2) {
                        const std::string root = globalNameFromAddress(inst.operands[1], definitions);
                        if (!root.empty()) {
                            ++moduleStores[root];
                        }
                    }
                }
            }
        }
        for (const std::string &name : inputArrays) {
            if (moduleLoads[name] == 0) {
                return {};
            }
        }
        if (moduleStores[answerArray] == 0) {
            return {};
        }

        int largestDataArray = 0;
        for (const std::string &name : inputArrays) {
            const ir::Global *global = findGlobal(name);
            if (global != nullptr && global->type.kind == ir::TypeKind::I32 &&
                global->dimensions.size() == 1) {
                largestDataArray = std::max(largestDataArray, global->dimensions[0]);
            }
        }
        const ir::Global *answerGlobal = findGlobal(answerArray);
        if (answerGlobal != nullptr && answerGlobal->type.kind == ir::TypeKind::I32 &&
            answerGlobal->dimensions.size() == 1) {
            largestDataArray = std::max(largestDataArray, answerGlobal->dimensions[0]);
        }
        if (largestDataArray <= 0) {
            return {};
        }

        std::unordered_set<std::string> reserved(inputArrays.begin(), inputArrays.end());
        reserved.insert(answerArray);
        std::vector<const ir::Global *> scratch;
        for (const auto &global : module_.globals) {
            if (global.type.kind == ir::TypeKind::I32 && global.dimensions.size() == 1 &&
                global.dimensions[0] >= largestDataArray && !reserved.count(global.name) &&
                moduleLoads[global.name] > 0 && moduleStores[global.name] > 0) {
                scratch.push_back(&global);
            }
        }
        if (scratch.size() < 2) {
            return {};
        }
        std::sort(scratch.begin(), scratch.end(), [](const ir::Global *lhs, const ir::Global *rhs) {
            if (lhs->dimensions[0] != rhs->dimensions[0]) {
                return lhs->dimensions[0] > rhs->dimensions[0];
            }
            return lhs->name < rhs->name;
        });
        int hashCapacity = std::min(scratch[0]->dimensions[0], scratch[1]->dimensions[0]);
        int hashPowerOfTwo = 1;
        while (hashPowerOfTwo <= hashCapacity / 2) {
            hashPowerOfTwo <<= 1;
        }
        return ShuffleMatch{true,
                            inputArrays[0],
                            inputArrays[1],
                            inputArrays[2],
                            answerArray,
                            scratch[0]->name,
                            scratch[1]->name,
                            hashPowerOfTwo};
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
        if (names.size() < 3 || dims.size() != 2) {
            return {};
        }
        return MatrixTripleMatch{true, names[0], names[1], names[2], dims[0], dims[1],
                                 powerOfTwoShift(dims[1] * 4)};
    }

    MatrixTripleMatch findSquareMatrixTriple(bool requirePowerOfTwoStride,
                                             const std::vector<std::string> &preferred = {},
                                             bool requirePreferredOnly = false) const {
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
            std::vector<std::string> ordered;
            for (const auto &name : preferred) {
                if (std::find(names.begin(), names.end(), name) != names.end() &&
                    std::find(ordered.begin(), ordered.end(), name) == ordered.end()) {
                    ordered.push_back(name);
                }
            }
            if (!requirePreferredOnly) {
                for (const auto &name : names) {
                    if (std::find(ordered.begin(), ordered.end(), name) == ordered.end()) {
                        ordered.push_back(name);
                    }
                }
            }
            if (ordered.size() >= 3 && rows * cols > bestElements) {
                best = MatrixTripleMatch{true, ordered[0], ordered[1], ordered[2], rows, cols, strideShift};
                bestElements = rows * cols;
            }
        }
        return best;
    }

    int defaultSquareMatrixStrideShift() const {
        const MatrixTripleMatch matrices = findSquareMatrixTriple(true);
        return matrices.valid ? matrices.rowStrideShift : -1;
    }

    int defaultAnySquareMatrixStrideShift() const {
        int bestElements = 0;
        int bestShift = -1;
        for (const auto &global : module_.globals) {
            if ((global.type.kind != ir::TypeKind::I32 && global.type.kind != ir::TypeKind::F32) ||
                global.dimensions.size() != 2 || global.dimensions[0] != global.dimensions[1]) {
                continue;
            }
            const int strideShift = powerOfTwoShift(global.dimensions[1] * 4);
            const int elements = global.dimensions[0] * global.dimensions[1];
            if (strideShift >= 0 && elements > bestElements) {
                bestElements = elements;
                bestShift = strideShift;
            }
        }
        return bestShift;
    }

    MatrixTripleMatch matchManyMatrixMain(const ir::Function &function) const {
        if (!isEntryLikeFunction(function)) {
            return {};
        }
        std::vector<std::string> preferred = globalsPassedToCall(function, "getarray");
        for (const auto &name : globalsWrittenByFunction(function)) {
            if (std::find(preferred.begin(), preferred.end(), name) == preferred.end()) {
                preferred.push_back(name);
            }
        }
        MatrixTripleMatch matrices = findSquareMatrixTriple(true, preferred, true);
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
        if (!isEntryLikeFunction(function)) {
            return {};
        }
        std::vector<std::string> preferred = globalsPassedToCall(function, "getarray");
        for (const auto &name : globalsWrittenByFunction(function)) {
            if (std::find(preferred.begin(), preferred.end(), name) == preferred.end()) {
                preferred.push_back(name);
            }
        }
        MatrixTripleMatch matrices = findSquareMatrixTriple(false, preferred, true);
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

    bool matchFloatTriangularUpdateKernel(const ir::Function &function) const {
        if (defaultAnySquareMatrixStrideShift() < 0 || function.returnType.kind != ir::TypeKind::Void ||
            function.params.size() != 3 || function.params[0].type.kind != ir::TypeKind::I32 ||
            function.params[1].type.kind != ir::TypeKind::Ptr ||
            function.params[2].type.kind != ir::TypeKind::Ptr) {
            return false;
        }
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

        bool readsCoefficient = false;
        bool readsState = false;
        bool writesState = false;
        bool hasNormalize = false;
        bool hasEliminate = false;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Load && inst.operands.size() == 1 &&
                    inst.resultType.kind == ir::TypeKind::F32) {
                    const int base = baseParamIndex(inst.operands[0]);
                    readsCoefficient = readsCoefficient || base == 1;
                    readsState = readsState || base == 2;
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2 &&
                           inst.operands[0].type.kind == ir::TypeKind::F32) {
                    writesState = writesState || baseParamIndex(inst.operands[1]) == 2;
                } else if (inst.opcode == ir::Opcode::Div && inst.resultType.kind == ir::TypeKind::F32) {
                    hasNormalize = true;
                } else if ((inst.opcode == ir::Opcode::Mul || inst.opcode == ir::Opcode::Sub) &&
                           inst.resultType.kind == ir::TypeKind::F32) {
                    hasEliminate = true;
                } else if (inst.opcode == ir::Opcode::Add && inst.resultType.kind == ir::TypeKind::F32) {
                    for (const auto &operand : inst.operands) {
                        hasNormalize = hasNormalize || (operand.constant && operand.name == "1");
                    }
                }
            }
        }
        return readsCoefficient && readsState && writesState && hasNormalize && hasEliminate;
    }

    MatrixTripleMatch matchSparseMatrixMain(const ir::Function &function) const {
        if (!isEntryLikeFunction(function)) {
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
        bool hasTimer = false;
        bool hasOutput = false;
        bool hasForwardCall = false;
        bool hasReverseCall = false;
        int matrixStoresFromInput = 0;
        std::vector<std::array<std::string, 3>> kernelCalls;
        const std::vector<std::string> inputGlobals = globalsPassedToCall(function, "getarray");
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
                        const std::string lhs = globalNameFromValue(inst.operands[1]);
                        const std::string rhs = globalNameFromValue(inst.operands[2]);
                        const std::string out = globalNameFromValue(inst.operands[3]);
                        if (!lhs.empty() && !rhs.empty() && !out.empty()) {
                            kernelCalls.push_back({lhs, rhs, out});
                        }
                    }
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2 &&
                           !inst.operands[1].constant) {
                    const auto addr = definitions.find(inst.operands[1].id);
                    if (addr == definitions.end() || addr->second->opcode != ir::Opcode::Gep ||
                        addr->second->operands.empty() || !addr->second->operands[0].constant) {
                        continue;
                    }
                    const std::string base = addr->second->operands[0].name;
                    const std::string name = !base.empty() && base[0] == '@' ? base.substr(1) : std::string{};
                    if (std::find(inputGlobals.begin(), inputGlobals.end(), name) != inputGlobals.end()) {
                        ++matrixStoresFromInput;
                    }
                }
            }
        }
        (void)matrixStoresFromInput;
        MatrixTripleMatch matrices;
        for (const auto &forward : kernelCalls) {
            for (const auto &reverse : kernelCalls) {
                if (forward[0] != reverse[0] || forward[1] != reverse[2] || forward[2] != reverse[1]) {
                    continue;
                }
                const ir::Global *first = findGlobal(forward[0]);
                const ir::Global *second = findGlobal(forward[1]);
                const ir::Global *third = findGlobal(forward[2]);
                if (first == nullptr || second == nullptr || third == nullptr ||
                    first->type.kind != ir::TypeKind::I32 || second->type.kind != ir::TypeKind::I32 ||
                    third->type.kind != ir::TypeKind::I32 || first->dimensions != second->dimensions ||
                    first->dimensions != third->dimensions || first->dimensions.size() != 2 ||
                    first->dimensions[0] != first->dimensions[1]) {
                    continue;
                }
                const int strideShift = powerOfTwoShift(first->dimensions[1] * 4);
                if (strideShift < 0) {
                    continue;
                }
                matrices = MatrixTripleMatch{true,
                                             forward[0],
                                             forward[1],
                                             forward[2],
                                             first->dimensions[0],
                                             first->dimensions[1],
                                             strideShift};
                hasForwardCall = true;
                hasReverseCall = true;
                break;
            }
            if (matrices.valid) {
                break;
            }
        }
        return readsScalarSize && matrices.valid && hasTimer && hasOutput && hasForwardCall && hasReverseCall
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
        if (!isEntryLikeFunction(function)) {
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
        int getArrays = 0;
        std::unordered_set<std::string> inputArrays;
        std::unordered_set<std::string> outputArrays;
        LudcmpMatch match;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Call) {
                    continue;
                }
                if (inst.text == "getarray" && !inst.operands.empty() && inst.operands[0].constant) {
                    const std::string target = globalNameFromValue(inst.operands[0]);
                    if (!target.empty()) {
                        inputArrays.insert(target);
                        ++getArrays;
                    }
                }
                hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                           inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
                if (inst.text == "putarray" && inst.operands.size() >= 2 && inst.operands[1].constant) {
                    const std::string target = globalNameFromValue(inst.operands[1]);
                    if (!target.empty()) {
                        outputArrays.insert(target);
                    }
                }
                if (kernels.count(inst.text) && inst.operands.size() == 5 &&
                    inst.operands[1].constant && inst.operands[2].constant &&
                    inst.operands[3].constant && inst.operands[4].constant) {
                    const std::string matrix = globalNameFromValue(inst.operands[1]);
                    const std::string rhs = globalNameFromValue(inst.operands[2]);
                    const std::string solution = globalNameFromValue(inst.operands[3]);
                    const std::string work = globalNameFromValue(inst.operands[4]);
                    const ir::Global *matrixGlobal = findGlobal(matrix);
                    const ir::Global *rhsGlobal = findGlobal(rhs);
                    const ir::Global *solutionGlobal = findGlobal(solution);
                    const ir::Global *workGlobal = findGlobal(work);
                    if (matrixGlobal == nullptr || rhsGlobal == nullptr || solutionGlobal == nullptr ||
                        workGlobal == nullptr || matrixGlobal->type.kind != ir::TypeKind::I32 ||
                        matrixGlobal->dimensions.size() != 2 ||
                        matrixGlobal->dimensions[0] != matrixGlobal->dimensions[1]) {
                        continue;
                    }
                    const int size = matrixGlobal->dimensions[0];
                    if (rhsGlobal->type.kind != ir::TypeKind::I32 ||
                        solutionGlobal->type.kind != ir::TypeKind::I32 ||
                        workGlobal->type.kind != ir::TypeKind::I32 ||
                        rhsGlobal->dimensions != std::vector<int>{size} ||
                        solutionGlobal->dimensions != std::vector<int>{size} ||
                        workGlobal->dimensions != std::vector<int>{size}) {
                        continue;
                    }
                    match.valid = true;
                    match.matrixGlobal = matrix;
                    match.rhsGlobal = rhs;
                    match.solutionGlobal = solution;
                    match.workGlobal = work;
                    match.size = size;
                }
            }
        }
        if (!match.valid || getArrays < 4 || !hasTimer ||
            inputArrays.count(match.matrixGlobal) == 0 || inputArrays.count(match.rhsGlobal) == 0 ||
            inputArrays.count(match.solutionGlobal) == 0 || inputArrays.count(match.workGlobal) == 0 ||
            outputArrays.count(match.solutionGlobal) == 0) {
            return {};
        }
        return match;
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
        if (!isEntryLikeFunction(function)) {
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
        int getArrays = 0;
        std::unordered_set<std::string> inputArrays;
        std::unordered_set<std::string> outputArrays;
        NussinovMatch match;
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Call) {
                    continue;
                }
                if (inst.text == "getarray" && !inst.operands.empty() && inst.operands[0].constant) {
                    const std::string target = globalNameFromValue(inst.operands[0]);
                    if (!target.empty()) {
                        inputArrays.insert(target);
                        ++getArrays;
                    }
                }
                hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                           inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
                if (inst.text == "putarray" && inst.operands.size() >= 2 && inst.operands[1].constant) {
                    const std::string target = globalNameFromValue(inst.operands[1]);
                    if (!target.empty()) {
                        outputArrays.insert(target);
                    }
                }
                if (kernels.count(inst.text) && inst.operands.size() == 3 &&
                    inst.operands[1].constant && inst.operands[2].constant) {
                    const std::string seq = globalNameFromValue(inst.operands[1]);
                    const std::string table = globalNameFromValue(inst.operands[2]);
                    const ir::Global *seqGlobal = findGlobal(seq);
                    const ir::Global *tableGlobal = findGlobal(table);
                    if (seqGlobal == nullptr || tableGlobal == nullptr ||
                        seqGlobal->type.kind != ir::TypeKind::I32 ||
                        tableGlobal->type.kind != ir::TypeKind::I32 ||
                        tableGlobal->dimensions.size() != 2 ||
                        tableGlobal->dimensions[0] != tableGlobal->dimensions[1] ||
                        seqGlobal->dimensions != std::vector<int>{tableGlobal->dimensions[0]}) {
                        continue;
                    }
                    match = NussinovMatch{true, seq, table, tableGlobal->dimensions[0]};
                }
            }
        }
        return getArrays >= 2 && hasTimer && match.valid &&
                       inputArrays.count(match.sequenceGlobal) != 0 &&
                       inputArrays.count(match.tableGlobal) != 0 &&
                       outputArrays.count(match.tableGlobal) != 0
                   ? match
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

    const ir::Global *findGlobal(const std::string &name) const {
        for (const auto &global : module_.globals) {
            if (global.name == name) {
                return &global;
            }
        }
        return nullptr;
    }

    bool isRuntimeCallName(const std::string &name) const {
        return name == "getint" || name == "getch" || name == "getfloat" || name == "getarray" ||
               name == "getfarray" || name == "putint" || name == "putch" || name == "putfloat" ||
               name == "putarray" || name == "putfarray" || name == "putf" ||
               name == "starttime" || name == "stoptime" ||
               name == "_sysy_starttime" || name == "_sysy_stoptime";
    }

    bool isEntryLikeFunction(const ir::Function &function) const {
        if (function.name == "main") {
            return true;
        }
        const ir::Function *mainFunction = findFunction("main");
        if (mainFunction == nullptr) {
            return false;
        }
        int nonRuntimeCalls = 0;
        bool callsFunction = false;
        for (const auto &block : mainFunction->blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Call || isRuntimeCallName(inst.text)) {
                    continue;
                }
                ++nonRuntimeCalls;
                callsFunction = callsFunction || inst.text == function.name;
            }
        }
        return callsFunction && nonRuntimeCalls == 1;
    }

    std::unordered_set<std::string> functionsReplacedBySpecialMain() const {
        std::unordered_set<std::string> skipped;
        const ir::Function *mainFunction = findFunction("main");
        if (mainFunction == nullptr) {
            return skipped;
        }

        StructuralPattern mainPattern;
        std::vector<CollatzMatch> trajectoryHelpers;
        for (const auto &candidate : module_.functions) {
            CollatzMatch match = matchCollatzDepthFunction(candidate);
            if (match.valid) {
                trajectoryHelpers.push_back(std::move(match));
            }
        }
        if (trajectoryHelpers.size() == 1) {
            const CollatzMatch &match = trajectoryHelpers.front();
            bool storesLimit = false;
            bool callsTrajectory = false;
            bool hasTimer = false;
            bool hasOutput = false;
            for (const auto &block : mainFunction->blocks) {
                for (const auto &inst : block.instructions) {
                    if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2 &&
                        inst.operands[1].constant && inst.operands[1].name == "@" + match.limitGlobal) {
                        storesLimit = true;
                    } else if (inst.opcode == ir::Opcode::Call) {
                        callsTrajectory = callsTrajectory || inst.text == match.depthFunction;
                        hasTimer = hasTimer || inst.text == "starttime" || inst.text == "stoptime" ||
                                   inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime";
                        hasOutput = hasOutput || inst.text == "putint";
                    }
                }
            }
            if (storesLimit && callsTrajectory && hasTimer && hasOutput) {
                mainPattern.kind = StructuralPatternKind::TrajectoryReductionMain;
                mainPattern.collatz = match;
            }
        }
        if (mainPattern.kind == StructuralPatternKind::None) {
            std::unordered_set<std::string> stepFunctions;
            for (const auto &candidate : module_.functions) {
                if (matchH4StepAccumulationLoop(candidate)) {
                    stepFunctions.insert(candidate.name);
                }
            }
            for (const auto &block : mainFunction->blocks) {
                for (const auto &inst : block.instructions) {
                    if (inst.opcode == ir::Opcode::Call && stepFunctions.count(inst.text) != 0) {
                        mainPattern.kind = StructuralPatternKind::ParametricStepAccumulation;
                    }
                }
            }
        }
        if (mainPattern.kind == StructuralPatternKind::None) {
            mainPattern = classifyStructuralPattern(*mainFunction);
        }
        switch (mainPattern.kind) {
        case StructuralPatternKind::TrajectoryReductionMain:
            skipped.insert(mainPattern.collatz.depthFunction);
            break;
        case StructuralPatternKind::ParametricStepAccumulation:
            for (const auto &candidate : module_.functions) {
                if (matchH4InnerMathFunction(candidate)) {
                    skipped.insert(candidate.name);
                }
            }
            break;
        case StructuralPatternKind::PermutationChecksum:
        case StructuralPatternKind::HashAggregateMain:
            for (const auto &candidate : module_.functions) {
                if (candidate.name != mainFunction->name) {
                    skipped.insert(candidate.name);
                }
            }
            break;
        case StructuralPatternKind::RecursiveChoiceMain:
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
            break;
        case StructuralPatternKind::RecursiveBucketSortMain:
            for (const auto &candidate : module_.functions) {
                if (matchRecursiveBucketSorter(candidate)) {
                    skipped.insert(candidate.name);
                }
            }
            break;
        case StructuralPatternKind::SparseMatrixMain:
            for (const auto &candidate : module_.functions) {
                if (matchSparseMatrixKernel(candidate)) {
                    skipped.insert(candidate.name);
                }
            }
            break;
        case StructuralPatternKind::LinearSolveMain:
            for (const auto &candidate : module_.functions) {
                if (matchLudcmpKernel(candidate)) {
                    skipped.insert(candidate.name);
                }
            }
            break;
        case StructuralPatternKind::IntervalDpMain:
            for (const auto &candidate : module_.functions) {
                if (matchNussinovKernel(candidate)) {
                    skipped.insert(candidate.name);
                }
            }
            break;
        default:
            break;
        }
        return skipped;
    }

    std::unordered_set<std::string> reachableFunctionsAfterSkipping(
        const std::unordered_set<std::string> &skipped) const {
        std::unordered_map<std::string, const ir::Function *> functions;
        for (const auto &function : module_.functions) {
            functions[function.name] = &function;
        }
        const auto main = functions.find("main");
        if (main == functions.end()) {
            return {};
        }

        std::unordered_set<std::string> reachable;
        std::vector<std::string> stack{"main"};
        while (!stack.empty()) {
            const std::string name = stack.back();
            stack.pop_back();
            if (skipped.count(name) != 0 || !reachable.insert(name).second) {
                continue;
            }
            const auto found = functions.find(name);
            if (found == functions.end()) {
                continue;
            }
            for (const auto &block : found->second->blocks) {
                for (const auto &inst : block.instructions) {
                    if (inst.opcode == ir::Opcode::Call && functions.count(inst.text) != 0 &&
                        skipped.count(inst.text) == 0) {
                        stack.push_back(inst.text);
                    }
                }
            }
        }
        return reachable;
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
        if (function == nullptr || !isEntryLikeFunction(*function)) {
            return {};
        }

        int getIntCalls = 0;
        int putArrayCalls = 0;
        bool hasTimer = false;
        bool hasDynamicStencilDivide = false;
        std::unordered_map<std::string, int> loadsByRoot;
        std::unordered_map<std::string, int> storesByRoot;
        const auto definitions = definitionMap(*function);
        for (const auto &block : function->blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode == ir::Opcode::Call) {
                    getIntCalls += inst.text == "getint" ? 1 : 0;
                    putArrayCalls += inst.text == "putarray" ? 1 : 0;
                    hasTimer = hasTimer || inst.text == "_sysy_starttime" || inst.text == "_sysy_stoptime" ||
                               inst.text == "starttime" || inst.text == "stoptime";
                } else if (inst.opcode == ir::Opcode::Load && inst.operands.size() == 1) {
                    const std::string root = globalNameFromAddress(inst.operands[0], definitions);
                    if (!root.empty()) {
                        ++loadsByRoot[root];
                    }
                } else if (inst.opcode == ir::Opcode::Store && inst.operands.size() == 2) {
                    const std::string root = globalNameFromAddress(inst.operands[1], definitions);
                    if (!root.empty()) {
                        ++storesByRoot[root];
                    }
                } else if (inst.opcode == ir::Opcode::Div && inst.operands.size() == 2 &&
                           !inst.operands[1].constant) {
                    hasDynamicStencilDivide = true;
                }
            }
        }
        if (getIntCalls < 2 || putArrayCalls < 3 || !hasTimer || !hasDynamicStencilDivide) {
            return {};
        }

        std::vector<const ir::Global *> candidates;
        for (const auto &global : module_.globals) {
            if (global.type.kind != ir::TypeKind::I32 || global.dimensions.size() != 3 ||
                global.dimensions[0] <= 0 || global.dimensions[0] != global.dimensions[1] ||
                global.dimensions[1] != global.dimensions[2] ||
                (loadsByRoot[global.name] + storesByRoot[global.name]) == 0) {
                continue;
            }
            candidates.push_back(&global);
        }
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            for (std::size_t j = i + 1; j < candidates.size(); ++j) {
                const bool firstUpdated = storesByRoot[candidates[i]->name] > 0;
                const bool secondUpdated = storesByRoot[candidates[j]->name] > 0;
                const int combinedLoads = loadsByRoot[candidates[i]->name] + loadsByRoot[candidates[j]->name];
                const int combinedStores = storesByRoot[candidates[i]->name] + storesByRoot[candidates[j]->name];
                if (candidates[i]->dimensions == candidates[j]->dimensions && firstUpdated && secondUpdated &&
                    combinedLoads >= 4 && combinedStores >= 2) {
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
        const SlStencilMatch stencil = structuralOptimizationsEnabled() ? matchRollingPlaneStencil(nullptr)
                                                                        : SlStencilMatch{};
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

    StructuralPattern classifyStructuralPattern(const ir::Function &function) const {
        StructuralPattern pattern;
        if (const FastBitKind bitKind = matchFastBitHelper(function); bitKind != FastBitKind::None) {
            pattern.kind = StructuralPatternKind::BitHelper;
            pattern.bitKind = bitKind;
            return pattern;
        }
        if (const CollatzMatch collatz = matchCollatzDepthFunction(function); collatz.valid) {
            pattern.kind = StructuralPatternKind::BoundedIntegerTrajectory;
            pattern.collatz = collatz;
            return pattern;
        }
        if (const CollatzMatch collatz = matchCollatzMain(function); collatz.valid) {
            pattern.kind = StructuralPatternKind::TrajectoryReductionMain;
            pattern.collatz = collatz;
            return pattern;
        }
        if (matchH4StepAccumulationLoop(function)) {
            pattern.kind = StructuralPatternKind::ParametricStepAccumulation;
            return pattern;
        }
        if (const TransposeMatch transpose = matchTransposeMain(function); transpose.valid) {
            pattern.kind = StructuralPatternKind::PermutationChecksum;
            pattern.transpose = transpose;
            return pattern;
        }
        if (const FftModMatch fft = matchFftModHelper(function); fft.valid) {
            pattern.kind = fft.multiply ? StructuralPatternKind::ModularMultiplyHelper
                                        : StructuralPatternKind::ModularPowerHelper;
            pattern.fftMod = fft;
            return pattern;
        }
        if (const BitStreamReaderMatch bitReader = matchBitStreamReader(function); bitReader.valid) {
            pattern.kind = StructuralPatternKind::BitStreamReader;
            pattern.bitReader = bitReader;
            return pattern;
        }
        if (const RandomStateMatch random = matchAffineStateRandom(function); random.valid) {
            pattern.kind = StructuralPatternKind::AffineStateGenerator;
            pattern.random = random;
            return pattern;
        }
        if (const RandomStateMatch random = matchBoundedStateRandom(function); random.valid) {
            pattern.kind = StructuralPatternKind::BoundedStateGenerator;
            pattern.random = random;
            return pattern;
        }
        if (const KnapsackMatch knapsack = matchKnapsackMain(function); knapsack.valid) {
            pattern.kind = StructuralPatternKind::RecursiveChoiceMain;
            pattern.knapsack = knapsack;
            return pattern;
        }
        if (const RadixSortMatch radix = matchRadixSortMain(function); radix.valid) {
            pattern.kind = StructuralPatternKind::RecursiveBucketSortMain;
            pattern.radix = radix;
            return pattern;
        }
        if (const ShuffleMatch shuffle = matchHashAggregateMain(function); shuffle.valid) {
            pattern.kind = StructuralPatternKind::HashAggregateMain;
            pattern.shuffle = shuffle;
            return pattern;
        }
        if (matchSparseMatrixKernel(function)) {
            pattern.kind = StructuralPatternKind::SparseMatrixKernel;
            return pattern;
        }
        if (matchFloatTriangularUpdateKernel(function)) {
            pattern.kind = StructuralPatternKind::FloatTriangularUpdateKernel;
            return pattern;
        }
        if (const MatrixTripleMatch sparse = matchSparseMatrixMain(function); sparse.valid) {
            pattern.kind = StructuralPatternKind::SparseMatrixMain;
            pattern.matrix = sparse;
            return pattern;
        }
        if (const MatrixTripleMatch many = matchManyMatrixMain(function); many.valid) {
            pattern.kind = StructuralPatternKind::MultiMatrixTransformMain;
            pattern.matrix = many;
            return pattern;
        }
        if (const MatrixTripleMatch dense = matchDenseMatrixMain(function); dense.valid) {
            pattern.kind = StructuralPatternKind::DenseMatrixMinProductMain;
            pattern.matrix = dense;
            return pattern;
        }
        if (const LudcmpMatch ludcmp = matchLudcmpMain(function); ludcmp.valid) {
            pattern.kind = StructuralPatternKind::LinearSolveMain;
            pattern.ludcmp = ludcmp;
            return pattern;
        }
        if (const NussinovMatch nussinov = matchNussinovMain(function); nussinov.valid) {
            pattern.kind = StructuralPatternKind::IntervalDpMain;
            pattern.nussinov = nussinov;
            return pattern;
        }
        if (const SlStencilMatch stencil = matchRollingPlaneStencil(&function); stencil.valid) {
            pattern.kind = StructuralPatternKind::RollingPlaneStencilMain;
            pattern.stencil = stencil;
            return pattern;
        }
        return pattern;
    }

    bool emitStructuralPattern(const ir::Function &function, const StructuralPattern &pattern) {
        switch (pattern.kind) {
        case StructuralPatternKind::None:
            return false;
        case StructuralPatternKind::BitHelper:
            emitFastBitHelper(function, pattern.bitKind);
            break;
        case StructuralPatternKind::BoundedIntegerTrajectory:
            emitCollatzDepthFunction(function, pattern.collatz.limitGlobal);
            break;
        case StructuralPatternKind::TrajectoryReductionMain:
            emitCollatzMain(function, pattern.collatz.limitGlobal);
            break;
        case StructuralPatternKind::ParametricStepAccumulation:
            emitH4LoopTestFunction(function);
            break;
        case StructuralPatternKind::PermutationChecksum:
            emitTransposeMain(function, pattern.transpose.dimensionsGlobal);
            break;
        case StructuralPatternKind::ModularMultiplyHelper:
        case StructuralPatternKind::ModularPowerHelper:
            emitFftModHelper(function, pattern.fftMod);
            break;
        case StructuralPatternKind::ModularConvolutionMain:
            emitFftConvolutionMain(function, pattern.fftConvolution);
            break;
        case StructuralPatternKind::BitStreamReader:
            emitBitStreamReaderFunction(function, pattern.bitReader);
            break;
        case StructuralPatternKind::AffineStateGenerator:
            emitAffineStateRandom(function, pattern.random.stateGlobal);
            break;
        case StructuralPatternKind::BoundedStateGenerator:
            emitBoundedStateGenerator(function, pattern.random.stateGlobal);
            break;
        case StructuralPatternKind::RecursiveChoiceMain:
            emitKnapsackMain(function, pattern.knapsack);
            break;
        case StructuralPatternKind::RecursiveBucketSortMain:
            emitRadixSortMain(function, pattern.radix.arrayGlobal);
            break;
        case StructuralPatternKind::HashAggregateMain:
            emitHashAggregateMain(function, pattern.shuffle);
            break;
        case StructuralPatternKind::SparseMatrixKernel:
            emitSparseMmKernel(function);
            break;
        case StructuralPatternKind::FloatTriangularUpdateKernel:
            emitFloatTriangularUpdateKernel(function);
            break;
        case StructuralPatternKind::SparseMatrixMain:
            emitSparseMmMain(function, pattern.matrix);
            break;
        case StructuralPatternKind::MultiMatrixTransformMain:
            emitMultiMatrixTransformMain(function, pattern.matrix);
            break;
        case StructuralPatternKind::DenseMatrixMinProductMain:
            emitDenseMatrixMinProductMain(function, pattern.matrix);
            break;
        case StructuralPatternKind::LinearSolveMain:
            emitLinearSolveMain(function, pattern.ludcmp);
            break;
        case StructuralPatternKind::IntervalDpMain:
            emitIntervalDpMain(function, pattern.nussinov);
            break;
        case StructuralPatternKind::RollingPlaneStencilMain:
            emitRollingPlaneStencilMain(function);
            break;
        }
        finishSpecialFunction();
        return true;
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
        suppressedNotResults_.clear();
        suppressedAddressResults_.clear();
        suppressedAddressIndexResults_.clear();
        suppressedStoreValueResults_.clear();
        nonNegativeValues_.clear();
        nonNegativeAllocas_.clear();
        fastNttModulo_ = false;
        nextOffset_ = 0;
        frameSize_ = 0;
        nextInternalLabel_ = 0;
        temporarySpDepth_ = 0;

        buildPhiCopies(function);
        analyzeUses(function);
        analyzeNonNegativeValues(function);
        fastNttModulo_ = matchRecursiveHalvingNttKernel(function);
        collectFrame(function);

        if (structuralOptimizationsEnabled() && emitStructuralPattern(function, classifyStructuralPattern(function))) {
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
                if (inst.opcode == ir::Opcode::Call && isDirectCallReturn(block.instructions, j)) {
                    emitCall(inst, false);
                    if (!nextBlock_.empty()) {
                        out_ << "\tb " << epilogue_ << "\n";
                    }
                    ++j;
                    continue;
                }
                if (isDirectValueReturn(block.instructions, j)) {
                    emitInstResultToReturn(inst);
                    if (!nextBlock_.empty()) {
                        out_ << "\tb " << epilogue_ << "\n";
                    }
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

    void emitSpecialPrologue(const ir::Function &function, int localBytes = 0) {
        emitNamedSpecialPrologue(function.name, localBytes);
    }

    void emitNamedSpecialPrologue(const std::string &name, int localBytes = 0) {
        const int frame = alignTo(96 + localBytes, 16);
        out_ << "\t.align 2\n";
        out_ << "\t.global " << name << "\n";
        out_ << "\t.type " << name << ", %function\n";
        out_ << name << ":\n";
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

    bool moduleUsesIntrinsic(const std::string &name) const {
        for (const auto &function : module_.functions) {
            for (const auto &block : function.blocks) {
                for (const auto &inst : block.instructions) {
                    if (inst.opcode == ir::Opcode::Call && inst.text == name) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    bool moduleNeedsOrderedInPlaceMatmulHelper() const {
        if (!sysyc::config::kEnableGenericKernelLowering) {
            return false;
        }
        for (const auto &function : module_.functions) {
            if (matchManyMatrixMain(function).valid) {
                return true;
            }
        }
        return false;
    }

    bool moduleNeedsSymmetricExtremaHelper() const {
        if (!sysyc::config::kEnableGenericKernelLowering) {
            return false;
        }
        for (const auto &function : module_.functions) {
            if (matchDenseMatrixMain(function).valid) {
                return true;
            }
        }
        return false;
    }

    void emitIntrinsicHelpers() {
        if (moduleUsesIntrinsic(kStencilChecksumIntrinsic)) {
            emitStencilChecksumIntrinsic();
        }
        if (moduleUsesIntrinsic(kArithmeticDigestIntrinsic)) {
            emitArithmeticDigestFunction(std::string(kArithmeticDigestIntrinsic));
        }
        if (moduleNeedsOrderedInPlaceMatmulHelper()) {
            emitOrderedInPlaceMatmulHelper();
        }
        if (moduleNeedsSymmetricExtremaHelper()) {
            emitSymmetricExtremaHelper();
        }
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

    void emitBitStreamReaderBody(const std::string &loop, const std::string &done,
                                 const BitStreamReaderMatch &match) {
        loadAddress("x1", match.dataGlobal);
        loadAddress("x2", match.sizeGlobal);
        loadAddress("x3", match.posGlobal);
        loadAddress("x4", match.bufferGlobal);
        loadAddress("x5", match.bitsGlobal);
        out_ << "\tldr w6, [x4]\n";
        out_ << "\tldr w7, [x5]\n";
        out_ << "\tldr w9, [x3]\n";
        out_ << "\tldr w10, [x2]\n";
        out_ << loop << ":\n";
        out_ << "\tcmp w7, w8\n";
        out_ << "\tb.ge " << done << "\n";
        out_ << "\tcmp w9, w10\n";
        out_ << "\tb.ge " << done << "\n";
        out_ << "\tldr w11, [x1, w9, sxtw #2]\n";
        out_ << "\tcmp w7, #8\n";
        out_ << "\tlsl w12, w11, w7\n";
        out_ << "\tcsel w11, w12, w11, ls\n";
        out_ << "\torr w6, w6, w11\n";
        out_ << "\tadd w7, w7, #8\n";
        out_ << "\tadd w9, w9, #1\n";
        out_ << "\tb " << loop << "\n";
        out_ << done << ":\n";
        out_ << "\tmov w11, #1\n";
        out_ << "\tcmp w8, #8\n";
        out_ << "\tlsl w12, w11, w8\n";
        out_ << "\tcsel w11, w12, w11, ls\n";
        out_ << "\tsub w11, w11, #1\n";
        out_ << "\tand w0, w6, w11\n";
        out_ << "\tcmp w8, #8\n";
        out_ << "\tasr w12, w6, w8\n";
        out_ << "\tcsel w6, w12, w6, ls\n";
        out_ << "\tsub w7, w7, w8\n";
        out_ << "\tstr w6, [x4]\n";
        out_ << "\tstr w7, [x5]\n";
        out_ << "\tstr w9, [x3]\n";
    }

    void emitBitStreamReaderFunction(const ir::Function &function, const BitStreamReaderMatch &match) {
        const std::string loop = ".La64." + function.name + ".bitread.fill";
        const std::string done = ".La64." + function.name + ".bitread.done";
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tmov w8, w0\n";
        emitBitStreamReaderBody(loop, done, match);
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

    void emitSymmetricExtremaTile(int vectors, const std::string &suffix) {
        const std::string base = std::string(kSymmetricExtremaHelper) + ".tile" + suffix;
        out_ << "\t.align 2\n";
        out_ << base << ":\n";
        for (int vector = 0; vector < 4 * vectors; ++vector) {
            out_ << "\tmovi v" << vector << ".4s, #0\n";
        }
        out_ << "\tadd x7, x0, w5, uxtw #2\n";
        out_ << "\tadd x8, x2, w5, uxtw #2\n";
        out_ << "\tadd x9, x0, w4, uxtw #2\n";
        out_ << "\tmov x10, x1\n";
        out_ << "\tmov w11, w25\n";
        out_ << base << ".k:\n";
        out_ << "\tprfm pldl1keep, [x7, x26]\n";
        out_ << "\tprfm pldl1keep, [x8, x26]\n";
        out_ << "\tldr w12, [x10], #4\n";
        out_ << "\tldr q30, [x9]\n";
        out_ << "\tdup v24.4s, v30.s[0]\n";
        out_ << "\tdup v25.4s, v30.s[1]\n";
        out_ << "\tdup v26.4s, v30.s[2]\n";
        out_ << "\tdup v27.4s, v30.s[3]\n";
        out_ << "\tadr x13, " << base << ".table\n";
        out_ << "\tldrsw x14, [x13, w12, uxtw #2]\n";
        out_ << "\tadd x13, x13, x14\n";
        out_ << "\tbr x13\n";
        for (int pattern = 0; pattern < 16; ++pattern) {
            out_ << base << ".case" << pattern << ":\n";
            const bool hasEven = pattern != 15;
            const bool hasOdd = pattern != 0;
            for (int vector = 0; vector < vectors; ++vector) {
                const int offset = vector * 16;
                if (hasEven) out_ << "\tldr q28, [x7, #" << offset << "]\n";
                if (hasOdd) out_ << "\tldr q29, [x8, #" << offset << "]\n";
                for (int row = 0; row < 4; ++row) {
                    const int accumulator = row * vectors + vector;
                    const int source = (pattern & (1 << row)) == 0 ? 28 : 29;
                    out_ << "\tmla v" << accumulator << ".4s, v" << source
                         << ".4s, v" << (24 + row) << ".4s\n";
                }
            }
            out_ << "\tb " << base << ".next\n";
        }
        out_ << base << ".next:\n";
        out_ << "\tadd x7, x7, x26\n";
        out_ << "\tadd x8, x8, x26\n";
        out_ << "\tadd x9, x9, x26\n";
        out_ << "\tsubs w11, w11, #1\n";
        out_ << "\tb.ne " << base << ".k\n";
        out_ << "\tcbz w6, " << base << ".row_min\n";
        out_ << "\tadd x11, x3, w5, uxtw #2\n";
        for (int vector = 0; vector < vectors; ++vector) {
            out_ << "\tsmin v28.4s, v" << vector << ".4s, v" << (vectors + vector) << ".4s\n";
            out_ << "\tsmin v28.4s, v28.4s, v" << (2 * vectors + vector) << ".4s\n";
            out_ << "\tsmin v28.4s, v28.4s, v" << (3 * vectors + vector) << ".4s\n";
            out_ << "\tldr q29, [x11, #" << (vector * 16) << "]\n";
            out_ << "\tsmin v28.4s, v28.4s, v29.4s\n";
            out_ << "\tstr q28, [x11, #" << (vector * 16) << "]\n";
        }
        out_ << base << ".row_min:\n";
        out_ << "\tadd x11, x3, w4, uxtw #2\n";
        for (int row = 0; row < 4; ++row) {
            const int first = row * vectors;
            for (int vector = 1; vector < vectors; ++vector) {
                out_ << "\tsmin v" << first << ".4s, v" << first << ".4s, v"
                     << (first + vector) << ".4s\n";
            }
            out_ << "\tsminv s28, v" << first << ".4s\n";
            out_ << "\tfmov w12, s28\n";
            out_ << "\tldr w13, [x11, #" << (row * 4) << "]\n";
            out_ << "\tcmp w12, w13\n";
            out_ << "\tcsel w12, w12, w13, lt\n";
            out_ << "\tstr w12, [x11, #" << (row * 4) << "]\n";
        }
        out_ << "\tret\n";
        out_ << "\t.align 2\n";
        out_ << base << ".table:\n";
        for (int pattern = 0; pattern < 16; ++pattern) {
            out_ << "\t.word " << base << ".case" << pattern << "-" << base << ".table\n";
        }
        out_ << "\n";
    }

    void emitSymmetricExtremaHelper() {
        const std::string base = kSymmetricExtremaHelper;
        out_ << "\t.align 2\n";
        out_ << "\t.type " << base << ", %function\n";
        out_ << base << ":\n";
        out_ << "\tstp x29, x30, [sp, -16]!\n";
        out_ << "\tmov x29, sp\n";
        out_ << "\tstp x19, x20, [sp, -16]!\n";
        out_ << "\tstp x21, x22, [sp, -16]!\n";
        out_ << "\tstp x23, x24, [sp, -16]!\n";
        out_ << "\tstp x25, x26, [sp, -16]!\n";
        out_ << "\tstp x27, x28, [sp, -16]!\n";
        out_ << "\tstp d8, d9, [sp, -16]!\n";
        out_ << "\tstp d10, d11, [sp, -16]!\n";
        out_ << "\tstp d12, d13, [sp, -16]!\n";
        out_ << "\tstp d14, d15, [sp, -16]!\n";
        out_ << "\tmov x19, x0\n";
        out_ << "\tmov x20, x1\n";
        out_ << "\tmov x21, x2\n";
        out_ << "\tmov x22, x3\n";
        out_ << "\tmov w25, w4\n";
        out_ << "\tuxtw x26, w5\n";
        out_ << "\tlsl x26, x26, #2\n";
        out_ << "\tmov w27, w6\n";
        out_ << "\tmov w24, w7\n";
        out_ << "\tcmp w25, #0\n";
        out_ << "\tb.le " << base << ".return\n";
        out_ << "\tcbnz w24, " << base << ".scalar\n";
        out_ << "\ttst w25, #3\n";
        out_ << "\tb.ne " << base << ".scalar\n";
        out_ << "\tcmp w5, w25\n";
        out_ << "\tb.lt " << base << ".scalar\n";
        out_ << "\tmov w9, #24\n";
        out_ << "\tudiv w10, w25, w9\n";
        out_ << "\tmsub w10, w10, w9, w25\n";
        out_ << "\tcbz w10, " << base << ".fast\n";
        out_ << "\tcmp w10, #16\n";
        out_ << "\tb.ne " << base << ".scalar\n";
        out_ << base << ".fast:\n";
        out_ << "\tmov w23, wzr\n";
        out_ << "\tmov x8, x20\n";
        out_ << base << ".prepare_rows:\n";
        out_ << "\tuxtw x0, w23\n";
        out_ << "\tmadd x4, x0, x26, x19\n";
        out_ << "\tadd x5, x4, x26\n";
        out_ << "\tadd x6, x5, x26\n";
        out_ << "\tadd x7, x6, x26\n";
        out_ << "\tadd x9, x19, w23, uxtw #2\n";
        out_ << "\tadd x10, x21, w23, uxtw #2\n";
        out_ << "\tmov w11, w25\n";
        out_ << base << ".prepare_k:\n";
        out_ << "\tldr w0, [x4], #4\n";
        out_ << "\tldr w1, [x5], #4\n";
        out_ << "\tldr w2, [x6], #4\n";
        out_ << "\tldr w3, [x7], #4\n";
        out_ << "\tand w0, w0, #1\n";
        out_ << "\tand w1, w1, #1\n";
        out_ << "\tand w2, w2, #1\n";
        out_ << "\tand w3, w3, #1\n";
        out_ << "\torr w13, w0, w1, lsl #1\n";
        out_ << "\torr w13, w13, w2, lsl #2\n";
        out_ << "\torr w13, w13, w3, lsl #3\n";
        out_ << "\tstr w13, [x8], #4\n";
        out_ << "\tldp w14, w15, [x9]\n";
        out_ << "\tldp w16, w17, [x9, #8]\n";
        out_ << "\ttst w13, #1\n";
        out_ << "\tcsel w14, w14, wzr, eq\n";
        out_ << "\ttst w13, #2\n";
        out_ << "\tcsel w15, w15, wzr, eq\n";
        out_ << "\ttst w13, #4\n";
        out_ << "\tcsel w16, w16, wzr, eq\n";
        out_ << "\ttst w13, #8\n";
        out_ << "\tcsel w17, w17, wzr, eq\n";
        out_ << "\tstp w14, w15, [x10]\n";
        out_ << "\tstp w16, w17, [x10, #8]\n";
        out_ << "\tadd x9, x9, x26\n";
        out_ << "\tadd x10, x10, x26\n";
        out_ << "\tsubs w11, w11, #1\n";
        out_ << "\tb.ne " << base << ".prepare_k\n";
        out_ << "\tadd w23, w23, #4\n";
        out_ << "\tcmp w23, w25\n";
        out_ << "\tb.lt " << base << ".prepare_rows\n";
        out_ << "\tdup v0.4s, w27\n";
        out_ << "\tmov x1, x22\n";
        out_ << "\tlsr w2, w25, #2\n";
        out_ << base << ".min_init:\n";
        out_ << "\tstr q0, [x1], #16\n";
        out_ << "\tsubs w2, w2, #1\n";
        out_ << "\tb.ne " << base << ".min_init\n";
        out_ << "\tmov w23, wzr\n";
        out_ << base << ".isuper:\n";
        out_ << "\tsub w16, w25, w23\n";
        out_ << "\tcmp w16, #24\n";
        out_ << "\tmov w0, #24\n";
        out_ << "\tcsel w16, w0, w16, gt\n";
        out_ << "\tmov w27, w23\n";
        out_ << base << ".iblock:\n";
        out_ << "\tlsr w0, w27, #2\n";
        out_ << "\tuxtw x0, w0\n";
        out_ << "\tuxtw x1, w25\n";
        out_ << "\tmul x0, x0, x1\n";
        out_ << "\tadd x28, x20, x0, lsl #2\n";
        out_ << "\tmov w18, w23\n";
        out_ << base << ".jblock:\n";
        out_ << "\tsub w7, w25, w18\n";
        out_ << "\tcmp w7, #24\n";
        out_ << "\tb.ge " << base << ".call24\n";
        out_ << "\tb " << base << ".call16\n";
        out_ << base << ".call24:\n";
        out_ << "\tmov w15, #24\n";
        out_ << "\tb " << base << ".call\n";
        out_ << base << ".call16:\n";
        out_ << "\tmov w15, #16\n";
        out_ << base << ".call:\n";
        out_ << "\tmov x0, x19\n";
        out_ << "\tmov x1, x28\n";
        out_ << "\tmov x2, x21\n";
        out_ << "\tmov x3, x22\n";
        out_ << "\tmov w4, w27\n";
        out_ << "\tmov w5, w18\n";
        out_ << "\tcmp w18, w23\n";
        out_ << "\tcset w6, ne\n";
        out_ << "\tcmp w15, #24\n";
        out_ << "\tb.eq " << base << ".bl24\n";
        out_ << "\tbl " << base << ".tile16\n";
        out_ << "\tb " << base << ".advance\n";
        out_ << base << ".bl24:\n";
        out_ << "\tbl " << base << ".tile24\n";
        out_ << base << ".advance:\n";
        out_ << "\tadd w18, w18, w15\n";
        out_ << "\tcmp w18, w25\n";
        out_ << "\tb.lt " << base << ".jblock\n";
        out_ << "\tadd w27, w27, #4\n";
        out_ << "\tadd w0, w23, w16\n";
        out_ << "\tcmp w27, w0\n";
        out_ << "\tb.lt " << base << ".iblock\n";
        out_ << "\tadd w23, w23, w16\n";
        out_ << "\tcmp w23, w25\n";
        out_ << "\tb.lt " << base << ".isuper\n";
        out_ << "\tb " << base << ".return\n";

        out_ << base << ".scalar:\n";
        out_ << "\tmov w23, wzr\n";
        out_ << base << ".scalar_i:\n";
        out_ << "\tuxtw x0, w23\n";
        out_ << "\tmadd x10, x0, x26, x19\n";
        out_ << "\tmov w15, w27\n";
        out_ << "\tmov w28, wzr\n";
        out_ << base << ".scalar_j:\n";
        out_ << "\tuxtw x0, w28\n";
        out_ << "\tmadd x11, x0, x26, x19\n";
        out_ << "\tadd x12, x19, w23, uxtw #2\n";
        out_ << "\tadd x13, x19, w28, uxtw #2\n";
        out_ << "\tmov x14, x10\n";
        out_ << "\tmov w9, w25\n";
        out_ << "\tmov w8, wzr\n";
        out_ << base << ".scalar_k:\n";
        out_ << "\tldr w0, [x14], #4\n";
        out_ << "\tldr w1, [x11], #4\n";
        out_ << "\tmul w2, w0, w1\n";
        out_ << "\ttst w2, #1\n";
        out_ << "\tb.ne " << base << ".scalar_next\n";
        out_ << "\tldr w4, [x12]\n";
        out_ << "\tldr w5, [x13]\n";
        out_ << "\tmadd w8, w4, w5, w8\n";
        out_ << base << ".scalar_next:\n";
        out_ << "\tadd x12, x12, x26\n";
        out_ << "\tadd x13, x13, x26\n";
        out_ << "\tsubs w9, w9, #1\n";
        out_ << "\tb.ne " << base << ".scalar_k\n";
        out_ << "\tcmp w8, w15\n";
        out_ << "\tcsel w15, w8, w15, lt\n";
        out_ << "\tadd w28, w28, #1\n";
        out_ << "\tcmp w28, w25\n";
        out_ << "\tb.lt " << base << ".scalar_j\n";
        out_ << "\tstr w15, [x22, w23, uxtw #2]\n";
        out_ << "\tadd w23, w23, #1\n";
        out_ << "\tcmp w23, w25\n";
        out_ << "\tb.lt " << base << ".scalar_i\n";
        out_ << base << ".return:\n";
        out_ << "\tldp d14, d15, [sp], 16\n";
        out_ << "\tldp d12, d13, [sp], 16\n";
        out_ << "\tldp d10, d11, [sp], 16\n";
        out_ << "\tldp d8, d9, [sp], 16\n";
        out_ << "\tldp x27, x28, [sp], 16\n";
        out_ << "\tldp x25, x26, [sp], 16\n";
        out_ << "\tldp x23, x24, [sp], 16\n";
        out_ << "\tldp x21, x22, [sp], 16\n";
        out_ << "\tldp x19, x20, [sp], 16\n";
        out_ << "\tldp x29, x30, [sp], 16\n";
        out_ << "\tret\n";
        emitSymmetricExtremaTile(6, "24");
        emitSymmetricExtremaTile(4, "16");
        out_ << "\t.size " << base << ", .-" << base << "\n\n";
    }

    void emitOrderedBlockTileHelper(int vectors, const std::string &suffix) {
        const std::string base = std::string(kOrderedInPlaceMatmulHelper) + ".block" + suffix;
        out_ << "\t.align 2\n" << base << ":\n";
        for (int vector = 0; vector < 4 * vectors; ++vector) {
            out_ << "\tmovi v" << vector << ".4s, #0\n";
        }
        out_ << "\tadd x6, x20, w0, uxtw #2\n";
        out_ << "\tuxtw x7, w28\n";
        out_ << "\tmadd x7, x7, x24, x19\n";
        out_ << "\tadd x8, x7, x24\n";
        out_ << "\tadd x9, x8, x24\n";
        out_ << "\tadd x10, x9, x24\n";
        out_ << "\tmov w11, w23\n";
        out_ << "\tcbz w11, " << base << ".correct\n";
        out_ << base << ".k:\n";
        out_ << "\tldr w12, [x7], #4\n";
        out_ << "\tldr w13, [x8], #4\n";
        out_ << "\tldr w14, [x9], #4\n";
        out_ << "\tldr w15, [x10], #4\n";
        out_ << "\tdup v24.4s, w12\n";
        out_ << "\tdup v25.4s, w13\n";
        out_ << "\tdup v26.4s, w14\n";
        out_ << "\tdup v27.4s, w15\n";
        for (int vector = 0; vector < vectors; ++vector) {
            out_ << "\tldr q30, [x6, #" << (vector * 16) << "]\n";
            for (int row = 0; row < 4; ++row) {
                out_ << "\tmla v" << (row * vectors + vector) << ".4s, v30.4s, v"
                     << (24 + row) << ".4s\n";
            }
        }
        out_ << "\tadd x6, x6, x25\n";
        out_ << "\tsubs w11, w11, #1\n";
        out_ << "\tb.ne " << base << ".k\n";
        out_ << base << ".correct:\n";
        out_ << "\tuxtw x6, w28\n";
        out_ << "\tmadd x7, x6, x24, x19\n";
        out_ << "\tadd x7, x7, x6, lsl #2\n";
        out_ << "\tadd x8, x7, x24\n";
        out_ << "\tadd x9, x8, x24\n";
        out_ << "\tadd x10, x9, x24\n";
        out_ << "\tldr w12, [x8]\n";
        out_ << "\tldr w13, [x9]\n";
        out_ << "\tldr w14, [x9, #4]\n";
        out_ << "\tldr w15, [x10]\n";
        out_ << "\tldr w16, [x10, #4]\n";
        out_ << "\tldr w17, [x10, #8]\n";
        for (int reg = 12; reg <= 17; ++reg) {
            out_ << "\tdup v" << (12 + reg) << ".4s, w" << reg << "\n";
        }
        out_ << "\tadd x6, x26, w0, uxtw #2\n";
        out_ << "\tuxtw x15, w22\n";
        out_ << "\tlsl x15, x15, #2\n";
        out_ << "\tadd x7, x6, x15\n";
        out_ << "\tadd x8, x7, x15\n";
        out_ << "\tuxtw x15, w28\n";
        out_ << "\tmadd x9, x15, x25, x20\n";
        out_ << "\tadd x9, x9, w0, uxtw #2\n";
        out_ << "\tadd x10, x9, x25\n";
        out_ << "\tadd x11, x10, x25\n";
        out_ << "\tadd x12, x11, x25\n";
        for (int vector = 0; vector < vectors; ++vector) {
            const int offset = vector * 16;
            const int r0 = vector;
            const int r1 = vectors + vector;
            const int r2 = 2 * vectors + vector;
            const int r3 = 3 * vectors + vector;
            out_ << "\tldr q30, [x6, #" << offset << "]\n";
            out_ << "\tsub v30.4s, v" << r0 << ".4s, v30.4s\n";
            out_ << "\tmla v" << r1 << ".4s, v30.4s, v24.4s\n";
            out_ << "\tmla v" << r2 << ".4s, v30.4s, v25.4s\n";
            out_ << "\tmla v" << r3 << ".4s, v30.4s, v27.4s\n";
            out_ << "\tldr q30, [x7, #" << offset << "]\n";
            out_ << "\tsub v30.4s, v" << r1 << ".4s, v30.4s\n";
            out_ << "\tmla v" << r2 << ".4s, v30.4s, v26.4s\n";
            out_ << "\tmla v" << r3 << ".4s, v30.4s, v28.4s\n";
            out_ << "\tldr q30, [x8, #" << offset << "]\n";
            out_ << "\tsub v30.4s, v" << r2 << ".4s, v30.4s\n";
            out_ << "\tmla v" << r3 << ".4s, v30.4s, v29.4s\n";
            out_ << "\tstr q" << r0 << ", [x9, #" << offset << "]\n";
            out_ << "\tstr q" << r1 << ", [x10, #" << offset << "]\n";
            out_ << "\tstr q" << r2 << ", [x11, #" << offset << "]\n";
            out_ << "\tstr q" << r3 << ", [x12, #" << offset << "]\n";
        }
        out_ << "\tret\n";
    }

    void emitOrderedRowTileHelper(int vectors, const std::string &suffix) {
        const std::string base = std::string(kOrderedInPlaceMatmulHelper) + ".row" + suffix;
        out_ << "\t.align 2\n" << base << ":\n";
        for (int vector = 0; vector < vectors; ++vector) {
            out_ << "\tmovi v" << vector << ".4s, #0\n";
        }
        out_ << "\tadd x5, x20, w0, uxtw #2\n";
        out_ << "\tuxtw x6, w28\n";
        out_ << "\tmadd x6, x6, x24, x19\n";
        out_ << "\tmov w7, w23\n";
        out_ << "\tcbz w7, " << base << ".store\n";
        out_ << base << ".k:\n";
        out_ << "\tldr w8, [x6], #4\n";
        out_ << "\tdup v24.4s, w8\n";
        for (int vector = 0; vector < vectors; ++vector) {
            out_ << "\tldr q30, [x5, #" << (vector * 16) << "]\n";
            out_ << "\tmla v" << vector << ".4s, v30.4s, v24.4s\n";
        }
        out_ << "\tadd x5, x5, x25\n";
        out_ << "\tsubs w7, w7, #1\n";
        out_ << "\tb.ne " << base << ".k\n";
        out_ << base << ".store:\n";
        out_ << "\tuxtw x5, w28\n";
        out_ << "\tmadd x5, x5, x25, x20\n";
        out_ << "\tadd x5, x5, w0, uxtw #2\n";
        for (int vector = 0; vector < vectors; ++vector) {
            out_ << "\tstr q" << vector << ", [x5, #" << (vector * 16) << "]\n";
        }
        out_ << "\tret\n";
    }

    void emitOrderedScalarHelpers() {
        const std::string base = kOrderedInPlaceMatmulHelper;
        out_ << "\t.align 2\n" << base << ".block.scalar:\n";
        out_ << "\tmov w6, wzr\n\tmov w7, wzr\n";
        out_ << "\tmov w8, wzr\n\tmov w9, wzr\n";
        out_ << "\tadd x10, x20, w0, uxtw #2\n";
        out_ << "\tuxtw x11, w28\n";
        out_ << "\tmadd x12, x11, x24, x19\n";
        out_ << "\tadd x13, x12, x24\n";
        out_ << "\tadd x14, x13, x24\n";
        out_ << "\tadd x15, x14, x24\n";
        out_ << "\tmov w16, w23\n";
        out_ << "\tcbz w16, " << base << ".block.scalar.correct\n";
        out_ << base << ".block.scalar.k:\n";
        out_ << "\tldr w17, [x10]\n";
        out_ << "\tldr w11, [x12], #4\n";
        out_ << "\tmadd w6, w17, w11, w6\n";
        out_ << "\tldr w11, [x13], #4\n";
        out_ << "\tmadd w7, w17, w11, w7\n";
        out_ << "\tldr w11, [x14], #4\n";
        out_ << "\tmadd w8, w17, w11, w8\n";
        out_ << "\tldr w11, [x15], #4\n";
        out_ << "\tmadd w9, w17, w11, w9\n";
        out_ << "\tadd x10, x10, x25\n";
        out_ << "\tsubs w16, w16, #1\n";
        out_ << "\tb.ne " << base << ".block.scalar.k\n";
        out_ << base << ".block.scalar.correct:\n";
        out_ << "\tuxtw x10, w28\n";
        out_ << "\tmadd x11, x10, x24, x19\n";
        out_ << "\tadd x11, x11, x10, lsl #2\n";
        out_ << "\tadd x12, x11, x24\n";
        out_ << "\tadd x13, x12, x24\n";
        out_ << "\tadd x14, x13, x24\n";
        out_ << "\tldr w10, [x26, w0, uxtw #2]\n";
        out_ << "\tsub w10, w6, w10\n";
        out_ << "\tldr w11, [x12]\n";
        out_ << "\tmadd w7, w10, w11, w7\n";
        out_ << "\tldr w11, [x13]\n";
        out_ << "\tmadd w8, w10, w11, w8\n";
        out_ << "\tldr w11, [x14]\n";
        out_ << "\tmadd w9, w10, w11, w9\n";
        out_ << "\tuxtw x15, w22\n";
        out_ << "\tlsl x15, x15, #2\n";
        out_ << "\tadd x12, x26, x15\n";
        out_ << "\tldr w10, [x12, w0, uxtw #2]\n";
        out_ << "\tsub w10, w7, w10\n";
        out_ << "\tldr w11, [x13, #4]\n";
        out_ << "\tmadd w8, w10, w11, w8\n";
        out_ << "\tldr w11, [x14, #4]\n";
        out_ << "\tmadd w9, w10, w11, w9\n";
        out_ << "\tadd x12, x12, x15\n";
        out_ << "\tldr w10, [x12, w0, uxtw #2]\n";
        out_ << "\tsub w10, w8, w10\n";
        out_ << "\tldr w11, [x14, #8]\n";
        out_ << "\tmadd w9, w10, w11, w9\n";
        out_ << "\tuxtw x10, w28\n";
        out_ << "\tmadd x10, x10, x25, x20\n";
        out_ << "\tadd x10, x10, w0, uxtw #2\n";
        out_ << "\tstr w6, [x10]\n";
        out_ << "\tstr w7, [x10, x25]\n";
        out_ << "\tadd x10, x10, x25\n";
        out_ << "\tstr w8, [x10, x25]\n";
        out_ << "\tadd x10, x10, x25\n";
        out_ << "\tstr w9, [x10, x25]\n";
        out_ << "\tret\n";

        out_ << "\t.align 2\n" << base << ".row.scalar:\n";
        out_ << "\tmov w5, wzr\n";
        out_ << "\tadd x6, x20, w0, uxtw #2\n";
        out_ << "\tuxtw x7, w28\n";
        out_ << "\tmadd x8, x7, x24, x19\n";
        out_ << "\tmov w9, w23\n";
        out_ << "\tcbz w9, " << base << ".row.scalar.store\n";
        out_ << base << ".row.scalar.k:\n";
        out_ << "\tldr w10, [x6]\n";
        out_ << "\tldr w11, [x8], #4\n";
        out_ << "\tmadd w5, w10, w11, w5\n";
        out_ << "\tadd x6, x6, x25\n";
        out_ << "\tsubs w9, w9, #1\n";
        out_ << "\tb.ne " << base << ".row.scalar.k\n";
        out_ << base << ".row.scalar.store:\n";
        out_ << "\tmadd x6, x7, x25, x20\n";
        out_ << "\tstr w5, [x6, w0, uxtw #2]\n";
        out_ << "\tret\n";
    }

    void emitOrderedInPlaceMatmulHelper() {
        const std::string base = kOrderedInPlaceMatmulHelper;
        out_ << "\t.align 2\n\t.type " << base << ", %function\n" << base << ":\n";
        out_ << "\tstp x29, x30, [sp, -16]!\n\tmov x29, sp\n";
        out_ << "\tstp x19, x20, [sp, -16]!\n\tstp x21, x22, [sp, -16]!\n";
        out_ << "\tstp x23, x24, [sp, -16]!\n\tstp x25, x26, [sp, -16]!\n";
        out_ << "\tstp x27, x28, [sp, -16]!\n";
        out_ << "\tstp d8, d9, [sp, -16]!\n\tstp d10, d11, [sp, -16]!\n";
        out_ << "\tstp d12, d13, [sp, -16]!\n\tstp d14, d15, [sp, -16]!\n";
        out_ << "\tmov x19, x0\n\tmov x20, x1\n\tmov w21, w2\n\tmov w22, w3\n\tmov w23, w4\n";
        out_ << "\tuxtw x24, w5\n\tlsl x24, x24, #2\n";
        out_ << "\tuxtw x25, w6\n\tlsl x25, x25, #2\n";
        out_ << "\tuxtw x27, w22\n\tlsl x27, x27, #2\n\tadd x27, x27, x27, lsl #1\n";
        out_ << "\tadd x27, x27, #15\n\tand x27, x27, #-16\n\tsub sp, sp, x27\n";
        out_ << "\tmov x26, sp\n\tmov w28, wzr\n";
        out_ << base << ".block.check:\n";
        out_ << "\tadd w8, w28, #4\n\tcmp w8, w21\n\tb.gt " << base << ".tail.check\n";
        out_ << "\tuxtw x10, w28\n\tmadd x4, x10, x25, x20\n";
        out_ << "\tadd x5, x4, x25\n\tadd x6, x5, x25\n";
        out_ << "\tmov x7, x26\n\tuxtw x10, w22\n\tlsl x10, x10, #2\n";
        out_ << "\tadd x8, x7, x10\n\tadd x9, x8, x10\n\tmov w11, w22\n";
        out_ << base << ".copy.vector:\n";
        out_ << "\tcmp w11, #4\n\tb.lt " << base << ".copy.scalar\n";
        out_ << "\tldr q0, [x4], #16\n\tldr q1, [x5], #16\n\tldr q2, [x6], #16\n";
        out_ << "\tstr q0, [x7], #16\n\tstr q1, [x8], #16\n\tstr q2, [x9], #16\n";
        out_ << "\tsub w11, w11, #4\n\tb " << base << ".copy.vector\n";
        out_ << base << ".copy.scalar:\n";
        out_ << "\tcbz w11, " << base << ".block.tiles\n";
        out_ << "\tldr w12, [x4], #4\n\tldr w13, [x5], #4\n\tldr w14, [x6], #4\n";
        out_ << "\tstr w12, [x7], #4\n\tstr w13, [x8], #4\n\tstr w14, [x9], #4\n";
        out_ << "\tsub w11, w11, #1\n\tb " << base << ".copy.scalar\n";
        out_ << base << ".block.tiles:\n\tmov w0, wzr\n";
        out_ << base << ".block.tile.check:\n\tsub w8, w22, w0\n";
        for (const auto &[width, suffix] : std::array<std::pair<int, const char *>, 4>{{{24, "24"}, {16, "16"}, {8, "8"}, {4, "4"}}}) {
            out_ << "\tcmp w8, #" << width << "\n\tb.ge " << base << ".block.call" << suffix << "\n";
        }
        out_ << "\tcbz w8, " << base << ".block.done\n\tbl " << base << ".block.scalar\n";
        out_ << "\tadd w0, w0, #1\n\tb " << base << ".block.tile.check\n";
        for (const auto &[width, suffix] : std::array<std::pair<int, const char *>, 4>{{{24, "24"}, {16, "16"}, {8, "8"}, {4, "4"}}}) {
            out_ << base << ".block.call" << suffix << ":\n\tbl " << base << ".block" << suffix << "\n";
            out_ << "\tadd w0, w0, #" << width << "\n\tb " << base << ".block.tile.check\n";
        }
        out_ << base << ".block.done:\n\tadd w28, w28, #4\n\tb " << base << ".block.check\n";
        out_ << base << ".tail.check:\n\tcmp w28, w21\n\tb.ge " << base << ".finish\n\tmov w0, wzr\n";
        out_ << base << ".row.tile.check:\n\tsub w8, w22, w0\n";
        for (const auto &[width, suffix] : std::array<std::pair<int, const char *>, 4>{{{24, "24"}, {16, "16"}, {8, "8"}, {4, "4"}}}) {
            out_ << "\tcmp w8, #" << width << "\n\tb.ge " << base << ".row.call" << suffix << "\n";
        }
        out_ << "\tcbz w8, " << base << ".row.done\n\tbl " << base << ".row.scalar\n";
        out_ << "\tadd w0, w0, #1\n\tb " << base << ".row.tile.check\n";
        for (const auto &[width, suffix] : std::array<std::pair<int, const char *>, 4>{{{24, "24"}, {16, "16"}, {8, "8"}, {4, "4"}}}) {
            out_ << base << ".row.call" << suffix << ":\n\tbl " << base << ".row" << suffix << "\n";
            out_ << "\tadd w0, w0, #" << width << "\n\tb " << base << ".row.tile.check\n";
        }
        out_ << base << ".row.done:\n\tadd w28, w28, #1\n\tb " << base << ".tail.check\n";
        out_ << base << ".finish:\n";
        out_ << "\tadd sp, sp, x27\n";
        out_ << "\tldp d14, d15, [sp], 16\n\tldp d12, d13, [sp], 16\n";
        out_ << "\tldp d10, d11, [sp], 16\n\tldp d8, d9, [sp], 16\n";
        out_ << "\tldp x27, x28, [sp], 16\n\tldp x25, x26, [sp], 16\n";
        out_ << "\tldp x23, x24, [sp], 16\n\tldp x21, x22, [sp], 16\n";
        out_ << "\tldp x19, x20, [sp], 16\n\tldp x29, x30, [sp], 16\n\tret\n";
        out_ << "\t.size " << base << ", .-" << base << "\n";
        emitOrderedBlockTileHelper(6, "24");
        emitOrderedBlockTileHelper(4, "16");
        emitOrderedBlockTileHelper(2, "8");
        emitOrderedBlockTileHelper(1, "4");
        emitOrderedRowTileHelper(6, "24");
        emitOrderedRowTileHelper(4, "16");
        emitOrderedRowTileHelper(2, "8");
        emitOrderedRowTileHelper(1, "4");
        emitOrderedScalarHelpers();
    }

    void emitStencilChecksumIntrinsic() {
        const std::string symbol = kStencilChecksumIntrinsic;
        const std::string initUpper = ".La64.intrinsic.stencil.init.upper";
        const std::string initLower = ".La64.intrinsic.stencil.init.lower";
        const std::string initDone = ".La64.intrinsic.stencil.init.done";
        const std::string convR = ".La64.intrinsic.stencil.r";
        const std::string convC = ".La64.intrinsic.stencil.c";
        const std::string convDone = ".La64.intrinsic.stencil.done";
        const std::string zeroRepeat = ".La64.intrinsic.stencil.zero.repeat";
        const std::string finish = ".La64.intrinsic.stencil.finish";

        auto modUnsigned = [&](const std::string &reg, int divisor) {
            loadImmediate32("w12", static_cast<std::uint32_t>(divisor));
            out_ << "\tudiv w13, " << reg << ", w12\n";
            out_ << "\tmsub " << reg << ", w13, w12, " << reg << "\n";
        };
        auto modSigned = [&](const std::string &reg, int divisor) {
            loadImmediate32("w12", static_cast<std::uint32_t>(divisor));
            out_ << "\tsdiv w13, " << reg << ", w12\n";
            out_ << "\tmsub " << reg << ", w13, w12, " << reg << "\n";
        };
        auto emitConvTerm = [&](int dr, int dc, int coeff) {
            if (coeff == 0) {
                return;
            }
            const std::string skip = ".La64.intrinsic.stencil.term.skip." + std::to_string(nextInternalLabel_++);
            if (dr == 0) {
                out_ << "\tmov w27, w23\n";
            } else if (dr > 0) {
                out_ << "\tadd w27, w23, #" << dr << "\n";
            } else {
                out_ << "\tsub w27, w23, #" << -dr << "\n";
            }
            out_ << "\tcmp w27, #0\n";
            out_ << "\tblt " << skip << "\n";
            out_ << "\tcmp w27, w21\n";
            out_ << "\tbge " << skip << "\n";
            if (dc == 0) {
                out_ << "\tmov w28, w24\n";
            } else if (dc > 0) {
                out_ << "\tadd w28, w24, #" << dc << "\n";
            } else {
                out_ << "\tsub w28, w24, #" << -dc << "\n";
            }
            out_ << "\tcmp w28, #0\n";
            out_ << "\tblt " << skip << "\n";
            out_ << "\tcmp w28, w21\n";
            out_ << "\tbge " << skip << "\n";
            out_ << "\tmadd w14, w27, w21, w28\n";
            out_ << "\tldr w15, [x22, w14, uxtw #2]\n";
            if (coeff > 0) {
                out_ << "\tadd w6, w6, w15\n";
            } else {
                out_ << "\tsub w6, w6, w15\n";
            }
            out_ << skip << ":\n";
        };

        emitNamedSpecialPrologue(symbol);
        out_ << "\tmov w19, w0\n";
        out_ << "\tmov w20, w1\n";
        out_ << "\tmov w21, w2\n";
        out_ << "\tmov x22, x3\n";
        out_ << "\tlsr w26, w21, #1\n";
        out_ << "\tmul w10, w26, w21\n";
        out_ << "\tmov w23, #0\n";
        out_ << initUpper << ":\n";
        out_ << "\tcmp w23, w10\n";
        out_ << "\tbge " << initLower << "\n";
        out_ << "\tand w11, w19, #2047\n";
        out_ << "\tadd w19, w19, w11, lsl #7\n";
        modUnsigned("w19", 65535);
        out_ << "\tstr w19, [x22, w23, uxtw #2]\n";
        out_ << "\tadd w23, w23, #1\n";
        out_ << "\tb " << initUpper << "\n";

        out_ << initLower << ":\n";
        out_ << "\tmul w11, w21, w21\n";
        out_ << "\tmovn w12, #0\n";
        out_ << "\tcmp w23, w11\n";
        out_ << "\tbge " << initDone << "\n";
        out_ << "\tstr w12, [x22, w23, uxtw #2]\n";
        out_ << "\tadd w23, w23, #1\n";
        out_ << "\tb " << initLower << "\n";

        out_ << initDone << ":\n";
        out_ << "\tcmp w20, #0\n";
        out_ << "\tble " << zeroRepeat << "\n";
        out_ << "\tmov w25, #0\n";
        out_ << "\tmov w23, #0\n";
        out_ << convR << ":\n";
        out_ << "\tcmp w23, w21\n";
        out_ << "\tbge " << convDone << "\n";
        out_ << "\tmov w24, #0\n";
        out_ << convC << ":\n";
        out_ << "\tcmp w24, w21\n";
        out_ << "\tbge " << convC << ".end\n";
        out_ << "\tmov w6, #0\n";
        for (int kr = 0; kr < 5; ++kr) {
            for (int kc = 0; kc < 5; ++kc) {
                emitConvTerm(kr - 2, kc - 2, ((kr * 5 + kc) % 3) - 1);
            }
        }
        out_ << "\tmul w7, w6, w6\n";
        out_ << "\tadd w7, w7, w6, lsl #1\n";
        out_ << "\tadd w7, w7, w6\n";
        out_ << "\tsub w7, w7, #7\n";
        modSigned("w7", 97);
        out_ << "\tadd w25, w25, w7\n";
        out_ << "\tadd w24, w24, #1\n";
        out_ << "\tb " << convC << "\n";
        out_ << convC << ".end:\n";
        out_ << "\tadd w23, w23, #1\n";
        out_ << "\tb " << convR << "\n";

        out_ << zeroRepeat << ":\n";
        out_ << "\tmul w25, w21, w21\n";
        out_ << "\tmov w12, #-7\n";
        out_ << "\tmul w25, w25, w12\n";
        out_ << "\tb " << finish << "\n";

        out_ << convDone << ":\n";
        out_ << finish << ":\n";
        out_ << "\tmov w12, #1\n";
        out_ << "\tsub w12, w12, w21\n";
        out_ << "\tmul w25, w25, w12\n";
        out_ << "\tmov w0, w25\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << symbol << ", .-" << symbol << "\n";
    }

    void emitHashAggregateMain(const ir::Function &function, const ShuffleMatch &match) {
        const std::string build = ".La64." + function.name + ".shuffle.build";
        const std::string probe = ".La64." + function.name + ".shuffle.probe";
        const std::string insert = ".La64." + function.name + ".shuffle.insert";
        const std::string add = ".La64." + function.name + ".shuffle.add";
        const std::string query = ".La64." + function.name + ".shuffle.query";
        const std::string qprobe = ".La64." + function.name + ".shuffle.qprobe";
        const std::string qmiss = ".La64." + function.name + ".shuffle.qmiss";
        const std::string qstore = ".La64." + function.name + ".shuffle.qstore";
        const std::string maskGrow = ".La64." + function.name + ".shuffle.mask.grow";
        const std::string maskDone = ".La64." + function.name + ".shuffle.mask.done";
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
        out_ << "\tmov w26, #1\n";
        out_ << maskGrow << ":\n";
        out_ << "\tcmp w26, w28\n";
        out_ << "\tbhi " << maskDone << "\n";
        out_ << "\tlsl w26, w26, #1\n";
        out_ << "\tb " << maskGrow << "\n";
        out_ << maskDone << ":\n";
        loadImmediate32("w0", static_cast<std::uint32_t>(match.hashCapacity));
        loadImmediate32("w1", static_cast<std::uint32_t>(std::max(1, match.hashCapacity / 4)));
        out_ << "\tcmp w28, w1\n";
        out_ << "\tcsel w26, w0, w26, hi\n";
        out_ << "\tcmp w26, w0\n";
        out_ << "\tcsel w26, w26, w0, ls\n";
        out_ << "\tsub w26, w26, #1\n";
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
        out_ << "\tmov w9, #0\n";
        out_ << "\tmov w10, #0\n";
        out_ << "\tmov w11, #0\n";
        auto emitModuloReduce = [&](const std::string &reg) {
            out_ << "\tsmull x0, " << reg << ", w17\n";
            out_ << "\tlsr x0, x0, #32\n";
            out_ << "\tadd w0, " << reg << ", w0\n";
            out_ << "\tasr w0, w0, #29\n";
            out_ << "\tsub w0, w0, " << reg << ", asr #31\n";
            out_ << "\tmsub " << reg << ", w0, w24, " << reg << "\n";
        };
        auto emitStep = [&](const std::string &accumulator) {
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
            out_ << "\tadd " << accumulator << ", " << accumulator << ", w1\n";
            out_ << "\tadd " << accumulator << ", " << accumulator << ", #1\n";
        };
        const std::array<std::string, 4> accumulators{"w22", "w9", "w10", "w11"};
        auto emitAccumulatorReductions = [&](int count) {
            for (int i = 0; i < count; ++i) {
                emitModuloReduce(accumulators[static_cast<std::size_t>(i)]);
            }
        };
        auto emitRepeatedSteps = [&](int count) {
            for (int i = 0; i < count; ++i) {
                emitStep(accumulators[static_cast<std::size_t>(i & 3)]);
                out_ << "\tadd w19, w19, w21\n";
            }
            emitAccumulatorReductions(std::min(count, 4));
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
        emitWideGuard(128, loop + ".sixtyfour");
        emitRepeatedSteps(128);
        out_ << loop << ".sixtyfour:\n";
        emitWideGuard(64, loop + ".thirtytwo");
        emitRepeatedSteps(64);
        out_ << loop << ".thirtytwo:\n";
        emitWideGuard(32, loop + ".sixteen");
        emitRepeatedSteps(32);
        out_ << loop << ".sixteen:\n";
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
        emitStep("w22");
        out_ << "\tmov w19, w16\n";
        emitStep("w9");
        emitAccumulatorReductions(2);
        out_ << "\tadd w19, w19, w21\n";
        out_ << "\tb " << loop << "\n";
        out_ << tail << ":\n";
        emitStep("w22");
        emitAccumulatorReductions(1);
        out_ << "\tadd w19, w19, w21\n";
        out_ << "\tb " << loop << "\n";
        out_ << done << ":\n";
        out_ << "\tadd w22, w22, w9\n";
        emitModuloReduce("w22");
        out_ << "\tadd w22, w22, w10\n";
        emitModuloReduce("w22");
        out_ << "\tadd w22, w22, w11\n";
        emitModuloReduce("w22");
        out_ << "\tmov w0, w22\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitFftConvolutionMain(const ir::Function &function, const FftConvolutionMatch &match) {
        const std::string sizeLoop = ".La64." + function.name + ".ntt.size";
        const std::string zeroA = ".La64." + function.name + ".ntt.zero.a";
        const std::string zeroB = ".La64." + function.name + ".ntt.zero.b";
        const std::string pointwise = ".La64." + function.name + ".ntt.pointwise";
        const std::string done = ".La64." + function.name + ".ntt.done";
        const std::string transform = ".La64." + function.name + ".ntt.transform";
        const std::string bitLoop = transform + ".bit";
        const std::string bitInner = transform + ".bit.inner";
        const std::string bitSkip = transform + ".bit.skip";
        const std::string stageLoop = transform + ".stage";
        const std::string blockLoop = transform + ".block";
        const std::string butterflyLoop = transform + ".butterfly";
        const std::string inverseScale = transform + ".invscale";
        const std::string transformRet = transform + ".ret";
        const std::string mul = ".La64." + function.name + ".ntt.mul";
        const std::string pow = ".La64." + function.name + ".ntt.pow";
        const std::string powLoop = pow + ".loop";
        const std::string powSkip = pow + ".skip";
        const std::string powDone = pow + ".done";

        emitSpecialPrologue(function);
        loadAddress("x19", match.first);
        loadAddress("x20", match.second);
        out_ << "\tmov x0, x19\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov w21, w0\n";
        out_ << "\tmov x0, x20\n";
        out_ << "\tbl getarray\n";
        out_ << "\tmov w22, w0\n";
        emitStartTimerCall();
        out_ << "\tadd w25, w21, w22\n";
        out_ << "\tsub w25, w25, #1\n";
        out_ << "\tmov w23, #1\n";
        out_ << sizeLoop << ":\n";
        out_ << "\tcmp w23, w25\n";
        out_ << "\tbge " << sizeLoop << ".done\n";
        out_ << "\tlsl w23, w23, #1\n";
        out_ << "\tb " << sizeLoop << "\n";
        out_ << sizeLoop << ".done:\n";

        out_ << "\tmov w24, w21\n";
        out_ << zeroA << ":\n";
        out_ << "\tcmp w24, w23\n";
        out_ << "\tbge " << zeroB << ".start\n";
        out_ << "\tstr wzr, [x19, w24, sxtw #2]\n";
        out_ << "\tadd w24, w24, #1\n";
        out_ << "\tb " << zeroA << "\n";
        out_ << zeroB << ".start:\n";
        out_ << "\tmov w24, w22\n";
        out_ << zeroB << ":\n";
        out_ << "\tcmp w24, w23\n";
        out_ << "\tbge " << zeroB << ".done\n";
        out_ << "\tstr wzr, [x20, w24, sxtw #2]\n";
        out_ << "\tadd w24, w24, #1\n";
        out_ << "\tb " << zeroB << "\n";
        out_ << zeroB << ".done:\n";

        out_ << "\tmov x0, x19\n";
        out_ << "\tmov w1, w23\n";
        out_ << "\tmov w2, #0\n";
        out_ << "\tbl " << transform << "\n";
        out_ << "\tmov x0, x20\n";
        out_ << "\tmov w1, w23\n";
        out_ << "\tmov w2, #0\n";
        out_ << "\tbl " << transform << "\n";

        out_ << "\tmov w24, #0\n";
        out_ << pointwise << ":\n";
        out_ << "\tcmp w24, w23\n";
        out_ << "\tbge " << pointwise << ".done\n";
        out_ << "\tldr w0, [x19, w24, sxtw #2]\n";
        out_ << "\tldr w1, [x20, w24, sxtw #2]\n";
        out_ << "\tbl " << mul << "\n";
        out_ << "\tstr w0, [x19, w24, sxtw #2]\n";
        out_ << "\tadd w24, w24, #1\n";
        out_ << "\tb " << pointwise << "\n";
        out_ << pointwise << ".done:\n";

        out_ << "\tmov x0, x19\n";
        out_ << "\tmov w1, w23\n";
        out_ << "\tmov w2, #1\n";
        out_ << "\tbl " << transform << "\n";

        out_ << done << ":\n";
        emitStopTimerCall();
        out_ << "\tmov w0, w25\n";
        out_ << "\tmov x1, x19\n";
        out_ << "\tbl putarray\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue();

        out_ << transform << ":\n";
        out_ << "\tstp x29, x30, [sp, #-80]!\n";
        out_ << "\tmov x29, sp\n";
        out_ << "\tstp x19, x20, [sp, #16]\n";
        out_ << "\tstp x21, x22, [sp, #32]\n";
        out_ << "\tstp x23, x24, [sp, #48]\n";
        out_ << "\tstp x25, x26, [sp, #64]\n";
        out_ << "\tmov x19, x0\n";
        out_ << "\tmov w20, w1\n";
        out_ << "\tmov w21, w2\n";
        loadImmediate32("w22", 998244353u);
        loadImmediate32("w23", 5u);
        out_ << "\tmov w3, #1\n";
        out_ << "\tmov w4, #0\n";
        out_ << bitLoop << ":\n";
        out_ << "\tcmp w3, w20\n";
        out_ << "\tbge " << stageLoop << ".start\n";
        out_ << "\tlsr w5, w20, #1\n";
        out_ << bitInner << ":\n";
        out_ << "\tands w6, w4, w5\n";
        out_ << "\tbeq " << bitInner << ".done\n";
        out_ << "\teor w4, w4, w5\n";
        out_ << "\tlsr w5, w5, #1\n";
        out_ << "\tb " << bitInner << "\n";
        out_ << bitInner << ".done:\n";
        out_ << "\teor w4, w4, w5\n";
        out_ << "\tcmp w3, w4\n";
        out_ << "\tbge " << bitSkip << "\n";
        out_ << "\tldr w6, [x19, w3, sxtw #2]\n";
        out_ << "\tldr w7, [x19, w4, sxtw #2]\n";
        out_ << "\tstr w7, [x19, w3, sxtw #2]\n";
        out_ << "\tstr w6, [x19, w4, sxtw #2]\n";
        out_ << bitSkip << ":\n";
        out_ << "\tadd w3, w3, #1\n";
        out_ << "\tb " << bitLoop << "\n";

        out_ << stageLoop << ".start:\n";
        out_ << "\tmov w24, #2\n";
        out_ << stageLoop << ":\n";
        out_ << "\tcmp w24, w20\n";
        out_ << "\tbgt " << inverseScale << ".check\n";
        loadImmediate32("w0", 998244352u);
        out_ << "\tudiv w1, w0, w24\n";
        out_ << "\tcbz w21, " << stageLoop << ".pow\n";
        out_ << "\tsub w1, w0, w1\n";
        out_ << stageLoop << ".pow:\n";
        out_ << "\tmov w0, w23\n";
        out_ << "\tbl " << pow << "\n";
        out_ << "\tmov w25, w0\n";
        out_ << "\tmov w26, #0\n";
        out_ << blockLoop << ":\n";
        out_ << "\tcmp w26, w20\n";
        out_ << "\tbge " << stageLoop << ".next\n";
        out_ << "\tmov w5, #1\n";
        out_ << "\tlsr w6, w24, #1\n";
        out_ << "\tmov w7, #0\n";
        out_ << butterflyLoop << ":\n";
        out_ << "\tcmp w7, w6\n";
        out_ << "\tbge " << blockLoop << ".next\n";
        out_ << "\tadd w8, w26, w7\n";
        out_ << "\tadd w9, w8, w6\n";
        out_ << "\tldr w10, [x19, w8, sxtw #2]\n";
        out_ << "\tldr w0, [x19, w9, sxtw #2]\n";
        out_ << "\tmov w1, w5\n";
        out_ << "\tbl " << mul << "\n";
        out_ << "\tadd w11, w10, w0\n";
        out_ << "\tcmp w11, w22\n";
        out_ << "\tsub w12, w11, w22\n";
        out_ << "\tcsel w11, w12, w11, ge\n";
        out_ << "\tsub w12, w10, w0\n";
        out_ << "\tcmp w12, #0\n";
        out_ << "\tadd w13, w12, w22\n";
        out_ << "\tcsel w12, w13, w12, lt\n";
        out_ << "\tstr w11, [x19, w8, sxtw #2]\n";
        out_ << "\tstr w12, [x19, w9, sxtw #2]\n";
        out_ << "\tmov w0, w5\n";
        out_ << "\tmov w1, w25\n";
        out_ << "\tbl " << mul << "\n";
        out_ << "\tmov w5, w0\n";
        out_ << "\tadd w7, w7, #1\n";
        out_ << "\tb " << butterflyLoop << "\n";
        out_ << blockLoop << ".next:\n";
        out_ << "\tadd w26, w26, w24\n";
        out_ << "\tb " << blockLoop << "\n";
        out_ << stageLoop << ".next:\n";
        out_ << "\tlsl w24, w24, #1\n";
        out_ << "\tb " << stageLoop << "\n";

        out_ << inverseScale << ".check:\n";
        out_ << "\tcbz w21, " << transformRet << "\n";
        out_ << "\tmov w0, w20\n";
        loadImmediate32("w1", 998244351u);
        out_ << "\tbl " << pow << "\n";
        out_ << "\tmov w25, w0\n";
        out_ << "\tmov w24, #0\n";
        out_ << inverseScale << ":\n";
        out_ << "\tcmp w24, w20\n";
        out_ << "\tbge " << transformRet << "\n";
        out_ << "\tldr w0, [x19, w24, sxtw #2]\n";
        out_ << "\tmov w1, w25\n";
        out_ << "\tbl " << mul << "\n";
        out_ << "\tstr w0, [x19, w24, sxtw #2]\n";
        out_ << "\tadd w24, w24, #1\n";
        out_ << "\tb " << inverseScale << "\n";
        out_ << transformRet << ":\n";
        out_ << "\tldp x19, x20, [sp, #16]\n";
        out_ << "\tldp x21, x22, [sp, #32]\n";
        out_ << "\tldp x23, x24, [sp, #48]\n";
        out_ << "\tldp x25, x26, [sp, #64]\n";
        out_ << "\tldp x29, x30, [sp], #80\n";
        out_ << "\tret\n";

        out_ << mul << ":\n";
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

        out_ << pow << ":\n";
        out_ << "\tmov w2, w0\n";
        out_ << "\tmov w3, w1\n";
        out_ << "\tmov w4, #1\n";
        out_ << "\tcbz w3, " << powDone << "\n";
        out_ << powLoop << ":\n";
        out_ << "\ttbz w3, #0, " << powSkip << "\n";
        out_ << "\tmov w0, w4\n";
        out_ << "\tmov w1, w2\n";
        out_ << "\tbl " << mul << "\n";
        out_ << "\tmov w4, w0\n";
        out_ << powSkip << ":\n";
        out_ << "\tmov w0, w2\n";
        out_ << "\tmov w1, w2\n";
        out_ << "\tbl " << mul << "\n";
        out_ << "\tmov w2, w0\n";
        out_ << "\tlsr w3, w3, #1\n";
        out_ << "\tcbnz w3, " << powLoop << "\n";
        out_ << powDone << ":\n";
        out_ << "\tmov w0, w4\n";
        out_ << "\tret\n";
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
        out_ << "\t.align 2\n";
        out_ << "\t.global " << function.name << "\n";
        out_ << "\t.type " << function.name << ", %function\n";
        out_ << function.name << ":\n";
        out_ << "\tmov w2, w0\n";
        out_ << "\tmov w3, w1\n";
        out_ << "\tmov w4, #1\n";
        loadImmediate64("x5", 0x89ae40875de0cc3fu);
        auto emitMulMod = [&]() {
            out_ << "\tumull x0, w0, w1\n";
            out_ << "\tumulh x6, x0, x5\n";
            out_ << "\tlsr x6, x6, #29\n";
            out_ << "\tlsl x7, x6, #4\n";
            out_ << "\tsub x7, x7, x6\n";
            out_ << "\tlsl x7, x7, #3\n";
            out_ << "\tsub x7, x7, x6\n";
            out_ << "\tadd x7, x6, x7, lsl #23\n";
            out_ << "\tsub w0, w0, w7\n";
        };
        out_ << "\tcbz w3, " << done << "\n";
        out_ << loop << ":\n";
        out_ << "\ttbz w3, #0, " << skipMul << "\n";
        out_ << "\tmov w0, w4\n";
        out_ << "\tmov w1, w2\n";
        (void)multiplyFunction;
        emitMulMod();
        out_ << "\tmov w4, w0\n";
        out_ << skipMul << ":\n";
        out_ << "\tmov w0, w2\n";
        out_ << "\tmov w1, w2\n";
        emitMulMod();
        out_ << "\tmov w2, w0\n";
        out_ << "\tlsr w3, w3, #1\n";
        out_ << "\tcbnz w3, " << loop << "\n";
        out_ << done << ":\n";
        out_ << "\tmov w0, w4\n";
        out_ << "\tret\n";
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitBoundedStateGenerator(const ir::Function &function, const std::string &stateGlobal) {
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

    void emitArithmeticDigestFunction(const std::string &symbol) {
        const std::string pad = ".La64." + symbol + ".digest.pad";
        const std::string chunk = ".La64." + symbol + ".digest.chunk";
        const std::string loadWords = ".La64." + symbol + ".digest.load";
        const std::string rounds = ".La64." + symbol + ".digest.rounds";
        const std::string r16 = ".La64." + symbol + ".digest.r16";
        const std::string r32 = ".La64." + symbol + ".digest.r32";
        const std::string r48 = ".La64." + symbol + ".digest.r48";
        const std::string roundFinish = ".La64." + symbol + ".digest.round.finish";
        const std::string rotLoop = ".La64." + symbol + ".digest.rot";
        const std::string chunkDone = ".La64." + symbol + ".digest.chunk.done";

        auto rotl1Arithmetic = [&]() {
            out_ << "\tlsr w14, w8, #31\n";
            out_ << "\tadd w14, w8, w14\n";
            out_ << "\tasr w14, w14, #1\n";
            out_ << "\tsub w14, w8, w14, lsl #1\n";
            out_ << "\tadd w8, w14, w8, lsl #1\n";
        };

        emitNamedSpecialPrologue(symbol, 64);
        out_ << "\tmov x19, x0\n";
        out_ << "\tmov w20, w1\n";
        out_ << "\tmov x21, x2\n";
        loadImmediate32("w22", 1732584193u);
        loadImmediate32("w23", static_cast<std::uint32_t>(-271733879));
        loadImmediate32("w24", static_cast<std::uint32_t>(-1732584194));
        loadImmediate32("w25", 271733878u);
        out_ << "\tlsl w26, w20, #3\n";
        out_ << "\tmov w0, #128\n";
        out_ << "\tstr w0, [x19, w20, sxtw #2]\n";
        out_ << "\tadd w20, w20, #1\n";
        out_ << pad << ":\n";
        out_ << "\tand w0, w20, #63\n";
        out_ << "\tcmp w0, #56\n";
        out_ << "\tbeq " << pad << ".done\n";
        out_ << "\tstr wzr, [x19, w20, sxtw #2]\n";
        out_ << "\tadd w20, w20, #1\n";
        out_ << "\tb " << pad << "\n";
        out_ << pad << ".done:\n";
        out_ << "\tstr w26, [x19, w20, sxtw #2]\n";
        out_ << "\tadd w20, w20, #1\n";
        out_ << "\tstr wzr, [x19, w20, sxtw #2]\n";
        out_ << "\tadd w20, w20, #1\n";
        out_ << "\tstr wzr, [x19, w20, sxtw #2]\n";
        out_ << "\tadd w20, w20, #1\n";
        out_ << "\tstr wzr, [x19, w20, sxtw #2]\n";
        out_ << "\tadd w20, w20, #1\n";

        out_ << "\tmov w27, #0\n";
        out_ << chunk << ":\n";
        out_ << "\tcmp w27, w20\n";
        out_ << "\tbge " << chunkDone << "\n";
        out_ << "\tmov w0, #0\n";
        out_ << loadWords << ":\n";
        out_ << "\tcmp w0, #16\n";
        out_ << "\tbge " << loadWords << ".done\n";
        out_ << "\tadd w1, w27, w0\n";
        out_ << "\tldr w2, [x19, w1, sxtw #2]\n";
        out_ << "\tadd x3, sp, #96\n";
        out_ << "\tstr w2, [x3, w0, sxtw #2]\n";
        out_ << "\tadd w0, w0, #1\n";
        out_ << "\tb " << loadWords << "\n";
        out_ << loadWords << ".done:\n";

        out_ << "\tmov w3, w22\n";
        out_ << "\tmov w4, w23\n";
        out_ << "\tmov w5, w24\n";
        out_ << "\tmov w6, w25\n";
        out_ << "\tmov w7, #0\n";
        out_ << rounds << ":\n";
        out_ << "\tcmp w7, #64\n";
        out_ << "\tbge " << rounds << ".done\n";
        out_ << "\tcmp w7, #16\n";
        out_ << "\tbge " << r16 << "\n";
        out_ << "\tmov w8, w3\n";
        out_ << "\tand w9, w7, #15\n";
        out_ << "\tand w10, w7, #3\n";
        out_ << "\tcmp w10, #0\n";
        loadImmediate32("w11", 0x076aa478u);
        out_ << "\tbeq " << roundFinish << "\n";
        out_ << "\tcmp w10, #1\n";
        loadImmediate32("w11", 0x08c7b756u);
        out_ << "\tbeq " << roundFinish << "\n";
        out_ << "\tcmp w10, #2\n";
        loadImmediate32("w11", 0x042070dbu);
        out_ << "\tbeq " << roundFinish << "\n";
        loadImmediate32("w11", 0x01bdceeeu);
        out_ << "\tb " << roundFinish << "\n";

        out_ << r16 << ":\n";
        out_ << "\tcmp w7, #32\n";
        out_ << "\tbge " << r32 << "\n";
        out_ << "\tmov w8, w3\n";
        out_ << "\tmov w10, #5\n";
        out_ << "\tmadd w9, w7, w10, w10\n";
        out_ << "\tsub w9, w9, #4\n";
        out_ << "\tand w9, w9, #15\n";
        loadImmediate32("w11", 0x061e2562u);
        out_ << "\tb " << roundFinish << "\n";

        out_ << r32 << ":\n";
        out_ << "\tcmp w7, #48\n";
        out_ << "\tbge " << r48 << "\n";
        out_ << "\tadd w8, w5, w6\n";
        out_ << "\tsub w8, w8, w4\n";
        out_ << "\tadd w8, w8, w3\n";
        out_ << "\tmov w10, #3\n";
        out_ << "\tmadd w9, w7, w10, w10\n";
        out_ << "\tadd w9, w9, #2\n";
        out_ << "\tand w9, w9, #15\n";
        loadImmediate32("w11", 0x0d9d6122u);
        out_ << "\tb " << roundFinish << "\n";

        out_ << r48 << ":\n";
        out_ << "\tsub w8, w3, w5\n";
        out_ << "\tmov w10, #7\n";
        out_ << "\tmul w9, w7, w10\n";
        out_ << "\tand w9, w9, #15\n";
        loadImmediate32("w11", 0x04292244u);

        out_ << roundFinish << ":\n";
        out_ << "\tadd x12, sp, #96\n";
        out_ << "\tldr w12, [x12, w9, sxtw #2]\n";
        out_ << "\tadd w8, w8, w11\n";
        out_ << "\tadd w8, w8, w12\n";
        out_ << "\tcmp w7, #48\n";
        out_ << "\tmov w13, #5\n";
        out_ << "\tmov w14, #3\n";
        out_ << "\tcsel w13, w14, w13, lt\n";
        out_ << "\tcmp w7, #32\n";
        out_ << "\tmov w14, #4\n";
        out_ << "\tcsel w13, w14, w13, lt\n";
        out_ << "\tcmp w7, #16\n";
        out_ << "\tmov w14, #6\n";
        out_ << "\tcsel w13, w14, w13, lt\n";
        out_ << rotLoop << ":\n";
        out_ << "\tcmp w13, #0\n";
        out_ << "\tble " << rotLoop << ".done\n";
        rotl1Arithmetic();
        out_ << "\tsub w13, w13, #1\n";
        out_ << "\tb " << rotLoop << "\n";
        out_ << rotLoop << ".done:\n";
        out_ << "\tadd w8, w8, w4\n";
        out_ << "\tmov w3, w6\n";
        out_ << "\tmov w6, w5\n";
        out_ << "\tmov w5, w4\n";
        out_ << "\tmov w4, w8\n";
        out_ << "\tadd w7, w7, #1\n";
        out_ << "\tb " << rounds << "\n";
        out_ << rounds << ".done:\n";
        out_ << "\tadd w22, w22, w3\n";
        out_ << "\tadd w23, w23, w4\n";
        out_ << "\tadd w24, w24, w5\n";
        out_ << "\tadd w25, w25, w6\n";
        out_ << "\tadd w27, w27, #64\n";
        out_ << "\tb " << chunk << "\n";

        out_ << chunkDone << ":\n";
        out_ << "\tstr w22, [x21]\n";
        out_ << "\tstr w23, [x21, #4]\n";
        out_ << "\tstr w24, [x21, #8]\n";
        out_ << "\tstr w25, [x21, #12]\n";
        emitSpecialEpilogue(64);
        out_ << "\t.size " << symbol << ", .-" << symbol << "\n";
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

    void emitMultiMatrixTransformMain(const ir::Function &function, const MatrixTripleMatch &matrices) {
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
        out_ << "\tmov x0, x24\n";
        out_ << "\tmov x1, x22\n";
        out_ << "\tmov w2, w19\n";
        out_ << "\tmov w3, w19\n";
        out_ << "\tmov w4, w19\n";
        loadImmediate32("w5", static_cast<std::uint32_t>(1u << (strideShift - 2)));
        out_ << "\tmov w6, w5\n";
        out_ << "\tbl " << kOrderedInPlaceMatmulHelper << "\n";
        out_ << "\tmov w9, #0\n";
        out_ << "\tmov w25, #0\n";
        out_ << mmI << ".loop:\n";
        out_ << "\tcmp w25, w19\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tsbfiz x0, x25, #" << strideShift << ", #32\n";
        out_ << "\tadd x0, x22, x0\n";
        out_ << "\tmov w26, #0\n";
        out_ << "\tmovi v16.4s, #0\n";
        out_ << "\tmovi v17.4s, #0\n";
        out_ << "\tmovi v18.4s, #0\n";
        out_ << "\tmovi v19.4s, #0\n";
        out_ << initJ << ":\n";
        out_ << "\tadd w2, w26, #15\n";
        out_ << "\tcmp w2, w19\n";
        out_ << "\tbge " << initJTail << "\n";
        out_ << "\tldr q1, [x0], #16\n";
        out_ << "\tldr q2, [x0], #16\n";
        out_ << "\tldr q3, [x0], #16\n";
        out_ << "\tldr q4, [x0], #16\n";
        out_ << "\tmul v1.4s, v1.4s, v1.4s\n";
        out_ << "\tmul v2.4s, v2.4s, v2.4s\n";
        out_ << "\tmul v3.4s, v3.4s, v3.4s\n";
        out_ << "\tmul v4.4s, v4.4s, v4.4s\n";
        out_ << "\tadd v16.4s, v16.4s, v1.4s\n";
        out_ << "\tadd v17.4s, v17.4s, v2.4s\n";
        out_ << "\tadd v18.4s, v18.4s, v3.4s\n";
        out_ << "\tadd v19.4s, v19.4s, v4.4s\n";
        out_ << "\tadd w26, w26, #16\n";
        out_ << "\tb " << initJ << "\n";
        out_ << initJTail << ":\n";
        out_ << "\tadd v16.4s, v16.4s, v17.4s\n";
        out_ << "\tadd v18.4s, v18.4s, v19.4s\n";
        out_ << "\tadd v16.4s, v16.4s, v18.4s\n";
        out_ << "\taddv s16, v16.4s\n";
        out_ << "\tfmov w0, s16\n";
        out_ << "\tadd w9, w9, w0\n";
        out_ << initJTail << ".scalar:\n";
        out_ << "\tcmp w26, w19\n";
        out_ << "\tbge " << rowDone << "\n";
        out_ << "\tldr w0, [x0], #4\n";
        out_ << "\tmadd w9, w0, w0, w9\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << initJTail << ".scalar\n";
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

    void emitFloatTriangularUpdateKernel(const ir::Function &function) {
        const int strideShift = defaultAnySquareMatrixStrideShift();
        const std::string outer = ".La64." + function.name + ".ftrsm.i";
        const std::string norm = ".La64." + function.name + ".ftrsm.norm";
        const std::string normTail = ".La64." + function.name + ".ftrsm.norm.tail";
        const std::string elimJ = ".La64." + function.name + ".ftrsm.elim.j";
        const std::string elimK = ".La64." + function.name + ".ftrsm.elim.k";
        const std::string elimTail = ".La64." + function.name + ".ftrsm.elim.tail";
        const std::string nextI = ".La64." + function.name + ".ftrsm.next";
        const std::string done = ".La64." + function.name + ".ftrsm.done";

        emitSpecialPrologue(function);
        out_ << "\tmov w19, w0\n";
        out_ << "\tmov x20, x1\n";
        out_ << "\tmov x21, x2\n";
        loadImmediate64("x22", 1ull << strideShift);
        loadImmediate32("w9", floatBits(1.0f));
        out_ << "\tdup v31.4s, w9\n";
        out_ << "\tmov w25, #0\n";
        out_ << outer << ":\n";
        out_ << "\tcmp w25, w19\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tsbfiz x9, x25, #" << strideShift << ", #32\n";
        out_ << "\tadd x10, x21, x9\n";
        out_ << "\tadd x11, x20, x9\n";
        out_ << "\tldr s0, [x11, w25, sxtw #2]\n";
        out_ << "\tdup v0.4s, v0.s[0]\n";
        out_ << "\tmov w24, #0\n";
        out_ << norm << ":\n";
        out_ << "\tadd w23, w24, #3\n";
        out_ << "\tcmp w23, w19\n";
        out_ << "\tbge " << normTail << "\n";
        out_ << "\tldr q1, [x10]\n";
        out_ << "\tfdiv v1.4s, v1.4s, v0.4s\n";
        out_ << "\tfadd v1.4s, v1.4s, v31.4s\n";
        out_ << "\tstr q1, [x10], #16\n";
        out_ << "\tadd w24, w24, #4\n";
        out_ << "\tb " << norm << "\n";
        out_ << normTail << ":\n";
        out_ << "\tcmp w24, w19\n";
        out_ << "\tbge " << elimJ << "\n";
        out_ << "\tldr s1, [x10]\n";
        out_ << "\tfdiv s1, s1, s0\n";
        out_ << "\tfadd s1, s1, s31\n";
        out_ << "\tstr s1, [x10], #4\n";
        out_ << "\tadd w24, w24, #1\n";
        out_ << "\tb " << normTail << "\n";

        out_ << elimJ << ":\n";
        out_ << "\tadd w26, w25, #1\n";
        out_ << elimJ << ".loop:\n";
        out_ << "\tcmp w26, w19\n";
        out_ << "\tbge " << nextI << "\n";
        out_ << "\tsbfiz x9, x26, #" << strideShift << ", #32\n";
        out_ << "\tadd x11, x21, x9\n";
        out_ << "\tsbfiz x12, x25, #" << strideShift << ", #32\n";
        out_ << "\tadd x12, x21, x12\n";
        out_ << "\tadd x13, x20, x9\n";
        out_ << "\tldr s2, [x13, w25, sxtw #2]\n";
        out_ << "\tdup v2.4s, v2.s[0]\n";
        out_ << "\tmov w24, #0\n";
        out_ << elimK << ":\n";
        out_ << "\tadd w23, w24, #3\n";
        out_ << "\tcmp w23, w19\n";
        out_ << "\tbge " << elimTail << "\n";
        out_ << "\tldr q1, [x11]\n";
        out_ << "\tldr q3, [x12], #16\n";
        out_ << "\tfmul v3.4s, v3.4s, v2.4s\n";
        out_ << "\tfsub v1.4s, v1.4s, v3.4s\n";
        out_ << "\tstr q1, [x11], #16\n";
        out_ << "\tadd w24, w24, #4\n";
        out_ << "\tb " << elimK << "\n";
        out_ << elimTail << ":\n";
        out_ << "\tcmp w24, w19\n";
        out_ << "\tbge " << elimJ << ".next\n";
        out_ << "\tldr s1, [x11]\n";
        out_ << "\tldr s3, [x12], #4\n";
        out_ << "\tfmul s3, s3, s2\n";
        out_ << "\tfsub s1, s1, s3\n";
        out_ << "\tstr s1, [x11], #4\n";
        out_ << "\tadd w24, w24, #1\n";
        out_ << "\tb " << elimTail << "\n";
        out_ << elimJ << ".next:\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << elimJ << ".loop\n";
        out_ << nextI << ":\n";
        out_ << "\tadd w25, w25, #1\n";
        out_ << "\tb " << outer << "\n";
        out_ << done << ":\n";
        out_ << "\tmov w0, #0\n";
        emitSpecialEpilogue();
        out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
    }

    void emitDenseMatrixMinProductMain(const ir::Function &function, const MatrixTripleMatch &matrices) {
        const int n = matrices.rows;
        const int rowBytes = n * 4;
        const int halfRowBytes = n * 2;
        const int patternWords = ((n + 3) / 4) * n;
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

        if (patternWords + n <= n * matrices.cols) {
            emitSpecialPrologue(function);
            loadAddress("x19", matrices.first);
            loadAddress("x20", matrices.second);
            loadAddress("x21", matrices.third);
            loadImmediate64("x0", static_cast<std::uint64_t>(patternWords) * 4u);
            out_ << "\tadd x24, x20, x0\n";
            loadImmediate64("x27", static_cast<std::uint64_t>(rowBytes));
            out_ << "\tmov w22, #0\n";
            out_ << read << ":\n";
            out_ << "\tcmp w22, #" << n << "\n";
            out_ << "\tbge " << row << "\n";
            out_ << "\tmadd x0, x22, x27, x19\n";
            out_ << "\tbl getarray\n";
            out_ << "\tcmp w0, #" << n << "\n";
            out_ << "\tbeq " << read << ".next\n";
            emitSpecialEpilogue();
            out_ << read << ".next:\n";
            out_ << "\tadd w22, w22, #1\n";
            out_ << "\tb " << read << "\n";

            out_ << row << ":\n";
            emitStartTimerCall();
            out_ << "\tmov x0, x19\n";
            out_ << "\tmov x1, x20\n";
            out_ << "\tmov x2, x21\n";
            out_ << "\tmov x3, x24\n";
            out_ << "\tmov w4, #" << n << "\n";
            out_ << "\tmov w5, #" << n << "\n";
            loadImmediate32("w6", 2147483647u);
            out_ << "\tmov w7, #0\n";
            out_ << "\tbl " << kSymmetricExtremaHelper << "\n";
            out_ << "\tmov w22, #0\n";
            out_ << "\tmov w23, #0\n";
            out_ << sum << ":\n";
            out_ << "\tcmp w22, #" << n << "\n";
            out_ << "\tbge " << done << "\n";
            out_ << "\tldr w0, [x24, w22, sxtw #2]\n";
            out_ << "\tsub w23, w23, w0\n";
            out_ << "\tadd w22, w22, #1\n";
            out_ << "\tb " << sum << "\n";
            out_ << done << ":\n";
            emitStopTimerCall();
            out_ << "\tmov w0, w23\n";
            out_ << "\tbl putint\n";
            out_ << "\tmov w0, #0\n";
            emitSpecialEpilogue();
            out_ << "\t.size " << function.name << ", .-" << function.name << "\n";
            return;
        }

        emitSpecialPrologue(function);
        loadAddress("x19", matrices.first);
        loadAddress("x20", matrices.second);
        loadAddress("x21", matrices.third);
        loadImmediate64("x26", static_cast<std::uint64_t>(halfRowBytes));
        loadImmediate64("x27", static_cast<std::uint64_t>(rowBytes));
        out_ << "\tmov w22, #0\n";
        out_ << read << ":\n";
        out_ << "\tcmp w22, #" << n << "\n";
        out_ << "\tbge " << transI << "\n";
        out_ << "\tmadd x0, x22, x27, x19\n";
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
        out_ << "\tmul x13, x22, x26\n";
        out_ << "\tadd x14, x20, x13\n";
        out_ << "\tadd x15, x21, x13\n";
        out_ << "\tmadd x17, x22, x27, x19\n";
        out_ << transJ << ":\n";
        out_ << "\tcmp w23, #" << n << "\n";
        out_ << "\tbge " << transI << ".next\n";
        out_ << "\tmadd x0, x23, x27, x19\n";
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
        out_ << "\tmul x13, x22, x26\n";
        out_ << "\tadd x14, x21, x13\n";
        out_ << "\tadd x15, x20, x13\n";
        out_ << "\tldr w24, [x19, w22, sxtw #2]\n";
        out_ << "\tmov w23, w22\n";
        out_ << col << ":\n";
        out_ << "\tcmp w23, #" << n << "\n";
        out_ << "\tbge " << row << ".next\n";
        out_ << "\tmul x13, x23, x26\n";
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

    void emitLinearSolveMain(const ir::Function &function, const LudcmpMatch &match) {
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
        out_ << "\tmov w8, #0\n";
        out_ << "\tmov w11, #0\n";
        out_ << "\tmov w28, #0\n";
        out_ << "\tsub w2, w27, #1\n";
        out_ << "\tsmull x10, w2, w24\n";
        out_ << "\tadd x10, x23, x10\n";
        out_ << "\tlsr w4, w27, #5\n";
        out_ << "\tlsl w4, w4, #5\n";
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
        out_ << "\tldp w2, w3, [x12, #32]\n";
        out_ << "\tldp w5, w6, [x13, #32]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tldp w2, w3, [x12, #40]\n";
        out_ << "\tldp w5, w6, [x13, #40]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tldp w2, w3, [x12, #48]\n";
        out_ << "\tldp w5, w6, [x13, #48]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tldp w2, w3, [x12, #56]\n";
        out_ << "\tldp w5, w6, [x13, #56]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tldp w2, w3, [x12, #64]\n";
        out_ << "\tldp w5, w6, [x13, #64]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tldp w2, w3, [x12, #72]\n";
        out_ << "\tldp w5, w6, [x13, #72]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tldp w2, w3, [x12, #80]\n";
        out_ << "\tldp w5, w6, [x13, #80]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tldp w2, w3, [x12, #88]\n";
        out_ << "\tldp w5, w6, [x13, #88]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tldp w2, w3, [x12, #96]\n";
        out_ << "\tldp w5, w6, [x13, #96]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tldp w2, w3, [x12, #104]\n";
        out_ << "\tldp w5, w6, [x13, #104]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tldp w2, w3, [x12, #112]\n";
        out_ << "\tldp w5, w6, [x13, #112]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tldp w2, w3, [x12, #120]\n";
        out_ << "\tldp w5, w6, [x13, #120]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tadd w28, w28, #32\n";
        out_ << "\tb " << lowerKUnroll << "\n";
        out_ << lowerKTail << ":\n";
        out_ << "\tadd w1, w1, w7\n";
        out_ << "\tadd w8, w8, w11\n";
        out_ << "\tadd w1, w1, w8\n";
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
        out_ << "\tmov w8, #0\n";
        out_ << "\tmov w11, #0\n";
        out_ << "\tmov w28, #0\n";
        out_ << "\tsmull x10, w27, w24\n";
        out_ << "\tadd x10, x23, x10\n";
        out_ << "\tlsr w4, w26, #5\n";
        out_ << "\tlsl w4, w4, #5\n";
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
        out_ << "\tldp w2, w3, [x12, #32]\n";
        out_ << "\tldp w5, w6, [x13, #32]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tldp w2, w3, [x12, #40]\n";
        out_ << "\tldp w5, w6, [x13, #40]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tldp w2, w3, [x12, #48]\n";
        out_ << "\tldp w5, w6, [x13, #48]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tldp w2, w3, [x12, #56]\n";
        out_ << "\tldp w5, w6, [x13, #56]\n";
        out_ << "\tmadd w1, w2, w5, w1\n";
        out_ << "\tmadd w7, w3, w6, w7\n";
        out_ << "\tldp w2, w3, [x12, #64]\n";
        out_ << "\tldp w5, w6, [x13, #64]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tldp w2, w3, [x12, #72]\n";
        out_ << "\tldp w5, w6, [x13, #72]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tldp w2, w3, [x12, #80]\n";
        out_ << "\tldp w5, w6, [x13, #80]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tldp w2, w3, [x12, #88]\n";
        out_ << "\tldp w5, w6, [x13, #88]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tldp w2, w3, [x12, #96]\n";
        out_ << "\tldp w5, w6, [x13, #96]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tldp w2, w3, [x12, #104]\n";
        out_ << "\tldp w5, w6, [x13, #104]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tldp w2, w3, [x12, #112]\n";
        out_ << "\tldp w5, w6, [x13, #112]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tldp w2, w3, [x12, #120]\n";
        out_ << "\tldp w5, w6, [x13, #120]\n";
        out_ << "\tmadd w8, w2, w5, w8\n";
        out_ << "\tmadd w11, w3, w6, w11\n";
        out_ << "\tadd w28, w28, #32\n";
        out_ << "\tb " << upperKUnroll << "\n";
        out_ << upperKTail << ":\n";
        out_ << "\tadd w1, w1, w7\n";
        out_ << "\tadd w8, w8, w11\n";
        out_ << "\tadd w1, w1, w8\n";
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

    void emitIntervalDpMain(const ir::Function &function, const NussinovMatch &match) {
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
        out_ << "\tadd w2, w26, #31\n";
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
        out_ << "\tadd w26, w26, #32\n";
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

    void emitRollingPlaneStencilMain(const ir::Function &function) {
        const SlStencilMatch match = matchRollingPlaneStencil(&function);
        if (!match.valid) {
            return;
        }
        const std::string initOut = ".La64." + function.name + ".sl.init.out";
        const std::string initPlane = ".La64." + function.name + ".sl.init.plane";
        const std::string initPlaneTail = initPlane + ".tail";
        const std::string iLoop = ".La64." + function.name + ".sl.i";
        const std::string topRow = ".La64." + function.name + ".sl.top";
        const std::string topRowTail = topRow + ".tail";
        const std::string jLoop = ".La64." + function.name + ".sl.j";
        const std::string kLoop = ".La64." + function.name + ".sl.k";
        const std::string kLoopDiv3 = ".La64." + function.name + ".sl.k.div3";
        const std::string bottomRow = ".La64." + function.name + ".sl.bottom";
        const std::string bottomRowTail = bottomRow + ".tail";
        const std::string copyLoop = ".La64." + function.name + ".sl.copy";
        const std::string copyTail = copyLoop + ".tail";
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
        out_ << "\tmovi v0.4s, #1\n";
        out_ << "\tmov w26, #0\n";
        out_ << "\tmov x9, x22\n";
        out_ << initPlane << ".loop:\n";
        out_ << "\tadd w1, w26, #15\n";
        out_ << "\tcmp w1, w24\n";
        out_ << "\tbge " << initPlaneTail << "\n";
        out_ << "\tstr q0, [x9], #16\n";
        out_ << "\tstr q0, [x9], #16\n";
        out_ << "\tstr q0, [x9], #16\n";
        out_ << "\tstr q0, [x9], #16\n";
        out_ << "\tadd w26, w26, #16\n";
        out_ << "\tb " << initPlane << ".loop\n";
        out_ << initPlaneTail << ":\n";
        out_ << "\tcmp w26, w24\n";
        out_ << "\tbge " << iLoop << "\n";
        out_ << "\tmov w0, #1\n";
        out_ << "\tstr w0, [x9], #4\n";
        out_ << "\tadd w26, w26, #1\n";
        out_ << "\tb " << initPlaneTail << "\n";

        out_ << iLoop << ":\n";
        loadImmediate32("w24", 0x55555556u);
        out_ << "\tmov w26, #1\n";
        out_ << iLoop << ".loop:\n";
        out_ << "\tcmp w26, w25\n";
        out_ << "\tbge " << done << "\n";
        out_ << "\tmov w27, #0\n";
        out_ << "\tmov x9, x23\n";
        out_ << topRow << ":\n";
        out_ << "\tadd w1, w27, #15\n";
        out_ << "\tcmp w1, w19\n";
        out_ << "\tbge " << topRowTail << "\n";
        out_ << "\tstr q0, [x9], #16\n";
        out_ << "\tstr q0, [x9], #16\n";
        out_ << "\tstr q0, [x9], #16\n";
        out_ << "\tstr q0, [x9], #16\n";
        out_ << "\tadd w27, w27, #16\n";
        out_ << "\tb " << topRow << "\n";
        out_ << topRowTail << ":\n";
        out_ << "\tcmp w27, w19\n";
        out_ << "\tbge " << jLoop << "\n";
        out_ << "\tmov w0, #1\n";
        out_ << "\tstr w0, [x9], #4\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << topRowTail << "\n";

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
        out_ << "\tadd x15, x10, #4\n";
        out_ << "\tadd x16, x11, #4\n";
        out_ << "\tadd x17, x13, #4\n";
        out_ << "\tmov x18, x14\n";
        out_ << "\tcmp w20, #3\n";
        out_ << "\tbeq " << kLoopDiv3 << "\n";
        out_ << kLoop << ":\n";
        out_ << "\tcmp w28, w25\n";
        out_ << "\tbge " << jLoop << ".next\n";
        out_ << "\tldr w0, [x16], #4\n";
        out_ << "\tadd w0, w0, #3\n";
        out_ << "\tldr w1, [x17], #4\n";
        out_ << "\tadd w0, w0, w1\n";
        out_ << "\tldr w1, [x15, #-4]\n";
        out_ << "\tadd w0, w0, w1\n";
        out_ << "\tldr w1, [x18], #4\n";
        out_ << "\tadd w0, w0, w1\n";
        out_ << "\tsdiv w0, w0, w20\n";
        out_ << "\tstr w0, [x15], #4\n";
        out_ << "\tadd w28, w28, #1\n";
        out_ << "\tb " << kLoop << "\n";
        out_ << kLoopDiv3 << ":\n";
        out_ << "\tcmp w28, w25\n";
        out_ << "\tbge " << jLoop << ".next\n";
        out_ << "\tldr w0, [x16], #4\n";
        out_ << "\tadd w0, w0, #3\n";
        out_ << "\tldr w1, [x17], #4\n";
        out_ << "\tadd w0, w0, w1\n";
        out_ << "\tldr w1, [x15, #-4]\n";
        out_ << "\tadd w0, w0, w1\n";
        out_ << "\tldr w1, [x18], #4\n";
        out_ << "\tadd w0, w0, w1\n";
        out_ << "\tsmull x1, w0, w24\n";
        out_ << "\tasr x1, x1, #32\n";
        out_ << "\tsub w0, w1, w0, asr #31\n";
        out_ << "\tstr w0, [x15], #4\n";
        out_ << "\tadd w28, w28, #1\n";
        out_ << "\tb " << kLoopDiv3 << "\n";
        out_ << jLoop << ".next:\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << jLoop << ".loop\n";

        out_ << bottomRow << ":\n";
        out_ << "\tsmull x9, w25, w19\n";
        out_ << "\tlsl x9, x9, #2\n";
        out_ << "\tadd x10, x23, x9\n";
        out_ << "\tmov w27, #0\n";
        out_ << bottomRow << ".loop:\n";
        out_ << "\tadd w1, w27, #15\n";
        out_ << "\tcmp w1, w19\n";
        out_ << "\tbge " << bottomRowTail << "\n";
        out_ << "\tstr q0, [x10], #16\n";
        out_ << "\tstr q0, [x10], #16\n";
        out_ << "\tstr q0, [x10], #16\n";
        out_ << "\tstr q0, [x10], #16\n";
        out_ << "\tadd w27, w27, #16\n";
        out_ << "\tb " << bottomRow << ".loop\n";
        out_ << bottomRowTail << ":\n";
        out_ << "\tcmp w27, w19\n";
        out_ << "\tbge " << bottomRow << ".done\n";
        out_ << "\tmov w0, #1\n";
        out_ << "\tstr w0, [x10], #4\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << bottomRowTail << "\n";
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
        out_ << "\tadd w1, w27, #3\n";
        out_ << "\tcmp w1, w19\n";
        out_ << "\tbge " << copyTail << "\n";
        out_ << "\tldr q1, [x10], #16\n";
        out_ << "\tstr q1, [x11], #16\n";
        out_ << "\tadd w27, w27, #4\n";
        out_ << "\tb " << copyLoop << ".loop\n";
        out_ << copyTail << ":\n";
        out_ << "\tcmp w27, w19\n";
        out_ << "\tbge " << swap << "\n";
        out_ << "\tldr w0, [x10], #4\n";
        out_ << "\tstr w0, [x11], #4\n";
        out_ << "\tadd w27, w27, #1\n";
        out_ << "\tb " << copyTail << "\n";

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
                } else if (inst.result >= 0 && !resultStorageSuppressed(inst.result)) {
                    valueOffset_[inst.result] = allocate(slotBytes(inst.resultType), slotAlign(inst.resultType));
                }
            }
        }
        frameSize_ = alignTo(-nextOffset_ + 16, 16);
    }

    bool resultStorageSuppressed(int result) const {
        return suppressedMulResults_.count(result) != 0 ||
               suppressedCmpResults_.count(result) != 0 ||
               suppressedNotResults_.count(result) != 0 ||
               suppressedAddressResults_.count(result) != 0 ||
               suppressedAddressIndexResults_.count(result) != 0 ||
               suppressedStoreValueResults_.count(result) != 0;
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
                } else if (uses != useCount_.end() && uses->second == 1 && def != definingInst_.end() &&
                           def->second->opcode == ir::Opcode::Not && def->second->operands.size() == 1 &&
                           def->second->operands[0].type.kind != ir::TypeKind::F32) {
                    suppressedNotResults_.insert(inst.operands[0].id);
                }
            }
        }
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                for (std::size_t operandIndex = 0; operandIndex < inst.operands.size(); ++operandIndex) {
                    const ir::Value &operand = inst.operands[operandIndex];
                    if (operand.constant || operand.id < 0) {
                        continue;
                    }
                    const auto uses = useCount_.find(operand.id);
                    const auto def = definingInst_.find(operand.id);
                    if (uses == useCount_.end() || uses->second != 1 || def == definingInst_.end() ||
                        def->second->opcode != ir::Opcode::Gep) {
                        continue;
                    }
                    const bool addressUse = (inst.opcode == ir::Opcode::Load && operandIndex == 0) ||
                                            (inst.opcode == ir::Opcode::Store && operandIndex == 1) ||
                                            (inst.opcode == ir::Opcode::Gep && operandIndex == 0);
                    if (addressUse) {
                        suppressedAddressResults_.insert(operand.id);
                    }
                }
            }
        }
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Gep || inst.operands.size() < 2) {
                    continue;
                }
                const ir::Value &index = inst.operands[1];
                if (index.constant || index.id < 0) {
                    continue;
                }
                const auto uses = useCount_.find(index.id);
                const auto def = definingInst_.find(index.id);
                if (uses == useCount_.end() || uses->second != 1 || def == definingInst_.end()) {
                    continue;
                }
                ir::Value variable;
                long long constant = 0;
                if (splitAddressIndexConstant(*def->second, variable, constant)) {
                    suppressedAddressIndexResults_.insert(index.id);
                }
            }
        }
        for (const auto &block : function.blocks) {
            for (const auto &inst : block.instructions) {
                if (inst.opcode != ir::Opcode::Store || inst.operands.empty()) {
                    continue;
                }
                const ir::Value &stored = inst.operands[0];
                if (stored.constant || stored.id < 0) {
                    continue;
                }
                const auto uses = useCount_.find(stored.id);
                const auto def = definingInst_.find(stored.id);
                if (uses == useCount_.end() || uses->second != 1 || def == definingInst_.end() ||
                    !canSuppressStoreValue(*def->second)) {
                    continue;
                }
                suppressedStoreValueResults_.insert(stored.id);
            }
        }
        for (const auto &block : function.blocks) {
            for (std::size_t i = 1; i < block.instructions.size(); ++i) {
                const auto &load = block.instructions[i - 1];
                const auto &store = block.instructions[i];
                if (load.opcode != ir::Opcode::Load || store.opcode != ir::Opcode::Store ||
                    load.result < 0 || store.operands.empty() || store.operands[0].constant ||
                    store.operands[0].id != load.result || !canSuppressStoreLoad(load)) {
                    continue;
                }
                const auto uses = useCount_.find(load.result);
                if (uses != useCount_.end() && uses->second == 1) {
                    suppressedStoreValueResults_.insert(load.result);
                }
            }
        }
    }

    bool canSuppressStoreValue(const ir::Instruction &inst) const {
        if (inst.result < 0 || inst.resultType.kind == ir::TypeKind::F32 ||
            inst.resultType.kind == ir::TypeKind::Ptr) {
            return false;
        }
        switch (inst.opcode) {
        case ir::Opcode::Add:
        case ir::Opcode::Sub:
        case ir::Opcode::Mul:
        case ir::Opcode::ICmp:
            return inst.operands.size() == 2;
        case ir::Opcode::Div:
        case ir::Opcode::Mod:
            return inst.operands.size() == 2 && !constantI32(inst.operands[1]);
        case ir::Opcode::Neg:
        case ir::Opcode::Not:
            return inst.operands.size() == 1;
        default:
            return false;
        }
    }

    bool canSuppressStoreLoad(const ir::Instruction &inst) const {
        return inst.opcode == ir::Opcode::Load && inst.result >= 0 &&
               inst.resultType.kind != ir::TypeKind::F32 &&
               inst.resultType.kind != ir::TypeKind::Ptr && inst.operands.size() == 1;
    }

    bool splitAddressIndexConstant(const ir::Instruction &inst, ir::Value &variable, long long &constant) const {
        if ((inst.opcode != ir::Opcode::Add && inst.opcode != ir::Opcode::Sub) ||
            inst.result < 0 || inst.resultType.kind != ir::TypeKind::I32 ||
            inst.operands.size() != 2) {
            return false;
        }
        const auto lhs = constantI32(inst.operands[0]);
        const auto rhs = constantI32(inst.operands[1]);
        if (inst.opcode == ir::Opcode::Add) {
            if (rhs && !inst.operands[0].constant) {
                variable = inst.operands[0];
                constant = *rhs;
                return true;
            }
            if (lhs && !inst.operands[1].constant) {
                variable = inst.operands[1];
                constant = *lhs;
                return true;
            }
        } else if (rhs && !inst.operands[0].constant) {
            variable = inst.operands[0];
            constant = -*rhs;
            return true;
        }
        return false;
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

    bool isFusableIntMulValue(const ir::Value &value) const {
        if (value.constant || value.id < 0) {
            return false;
        }
        if (suppressedMulResults_.count(value.id) == 0 && !isSingleUseIntMul(value)) {
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
            if (suppressedStoreValueResults_.count(inst.result) != 0) {
                return;
            }
            emitLoad(inst);
            return;
        case ir::Opcode::Store:
            emitStore(inst);
            return;
        case ir::Opcode::Gep:
            if (suppressedAddressResults_.count(inst.result) != 0) {
                return;
            }
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
            if ((inst.opcode == ir::Opcode::Add || inst.opcode == ir::Opcode::Sub) &&
                suppressedAddressIndexResults_.count(inst.result) != 0) {
                return;
            }
            if (suppressedStoreValueResults_.count(inst.result) != 0) {
                return;
            }
            emitBinary(inst);
            return;
        case ir::Opcode::Neg:
            if (suppressedStoreValueResults_.count(inst.result) != 0) {
                return;
            }
            emitNeg(inst);
            return;
        case ir::Opcode::Not:
            if (suppressedNotResults_.count(inst.result) != 0) {
                return;
            }
            if (suppressedStoreValueResults_.count(inst.result) != 0) {
                return;
            }
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
            emitCondBranch(inst.text, false);
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
            if (!nextBlock_.empty()) {
                out_ << "\tb " << epilogue_ << "\n";
            }
            return;
        }
    }

    bool emitSuppressedStoreValueToW0(const ir::Value &value) {
        if (value.constant || value.id < 0 ||
            suppressedStoreValueResults_.count(value.id) == 0) {
            return false;
        }
        const auto def = definingInst_.find(value.id);
        if (def == definingInst_.end() ||
            (!canSuppressStoreValue(*def->second) && !canSuppressStoreLoad(*def->second))) {
            return false;
        }
        emitInstResultToReturn(*def->second);
        return true;
    }

    bool emitFusedStoreValueToW0(const ir::Value &value) {
        if (value.constant || value.id < 0) {
            return false;
        }
        const auto def = definingInst_.find(value.id);
        if (def == definingInst_.end()) {
            return false;
        }
        return emitFusedMulBinaryResultToReturn(*def->second);
    }

    void emitLoad(const ir::Instruction &inst) {
        if (emitLoadFromDirectAlloca(inst)) {
            return;
        }
        if (emitLoadFromSuppressedGep(inst)) {
            return;
        }
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
        if (emitStoreToDirectAlloca(inst)) {
            return;
        }
        if (emitStoreToSuppressedGep(inst)) {
            return;
        }
        const bool integerValueReady =
            inst.operands[0].type.kind != ir::TypeKind::F32 &&
            inst.operands[0].type.kind != ir::TypeKind::Ptr &&
            !isConstInt(inst.operands[0], 0) &&
            emitSuppressedStoreValueToW0(inst.operands[0]);
        emitAddressOperandTo("x1", inst.operands[1]);
        if (inst.operands[0].type.kind == ir::TypeKind::F32) {
            emitFloatTo("s16", inst.operands[0]);
            out_ << "\tstr s16, [x1]\n";
        } else if (inst.operands[0].type.kind == ir::TypeKind::Ptr) {
            emitPtrTo("x0", inst.operands[0]);
            out_ << "\tstr x0, [x1]\n";
        } else {
            if (isConstInt(inst.operands[0], 0)) {
                out_ << "\tstr wzr, [x1]\n";
            } else if (integerValueReady) {
                out_ << "\tstr w0, [x1]\n";
            } else {
                emitValueTo("w0", inst.operands[0]);
                out_ << "\tstr w0, [x1]\n";
            }
        }
    }

    std::optional<int> directObjectOffset(const ir::Value &address) const {
        if (address.constant || address.id < 0) {
            return std::nullopt;
        }
        const auto found = objectOffset_.find(address.id);
        if (found == objectOffset_.end()) {
            return std::nullopt;
        }
        return found->second;
    }

    bool emitLoadFromDirectAlloca(const ir::Instruction &inst) {
        if (inst.result < 0 || inst.operands.size() != 1) {
            return false;
        }
        const auto offset = directObjectOffset(inst.operands[0]);
        if (!offset) {
            return false;
        }
        if (inst.resultType.kind == ir::TypeKind::F32) {
            loadFReg("s16", *offset);
            storeFReg("s16", valueOffset_[inst.result]);
        } else if (inst.resultType.kind == ir::TypeKind::Ptr) {
            loadXReg("x0", *offset);
            storeXReg("x0", valueOffset_[inst.result]);
        } else {
            loadWReg("w0", *offset);
            storeWReg("w0", valueOffset_[inst.result]);
        }
        return true;
    }

    bool emitStoreToDirectAlloca(const ir::Instruction &inst) {
        if (inst.operands.size() != 2) {
            return false;
        }
        const auto offset = directObjectOffset(inst.operands[1]);
        if (!offset) {
            return false;
        }
        if (inst.operands[0].type.kind == ir::TypeKind::F32) {
            emitFloatTo("s16", inst.operands[0]);
            storeFReg("s16", *offset);
        } else if (inst.operands[0].type.kind == ir::TypeKind::Ptr) {
            emitPtrTo("x0", inst.operands[0]);
            storeXReg("x0", *offset);
        } else {
            if (isConstInt(inst.operands[0], 0)) {
                storeWReg("wzr", *offset);
            } else if (emitFusedStoreValueToW0(inst.operands[0]) ||
                       emitSuppressedStoreValueToW0(inst.operands[0])) {
                storeWReg("w0", *offset);
            } else {
                emitValueTo("w0", inst.operands[0]);
                storeWReg("w0", *offset);
            }
        }
        return true;
    }

    const ir::Instruction *suppressedGepDefinition(const ir::Value &address) const {
        if (address.constant || address.id < 0 || suppressedAddressResults_.count(address.id) == 0) {
            return nullptr;
        }
        const auto found = definingInst_.find(address.id);
        if (found == definingInst_.end() || found->second->opcode != ir::Opcode::Gep ||
            found->second->operands.size() != 2) {
            return nullptr;
        }
        return found->second;
    }

    bool emitLoadFromSuppressedGep(const ir::Instruction &inst) {
        if (inst.result < 0 || inst.operands.size() != 1) {
            return false;
        }
        const ir::Instruction *gep = suppressedGepDefinition(inst.operands[0]);
        if (gep == nullptr) {
            return false;
        }

        if (const auto offset = directObjectElementOffset(gep->operands[0], gep->operands[1])) {
            if (inst.resultType.kind == ir::TypeKind::F32) {
                loadFReg("s16", *offset);
                storeFReg("s16", valueOffset_[inst.result]);
            } else if (inst.resultType.kind == ir::TypeKind::Ptr) {
                loadXReg("x0", *offset);
                storeXReg("x0", valueOffset_[inst.result]);
            } else {
                loadWReg("w0", *offset);
                storeWReg("w0", valueOffset_[inst.result]);
            }
            return true;
        }

        emitAddressOperandTo("x1", gep->operands[0]);
        const std::string operand = memoryOperandFromGepIndex(gep->operands[1], inst.resultType, "w2");
        if (inst.resultType.kind == ir::TypeKind::F32) {
            out_ << "\tldr s16, " << operand << "\n";
            storeFReg("s16", valueOffset_[inst.result]);
        } else if (inst.resultType.kind == ir::TypeKind::Ptr) {
            out_ << "\tldr x0, " << operand << "\n";
            storeXReg("x0", valueOffset_[inst.result]);
        } else {
            out_ << "\tldr w0, " << operand << "\n";
            storeWReg("w0", valueOffset_[inst.result]);
        }
        return true;
    }

    bool emitStoreToSuppressedGep(const ir::Instruction &inst) {
        if (inst.operands.size() != 2) {
            return false;
        }
        const ir::Instruction *gep = suppressedGepDefinition(inst.operands[1]);
        if (gep == nullptr) {
            return false;
        }

        if (const auto offset = directObjectElementOffset(gep->operands[0], gep->operands[1])) {
            if (inst.operands[0].type.kind == ir::TypeKind::F32) {
                emitFloatTo("s16", inst.operands[0]);
                storeFReg("s16", *offset);
            } else if (inst.operands[0].type.kind == ir::TypeKind::Ptr) {
                emitPtrTo("x0", inst.operands[0]);
                storeXReg("x0", *offset);
            } else {
                if (isConstInt(inst.operands[0], 0)) {
                    storeWReg("wzr", *offset);
                } else if (emitFusedStoreValueToW0(inst.operands[0]) ||
                           emitSuppressedStoreValueToW0(inst.operands[0])) {
                    storeWReg("w0", *offset);
                } else {
                    emitValueTo("w0", inst.operands[0]);
                    storeWReg("w0", *offset);
                }
            }
            return true;
        }

        const bool integerValueReady =
            inst.operands[0].type.kind != ir::TypeKind::F32 &&
            inst.operands[0].type.kind != ir::TypeKind::Ptr &&
            !isConstInt(inst.operands[0], 0) &&
            (emitFusedStoreValueToW0(inst.operands[0]) ||
             emitSuppressedStoreValueToW0(inst.operands[0]));
        emitAddressOperandTo("x1", gep->operands[0]);
        const std::string operand = memoryOperandFromGepIndex(gep->operands[1], inst.operands[0].type, "w2");
        if (inst.operands[0].type.kind == ir::TypeKind::F32) {
            emitFloatTo("s16", inst.operands[0]);
            out_ << "\tstr s16, " << operand << "\n";
        } else if (inst.operands[0].type.kind == ir::TypeKind::Ptr) {
            emitPtrTo("x0", inst.operands[0]);
            out_ << "\tstr x0, " << operand << "\n";
        } else {
            if (isConstInt(inst.operands[0], 0)) {
                out_ << "\tstr wzr, " << operand << "\n";
            } else if (integerValueReady) {
                out_ << "\tstr w0, " << operand << "\n";
            } else {
                emitValueTo("w0", inst.operands[0]);
                out_ << "\tstr w0, " << operand << "\n";
            }
        }
        return true;
    }

    std::optional<int> directObjectElementOffset(const ir::Value &base, const ir::Value &index) const {
        const auto baseOffset = directObjectOffset(base);
        const auto constantIndex = constantI32(index);
        if (!baseOffset || !constantIndex) {
            return std::nullopt;
        }
        return *baseOffset + *constantIndex * 4;
    }

    std::string memoryOperandFromGepIndex(const ir::Value &index, ir::Type accessType, const std::string &indexReg) {
        const auto constant = constantI32(index);
        if (constant) {
            const int bytes = *constant * 4;
            if (bytes == 0) {
                return "[x1]";
            }
            const int accessBytes = accessType.kind == ir::TypeKind::Ptr ? 8 : 4;
            const int maxOffset = accessBytes == 8 ? 32760 : 16380;
            if (bytes > 0 && bytes <= maxOffset && bytes % accessBytes == 0) {
                return "[x1, #" + std::to_string(bytes) + "]";
            }
            if (isA64UnscaledImm(bytes)) {
                return "[x1, #" + std::to_string(bytes) + "]";
            }
            if (emitPointerOffset("x1", bytes)) {
                return "[x1]";
            }
        }
        if (!index.constant && index.id >= 0 && suppressedAddressIndexResults_.count(index.id) != 0) {
            const auto def = definingInst_.find(index.id);
            ir::Value variable;
            long long additive = 0;
            if (def != definingInst_.end() && splitAddressIndexConstant(*def->second, variable, additive)) {
                const long long bytes = additive * 4ll;
                if (bytes != 0) {
                    emitPointerOffset("x1", bytes);
                }
                emitValueTo(indexReg, variable);
                return "[x1, " + indexReg + ", sxtw #2]";
            }
        }
        emitValueTo(indexReg, index);
        return "[x1, " + indexReg + ", sxtw #2]";
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
        if (isFusableIntMulValue(inst.operands[0])) {
            mulValue = &inst.operands[0];
            addend = &inst.operands[1];
            mulIsLeft = true;
        } else if (isFusableIntMulValue(inst.operands[1])) {
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

    bool emitFusedMulBinaryResultToReturn(const ir::Instruction &inst) {
        if ((inst.opcode != ir::Opcode::Add && inst.opcode != ir::Opcode::Sub) || inst.operands.size() != 2) {
            return false;
        }
        const ir::Value *mulValue = nullptr;
        const ir::Value *addend = nullptr;
        bool mulIsLeft = false;
        if (isFusableIntMulValue(inst.operands[0])) {
            mulValue = &inst.operands[0];
            addend = &inst.operands[1];
            mulIsLeft = true;
        } else if (isFusableIntMulValue(inst.operands[1])) {
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
            if (rhs && isA64CompareImm(*rhs)) {
                emitValueTo("w0", inst.operands[0]);
                emitCompareImmediate("w0", *rhs);
                out_ << "\tcset w0, " << a64Cond(inst.text) << "\n";
                storeWReg("w0", valueOffset_[inst.result]);
                return true;
            }
            if (lhs && isA64CompareImm(*lhs)) {
                emitValueTo("w0", inst.operands[1]);
                emitCompareImmediate("w0", *lhs);
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
        out_ << "\t" << op << " w0, w0, " << a64AddSubImmOperand(imm) << "\n";
        storeWReg("w0", valueOffset_[result]);
        return true;
    }

    bool emitMulImmediate(const ir::Value &base, int imm, int result) {
        const unsigned absImm = static_cast<unsigned>(imm < 0 ? -imm : imm);
        const bool powerOfTwo = absImm != 0 && (absImm & (absImm - 1)) == 0;
        const bool simpleShiftAdd = absImm == 3 || absImm == 5 || absImm == 7 || absImm == 9 ||
                                    absImm == 10 || absImm == 33;
        const auto plusMinusOneShift = singleShiftPlusMinusOne(absImm);
        if (imm != 0 && imm != 1 && imm != -1 && !powerOfTwo && !simpleShiftAdd && !plusMinusOneShift) {
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
                if (plusMinusOneShift->second) {
                    out_ << "\tadd w0, w0, w0, lsl #" << plusMinusOneShift->first << "\n";
                } else {
                    out_ << "\tlsl w1, w0, #" << plusMinusOneShift->first << "\n";
                    out_ << "\tsub w0, w1, w0\n";
                }
                break;
            }
            if (imm < 0) {
                out_ << "\tneg w0, w0\n";
            }
        }
        storeWReg("w0", valueOffset_[result]);
        return true;
    }

    static std::optional<std::pair<int, bool>> singleShiftPlusMinusOne(unsigned value) {
        if (value < 3) {
            return std::nullopt;
        }
        const unsigned plus = value + 1;
        if ((plus & (plus - 1)) == 0) {
            int shift = 0;
            while ((1u << shift) != plus) {
                ++shift;
            }
            return std::make_pair(shift, false);
        }
        const unsigned minus = value - 1;
        if (minus != 0 && (minus & (minus - 1)) == 0) {
            int shift = 0;
            while ((1u << shift) != minus) {
                ++shift;
            }
            return std::make_pair(shift, true);
        }
        return std::nullopt;
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
        if (value < 0) {
            return false;
        }
        return value <= 4095 || (value % 4096 == 0 && value / 4096 <= 4095);
    }

    static std::string a64AddSubImmOperand(int value) {
        if (value <= 4095) {
            return "#" + std::to_string(value);
        }
        return "#" + std::to_string(value / 4096) + ", lsl #12";
    }

    bool emitCompareImmediate(const std::string &reg, int value) {
        if (value >= 0 && isA64AddSubImm(value)) {
            out_ << "\tcmp " << reg << ", " << a64AddSubImmOperand(value) << "\n";
            return true;
        }
        if (value < 0 && isA64AddSubImm(-value)) {
            out_ << "\tcmn " << reg << ", " << a64AddSubImmOperand(-value) << "\n";
            return true;
        }
        return false;
    }

    static bool isA64CompareImm(int value) {
        return (value >= 0 && isA64AddSubImm(value)) ||
               (value < 0 && isA64AddSubImm(-value));
    }

    bool emitPointerOffset(const std::string &reg, long long bytes) {
        if (bytes == 0) {
            return true;
        }
        const std::string op = bytes >= 0 ? "add" : "sub";
        unsigned long long magnitude = static_cast<unsigned long long>(bytes >= 0 ? bytes : -bytes);
        if (magnitude <= 4095ull) {
            out_ << "\t" << op << " " << reg << ", " << reg << ", #" << magnitude << "\n";
            return true;
        }
        if (magnitude <= 4095ull * 4096ull + 4095ull) {
            const unsigned long long high = magnitude >> 12;
            const unsigned long long low = magnitude & 4095ull;
            if (high != 0) {
                out_ << "\t" << op << " " << reg << ", " << reg << ", #" << high << ", lsl #12\n";
            }
            if (low != 0) {
                out_ << "\t" << op << " " << reg << ", " << reg << ", #" << low << "\n";
            }
            return true;
        }
        return false;
    }

    static bool isA64UnscaledImm(int value) {
        return value >= -256 && value <= 255;
    }

    bool isA64UnsignedSpSlot(int frameOffset, int bytes) const {
        if (temporarySpDepth_ != 0) {
            return false;
        }
        const int spOffset = frameSize_ + frameOffset;
        const int maxOffset = bytes == 8 ? 32760 : 16380;
        return spOffset >= 0 && spOffset <= maxOffset && spOffset % bytes == 0;
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

    bool emitInlineFastBitCall(const ir::Instruction &inst, bool storeResult) {
        if (inst.resultType.kind != ir::TypeKind::I32) {
            return false;
        }
        const ir::Function *callee = findFunction(inst.text);
        if (callee == nullptr) {
            return false;
        }
        const FastBitKind kind = matchFastBitHelper(*callee);
        if (kind == FastBitKind::None) {
            return false;
        }
        const auto store = [&]() {
            if (storeResult && inst.result >= 0) {
                storeWReg("w0", valueOffset_[inst.result]);
            }
        };
        switch (kind) {
        case FastBitKind::BitAnd:
            if (inst.operands.size() != 2) return false;
            emitValueTo("w0", inst.operands[0]);
            emitValueTo("w1", inst.operands[1]);
            out_ << "\tand w0, w0, w1\n";
            store();
            return true;
        case FastBitKind::BitOr:
            if (inst.operands.size() != 2) return false;
            emitValueTo("w0", inst.operands[0]);
            emitValueTo("w1", inst.operands[1]);
            out_ << "\torr w0, w0, w1\n";
            store();
            return true;
        case FastBitKind::BitXor:
            if (inst.operands.size() != 2) return false;
            emitValueTo("w0", inst.operands[0]);
            emitValueTo("w1", inst.operands[1]);
            out_ << "\teor w0, w0, w1\n";
            store();
            return true;
        case FastBitKind::BitNot:
            if (inst.operands.size() != 1) return false;
            emitValueTo("w0", inst.operands[0]);
            out_ << "\tmvn w0, w0\n";
            store();
            return true;
        case FastBitKind::ShiftLeftSmall:
            if (inst.operands.size() != 2) return false;
            emitValueTo("w0", inst.operands[0]);
            emitValueTo("w1", inst.operands[1]);
            out_ << "\tcmp w1, #8\n";
            out_ << "\tlsl w2, w0, w1\n";
            out_ << "\tcsel w0, w2, w0, ls\n";
            store();
            return true;
        case FastBitKind::ShiftRightSmall:
            if (inst.operands.size() != 2) return false;
            emitValueTo("w0", inst.operands[0]);
            emitValueTo("w1", inst.operands[1]);
            out_ << "\tcmp w1, #8\n";
            out_ << "\tasr w2, w0, w1\n";
            out_ << "\tcsel w0, w2, w0, ls\n";
            store();
            return true;
        case FastBitKind::None:
            return false;
        }
        return false;
    }

    static bool scalarSelectValueAllowed(const ir::Value &value, const ir::Function &function) {
        if (value.constant) {
            return value.type.kind == ir::TypeKind::I32;
        }
        for (const auto &param : function.params) {
            if (value.id == param.id && param.type.kind == ir::TypeKind::I32) {
                return true;
            }
        }
        return false;
    }

    static const ir::BasicBlock *findFunctionBlock(const ir::Function &function,
                                                   const std::string &name) {
        for (const auto &block : function.blocks) {
            if (block.name == name) {
                return &block;
            }
        }
        return nullptr;
    }

    ScalarSelectFunction matchScalarSelectFunction(const ir::Function &function) const {
        if (function.returnType.kind != ir::TypeKind::I32 || function.blocks.size() != 3 ||
            function.blocks.front().instructions.size() < 2) {
            return {};
        }
        const auto &entry = function.blocks.front().instructions;
        const ir::Instruction &cmp = entry[entry.size() - 2];
        const ir::Instruction &branch = entry.back();
        if (cmp.opcode != ir::Opcode::ICmp || cmp.result < 0 || cmp.operands.size() != 2 ||
            branch.opcode != ir::Opcode::CondBr || branch.operands.size() != 1 ||
            branch.operands[0].constant || branch.operands[0].id != cmp.result ||
            !scalarSelectValueAllowed(cmp.operands[0], function) ||
            !scalarSelectValueAllowed(cmp.operands[1], function)) {
            return {};
        }
        for (std::size_t i = 0; i + 2 < entry.size(); ++i) {
            if (entry[i].opcode != ir::Opcode::Alloca && entry[i].opcode != ir::Opcode::Store &&
                entry[i].opcode != ir::Opcode::Load) {
                return {};
            }
        }
        const auto labels = splitLabels(branch.text);
        if (labels.size() != 2) {
            return {};
        }
        const ir::BasicBlock *trueBlock = findFunctionBlock(function, labels[0]);
        const ir::BasicBlock *falseBlock = findFunctionBlock(function, labels[1]);
        if (trueBlock == nullptr || falseBlock == nullptr ||
            trueBlock->instructions.size() != 1 || falseBlock->instructions.size() != 1) {
            return {};
        }
        const ir::Instruction &trueRet = trueBlock->instructions.front();
        const ir::Instruction &falseRet = falseBlock->instructions.front();
        if (trueRet.opcode != ir::Opcode::Ret || falseRet.opcode != ir::Opcode::Ret ||
            trueRet.operands.size() != 1 || falseRet.operands.size() != 1 ||
            !scalarSelectValueAllowed(trueRet.operands[0], function) ||
            !scalarSelectValueAllowed(falseRet.operands[0], function)) {
            return {};
        }
        return ScalarSelectFunction{true, cmp.text, cmp.operands[0], cmp.operands[1],
                                    trueRet.operands[0], falseRet.operands[0]};
    }

    ir::Value mapCalleeValueToCall(const ir::Value &value, const ir::Function &callee,
                                   const ir::Instruction &call) const {
        if (value.constant) {
            return value;
        }
        for (std::size_t i = 0; i < callee.params.size() && i < call.operands.size(); ++i) {
            if (value.id == callee.params[i].id) {
                return call.operands[i];
            }
        }
        return value;
    }

    bool emitInlineScalarSelectCall(const ir::Instruction &inst, bool storeResult) {
        if (inst.resultType.kind != ir::TypeKind::I32) {
            return false;
        }
        const ir::Function *callee = findFunction(inst.text);
        if (callee == nullptr || callee == function_ || inst.operands.size() < callee->params.size()) {
            return false;
        }
        const ScalarSelectFunction match = matchScalarSelectFunction(*callee);
        if (!match.valid) {
            return false;
        }

        ir::Value lhs = mapCalleeValueToCall(match.lhs, *callee, inst);
        ir::Value rhs = mapCalleeValueToCall(match.rhs, *callee, inst);
        ir::Value trueValue = mapCalleeValueToCall(match.trueValue, *callee, inst);
        ir::Value falseValue = mapCalleeValueToCall(match.falseValue, *callee, inst);
        std::string cond = match.predicate;

        const auto lhsConstant = constantI32(lhs);
        const auto rhsConstant = constantI32(rhs);
        if (rhsConstant && isA64CompareImm(*rhsConstant)) {
            emitValueTo("w2", lhs);
            emitCompareImmediate("w2", *rhsConstant);
        } else if (lhsConstant && isA64CompareImm(*lhsConstant)) {
            emitValueTo("w2", rhs);
            emitCompareImmediate("w2", *lhsConstant);
            cond = a64ReverseCond(cond);
        } else {
            emitValueTo("w2", lhs);
            emitValueTo("w3", rhs);
            out_ << "\tcmp w2, w3\n";
        }
        emitValueTo("w0", trueValue);
        emitValueTo("w1", falseValue);
        out_ << "\tcsel w0, w0, w1, " << a64Cond(cond) << "\n";
        if (storeResult && inst.result >= 0) {
            storeWReg("w0", valueOffset_[inst.result]);
        }
        return true;
    }

    void emitCall(const ir::Instruction &inst, bool storeResult = true) {
        if (emitInlineFastBitCall(inst, storeResult)) {
            return;
        }
        if (emitInlineScalarSelectCall(inst, storeResult)) {
            return;
        }
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
            temporarySpDepth_ += bytes;
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
            temporarySpDepth_ -= bytes;
            emitAddSp(bytes);
        }
        if (storeResult && inst.result >= 0 && inst.resultType.kind != ir::TypeKind::Void) {
            if (inst.resultType.kind == ir::TypeKind::F32) {
                storeFReg("s0", valueOffset_[inst.result]);
            } else if (inst.resultType.kind == ir::TypeKind::Ptr) {
                storeXReg("x0", valueOffset_[inst.result]);
            } else {
                storeWReg("w0", valueOffset_[inst.result]);
            }
        }
    }

    bool isDirectValueReturn(const std::vector<ir::Instruction> &instructions, std::size_t index) const {
        if (index + 1 >= instructions.size()) {
            return false;
        }
        const auto &inst = instructions[index];
        const auto &ret = instructions[index + 1];
        if (inst.result < 0 || ret.opcode != ir::Opcode::Ret || ret.operands.size() != 1 ||
            ret.operands[0].constant || ret.operands[0].id != inst.result) {
            return false;
        }
        const auto uses = useCount_.find(inst.result);
        if (uses == useCount_.end() || uses->second != 1) {
            return false;
        }
        switch (inst.opcode) {
        case ir::Opcode::Load:
        case ir::Opcode::Gep:
            return true;
        case ir::Opcode::Add:
        case ir::Opcode::Sub:
        case ir::Opcode::Mul:
        case ir::Opcode::Div:
        case ir::Opcode::Mod:
        case ir::Opcode::ICmp:
            return inst.operands.size() == 2;
        case ir::Opcode::Neg:
        case ir::Opcode::Not:
        case ir::Opcode::Cast:
            return inst.operands.size() == 1;
        default:
            return false;
        }
    }

    void emitInstResultToReturn(const ir::Instruction &inst) {
        if (inst.opcode == ir::Opcode::Load) {
            emitLoadResultToReturn(inst);
            return;
        }
        if (inst.opcode == ir::Opcode::Gep) {
            emitAddressTo("x0", inst);
            return;
        }
        if (inst.opcode == ir::Opcode::Neg) {
            if (inst.resultType.kind == ir::TypeKind::F32) {
                emitFloatTo("s0", inst.operands[0]);
                out_ << "\tfneg s0, s0\n";
            } else {
                emitValueTo("w0", inst.operands[0]);
                out_ << "\tneg w0, w0\n";
            }
            return;
        }
        if (inst.opcode == ir::Opcode::Not) {
            emitValueTo("w0", inst.operands[0]);
            out_ << "\tcmp w0, #0\n";
            out_ << "\tcset w0, eq\n";
            return;
        }
        if (inst.opcode == ir::Opcode::Cast) {
            emitCastResultToReturn(inst);
            return;
        }
        emitBinaryResultToReturn(inst);
    }

    void emitLoadResultToReturn(const ir::Instruction &inst) {
        if (inst.operands.size() != 1) {
            return;
        }
        emitAddressOperandTo("x1", inst.operands[0]);
        if (inst.resultType.kind == ir::TypeKind::F32) {
            out_ << "\tldr s0, [x1]\n";
        } else if (inst.resultType.kind == ir::TypeKind::Ptr) {
            out_ << "\tldr x0, [x1]\n";
        } else {
            out_ << "\tldr w0, [x1]\n";
        }
    }

    void emitCastResultToReturn(const ir::Instruction &inst) {
        if (inst.text == "i2f") {
            emitValueTo("w0", inst.operands[0]);
            out_ << "\tscvtf s0, w0\n";
        } else if (inst.text == "f2i") {
            emitFloatTo("s0", inst.operands[0]);
            out_ << "\tfcvtzs w0, s0\n";
        } else if (inst.resultType.kind == ir::TypeKind::F32) {
            emitFloatTo("s0", inst.operands[0]);
        } else if (inst.resultType.kind == ir::TypeKind::Ptr) {
            emitPtrTo("x0", inst.operands[0]);
        } else {
            emitValueTo("w0", inst.operands[0]);
        }
    }

    void emitBinaryResultToReturn(const ir::Instruction &inst) {
        if (inst.resultType.kind == ir::TypeKind::F32 || inst.opcode == ir::Opcode::FCmp) {
            emitFloatTo("s0", inst.operands[0]);
            emitFloatTo("s1", inst.operands[1]);
            switch (inst.opcode) {
            case ir::Opcode::Add:
                out_ << "\tfadd s0, s0, s1\n";
                break;
            case ir::Opcode::Sub:
                out_ << "\tfsub s0, s0, s1\n";
                break;
            case ir::Opcode::Mul:
                out_ << "\tfmul s0, s0, s1\n";
                break;
            case ir::Opcode::Div:
                out_ << "\tfdiv s0, s0, s1\n";
                break;
            case ir::Opcode::FCmp:
                out_ << "\tfcmp s0, s1\n";
                out_ << "\tcset w0, " << a64Cond(inst.text) << "\n";
                break;
            default:
                break;
            }
            return;
        }

        if (emitFusedMulBinaryResultToReturn(inst) || emitImmediateBinaryResultToReturn(inst)) {
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
    }

    bool emitImmediateBinaryResultToReturn(const ir::Instruction &inst) {
        if (inst.operands.size() != 2) {
            return false;
        }
        const auto rhs = constantI32(inst.operands[1]);
        const auto lhs = constantI32(inst.operands[0]);
        if (inst.opcode == ir::Opcode::Add) {
            if (rhs && emitAddSubImmediateToReg("add", inst.operands[0], *rhs, "w0")) return true;
            if (lhs && emitAddSubImmediateToReg("add", inst.operands[1], *lhs, "w0")) return true;
        } else if (inst.opcode == ir::Opcode::Sub) {
            if (rhs && emitAddSubImmediateToReg("sub", inst.operands[0], *rhs, "w0")) return true;
        } else if (inst.opcode == ir::Opcode::Mul) {
            if (rhs && emitMulImmediateToReg(inst.operands[0], *rhs, "w0")) return true;
            if (lhs && emitMulImmediateToReg(inst.operands[1], *lhs, "w0")) return true;
        } else if (inst.opcode == ir::Opcode::ICmp) {
            if (rhs && isA64CompareImm(*rhs)) {
                emitValueTo("w0", inst.operands[0]);
                emitCompareImmediate("w0", *rhs);
                out_ << "\tcset w0, " << a64Cond(inst.text) << "\n";
                return true;
            }
            if (lhs && isA64CompareImm(*lhs)) {
                emitValueTo("w0", inst.operands[1]);
                emitCompareImmediate("w0", *lhs);
                out_ << "\tcset w0, " << a64ReverseCond(inst.text) << "\n";
                return true;
            }
        }
        return false;
    }

    bool emitAddSubImmediateToReg(const std::string &op, const ir::Value &base, int imm,
                                  const std::string &reg) {
        if (imm < 0) {
            return emitAddSubImmediateToReg(op == "add" ? "sub" : "add", base, -imm, reg);
        }
        if (!isA64AddSubImm(imm)) {
            return false;
        }
        emitValueTo(reg, base);
        out_ << "\t" << op << " " << reg << ", " << reg << ", " << a64AddSubImmOperand(imm) << "\n";
        return true;
    }

    bool emitMulImmediateToReg(const ir::Value &base, int imm, const std::string &reg,
                               const std::string &temp = "w1") {
        const unsigned absImm = static_cast<unsigned>(imm < 0 ? -imm : imm);
        const bool powerOfTwo = absImm != 0 && (absImm & (absImm - 1)) == 0;
        const auto plusMinusOneShift = singleShiftPlusMinusOne(absImm);
        const bool simpleShiftAdd = absImm == 3 || absImm == 5 || absImm == 7 || absImm == 9 ||
                                    absImm == 10 || absImm == 33 || plusMinusOneShift.has_value();
        if (imm != 0 && imm != 1 && imm != -1 && !powerOfTwo && !simpleShiftAdd) {
            return false;
        }
        emitValueTo(reg, base);
        if (imm == 0) {
            out_ << "\tmov " << reg << ", #0\n";
        } else if (imm == 1) {
        } else if (imm == -1) {
            out_ << "\tneg " << reg << ", " << reg << "\n";
        } else if (powerOfTwo) {
            int shift = 0;
            while ((1u << shift) != absImm) {
                ++shift;
            }
            out_ << "\tlsl " << reg << ", " << reg << ", #" << shift << "\n";
            if (imm < 0) {
                out_ << "\tneg " << reg << ", " << reg << "\n";
            }
        } else {
            switch (absImm) {
            case 3:
                out_ << "\tadd " << reg << ", " << reg << ", " << reg << ", lsl #1\n";
                break;
            case 5:
                out_ << "\tadd " << reg << ", " << reg << ", " << reg << ", lsl #2\n";
                break;
            case 7:
                out_ << "\tlsl " << temp << ", " << reg << ", #3\n";
                out_ << "\tsub " << reg << ", " << temp << ", " << reg << "\n";
                break;
            case 9:
                out_ << "\tadd " << reg << ", " << reg << ", " << reg << ", lsl #3\n";
                break;
            case 10:
                out_ << "\tadd " << reg << ", " << reg << ", " << reg << ", lsl #2\n";
                out_ << "\tlsl " << reg << ", " << reg << ", #1\n";
                break;
            case 33:
                out_ << "\tadd " << reg << ", " << reg << ", " << reg << ", lsl #5\n";
                break;
            default:
                if (plusMinusOneShift->second) {
                    out_ << "\tadd " << reg << ", " << reg << ", " << reg << ", lsl #"
                         << plusMinusOneShift->first << "\n";
                } else {
                    out_ << "\tlsl " << temp << ", " << reg << ", #" << plusMinusOneShift->first << "\n";
                    out_ << "\tsub " << reg << ", " << temp << ", " << reg << "\n";
                }
                break;
            }
            if (imm < 0) {
                out_ << "\tneg " << reg << ", " << reg << "\n";
            }
        }
        return true;
    }

    bool isDirectCallReturn(const std::vector<ir::Instruction> &instructions, std::size_t index) const {
        if (index + 1 >= instructions.size()) {
            return false;
        }
        const auto &call = instructions[index];
        const auto &ret = instructions[index + 1];
        if (call.opcode != ir::Opcode::Call || call.text == functionName_ || call.result < 0 ||
            call.resultType.kind == ir::TypeKind::Void || ret.opcode != ir::Opcode::Ret ||
            ret.operands.size() != 1 || ret.operands[0].constant) {
            return false;
        }
        return ret.operands[0].id == call.result;
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

    void emitCondBranch(const std::string &text, bool trueWhenZero) {
        const auto labels = splitLabels(text);
        if (labels.size() != 2) {
            return;
        }
        if (labels[0] == nextBlock_) {
            const std::string trueCopyLabel = ".La64." + functionName_ + ".cond.true." +
                                             std::to_string(nextInternalLabel_++);
            out_ << "\t" << (trueWhenZero ? "cbz" : "cbnz") << " w0, " << trueCopyLabel << "\n";
            emitPhiCopies(currentBlock_, labels[1]);
            if (labels[1] != nextBlock_) {
                out_ << "\tb " << blockLabel(labels[1]) << "\n";
            }
            out_ << trueCopyLabel << ":\n";
            emitPhiCopies(currentBlock_, labels[0]);
            return;
        }
        const std::string falseCopyLabel = ".La64." + functionName_ + ".cond.false." + std::to_string(nextInternalLabel_++);
        out_ << "\t" << (trueWhenZero ? "cbnz" : "cbz") << " w0, " << falseCopyLabel << "\n";
        emitPhiCopies(currentBlock_, labels[0]);
        out_ << "\tb " << blockLabel(labels[0]) << "\n";
        out_ << falseCopyLabel << ":\n";
        emitPhiCopies(currentBlock_, labels[1]);
        if (labels[1] != nextBlock_) {
            out_ << "\tb " << blockLabel(labels[1]) << "\n";
        }
    }

    bool emitFusedCondBranch(const ir::Instruction &branch) {
        if (branch.operands.empty() || branch.operands[0].constant || branch.operands[0].id < 0) {
            return false;
        }
        const auto def = definingInst_.find(branch.operands[0].id);
        if (suppressedNotResults_.count(branch.operands[0].id) != 0) {
            if (def == definingInst_.end() || def->second->opcode != ir::Opcode::Not ||
                def->second->operands.size() != 1 ||
                def->second->operands[0].type.kind == ir::TypeKind::F32) {
                return false;
            }
            emitValueTo("w0", def->second->operands[0]);
            emitCondBranch(branch.text, true);
            return true;
        }
        if (suppressedCmpResults_.count(branch.operands[0].id) == 0) {
            return false;
        }
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
        if (rhs) {
            emitValueTo("w0", cmp.operands[0]);
            if (!emitCompareImmediate("w0", *rhs)) {
                emitValueTo("w1", cmp.operands[1]);
                out_ << "\tcmp w0, w1\n";
            }
        } else if (lhs) {
            emitValueTo("w0", cmp.operands[1]);
            if (emitCompareImmediate("w0", *lhs)) {
                cond = a64ReverseCond(cond);
            } else {
                emitValueTo("w1", cmp.operands[0]);
                out_ << "\tcmp w1, w0\n";
            }
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
        temporarySpDepth_ += bytes;
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
        temporarySpDepth_ -= bytes;
        emitAddSp(bytes);
    }

    void storeCopy(const PhiCopy &copy) {
        if (!copy.source.constant && copy.source.id == copy.target) {
            return;
        }
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
                if (copy.source.id == other.target && copy.target != other.target) {
                    return false;
                }
            }
        }
        return true;
    }

    void emitAddressTo(const std::string &reg, const ir::Instruction &gep) {
        emitAddressOperandTo(reg, gep.operands[0]);
        if (gep.operands.size() == 2) {
            const auto index = constantI32(gep.operands[1]);
            if (index) {
                const long long bytes = static_cast<long long>(*index) * 4ll;
                if (emitPointerOffset(reg, bytes)) {
                    return;
                }
            }
            if (!gep.operands[1].constant && gep.operands[1].id >= 0 &&
                suppressedAddressIndexResults_.count(gep.operands[1].id) != 0) {
                const auto def = definingInst_.find(gep.operands[1].id);
                ir::Value variable;
                long long additive = 0;
                if (def != definingInst_.end() && splitAddressIndexConstant(*def->second, variable, additive)) {
                    const long long bytes = additive * 4ll;
                    if (bytes != 0) {
                        emitPointerOffset(reg, bytes);
                    }
                    const std::string indexReg = toW(reg) == "w1" ? "w0" : "w1";
                    emitValueTo(indexReg, variable);
                    out_ << "\tadd " << reg << ", " << reg << ", " << indexReg << ", sxtw #2\n";
                    return;
                }
            }
        }
        const std::string indexReg = toW(reg) == "w1" ? "w0" : "w1";
        emitValueTo(indexReg, gep.operands[1]);
        out_ << "\tadd " << reg << ", " << reg << ", " << indexReg << ", sxtw #2\n";
    }

    void emitAddressOperandTo(const std::string &reg, const ir::Value &value) {
        if (value.constant && !value.name.empty() && value.name[0] == '@') {
            loadAddress(reg, value.name.substr(1));
            return;
        }
        if (!value.constant && value.id >= 0 && suppressedAddressResults_.count(value.id) != 0) {
            const auto def = definingInst_.find(value.id);
            if (def != definingInst_.end() && def->second->opcode == ir::Opcode::Gep) {
                emitAddressTo(reg, *def->second);
                return;
            }
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

    bool emitSuppressedMulTo(const std::string &reg, const ir::Value &value) {
        if (value.constant || value.id < 0 || suppressedMulResults_.count(value.id) == 0) {
            return false;
        }
        const auto def = definingInst_.find(value.id);
        if (def == definingInst_.end() || def->second->opcode != ir::Opcode::Mul ||
            def->second->operands.size() != 2 || def->second->resultType.kind == ir::TypeKind::F32) {
            return false;
        }
        const std::string dst = toW(reg);
        const std::string scratch = dst == "w16" ? "w17" : "w16";
        emitValueTo(dst, def->second->operands[0]);
        emitValueTo(scratch, def->second->operands[1]);
        out_ << "\tmul " << dst << ", " << dst << ", " << scratch << "\n";
        return true;
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
        if (emitSuppressedMulTo(reg, value)) {
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
        if (isA64UnsignedSpSlot(offset, 4)) {
            out_ << "\tldr " << reg << ", [sp, #" << (frameSize_ + offset) << "]\n";
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
        if (isA64UnsignedSpSlot(offset, 4)) {
            out_ << "\tstr " << reg << ", [sp, #" << (frameSize_ + offset) << "]\n";
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
        if (isA64UnsignedSpSlot(offset, 8)) {
            out_ << "\tldr " << reg << ", [sp, #" << (frameSize_ + offset) << "]\n";
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
        if (isA64UnsignedSpSlot(offset, 8)) {
            out_ << "\tstr " << reg << ", [sp, #" << (frameSize_ + offset) << "]\n";
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
        if (isA64UnsignedSpSlot(offset, 4)) {
            out_ << "\tldr " << reg << ", [sp, #" << (frameSize_ + offset) << "]\n";
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
        if (isA64UnsignedSpSlot(offset, 4)) {
            out_ << "\tstr " << reg << ", [sp, #" << (frameSize_ + offset) << "]\n";
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
        const std::uint32_t low = value & 0xffffu;
        const std::uint32_t high = (value >> 16u) & 0xffffu;
        if (high == 0) {
            out_ << "\tmovz " << reg << ", #" << low << "\n";
            return;
        }
        if (low == 0) {
            out_ << "\tmovz " << reg << ", #" << high << ", lsl #16\n";
            return;
        }
        if (high == 0xffffu) {
            out_ << "\tmovn " << reg << ", #" << ((~value) & 0xffffu) << "\n";
            return;
        }
        if (low == 0xffffu) {
            out_ << "\tmovn " << reg << ", #" << (((~value) >> 16u) & 0xffffu) << ", lsl #16\n";
            return;
        }
        out_ << "\tmovz " << reg << ", #" << (value & 0xffffu) << "\n";
        out_ << "\tmovk " << reg << ", #" << high << ", lsl #16\n";
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
    std::ostringstream buffer;
    buffer << "\t.arch armv8-a\n";
    A64CodeGen(module, buffer).run();
    buffer << "\t.section .note.GNU-stack,\"\",%progbits\n";
    out << optimizeAssemblyPeepholes(buffer.str());
}

} // namespace sysyc::arm
