#include "backend/riscv/emit.hpp"
#include "frontend/lexer.hpp"
#include "frontend/parser.hpp"
#include "frontend/sema.hpp"
#include "ir/builder.hpp"
#include "ir/ir.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

struct Options {
    std::string inputPath;
    std::string outputPath;
    bool emitAssembly = false;
    bool optimize = false;
    bool dumpTokens = false;
    bool parseOnly = false;
    bool semaOnly = false;
    bool dumpIr = false;
};

void printUsage(const char *argv0) {
    std::cerr << "usage: " << argv0 << " input.sysy -S -o output.s [-O1]\n";
    std::cerr << "       " << argv0 << " --dump-tokens input.sysy\n";
    std::cerr << "       " << argv0 << " --parse-only input.sysy\n";
    std::cerr << "       " << argv0 << " --sema-only input.sysy\n";
    std::cerr << "       " << argv0 << " --dump-ir input.sysy\n";
}

bool parseArgs(int argc, char **argv, Options &options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-S") {
            options.emitAssembly = true;
        } else if (arg == "-O1") {
            options.optimize = true;
        } else if (arg == "--dump-tokens") {
            options.dumpTokens = true;
        } else if (arg == "--parse-only") {
            options.parseOnly = true;
        } else if (arg == "--sema-only") {
            options.semaOnly = true;
        } else if (arg == "--dump-ir") {
            options.dumpIr = true;
        } else if (arg == "-o") {
            if (i + 1 >= argc) {
                std::cerr << "missing output path after -o\n";
                return false;
            }
            options.outputPath = argv[++i];
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "unknown option: " << arg << '\n';
            return false;
        } else if (options.inputPath.empty()) {
            options.inputPath = arg;
        } else {
            std::cerr << "unexpected extra input: " << arg << '\n';
            return false;
        }
    }

    if (options.inputPath.empty()) {
        return false;
    }
    if (options.dumpTokens || options.parseOnly || options.semaOnly || options.dumpIr) {
        return true;
    }
    if (options.outputPath.empty() || !options.emitAssembly) {
        return false;
    }
    return true;
}

bool readFile(const std::string &path, std::string &contents) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "failed to open input file: " << path << '\n';
        return false;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    contents = buffer.str();
    return true;
}

} // namespace

int main(int argc, char **argv) {
    Options options;
    if (!parseArgs(argc, argv, options)) {
        printUsage(argv[0]);
        return 1;
    }

    std::string source;
    if (!readFile(options.inputPath, source)) {
        return 1;
    }

    sysyc::Lexer lexer(source);
    auto tokens = lexer.tokenize();
    if (!lexer.errors().empty()) {
        for (const auto &error : lexer.errors()) {
            std::cerr << error.format() << '\n';
        }
        return 1;
    }

    if (options.dumpTokens) {
        for (const auto &token : tokens) {
            std::cout << token.location.line << ':' << token.location.column << ' '
                      << sysyc::tokenKindName(token.kind);
            if (!token.text.empty()) {
                std::cout << ' ' << token.text;
            }
            std::cout << '\n';
        }
        return 0;
    }

    sysyc::Parser parser(std::move(tokens));
    const auto unit = parser.parse();
    if (!parser.errors().empty()) {
        for (const auto &error : parser.errors()) {
            std::cerr << error.format() << '\n';
        }
        return 1;
    }
    if (options.parseOnly) {
        return 0;
    }

    sysyc::SemanticAnalyzer sema;
    if (!sema.analyze(unit)) {
        for (const auto &error : sema.errors()) {
            std::cerr << error.format() << '\n';
        }
        return 1;
    }
    if (options.semaOnly) {
        return 0;
    }
    if (options.dumpIr) {
        sysyc::ir::IRBuilder builder;
        sysyc::ir::dumpModule(builder.build(unit), std::cout);
        return 0;
    }

    std::ofstream out(options.outputPath);
    if (!out) {
        std::cerr << "failed to open output file: " << options.outputPath << '\n';
        return 1;
    }

    (void)options.optimize;
    sysyc::riscv::emitAssembly(unit, out);
    return 0;
}
