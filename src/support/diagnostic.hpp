#pragma once

#include <string>

namespace sysyc {

struct SourceLocation {
    int line = 1;
    int column = 1;
};

class CompileError {
public:
    CompileError(SourceLocation location, std::string message);

    const SourceLocation &location() const;
    const std::string &message() const;
    std::string format() const;

private:
    SourceLocation location_;
    std::string message_;
};

} // namespace sysyc
