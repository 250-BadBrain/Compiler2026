#include "support/diagnostic.hpp"

#include <sstream>
#include <utility>

namespace sysyc {

CompileError::CompileError(SourceLocation location, std::string message)
    : location_(location), message_(std::move(message)) {}

const SourceLocation &CompileError::location() const {
    return location_;
}

const std::string &CompileError::message() const {
    return message_;
}

std::string CompileError::format() const {
    std::ostringstream os;
    os << location_.line << ':' << location_.column << ": error: " << message_;
    return os.str();
}

} // namespace sysyc
