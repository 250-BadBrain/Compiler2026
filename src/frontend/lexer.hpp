#pragma once

#include "token.hpp"
#include "../support/diagnostic.hpp"

#include <string>
#include <vector>

namespace sysyc {

class Lexer {
public:
    explicit Lexer(std::string source);

    std::vector<Token> tokenize();
    const std::vector<CompileError> &errors() const;

private:
    bool eof() const;
    char peek(int offset = 0) const;
    char advance();
    void skipWhitespaceAndComments();
    void add(TokenKind kind, std::string text, SourceLocation location);
    void lexIdentifierOrKeyword();
    void lexNumber();
    void lexString();

    std::string source_;
    std::size_t pos_ = 0;
    SourceLocation location_;
    std::vector<Token> tokens_;
    std::vector<CompileError> errors_;
};

} // namespace sysyc
