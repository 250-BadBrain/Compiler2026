#pragma once

#include "../support/diagnostic.hpp"

#include <string>

namespace sysyc {

enum class TokenKind {
    End,
    Identifier,
    IntLiteral,
    FloatLiteral,
    StringLiteral,
    KeywordConst,
    KeywordInt,
    KeywordFloat,
    KeywordVoid,
    KeywordIf,
    KeywordElse,
    KeywordWhile,
    KeywordBreak,
    KeywordContinue,
    KeywordReturn,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Bang,
    AmpAmp,
    PipePipe,
    Assign,
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    Comma,
    Semicolon,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
};

struct Token {
    TokenKind kind = TokenKind::End;
    std::string text;
    SourceLocation location;
};

const char *tokenKindName(TokenKind kind);

} // namespace sysyc
