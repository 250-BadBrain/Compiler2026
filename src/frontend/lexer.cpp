#include "frontend/lexer.hpp"

#include <cctype>
#include <unordered_map>
#include <utility>

namespace sysyc {

const char *tokenKindName(TokenKind kind) {
    switch (kind) {
    case TokenKind::End:
        return "End";
    case TokenKind::Identifier:
        return "Identifier";
    case TokenKind::IntLiteral:
        return "IntLiteral";
    case TokenKind::FloatLiteral:
        return "FloatLiteral";
    case TokenKind::StringLiteral:
        return "StringLiteral";
    case TokenKind::KeywordConst:
        return "KeywordConst";
    case TokenKind::KeywordInt:
        return "KeywordInt";
    case TokenKind::KeywordFloat:
        return "KeywordFloat";
    case TokenKind::KeywordVoid:
        return "KeywordVoid";
    case TokenKind::KeywordIf:
        return "KeywordIf";
    case TokenKind::KeywordElse:
        return "KeywordElse";
    case TokenKind::KeywordWhile:
        return "KeywordWhile";
    case TokenKind::KeywordBreak:
        return "KeywordBreak";
    case TokenKind::KeywordContinue:
        return "KeywordContinue";
    case TokenKind::KeywordReturn:
        return "KeywordReturn";
    case TokenKind::Plus:
        return "Plus";
    case TokenKind::Minus:
        return "Minus";
    case TokenKind::Star:
        return "Star";
    case TokenKind::Slash:
        return "Slash";
    case TokenKind::Percent:
        return "Percent";
    case TokenKind::Bang:
        return "Bang";
    case TokenKind::AmpAmp:
        return "AmpAmp";
    case TokenKind::PipePipe:
        return "PipePipe";
    case TokenKind::Assign:
        return "Assign";
    case TokenKind::Equal:
        return "Equal";
    case TokenKind::NotEqual:
        return "NotEqual";
    case TokenKind::Less:
        return "Less";
    case TokenKind::Greater:
        return "Greater";
    case TokenKind::LessEqual:
        return "LessEqual";
    case TokenKind::GreaterEqual:
        return "GreaterEqual";
    case TokenKind::Comma:
        return "Comma";
    case TokenKind::Semicolon:
        return "Semicolon";
    case TokenKind::LParen:
        return "LParen";
    case TokenKind::RParen:
        return "RParen";
    case TokenKind::LBracket:
        return "LBracket";
    case TokenKind::RBracket:
        return "RBracket";
    case TokenKind::LBrace:
        return "LBrace";
    case TokenKind::RBrace:
        return "RBrace";
    }
    return "Unknown";
}

Lexer::Lexer(std::string source) : source_(std::move(source)) {}

std::vector<Token> Lexer::tokenize() {
    while (!eof()) {
        skipWhitespaceAndComments();
        if (eof()) {
            break;
        }

        const SourceLocation start = location_;
        const char c = peek();
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            lexIdentifierOrKeyword();
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && std::isdigit(static_cast<unsigned char>(peek(1))))) {
            lexNumber();
            continue;
        }
        if (c == '"') {
            lexString();
            continue;
        }

        switch (c) {
        case '+':
            advance();
            add(TokenKind::Plus, "+", start);
            break;
        case '-':
            advance();
            add(TokenKind::Minus, "-", start);
            break;
        case '*':
            advance();
            add(TokenKind::Star, "*", start);
            break;
        case '/':
            advance();
            add(TokenKind::Slash, "/", start);
            break;
        case '%':
            advance();
            add(TokenKind::Percent, "%", start);
            break;
        case '!':
            advance();
            if (peek() == '=') {
                advance();
                add(TokenKind::NotEqual, "!=", start);
            } else {
                add(TokenKind::Bang, "!", start);
            }
            break;
        case '&':
            advance();
            if (peek() == '&') {
                advance();
                add(TokenKind::AmpAmp, "&&", start);
            } else {
                errors_.emplace_back(start, "expected '&' after '&'");
            }
            break;
        case '|':
            advance();
            if (peek() == '|') {
                advance();
                add(TokenKind::PipePipe, "||", start);
            } else {
                errors_.emplace_back(start, "expected '|' after '|'");
            }
            break;
        case '=':
            advance();
            if (peek() == '=') {
                advance();
                add(TokenKind::Equal, "==", start);
            } else {
                add(TokenKind::Assign, "=", start);
            }
            break;
        case '<':
            advance();
            if (peek() == '=') {
                advance();
                add(TokenKind::LessEqual, "<=", start);
            } else {
                add(TokenKind::Less, "<", start);
            }
            break;
        case '>':
            advance();
            if (peek() == '=') {
                advance();
                add(TokenKind::GreaterEqual, ">=", start);
            } else {
                add(TokenKind::Greater, ">", start);
            }
            break;
        case ',':
            advance();
            add(TokenKind::Comma, ",", start);
            break;
        case ';':
            advance();
            add(TokenKind::Semicolon, ";", start);
            break;
        case '(':
            advance();
            add(TokenKind::LParen, "(", start);
            break;
        case ')':
            advance();
            add(TokenKind::RParen, ")", start);
            break;
        case '[':
            advance();
            add(TokenKind::LBracket, "[", start);
            break;
        case ']':
            advance();
            add(TokenKind::RBracket, "]", start);
            break;
        case '{':
            advance();
            add(TokenKind::LBrace, "{", start);
            break;
        case '}':
            advance();
            add(TokenKind::RBrace, "}", start);
            break;
        default:
            errors_.emplace_back(start, std::string("unexpected character '") + c + "'");
            advance();
            break;
        }
    }

    add(TokenKind::End, "", location_);
    return tokens_;
}

const std::vector<CompileError> &Lexer::errors() const {
    return errors_;
}

bool Lexer::eof() const {
    return pos_ >= source_.size();
}

char Lexer::peek(int offset) const {
    const std::size_t index = pos_ + static_cast<std::size_t>(offset);
    return index < source_.size() ? source_[index] : '\0';
}

char Lexer::advance() {
    const char c = source_[pos_++];
    if (c == '\n') {
        ++location_.line;
        location_.column = 1;
    } else {
        ++location_.column;
    }
    return c;
}

void Lexer::skipWhitespaceAndComments() {
    bool moved = true;
    while (moved && !eof()) {
        moved = false;
        while (!eof() && std::isspace(static_cast<unsigned char>(peek()))) {
            advance();
            moved = true;
        }

        if (peek() == '/' && peek(1) == '/') {
            while (!eof() && peek() != '\n') {
                advance();
            }
            moved = true;
        } else if (peek() == '/' && peek(1) == '*') {
            const SourceLocation start = location_;
            advance();
            advance();
            while (!eof() && !(peek() == '*' && peek(1) == '/')) {
                advance();
            }
            if (eof()) {
                errors_.emplace_back(start, "unterminated block comment");
                return;
            }
            advance();
            advance();
            moved = true;
        }
    }
}

void Lexer::add(TokenKind kind, std::string text, SourceLocation location) {
    tokens_.push_back(Token{kind, std::move(text), location});
}

void Lexer::lexIdentifierOrKeyword() {
    static const std::unordered_map<std::string, TokenKind> keywords = {
        {"const", TokenKind::KeywordConst},
        {"int", TokenKind::KeywordInt},
        {"float", TokenKind::KeywordFloat},
        {"void", TokenKind::KeywordVoid},
        {"if", TokenKind::KeywordIf},
        {"else", TokenKind::KeywordElse},
        {"while", TokenKind::KeywordWhile},
        {"break", TokenKind::KeywordBreak},
        {"continue", TokenKind::KeywordContinue},
        {"return", TokenKind::KeywordReturn},
    };

    const SourceLocation start = location_;
    std::string text;
    while (!eof() && (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_')) {
        text.push_back(advance());
    }

    const auto keyword = keywords.find(text);
    if (keyword != keywords.end()) {
        add(keyword->second, text, start);
    } else {
        add(TokenKind::Identifier, text, start);
    }
}

void Lexer::lexNumber() {
    const SourceLocation start = location_;
    std::string text;
    bool isFloat = false;

    if (peek() == '0' && (peek(1) == 'x' || peek(1) == 'X')) {
        text.push_back(advance());
        text.push_back(advance());
        while (!eof() && std::isxdigit(static_cast<unsigned char>(peek()))) {
            text.push_back(advance());
        }
        if (peek() == '.') {
            isFloat = true;
            text.push_back(advance());
            while (!eof() && std::isxdigit(static_cast<unsigned char>(peek()))) {
                text.push_back(advance());
            }
        }
        if (peek() == 'p' || peek() == 'P') {
            isFloat = true;
            text.push_back(advance());
            if (peek() == '+' || peek() == '-') {
                text.push_back(advance());
            }
            if (!std::isdigit(static_cast<unsigned char>(peek()))) {
                errors_.emplace_back(location_, "expected decimal exponent after hexadecimal float literal");
            }
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
                text.push_back(advance());
            }
        }
        add(isFloat ? TokenKind::FloatLiteral : TokenKind::IntLiteral, text, start);
        return;
    }

    while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
        text.push_back(advance());
    }
    if (peek() == '.') {
        isFloat = true;
        text.push_back(advance());
        while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
            text.push_back(advance());
        }
    }
    if (peek() == 'e' || peek() == 'E') {
        isFloat = true;
        text.push_back(advance());
        if (peek() == '+' || peek() == '-') {
            text.push_back(advance());
        }
        if (!std::isdigit(static_cast<unsigned char>(peek()))) {
            errors_.emplace_back(location_, "expected digit in floating-point exponent");
        }
        while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
            text.push_back(advance());
        }
    }
    add(isFloat ? TokenKind::FloatLiteral : TokenKind::IntLiteral, text, start);
}

void Lexer::lexString() {
    const SourceLocation start = location_;
    std::string text;
    text.push_back(advance());
    while (!eof() && peek() != '"') {
        if (peek() == '\n') {
            errors_.emplace_back(start, "unterminated string literal");
            return;
        }
        if (peek() == '\\') {
            text.push_back(advance());
            if (eof()) {
                errors_.emplace_back(start, "unterminated string literal");
                return;
            }
            const char escaped = peek();
            if (escaped != 'n' && escaped != 't' && escaped != '\\' && escaped != '"') {
                errors_.emplace_back(location_, std::string("unsupported escape sequence '\\") + escaped + "'");
            }
            text.push_back(advance());
            continue;
        }
        text.push_back(advance());
    }
    if (eof()) {
        errors_.emplace_back(start, "unterminated string literal");
        return;
    }
    text.push_back(advance());
    add(TokenKind::StringLiteral, text, start);
}

} // namespace sysyc
