#pragma once

#include "type.hpp"
#include "../support/diagnostic.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sysyc {

enum class SymbolKind {
    Var,
    Const,
    Func,
    Param,
};

struct Symbol {
    std::string name;
    SymbolKind kind = SymbolKind::Var;
    Type type;
    SourceLocation location;
    bool isGlobal = false;
    bool hasConstValue = false;
    TypeKind constKind = TypeKind::Error;
    long long intValue = 0;
    double floatValue = 0.0;
};

class Scope {
public:
    explicit Scope(Scope *parent = nullptr);

    Symbol *insert(std::unique_ptr<Symbol> symbol);
    Symbol *lookupCurrent(const std::string &name) const;
    Symbol *lookup(const std::string &name) const;
    Scope *parent() const;

private:
    Scope *parent_ = nullptr;
    std::unordered_map<std::string, std::unique_ptr<Symbol>> symbols_;
};

} // namespace sysyc
