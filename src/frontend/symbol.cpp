#include "frontend/symbol.hpp"

#include <utility>

namespace sysyc {

Scope::Scope(Scope *parent) : parent_(parent) {}

Symbol *Scope::insert(std::unique_ptr<Symbol> symbol) {
    const std::string name = symbol->name;
    auto [it, inserted] = symbols_.emplace(name, std::move(symbol));
    return inserted ? it->second.get() : nullptr;
}

Symbol *Scope::lookupCurrent(const std::string &name) const {
    const auto it = symbols_.find(name);
    return it == symbols_.end() ? nullptr : it->second.get();
}

Symbol *Scope::lookup(const std::string &name) const {
    if (auto *symbol = lookupCurrent(name)) {
        return symbol;
    }
    return parent_ ? parent_->lookup(name) : nullptr;
}

Scope *Scope::parent() const {
    return parent_;
}

} // namespace sysyc
