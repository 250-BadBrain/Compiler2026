#include "frontend/type.hpp"

#include <sstream>

namespace sysyc {

Type Type::voidType() {
    Type type;
    type.kind = TypeKind::Void;
    return type;
}

Type Type::intType() {
    Type type;
    type.kind = TypeKind::Int;
    return type;
}

Type Type::floatType() {
    Type type;
    type.kind = TypeKind::Float;
    return type;
}

Type Type::stringType() {
    Type type;
    type.kind = TypeKind::String;
    return type;
}

Type Type::errorType() {
    Type type;
    type.kind = TypeKind::Error;
    return type;
}

Type Type::arrayType(TypeKind element, std::vector<int> dims) {
    Type type;
    type.kind = TypeKind::Array;
    type.element = element;
    type.dims = std::move(dims);
    return type;
}

Type Type::functionType(TypeKind returnKind, std::vector<Type> params, bool variadic) {
    Type type;
    type.kind = TypeKind::Function;
    type.returnKind = returnKind;
    type.params = std::move(params);
    type.variadic = variadic;
    return type;
}

bool Type::isNumeric() const {
    return kind == TypeKind::Int || kind == TypeKind::Float;
}

bool Type::isScalar() const {
    return isNumeric();
}

bool Type::isArray() const {
    return kind == TypeKind::Array;
}

bool Type::isError() const {
    return kind == TypeKind::Error;
}

std::string Type::str() const {
    switch (kind) {
    case TypeKind::Void:
        return "void";
    case TypeKind::Int:
        return "int";
    case TypeKind::Float:
        return "float";
    case TypeKind::String:
        return "string";
    case TypeKind::Error:
        return "<error>";
    case TypeKind::Array: {
        std::ostringstream os;
        os << (element == TypeKind::Float ? "float" : "int");
        for (int dim : dims) {
            os << '[' << dim << ']';
        }
        return os.str();
    }
    case TypeKind::Function:
        return "<function>";
    }
    return "<unknown>";
}

bool sameType(const Type &lhs, const Type &rhs) {
    if (lhs.kind != rhs.kind) {
        return false;
    }
    if (lhs.kind == TypeKind::Array) {
        return lhs.element == rhs.element && lhs.dims == rhs.dims;
    }
    if (lhs.kind == TypeKind::Function) {
        if (lhs.returnKind != rhs.returnKind || lhs.params.size() != rhs.params.size()) {
            return false;
        }
        for (std::size_t i = 0; i < lhs.params.size(); ++i) {
            if (!sameType(lhs.params[i], rhs.params[i])) {
                return false;
            }
        }
    }
    return true;
}

bool canConvert(const Type &from, const Type &to) {
    if (from.isError() || to.isError()) {
        return true;
    }
    if (sameType(from, to)) {
        return true;
    }
    if (from.kind == TypeKind::Array && to.kind == TypeKind::Array) {
        if (from.element != to.element || from.dims.size() < to.dims.size()) {
            return false;
        }
        for (std::size_t i = 0; i < to.dims.size(); ++i) {
            if (to.dims[i] >= 0 && from.dims[i] != to.dims[i]) {
                return false;
            }
        }
        return true;
    }
    return from.isNumeric() && to.isNumeric();
}

} // namespace sysyc
