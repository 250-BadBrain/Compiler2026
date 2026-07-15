#pragma once

#include <string>
#include <vector>

namespace sysyc {

enum class TypeKind {
    Void,
    Int,
    Float,
    String,
    Array,
    Function,
    Error,
};

struct Type {
    TypeKind kind = TypeKind::Error;
    TypeKind element = TypeKind::Error;
    std::vector<int> dims;
    std::vector<Type> params;
    TypeKind returnKind = TypeKind::Error;
    bool variadic = false;

    static Type voidType();
    static Type intType();
    static Type floatType();
    static Type stringType();
    static Type errorType();
    static Type arrayType(TypeKind element, std::vector<int> dims);
    static Type functionType(TypeKind returnKind, std::vector<Type> params, bool variadic = false);

    bool isNumeric() const;
    bool isScalar() const;
    bool isArray() const;
    bool isError() const;
    std::string str() const;
};

bool sameType(const Type &lhs, const Type &rhs);
bool canConvert(const Type &from, const Type &to);

} // namespace sysyc
