CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
DEPFLAGS := -MMD -MP

SRC := \
	src/main.cpp \
	src/frontend/lexer.cpp \
	src/frontend/parser.cpp \
	src/frontend/type.cpp \
	src/frontend/symbol.cpp \
	src/frontend/sema.cpp \
	src/ir/ir.cpp \
	src/ir/builder.cpp \
	src/backend/riscv/emit.cpp \
	src/support/diagnostic.cpp

OBJ := $(SRC:.cpp=.o)
DEP := $(OBJ:.o=.d)
TARGET := compiler

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -Isrc -c -o $@ $<

clean:
	$(RM) $(OBJ) $(DEP) $(TARGET)

-include $(DEP)
