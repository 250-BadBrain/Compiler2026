# 开发规划

## 阶段 0：项目闭环

状态：已完成最小版本。

目标文件：
- `src/main.cpp`
- `src/frontend/token.hpp`
- `src/frontend/lexer.*`
- `src/frontend/parser.*`
- `src/frontend/ast.hpp`
- `src/backend/riscv/emit.*`
- `src/support/diagnostic.*`
- `tests/run_smoke.sh`

实现步骤：
1. 命令行支持 `input.sysy -S -o output.s [-O1]`。
2. 读取输入文件，失败时返回非 0。
3. Lexer 输出 token 序列，错误包含行列号。
4. Parser 构造最小 AST。
5. RISC-V emitter 输出 `.text`、`.globl main`、`main:`、`li a0, imm`、`ret`。
6. `make` 生成 `compiler`。
7. `tests/run_smoke.sh` 编译 `tests/mini/return_42.sy` 并检查汇编。

验收：
```bash
make
./tests/run_smoke.sh
```

## 阶段 1：Lexer 完整化

状态：已完成基础实现，已通过 `tests/run_lexer.sh`。

目标文件：
- `src/frontend/token.hpp`
- `src/frontend/lexer.*`
- `tests/lexer/`

Token 分类：
- EOF：`End`
- 标识符：`Identifier`
- 字面量：`IntLiteral`、`FloatLiteral`、`StringLiteral`
- 关键字：`const`、`int`、`float`、`void`、`if`、`else`、`while`、`break`、`continue`、`return`
- 运算符：`+`、`-`、`*`、`/`、`%`、`!`、`&&`、`||`、`=`、`==`、`!=`、`<`、`>`、`<=`、`>=`
- 分隔符：`,`、`;`、`(`、`)`、`[`、`]`、`{`、`}`

实现步骤：
1. 将 `TokenKind` 扩展为完整枚举。
2. `Token` 增加必要字段：
   - `kind`
   - `text`
   - `location`
3. 实现空白跳过：
   - 空格、`\t`、`\r`、`\n`
   - 正确维护行列号
4. 实现注释：
   - `//` 到行尾
   - `/* ... */`
   - 未闭合块注释报错
5. 实现标识符和关键字：
   - `[A-Za-z_][A-Za-z0-9_]*`
   - keyword 表驱动识别
6. 实现整数：
   - 十进制
   - 八进制
   - 十六进制 `0x` / `0X`
7. 实现浮点：
   - 小数形式：`1.0`、`.5`、`1.`
   - 指数形式：`1e3`、`1.0e-3`
   - 十六进制浮点按 C 规则保留文本，后续常量求值解析
8. 实现字符串：
   - 只用于 `putf` 格式串
   - 支持常见转义：`\n`、`\t`、`\\`、`\"`
9. 实现双字符运算符优先匹配：
   - `&&`、`||`、`==`、`!=`、`<=`、`>=`
10. 未知字符报错后继续扫描。

验收：
1. 新增 lexer 单元测试：
   - `tests/lexer/token_basic.sy`
   - `tests/lexer/comment.sy`
   - `tests/lexer/number.sy`
   - `tests/lexer/string.sy`
2. 测试脚本能 dump token 并比对期望。
3. 公开功能用例的 lexer 阶段无误报。

## 阶段 2：AST 和 Parser 完整化

状态：已完成基础实现，已通过 `tests/run_parser.sh` 和公开功能用例 parse-only 扫描。

目标文件：
- `src/frontend/ast.hpp`
- `src/frontend/parser.*`
- `tests/parser/`

AST 类型：
- `TranslationUnit`
- `Decl`
- `ConstDecl`
- `VarDecl`
- `VarDef`
- `InitVal`
- `FuncDef`
- `FuncParam`
- `Block`
- `Stmt`
- `Expr`
- `LValue`

表达式节点：
- `IntegerLiteral`
- `FloatLiteral`
- `StringLiteral`
- `DeclRefExpr`
- `ArraySubscriptExpr`
- `CallExpr`
- `UnaryExpr`
- `BinaryExpr`

语句节点：
- `BlockStmt`
- `DeclStmt`
- `ExprStmt`
- `AssignStmt`
- `IfStmt`
- `WhileStmt`
- `BreakStmt`
- `ContinueStmt`
- `ReturnStmt`

实现步骤：
1. 用 `std::unique_ptr` 管理 AST 节点。
2. 给所有 AST 节点保存 `SourceLocation`。
3. Parser 入口实现 `parseTranslationUnit()`：
   - 循环解析顶层 `Decl` 或 `FuncDef`
   - 通过 `BType Ident (` 判断函数定义
4. 实现声明解析：
   - `ConstDecl`
   - `VarDecl`
   - 多个 `VarDef` 逗号分隔
   - 数组维度列表
   - 初始化树
5. 实现函数解析：
   - `void/int/float` 返回类型
   - 形参列表
   - 数组形参首维空
6. 实现块解析：
   - `{ BlockItem* }`
   - `BlockItem = Decl | Stmt`
7. 实现语句解析：
   - 赋值语句需要区分 `LVal = Exp ;` 和普通表达式语句
   - `if` 使用最近 `else`
   - `while`
   - `break` / `continue`
   - `return`
8. 实现 Pratt parser 或递归下降表达式：
   - `PrimaryExp`
   - `UnaryExp`
   - `MulExp`
   - `AddExp`
   - `RelExp`
   - `EqExp`
   - `LAndExp`
   - `LOrExp`
9. 保留短路表达式结构，不在 parser 阶段降成普通二元运算。
10. 解析失败时跳到同步点：
   - `;`
   - `}`
   - 顶层声明起始 token

验收：
1. `tests/parser/` 覆盖：
   - 声明
   - 函数
   - if/while
   - 表达式优先级
   - 数组初始化
2. 对所有公开功能 `.sy` 执行 parse-only，无崩溃。

## 阶段 3：类型系统、符号表、语义分析

状态：已完成基础实现，已通过 `tests/run_sema.sh` 和 140 个公开功能用例 sema-only 扫描。

目标文件：
- `src/frontend/type.hpp`
- `src/frontend/symbol.hpp`
- `src/frontend/sema.*`
- `src/frontend/const_eval.*`
- `tests/sema/`

类型设计：
```text
Type
  kind: Void | Int | Float | Array | Function
ArrayType
  element: Type*
  dims: vector<int>
FunctionType
  returnType: Type*
  params: vector<Type*>
```

符号设计：
```text
Symbol
  name
  kind: Var | Const | Func | Param
  type
  location
  isGlobal
  constValue optional
```

实现步骤：
1. 实现 `Scope`：
   - `parent`
   - `unordered_map<string, Symbol*>`
   - `insert`
   - `lookupCurrent`
   - `lookup`
2. 预置运行库函数：
   - `getint`
   - `getch`
   - `getfloat`
   - `getarray`
   - `getfarray`
   - `putint`
   - `putch`
   - `putfloat`
   - `putarray`
   - `putfarray`
   - `putf`
   - `starttime`
   - `stoptime`
3. 顶层检查：
   - 必须存在唯一 `int main()`
   - 顶层变量、常量、函数不能重名
4. 声明检查：
   - 先定义后使用
   - 同一作用域不能重复定义
   - `const` 必须初始化
   - 数组维度必须是非负整数常量
5. 初始化检查：
   - 标量初始化不能使用 `{...}`
   - 数组初始化递归展平
   - 缺失元素补 0
   - 元素数量不能超过总元素数
   - 全局变量初始化必须是常量表达式
6. 表达式检查：
   - 给每个表达式标注类型
   - `LVal` 左值检查
   - 数组下标必须为 int
   - 数组实参可以退化为地址
   - 函数调用参数数量和类型匹配
7. 隐式转换：
   - int 到 float
   - float 到 int
   - 条件表达式转 bool 语义，结果仍按 int 0/1 处理
8. 控制流语义：
   - `break` / `continue` 必须在循环内
   - `void` 函数不能返回值
   - `int/float` 函数返回表达式可隐式转换
9. 常量求值：
   - 整数运算
   - 浮点运算
   - 关系和逻辑运算
   - 常量数组元素引用
10. 语义产物：
   - AST 节点绑定 `Type*`
   - `DeclRefExpr` / `LValue` 绑定 `Symbol*`
   - 初始化树变成线性初始化值

验收：
1. `tests/sema/` 覆盖重名、未定义、类型错误、初始化错误。
2. 公开功能用例通过 sema-only。

## 阶段 4：IR 设计和生成

状态：已完成文本 IR 初版，已通过 `tests/run_ir.sh` 和 140 个公开功能用例 IR 生成扫描。

目标文件：
- `src/ir/ir.hpp`
- `src/ir/module.hpp`
- `src/ir/builder.*`
- `src/ir/dump.*`
- `tests/ir/`

IR 数据结构：
```text
Module
  globals
  functions

Function
  name
  returnType
  params
  basicBlocks

BasicBlock
  name
  instructions
  predecessors
  successors

Value
  id
  type

Instruction : Value
  opcode
  operands
```

指令集合：
- `Alloca`
- `Load`
- `Store`
- `GlobalAddr`
- `Gep`
- `Binary`
- `Unary`
- `Cast`
- `Cmp`
- `Call`
- `Br`
- `CondBr`
- `Return`
- `Phi`，先预留，SSA 阶段使用

实现步骤：
1. 建立 IR 类型：
   - `I32`
   - `F32`
   - `Void`
   - `Pointer`
   - `Array`
2. 建立 `IRBuilder`：
   - 当前函数
   - 当前基本块
   - 临时 value 创建
3. 生成全局对象：
   - 全局变量
   - 全局常量
   - 字符串常量
4. 生成函数：
   - 参数映射到符号
   - 局部变量 `Alloca`
   - 入口块初始化参数
5. 生成表达式：
   - 算术
   - 比较
   - 逻辑短路
   - 函数调用
   - 数组下标地址
   - lvalue/rvalue 区分
6. 生成语句：
   - `if`：then、else、merge
   - `while`：cond、body、exit
   - `break` / `continue` 使用循环栈
   - `return`
7. 生成初始化：
   - 全局初始化落到数据段描述
   - 局部数组初始化生成 store 序列
8. IR dump：
   - 稳定文本格式
   - 用于测试和调试

验收：
1. `tests/ir/` 比对 IR dump。
2. 基础功能用例可生成 IR。

## 阶段 5：朴素 RISC-V 后端

状态：已完成。已支持整数、float/FPU、数组、短路、递归、大偏移栈帧、超过 8 个整数/浮点/混合参数、32 位 int 算术语义和常量 RHS 指令选择。

目标文件：
- `src/backend/riscv/isa.hpp`
- `src/backend/riscv/machine_ir.hpp`
- `src/backend/riscv/isel.*`
- `src/backend/riscv/frame.*`
- `src/backend/riscv/emit.*`
- `tests/backend/`

实现步骤：
1. 定义机器寄存器：
   - 整数：`zero`、`ra`、`sp`、`gp`、`tp`、`t0-t6`、`s0-s11`、`a0-a7`
   - 浮点：`ft0-ft11`、`fs0-fs11`、`fa0-fa7`
2. 定义机器指令：
   - 算术：`add`、`sub`、`mul`、`div`、`rem`
   - 立即数：`addi`、`li`
   - 访存：`lw`、`sw`、`flw`、`fsw`、`ld`、`sd`
   - 分支：`beq`、`bne`、`blt`、`bge`、`j`
   - 调用：`call`、`ret`
   - 浮点：`fadd.s`、`fsub.s`、`fmul.s`、`fdiv.s`
   - 转换：`fcvt.s.w`、`fcvt.w.s`
3. 先实现栈槽分配：
   - 每个 IR value 分配栈槽
   - 每条指令从栈加载操作数，计算后写回栈
4. 实现栈帧：
   - 保存 `ra`
   - 保存用到的 callee-saved 寄存器
   - 栈 16 字节对齐
   - 大 offset 使用临时寄存器计算地址
5. 实现函数调用 ABI：
   - 前 8 个整数或指针参数放 `a0-a7`
   - 前 8 个浮点参数放 `fa0-fa7`
   - 多余参数放调用者栈区
   - 返回值从 `a0` 或 `fa0` 取
6. 实现全局寻址：
   - 使用 `la reg, symbol`
   - 后续必要时改成显式 `%pcrel_hi/%pcrel_lo`
7. 实现数据段：
   - `.data`
   - `.bss`
   - `.rodata`
   - `.word`
   - `.zero`
   - 字符串 `.asciz`
8. 实现数组地址：
   - 行主序
   - `offset = (((i0 * dim1 + i1) * dim2 + i2)...) * elemSize`
9. 实现浮点字面量：
   - 放入 `.rodata`
   - 通过地址加载
10. 汇编输出稳定化：
   - 函数标签
   - 基本块标签
   - `.size`

验收：
1. 支持整数变量、算术、if、while、函数调用。
2. 支持数组读写。
3. 支持 float 基础运算。
4. `tests/run_all.sh` 通过。
5. 公开功能用例 140/140 通过 RISC-V 链接运行比对。
6. 公开性能用例 60/60 通过 RISC-V 链接运行比对。

## 阶段 6：功能用例全通

状态：已完成。

目标文件：
- `tests/run_functional.sh`
- `tests/run_public_functional.sh`
- `tests/runtime/sylib.c`

实现步骤：
1. 编写批量测试脚本：
   - 支持传入用例根目录
   - 递归查找 `.sy`
   - 使用独立临时目录
2. 每个用例执行：
   - 生成 `.s`
   - 汇编链接
   - 如有 `.in`，重定向输入
   - 捕获 stdout
   - 比对 `.out`
   - 检查程序返回值
3. 输出汇总：
   - passed
   - failed compile
   - failed assemble
   - failed run
   - wrong answer
4. 修复顺序：
   - parser 崩溃
   - sema 误判
   - 后端汇编错误
   - 运行时错误
   - 输出错误
5. 已修复问题：
   - float 数组实参按地址寄存器传递
   - 调用实参按函数形参 ABI 分类
   - 数组形参维度使用常量求值
   - 大调用栈区使用大立即数安全展开
   - 超过 8 个混合参数按调用顺序读取栈参数
   - RV64 后端使用 `addw/subw/mulw/divw/remw` 保持 32 位 int 语义
   - 本地 runtime 的 `putarray` / `putfarray` 输出换行

验收：
1. 公开普通功能 100 个全通。
2. 公开高阶功能 40 个全通。
3. 公开性能用例 60 个正确性全通。
4. smoke、lexer、parser、sema、ir、backend、functional 全部可一键运行。

## 阶段 7：基础 IR 优化

状态：下一阶段。

目标文件：
- `src/ir/pass.hpp`
- `src/ir/pass_manager.*`
- `src/ir/passes/`
- `tests/opt/`

Pass 顺序：
1. `SimplifyCFG`
2. `ConstantFold`
3. `AlgebraicSimplify`
4. `DeadCodeElimination`
5. `LocalValueNumbering`
6. `CopyPropagation`
7. `CommonSubexprElimination`
8. `GlobalValueNumbering`
9. `LoopInvariantCodeMotion`

实现步骤：
1. 建立 `PassManager`：
   - `run(Module&)`
   - 每个 pass 返回是否修改
   - 修改后可循环运行固定点
2. `SimplifyCFG`：
   - 删除不可达块
   - 合并只有一个前驱和一个后继的块
   - 常量条件分支转无条件分支
3. `ConstantFold`：
   - 整数常量计算
   - 浮点常量计算
   - 比较常量计算
   - cast 常量计算
4. `AlgebraicSimplify`：
   - `x + 0 -> x`
   - `x - 0 -> x`
   - `x * 1 -> x`
   - `x * 0 -> 0`
   - `x / 1 -> x`
   - `x && true -> x`
   - `x || false -> x`
5. `DeadCodeElimination`：
   - 删除无副作用且结果未使用的指令
   - 保留 `Store`、`Call`、terminator
6. `LocalValueNumbering`：
   - 基本块内表达式编号
   - 替换重复计算
7. `CopyPropagation`：
   - 消除等价 move/cast
8. `CommonSubexprElimination`：
   - 在支配关系允许时复用表达式
9. `LoopInvariantCodeMotion`：
   - 构造 loop info
   - 找 preheader
   - 外提无副作用且操作数循环外不变的指令

验收：
1. 所有功能测试仍全通。
2. `-O1` 开启 pass pipeline。
3. `tests/opt/` 比对优化前后 IR。

## 阶段 8：寄存器分配

目标文件：
- `src/backend/riscv/liveness.*`
- `src/backend/riscv/reg_alloc.*`
- `tests/regalloc/`

实现步骤：
1. 在机器基本块上计算：
   - `use`
   - `def`
   - `liveIn`
   - `liveOut`
2. 实现线性扫描寄存器分配：
   - 建立 live interval
   - 分配 caller-saved 优先
   - 跨调用活跃值优先使用 callee-saved 或 spill
3. 整数和浮点分别分配：
   - int/pointer value 用 GPR
   - float value 用 FPR
4. Spill 策略：
   - 选择结束点最远的 interval
   - 插入 load/store
   - 复用栈槽
5. 调用点处理：
   - 保存调用后仍活跃的 caller-saved 寄存器
   - 恢复后继续使用
6. 后端清理：
   - 删除冗余 move
   - 合并相邻 load/store

验收：
1. 功能测试全通。
2. `92_register_alloc.sy`、`many_locals`、`many_params` 类用例通过。
3. 性能用例相对朴素栈分配有明显提升。

## 阶段 9：循环和数组性能优化

目标文件：
- `src/ir/analysis/loop_info.*`
- `src/ir/analysis/dominators.*`
- `src/ir/passes/induction.*`
- `src/ir/passes/strength_reduce.*`
- `src/ir/passes/loop_unroll.*`

实现步骤：
1. Dominator tree：
   - 计算 immediate dominator
   - 提供 `dominates(a, b)`
2. LoopInfo：
   - 识别 back edge
   - 找 header
   - 收集 loop blocks
   - 构造 preheader
3. 归纳变量识别：
   - 初值
   - 步长
   - 循环边界
4. 强度削弱：
   - 数组下标乘法改为地址递增
   - 常量乘除优化
5. 循环展开：
   - 小固定次数完全展开
   - 常规循环按 2 或 4 展开
   - 生成 remainder loop
6. 数组访问优化：
   - 公共基址外提
   - 连续访问改为指针递增
7. 函数内联：
   - 小函数
   - 非递归
   - 调用次数有限
   - 控制代码体积

验收：
1. 功能测试全通。
2. 性能用例生成结果正确。
3. 矩阵、转置、卷积类用例运行时间下降。

## 阶段 10：提交前检查

目标文件：
- `README.md`
- `docs/DEVELOPMENT_PLAN.md`
- `docs/DESIGN.md`
- `tests/run_all.sh`

实现步骤：
1. 补 `docs/DESIGN.md`：
   - 架构
   - 前端
   - IR
   - 后端
   - 优化
   - 测试
2. 补 `tests/run_all.sh`：
   - smoke
   - unit
   - functional
   - performance correctness
3. 清理生成物：
   - `compiler`
   - `*.o`
   - `*.s`
   - 临时输出
4. 检查构建：
   - clean tree 下 `make`
   - 产物名为 `compiler`
5. 检查命令：
   - `./compiler testcase.sysy -S -o testcase.s`
   - `./compiler testcase.sysy -S -o testcase.s -O1`
6. 检查内层赛题目录：
   - 只作为资料目录
   - 不作为子 Git 仓库

验收：
1. `make clean && make` 通过。
2. `tests/run_all.sh` 通过。
3. `git status` 只包含应提交源码、测试和必要文档。

## 下一次开发入口

按顺序执行：
1. 阶段 7：基础 IR 优化。
2. 阶段 8：寄存器分配。
3. 阶段 9：循环和数组性能优化。
