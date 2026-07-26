# 后续优化计划

## 阶段 1：合规性与基线固定

目标：
- 只保留 IR -> AArch64 后端路径。
- 不使用用例名、函数名、固定公开尺寸或给定输出触发优化。
- 每次优化前后都有可对比的本地基线。

实现：
1. 保持 `src/main.cpp` 默认只走 IR 后端。
2. 保持 `src/backend/arm/emit.cpp` 中旧 ARM32、AST 后端、未使用硬编码入口全部删除。
3. 只允许以下触发条件：
   - IR 指令结构。
   - 函数参数类型和调用关系。
   - 全局数组维度的通用形态。
   - 循环、访存、递归、DP、归约等抽象模式。
4. 建立固定统计脚本输出：
   - 正确性：AC/WA/CE/LE/TLE。
   - 运行时间。
   - 汇编行数。
   - 总指令数。
   - `ldr/str/ldp/stp` 数量。
   - `mov/fmov` 数量。
   - `sdiv/udiv` 数量。

验收：
```bash
bash build.sh
bash tests/run_arm.sh
```

## 阶段 2：全局寄存器分配

目标：
- 减少通用 IR 后端生成的栈访存。
- 减少 Phi copy 和普通 `mov`。

实现：
1. 构建 CFG、前驱、后继、反向后序。
2. 对每个 SSA 值计算：
   - 定义点。
   - 使用点。
   - 跨基本块活跃范围。
   - 调用点穿越情况。
3. 实现线性扫描寄存器分配：
   - caller-saved：`x9-x15`。
   - callee-saved：`x19-x28`。
   - 浮点：`v16-v31`。
4. 调用点处理：
   - 穿越调用的值优先分配 callee-saved。
   - 必须使用 caller-saved 时在调用前后保存恢复。
5. Phi coalescing：
   - Phi 目标和同类型来源值优先同寄存器。
   - 同寄存器 Phi copy 不发射 `mov`。
6. Copy coalescing：
   - `add x, y, #0`、`mov x, y`、bitcast-like copy 合并寄存器。
7. Spill 策略：
   - 按循环深度和使用次数计算 spill cost。
   - 只为实际 spill 值分配栈槽。

验收：
1. 全量功能用例 AC。
2. 全量性能用例 AC。
3. 代表用例 `ldr/str` 和 `mov/fmov` 数量下降。

## 阶段 3：IR 内存优化

目标：
- 减少重复 load/store。
- 将循环内可证明不变的地址计算移出循环。

实现：
1. Mem2Reg：
   - 只提升局部标量 alloca。
   - 构建支配树和支配边界。
   - 插入 Phi。
   - SSA 重命名。
2. 基于到达定值的 load forwarding：
   - 跟踪同一 alloca/global+index 表达式最近 store。
   - 遇到可能写内存的 call 时失效。
   - 遇到别名不明指针写入时失效。
3. Store forwarding：
   - 连续 store 同地址且中间无读取时删除旧 store。
4. 地址表达式 GVN：
   - `base + i * stride + c` 建值编号。
   - 同一基本块和支配路径复用已计算地址。
5. 循环不变地址提升：
   - 对自然循环构建 preheader。
   - 提升数组基址、行基址、常量步长。
   - 寄存器压力过高时不提升。

验收：
1. 全量性能用例 AC。
2. 代表用例 `ldr/str` 数量下降。
3. 运行时间没有系统性回退。

## 阶段 4：GVN、CSE、LICM

目标：
- 降低总指令数。
- 减少循环内重复整数运算。

实现：
1. Local CSE：
   - 同基本块内纯表达式复用。
   - 支持 `add/sub/mul/div/mod/icmp/gep/cast`。
2. Global Value Numbering：
   - 只复用支配当前点的值。
   - 对可能溢出语义保持 SysY/C int 行为。
   - 对 load 只在内存版本一致时复用。
3. LICM：
   - 识别自然循环。
   - 只提升无副作用且操作数循环不变的表达式。
   - 结合寄存器压力估计限制提升。
4. ADCE：
   - 从 return/store/call/branch 等根指令反向标记。
   - 删除无副作用死计算。
5. CFG 简化：
   - 删除不可达块。
   - 合并单前驱单后继块。
   - 删除恒真/恒假分支。

验收：
1. 全量功能和性能用例 AC。
2. 汇编总指令数下降。
3. 热点用例运行时间下降。

## 阶段 5：AArch64 指令选择优化

目标：
- 生成更接近工业编译器的 AArch64 指令形态。

实现：
1. 立即数选择：
   - `add/sub #imm`。
   - logical immediate。
   - `movz/movk` 最少段数。
2. 地址模式折叠：
   - `[base, #imm]`。
   - `[base, index, sxtw #2]`。
   - 连续访问使用 post-index。
3. Compare/branch 融合：
   - `cmp + b.cond`。
   - `cbz/cbnz`。
   - `tbz/tbnz`。
4. 乘加融合：
   - `madd`。
   - `msub`。
   - `smaddl/umaddl`。
5. 常量除法：
   - 正数除法使用 unsigned magic。
   - 一般有符号除法使用 signed magic。
   - 不能证明安全时保留 `sdiv`。
6. 栈帧优化：
   - 叶子函数省略不必要保存。
   - 相邻保存恢复合并为 `stp/ldp`。
   - 大栈调整避免重复装载常量。

验收：
1. 全量性能用例 AC。
2. `sdiv/udiv` 数量下降。
3. `mov/fmov` 数量下降。
4. 运行时间下降。

## 阶段 6：结构模式优化合规化

目标：
- 保留有效的结构化 lowering。
- 所有触发条件都来自通用 IR/数组/循环结构。

实现：
1. 每个结构优化必须拆成：
   - matcher：只识别抽象结构。
   - analyzer：从 IR 和类型推导尺寸、步长、数组角色。
   - emitter：只使用 analyzer 结果。
2. 禁止 matcher 依赖：
   - 公开用例名。
   - 固定函数名。
   - 固定全局变量名。
   - 固定公开尺寸。
3. 允许 matcher 依赖：
   - 参数个数和类型。
   - 调用图形态。
   - load/store 访问角色。
   - 循环嵌套和递推依赖。
   - 数组维度之间的关系。
4. 对现有结构化 lowering 逐个审计：
   - Collatz-like 递归深度。
   - NTT/FFT 模乘。
   - Radix-like 分桶排序。
   - Hash aggregate。
   - Matrix multiply/update。
   - LU-like 消元。
   - Nussinov-like DP。
   - Rolling-plane stencil。

验收：
1. 静态扫描不出现固定公开尺寸触发条件。
2. 全量性能用例 AC。
3. 结构化 lowering 生成汇编与原通用后端输出语义一致。

## 阶段 7：性能回归流程

目标：
- 每轮优化只保留有证据的收益。

实现：
1. 每轮优化前跑一次基线。
2. 每轮优化后跑：
   - `bash build.sh`
   - `bash tests/run_arm.sh`
   - 全量性能用例。
3. 对比输出：
   - AC 数量。
   - 总时间。
   - 每个用例时间差。
   - 指令统计差。
4. 保留条件：
   - 正确性不下降。
   - 至少 3 个性能用例时间下降。
   - 或明确减少访存/除法且没有系统性时间回退。
5. 回退条件：
   - 任一用例 WA/CE/LE/TLE。
   - 单次优化导致多数用例变慢且无明确结构性收益。

验收：
```bash
bash build.sh
bash tests/run_arm.sh
python3 scripts/run_arm_performance.py
python3 scripts/compare_perf.py before.tsv after.tsv
```
