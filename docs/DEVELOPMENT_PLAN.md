# 开发规划

## 阶段 1：ARMv8-A 正确性基线

目标：
- 后端只输出 ARMv8-A/AArch64 GNU 汇编。
- 官方编译入口 `compiler testcase.sysy -S -o testcase.s [-O1]` 可用。
- 公开功能用例可完成汇编生成、链接和 qemu 运行。

实现：
1. 后端入口保留 `src/backend/arm/emit.cpp`，汇编头输出 `.arch armv8-a`。
2. 遵循 AAPCS64：
   - 整数和指针参数：`x0-x7`/`w0-w7`
   - 浮点参数：`s0-s7`
   - 整数返回值：`w0`
   - 指针返回值：`x0`
   - 浮点返回值：`s0`
   - 栈 16 字节对齐
3. 支持全局变量、局部数组、数组参数、函数调用、递归、短路逻辑、整数/浮点表达式。
4. 所有测试脚本默认使用：
   - `aarch64-linux-gnu-gcc`
   - `qemu-aarch64`

验收：
```bash
make -B
./tests/run_smoke.sh
./tests/run_backend.sh
./tests/run_functional.sh compiler2026/2026初赛ARM赛道功能用例
./tests/run_public_functional.sh compiler2026/2026初赛ARM赛道功能用例
```

## 阶段 2：公开性能用例正确性

目标：
- 公开性能用例全部正确运行。
- 单个用例不能生成异常巨大的汇编文件。

实现：
1. 对性能用例逐个生成 AArch64 汇编。
2. 链接 `tests/runtime/sylib.c` 后运行并对比 `.out`。
3. 检查汇编文件大小、总指令数、访存指令数、`mov` 指令数。
4. 对超大汇编优先修复：
   - 基本块重复输出
   - Phi 拷贝爆炸
   - 数组初始化展开爆炸
   - 常量表达式重复生成

验收：
```bash
./tests/run_public_functional.sh compiler2026/2026初赛ARM赛道性能用例
```

## 阶段 3：AArch64 后端寄存器分配

目标：
- 显著减少栈访存和 `mov`。
- 避免所有 SSA 值默认落栈。

实现：
1. 建立基本块级活跃区间。
2. 整数/指针值分配 `x9-x15`、`x19-x28`。
3. 浮点值分配 `s16-s31`。
4. 调用点前保存仍活跃的 caller-saved 值。
5. 使用 callee-saved 寄存器时在函数入口/出口保存恢复。
6. Phi coalescing：前驱值与目标值优先分配同一寄存器。
7. Copy coalescing：消除同寄存器 `mov/fmov`。
8. 溢出值按需落栈，不为所有临时值预留栈槽。

验收：
1. 功能和性能用例全量正确。
2. 代表性能用例访存指令数下降。
3. `mv/mov/fmov` 指令数下降。

## 阶段 4：IR 全局优化

目标：
- 降低循环内重复计算、重复访存和冗余分支。

实现：
1. Mem2Reg：
   - 构建支配树和支配边界。
   - 为可提升局部标量插入 Phi。
   - 重命名 SSA 值。
2. GVN/CSE：
   - 对纯表达式建立值编号。
   - 跨基本块复用支配路径上的等价值。
   - 对寄存器压力高的块限制表达式提升。
3. Load/store forwarding：
   - 基于到达定值跟踪同一地址最近 store。
   - 未被调用、别名写入或控制流破坏时转发 load。
4. LICM：
   - 识别自然循环。
   - 将循环不变式、数组基址、常量地址提升到 preheader。
   - 避免提升会增加寄存器压力且收益低的表达式。
5. DCE/ADCE：
   - 删除无副作用死指令。
   - 删除不可达块和不可达分支。

验收：
1. IR 优化前后语义一致。
2. 全量公开用例正确。
3. 性能用例总指令数下降。

## 阶段 5：AArch64 指令选择优化

目标：
- 接近工业编译器常见 AArch64 输出形态。

实现：
1. 立即数选择：
   - `movz/movk`
   - `add/sub #imm`
   - logical immediate
2. 地址模式折叠：
   - `[base, #imm]`
   - `[base, index, lsl #2]`
   - `adrp + add :lo12:`
3. 比较和分支融合：
   - `cmp` + `b.cond`
   - `cbz/cbnz`
   - `cset`
4. 乘加融合：
   - `madd`
   - `msub`
5. 栈帧优化：
   - 小叶子函数省略帧指针。
   - 合并相邻保存恢复为 `stp/ldp`。
   - 减少大立即数栈调整序列。

验收：
1. 全量公开用例正确。
2. 汇编文件大小下降。
3. 代表性能用例运行时间下降。
