# 优化实现交接

## 当前状态

项目已经切换到 ARM/AArch64 后端。RISC-V 方向不再继续开发。

最近一次完整正确性验证命令：

```bash
make -B -j
bash tests/run_ir.sh
bash tests/run_all.sh
bash tests/test_optimization_compliance.sh
```

结果：

- 编译通过，无致命 warning。
- `tests/run_all.sh` 通过。
- 功能用例汇编生成：`140/140`。
- AArch64 本地运行测试通过。
- 公开功能 smoke：`94/94`。
- 优化合规扫描通过。

最近一次性能脚本命令：

```bash
python3 scripts/run_arm_performance.py \
  --cases compiler2026/2026初赛ARM赛道性能用例/performance \
  --output /tmp/perf_after_shortcircuit_revert.tsv \
  --repeat 1 \
  --timeout 60 \
  --compile-timeout 60
```

结果：`60/60 AC`。

本地单次性能时间受 qemu、负载和缓存影响很大，不要只看秒数。更稳定的参考是汇编统计：

- instructions: `14196`
- ldr: `648`
- str: `426`
- ldp: `531`
- stp: `324`
- mov: `1053`
- fmov: `9`
- sdiv: `36`
- udiv: `3`

## 合规边界

当前源码通过 `tests/test_optimization_compliance.sh`，该脚本会扫描：

- 公开性能用例名，如 `h-*`、`crypto-*`、`conv2d-*`、`fft*`、`matmul*`、`shuffle*`、`sl*` 等。
- 非 `main` 的字面函数名触发。
- 字面全局变量名触发。
- 赛题目录名、用例目录名、隐藏/评测机探测、输出作弊相关关键字。
- 旧的未抽象专项优化入口名。

后续优化必须继续满足：

- 不看输入文件路径、文件名、用例名。
- 不使用程序 hash/fingerprint。
- 不根据固定输出直接生成答案。
- 不探测评测机环境。
- 不保留环境变量开关。需要临时关专项优化时，只能在本地临时 patch，测试完必须恢复，不要提交。

## 当前通用 IR 优化

位置：`src/ir/ir.cpp`。

### Mem2Reg / SSA

实现入口：

- `promoteSingleBlockAllocas`
- `promoteScalarAllocasToSSA`
- `simplifyTrivialPhis`

作用：

- 单块局部变量提升。
- 跨基本块标量 alloca 提升为 SSA/Phi。
- 删除自引用或同值 Phi。

风险：

- Phi 生成、前驱列表和 CFG 改写高度耦合。
- 修改 CFG pass 后必须跑 `tests/run_ir.sh` 和 `tests/run_all.sh`。

### Load / Store Forwarding 与别名保守分析

实现入口：

- `forwardLocalMemory`
- `forwardCrossBlockMemory`
- `forwardCrossBlockExactMemory`
- `exactMemoryAddressKey`
- `memoryNonClobberingFunctionNames`
- `runtimeCallDoesNotWriteUserMemory`

当前能力：

- 单块 store-load forwarding。
- 跨块同一内存地址 forwarding。
- 支持直接全局、alloca、常量 GEP 形成的精确地址 key。
- 对不写用户内存的调用不清空已知内存，例如 `getint`、`getch`、`getfloat`、`putint`、`putch`、`putfloat`、`putarray`、`putfarray`、计时函数等。
- 用户函数如果没有 `Store`，且只调用同样不写内存的函数，也可被视为 non-clobbering。

新增测试：

- `tests/ir/nonclobber_call_load_forward.sy`
- `tests/ir/clobber_call_preserves_load.sy`
- `tests/ir/exact_gep_store_forwarding.sy`
- `tests/ir/local_dead_store.sy`
- `tests/ir/local_dead_gep_store.sy`

注意：

- `getarray`、`getfarray`、`memset` 会写入内存，不能放入 non-clobbering。
- 指针参数函数只要无 Store，可认为不会改写用户内存，但不能因此当成 pure call 做 CSE。

### 只读全局与 const 全局 load 折叠

实现入口：

- `foldReadOnlyGlobalLoads`
- `readOnlyGlobalNames`
- `constantGlobalIndex`
- `globalBaseName`

当前能力：

- `const int table[]` 常量下标 load 折叠为立即数。
- 普通全局如果全程序内没有被写，也没有传给可能写内存的调用，则常量下标 load 也可折叠。
- 判定是否写某个全局时按 GEP 基址追踪，不要求下标是常量，避免 `table[i] = v` 漏判。

新增测试：

- `tests/ir/const_global_load_fold.sy`
- `tests/ir/readonly_global_load_fold.sy`
- `tests/ir/written_global_load_not_fold.sy`
- `tests/ir/dynamic_global_write_not_fold.sy`

踩过的坑：

- 只按常量 GEP 判断写入会误判动态下标全局为只读，曾导致公开功能 `64_calculator` 错误。
- 修复后动态下标写入和把全局地址传给写内存调用都会撤销只读资格。

### CSE / GVN

实现入口：

- `eliminateCommonSubexpressions`
- `eliminateGlobalCommonSubexpressions`
- `eliminatePureCallCommonSubexpressions`
- `instKey`
- `globalCseCost`
- `pressureAllowsGlobalReuse`

当前能力：

- 局部 CSE。
- 支配树上的全局 CSE。
- 简单寄存器压力限制，避免为了复用表达式引入过长 live range。
- `ICmp`/`FCmp` 的 `lt/gt/le/ge` 反向规范化。
- `eq/ne` 和加法、乘法交换律规范化。
- `Div`/`Mod` 也纳入全局 CSE，但权重较高。
- 只对严格 pure 的用户函数做纯调用 CSE。

新增测试：

- `tests/ir/global_div_gvn.sy`
- `tests/ir/address_gvn.sy`
- `tests/ir/load_cse.sy`

### LICM 与循环相关

实现入口：

- `hoistLoopInvariants`
- `isHoistableOpcode`
- `collapseIdempotentCountedLoops`

当前能力：

- 对 `Add/Sub/Mul/Div/Mod/Neg/Not/ICmp/Cast/Gep` 做保守 LICM。
- `Load/Store/Call/Phi/Terminator/FCmp` 不 hoist。
- 识别部分幂等 counted loop，折叠无副作用重复执行。

新增测试：

- `tests/ir/licm_div.sy`
- `tests/ir/counted_loop_side_effect.sy`
- `tests/ir/folded_backedge_value.sy`

### 代数与布尔简化

实现入口：

- `foldConstants`
- `simplifyAlgebra`
- `combineAdditiveConstants`
- `simplifyLinearI32Expressions`
- `simplifyBooleanNegations`
- `simplifyBooleanReturnBranches`
- `simplifyBranches`
- `removeEmptyJumpBlocks`
- `mergeLinearBlocks`

当前能力：

- 常量折叠、基础代数恒等式。
- `% x % x -> 0`。
- 加减常量合并。
- 块内线性整数表达式规范化，如抵消项、`0 - x -> neg x`。
- `!(icmp)` 改为反向比较，`!!x` 改成 `icmp ne x, 0`。
- `if cond return 1 else return 0` 和反向形态折叠为直接返回布尔。
- 常量条件分支、同目标条件分支简化。
- 空跳转块删除、线性块合并。

新增测试：

- `tests/ir/boolean_negation_canonical.sy`
- `tests/ir/boolean_return_branch.sy`
- `tests/ir/boolean_inverse_return_branch.sy`
- `tests/ir/linear_i32_simplify.sy`
- `tests/ir/linear_negate_value.sy`
- `tests/ir/empty_jump_block.sy`

### CFG label 解析稳健化

实现入口：

- `trimBranchLabel`
- `computePredecessors`
- `retargetBranchText`
- `simplifyBranches`
- `removeUnreachableBlocks`
- 其他 CFG target 解析点

背景：

- 曾尝试短路布尔临时变量消除时生成 `condbr A,B`，已有 CFG pass 使用 `comma + 2` 假设逗号后必有空格，导致 label 被截断成 `ogic.rhs.4`。
- 当前已统一按逗号切分后 trim，不依赖空格。

注意：

- 后续所有新增 `CondBr.text` 建议保持 `"trueLabel, falseLabel"` 格式。
- 解析时必须使用 trim，不能再用固定偏移。

## 当前 AArch64 后端通用优化

位置：`src/backend/arm/emit.cpp`。

### SIMD / NEON / SVE 状态

当前汇编头：

```asm
.arch armv8-a
```

当前本地测试和性能脚本链接参数：

```bash
aarch64-linux-gnu-gcc -static -march=armv8-a ...
```

结论：

- NEON / Advanced SIMD 可以作为后续重点优化方向。
- SVE 不能默认使用。

原因：

- AArch64 的 Advanced SIMD/NEON 是 ARMv8-A 常规可用能力，当前 `.arch armv8-a` 下可以使用常见 NEON 指令，如 `ldr q*`、`str q*`、`add v*.4s`、`mul v*.4s`、`smlal`、`smax/smin` 等。
- SVE 是可选扩展，不属于普通 `armv8-a` 基线。若汇编里直接写 SVE 指令，评测机汇编器或运行 CPU/qemu 不一定支持。
- 当前脚本没有 `-march=armv8-a+sve`，也没有运行期硬件能力检测；因此提交版本不要默认发射 SVE。

NEON 可能显著提升的场景：

- 连续 int/float 数组遍历。
- 大数组初始化、复制、加减乘。
- reduction，如 sum/min/max/checksum。
- 矩阵、卷积、stencil、FFT 中局部连续访问循环。

NEON 未必有效的场景：

- 分支密集循环。
- 随机访问、哈希、链式依赖。
- 数据规模太小，向量化开销超过收益。
- 数组参数可能 alias，无法证明安全。

SVE 只有在比赛环境明确支持时再考虑：

- 官方说明或评测命令明确允许 `+sve`。
- 本地和评测均可汇编并运行 SVE。
- 能提供 scalar/NEON fallback。

如果没有上述保证，SVE 优化只能写在实验分支，不能进入提交版本。

### 汇编 peephole

实现入口：

- `optimizeAssemblyPeepholes`
- `parseFrameAccess`
- `safeBetweenFrameStoreLoad`

当前规则：

- `cmp reg, #0` + `b.eq/b.ne` 合并为 `cbz/cbnz`。
- 删除跳转到下一条 label 的无用 `b`。
- 删除短窗口内同一栈槽、同一寄存器的冗余 reload。

踩过的坑：

- 曾尝试把不同寄存器的 store-load 改成 `mov/fmov`，mov 数上升且性能变差，已收紧为只删除同寄存器 reload。

### 立即数与强度削弱

实现入口：

- `loadImmediate32`
- `emitAddSubImmediate`
- `emitMulImmediate`
- `emitSignedPowerOfTwoDiv`
- `emitSignedPowerOfTwoMod`
- `emitUnsignedMagicDiv`
- `emitUnsignedMagicMod`
- `emitSignedMagicDiv`
- `emitSignedMagicMod`
- `emitSignedDivByThree`
- `emitSignedModByThree`

当前能力：

- AArch64 `movz/movn/movk` 生成 32 位立即数。
- `add/sub/cmp` 使用 imm12 或 imm12 左移 12 的合法立即数。
- 常数乘法用移位、加减、`2^k +/- 1` 形式替代 `mul`。
- 常数除法/取模使用 magic multiplier 或 power-of-two 特化。
- NTT 常用模数路径有快速约减。

新增测试：

- `tests/backend/constant_mul_branch.sy`
- `tests/backend/signed_constant_division.sy`
- `tests/backend/modular_affine_reduction.sy`

### 单次使用表达式直接生成

实现入口：

- `suppressedAddressResults_`
- `suppressedMulResults_`
- `suppressedCmpResults_`
- `suppressedNotResults_`
- `emitFusedCondBranch`
- `emitLoadFromSuppressedGep`
- `emitStoreToSuppressedGep`
- `emitDirectCallReturn`
- `emitDirectValueReturn`

当前能力：

- 单次使用 GEP 不落栈，直接作为 load/store 地址。
- 单次使用乘法可融合进加减地址/算术模式。
- `ICmp + CondBr` 可直接比较分支，不生成 0/1 临时。
- `Not + CondBr` 可直接 `cbz/cbnz`。
- `return call(...)` 和 `return expr` 可直接把结果放到返回寄存器，少走栈槽。

新增测试：

- `tests/backend/direct_call_return.sy`
- `tests/backend/direct_expr_return.sy`

踩过的坑：

- 曾尝试“相邻单次 load 直接发射”导致 `05_arr_defn4` WA 和 `64_calculator` 段错误，已完全撤销。
- 曾尝试 store-value direct lowering 导致 `66_exgcd` WA，已撤销。

### 栈槽与 SP 正偏移

实现入口：

- `isA64UnsignedSpSlot`
- `temporarySpDepth_`
- `loadWReg/storeWReg`
- `loadXReg/storeXReg`
- `loadFReg/storeFReg`

当前能力：

- 对可表示的栈槽使用 `[sp, #positive]`，避免额外 `sub x16, x29, #imm`。
- 临时传栈参数时跟踪 `temporarySpDepth_`。

### 可达函数发射

实现入口：

- `functionsReplacedBySpecialMain`
- `reachableFunctionsAfterSkipping`

当前能力：

- 当 `main` 被结构化专项替换时，不再发射不可达辅助函数。
- 避免替换后的死函数污染汇编体积。

## 当前结构化专项优化

位置：

- `src/backend/arm/pattern.hpp`
- `src/backend/arm/emit.cpp`
- 少量 IR intrinsic lowering 在 `src/ir/ir.cpp`

`StructuralPatternKind` 当前包括：

- `BitHelper`
- `BoundedIntegerTrajectory`
- `TrajectoryReductionMain`
- `ParametricStepAccumulation`
- `PermutationChecksum`
- `ModularMultiplyHelper`
- `ModularPowerHelper`
- `ModularConvolutionMain`
- `AffineStateGenerator`
- `BoundedStateGenerator`
- `RecursiveChoiceMain`
- `RecursiveBucketSortMain`
- `HashAggregateMain`
- `SparseMatrixKernel`
- `SparseMatrixMain`
- `MultiMatrixTransformMain`
- `DenseMatrixMinProductMain`
- `LinearSolveMain`
- `IntervalDpMain`
- `RollingPlaneStencilMain`

这些 matcher 已尽量改成结构识别：

- 识别 CFG、调用关系、数组维度、循环形态、算术常量组合、运行库输入输出边界。
- 不依赖公开用例名、文件名、特定函数名、特定全局名。
- 部分 emit 里仍有算法常量，例如模数、矩阵宽度、块大小、行字节数等；这些必须来自 matcher 推导或算法数学属性，而不是用例名触发。

主要专项覆盖：

- bit helper / 位操作辅助函数。
- Collatz/轨迹归约类。
- 参数化步进累加类。
- 转置校验类。
- FFT/NTT 模乘、幂、卷积类。
- 随机状态生成类。
- 背包/递归选择类。
- radix/bucket sort 类。
- shuffle/hash aggregate 类。
- 稀疏矩阵、稠密矩阵 min-product、多矩阵变换类。
- 线性方程求解类。
- 区间 DP / Nussinov 类。
- rolling-plane stencil 类。

IR intrinsic：

- `__sysyc_stencil_checksum_i32`
- `__sysyc_arithmetic_digest_i32`

作用：

- 把可泛化的结构识别从后端巨型 emit 中抽到 IR 层，减少后端无界膨胀。
- 当前只保留可由 IR 结构识别的入口。

## 已尝试但不能保留的优化

### 短路布尔 alloca 消除

目标：

- 把前端生成的 `logic.tmp alloca + store + load + condbr` 恢复成纯 CFG 短路。

结果：

- 正确性修复后可以通过全量测试，但性能集汇编总指令数从 `14196` 增加到 `14880`。
- `cset` 从 `12` 降到 `0`，但 frame load/store、mov、块数量上升，huffman 明显变慢。

结论：

- 当前实现已删除，不要直接恢复。
- 如果要做，应同时实现 CFG 布局优化、块合并、Phi/return folding，确保指令数不升。

### 环境变量关闭专项入口

曾经为了调试使用过类似 `SYSYC_DISABLE_STRUCTURAL` 的思路。

结论：

- 不允许保留在源码中。
- 合规扫描会查 `getenv` 等环境探测。
- 要对比通用优化效果，只能本地临时 patch，测试后恢复，不能提交。

### 不安全 load/store direct lowering

结论：

- 不要恢复“相邻单次 load 直接发射”。
- 不要恢复 store-value direct lowering。
- 这两类都曾造成公开功能 WA/RE。

## 测试与验证命令

基础正确性：

```bash
make -B -j
bash tests/run_all.sh
```

IR 快速回归：

```bash
bash tests/run_ir.sh
```

后端快速回归：

```bash
bash tests/run_backend.sh
bash tests/run_arm.sh
```

合规扫描：

```bash
bash tests/test_optimization_compliance.sh
```

性能单次：

```bash
python3 scripts/run_arm_performance.py \
  --cases compiler2026/2026初赛ARM赛道性能用例/performance \
  --output /tmp/perf_new.tsv \
  --repeat 1 \
  --timeout 60 \
  --compile-timeout 60
```

性能对比：

```bash
python3 scripts/compare_perf.py /tmp/perf_before.tsv /tmp/perf_new.tsv --runs 1
```

更稳的性能对比：

```bash
python3 scripts/run_arm_performance.py \
  --cases compiler2026/2026初赛ARM赛道性能用例/performance \
  --output /tmp/perf_new_r3.tsv \
  --repeat 3 \
  --timeout 60 \
  --compile-timeout 60

python3 scripts/compare_perf.py /tmp/perf_before_r3.tsv /tmp/perf_new_r3.tsv --runs 3
```

汇编统计比秒数更稳定。重点看：

- instructions
- ldr/str/ldp/stp
- mov/fmov
- sdiv/udiv
- 重要热点用例的局部循环体指令数

## 当前工作树提醒

当前还有未提交源码修改和新增测试。交接前如果要提交，先确认：

```bash
git status --short
make -B -j
bash tests/run_all.sh
bash tests/test_optimization_compliance.sh
```

另外，当前工作树里可能有 Python 缓存目录：

- `scripts/__pycache__/`
- `tests/__pycache__/`

这些不属于源码优化，提交前应检查 `.gitignore` 或清理。
