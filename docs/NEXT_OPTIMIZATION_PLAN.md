# 后续优化计划

目标：保留当前结构化专项优化，同时补强未知用例也能受益的通用优化。优先减少访存、`mov`、总指令数，其次减少除法、分支和调用开销。

## 每轮迭代流程

1. 建立基线。

```bash
make -B -j
bash tests/run_all.sh
bash tests/test_optimization_compliance.sh
python3 scripts/run_arm_performance.py \
  --cases compiler2026/2026初赛ARM赛道性能用例/performance \
  --output /tmp/perf_before.tsv \
  --repeat 1 \
  --timeout 60 \
  --compile-timeout 60
```

2. 选择一个通用优化方向，只改一类问题。

3. 添加最小 IR/backend 回归测试。

4. 跑正确性。

```bash
make -B -j
bash tests/run_ir.sh
bash tests/run_backend.sh
bash tests/run_all.sh
bash tests/test_optimization_compliance.sh
```

5. 跑性能回归。

```bash
python3 scripts/run_arm_performance.py \
  --cases compiler2026/2026初赛ARM赛道性能用例/performance \
  --output /tmp/perf_after.tsv \
  --repeat 1 \
  --timeout 60 \
  --compile-timeout 60

python3 scripts/compare_perf.py /tmp/perf_before.tsv /tmp/perf_after.tsv --runs 1
```

6. 判断是否保留。

保留条件：

- 全量正确性通过。
- 合规扫描通过。
- 汇编统计不变差，或明确减少目标指标。
- 若秒数变差但指令统计改善，需要 repeat 3 复测。

撤回条件：

- 任一公开功能或性能用例 WA/RE/CE。
- 指令数明显上升且没有补偿收益。
- 触发合规扫描。
- 只能靠用例名、固定维度、固定函数名解释收益。

## 如何临时关闭专项优化进行测试

目的：观察通用优化在普通后端路径上的真实效果，避免被结构化专项 emit 覆盖。

允许方式：

- 本地临时 patch `detectPattern` 或 `emitFunction`，让 `StructuralPatternKind` 始终为 `None`。
- 或临时注释 `functionsReplacedBySpecialMain` 的跳过逻辑。
- 测试完成后必须恢复，不要提交。

禁止方式：

- 不要加入环境变量开关。
- 不要加入命令行隐藏参数。
- 不要根据输入路径或用例目录关闭/打开优化。

建议对比：

```bash
# 正常路径
python3 scripts/run_arm_performance.py --cases compiler2026/2026初赛ARM赛道性能用例/performance --output /tmp/perf_structural_on.tsv --repeat 1 --timeout 60 --compile-timeout 60

# 临时 patch 关闭专项后
python3 scripts/run_arm_performance.py --cases compiler2026/2026初赛ARM赛道性能用例/performance --output /tmp/perf_structural_off.tsv --repeat 1 --timeout 60 --compile-timeout 60
```

如果通用优化只在 `structural_off` 下改善，而 `structural_on` 不变，也可以保留，因为隐藏用例可能走普通路径。

## 优先级 1：通用后端栈槽访存削减

现状：

- 当前性能集统计中 frame load/store 仍然很多。
- 通用后端仍以栈槽为主，很多 IR 临时值 store 后很快 load。

建议实现：

- 基本块内轻量寄存器缓存。
- 记录最近一次写入栈槽的物理寄存器。
- 后续读取同一栈槽时，如果寄存器未被 clobber，直接复用寄存器。
- 对不同寄存器目标，不要盲目插入 `mov`；只有减少至少一次内存访问且不会增加关键路径时才替换。
- 函数调用、写同一寄存器、未知内存写、分支 label 必须清空缓存。

测试：

- 新增 backend 小用例覆盖同寄存器 reload 删除。
- 新增负例覆盖调用后不能复用 caller-saved 寄存器。
- 跑公开功能 `05_arr_defn4`、`64_calculator`、`66_exgcd`，这些曾经暴露过不安全 direct lowering。

风险：

- AArch64 caller-saved/callee-saved 规则必须正确。
- `w0/x0` 作为调用返回和参数寄存器很容易被 clobber。
- 浮点 `s/d` 寄存器要单独处理。

## 优先级 2：局部值编号和跨块 GVN 增强

现状：

- 已有局部 CSE 和简单全局 CSE。
- 仍缺完整内存版本号和别名集，Load CSE 保守。

建议实现：

- 为每个基本块维护 `Expr -> Value` available map。
- 为内存维护抽象版本：
  - 全局基址版本。
  - alloca 基址版本。
  - unknown memory version。
- `Store` 只 kill 可能 alias 的 load 表达式。
- `Call` 根据 `memoryNonClobberingFunctionNames` 或 runtime 分类 kill。
- `Load` 的 key 使用 `exactMemoryAddressKey`，无法精确时保守。
- 合并块入口 available 时只保留所有前驱同值表达式。

收益目标：

- 减少重复 GEP、Load、Div、Mod、ICmp。
- 给后端减少栈槽 pressure。

测试：

- exact GEP load CSE。
- 动态 GEP 不误优化。
- call clobber 与 non-clobber 边界。

## 优先级 3：循环优化

现状：

- LICM 已 hoist 纯算术和 GEP。
- 没有完整归纳变量分析、循环强度削弱、地址递推。

建议实现：

- 识别 canonical loop：
  - header Phi 作为 induction variable。
  - backedge `iv + const`。
  - header compare bound。
- 建立 `SCEV-like` 线性表达式：
  - `base + iv * stride + const`
  - `iv * const`
  - `iv + const`
- 对循环内 GEP 地址：
  - 把 `base + iv * scale` 改为 preheader 初始化指针。
  - loop latch 中递增指针。
- 对 `mul iv, const`：
  - 变为递推累加。
- 对循环不变除法/取模：
  - 尽量 hoist magic 参数或 hoist 不变表达式。

风险：

- SysY int 溢出语义应按 32 位有符号行为保持。
- 指针递推必须保证循环步长和退出条件一致。
- 不要对可能越界但原程序未执行的地址提前求值。

测试：

- 单层 for/while 数组求和。
- 负步长循环。
- 非 1 步长循环。
- 循环体内有 break/continue 时先保守不做。

## 优先级 4：跨块寄存器分配

现状：

- 后端仍有大量值落栈。
- 有局部 suppressed result，但不是完整寄存器分配。

建议实现：

- 先做基本块内 linear scan。
- 再做 loop-aware live interval：
  - CFG 顺序编号。
  - live-in/live-out。
  - loop depth 加权。
- 寄存器优先级：
  - 循环内 hot value。
  - 地址基址。
  - induction variable。
  - 函数参数中频繁使用者。
- Phi coalescing：
  - 同一物理寄存器承接 Phi target 和主要前驱 source。
  - 无法 coalesce 再插入 edge copy。

风险：

- AArch64 ABI：
  - `x0-x7/w0-w7` 参数/返回，caller-saved。
  - `x19-x28` callee-saved，使用时必须保存恢复。
  - `x29` frame pointer，`x30` link register。
- 不能破坏现有结构化专项 emit。

验证：

- `tests/test_register_allocation.sh`
- 所有函数调用、递归、数组参数、浮点参数测试。

## 优先级 5：通用 NEON 循环向量化

现状：

- 部分结构化专项 emit 已手写 NEON。
- 普通 IR 后端没有通用 vectorizer。
- 当前汇编和测试链接均使用 `armv8-a`，可以优先使用 NEON / Advanced SIMD。
- 不要默认发射 SVE，除非官方评测环境明确支持 `armv8-a+sve`。

建议实现范围：

- 先支持最窄模式：
  - 单循环。
  - 连续 int 数组。
  - 无 loop-carried dependency。
  - 操作是 add/sub/mul/and/or/xor/min/max 或简单 reduction。
- 生成 scalar remainder。
- 支持 4x i32 或 8x i16 视数据宽度而定。

可识别模式：

- `for i: c[i] = a[i] + b[i]`
- `sum += a[i]`
- `for i: a[i] = const`
- `for i: a[i] = b[i]`

风险：

- 内存 alias。如果 `a`、`b`、`c` 可能重叠，必须保守或加入运行时别名检查。
- 对 SysY 数组参数通常缺少 restrict 信息，不要默认不别名。
- 对全局不同数组可较安全。

### NEON 实现建议

第一阶段只做最保守的向量化：

- 只处理全局数组或确定不 alias 的局部数组。
- 循环计数可静态确定，或能生成清晰 remainder。
- 步长为 1。
- 数组元素连续，GEP 形如 `base + i` 或 `base + i + const`。
- 循环体无函数调用、无 early exit、无复杂控制流。

推荐先实现这些模式：

- `a[i] = b[i]`
- `a[i] = 0`
- `a[i] = b[i] + c[i]`
- `a[i] = b[i] + const`
- `sum += a[i]`
- `max/min` reduction

可用指令方向：

- 4x i32：`ldr qN` / `str qN` / `add vN.4s` / `sub vN.4s` / `mul vN.4s`
- 4x f32：`fadd vN.4s` / `fsub` / `fmul`
- widen multiply/accumulate：`smlal` / `smlal2`
- reduction 可先用 vector accumulator，循环后横向归约。

验证方法：

- 为每个向量化模式建立 scalar 对照测试。
- 对长度不是 4 的倍数必须测 remainder。
- 对小长度、大长度、0 长度都测。
- 对可能 alias 的参数数组必须保持 scalar，除非生成运行时 alias check。

### SVE 判断

SVE 理论上对大循环更强：

- 向量长度可扩展。
- predication 可以减少 remainder 分支。
- 宽度可能超过 NEON 128 bit。

但当前不应默认采用：

- SVE 是 ARM 可选扩展，不是普通 `armv8-a` 基线。
- 当前脚本使用 `-march=armv8-a`，SVE 指令可能无法汇编。
- 评测机/qemu 不一定开启 SVE。

只有满足以下条件才进入提交版本：

- 官方评测文档明确允许 SVE。
- 编译命令可改为或已包含 `-march=armv8-a+sve`。
- 本地 `aarch64-linux-gnu-gcc` 和 `qemu-aarch64` 可稳定运行 SVE 程序。
- 后端可以在无 SVE 时回退到 scalar/NEON。

否则，SVE 只作为研究方向记录，不作为主线优化。

## 优先级 6：短路布尔优化的正确做法

不要恢复已删除的简单 CFG pass。

如果继续做，建议：

- 先优化前端/IR 构建，直接生成短路 CFG，不生成 `logic.tmp alloca`。
- 或在 mem2reg 之前处理，并确保后续 CFG 简化可以合并块。
- 必须保证最终汇编指令数下降。

验证重点：

- `tests/backend/short_circuit.sy`
- 公开功能 `38_op_priority4.sy`
- huffman 性能用例。

之前失败数据：

- `cset` 从 `12` 降到 `0`。
- 但 instructions 从 `14196` 增到 `14880`。
- huffman 单次时间明显变差。

结论：

- 只有同时减少 CFG/栈槽指令时才值得保留。

## 对比工业编译器的方法

可以用 GCC/Clang 编译公开用例，对比热点循环汇编，但不能复制代码。

建议：

```bash
aarch64-linux-gnu-gcc -O2 -S case.c -o /tmp/gcc.s
./compiler case.sy -S -o /tmp/ours.s
```

比较：

- 内层循环指令数。
- load/store 是否重复。
- induction variable 是否用递推。
- 常数除法是否 magic lowering。
- 是否使用 `cbz/cbnz`、`csel`、`madd/msub`、post-index addressing。
- 是否使用 NEON。

注意：

- SysY 不是 C，直接转 C 对比只用于学习思路。
- 不要把 GCC 源码或他队源码照搬。
- 借鉴优化思想，自己按本项目 IR/后端实现。

## 提交前检查清单

每次提交前必须做：

```bash
git status --short
make -B -j
bash tests/run_all.sh
bash tests/test_optimization_compliance.sh
```

推荐再做：

```bash
python3 scripts/run_arm_performance.py \
  --cases compiler2026/2026初赛ARM赛道性能用例/performance \
  --output /tmp/perf_submit.tsv \
  --repeat 1 \
  --timeout 60 \
  --compile-timeout 60
```

若改动涉及性能关键路径，使用 repeat 3：

```bash
python3 scripts/run_arm_performance.py \
  --cases compiler2026/2026初赛ARM赛道性能用例/performance \
  --output /tmp/perf_submit_r3.tsv \
  --repeat 3 \
  --timeout 60 \
  --compile-timeout 60
```

提交前不要留下：

- `__pycache__/`
- 临时关闭专项优化的 patch。
- 调试输出。
- 环境变量/路径探测。
- 用例名、程序名、固定输出相关逻辑。
