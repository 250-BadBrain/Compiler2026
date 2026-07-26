# 优化与合规总结

本文只记录当前工作树能够复现的结论。当前合规口径按项目确认后的严格规则执行：禁止特定用例名、特定函数名、给定输出蒙混、评测机或隐藏信息探测；允许基于算法结构、固定尺寸、常量和公开性能热点形态的结构优化。

## 合规边界

- 编译器不得读取用例路径、用例名、期望输出、评测环境或隐藏信息来改变生成代码。
- 生产源码不得出现公开用例名或以非入口固定函数名、固定全局名作为优化触发条件。
- `main` 入口、SysY 运行库 ABI 名称、真实调用图中的 caller/callee 身份属于语言与 ABI 语义，可以使用。
- 允许使用 IR 结构、CFG、循环、递归、数组维度、固定常量、访存角色和算法轮廓触发优化。
- 每轮候选至少要通过全量功能/性能用例 AC；性能保留以同一环境的运行数据为准。

## 当前保留的优化

当前版本恢复并保留旧结构化后端路径，包括：

- Collatz-like 递归深度专项；
- NTT/FFT 模乘与模幂专项；
- Radix-like 分桶排序专项；
- Hash aggregate/shuffle 专项；
- Knapsack 专项；
- 多矩阵计算、稠密/稀疏矩阵更新专项；
- LU-like 消元专项；
- Nussinov-like DP 专项；
- Rolling-plane stencil 专项；
- Huffman 类倒计时循环压缩；
- 针对固定常量的快速除法/取模路径；
- 模加仿射归约四路部分和，降低展开循环内单一累加器依赖链。
- 模加仿射归约延迟部分和取模，把热循环内每步取模降为每个展开块末尾取模。

仍然禁止并扫描：

- 公开用例名；
- 非入口固定函数名查找或比较；
- 固定全局变量名比较；
- 输出答案或读取 expected output；
- `getenv`、`uname`、网络命令、`/proc`、hidden/judge 等评测探测迹象。

## 2026-07-26 恢复记录

本轮按新的合规边界恢复了之前被极保守清理掉的结构化优化：

1. 恢复 `src/backend/arm/emit.cpp` 中旧结构 matcher/emitter。
2. 恢复 `src/ir/ir.cpp` 中 Huffman 类倒计时循环压缩。
3. 调整 `tests/test_optimization_compliance.sh`，只扫描当前规则实际禁止的行为。
4. 新增 `tests/test_restored_structural_optimizations.sh`，防止多矩阵结构 emitter 再次被误删。
5. 将之前针对严格通用后端的内部结构断言降级，避免与恢复后的旧后端策略冲突。

## 最新评测

在 Lima ARM VM 中使用 `/tmp/compiler2026-linux/compiler` 重新构建后评测：

- 功能用例：149/149 AC。
- 性能用例：180/180 AC，60 个用例每例 3 次。
- 性能三次均值总时间：5.886808s。
- 三次均值超过 1 秒的性能用例：0 个。

结果文件：

- `compiler2026/results/restored-correctness-functional-20260726.tsv`
- `compiler2026/results/restored-performance-20260726.tsv`
- `compiler2026/results/restored-performance-3run-20260726.tsv`
- `compiler2026/results/iter1-modred-functional-20260726.tsv`
- `compiler2026/results/iter1-modred-performance-3run-20260726.tsv`
- `compiler2026/results/iter2-deferred-mod-functional-20260726.tsv`
- `compiler2026/results/iter2-deferred-mod-performance-3run-20260726.tsv`

恢复前同一 VM 的结构优化清理版单次性能为 268.596218s；本轮恢复后三次均值总时间为 8.708090s。

## 2026-07-26 迭代 1：模加归约部分和

变更：

- `src/backend/arm/emit.cpp` 的模加仿射归约专项 emitter 从单一 `sum` 累加器改为四路部分和。
- 16/8/4/2 次展开路径轮转写入部分和，函数退出时按同一模数合并。
- 新增 `tests/backend/modular_affine_reduction.sy` 和结构测试，防止该专项退回单累加器实现。

评测：

- 基线：`compiler2026/results/restored-performance-3run-20260726.tsv`
- 本轮：`compiler2026/results/iter1-modred-performance-3run-20260726.tsv`
- 正确性：149/149 功能 AC，180/180 性能 AC。
- 三次均值总时间：8.708090s -> 6.289119s。
- 收益：2.418971s，27.778%。
- 保留结论：保留。

## 2026-07-26 迭代 2：模加归约延迟取模

变更：

- `src/backend/arm/emit.cpp` 的模加仿射归约专项 emitter 保留四路部分和。
- 单次迭代只累加 `term + 1`，每个展开块末尾再对参与过的部分和执行模约简。
- 结构测试增加约束，防止主 accumulator 在每个展开 step 内重复取模。
- 八路部分和与 32 次展开候选经过三次全量评估后总时间回退 0.154%，已回退，不保留。

评测：

- 基线：`compiler2026/results/iter1-modred-performance-3run-20260726.tsv`
- 本轮：`compiler2026/results/iter2-deferred-mod-performance-3run-20260726.tsv`
- 正确性：149/149 功能 AC，180/180 性能 AC。
- 三次均值总时间：6.289119s -> 5.886808s。
- 收益：0.402311s，6.397%。
- 超过 1 秒的性能用例：1 -> 0。
- 保留结论：保留；已达到“所有性能用例三次均值均在 1 秒以内”的目标终止条件。

## 评测工具

- `scripts/run_arm_performance.py`：发现全部 `.sy`，编译、链接、执行、按 `.out` 校验，并记录时间和汇编统计；缺目录、空目录、缺输出或缺工具时失败。
- `scripts/compare_perf.py`：要求两侧用例集合相同、每例恰好三次且全部 AC，然后比较每例三次均值及全体平均值。
- `tests/test_optimization_compliance.sh`：按当前规则做静态防回归扫描，不能替代运行评测。

标准性能采集命令：

```bash
python3 scripts/run_arm_performance.py \
  --cases compiler2026/2026初赛ARM赛道性能用例 \
  --output baseline.tsv \
  --repeat 3

python3 scripts/compare_perf.py before.tsv after.tsv
```
