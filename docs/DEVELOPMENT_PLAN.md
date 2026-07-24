# 开发规划

## 阶段 1：ARM 正确性基线

目标文件：
- `src/backend/arm/emit.*`
- `src/main.cpp`
- `Makefile`
- `CMakeLists.txt`
- `tests/run_arm.sh`
- `tests/run_public_functional.sh`
- `tests/run_public_smoke.sh`
- `tests/run_runtime.sh`

实现步骤：
1. 后端命名空间切换为 `sysyc::arm`。
2. 构建入口只编译 `src/backend/arm/emit.cpp`。
3. 输出 ARMv7 A32 GNU 汇编：
   - `.syntax unified`
   - `.arch armv7-a`
   - `.fpu vfpv3`
   - `.global`
   - `.type`
4. 按 AAPCS 生成函数：
   - `r0-r3` 传递整数和指针参数
   - `s0-s15` 传递浮点参数
   - `r0` 返回整数和指针
   - `s0` 返回浮点
   - `push {fp, lr}`
   - `mov fp, sp`
   - `mov sp, fp`
   - `pop {fp, pc}`
5. 所有 IR SSA 值先落栈。
6. `alloca` 分配函数栈帧对象。
7. `load/store/gep` 支持局部对象、全局变量、数组元素。
8. `add/sub/mul/div/mod/icmp` 支持整数。
9. `fadd/fsub/fmul/fdiv/fcmp/cast` 支持 VFP 单精度。
10. `call/br/condbr/ret` 支持函数调用和控制流。
11. Phi 在前驱边插入保守拷贝。
12. 测试脚本默认使用 ARM 用例目录。

验收：
```bash
make -B
./tests/run_smoke.sh
./tests/run_lexer.sh
./tests/run_parser.sh
./tests/run_sema.sh
./tests/run_ir.sh
./tests/run_backend.sh
./tests/run_functional.sh compiler2026/2026初赛ARM赛道功能用例
```

安装 ARM 工具链后追加验收：
```bash
./tests/run_arm.sh
./tests/run_public_functional.sh compiler2026/2026初赛ARM赛道功能用例
./tests/run_public_functional.sh compiler2026/2026初赛ARM赛道性能用例
```

## 阶段 2：ARM 全量功能修复

实现步骤：
1. 安装并确认：
   - `arm-linux-gnueabihf-gcc`
   - `qemu-arm`
2. 逐个运行 ARM 功能用例。
3. 修复真实链接运行暴露的问题：
   - 大栈帧偏移超限
   - 栈参数读取
   - 混合 int/float 参数
   - 浮点返回值
   - 外部运行时函数调用
   - Phi 拷贝环
4. 每次修复后运行完整 ARM 功能集。

验收：
```bash
./tests/run_public_functional.sh compiler2026/2026初赛ARM赛道功能用例
```

## 阶段 3：ARM 后端性能基线

实现步骤：
1. 统计公开性能用例汇编：
   - 总指令数
   - `ldr/str/vldr/vstr`
   - `mov/vmov`
   - `bl`
2. 与 `arm-linux-gnueabihf-gcc -O2` 输出对比。
3. 建立代表用例集合：
   - `03_sort`
   - `conv2d`
   - `crypto`
   - `fft`
   - `huffman`
   - `matmul`
4. 建立每轮优化后的统计表。

验收：
```bash
./tests/run_public_functional.sh compiler2026/2026初赛ARM赛道性能用例
```

## 阶段 4：ARM 寄存器分配

实现步骤：
1. 为整数值实现跨基本块线性扫描寄存器分配。
2. 可用寄存器：
   - caller-saved：`r0-r3`, `r12`
   - callee-saved：`r4-r10`
3. 为浮点值实现 VFP 线性扫描：
   - caller-saved：`s0-s15`
   - callee-saved：`s16-s31`
4. 根据函数调用情况决定保存 callee-saved 寄存器。
5. Phi coalescing 合并同源同目标值。
6. 消除冗余 `mov/vmov`。
7. 降低栈槽 `ldr/str/vldr/vstr`。

验收：
1. ARM 功能用例全过。
2. ARM 性能用例全过。
3. 代表性能用例访存数显著下降。

## 阶段 5：ARM 指令选择优化

实现步骤：
1. 立即数指令选择：
   - `mov`
   - `mvn`
   - `add/sub #imm`
   - literal pool fallback
2. 地址模式折叠：
   - `[base, #imm]`
   - `[base, index, lsl #2]`
   - `ldr/str` 直接使用数组元素地址
3. 乘加模式：
   - `mla`
   - `mls`
4. 除法和取模：
   - 常量 2 的幂强度削弱
   - 常量除法 magic number
   - 必要时调用 `__aeabi_idiv`
5. 条件分支融合：
   - `cmp` + `b<cond>`
   - 删除物化布尔临时值
6. 函数调用参数直传：
   - 避免先落栈再装入 `r0-r3/s0-s15`

验收：
1. 汇编总指令数下降。
2. `ldr/str` 数量下降。
3. `mov/vmov` 数量下降。

## 阶段 6：IR 优化继续迁移

实现步骤：
1. 保留并验证现有 IR 优化：
   - mem2reg
   - Phi
   - GVN
   - LICM
   - DCE
   - load/store forwarding
2. 针对 ARM 后端重新评估寄存器压力。
3. 高压力循环中限制 GVN/LICM 过度提升。
4. 对数组地址表达式做循环内提升。
5. 对全局标量做安全寄存器缓存。

验收：
1. ARM 全量功能和性能用例通过。
2. 代表性能用例运行时间下降。

## 阶段 7：提交前检查

执行：
```bash
make -B
./tests/run_smoke.sh
./tests/run_backend.sh
./tests/run_functional.sh compiler2026/2026初赛ARM赛道功能用例
./tests/run_public_functional.sh compiler2026/2026初赛ARM赛道功能用例
./tests/run_public_functional.sh compiler2026/2026初赛ARM赛道性能用例
git status --short
```
