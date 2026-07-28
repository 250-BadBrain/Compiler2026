# Compiler2026 交接文档

本目录只保留队友继续开发需要的文档。

- [OPTIMIZATION_HANDOFF.md](OPTIMIZATION_HANDOFF.md)：当前优化实现、验证状态、已知风险、不能再踩的坑。
- [NEXT_OPTIMIZATION_PLAN.md](NEXT_OPTIMIZATION_PLAN.md)：后续优化优先级、每轮迭代流程、测试与对比方法。

当前赛道：ARM / AArch64。

当前原则：

- 保留已经通过正确性验证的结构化专项优化。
- 后续重点补强未知用例也能受益的通用优化。
- 不使用用例名、文件名、函数名、输出答案、程序 fingerprint、评测机探测等违规触发方式。
- 每轮优化必须跑全量正确性测试和性能回归。
