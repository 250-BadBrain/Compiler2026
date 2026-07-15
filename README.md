# Compiler2026

SysY2026 到 RISC-V 64GC 汇编的 C++17 编译器项目。

## 团队

OvO

[250-Badbrain](https://github.com/250-BadBrain)

[pointhub-a](https://github.com/pointhub-a)

## 构建

```bash
make
```

生成的编译器可执行文件为 `compiler`。

## 使用

```bash
./compiler input.sysy -S -o output.s
./compiler input.sysy -S -o output.s -O1
```

## 测试

```bash
./tests/run_smoke.sh
./tests/run_all.sh
./tests/run_public_functional.sh compiler2026/2026初赛RISCV赛道功能用例
./tests/run_public_functional.sh compiler2026/2026初赛RISCV赛道性能用例
```
