# Compiler2026

SysY2026 到 ARMv8-A/AArch64 汇编的 C++17 编译器项目。

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

ARM 链接运行测试需要：

```bash
sudo apt install gcc-aarch64-linux-gnu qemu-user
```

```bash
./tests/run_smoke.sh
./tests/run_all.sh
./tests/run_public_functional.sh compiler2026/2026初赛ARM赛道功能用例
./tests/run_public_functional.sh compiler2026/2026初赛ARM赛道性能用例
```

官方 ARM 赛道目标为 ARMv8-A 64 位，汇编/链接口径应使用：

```bash
aarch64-linux-gnu-gcc -march=armv8-a output.s tests/runtime/sylib.c -o output
qemu-aarch64 -L /usr/aarch64-linux-gnu ./output
```
