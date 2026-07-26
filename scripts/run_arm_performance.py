#!/usr/bin/env python3
"""Compile, execute, validate, and measure every supplied SysY performance case."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, Sequence


FIELDS = (
    "case",
    "run",
    "status",
    "seconds",
    "asm_lines",
    "instructions",
    "ldr",
    "str",
    "ldp",
    "stp",
    "mov",
    "fmov",
    "sdiv",
    "udiv",
)
COUNTED_MNEMONICS = ("ldr", "str", "ldp", "stp", "mov", "fmov", "sdiv", "udiv")
INSTRUCTION_RE = re.compile(r"^\s+([a-z][a-z0-9.]*)\b", re.IGNORECASE)


class HarnessError(RuntimeError):
    """Raised when a complete, trustworthy measurement cannot be made."""


def compose_actual_output(stdout: bytes, returncode: int, expected: bytes) -> bytes:
    actual = stdout
    if actual and not actual.endswith(b"\n"):
        actual += b"\n"
    actual += str(returncode).encode("ascii") + b"\n"
    if expected and not expected.endswith(b"\n"):
        actual = actual[:-1]
    return actual


def assembly_stats(assembly: str) -> Dict[str, int]:
    lines = assembly.splitlines()
    stats = {metric: 0 for metric in COUNTED_MNEMONICS}
    instruction_count = 0
    for line in lines:
        match = INSTRUCTION_RE.match(line)
        if not match:
            continue
        mnemonic = match.group(1).lower()
        if mnemonic.startswith("."):
            continue
        instruction_count += 1
        if mnemonic in stats:
            stats[mnemonic] += 1
    return {"asm_lines": len(lines), "instructions": instruction_count, **stats}


def _resolve_tool(command: str, label: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        candidate = Path(command)
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            resolved = str(candidate.resolve())
    if resolved is None:
        raise HarnessError(f"missing {label}: {command}")
    return resolved


def _cases_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise HarnessError(f"case directory does not exist: {directory}")
    cases = sorted(directory.rglob("*.sy"))
    if not cases:
        raise HarnessError(f"case directory contains no .sy files: {directory}")
    missing = [case.with_suffix(".out") for case in cases if not case.with_suffix(".out").is_file()]
    if missing:
        raise HarnessError(f"missing expected output: {missing[0]}")
    return cases


def _run(
    command: Sequence[str], *, timeout: float, stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _failed_rows(case: str, repeat: int, status: str, stats: Dict[str, int]) -> Iterable[dict[str, object]]:
    for run in range(1, repeat + 1):
        yield {"case": case, "run": run, "status": status, "seconds": "0.000000000", **stats}


def collect(args: argparse.Namespace) -> int:
    cases = _cases_in(args.cases.resolve())
    if args.repeat <= 0:
        raise HarnessError("repeat must be positive")
    if args.timeout <= 0 or args.compile_timeout <= 0:
        raise HarnessError("timeouts must be positive")

    compiler = _resolve_tool(args.compiler, "compiler")
    cc = _resolve_tool(args.cc, "AArch64 compiler")
    runner = _resolve_tool(args.runner, "AArch64 runner")
    runtime = args.runtime.resolve()
    if not runtime.is_file():
        raise HarnessError(f"runtime source does not exist: {runtime}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    any_failure = False
    with tempfile.TemporaryDirectory(prefix="compiler2026-perf-") as temp_name, args.output.open(
        "w", newline="", encoding="utf-8"
    ) as output:
        temp = Path(temp_name)
        runtime_object = temp / "sylib.o"
        runtime_compile = _run(
            [cc, "-O2", "-static", "-march=armv8-a", "-c", str(runtime), "-o", str(runtime_object)],
            timeout=args.compile_timeout,
        )
        if runtime_compile.returncode != 0:
            message = runtime_compile.stderr.decode(errors="replace")
            raise HarnessError(f"failed to compile runtime:\n{message}")

        writer = csv.DictWriter(output, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for case_path in cases:
            relative = case_path.relative_to(args.cases.resolve())
            case = relative.with_suffix("").as_posix()
            stem = re.sub(r"[^A-Za-z0-9_.-]", "_", case)
            assembly_path = temp / f"{stem}.s"
            executable_path = temp / f"{stem}.exe"
            compile_result = _run(
                [compiler, str(case_path), "-S", "-o", str(assembly_path)], timeout=args.compile_timeout
            )
            if compile_result.returncode != 0 or not assembly_path.is_file():
                stats = {"asm_lines": 0, "instructions": 0, **{name: 0 for name in COUNTED_MNEMONICS}}
                writer.writerows(_failed_rows(case, args.repeat, "CE", stats))
                any_failure = True
                continue

            stats = assembly_stats(assembly_path.read_text(encoding="utf-8"))
            link_result = _run(
                [cc, "-static", "-march=armv8-a", str(assembly_path), str(runtime_object), "-o", str(executable_path)],
                timeout=args.compile_timeout,
            )
            if link_result.returncode != 0:
                writer.writerows(_failed_rows(case, args.repeat, "LE", stats))
                any_failure = True
                continue

            input_path = case_path.with_suffix(".in")
            stdin = input_path.read_bytes() if input_path.is_file() else None
            expected = case_path.with_suffix(".out").read_bytes()
            command = [runner]
            if args.sysroot:
                command.extend(["-L", args.sysroot])
            command.append(str(executable_path))
            for run in range(1, args.repeat + 1):
                start = time.perf_counter()
                try:
                    result = _run(command, timeout=args.timeout, stdin=stdin)
                    seconds = time.perf_counter() - start
                    actual = compose_actual_output(result.stdout, result.returncode, expected)
                    status = "AC" if actual == expected else "WA"
                except subprocess.TimeoutExpired:
                    seconds = args.timeout
                    status = "TLE"
                writer.writerow(
                    {
                        "case": case,
                        "run": run,
                        "status": status,
                        "seconds": f"{seconds:.9f}",
                        **stats,
                    }
                )
                output.flush()
                any_failure = any_failure or status != "AC"
                print(f"{case} run {run}/{args.repeat}: {status} {seconds:.6f}s", file=sys.stderr)
    return 1 if any_failure else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="directory containing .sy/.in/.out cases")
    parser.add_argument("--output", type=Path, required=True, help="raw TSV destination")
    parser.add_argument("--compiler", default="./compiler")
    parser.add_argument("--cc", default="aarch64-linux-gnu-gcc")
    parser.add_argument("--runner", default="qemu-aarch64")
    parser.add_argument("--sysroot", default="/usr/aarch64-linux-gnu")
    parser.add_argument("--runtime", type=Path, default=Path("tests/runtime/sylib.c"))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--compile-timeout", type=float, default=60.0)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        return collect(args)
    except (HarnessError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
