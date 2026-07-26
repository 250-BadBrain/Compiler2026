#!/usr/bin/env python3
"""Validate and compare three-run compiler performance result sets."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, Mapping


METRICS = (
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
REQUIRED_COLUMNS = ("case", "run", "status", "seconds", *METRICS)


class ResultsError(ValueError):
    """Raised when a result file is incomplete or cannot be compared safely."""


@dataclass(frozen=True)
class CaseResult:
    seconds: float
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class Comparison:
    before_seconds: float
    after_seconds: float
    gain_percent: float
    retain: bool
    exceeds_ten_percent: bool
    improved_cases: int
    total_cases: int
    all_under_one_second: bool
    before_metrics: Mapping[str, float]
    after_metrics: Mapping[str, float]
    metric_changes: Mapping[str, float]


def _parse_nonnegative_float(row: Mapping[str, str], column: str, context: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, ValueError) as error:
        raise ResultsError(f"{context}: invalid {column}") from error
    if not math.isfinite(value) or value < 0:
        raise ResultsError(f"{context}: invalid {column}")
    return value


def load_results(path: Path | str, expected_runs: int = 3) -> Dict[str, CaseResult]:
    if expected_runs <= 0:
        raise ResultsError("expected_runs must be positive")
    source = Path(path)
    try:
        handle = source.open(newline="", encoding="utf-8")
    except OSError as error:
        raise ResultsError(f"cannot read results: {source}") from error

    grouped: Dict[str, Dict[int, Mapping[str, str]]] = {}
    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or ())]
        if missing:
            raise ResultsError(f"{source}: missing columns: {', '.join(missing)}")
        for line, row in enumerate(reader, start=2):
            case = row["case"].strip()
            if not case:
                raise ResultsError(f"{source}:{line}: empty case")
            try:
                run = int(row["run"])
            except ValueError as error:
                raise ResultsError(f"{source}:{line}: invalid run") from error
            if run in grouped.setdefault(case, {}):
                raise ResultsError(f"{case}: duplicate run {run}")
            grouped[case][run] = row

    if not grouped:
        raise ResultsError(f"{source}: no result rows")

    expected = set(range(1, expected_runs + 1))
    aggregated: Dict[str, CaseResult] = {}
    for case, runs in sorted(grouped.items()):
        if set(runs) != expected:
            raise ResultsError(f"{case}: expected {expected_runs} runs, got {sorted(runs)}")
        rows = [runs[index] for index in sorted(runs)]
        for index, row in zip(sorted(runs), rows):
            status = row["status"].strip()
            if status != "AC":
                raise ResultsError(f"{case}: run {index} status {status or '<empty>'}")
        seconds = fmean(_parse_nonnegative_float(row, "seconds", case) for row in rows)
        metrics = {
            metric: fmean(_parse_nonnegative_float(row, metric, case) for row in rows)
            for metric in METRICS
        }
        aggregated[case] = CaseResult(seconds=seconds, metrics=metrics)
    return aggregated


def compare_results(before: Mapping[str, CaseResult], after: Mapping[str, CaseResult]) -> Comparison:
    if set(before) != set(after):
        only_before = sorted(set(before) - set(after))
        only_after = sorted(set(after) - set(before))
        raise ResultsError(f"case sets differ: only before={only_before}, only after={only_after}")
    if not before:
        raise ResultsError("cannot compare empty result sets")

    before_seconds = fmean(result.seconds for result in before.values())
    after_seconds = fmean(after[name].seconds for name in before)
    gain_percent = 100.0 * (before_seconds - after_seconds) / before_seconds if before_seconds else 0.0
    before_metrics = {
        metric: fmean(result.metrics[metric] for result in before.values()) for metric in METRICS
    }
    after_metrics = {
        metric: fmean(after[name].metrics[metric] for name in before) for metric in METRICS
    }
    return Comparison(
        before_seconds=before_seconds,
        after_seconds=after_seconds,
        gain_percent=gain_percent,
        retain=after_seconds < before_seconds,
        exceeds_ten_percent=gain_percent > 10.0,
        improved_cases=sum(after[name].seconds < before[name].seconds for name in before),
        total_cases=len(before),
        all_under_one_second=all(result.seconds < 1.0 for result in after.values()),
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        metric_changes={metric: after_metrics[metric] - before_metrics[metric] for metric in METRICS},
    )


def _case_rows(
    before: Mapping[str, CaseResult], after: Mapping[str, CaseResult]
) -> Iterable[tuple[str, float, float, float]]:
    for name in sorted(before):
        old = before[name].seconds
        new = after[name].seconds
        gain = 100.0 * (old - new) / old if old else 0.0
        yield name, old, new, gain


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--runs", type=int, default=3, help="required runs per case (default: 3)")
    args = parser.parse_args()

    try:
        before = load_results(args.before, args.runs)
        after = load_results(args.after, args.runs)
        comparison = compare_results(before, after)
    except ResultsError as error:
        parser.error(str(error))

    print("case\tbefore_seconds\tafter_seconds\tgain_percent")
    for name, old, new, gain in _case_rows(before, after):
        print(f"{name}\t{old:.9f}\t{new:.9f}\t{gain:.3f}")
    print(
        "SUMMARY"
        f"\t{comparison.before_seconds:.9f}"
        f"\t{comparison.after_seconds:.9f}"
        f"\t{comparison.gain_percent:.3f}"
        f"\tretain={'yes' if comparison.retain else 'no'}"
        f"\timproved={comparison.improved_cases}/{comparison.total_cases}"
        f"\tall_under_1s={'yes' if comparison.all_under_one_second else 'no'}"
    )
    print("metric\tbefore_mean\tafter_mean\tchange")
    for metric in METRICS:
        print(
            f"{metric}\t{comparison.before_metrics[metric]:.3f}"
            f"\t{comparison.after_metrics[metric]:.3f}"
            f"\t{comparison.metric_changes[metric]:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
