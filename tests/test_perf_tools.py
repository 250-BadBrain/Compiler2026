import csv
import tempfile
import unittest
from pathlib import Path


FIELDS = [
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
]


class ComparePerfTests(unittest.TestCase):
    def write_results(self, rows):
        handle = tempfile.NamedTemporaryFile("w", newline="", suffix=".tsv", delete=False)
        with handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(Path(handle.name).unlink)
        return Path(handle.name)

    @staticmethod
    def row(case, run, seconds, status="AC", instructions=100):
        return {
            "case": case,
            "run": run,
            "status": status,
            "seconds": seconds,
            "asm_lines": 120,
            "instructions": instructions,
            "ldr": 10,
            "str": 8,
            "ldp": 1,
            "stp": 1,
            "mov": 5,
            "fmov": 0,
            "sdiv": 2,
            "udiv": 0,
        }

    def test_load_results_averages_exactly_three_ac_runs(self):
        from scripts.compare_perf import load_results

        path = self.write_results(
            [
                self.row("alpha", 1, 3.0, instructions=90),
                self.row("alpha", 2, 2.0, instructions=120),
                self.row("alpha", 3, 1.0, instructions=150),
            ]
        )

        result = load_results(path)

        self.assertAlmostEqual(result["alpha"].seconds, 2.0)
        self.assertAlmostEqual(result["alpha"].metrics["instructions"], 120.0)

    def test_load_results_rejects_non_ac_run(self):
        from scripts.compare_perf import ResultsError, load_results

        path = self.write_results(
            [self.row("alpha", 1, 1.0), self.row("alpha", 2, 1.0, status="WA"), self.row("alpha", 3, 1.0)]
        )

        with self.assertRaisesRegex(ResultsError, "alpha.*WA"):
            load_results(path)

    def test_load_results_rejects_missing_run(self):
        from scripts.compare_perf import ResultsError, load_results

        path = self.write_results([self.row("alpha", 1, 1.0), self.row("alpha", 2, 1.0)])

        with self.assertRaisesRegex(ResultsError, "alpha.*3 runs"):
            load_results(path)

    def test_load_results_rejects_nonfinite_numbers(self):
        from scripts.compare_perf import ResultsError, load_results

        path = self.write_results([self.row("alpha", run, "nan") for run in range(1, 4)])

        with self.assertRaisesRegex(ResultsError, "alpha.*seconds"):
            load_results(path)

    def test_compare_results_rejects_different_case_sets(self):
        from scripts.compare_perf import ResultsError, compare_results, load_results

        before = self.write_results([self.row("alpha", run, 1.0) for run in range(1, 4)])
        after = self.write_results([self.row("beta", run, 1.0) for run in range(1, 4)])

        with self.assertRaisesRegex(ResultsError, "case sets differ"):
            compare_results(load_results(before), load_results(after))

    def test_compare_results_reports_mean_gain_and_retention(self):
        from scripts.compare_perf import compare_results, load_results

        before = self.write_results(
            [self.row(case, run, seconds) for case, seconds in (("alpha", 2.0), ("beta", 4.0)) for run in range(1, 4)]
        )
        after = self.write_results(
            [self.row(case, run, seconds) for case, seconds in (("alpha", 1.0), ("beta", 3.0)) for run in range(1, 4)]
        )

        comparison = compare_results(load_results(before), load_results(after))

        self.assertAlmostEqual(comparison.before_seconds, 3.0)
        self.assertAlmostEqual(comparison.after_seconds, 2.0)
        self.assertAlmostEqual(comparison.gain_percent, 100.0 / 3.0)
        self.assertTrue(comparison.retain)
        self.assertTrue(comparison.exceeds_ten_percent)
        self.assertAlmostEqual(comparison.before_metrics["instructions"], 100.0)
        self.assertAlmostEqual(comparison.after_metrics["instructions"], 100.0)
        self.assertAlmostEqual(comparison.metric_changes["instructions"], 0.0)


class RunArmPerformanceTests(unittest.TestCase):
    def test_compose_actual_output_matches_sysy_judging_convention(self):
        from scripts.run_arm_performance import compose_actual_output

        self.assertEqual(compose_actual_output(b"42", 0, b"42\n0\n"), b"42\n0\n")
        self.assertEqual(compose_actual_output(b"42\n", 7, b"42\n7"), b"42\n7")
        self.assertEqual(compose_actual_output(b"", 3, b"3\n"), b"3\n")

    def test_assembly_stats_counts_only_instruction_mnemonics(self):
        from scripts.run_arm_performance import assembly_stats

        assembly = """\
\t.arch armv8-a
main:
\tstp x29, x30, [sp, #-16]!
\tmov x29, sp
\tldr w0, [x1]
\tstr w0, [x2]
\tsdiv w0, w0, w3
\tfmov s0, w0
\tldp x29, x30, [sp], #16
\tret
\t.size main, .-main
"""

        stats = assembly_stats(assembly)

        self.assertEqual(stats["asm_lines"], 11)
        self.assertEqual(stats["instructions"], 8)
        self.assertEqual(stats["ldr"], 1)
        self.assertEqual(stats["str"], 1)
        self.assertEqual(stats["ldp"], 1)
        self.assertEqual(stats["stp"], 1)
        self.assertEqual(stats["mov"], 1)
        self.assertEqual(stats["fmov"], 1)
        self.assertEqual(stats["sdiv"], 1)
        self.assertEqual(stats["udiv"], 0)


if __name__ == "__main__":
    unittest.main()
