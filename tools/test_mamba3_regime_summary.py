"""Retained captures must not turn partial or inconsistent runs into evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from mamba3_regime_summary import summarize


class RegimeSummaryTests(unittest.TestCase):
    def setUp(self):
        self.lines = (Path(__file__).resolve().parents[1] /
            "bench/results/mamba_regime_2026-09-10/m4-tiny.log").read_text().splitlines()

    def run_log(self, lines):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.log"
            path.write_text("\n".join(lines) + "\n")
            return summarize(path)

    def test_retained_cold_call_stays_separate(self):
        report = self.run_log(self.lines)
        self.assertFalse(report["instrumented"])
        visit = report["visits"][0]
        self.assertEqual(visit["first_call_ms"], 385.846542)
        self.assertEqual(visit["later_calls_median_ms"], 13.75025)
        self.assertIsNone(visit["predecessor"])

    def test_truncated_and_duplicate_calls_refused(self):
        with self.assertRaises(ValueError):
            self.run_log(self.lines[:-1])
        index = next(i for i, line in enumerate(self.lines) if '"kind": "call_end"' in line)
        with self.assertRaises(ValueError):
            self.run_log(self.lines[:index] + [self.lines[index]] + self.lines[index:])

    def test_changed_output_refused(self):
        lines = self.lines.copy()
        index = next(i for i, line in enumerate(lines) if '"kind": "call_end"' in line)
        event = json.loads(lines[index])
        event["output_sha256"] = "0" * 64
        lines[index] = json.dumps(event)
        with self.assertRaises(ValueError):
            self.run_log(lines)

    def test_phases_attached_without_summing_nested_scopes(self):
        lines = []
        for line in self.lines:
            lines.append(line)
            if '"kind": "call_begin"' in line:
                lines += ["M3_PHASE block.core 2.5", "M3_PHASE core.inner 1.0"]
        report = self.run_log(lines)
        self.assertTrue(report["instrumented"])
        self.assertEqual(report["visits"][0]["phases_ms"],
                         {"block.core": [2.5, 2.5], "core.inner": [1.0, 1.0]})
        lines.remove("M3_PHASE core.inner 1.0")
        with self.assertRaises(ValueError):
            self.run_log(lines)


if __name__ == "__main__":
    unittest.main()
