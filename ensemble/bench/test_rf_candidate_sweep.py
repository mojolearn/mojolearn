# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""CPU-only tests of the RF sweep's measurement/identity rejection gates."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).with_name("rf_candidate_sweep.py")


class CandidateSweepGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.binary("baseline", 1, "1234", 10)
        self.binary("items4", 4, "1234", 8)

    def binary(self, variant, items, model, timing):
        path = self.out / ("fast_" + variant)
        path.write_text(
            f"#!{sys.executable}\n"
            f"print('CONFIG 0 sorted False items {items} copies 1')\n"
            "print('CANARY warmup 20 nodes 3')\n"
            "print('CANARY pre 10 nodes 3')\n"
            "for i in range(5):\n"
            f" print('ARM rf-clf@128x3', 1000 if i == 0 else {timing})\n"
            f" print('MODEL rf-clf@128x3 {model}')\n"
            "print('CANARY post 10 nodes 3')\n"
        )
        path.chmod(0o755)

    def run_sweep(self):
        return subprocess.run(
            [sys.executable, str(RUNNER), "run", "--out", str(self.out),
             "--modes", "fast", "--variants", "baseline", "items4",
             "--tasks", "clf", "--rows", "128", "--cols", "3", "--rounds", "1"],
            capture_output=True, text=True,
        )

    def test_matching_models_and_discarded_first_timing(self):
        result = self.run_sweep()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["identity_failures"], [])
        self.assertEqual(summary["records"][1]["samples_ms"], [8.0] * 4)
        self.assertEqual(summary["records"][1]["baseline_over_candidate"], 1.25)

    def test_changed_model_rejected_even_when_faster(self):
        self.binary("items4", 4, "5678", 1)
        result = self.run_sweep()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("full-model identity mismatch", result.stderr)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["identity_failures"], ["fast_items4_clf"])

    def test_wrong_binary_rejected_and_old_summary_removed(self):
        self.assertEqual(self.run_sweep().returncode, 0)
        self.binary("items4", 1, "1234", 1)
        result = self.run_sweep()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("compiled config", result.stderr)
        self.assertFalse((self.out / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
