#!/usr/bin/env python3
"""tools/ops/measure_candidate.py stage refuses a B arm whose wheels lack the arm's define (lane ab-prep, 2026-11-23).

Plain unittest on a temporary owed file and hand-built manifests; reads no wheel, touches no box.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "ops"))
import measure_candidate as M  # noqa: E402


def manifest(defines):
    files = [dict(file=f"mojolearn_{d}-x.whl", candidate=dict(defines=defines)) for d in ("core", "nvidia")]
    return dict(files=files, defines=defines)


class ArmWheel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.owed = Path(self.tmp.name) / "owed.json"
        self.owed.write_text(json.dumps({"b_arms": {
            "one": {"define": "MOJOLEARN_A_OFF", "owed": "RUN"},
            "pair": {"define": "MOJOLEARN_A_OFF,MOJOLEARN_B=1", "owed": "RUN"},
            "none": {"owed": "RUN"}}}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_set_is_staged(self):
        M.check_arm_wheel("one", self.owed, manifest("MOJOLEARN_A_OFF"))
        M.check_arm_wheel("pair", self.owed, manifest("MOJOLEARN_A_OFF,MOJOLEARN_B=1"))

    def test_combined_set_is_staged(self):
        M.check_arm_wheel("one", self.owed, manifest("MOJOLEARN_A_OFF,MOJOLEARN_C_FAST_X"))

    def test_candidate_wheel_is_refused(self):
        with self.assertRaises(SystemExit):
            M.check_arm_wheel("one", self.owed, manifest(""))

    def test_other_arm_wheel_is_refused(self):
        with self.assertRaises(SystemExit):
            M.check_arm_wheel("pair", self.owed, manifest("MOJOLEARN_A_OFF"))
        with self.assertRaises(SystemExit):
            M.check_arm_wheel("pair", self.owed, manifest("MOJOLEARN_A_OFF,MOJOLEARN_B=2"))

    def test_arm_without_define_is_refused(self):
        with self.assertRaises(SystemExit):
            M.check_arm_wheel("none", self.owed, manifest("MOJOLEARN_A_OFF"))


if __name__ == "__main__":
    unittest.main()
