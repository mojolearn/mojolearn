# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What each arm of tools/algos_lane_check.py runs, without running anything.

Pure Python: no binding, no GPU. The `par-*` device axis these tests also
covered (tools/test_algos_lane_check_par.py) was removed when multi-GPU
verification was retired (owner, 2026-10-31).

    python tools/test_algos_lane_check_arms.py      (or pytest)
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import algos_lane_check as alc  # noqa: E402


class RunSidesCommand(unittest.TestCase):
    """Both columns come from ONE harness process (`--side both`, 2026-11-24):
    the CPU column through device="cpu", never a MOJOLEARN_VENDOR=cpu child."""

    def setUp(self):
        self.calls = []
        self.real_run = alc.subprocess.run
        self.rc = 0
        self.write = True

        def fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None, timeout=None, **kw):
            self.calls.append((cmd, env))
            if self.write:
                for flag in ("--json", "--cpu-json"):
                    Path(cmd[cmd.index(flag) + 1]).write_text("{}")
            return subprocess.CompletedProcess(cmd, self.rc)

        alc.subprocess.run = fake_run
        self.d = Path(tempfile.mkdtemp())

    def tearDown(self):
        alc.subprocess.run = self.real_run

    def _run(self, **kw):
        return alc.run_sides("gmm", "cuda", "", self.d / "g.json", self.d / "c.json", self.d / "log", **kw)

    def test_one_process_records_both_sides(self):
        os.environ["MOJOLEARN_PAR_DEVICES"] = "0,1"          # an ambient group never chooses a column
        os.environ["MOJOLEARN_VENDOR"] = "cpu"               # an ambient selector never chooses a side
        try:
            self._run()
        finally:
            os.environ.pop("MOJOLEARN_PAR_DEVICES")
            os.environ.pop("MOJOLEARN_VENDOR")
        self.assertEqual(len(self.calls), 1)
        cmd, env = self.calls[0]
        self.assertIn(str(alc.HARNESS), cmd)
        self.assertEqual(cmd[cmd.index("--side") + 1], "both")
        self.assertEqual(cmd[cmd.index("--require-backend") + 1], "cuda")
        self.assertEqual(cmd[cmd.index("--json") + 1], str(self.d / "g.json"))
        self.assertEqual(cmd[cmd.index("--cpu-json") + 1], str(self.d / "c.json"))
        self.assertNotIn("--require-cpu", cmd)
        self.assertNotIn("MOJOLEARN_VENDOR", env)
        self.assertNotIn("MOJOLEARN_PAR_DEVICES", env)

    def test_a_refusal_on_either_side_fails(self):
        self.rc = 1
        with self.assertRaises(alc.Fail):
            self._run()

    def test_a_missing_column_fails(self):
        self.write = False
        with self.assertRaises(alc.Fail):
            self._run()

    def test_a_moved_part_in_the_sabotaged_stage_is_left_to_the_diff(self):
        self.rc = 1
        self.assertEqual(self._run(moved_ok=True), 1)


if __name__ == "__main__":
    unittest.main()
