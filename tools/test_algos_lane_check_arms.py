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


class RunArmCommands(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.real_run = alc.subprocess.run
        self.rc = 0

        def fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None, timeout=None, **kw):
            self.calls.append((cmd, env))
            out = Path(cmd[cmd.index("--json") + 1])
            out.write_text("{}")
            return subprocess.CompletedProcess(cmd, self.rc)

        alc.subprocess.run = fake_run
        self.d = Path(tempfile.mkdtemp())

    def tearDown(self):
        alc.subprocess.run = self.real_run

    def test_gpu_and_cpu_arms_run_the_harness_on_their_backends(self):
        os.environ["MOJOLEARN_PAR_DEVICES"] = "0,1"          # an ambient group never chooses a column
        try:
            alc.run_arm("gpu", "gmm", "cuda", "", self.d / "g.json", self.d / "log")
            alc.run_arm("cpu", "gmm", "cuda", "", self.d / "c.json", self.d / "log")
        finally:
            os.environ.pop("MOJOLEARN_PAR_DEVICES")
        (gpu_cmd, gpu_env), (cpu_cmd, cpu_env) = self.calls
        self.assertIn(str(alc.HARNESS), gpu_cmd)
        self.assertEqual(gpu_cmd[gpu_cmd.index("--require-backend") + 1], "cuda")
        self.assertNotIn("--require-cpu", gpu_cmd)
        self.assertNotIn("MOJOLEARN_VENDOR", gpu_env)
        self.assertIn("--require-cpu", cpu_cmd)
        self.assertEqual(cpu_env["MOJOLEARN_VENDOR"], "cpu")
        for _, env in self.calls:
            self.assertNotIn("MOJOLEARN_PAR_DEVICES", env)

    def test_a_cpu_refusal_fails(self):
        self.rc = 1
        with self.assertRaises(alc.Fail):
            alc.run_arm("cpu", "gmm", "cuda", "", self.d / "c.json", self.d / "log")

    def test_a_moved_part_in_the_sabotaged_stage_is_left_to_the_diff(self):
        self.rc = 1
        self.assertEqual(alc.run_arm("gpu", "gmm", "cuda", "", self.d / "g.json", self.d / "log",
                                     moved_ok=True), 1)


if __name__ == "__main__":
    unittest.main()
