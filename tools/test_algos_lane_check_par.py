# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The `par-*` device axis of tools/algos_lane_check.py and the per-cell
witness of tools/par_witness_arm.py (lane/par-harness, 2026-09-28).

Pure Python: no binding, no GPU. The two-GPU behavior itself is shown by the
lane check on a two-GPU box; this pins the verdict logic that reads it, and
the 1-GPU NOT APPLICABLE path every Mac takes.

    python tools/test_algos_lane_check_par.py      (or pytest)
"""
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import algos_lane_check as alc  # noqa: E402
import par_witness_arm as pwa  # noqa: E402


def _witness(cells):
    return dict(format="mojolearn.par-witness-arm.v1", vendor="cuda", devices=[0, 1], cells=cells)


def _cell(refusal=None, pools=1, sessions=0, one_device=False):
    return dict(refusal=refusal, one_device_by_design=one_device,
                witness=dict(pools=pools, native_sessions=sessions, workers=pools))


class FakeHarness:
    FIXTURES = ["base", "ties"]


class DeviceCount(unittest.TestCase):
    def setUp(self):
        self.saved = dict(alc._DEVICE_COUNT)

    def tearDown(self):
        alc._DEVICE_COUNT.clear()
        alc._DEVICE_COUNT.update(self.saved)

    def test_metal_is_one_device_by_structure(self):
        self.assertEqual(alc.device_count("metal"), 1)
        self.assertFalse(alc.par_axis("par-gmm", "metal"))

    def test_axis_needs_two_devices_and_a_par_lane(self):
        alc._DEVICE_COUNT["cuda"] = 1
        self.assertFalse(alc.par_axis("par-gmm", "cuda"))
        alc._DEVICE_COUNT["cuda"] = 2
        self.assertTrue(alc.par_axis("par-gmm", "cuda"))
        self.assertFalse(alc.par_axis("gmm", "cuda"))

    def test_axis_drops_host_bindings(self):
        alc._DEVICE_COUNT["cuda"] = 2
        b = ["_mojolearn", "_mojolearn_core_host", "_mojolearn_mixture"]
        self.assertEqual(alc.arms_bindings("par-gmm", b, "cuda"), ["_mojolearn", "_mojolearn_mixture"])
        self.assertEqual(alc.arms_bindings("gmm", b, "cuda"), b)
        alc._DEVICE_COUNT["cuda"] = 1
        self.assertEqual(alc.arms_bindings("par-gmm", b, "cuda"), b)


class WitnessVerdict(unittest.TestCase):
    want = ["base", "ties"]

    def v(self, verdict, witness):
        return alc.par_witness_verdict("par-gmm", verdict, "compared train 2", witness, self.want)

    def test_agree_needs_a_witness_for_every_cell(self):
        good = _witness({"par-gmm/base": _cell(), "par-gmm/ties": _cell()})
        self.assertEqual(self.v("AGREE", good)[0], "AGREE")
        self.assertEqual(self.v("AGREE", None)[0], alc.WITNESS_REFUSED)
        self.assertEqual(self.v("AGREE", _witness({"par-gmm/base": _cell()}))[0], alc.WITNESS_REFUSED)

    def test_a_refused_witness_fails_an_agreement(self):
        w = _witness({"par-gmm/base": _cell(), "par-gmm/ties": _cell(refusal="started NO device pool")})
        verdict, detail = self.v("AGREE", w)
        self.assertEqual(verdict, alc.WITNESS_REFUSED)
        self.assertIn("NO device pool", detail)
        self.assertNotIn(verdict, alc.PASSING)

    def test_disagree_stays_disagree(self):
        self.assertEqual(self.v("DISAGREE", None)[0], "DISAGREE")

    def test_nothing_compared_is_not_upgraded(self):
        good = _witness({"par-gmm/base": _cell(), "par-gmm/ties": _cell()})
        self.assertEqual(self.v("NOTHING COMPARED", good)[0], "NOTHING COMPARED")

    def test_one_device_by_design_is_not_applicable(self):
        w = _witness({"par-gmm/base": _cell(pools=0, one_device=True),
                      "par-gmm/ties": _cell(pools=0, one_device=True)})
        self.assertEqual(self.v("AGREE", w)[0], alc.NOT_APPLICABLE)
        self.assertEqual(self.v("DISAGREE", w)[0], "DISAGREE")


class OneDeviceBox(unittest.TestCase):
    """A Mac (metal): the par lane's CPU refusal reads NOT APPLICABLE, which
    passes the clean check and is not AGREE."""

    def columns(self, cpu_error):
        d = Path(tempfile.mkdtemp())
        gpu = dict(vendor="metal", cells={f"par-gmm/{f}": dict(verdict="STABLE", hashes=["ab"]) for f in
                                          FakeHarness.FIXTURES})
        cpu = dict(vendor="cpu", cells={f"par-gmm/{f}": dict(verdict="REFUSED", hashes=[], error=cpu_error)
                                        for f in FakeHarness.FIXTURES})
        (d / "g.json").write_text(json.dumps(gpu))
        (d / "c.json").write_text(json.dumps(cpu))
        return d / "g.json", d / "c.json", d / "log"

    def test_by_design_refusal_is_not_applicable(self):
        g, c, log = self.columns("NotImplementedError: " + alc.CPU_BY_DESIGN_REFUSAL + " gmm_fit yet")
        verdict, detail = alc.compare(FakeHarness, "par-gmm", g, c, "", log, "metal")
        self.assertEqual(verdict, alc.NOT_APPLICABLE)
        self.assertIn("needs 2 devices", detail)
        self.assertIn(verdict, alc.PASSING)
        self.assertNotEqual(verdict, "AGREE")

    def test_another_refusal_is_not_by_design(self):
        g, c, _ = self.columns("RuntimeError: something else")
        self.assertFalse(alc.known_cpu_refusal(FakeHarness, "par-gmm", g, c, "")[0])


class RunArmCommands(unittest.TestCase):
    """What each slot runs, on each box, without running anything."""

    def setUp(self):
        self.saved = dict(alc._DEVICE_COUNT)
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
        alc._DEVICE_COUNT.clear()
        alc._DEVICE_COUNT.update(self.saved)

    def test_two_gpu_box_runs_two_device_and_one_device_gpu_arms(self):
        alc._DEVICE_COUNT["cuda"] = 2
        alc.run_arm("gpu", "par-gmm", "cuda", "", self.d / "g.json", self.d / "log")
        alc.run_arm("cpu", "par-gmm", "cuda", "", self.d / "c.json", self.d / "log")
        (lane_cmd, lane_env), (ref_cmd, ref_env) = self.calls
        self.assertIn(str(alc.WITNESS_ARM), lane_cmd)
        self.assertEqual(lane_cmd[lane_cmd.index("--witness-json") + 1], str(self.d / "g.json.witness.json"))
        self.assertEqual(lane_env["MOJOLEARN_PAR_DEVICES"], "0,1")
        self.assertEqual(ref_env["MOJOLEARN_PAR_DEVICES"], "0")
        for cmd, env in self.calls:
            self.assertNotIn("--require-cpu", cmd)
            self.assertEqual(cmd[cmd.index("--require-backend") + 1], "cuda")
            self.assertNotIn("MOJOLEARN_VENDOR", env)

    def test_two_device_arm_exit_one_is_left_to_the_diff(self):
        alc._DEVICE_COUNT["cuda"] = 2
        self.rc = 1
        self.assertEqual(alc.run_arm("gpu", "par-gmm", "cuda", "", self.d / "g.json", self.d / "log"), 1)
        with self.assertRaises(alc.Fail):   # the one-device reference arm failing is a failure
            alc.run_arm("cpu", "par-gmm", "cuda", "", self.d / "c.json", self.d / "log")

    def test_one_gpu_box_keeps_the_cpu_arm(self):
        alc._DEVICE_COUNT["cuda"] = 1
        os.environ["MOJOLEARN_PAR_DEVICES"] = "0,1"          # an ambient group never chooses a column
        try:
            self.rc = 1
            self.assertEqual(alc.run_arm("cpu", "par-gmm", "cuda", "", self.d / "c.json", self.d / "log"), 1)
        finally:
            os.environ.pop("MOJOLEARN_PAR_DEVICES")
        cmd, env = self.calls[0]
        self.assertIn("--require-cpu", cmd)
        self.assertEqual(env["MOJOLEARN_VENDOR"], "cpu")
        self.assertNotIn("MOJOLEARN_PAR_DEVICES", env)
        self.assertNotIn(str(alc.WITNESS_ARM), cmd)

    def test_a_plain_lane_cpu_refusal_still_fails(self):
        self.rc = 1
        with self.assertRaises(alc.Fail):
            alc.run_arm("cpu", "gmm", "cuda", "", self.d / "c.json", self.d / "log")


class CellWitnessBoundaries(unittest.TestCase):
    """One witness per cell, closed at the next cell and at exit."""

    def setUp(self):
        opened = self.opened = []

        class FakeWitness:
            def __init__(self, vendor, devices, one_device_driver=None):
                self.one_device_driver = one_device_driver
                self.n = len(opened)
                opened.append(self)

            def watching(self):
                import contextlib
                return contextlib.nullcontext(self)

            def refusal(self):
                return None if self.n == 0 else "started NO device pool"

            def summary(self):
                return dict(pools=1 - self.n)

        fake = types.ModuleType("mojolearn._verify_par")
        fake.PoolWitness = FakeWitness
        fake.ONE_DEVICE_BY_DESIGN = {"par-byte-lm-offload": ("OffloadedByteLanguageModelTrainer", "n/a")}
        backend = types.ModuleType("mojolearn._backend")
        backend.vendor = lambda: "cuda"
        pkg = types.ModuleType("mojolearn")
        pkg._backend = backend
        self.saved = {k: sys.modules.get(k) for k in ("mojolearn", "mojolearn._verify_par", "mojolearn._backend")}
        sys.modules.update({"mojolearn": pkg, "mojolearn._verify_par": fake, "mojolearn._backend": backend})

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_each_cell_gets_its_own_record(self):
        cells = pwa.CellWitnesses((0, 1))
        cells.open("par-gmm/base")
        cells.open("par-byte-lm-offload/base")
        cells.close()
        cells.close()                                        # idempotent
        self.assertEqual(cells.vendor, "cuda")
        self.assertIsNone(cells.cells["par-gmm/base"]["refusal"])
        self.assertEqual(cells.cells["par-byte-lm-offload/base"]["refusal"], "started NO device pool")
        self.assertTrue(cells.cells["par-byte-lm-offload/base"]["one_device_by_design"])
        self.assertFalse(cells.cells["par-gmm/base"]["one_device_by_design"])

    def test_the_wrapper_refuses_one_device(self):
        for raw in ("", "0", "0,0", "1,-1", "a,b"):
            with self.assertRaises(SystemExit):
                pwa.parse_devices(raw)
        self.assertEqual(pwa.parse_devices("0,1"), (0, 1))


if __name__ == "__main__":
    unittest.main()
