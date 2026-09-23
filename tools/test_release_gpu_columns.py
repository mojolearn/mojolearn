#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The NVIDIA and AMD release columns, Mac side (lane/release-gpu-columns,
2026-09-22): placing a fetched column, the diff against the CPU column, the
release step that stops on a DIVERGENT cell and NAMES it, and the legs that
count as unfinished until their column came home. Synthetic columns in a
temporary release-check directory; nothing is rented and no lane runs.

    .pixi/envs/test/bin/python -m pytest tools/test_release_gpu_columns.py -q
"""
import argparse
import json
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402
import release_gpu_columns as rgc  # noqa: E402

COMMIT = "c" * 40
FIX = ("base", "denormal", "odd")


def column(vendor, lanes, model="m", perturb=None, cpu=False):
    """A minimal one-fit column: every lane/fixture STABLE with a train hash,
    an infer hash and a model hash (n/a on a CPU column, which loads the GPU
    column's saved model and writes none)."""
    cells = {}
    for lane in lanes:
        for f in FIX:
            h = f"{abs(hash((lane, f))) % 16**16:016x}"
            key = f"{lane}/{f}"
            if perturb == key:
                h = ("0" if h[0] != "0" else "1") + h[1:]
            cells[key] = dict(verdict="STABLE", hashes=[h], parts=[dict(out=h)],
                              infer=[h[::-1]], infer_verdict="STABLE", reload=[h[::-1]],
                              model=["n/a:gpu-saved-file (a CPU column loads the GPU column's model and writes none)"]
                              if cpu else [model + h[:15]], model_verdict="N/A" if cpu else "STABLE")
    return dict(mode="identical", repeats=1, vendor=vendor, commit=COMMIT, complete=True,
                fixtures={f: "fx-" + f for f in FIX}, heldout={f: "ho-" + f for f in FIX}, cells=cells)


def record(d, col, lanes):
    d.mkdir(parents=True, exist_ok=True)
    (d / "column.json").write_text(json.dumps(col))
    (d / "manifest.json").write_text(json.dumps(dict(commit=COMMIT, lanes=lanes)))
    (d / "run-summary.json").write_text(json.dumps(dict(complete=True, validation_failures=[])))


LANES = ["kmeans", "ols"]


class Base(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._t.name)
        self._env = os.environ.get("MOJOLEARN_RELEASE_CHECK_DIR")
        os.environ["MOJOLEARN_RELEASE_CHECK_DIR"] = str(self.tmp / "rc")
        record(rgc.check_dir(COMMIT) / "cpu", column("cpu-test", LANES, cpu=True), LANES)

    def tearDown(self):
        if self._env is None:
            os.environ.pop("MOJOLEARN_RELEASE_CHECK_DIR", None)
        else:
            os.environ["MOJOLEARN_RELEASE_CHECK_DIR"] = self._env
        self._t.cleanup()

    def legs_columns(self, perturb=None):
        record(self.tmp / "legcol-cuda" / "cuda", column("nvidia-h100", LANES), LANES)
        record(self.tmp / "legcol-hip" / "hip", column("amd-mi325x", LANES, perturb=perturb), LANES)
        rgc.place(self.tmp / "legcol-cuda", "cuda", COMMIT)
        rgc.place(self.tmp / "legcol-hip", "hip", COMMIT)


class CompareTests(Base):
    def test_a_perturbed_amd_cell_fails_and_is_named(self):
        """THE NEGATIVE CONTROL, run first: one AMD train hash moved."""
        self.legs_columns(perturb="kmeans/denormal")
        lines = []
        v = rgc.compare(COMMIT, supplement=False, out=lines.append)
        self.assertFalse(v["ok"])
        # the perturbed hash feeds train, infer and model in this synthetic cell
        self.assertEqual({d["cell"] for d in v["divergent"]}, {"kmeans/denormal"})
        self.assertEqual([d["part"] for d in v["divergent"]], ["train", "infer", "model"])
        self.assertTrue(any(ln.startswith("# DIVERGENT: kmeans/denormal part=train") for ln in lines), lines)
        self.assertFalse(rgc.verdict_ok(COMMIT))

    def test_identical_columns_pass_and_gpu_only_parts_meet_each_other(self):
        self.legs_columns()
        v = rgc.compare(COMMIT, supplement=False, out=lambda m: None)
        self.assertTrue(v["ok"], v)
        # the model part the CPU column does not write was held to the other GPU
        self.assertEqual(v["uncompared"], [])
        self.assertTrue(rgc.verdict_ok(COMMIT))

    def test_a_missing_column_fails(self):
        record(self.tmp / "legcol-cuda" / "cuda", column("nvidia-h100", LANES), LANES)
        rgc.place(self.tmp / "legcol-cuda", "cuda", COMMIT)
        v = rgc.compare(COMMIT, supplement=False, out=lambda m: None)
        self.assertFalse(v["ok"])
        self.assertEqual(v["missing"], ["hip"])

    def test_a_lane_the_cpu_column_lacks_is_supplemented_or_fails(self):
        more = LANES + ["ridge"]
        record(self.tmp / "legcol-cuda" / "cuda", column("nvidia-h100", more), more)
        record(self.tmp / "legcol-hip" / "hip", column("amd-mi325x", more), more)
        rgc.place(self.tmp / "legcol-cuda", "cuda", COMMIT)
        rgc.place(self.tmp / "legcol-hip", "hip", COMMIT)
        v = rgc.compare(COMMIT, supplement=False, out=lambda m: None)
        self.assertFalse(v["ok"])
        self.assertEqual(v["missing"], ["cpu:ridge"])
        ran = []

        def fake_run(cmd, cwd=None):
            ran.append(cmd)
            out = pathlib.Path(cmd[cmd.index("--out") + 1])
            record(out, column("cpu-test", ["ridge"], cpu=True), ["ridge"])
            return 0
        v = rgc.compare(COMMIT, supplement=True, run=fake_run, out=lambda m: None)
        self.assertEqual(v["supplement"], ["ridge"])
        self.assertIn("--lanes", ran[0])
        self.assertEqual(ran[0][ran[0].index("--lanes") + 1], "ridge")
        self.assertTrue(v["ok"], v)

    def test_place_refuses_an_incomplete_column(self):
        d = self.tmp / "legcol-cuda" / "cuda"
        record(d, column("nvidia-h100", LANES), LANES)
        (d / "run-summary.json").write_text(json.dumps(dict(complete=False)))
        with self.assertRaises(RuntimeError):
            rgc.place(self.tmp / "legcol-cuda", "cuda", COMMIT)


def args(**kw):
    base = dict(version="0.8.15", dry_run=False, publish=None, only="", redo="", build_backend="gpu-legs",
                amd_expect_from="", smoke_gpu="", state_dir="")
    base.update(kw)
    return argparse.Namespace(**base)


class ReleaseStepTests(Base):
    def make(self, **kw):
        r = release.Release(args(state_dir=str(self.tmp / "state"), **kw),
                            runner=lambda cmd, env, log, detach=False: 0)
        r.state["commit"] = COMMIT
        r.lines = []
        r.say = r.lines.append
        return r

    def put_leg_columns(self, r, perturb=None):
        legs = {l.name: l for l in release.gpu_legs(r)}
        record(legs["cuda-sm_90a"].column_dir / "cuda", column("nvidia-h100", LANES), LANES)
        record(legs["hip-gfx942"].column_dir / "hip", column("amd-mi325x", LANES, perturb=perturb), LANES)
        return legs

    def test_the_release_stops_on_a_divergent_cell_and_names_it(self):
        r = self.make(only="gpu-columns")
        self.put_leg_columns(r, perturb="ols/odd")
        self.assertEqual(r.go(), 1)
        text = "\n".join(r.lines)
        self.assertIn("STOPPED at gpu-columns", text)
        self.assertIn("DIVERGENT ols/odd part=train", text)
        # and the Linux wheel cannot be published past it
        r2 = self.make(only="publish-linux", publish="pypi")
        self.assertEqual(r2.go(), 1)
        self.assertIn("have not passed", "\n".join(r2.lines))

    def test_the_release_passes_identical_columns(self):
        r = self.make(only="gpu-columns")
        self.put_leg_columns(r)
        self.assertEqual(r.go(), 0, "\n".join(r.lines))
        self.assertTrue(r.recorded("gpu-columns"))
        self.assertTrue((rgc.check_dir(COMMIT) / "cuda" / "column.json").is_file(),
                        "the NVIDIA column must be placed where the next pass finds its anchor")

    def test_a_built_leg_without_its_column_is_not_done(self):
        r = self.make()
        legs = {l.name: l for l in release.gpu_legs(r)}
        amd = legs["hip-gfx942"]
        sets = amd.release_build / "build" / "sets" / "hip" / "gfx942"
        sets.mkdir(parents=True)
        (amd.release_build / "build" / "build-provenance.json").write_text(
            json.dumps(dict(complete=True, build_exit=0, source_commit=COMMIT)))
        amd.workdir.mkdir(parents=True, exist_ok=True)
        amd.exit_file.write_text("0\n")
        self.assertTrue(amd.proof_ok(COMMIT))
        self.assertFalse(amd.done(COMMIT), "a leg whose column never came home must be relaunched")
        record(amd.column_dir / "hip", column("amd-mi325x", LANES), LANES)
        self.assertTrue(amd.done(COMMIT))
        self.assertTrue(legs["cuda-sm_89"].column_ok(COMMIT), "sm_89 records no column")

    def test_the_column_legs_carry_the_selection_and_a_longer_lease(self):
        r = self.make()
        legs = {l.name: l for l in release.gpu_legs(r)}
        self.assertEqual(legs["cuda-sm_90a"].env["MOJOLEARN_RELEASE_COLUMN_SELECTION"],
                         str(r.column_selection("cuda")))
        self.assertEqual(legs["hip-gfx942"].env["MOJOLEARN_RELEASE_COLUMN_SELECTION"],
                         str(r.column_selection("hip")))
        self.assertNotIn("MOJOLEARN_RELEASE_COLUMN_SELECTION", legs["cuda-sm_89"].env)
        cmd = legs["cuda-sm_90a"].command
        self.assertGreater(int(cmd[cmd.index("--minutes") + 1]), 60)


class ParseTests(unittest.TestCase):
    def test_parse_both_tables(self):
        text = "\n".join([
            "| lane/fixture | verdict | cpu-x | nvidia |",
            "|---|---|---|---|",
            "| kmeans/base | IDENTICAL x2 | aa | aa |",
            "| ols/odd | DIVERGENT | aa | bb |",
            "",
            "| lane/fixture | column | verdict | cpu-x | nvidia |",
            "| ols/odd | infer | DIVERGENT | cc | dd |",
            "| ols/base | model | ONE-COLUMN | n/a | ee |",
            "REQUIRE FAIL ols/base model: ONE-COLUMN rests on 1 real hash(es)",
            "FIXTURE MISMATCH on 'base': x"])
        bad, unc, mis = rgc.parse_diff(text)
        self.assertEqual([(k, p) for k, p, _, _ in bad], [("ols/odd", "train"), ("ols/odd", "infer")])
        self.assertEqual(bad[1][3], {"cpu-x": "cc", "nvidia": "dd"})
        self.assertEqual(len(unc), 1)
        self.assertEqual(len(mis), 1)


if __name__ == "__main__":
    unittest.main()
