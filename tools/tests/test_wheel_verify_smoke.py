"""Tooling tests for the verify smoke's GPU mode (lane gpu-smoke, 2026-11-25): tools/ops/wheel_verify_smoke.py (the
lane units on both sides, the per-side record check that never hides a lane, the GPU worker cap, the plan header),
tools/ops/wheel_verify_smoke.sh (SMOKE_SIDE=gpu install refusals) and the GPU smoke lines tools/release.py candidate
prints. Pure Python and bash on temp dirs: nothing is installed, compiled or dispatched; no mojolearn import."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "ops"))
sys.path.insert(0, str(ROOT / "tools"))
import wheel_verify_smoke as smoke   # noqa: E402

SMOKE_SH = ROOT / "tools" / "ops" / "wheel_verify_smoke.sh"
JOB = ROOT / "tools" / "ops" / "gpu_smoke_job.sh"


def _record(path, cells):
    Path(path).write_text(json.dumps({"cells": cells, "partial_column": True}))


OK = {"verdict": "STABLE", "hashes": ["ab" * 8]}
NA = {"verdict": "N/A", "hashes": ["n/a:one-sided: no CPU implementation: GPU only"]}
REFUSED = {"verdict": "REFUSED", "error": "the GPU side did not read a device-resident input in place"}
NOT_RUN = {"verdict": "REFUSED", "error": "not run after interruption: 3 consecutive cell workers crashed"}


class LaneUnits(unittest.TestCase):
    def test_gpu_lane_runs_both_sides_on_its_vendor(self):
        argv = smoke.lane_argv("py", "ols", Path("/j"), True, "cuda")
        self.assertIn("--side", argv)
        self.assertEqual(argv[argv.index("--side") + 1], "both")
        self.assertEqual(argv[argv.index("--require-backend") + 1], "cuda")    # never forced to cpu on a GPU install
        self.assertEqual(argv[argv.index("--json") + 1], "/j/ols.gpu.json")
        self.assertEqual(argv[argv.index("--cpu-json") + 1], "/j/ols.cpu.json")
        self.assertIn("--fail-on-refused", argv)
        self.assertIn("--no-batch", argv)                                       # partial_column: never a record

    def test_cpu_lane_unchanged(self):
        argv = smoke.lane_argv("py", "ols", Path("/j"), False, "cpu")
        self.assertEqual(argv[argv.index("--require-backend") + 1], "cpu")
        self.assertNotIn("--side", argv)
        self.assertEqual(argv[argv.index("--json") + 1], "/j/ols.json")
        self.assertEqual(argv[:4], ["py", "-I", "-m", "mojolearn._identity_break"])


class SideStates(unittest.TestCase):
    def test_states(self):
        with tempfile.TemporaryDirectory() as td:
            cases = {"ok": {"a/base": OK}, "n/a": {"a/base": NA}, "refused": {"a/base": OK, "b/base": REFUSED},
                     "not-run": {"a/base": NOT_RUN}, "missing": {}}
            for want, cells in cases.items():
                p = Path(td) / "x.json"
                _record(p, cells)
                self.assertEqual(smoke.side_state(p)[0], want, cells)
            self.assertEqual(smoke.side_state(Path(td) / "absent.json")[0], "missing")
            (Path(td) / "bad.json").write_text("{")
            self.assertEqual(smoke.side_state(Path(td) / "bad.json")[0], "missing")

    def test_failed_probes_and_malformed_cells_fail_the_side(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.json"
            for cell, state in (
                (dict(OK, batch_verdict="BATCH_MOVED"), "divergent"),
                (dict(OK, model_verdict="RELOAD-MOVED"), "divergent"),
                (dict(OK, probe_error="model probe failed"), "refused"),
                (dict(OK, model_verdict="REFUSED"), "refused"),
                (dict(OK, verdict="unexpected"), "invalid"),
                (dict(OK, hashes=["corrupt"]), "invalid"), (dict(NA, hashes=[""]), "invalid"),
                ({"verdict": "STABLE"}, "invalid"), (dict(OK, infer_verdict="STABLE"), "invalid"),
                ({}, "invalid"), (None, "invalid"),
            ):
                with self.subTest(cell=cell):
                    _record(path, {"x/base": cell})
                    self.assertEqual(smoke.side_state(path)[0], state)

    def test_check_fails_a_lane_missing_a_side_or_refused_but_not_one_sided(self):
        with tempfile.TemporaryDirectory() as td:
            j = Path(td)
            _record(j / "gpuonly.gpu.json", {"gpuonly/base": OK})
            _record(j / "gpuonly.cpu.json", {"gpuonly/base": NA})                # declared one-sided: n/a, passes
            _record(j / "refgpu.gpu.json", {"refgpu/base": REFUSED})
            _record(j / "refgpu.cpu.json", {"refgpu/base": OK})
            _record(j / "nogpu.cpu.json", {"nogpu/base": OK})                    # its GPU side never wrote a record
            _record(j / "moved.gpu.json", {"moved/base": OK})
            _record(j / "moved.cpu.json", {"moved/base": {"verdict": "DIVERGENT", "oracle_errors": ["wrong bytes"]}})
            check = smoke.lane_check(j, True)
            self.assertRegex(check("moved"), "cpu side divergent.*wrong bytes")
            self.assertIsNone(check("gpuonly"))
            self.assertIn("gpu side refused", check("refgpu"))
            self.assertIn("gpu side missing", check("nogpu"))


class PoolNeverHidesALane(unittest.TestCase):
    """A lane that crashed, one whose record holds never-run cells, and one that exited 0 without a GPU record are
    all failures, and every listed lane has a row per side in SIDES.tsv (item 4: no stop hides a lane)."""

    def test_pool(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            jdir = out / "lanes" / "json"
            jdir.mkdir(parents=True)

            def writer(lane, gpu_cells, cpu_cells, rc=0, crash=False):
                code = ("import json, os, signal, sys\n"
                        f"g, c = {gpu_cells!r}, {cpu_cells!r}\n"
                        f"if g is not None: open({str(jdir / (lane + '.gpu.json'))!r}, 'w').write(json.dumps({{'cells': g}}))\n"
                        f"if c is not None: open({str(jdir / (lane + '.cpu.json'))!r}, 'w').write(json.dumps({{'cells': c}}))\n"
                        + ("os.kill(os.getpid(), signal.SIGKILL)\n" if crash else f"sys.exit({rc})\n"))
                return (lane, [sys.executable, "-c", code])
            units = [writer("good", {"good/base": OK}, {"good/base": OK}),
                     writer("onesided", {"onesided/base": OK}, {"onesided/base": NA}),
                     writer("silent", None, {"silent/base": OK}),                  # rc 0, no GPU record
                     writer("stopped", {"stopped/base": NOT_RUN}, {"stopped/base": OK}),
                     writer("crashed", None, None, crash=True)]
            args = argparse.Namespace(out=str(out), jobs=2, memory_from="", gpu=False, gpu_mb_per_worker=2048)
            rc = smoke.run_pool("lanes", units, args, str(td), dict(os.environ), 60,
                                check=smoke.lane_check(jdir, True))
            self.assertEqual(rc, 1)
            rows = dict((line.split("\t")[0], line.split("\t")[2])
                        for line in (out / "lanes" / "results.tsv").read_text().splitlines()[1:])
            self.assertEqual(rows["good"], "ok")
            self.assertEqual(rows["onesided"], "ok")
            self.assertEqual(rows["silent"], "failed")
            self.assertEqual(rows["stopped"], "failed")
            self.assertEqual(rows["crashed"], "crash")
            smoke.sides_report(str(out), jdir, [u for u, _ in units], True)
            sides = (out / "lanes" / "SIDES.tsv").read_text().splitlines()[1:]
            self.assertEqual(len(sides), 10)                                       # 5 lanes x 2 sides, none left out
            summary = (out / "SUMMARY.txt").read_text()
            self.assertIn("lanes rc=1 units=5 failed=3", summary)
            self.assertIn("lanes sides: gpu ok=2 n/a=0 refused=0 not-run=1 missing=2", summary)
            self.assertIn("cpu ok=3 n/a=1", summary)


class GpuPlan(unittest.TestCase):
    def test_worker_cap(self):
        info = dict(name="NVIDIA L40S", driver="595", total_mb=46068, free_mb=45000, tool="nvidia-smi")
        self.assertEqual(smoke.gpu_worker_cap(info, 2048), 19)
        self.assertEqual(smoke.gpu_worker_cap(dict(info, free_mb=100), 2048), 1)
        self.assertIsNone(smoke.gpu_worker_cap(None, 2048))
        self.assertGreaterEqual(smoke.cores(), 1)

    def test_plan_header_without_a_gpu_tool(self):
        env = dict(os.environ, PATH="/usr/bin:/bin")
        p = subprocess.run([sys.executable, str(ROOT / "tools/ops/wheel_verify_smoke.py"), "plan", "--gpu"],
                           capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("# allocation cores=", p.stdout)
        self.assertIn("# parallel plan:", p.stdout)

    def test_plan_header_with_a_gpu(self):
        with tempfile.TemporaryDirectory() as td:
            fake = Path(td) / "nvidia-smi"
            fake.write_text("#!/bin/sh\necho 'NVIDIA L40S, 595.112.07, 46068, 45000'\n")
            fake.chmod(0o755)
            env = dict(os.environ, PATH=f"{td}:/usr/bin:/bin")
            p = subprocess.run([sys.executable, str(ROOT / "tools/ops/wheel_verify_smoke.py"), "plan", "--gpu"],
                               capture_output=True, text=True, env=env, timeout=60)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("# gpu NVIDIA L40S driver=595.112.07 memory_mb=46068", p.stdout)
            self.assertIn("/ 2048 MB a worker = 19)", p.stdout)


class SmokeShellGpuMode(unittest.TestCase):
    def run_smoke(self, td, wheels, side="gpu"):
        wd = Path(td) / "w"
        wd.mkdir()
        for name in wheels:
            (wd / name).write_bytes(b"x")
        out = Path(td) / "out"
        env = dict(os.environ, SMOKE_SIDE=side, STEPS="install", VENV=str(Path(td) / "v" / "venv"),
                   PATH=f"{td}:/usr/bin:/bin")
        p = subprocess.run(["bash", str(SMOKE_SH), str(wd), str(out)], capture_output=True, text=True, env=env,
                           timeout=120)
        return p, (out / "SUMMARY.txt").read_text()

    def test_gpu_mode_needs_one_plugin_wheel(self):
        with tempfile.TemporaryDirectory() as td:
            p, summary = self.run_smoke(td, ["mojolearn_core-1.0-py3-none-any.whl"])
            self.assertEqual(p.returncode, 1)
            self.assertIn("GPU mode expects exactly one GPU plugin wheel", summary)
            self.assertIn("side=gpu", summary)
            self.assertIn("overall rc=1 failed: install", summary)

    def test_bad_side(self):
        with tempfile.TemporaryDirectory() as td:
            p = subprocess.run(["bash", str(SMOKE_SH), td, td], capture_output=True, text=True, timeout=60,
                               env=dict(os.environ, SMOKE_SIDE="both"))
            self.assertEqual(p.returncode, 2)

    def test_parses(self):
        for f in (SMOKE_SH, JOB):
            p = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stderr)


class CandidateLines(unittest.TestCase):
    def test_stage_and_submit(self):
        import release_candidate as rc
        lines = rc.gpu_smoke_lines("0.8.67.dev0", "a" * 40)
        self.assertTrue(lines[0].startswith("GPU-SMOKE-STAGE python3 tools/ops/measure_candidate.py stage nvidia "))
        self.assertIn("--source-commit " + "a" * 40, lines[0])
        self.assertIn("--lane gpu-smoke", lines[0])
        self.assertEqual(lines[1], "GPU-SMOKE tools/nvidia_central.sh submit gpu-smoke --cap 45 --results "
                         "box-out/gsmoke-aaaaaaaaaaaa --env GS_WHEEL_DIR=/root/pc-in/aaaaaaaaaaaa/nvidia "
                         "tools/ops/gpu_smoke_job.sh")
        self.assertIn("gpu_smoke_lines(version, a.commit)", (ROOT / "tools/release_candidate.py").read_text())
        self.assertIn("\n# job-preflight:", JOB.read_text())


if __name__ == "__main__":
    unittest.main()
