"""A release is LIGHT and PARALLEL (Andrew, 2026-09-25), proved without a
rental: the NVIDIA and AMD wheel columns are launched together and both
awaited, one failing does not stop the other, they are diffed against the
Apple column (the CPU column only with --cpu-column), a DIVERGENT cell
anywhere stops the release. Nothing here
reaches a cloud: the runner stands in for every process."""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402
import release_check  # noqa: E402

COMMIT = "a" * 40
WHEEL = "mojolearn-0.8.14-py3-none-manylinux_2_35_x86_64.whl"


def args(**kw):
    base = dict(version="0.8.14", dry_run=False, publish=None, only="", redo="", build_backend="gpu-legs",
                amd_expect_from="", smoke_gpu="", state_dir="", amd_build_provider=None, amd_provider="auto",
                cpu_column=False)
    base.update(kw)
    return argparse.Namespace(**base)


def column(vendor, h="aaaa"):
    return dict(mode="identical", repeats=1, vendor=vendor, commit=COMMIT, complete=True,
                cells={"rf-clf/base": {"verdict": "STABLE", "hashes": [h], "parts": [{"predict": h}]}})


class Columns(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._t.name)
        self.check = self.tmp / "release-check"
        self._env = mock.patch.dict(os.environ, MOJOLEARN_RELEASE_CHECK_DIR=str(self.check))
        self._env.start()
        self.spawned, self.events, self.hooks = [], [], []

    def tearDown(self):
        self._env.stop()
        self._t.cleanup()

    def release(self, **kw):
        def runner(cmd, env, log, detach=False):
            if detach:
                self.spawned.append(cmd)
                self.events.append("spawn")
                return os.getpid()          # alive until its exit file is written
            return 0
        r = release.Release(args(state_dir=str(self.tmp / "state"), **kw), runner=runner)
        r.state["commit"] = COMMIT
        r.lines = []
        r.say = r.lines.append
        r.gpu_selection = lambda vendor: self.tmp / f"selection-{vendor}.json"

        def sleep(s):
            self.events.append("sleep")
            if self.hooks:
                self.hooks.pop(0)(r)
            elif self.events.count("sleep") > 20:
                raise AssertionError("waited on a leg that was never launched alongside the others")
        r.sleep = sleep
        final = r.rel / "linux" / "final" / WHEEL
        final.parent.mkdir(parents=True)
        with zipfile.ZipFile(final, "w") as z:
            z.writestr("mojolearn/identity_columns/COMMIT", COMMIT + "\n")
        self.final = final
        return r

    def reference(self, backend, h="aaaa"):
        d = self.check / COMMIT[:12] / backend
        d.mkdir(parents=True, exist_ok=True)
        (d / "column.json").write_text(json.dumps(column("apple-m4" if backend == "metal" else "cpu-m4", h)))
        (d / "run-summary.json").write_text(json.dumps(dict(complete=True, validation_failures=[])))

    def finish(self, r, name, rc=0, h="aaaa", diff="summary: IDENTICAL=1\n", log=""):
        """What a column leg leaves behind when it exits."""
        vendor = "cuda" if name == "nvidia" else "hip"
        out = r.rel / ("smoke-linux" if name == "nvidia" else "column-amd")
        out.mkdir(parents=True, exist_ok=True)
        if rc == 0:
            if name == "nvidia":
                (out / "results.json").write_text(json.dumps(dict(
                    status="PASSED", scope="expanded", source_commit=COMMIT,
                    wheel_sha256=hashlib.sha256(self.final.read_bytes()).hexdigest())))
            (out / f"column-{vendor}.json").write_text(json.dumps(column(vendor, h)))
            (out / f"diff-ref-{vendor}.txt").write_text(diff)
        (r.rel / "columns").mkdir(parents=True, exist_ok=True)
        (r.rel / "columns" / f"{name}.log").write_text(log)
        (r.rel / "columns" / f"{name}.exit").write_text(f"{rc}\n")

    # ------------------------------------------------------------------ tests
    def test_both_columns_launch_together_and_both_are_awaited(self):
        self.reference("metal")
        r = self.release()
        # AMD fails first; NVIDIA is still running and must still be awaited
        self.hooks = [lambda r: self.finish(r, "amd", rc=1, log="boom"),
                      lambda r: None,
                      lambda r: self.finish(r, "nvidia")]
        with self.assertRaises(release.StepFailed) as cm:
            r.step_gpu_columns()
        self.assertEqual(self.events[:2], ["spawn", "spawn"], "both launched before any wait")
        self.assertEqual(self.events.count("sleep"), 3, "NVIDIA was awaited after AMD failed")
        self.assertEqual(len(self.spawned), 2)
        nv, amd = (c[2] for c in self.spawned)
        self.assertIn("release_wheel_smoke.sh", nv)
        self.assertIn("--gpu 'NVIDIA GeForce RTX 4090'", nv)
        self.assertIn("--vendor hip", amd)
        self.assertIn("--provider auto", amd)
        # both diffed against the Apple column, not the CPU column
        for c in (nv, amd):
            self.assertIn("--ref-column " + str(self.check / COMMIT[:12] / "metal" / "column.json"), c)
            self.assertNotIn("cpu/column.json", c)
            self.assertNotIn("--cpu-column", c)
        msg = str(cm.exception)
        self.assertIn("AMD column", msg)
        self.assertNotIn("NVIDIA column", msg)
        self.assertIn("NVIDIA column: PASSED", "\n".join(r.lines))

    def test_both_failing_are_both_reported(self):
        self.reference("metal")
        r = self.release()
        self.hooks = [lambda r: (self.finish(r, "amd", rc=1), self.finish(r, "nvidia", rc=3))]
        with self.assertRaises(release.StepFailed) as cm:
            r.step_gpu_columns()
        self.assertIn("NVIDIA column", str(cm.exception))
        self.assertIn("AMD column", str(cm.exception))
        self.assertIn("exit 3", str(cm.exception))

    def test_all_identical_passes_and_a_divergent_amd_cell_fails(self):
        self.reference("metal")
        r = self.release()
        self.hooks = [lambda r: (self.finish(r, "amd"), self.finish(r, "nvidia"))]
        result = r.step_gpu_columns()
        self.assertIn("no DIVERGENT cell across the columns", result)
        self.assertTrue(any(l.startswith("  summary: IDENTICAL=1") for l in r.lines), r.lines)
        # NVIDIA and AMD each agreed with Apple in their own diff, but disagree with
        # each other: the all-columns diff must stop the release
        r2 = self.release_again()
        self.hooks = [lambda r: (self.finish(r, "amd", h="bbbb"), self.finish(r, "nvidia"))]
        with self.assertRaises(release.StepFailed) as cm:
            r2.step_gpu_columns()
        self.assertIn("DIVERGENT", str(cm.exception))
        self.assertIn("rf-clf/base", str(cm.exception))

    def release_again(self):
        self._t2 = tempfile.TemporaryDirectory()
        self.addCleanup(self._t2.cleanup)
        self.tmp = pathlib.Path(self._t2.name)
        return self.release()

    def test_a_divergent_reference_diff_is_not_a_pass(self):
        self.reference("metal")
        r = self.release()
        self.hooks = [lambda r: (self.finish(r, "amd", diff="| x/base | DIVERGENT |\nsummary: DIVERGENT=1\n"),
                                 self.finish(r, "nvidia"))]
        with self.assertRaises(release.StepFailed) as cm:
            r.step_gpu_columns()
        self.assertIn("AMD column", str(cm.exception))

    def test_a_passed_column_is_not_rerun(self):
        self.reference("metal")
        r = self.release()
        (r.rel / "columns").mkdir(parents=True)
        self.finish(r, "nvidia")
        self.hooks = [lambda r: self.finish(r, "amd")]
        r.step_gpu_columns()
        self.assertEqual(len(self.spawned), 1)
        self.assertIn("--vendor hip", self.spawned[0][2])
        self.assertIn("  nvidia: PASSED for this wheel (record exists), not rerun", r.lines)

    def test_nvidia_walks_to_the_next_gpu_on_no_stock(self):
        self.reference("metal")
        r = self.release()
        self.hooks = [lambda r: (self.finish(r, "amd"),
                                 self.finish(r, "nvidia", rc=1, log="create: There are no instances currently available")),
                      lambda r: self.finish(r, "nvidia")]
        r.step_gpu_columns()
        self.assertEqual(len(self.spawned), 3)
        self.assertIn("--gpu 'NVIDIA L40S'", self.spawned[2][2])
        self.assertEqual((r.rel / "columns" / "nvidia.gpu").read_text(), "1")
        self.assertEqual(len(list((r.rel / "columns").glob("nvidia.log.failed-*"))), 1)

    def test_no_reference_column_refuses_before_any_rental(self):
        r = self.release()
        with self.assertRaises(release.StepFailed) as cm:
            r.step_gpu_columns()
        self.assertIn("metal/column.json", str(cm.exception))
        self.assertEqual(self.spawned, [])

    def test_cpu_column_is_opt_in(self):
        self.reference("metal")
        r = self.release()
        self.assertEqual(r.check_backends(), ("metal",))
        self.assertEqual(r.column_refs(), [self.check / COMMIT[:12] / "metal" / "column.json"])
        # complete WITHOUT any CPU record: the GPU columns no longer need one
        self.assertTrue(r.release_check_complete())
        calls = []
        r.runner = lambda cmd, env, log, detach=False: calls.append([str(c) for c in cmd]) or 0
        self.assertIn("complete", r.step_release_check())
        self.assertEqual(calls, [])
        (self.check / COMMIT[:12] / "metal" / "column.json").unlink()
        with mock.patch.object(release, "git", return_value=COMMIT):
            with self.assertRaises(release.StepFailed):   # the stand-in runner writes no record
                r.step_release_check()
        self.assertEqual(calls[-1], ["pixi", "run", "-e", "test", "release-check"])

        on = self.release_again()
        on.args.cpu_column = True
        self.assertEqual(on.check_backends(), ("metal", "cpu"))
        self.assertFalse(on.release_check_complete(), "the CPU record is required once asked for")
        self.reference("metal")
        self.reference("cpu")
        self.assertTrue(on.release_check_complete())
        self.assertIn(self.check / COMMIT[:12] / "cpu" / "column.json", on.column_refs())
        (self.check / COMMIT[:12] / "cpu" / "column.json").unlink()
        calls.clear()
        on.runner = lambda cmd, env, log, detach=False: calls.append([str(c) for c in cmd]) or 0
        with mock.patch.object(release, "git", return_value=COMMIT):
            with self.assertRaises(release.StepFailed):
                on.step_release_check()
        self.assertEqual(calls[-1][-1], "--cpu-column")
        # and the GPU columns are diffed against both
        self.reference("cpu")
        legs = on.column_legs()
        for l in legs:
            joined = " ".join(l.command)
            self.assertIn("metal/column.json", joined)
            self.assertIn("cpu/column.json", joined)

    def test_the_flag_parses(self):
        with mock.patch.object(release.Release, "go", lambda self: self.args):
            self.assertFalse(release.main(["0.8.14"]).cpu_column)
            self.assertTrue(release.main(["0.8.14", "--cpu-column"]).cpu_column)
            self.assertEqual(release.main(["0.8.14"]).build_backend, "gpu-legs")


class JointDiff(unittest.TestCase):
    def test_verdicts(self):
        p = pathlib.Path("diff-columns.txt")
        ok = "summary: IDENTICAL=3, ONE-COLUMN=2\nsummary (infer/model): IDENTICAL=6\n"
        self.assertIn("2 cell(s) hashed by one column only", release.judge_joint_diff(ok, p, say=lambda m: None))
        for bad in ("summary: IDENTICAL=3, DIVERGENT=1\n",
                    "summary: IDENTICAL=3\nsummary (infer/model): RELOAD-MOVED=1\n",
                    "summary: IDENTICAL=3\nsummary (batch): BATCH_MOVED=2\n",
                    "| a/base | DIVERGENT |\nsummary: IDENTICAL=3\n",
                    "Traceback: refused a mix of columns\n"):
            with self.assertRaises(release.StepFailed, msg=bad):
                release.judge_joint_diff(bad, p, say=lambda m: None)


class ReleaseCheck(unittest.TestCase):
    def test_cpu_pass_is_opt_in(self):
        self.assertEqual(release_check.default_backends([]), ("metal",))
        self.assertEqual(release_check.default_backends(["--cpu-column"]), ("cpu", "metal"))

    def run_main(self, argv):
        started = []

        class Proc:
            def __init__(self, cmd, **kw):
                started.append(cmd)

            def wait(self):
                return 0

            def poll(self):
                return 0
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, MOJOLEARN_RELEASE_CHECK_DIR=d), \
                mock.patch.object(sys, "argv", ["release_check.py", *argv]), \
                mock.patch.object(release_check, "subprocess",
                                  argparse.Namespace(Popen=Proc, STDOUT=release_check.subprocess.STDOUT)):
            self.assertEqual(release_check.main(), 0)
        return [c[-1] for c in started]

    def test_default_runs_the_apple_pass_alone(self):
        self.assertEqual(self.run_main([]), ["--apple-pass"])

    def test_cpu_column_runs_both_at_once(self):
        self.assertEqual(self.run_main(["--cpu-column"]), ["--apple-pass", "--cpu-pass"])


if __name__ == "__main__":
    unittest.main()
