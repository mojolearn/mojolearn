"""A release is LIGHT and PARALLEL (2026-09-25), proved without a
rental: the NVIDIA and AMD wheel columns are launched together and both
awaited, one failing does not stop the other. Each column is the installed
wheel's smoke against the admitted reference table only (Andrew 2026-10-10: identity runs ONCE; a mismatch is a bug to fix, never a reason to rerun):
no second identity pass, no Apple reference, no joint diff. Nothing here
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
                smoke_via="rent", hopper_box="", smoke_branch="main",   # the rented route's mechanics; lq: test_release_lq_smoke.py
                cpu_column=False)
    base.update(kw)
    return argparse.Namespace(**base)


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
        # the split set: the core above and both plugins beside it
        self.plugins = {}
        for prefix in ("mojolearn_nvidia", "mojolearn_amd"):
            p = final.parent / WHEEL.replace("mojolearn-", prefix + "-", 1)
            with zipfile.ZipFile(p, "w") as z:
                z.writestr(f"mojolearn/{prefix}.txt", prefix)
            self.plugins[prefix] = p
        return r

    def columns(self, r):
        """Both GPU columns launched and awaited together: what the
        gpu-column-nvidia and gpu-column-amd steps run between them."""
        return r.run_columns(("nvidia", "amd"))

    def finish(self, r, name, rc=0, log=""):
        """What a column leg leaves behind when it exits."""
        vendor = "cuda" if name == "nvidia" else "hip"
        out = r.rel / ("smoke-linux" if name == "nvidia" else "column-amd")
        out.mkdir(parents=True, exist_ok=True)
        if rc == 0:
            # each column's smoke receipt: the core, with both plugins installed beside it
            (out / "results.json").write_text(json.dumps(dict(
                status="PASSED", scope="expanded", source_commit=COMMIT,
                # the loader's selection on the box (Release.column_arch_ok)
                installed=dict(vendor=vendor, gpu_arch="sm_89" if vendor == "cuda" else "gfx942"),
                wheel_sha256=hashlib.sha256(self.final.read_bytes()).hexdigest(),
                plugins=[dict(wheel=str(p), wheel_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                         for p in self.plugins.values()])))
        (r.rel / "columns").mkdir(parents=True, exist_ok=True)
        (r.rel / "columns" / f"{name}.log").write_text(log)
        (r.rel / "columns" / f"{name}.exit").write_text(f"{rc}\n")

    # ------------------------------------------------------------------ tests
    def test_both_columns_launch_together_and_both_are_awaited(self):
        r = self.release()
        # AMD fails first; NVIDIA is still running and must still be awaited
        self.hooks = [lambda r: self.finish(r, "amd", rc=1, log="boom"),
                      lambda r: None,
                      lambda r: self.finish(r, "nvidia")]
        with self.assertRaises(release.StepFailed) as cm:
            self.columns(r)
        self.assertEqual(self.events[:2], ["spawn", "spawn"], "both launched before any wait")
        self.assertEqual(self.events.count("sleep"), 3, "NVIDIA was awaited after AMD failed")
        self.assertEqual(len(self.spawned), 2)
        nv, amd = (c[2] for c in self.spawned)
        self.assertIn("release_wheel_smoke.sh", nv)
        self.assertIn(f"--gpu '{release.NVIDIA_WALK['sm_89'][0]}'", nv)   # the sm_89 walk's head
        self.assertIn("--vendor hip", amd)
        self.assertIn("--provider auto", amd)
        # Andrew 2026-10-10: identity runs ONCE; a mismatch is a bug to fix, never a reason to rerun: the smoke only, no second identity pass
        for c in (nv, amd):
            self.assertNotIn("--column", c)
            self.assertNotIn("--ref-column", c)
        msg = str(cm.exception)
        self.assertIn("amd column", msg)
        self.assertNotIn("nvidia column", msg)
        self.assertIn("nvidia column: wheel smoke PASSED", "\n".join(r.lines))

    def test_both_failing_are_both_reported(self):
        r = self.release()
        self.hooks = [lambda r: (self.finish(r, "amd", rc=1), self.finish(r, "nvidia", rc=3))]
        with self.assertRaises(release.StepFailed) as cm:
            self.columns(r)
        self.assertIn("nvidia column", str(cm.exception))
        self.assertIn("amd column", str(cm.exception))
        self.assertIn("exit 3", str(cm.exception))

    def test_a_passed_column_is_not_rerun(self):
        r = self.release()
        (r.rel / "columns").mkdir(parents=True)
        self.finish(r, "nvidia")
        self.hooks = [lambda r: self.finish(r, "amd")]
        self.columns(r)
        self.assertEqual(len(self.spawned), 1)
        self.assertIn("--vendor hip", self.spawned[0][2])
        self.assertIn("  nvidia: PASSED for this wheel (record exists), not rerun", r.lines)

    def test_nvidia_walks_to_the_next_gpu_on_no_stock(self):
        r = self.release()
        self.hooks = [lambda r: (self.finish(r, "amd"),
                                 self.finish(r, "nvidia", rc=1, log="create: There are no instances currently available")),
                      lambda r: self.finish(r, "nvidia")]
        self.columns(r)
        self.assertEqual(len(self.spawned), 3)
        self.assertIn(f"--gpu '{release.NVIDIA_WALK['sm_89'][1]}'", self.spawned[2][2])
        self.assertEqual((r.rel / "columns" / "nvidia.gpu").read_text(), "1")
        self.assertEqual(len(list((r.rel / "columns").glob("nvidia.log.failed-*"))), 1)

    def test_the_walk_does_not_wait_for_the_other_column(self):
        # 0.8.19: an out-of-stock leg sat idle until every other leg finished
        r = self.release()
        seen = []
        self.hooks = [lambda r: self.finish(r, "nvidia", rc=1, log="create: There are no instances currently available"),
                      lambda r: (seen.append(len(self.spawned)), self.finish(r, "amd")),
                      lambda r: self.finish(r, "nvidia")]
        self.columns(r)
        self.assertEqual(seen, [3], "the NVIDIA walk launched while AMD was still running")

    def test_the_flag_parses(self):
        with mock.patch.object(release.Release, "go", lambda self: self.args):
            self.assertFalse(release.main(["0.8.14"]).cpu_column)
            self.assertTrue(release.main(["0.8.14", "--cpu-column"]).cpu_column)
            self.assertEqual(release.main(["0.8.14"]).build_backend, "github")


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
