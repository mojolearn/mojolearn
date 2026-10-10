"""The release's cross-compile step (tools/release.py step_cross_compile), with
`gh` stubbed on PATH: it dispatches .github/workflows/cross-compile-check.yml
on the FROZEN commit against the previous release's source commit, finds its
run by token, waits, and fails BY NAME on any job that did not pass; it gates
linux-builds (the first rental) and nothing on the Mac.

    python3 -m pytest -q tools/test_release_cross_compile.py
"""
import argparse
import json
import os
import pathlib
import stat
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402

FROZEN = "c" * 40
BASE = "b" * 40

GH_STUB = r'''#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
with open(os.environ["GH_RECORD"], "a") as fh:
    fh.write(json.dumps(a) + "\n")
if a[:2] == ["workflow", "run"]:
    sys.exit(int(os.environ.get("GH_DISPATCH_RC", "0")))
if a[:2] == ["run", "list"]:
    token = ""
    for line in open(os.environ["GH_RECORD"]):
        for f in json.loads(line):
            if f.startswith("token="):
                token = f[6:]
    print(json.dumps([{"databaseId": 1, "displayTitle": "someone else's run", "url": "u1"},
                      {"databaseId": 42, "displayTitle": "cross-compile-check x vs y " + token,
                       "url": "https://github.com/o/r/actions/runs/42"}]))
    sys.exit(0)
if a[:2] == ["run", "watch"]:
    print("watching")
    sys.exit(int(os.environ.get("GH_WATCH_RC", "0")))
if a[:2] == ["run", "view"]:
    print(os.environ.get("GH_FAILED_JOBS", "").replace(";", "\n"))
    sys.exit(0)
sys.exit(3)
'''


def args(**kw):
    base = dict(version="0.8.99", dry_run=False, publish=None, only="", redo="", build_backend="gpu-legs",
                amd_expect_from="", smoke_gpu="", state_dir="", amd_build_provider=None, amd_provider="auto",
                cpu_column=False, status=False, refreeze=False, source_checkout="")
    base.update(kw)
    return argparse.Namespace(**base)


class CrossCompileStep(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._t.name)
        bindir = self.tmp / "bin"
        bindir.mkdir()
        gh = bindir / "gh"
        gh.write_text(GH_STUB.replace("/usr/bin/env python3", sys.executable, 1))
        gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
        self.record = self.tmp / "gh.jsonl"
        self._env = mock.patch.dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}",
                                    GH_RECORD=str(self.record), MOJOLEARN_EVIDENCE_ROOT=str(self.tmp / "ev"),
                                    MOJOLEARN_RELEASE_LEDGER="off")
        self._env.start()
        self._prev = mock.patch.object(release.release_reuse, "previous_release",
                                       return_value=dict(version="0.8.22", source_commit=BASE))
        self._prev.start()

    def tearDown(self):
        self._prev.stop()
        self._env.stop()
        self._t.cleanup()

    def release(self, **kw):
        r = release.Release(args(state_dir=str(self.tmp / "state"), **kw))
        r.state["commit"] = FROZEN
        r.lines = []
        r.say = r.lines.append
        r.sleep = lambda s: None
        return r

    def calls(self):
        return [json.loads(x) for x in self.record.read_text().splitlines()] if self.record.exists() else []

    def test_dispatches_the_frozen_commit_against_the_previous_release_and_passes(self):
        r = self.release()
        out, data = r.step_cross_compile()
        self.assertIn("PASS vs 0.8.22", out)
        self.assertEqual(data["run"], "https://github.com/o/r/actions/runs/42")
        calls = self.calls()
        run = calls[0]
        self.assertEqual(run[:4], ["workflow", "run", "cross-compile-check.yml", "--ref"])
        self.assertIn(f"ref={BASE}", run)
        self.assertIn(f"commit={FROZEN}", run)
        self.assertIn("archs=sm_90a,sm_89,gfx942", run)
        watch = [c for c in calls if c[:2] == ["run", "watch"]]
        self.assertEqual(watch[0][2], "42", "it waits on ITS run, found by token, not the newest")
        self.assertIn("--exit-status", watch[0])

    def test_a_failed_job_fails_the_step_by_name(self):
        with mock.patch.dict(os.environ, GH_WATCH_RC="1",
                             GH_FAILED_JOBS="_mojolearn_gp fast gfx942;_mojolearn_mixture identical gfx942;summary"):
            with self.assertRaises(release.StepFailed) as cm:
                self.release().step_cross_compile()
        msg = str(cm.exception)
        self.assertIn("_mojolearn_gp fast gfx942", msg)
        self.assertIn("_mojolearn_mixture identical gfx942", msg)
        self.assertNotIn("summary", msg)
        self.assertIn("runs/42", msg)

    def test_a_failed_dispatch_fails_the_step(self):
        with mock.patch.dict(os.environ, GH_DISPATCH_RC="1"):
            with self.assertRaises(release.StepFailed):
                self.release().step_cross_compile()
        self.assertFalse([c for c in self.calls() if c[:2] == ["run", "watch"]])

    def test_no_previous_source_commit_refuses_it_never_widens(self):
        with mock.patch.object(release.release_reuse, "previous_release", return_value=None):
            with self.assertRaises(release.StepFailed) as cm:
                self.release().step_cross_compile()
        self.assertIn("no published release record", str(cm.exception))
        self.assertEqual(self.calls(), [])

    def test_dry_run_dispatches_nothing(self):
        out = self.release(dry_run=True).step_cross_compile()
        self.assertIn("would dispatch", out)
        self.assertEqual(self.calls(), [])

    def test_it_gates_the_first_rental_and_nothing_on_the_mac(self):
        self.assertIn("cross-compile", release.NEEDS["linux-builds"])
        self.assertEqual(release.NEEDS["cross-compile"], ["freeze-commit"])
        self.assertEqual(release.PIPELINE_OF["cross-compile"], "core-linux")
        for mac in ("macos-build", "macos-smoke", "publish-macos", "reuse-plan", "rehearsal"):
            self.assertNotIn("cross-compile", release.NEEDS[mac])
        self.assertIn("cross-compile", release.Release.SKIP_IF_RECORDED)

    def test_the_workflow_is_dispatch_only(self):
        text = (ROOT / ".github" / "workflows" / "cross-compile-check.yml").read_text()
        on = text.split("\non:\n", 1)[1].split("\n\n", 1)[0]
        self.assertIn("workflow_dispatch", on)
        for trigger in ("push:", "pull_request", "schedule:"):
            self.assertNotIn(trigger, on)


if __name__ == "__main__":
    unittest.main()
