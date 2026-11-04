# SPDX-License-Identifier: Apache-2.0
"""The installed-runtime qualifier's per-test and run budgets and its partial junit; a toy pytest run only."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("installed_native_audit", HERE / "qualify_native_runtime_installed.py")
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)

TOY = textwrap.dedent('''
    import time, pytest
    def test_passes():
        assert True
    @pytest.mark.skip(reason="declared")
    def test_skipped():
        pass
    def test_fails():
        assert False, "boom"
    class TestSlow:
        def test_hangs(self):
            time.sleep(60)
    def test_never_reached():
        pass
''')

WORKER = textwrap.dedent('''
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("q", sys.argv[1])
    q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
    import pytest
    sys.exit(int(pytest.main(["-q", "-p", "no:cacheprovider", sys.argv[3]], plugins=[q.ProgressLog(sys.argv[2])])))
''')


class StoppedRunKeepsFinishedTests(unittest.TestCase):
    def run_toy(self, root, *, run_timeout, test_timeout):
        (root / "test_toy.py").write_text(TOY)
        progress = root / q.PROGRESS
        command = [sys.executable, "-c", WORKER, str(HERE / "qualify_native_runtime_installed.py"),
                   str(progress), "test_toy.py"]
        with (root / "tests.log").open("wb") as log:
            return progress, q.run_bounded(command, cwd=root, env=None, log=log, progress=progress,
                                           run_timeout=run_timeout, test_timeout=test_timeout, poll=0.2)

    def check_partial(self, root, progress, stop):
        summary = q.partial_junit(progress, root / q.PARTIAL_JUNIT, stop["kind"] + " timeout")
        outcomes = {case["name"]: case["outcome"] for case in summary["cases"]}
        self.assertEqual(outcomes, {"test_passes": "passed", "test_skipped": "skipped",
                                    "test_fails": "failed", "test_hangs": "failed"})
        self.assertEqual((summary["total"], summary["passed"]), (4, 1))
        hang = next(c for c in summary["cases"] if c["name"] == "test_hangs")
        self.assertEqual(hang["classname"], "test_toy.TestSlow")
        # test_scope_summary maps cases to files by the module stem in the classname
        self.assertTrue(all("test_toy" in c["classname"].split(".") for c in summary["cases"]))

    def test_a_hung_test_stops_on_the_per_test_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            progress, (code, stop, elapsed) = self.run_toy(root, run_timeout=120, test_timeout=3)
            self.assertNotEqual(code, 0)
            self.assertEqual(stop["kind"], "test")
            self.assertLess(elapsed, 30)
            self.check_partial(root, progress, stop)

    def test_the_run_budget_stops_a_long_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            progress, (code, stop, elapsed) = self.run_toy(root, run_timeout=4, test_timeout=120)
            self.assertEqual(stop["kind"], "run")
            self.check_partial(root, progress, stop)

    def test_a_torn_last_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            progress = root / q.PROGRESS
            rows = [dict(event="collect", nodeid="t.py"), dict(event="start", nodeid="t.py::test_a"),
                    dict(event="report", nodeid="t.py::test_a", when="call", outcome="passed"),
                    dict(event="report", nodeid="t.py::test_a", when="teardown", outcome="passed")]
            progress.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"event": "sta')
            summary = q.partial_junit(progress, root / q.PARTIAL_JUNIT)
            self.assertEqual((summary["total"], summary["passed"]), (1, 1))

    def test_defaults_cover_the_measured_full_run(self):
        # 0.8.65: 12% of the selection in 900 s on NVIDIA, about 7,500 s for the whole run.
        self.assertGreaterEqual(q.RUN_TIMEOUT_S, 7500 * 1.4)
        self.assertLess(q.TEST_TIMEOUT_S, q.RUN_TIMEOUT_S)


if __name__ == "__main__":
    unittest.main()
