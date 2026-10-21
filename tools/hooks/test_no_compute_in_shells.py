#!/usr/bin/env python3
"""Tests for tools/hooks/no_compute_in_shells.py on a tiny temporary git tree.

Run: python3 tools/hooks/test_no_compute_in_shells.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.join(HERE, "no_compute_in_shells.py")
BASELINE = "tools/hooks/compute_in_shells_baseline.txt"
sys.path.insert(0, HERE)
import no_compute_in_shells as ncs  # noqa: E402

LOOPY = "def f(n: Int):\n    for i in range(n):\n        pass\n"
CLEAN = "def f(n: Int) -> Int:\n    return kernel(n)\n"


class ScanTest(unittest.TestCase):
    def kinds(self, text):
        return [(n, k) for n, k, _c in ncs.scan(text)]

    def test_loops_tags_and_launches(self):
        text = "\n".join([
            "fn a(n: Int):",
            "    for i in range(n):",                                   # 2 untagged
            "    for i in range(3):  # small-loop(three shape words)",  # 3 tagged
            "    # driver-loop: over model layers",
            "    for b in blocks:",                                     # 5 tagged above
            "    while x < n:",                                         # 6 untagged
            "    var ctx = DeviceContext()",                            # 7 launch
            "    ctx.enqueue_function[k](grid_dim=1)",                  # 8 launch
            "    parallelize[work](n)",                                 # 9 launch
            "    vectorize[body, 4](n)",                                # 10 launch
            "    # parallelize(n) in a comment",                        # 11 comment
            "    var s = format(n)  # for i in a trailing comment",     # 12 not a loop
            "    var forward = 1",                                      # 13 not a loop
            "    var t = \"#\"; for_each(n)",                           # 14 nothing
            "    sync_parallelize[w](n)  # driver-loop: tags never excuse a launch",  # 15 launch
        ])
        self.assertEqual(self.kinds(text), [(2, "loop"), (6, "loop"), (7, "launch"), (8, "launch"),
                                            (9, "launch"), (10, "launch"), (15, "launch")])

    def test_docstrings_are_skipped(self):
        text = '\n'.join([
            'fn a():',
            '    """Sum the rows.',
            '',
            '    for each row, add it; parallelize(n) is never called here.',
            '    """',
            '    """One line docstring."""',
            '    for i in range(2):',
        ])
        self.assertEqual(self.kinds(text), [(7, "loop")])


class TreeTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ncis-")
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "t@localhost")
        self.git("config", "user.name", "t")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.dir, capture_output=True, text=True, check=True).stdout

    def write(self, path, text):
        full = os.path.join(self.dir, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)

    def read(self, path):
        with open(os.path.join(self.dir, path), encoding="utf-8") as f:
            return f.read()

    def commit(self, msg="c"):
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", msg)

    def run_checker(self, *args):
        r = subprocess.run([sys.executable, CHECKER, *args], cwd=self.dir, capture_output=True, text=True)
        return r.returncode, r.stderr

    def test_only_shell_files_are_judged(self):
        for p in ("x_decomp/pca.mojo", "bench/b.mojo", "experiments/e.mojo", "checks/c.mojo", "tools/t.py"):
            self.write(p, LOOPY)
        self.write("bindings/x.mojo", CLEAN)
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 0, err)
        self.assertIn("0 hits in 0 shell .mojo files", err)

    def test_shell_hits_fail_with_per_directory_counts(self):
        self.write("bindings/entry/x.mojo", LOOPY)
        self.write("x_decomp/pca_pyapi.mojo", "fn f():\n    var c = DeviceContext()\n")
        self.write("svm/svc_py.mojo", LOOPY)
        self.write("bindings/ok.mojo", "fn f(n: Int):\n    for i in range(n):  # small-loop(n argument words)\n")
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 1)
        self.assertIn("bindings/entry/x.mojo:2: untagged loop", err)
        self.assertIn("x_decomp/pca_pyapi.mojo:2: kernel launch", err)
        self.assertIn("svm/svc_py.mojo:2: untagged loop", err)
        self.assertNotIn("bindings/ok.mojo:", err)
        self.assertIn("per directory: bindings/entry 1, svm 1, x_decomp 1", err)

    def test_baseline_survives_line_shifts_and_prunes_stale(self):
        self.write("bindings/x.mojo", LOOPY)
        self.write("bindings/y.mojo", CLEAN)
        self.write(BASELINE, "# comment\nbindings/x.mojo:2: for i in range(n):\n"
                             "bindings/y.mojo:9: while True:\n")
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 0, err)
        self.assertIn("stale baseline row bindings/y.mojo: while True:", err)
        self.write("bindings/x.mojo", "# a new header line\n\n" + LOOPY)   # the loop moves to line 4
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 0, err)
        rc, err = self.run_checker("--prune-baseline")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.read(BASELINE), "# comment\nbindings/x.mojo:4: for i in range(n):\n")

    def test_a_second_identical_loop_is_new(self):
        self.write("bindings/x.mojo", LOOPY)
        self.write(BASELINE, "bindings/x.mojo:2: for i in range(n):\n")
        self.commit()
        self.write("bindings/x.mojo", LOOPY + LOOPY.replace("def f", "def g"))
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 1)
        self.assertIn("bindings/x.mojo:5: untagged loop", err)

    def test_branch_judges_only_its_changes_and_cannot_grow_baseline(self):
        self.write("bindings/x.mojo", LOOPY)        # main's debt, not in the baseline
        self.write("bindings/y.mojo", LOOPY)
        self.write("bindings/z.mojo", LOOPY)        # main's debt, untouched by the lane
        self.write(BASELINE, "bindings/y.mojo:2: for i in range(n):\n")
        self.commit("main")
        self.git("checkout", "-q", "-b", "lane/x")
        self.write("bindings/x.mojo", LOOPY + "\n# touched\n")   # already a hit at the merge-base
        self.write("bindings/y.mojo", LOOPY + "\n# touched\n")   # in the baseline
        self.commit("lane")
        rc, err = self.run_checker("--tree", "HEAD", "--branch")
        self.assertEqual(rc, 0, err)
        self.write("bindings/new_entry.mojo", LOOPY)
        self.write(BASELINE, "bindings/y.mojo:2: for i in range(n):\nbindings/new_entry.mojo:2: for i in range(n):\n")
        self.commit("grow")
        rc, err = self.run_checker("--tree", "HEAD", "--branch")
        self.assertEqual(rc, 1)
        self.assertIn("bindings/new_entry.mojo:2: untagged loop", err)
        self.assertIn("baseline row bindings/new_entry.mojo: for i in range(n): (x1) is not in main's baseline", err)
        self.assertNotIn("bindings/z.mojo:2", err)
        rc, err = self.run_checker("--tree", "HEAD")
        self.assertEqual(rc, 1)
        self.assertIn("bindings/x.mojo:2: untagged loop", err)
        self.assertIn("bindings/z.mojo:2: untagged loop", err)

    def test_usage(self):
        self.commit()
        self.assertEqual(self.run_checker("--bogus")[0], 2)
        self.assertEqual(self.run_checker("--tree", "no-such-ref")[0], 2)


if __name__ == "__main__":
    unittest.main()
