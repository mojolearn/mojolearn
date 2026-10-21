#!/usr/bin/env python3
"""Tests for tools/hooks/no_python_in_kernels.py on a tiny temporary git tree.

Run: python3 tools/hooks/test_no_python_in_kernels.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

CHECKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "no_python_in_kernels.py")
BASELINE = "tools/hooks/python_in_kernels_baseline.txt"
IMPORT = "from std.python import PythonObject\n\nfn f() -> Int:\n    return 1\n"
CLEAN = "from std.math import sqrt\n\nfn f() -> Int:\n    return 1\n"


class TreeTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="npik-")
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

    def commit(self, msg="c"):
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", msg)

    def run_checker(self, *args):
        r = subprocess.run([sys.executable, CHECKER, *args], cwd=self.dir, capture_output=True, text=True)
        return r.returncode, r.stderr

    def test_shell_files_are_allowed(self):
        for p in ("bindings/x.mojo", "x_decomp/pca_pyapi.mojo", "svm/svc_py.mojo", "bench/b.mojo",
                  "experiments/e.mojo", "checks/c.mojo", "autograd/checks/c.mojo", "x_decomp/pyapi_util.mojo"):
            self.write(p, IMPORT)
        self.write("x_decomp/pca.mojo", CLEAN)
        self.write("tools/notes.py", "import python\n")
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 0, err)
        self.assertIn("0 kernel .mojo files", err)

    def test_kernel_hit_fails(self):
        self.write("x_decomp/pca.mojo", IMPORT)
        self.write("svm/svc.mojo", 'fn f():\n    external_call["PyGILState_Ensure", Int]()\n')
        self.write("core/k.mojo", "    import python\n")
        self.write("core/ok.mojo", "# from python import x is a comment\nfn f():\n    pass\n")
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 1)
        for p in ("x_decomp/pca.mojo", "svm/svc.mojo", "core/k.mojo"):
            self.assertIn(p + ": kernel .mojo file", err)
        self.assertNotIn("core/ok.mojo", err)

    def test_baseline_grandfathers_and_reports_stale(self):
        self.write("x_decomp/pca.mojo", IMPORT)
        self.write("x_decomp/ica.mojo", CLEAN)
        self.write(BASELINE, "# comment\nx_decomp/pca.mojo\nx_decomp/ica.mojo\n")
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 0, err)
        self.assertIn("stale baseline row x_decomp/ica.mojo", err)
        rc, err = self.run_checker("--prune-baseline")
        self.assertEqual(rc, 0, err)
        with open(os.path.join(self.dir, BASELINE), encoding="utf-8") as f:
            self.assertEqual(f.read(), "# comment\nx_decomp/pca.mojo\n")

    def test_branch_judges_only_its_changes_and_cannot_grow_baseline(self):
        self.write("x_decomp/pca.mojo", IMPORT)          # main's debt, not in the baseline
        self.write("x_decomp/ica.mojo", IMPORT)
        self.write(BASELINE, "x_decomp/ica.mojo\n")
        self.commit("main")
        self.git("checkout", "-q", "-b", "lane/x")
        self.write("x_decomp/pca.mojo", IMPORT + "\n# touched\n")   # already a hit at the merge-base
        self.write("x_decomp/ica.mojo", IMPORT + "\n# touched\n")   # in the baseline
        self.commit("lane")
        rc, err = self.run_checker("--tree", "HEAD", "--branch")
        self.assertEqual(rc, 0, err)
        self.write("svm/new.mojo", IMPORT)
        self.write(BASELINE, "x_decomp/ica.mojo\nsvm/new.mojo\n")
        self.commit("grow")
        rc, err = self.run_checker("--tree", "HEAD", "--branch")
        self.assertEqual(rc, 1)
        self.assertIn("svm/new.mojo: kernel .mojo file", err)
        self.assertIn("baseline row svm/new.mojo is not in main's baseline", err)
        rc, err = self.run_checker("--tree", "HEAD")
        self.assertEqual(rc, 1)
        self.assertIn("x_decomp/pca.mojo: kernel .mojo file", err)

    def test_usage(self):
        self.commit()
        self.assertEqual(self.run_checker("--bogus")[0], 2)
        self.assertEqual(self.run_checker("--tree", "no-such-ref")[0], 2)


if __name__ == "__main__":
    unittest.main()
