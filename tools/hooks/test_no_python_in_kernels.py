#!/usr/bin/env python3
"""Tests for tools/hooks/no_python_in_kernels.py on a tiny temporary git tree.

Run: python3 tools/hooks/test_no_python_in_kernels.py
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

CHECKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "no_python_in_kernels.py")
BASELINE = "tools/hooks/python_execution_baseline.txt"
LEGACY_BASELINE = "tools/hooks/python_in_kernels_baseline.txt"
IMPORT = "from std.python import PythonObject\n\nfn f() -> Int:\n    return 1\n"
EVALUATE = 'from std.python import Python\ndef f():\n    return Python.evaluate("1 + 2")\n'
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

    def test_native_imports_and_capi_are_allowed_in_doors(self):
        native = IMPORT + 'def native(entry, args):\n    external_call["PyGILState_Ensure", Int]()\n    external_call["Py_INCREF", NoneType](entry)\n    external_call["PyObject_Call", Int](entry, args, 0)\n    for i in range(3):\n        external_call["PyDict_GetItemWithError", Int](entry, args)\n'
        for path in ("bindings/x.mojo", "providers/python/x.mojo", "providers/python/x_pyapi.mojo"):
            self.write(path, native)
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 0, err)
        self.assertIn("0 product .mojo files", err)
        self.assertIn("dynamic calls/hooks require target and type-admission review", err)

    def test_explicit_evaluation_rejected_in_provider_and_binding_too(self):
        for path in ("core/x.mojo", "bindings/x.mojo", "providers/python/x.mojo", "x/pca_pyapi.mojo"):
            self.write(path, EVALUATE)
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 1, err)
        for path in ("core/x.mojo", "bindings/x.mojo", "providers/python/x.mojo", "x/pca_pyapi.mojo"):
            self.assertIn(path + ":3: explicit Python evaluator call Python.evaluate", err)
        # core/ and x/ are kernel files: importing Python there is refused on its own
        self.assertIn("core/x.mojo:1: kernel file imports std.python", err)
        self.assertIn("x/pca_pyapi.mojo:1: kernel file imports std.python", err)
        self.assertNotIn("bindings/x.mojo:1: kernel file", err)
        self.assertNotIn("move the Python door", err)

    def test_kernel_files_import_and_name_no_python(self):
        cases = {
            "core/a.mojo": "from std.python import PythonObject\n",
            "autograd/b.mojo": "from python import Python\n",
            "gemm/c.mojo": "import std.python\n",
            "x_decomp/d.mojo": "from bindings.family_common import family_call\n",
            "svm/e.mojo": "from providers.python.x import y\n",
            "metrics/f.mojo": "fn f(o: PythonObject):\n    pass\n",
            "glm/g.mojo": "fn f():\n    var m = Python.import_module(\"os\")\n",
            "umap/h.mojo": "fn f(x: Int):\n    _ = py_call(x)\n",
            "adapters/i.mojo": "from std.python import PythonObject\n",
            "x/j_pyapi.mojo": "from std.python import PythonObject\n",
        }
        for path, text in cases.items():
            self.write(path, text)
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 1, err)
        for path in cases:
            self.assertRegex(err, path.replace(".", r"\.") + r":\d+: kernel file ")
        self.assertIn(f"{len(cases)} of {len(cases)} kernel/algorithm/core (K) .mojo files touch Python", err)

    def test_kernel_rule_ignores_strings_comments_local_helpers_and_doors(self):
        self.write("core/a.mojo", '"""Calls bindings/x.mojo; see Python docs."""\n# PythonObject in a comment\n'
                                  'fn py_repr(s: String) -> String:\n    return "Python." + s\n'
                                  'fn g(o: Int):\n    _ = o.py_call\n')
        self.write("bindings/b.mojo", IMPORT + "fn h(o: PythonObject):\n    _ = py_call(o)\n")
        self.write("providers/python/c.mojo", IMPORT)
        self.write("checks/d.mojo", IMPORT)
        self.write("gemm/tests/e.mojo", IMPORT)
        self.write("gemm/test_f.mojo", IMPORT)
        self.write("packaging/cases/g.mojo", IMPORT)
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 0, err)
        self.assertIn("0 of 1 kernel/algorithm/core (K) .mojo files touch Python", err)

    def test_kernel_rule_in_branch_mode_judges_only_new_touches(self):
        self.write("core/old.mojo", IMPORT)
        self.commit("main debt")
        self.git("checkout", "-q", "-b", "lane/k")
        self.write("core/old.mojo", IMPORT + "\n# touched\n")
        self.commit("touch")
        self.assertEqual(self.run_checker("--tree", "HEAD", "--branch")[0], 0)
        self.write("gemm/new.mojo", "from python import Python\n")
        self.commit("new")
        rc, err = self.run_checker("--tree", "HEAD", "--branch")
        self.assertEqual(rc, 1, err)
        self.assertIn("gemm/new.mojo:1: kernel file imports python", err)
        self.assertNotIn("core/old.mojo:1", err)

    def test_layer_table(self):
        for path, want in (("bindings/x.mojo", "D"), ("providers/python/x.mojo", "D"), ("providers/__init__.mojo", "K"),
                           ("adapters/x.mojo", "K"), ("core/x.mojo", "K"), ("x/y_pyapi.mojo", "K"),
                           ("training/checks/c.mojo", "T"), ("bench/b.mojo", "T"), ("packaging/p.mojo", "T"),
                           ("gemm/test_g.mojo", "T"), ("tests/t.mojo", "T")):
            self.assertEqual(_guard.layer(path), want, path)

    def test_test_fixtures_are_not_product(self):
        for path in ("bench/b.mojo", "experiments/e.mojo", "tests/x.mojo", "checks/c.mojo", "autograd/checks/c.mojo"):
            self.write(path, EVALUATE)
        self.commit()
        self.assertEqual(self.run_checker("--tree")[0], 0)

    def test_old_import_baseline_does_not_authorize_execution(self):
        self.write("bindings/x.mojo", EVALUATE)
        self.write(LEGACY_BASELINE, "# historical import debt\nbindings/x.mojo\n")
        self.commit()
        self.assertEqual(self.run_checker("--tree")[0], 1)
        with open(os.path.join(self.dir, LEGACY_BASELINE), encoding="utf-8") as source:
            self.assertEqual(source.read(), "# historical import debt\nbindings/x.mojo\n")

    def test_baseline_grandfathers_and_reports_stale(self):
        self.write("bindings/pca_door.mojo", EVALUATE)
        self.write("bindings/ica_door.mojo", CLEAN)
        self.write(BASELINE, "# comment\nbindings/pca_door.mojo\nbindings/ica_door.mojo\n")
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 0, err)
        self.assertIn("stale baseline row bindings/ica_door.mojo", err)
        rc, err = self.run_checker("--prune-baseline")
        self.assertEqual(rc, 0, err)
        with open(os.path.join(self.dir, BASELINE), encoding="utf-8") as f:
            self.assertEqual(f.read(), "# comment\nbindings/pca_door.mojo\n")

    def test_branch_judges_only_its_changes_and_cannot_grow_baseline(self):
        self.write("bindings/pca_door.mojo", EVALUATE)          # main's debt, not in the baseline
        self.write("bindings/ica_door.mojo", EVALUATE)
        self.write(BASELINE, "bindings/ica_door.mojo\n")
        self.commit("main")
        self.git("checkout", "-q", "-b", "lane/x")
        self.write("bindings/pca_door.mojo", EVALUATE + "\n# touched\n")   # already a hit at the merge-base
        self.write("bindings/ica_door.mojo", EVALUATE + "\n# touched\n")   # in the baseline
        self.commit("lane")
        rc, err = self.run_checker("--tree", "HEAD", "--branch")
        self.assertEqual(rc, 0, err)
        self.write("bindings/svm_new.mojo", EVALUATE)
        self.write(BASELINE, "bindings/ica_door.mojo\nbindings/svm_new.mojo\n")
        self.commit("grow")
        rc, err = self.run_checker("--tree", "HEAD", "--branch")
        self.assertEqual(rc, 1)
        self.assertIn("bindings/svm_new.mojo:3: explicit Python evaluator call", err)
        self.assertIn("baseline row bindings/svm_new.mojo is not in main's baseline", err)
        rc, err = self.run_checker("--tree", "HEAD")
        self.assertEqual(rc, 1)
        self.assertIn("bindings/pca_door.mojo:3: explicit Python evaluator call", err)

    def test_new_call_in_preexisting_file_is_not_grandfathered(self):
        self.write("bindings/existing.mojo", EVALUATE)
        self.commit("main debt")
        self.git("checkout", "-q", "-b", "lane/add-call")
        self.write("bindings/existing.mojo", EVALUATE + 'def extra():\n    return Python.evaluate("3+4")\n')
        self.commit("add second call")
        rc, err = self.run_checker("--tree", "HEAD", "--branch")
        self.assertEqual(rc, 1, err)
        self.assertIn("bindings/existing.mojo:5: explicit Python evaluator call", err)
        self.assertNotIn("bindings/existing.mojo:3: explicit Python evaluator call", err)

    def test_new_baseline_cannot_bootstrap_an_exception(self):
        self.write("bindings/x.mojo", CLEAN)
        self.commit("main without evaluator baseline")
        self.git("checkout", "-q", "-b", "lane/bootstrap")
        self.write("bindings/x.mojo", EVALUATE)
        self.write(BASELINE, "bindings/x.mojo\n")
        self.commit("unapproved exception")
        rc, err = self.run_checker("--tree", "HEAD", "--branch")
        self.assertEqual(rc, 1, err)
        self.assertIn("baseline row bindings/x.mojo is not in main's baseline", err)

    def test_usage(self):
        self.commit()
        self.assertEqual(self.run_checker("--bogus")[0], 2)
        self.assertEqual(self.run_checker("--tree", "no-such-ref")[0], 2)


_spec = importlib.util.spec_from_file_location("native_python_execution_guard", CHECKER)
_guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard)


class ScannerTest(unittest.TestCase):
    def test_import_aliases_and_direct_eval_symbols(self):
        cases = (
            ('from std.python import Python as P\nP.evaluate("x")\n', 'P.evaluate'),
            ('from std.python import (\n Python as P, PythonObject\n)\nP.evaluate("x")\n', 'P.evaluate'),
            ('import std.python as py\npy.Python.evaluate("x")\n', 'py.Python.evaluate'),
            ('from std.python import _cpython as cpy\ncpy.PyRun_SimpleString(code)\n', 'PyRun_SimpleString'),
            ('from std.python._cpython import PyEval_EvalCode as execute\nexecute(code, env, env)\n', 'PyEval_EvalCode'),
            ('from std.ffi import external_call as native\nnative[\n "PyRun_StringFlags", Int\n](code, mode, env, env, 0)\n', 'PyRun_StringFlags'),
            ('from std.python import _cpython as cpy\ncpy.lib.get_function[Int]("PyRun_String")(code)\n', 'PyRun_String'),
        )
        for source, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual([callee for _, callee in _guard.execution_findings(source)], [expected])

    def test_comments_strings_and_symbol_references_are_not_execution(self):
        source = '''from std.python import Python
from std.ffi import external_call
# Python.evaluate("comment")
var doc = "Python.evaluate(source) and PyRun_String"
var target = "PyRun_SimpleString"
var function = external_call["PyRun_SimpleString", Int]
'''
        self.assertEqual(_guard.execution_findings(source), [])

    def test_unrelated_evaluate_and_safe_native_calls_are_not_condemned(self):
        self.assertEqual(_guard.execution_findings('''from std.python import PythonObject
from std.python import _cpython as cpy
from std.ffi import external_call
model.evaluate(inputs)
cpy.PyObject_Call(admitted_native_entry, args, kwargs)
external_call["PyGILState_Ensure", Int]()
external_call["PyDict_GetItemWithError", Int](mapping, key)
'''), [])


if __name__ == "__main__":
    unittest.main()
