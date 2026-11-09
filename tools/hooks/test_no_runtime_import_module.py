#!/usr/bin/env python3
"""Tests for tools/hooks/no_runtime_import_module.py on a tiny temporary git tree.

Run: python3 tools/hooks/test_no_runtime_import_module.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.join(HERE, "no_runtime_import_module.py")
BASELINE = "tools/hooks/runtime_import_module_baseline.txt"
sys.path.insert(0, HERE)
import no_runtime_import_module as nrim  # noqa: E402

IMPORTING = 'def f() raises -> PythonObject:\n    return Python.import_module("numpy")\n'
CLEAN = 'def f() raises -> PythonObject:\n    return native_loaded_module("builtins")\n'
ROW = 'return Python.import_module("numpy")'


class ScanTest(unittest.TestCase):
    def lines(self, text):
        return [n for n, _c in nrim.scan(text)]

    def test_calls_tags_initializers_comments(self):
        text = "\n".join([
            "def a() raises:",
            '    var np = Python.import_module("numpy")',                        # 2 hit
            '    var b = Python.import_module("builtins")  # import-time: global fill',  # 3 tagged
            "    # import-time: process state, filled once",
            '    var s = Python.import_module("sys")',                           # 5 tagged above
            '    # Python.import_module("os") in a comment',                     # 6 comment
            '    var t = native_loaded_module("posix")  # not Python.import_module(', # 7 trailing comment
            '    var u = "Python.import_module(x)"',                             # 8 string literal: still code
            "@export",
            'def PyInit_mod() abi("C") -> PythonObject:',
            "    try:",
            '        var m = Python.import_module("builtins")',                  # 12 inside PyInit_
            "    except e:",
            "        abort(String(e))",
            "def after() raises:",
            '    _ = Python . import_module ("array")',                          # 16 hit, spacing
        ])
        self.assertEqual(self.lines(text), [2, 8, 16])

    def test_nested_def_inside_pyinit_is_exempt_and_after_is_not(self):
        text = "\n".join([
            'def PyInit_x() abi("C") -> PythonObject:',
            "    def inner() raises:",
            '        _ = Python.import_module("a")',     # 3 inside PyInit_ (nested)
            "    return m",
            "def g() raises:",
            '    _ = Python.import_module("b")',         # 6 hit
        ])
        self.assertEqual(self.lines(text), [6])

    def test_docstrings_are_skipped(self):
        text = "\n".join([
            "def a() raises:",
            '    """Never call Python.import_module("numpy") here.',
            "",
            '    Python.import_module("x") in prose.',
            '    """',
            '    """One line: Python.import_module("y")."""',
            '    _ = Python.import_module("z")',
        ])
        self.assertEqual(self.lines(text), [7])


class ModuleReadKindsTest(unittest.TestCase):
    def test_kind_of(self):
        cases = {
            'var b = native_loaded_module("builtins")': None,
            'var o = native_loaded_module("operator").neg': None,
            'var m = native_loaded_module("mojolearn._mode")': "loaded_module",
            'var n = native_loaded_module("numpy")': "loaded_module",
            'var x = native_loaded_module(name)': "loaded_module",
            'var s = py_get(native_loaded_module("sys").modules, "numpy")': "sys_modules",
            'var s = py_get(call.module("sys").modules, name)': "sys_modules",
            'var v = native_loaded_module("sys").byteorder': None,
            'var f = py_fields(module)': "module_fields",
            'var f = py_fields(numpy_module)': "module_fields",
            'var f = py_fields(model)': None,
            'var c = py_fields(native_loaded_module("mojolearn._expansion_trees"))["_C"]': "loaded_module",
            'var p = Python.import_module("os")': "import_module",
        }
        for code, want in cases.items():
            self.assertEqual(nrim.kind_of(code), want, code)

    def test_scan_new_kinds_and_exemptions(self):
        text = "\n".join([
            "def a(name: String) raises:",                                   # 1 def line: declares
            '    var m = native_loaded_module("mojolearn._array")',             # 2 hit
            '    var c = py_fields(module)  # import-time: connect once',        # 3 tagged
            "    pass",                                                       # 4
            '    var s = native_loaded_module("sys").modules',                  # 5 hit
            "def native_loaded_module(name: String) raises -> PythonObject:",  # 5 def line
            "    return x",
        ])
        self.assertEqual([n for n, _c in nrim.scan(text)], [2, 5])
        self.assertEqual(nrim.scan('    _ = native_loaded_module(name)', "bindings/native_loaded_module.mojo"), [])
        self.assertEqual(len(nrim.scan('    _ = native_loaded_module(name)', "bindings/other.mojo")), 1)

    def test_parse_kinds(self):
        self.assertEqual(nrim.parse_kinds("# old\nbindings/a.mojo:1: x\n"), frozenset({"import_module"}))
        self.assertEqual(nrim.parse_kinds("# kinds: import_module loaded_module\n"),
                         frozenset({"import_module", "loaded_module"}))


class TreeTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="nrim-")
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

    def test_only_bindings_and_providers_are_judged(self):
        for p in ("core/a.mojo", "bench/b.mojo", "tools/t.mojo", "x_decomp/pca.mojo", "bindings/notes.py"):
            self.write(p, IMPORTING)
        self.write("bindings/x.mojo", CLEAN)
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 0, err)
        self.assertIn("0 runtime module reads (import_module 0, loaded_module 0, sys_modules 0, module_fields 0) in 0 of 1 bindings/providers .mojo files", err)

    def test_hits_fail_with_per_directory_counts(self):
        self.write("bindings/entry/x.mojo", IMPORTING)
        self.write("providers/python/y_pyapi.mojo", IMPORTING)
        self.write("bindings/ok.mojo", 'def f() raises:\n    _ = Python.import_module("a")  # import-time: once\n')
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 1)
        self.assertIn("bindings/entry/x.mojo:2: runtime module read (import_module)", err)
        self.assertIn("providers/python/y_pyapi.mojo:2: runtime module read (import_module)", err)
        self.assertNotIn("bindings/ok.mojo:", err)
        self.assertIn("per directory: bindings/entry 1, providers/python 1", err)

    def test_baseline_survives_line_shifts_and_prunes_stale(self):
        self.write("bindings/x.mojo", IMPORTING)
        self.write("bindings/y.mojo", CLEAN)
        self.write(BASELINE, "# comment\nbindings/x.mojo:2: " + ROW + "\n"
                             'bindings/y.mojo:9: var s = Python.import_module("sys")\n')
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 0, err)
        self.assertIn('stale baseline row bindings/y.mojo: var s = Python.import_module("sys")', err)
        self.write("bindings/x.mojo", "# a new header line\n\n" + IMPORTING)   # the call moves to line 4
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 0, err)
        rc, err = self.run_checker("--prune-baseline")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.read(BASELINE), "# comment\nbindings/x.mojo:4: " + ROW + "\n")

    def test_a_second_identical_call_is_new(self):
        self.write("bindings/x.mojo", IMPORTING)
        self.write(BASELINE, "bindings/x.mojo:2: " + ROW + "\n")
        self.commit()
        self.write("bindings/x.mojo", IMPORTING + IMPORTING.replace("def f", "def g"))
        self.commit()
        rc, err = self.run_checker("--tree")
        self.assertEqual(rc, 1)
        self.assertIn("bindings/x.mojo:4: runtime module read (import_module)", err)

    def test_branch_judges_only_its_changes_and_cannot_grow_baseline(self):
        self.write("bindings/x.mojo", IMPORTING)        # main's debt, not in the baseline
        self.write("bindings/y.mojo", IMPORTING)
        self.write("providers/z.mojo", IMPORTING)       # main's debt, untouched by the lane
        self.write(BASELINE, "bindings/y.mojo:2: " + ROW + "\n")
        self.commit("main")
        self.git("checkout", "-q", "-b", "lane/x")
        self.write("bindings/x.mojo", IMPORTING + "\n# touched\n")   # already a hit at the merge-base
        self.write("bindings/y.mojo", IMPORTING + "\n# touched\n")   # in the baseline
        self.commit("lane")
        rc, err = self.run_checker("--tree", "HEAD", "--branch")
        self.assertEqual(rc, 0, err)
        self.write("bindings/new_entry.mojo", IMPORTING)
        self.write(BASELINE, "bindings/y.mojo:2: " + ROW + "\nbindings/new_entry.mojo:2: " + ROW + "\n")
        self.commit("grow")
        rc, err = self.run_checker("--tree", "HEAD", "--branch")
        self.assertEqual(rc, 1)
        self.assertIn("bindings/new_entry.mojo:2: runtime module read (import_module)", err)
        self.assertIn("baseline row bindings/new_entry.mojo: " + ROW + " (x1) is not in main's baseline", err)
        self.assertNotIn("providers/z.mojo:2", err)
        rc, err = self.run_checker("--tree", "HEAD")
        self.assertEqual(rc, 1)
        self.assertIn("bindings/x.mojo:2: runtime module read (import_module)", err)
        self.assertIn("providers/z.mojo:2: runtime module read (import_module)", err)

    def test_new_kind_bootstraps_once_then_only_shrinks(self):
        loaded = 'def f() raises -> PythonObject:\n    return native_loaded_module("mojolearn._mode")\n'
        lrow = 'return native_loaded_module("mojolearn._mode")'
        self.write("bindings/a.mojo", loaded)
        self.write(BASELINE, "# old header without kinds\n")
        self.commit("main before the kind existed")
        self.assertEqual(self.run_checker("--tree", "HEAD")[0], 1)
        self.git("checkout", "-q", "-b", "lane/kinds")
        self.write(BASELINE, "# kinds: import_module loaded_module sys_modules module_fields\nbindings/a.mojo:2: " + lrow + "\n")
        self.commit("bootstrap")
        rc, err = self.run_checker("--tree", "HEAD")
        self.assertEqual(rc, 0, err)                    # bootstrapped: main does not cover the kind yet
        self.git("checkout", "-q", "main")
        self.git("merge", "-q", "--ff-only", "lane/kinds")
        self.git("checkout", "-q", "-b", "lane/grow")
        self.write("bindings/b.mojo", loaded)
        self.write(BASELINE, "# kinds: import_module loaded_module sys_modules module_fields\nbindings/a.mojo:2: " + lrow +
                   "\nbindings/b.mojo:2: " + lrow + "\n")
        self.commit("grow")
        rc, err = self.run_checker("--tree", "HEAD")
        self.assertEqual(rc, 1, err)                    # main covers the kind now: no new rows
        self.assertIn("baseline row bindings/b.mojo: " + lrow + " (x1) is not in main's baseline", err)
        self.assertIn("bindings/b.mojo:2: runtime module read (loaded_module)", err)

    def test_write_baseline(self):
        self.write("bindings/a.mojo", IMPORTING)
        self.write(BASELINE, "# header kept\n")
        self.commit()
        self.assertEqual(self.run_checker("--write-baseline")[0], 0)
        with open(os.path.join(self.dir, BASELINE), encoding="utf-8") as f:
            text = f.read()
        self.assertIn("# header kept\n# kinds: import_module loaded_module sys_modules module_fields\n", text)
        self.assertIn("bindings/a.mojo:2: " + ROW, text)

    def test_usage(self):
        self.commit()
        self.assertEqual(self.run_checker("--bogus")[0], 2)
        self.assertEqual(self.run_checker("--tree", "no-such-ref")[0], 2)


if __name__ == "__main__":
    unittest.main()
