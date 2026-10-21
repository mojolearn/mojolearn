#!/usr/bin/env python3
"""Tests for tools/kernel_closure_check.py: the static pass and the stripped tree (no Mojo, no compile).

Run: python3 tools/test_kernel_closure_check.py
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel_closure_check as kcc  # noqa: E402


class StripTest(unittest.TestCase):
    def test_strip_keeps_line_numbers_and_blocks(self):
        text = "\n".join([
            "from std.python import PythonObject",
            "from std.python import (",
            "    Python,",
            "    PythonObject,",
            ")",
            "from std.math import sqrt",
            "fn f():",
            "    from python import Python",
            "    return",
        ])
        body, n = kcc.strip_python(text)
        self.assertEqual(n, 3)
        lines = body.split("\n")
        self.assertEqual(len(lines), 9)
        self.assertEqual(lines[:5], ["", "", "", "", ""])
        self.assertEqual(lines[5], "from std.math import sqrt")
        self.assertTrue(lines[7].startswith("    pass"))


class PlanTest(unittest.TestCase):
    def test_plan_classifies_and_finds_the_door(self):
        texts = {
            "core/a.mojo": "from core.b import g\nfn f():\n    pass\n",
            "core/b.mojo": "from bindings.hostptr import p\nfn g():\n    pass\n",
            "core/c.mojo": "from x.thing_pyapi import h\n",
            "core/prog.mojo": "def main():\n    pass\n",
            "core/d.mojo": "from .b import g\nimport core.a as A\n",
            "core/bad-name.mojo": "fn f():\n    pass\n",
            "bindings/hostptr.mojo": "fn p():\n    pass\n",
            "bindings/entry.mojo": "from std.python import PythonObject\nfrom core.a import f\n",
            "x/thing_pyapi.mojo": "from std.python import PythonObject\n",
            "checks/t.mojo": "from std.python import Python\n",
            "bench/b.mojo": "fn f():\n    pass\n",
        }
        pl = kcc.plan(texts)
        self.assertEqual(pl["kernels"], ["core/a.mojo", "core/b.mojo", "core/bad-name.mojo", "core/c.mojo",
                                         "core/d.mojo", "core/prog.mojo"])
        self.assertEqual([n for _p, n in pl["modules"]], ["core.a", "core.b", "core.c", "core.d"])
        self.assertEqual(pl["mains"], ["core/prog.mojo"])
        self.assertEqual([s["path"] for s in pl["skipped"]], ["core/bad-name.mojo"])
        self.assertEqual(pl["door"], [{"kernel": "core/c.mojo", "imports": "x/thing_pyapi.mojo", "kind": "shell"}])
        self.assertEqual(pl["shell_helpers"], [{"kernel": "core/b.mojo", "imports": "bindings/hostptr.mojo"}])
        self.assertEqual(kcc.imports_of("core/d.mojo", texts["core/d.mojo"], set(texts)), {"core/b.mojo", "core/a.mojo"})
        prog = kcc.program_text("0" * 40, pl["modules"])
        self.assertIn("import core.a as _k0000\n", prog)
        self.assertIn("def main():", prog)

        root = Path(tempfile.mkdtemp(prefix="kcc-"))
        try:
            stripped, deleted, _notes = kcc.write_tree(root / "tree", texts, root, pl["reach"])
            self.assertEqual(deleted, ["bindings/entry.mojo", "x/thing_pyapi.mojo"])
            self.assertTrue((root / "tree/bindings/hostptr.mojo").exists())
            self.assertEqual(stripped, {"checks/t.mojo": 1})
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
