# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""bindings/cpython_layout.mojo against live CPython objects (ctypes).

Run on each CI Python, this proves that interpreter's row of the table; the
`installed` case proves every python3.10 .. python3.14 on PATH at once (one
subprocess per interpreter, in parallel). The sabotage case shows the check
fails on the offsets that shipped wrong before 15cca53af1 (176 / 392).
"""
import importlib.util
import sys
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("cpython_layout_check", Path(__file__).resolve().parents[1] / "cpython_layout_check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class CPythonLayoutTest(unittest.TestCase):
    def test_table_parses(self):
        offsets, rows = check.parse_table()
        self.assertEqual(set(offsets), {"DICT_MA_USED", "DICT_MA_VERSION_TAG", "TYPE_TP_FLAGS", "TYPE_TP_VERSION_TAG",
                                        "ARRAY_OB_SIZE", "ARRAY_OB_ITEM", "ARRAY_ALLOCATED"})
        # every shipped minor version (python/pyproject.toml classifiers) has a row
        self.assertTrue({10, 11, 12, 13, 14} <= set(rows))

    def test_running_interpreter_row(self):
        offsets, rows = check.parse_table()
        lines, good = check.verdicts(check.measure(sys.executable, offsets), rows)
        self.assertTrue(good, "\n".join("\t".join(line) for line in lines))

    def test_installed_interpreters_rows(self):
        pythons = check.installed()
        if not pythons:
            self.skipTest("no python3.10 .. python3.14 on PATH")
        self.assertEqual(check.main(["--installed"]), 0)

    def test_old_wrong_offsets_fail(self):
        offsets, _ = check.parse_table()
        offsets = dict(offsets, TYPE_TP_FLAGS=176, TYPE_TP_VERSION_TAG=392)
        result = check.measure(sys.executable, offsets)
        forced = {result["minor"]: {"type"}}
        lines, good = check.verdicts(result, forced)
        self.assertFalse(good)
        self.assertIn("WRONG", [line[4] for line in lines if line[1] == "type.tp_flags"])


if __name__ == "__main__":
    unittest.main()
