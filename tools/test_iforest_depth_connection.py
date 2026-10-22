# SPDX-License-Identifier: Apache-2.0
"""Import-only depth metadata connection on supported CPython versions."""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class DepthConnectionTests(unittest.TestCase):
    def connect(self):
        source = ROOT / "python/mojolearn/_mode.py"
        tree = ast.parse(source.read_text())
        node = next(item for item in tree.body
                    if isinstance(item, ast.FunctionDef) and item.name == "_track_fit")
        mode = types.ModuleType("_depth_test._mode")
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), vars(mode))
        package = types.ModuleType("_depth_test")
        package._mode = mode
        spec = importlib.util.spec_from_file_location(
            "_depth_test._native_iforest_depth", ROOT / "python/mojolearn/_native_iforest_depth.py")
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        sentinel = object()
        context = {"modules": sys.modules, "warning_fields": sentinel}
        with patch.dict(sys.modules, {"_depth_test": package, mode.__name__: mode}):
            bridge.connect(context)
        return context, source, sentinel

    def test_warning_locations_match_actual_wrapper_arithmetic(self):
        context, source, sentinel = self.connect()
        lines = source.read_text().splitlines()
        expected = {"add": 'state.get("_mojolearn_fitting", 0) + 1',
                    "subtract": 'state.get("_mojolearn_fitting", 1) - 1',
                    "greater": 'if depth > 0:'}
        settings = context["host_depth_warning_contexts"]
        self.assertEqual(set(settings), set(expected))
        for operation, expression in expected.items():
            warning = settings[operation]["sequence_array_warnings"]
            self.assertIn(expression, lines[warning["warning_lineno"] - 1])
            self.assertEqual(warning["warning_source_line"], lines[warning["warning_lineno"] - 1].strip())
            self.assertIs(warning["warning_fields"], sentinel)
            self.assertIs(settings[operation]["module_type"], types.ModuleType)
        self.assertIsNot(settings["add"]["sequence_array_warnings"]["warning_seen"],
                         settings["subtract"]["sequence_array_warnings"]["warning_seen"])

    def test_fixed_primitive_identity_metadata_contains_no_numpy_dependency(self):
        context, _, _ = self.connect()
        self.assertEqual(context["host_depth_primitive_types"],
                         (type(None), bool, int, float, complex, str, bytes, bytearray))
        self.assertIs(context["host_depth_builtin_types"]["list"], list)
        self.assertNotIn("numpy", context)


if __name__ == "__main__":
    unittest.main()
