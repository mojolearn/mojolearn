#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/mojo_door_sites.py on small synthetic door files (source only).

Run: python3 -u tools/tests/test_mojo_door_sites.py
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mojo_door_sites as mds  # noqa: E402


def classes(code, names=(), tables=None):
    return [c for c, _ in mds.classify_line(code, mds.Names(names), tables)]


class ClassifyLineTest(unittest.TestCase):
    def test_object_layer_and_builtins(self):
        self.assertEqual(classes('var n = Int(py=x.shape[0])', ("x",)), ["a"])
        self.assertEqual(classes('var l = py_call(b.list, x)'), ["a"])
        self.assertEqual(classes('var l = py_call(_b().object.__new__, cls)'), ["a"])
        self.assertEqual(classes('var l = py_call(p[].bi.dict)'), ["a"])
        self.assertEqual(classes('_ = py_call(ints.extend, nodes)'), ["a"])
        self.assertEqual(classes('var m = native_loaded_module("operator").neg'), ["a"])
        self.assertEqual(classes('var t = py_call(py_type(Python.list()), x)'), ["a"])
        self.assertEqual(classes('var k = 3 + 4'), [])

    def test_compiled_entries(self):
        self.assertEqual(classes('var r = py_call(entries["seq_alloc"], x)'), ["a-native"])
        self.assertEqual(classes('var r = py_call(self._entry(_K, "x"), y)'), ["a-native"])
        self.assertEqual(classes('var r = py_call(_native_function(mods, "f"), y)'), ["a-native"])

    def test_calls_into_python_code(self):
        self.assertEqual(classes('var np = Python.import_module("numpy")'), ["b"])
        self.assertEqual(classes('var m = native_loaded_module("mojolearn._mode")'), ["b"])
        self.assertEqual(classes('var x = native_loaded_module(name)'), ["b"])
        self.assertEqual(classes('var a = py_call(context["ndarray"], shape)'), ["b"])
        self.assertEqual(classes('var c = py_call(init, x, k)'), ["b"])
        self.assertEqual(classes('var r = model.fit(x)', ("model",)), ["b"])
        self.assertEqual(classes('var r = model.append(x)', ("model",)), ["a"])   # C-type method: object layer
        self.assertIn("b", classes('var y = np.asarray(x)'))

    def test_table_entries(self):
        self.assertEqual(classes('var g = py_call(context["fields_get"], v)'), ["b-table"])
        self.assertEqual(classes('var g = py_call(context["prepare"], v)', tables={"prepare": "m.py:3 prepare"}), ["b"])

    def test_inline_iteration(self):
        self.assertIn("c-inline", classes('var s = py_call(b.sorted, labels)'))
        self.assertIn("c-inline", classes('var l = py_call(arr.tolist)', ("arr",)))
        self.assertIn("c-inline", classes('var it = cpy.PyObject_GetIter(x)'))

    def test_strings_are_not_code(self):
        self.assertEqual(classes('raise Error("pass a numpy array, not np.matrix(...)")'), [])


DOOR = '''from bindings.helper import admit, unrelated
from gemm.kernels import gemm_kernel


def entry_fit(model: PythonObject, operation: String, args: PythonObject) raises -> PythonObject:
    var x = admit(args)
    if operation == "family_fit":
        for i in range(len(x)):  # driver-loop: over rows
            _ = py_call(b.float, x[i])
        return x
    if operation == "family_predict":
        _ = py_call(model_hook, x)
        return x
    gemm_kernel(x)
    return x


def build_mod() -> PythonObject:
    var m = PythonModuleBuilder("_mod")
    m.def_function[entry_fit]("entry_fit")
    m.def_py_c_function(entry_c, "entry_c_call")
    return m.finalize()
'''
HELPER = '''def admit(args: PythonObject) raises -> PythonObject:
    for k in range(3):  # small-loop(3): the argument slots
        _ = py_get(args, k)
    return args


def unrelated(x: PythonObject) raises:
    _ = py_call(user_callback, x)
'''


class IndexTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        (root / "python" / "mojolearn").mkdir(parents=True)
        self.index = mds.DoorIndex(root, files={"bindings/door.mojo": DOOR, "bindings/helper.mojo": HELPER},
                                   workers=1, python_tables={})

    def tearDown(self):
        self.dir.cleanup()

    def test_registry(self):
        self.assertTrue(self.index.is_compiled_entry("entry_fit"))
        self.assertFalse(self.index.is_compiled_entry("admit"))
        self.assertIn("entry_c_call", self.index.unregistered)   # entry_c has no body in the index

    def test_key_filtered_reach(self):
        keys = frozenset({"family_fit", "family_predict"})
        fit = self.index.reach("entry_fit", "family_fit", keys)
        self.assertEqual(fit.level(), "glue-heavy")              # data-sized driver loop over rows
        self.assertEqual(fit.counts()["b"], 0)
        self.assertIn(("bindings/helper.mojo", "admit"), fit.functions)
        self.assertNotIn(("bindings/helper.mojo", "unrelated"), fit.functions)
        predict = self.index.reach("entry_fit", "family_predict", keys)
        self.assertEqual(predict.level(), "python-on-path")      # calls an unresolved Python callable
        self.assertEqual(predict.counts()["c-loops-data"], 0)
        whole = self.index.reach("entry_fit", None, keys)
        self.assertEqual(whole.level(), "python-on-path")
        self.assertIsNone(self.index.reach("not_an_entry", None, keys))

    def test_loops_and_tags(self):
        admit = self.index.funcs[("bindings/helper.mojo", "admit")]
        self.assertEqual([(t, d) for _, t, _, d, _ in admit.loops], [("small", False)])
        fit = self.index.funcs[("bindings/door.mojo", "entry_fit")]
        self.assertEqual([(t, d) for _, t, _, d, _ in fit.loops], [("driver", True)])

    def test_layers(self):
        self.assertTrue(mds.is_door("bindings/x.mojo"))
        self.assertTrue(mds.is_door("providers/python/x_pyapi.mojo"))
        self.assertFalse(mds.is_door("core/x.mojo"))
        self.assertFalse(mds.is_door("bindings/tests/x.mojo"))


if __name__ == "__main__":
    unittest.main()
