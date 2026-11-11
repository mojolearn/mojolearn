# SPDX-License-Identifier: Apache-2.0
"""Exact primitive scalar UNPACK into native words; no Python array staging."""
from std.memory import bitcast
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type
from bindings.cpython_table import cpython_table
from bindings.p3_labels_metrics_native import p3_item, p3_type_word
from bindings.label_input_sources import LABELS_LIST_INT, LABELS_LIST_FLOAT, LABELS_LIST_BOOL
from core.gp_native_label_input import GPLabelInput


def gp_label_input(value: PythonObject, kind: Int) raises -> Optional[GPLabelInput]:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    var tuple_input = cls is b.tuple
    if cls is not b.list and not tuple_input:
        return None
    if kind != LABELS_LIST_INT and kind != LABELS_LIST_FLOAT and kind != LABELS_LIST_BOOL:
        return None
    var expected_type = b.float if kind == LABELS_LIST_FLOAT else b.bool if kind == LABELS_LIST_BOOL else b.int
    var expected = Int(expected_type._obj_ptr)
    ref cpy = Python().cpython()
    var api = cpython_table()
    var n = len(value)
    if n == 0:
        return None
    var captured = GPLabelInput(n, kind == LABELS_LIST_FLOAT, kind == LABELS_LIST_BOOL)
    var overflow = List[Int32](length=1, fill=0)
    # This is the explicit scalar input boundary, matching array('q'/'d')'s
    # one scalar-slot conversion per admitted item. No arithmetic, sorting or
    # independent numerical kernel work occurs here; CPython reads need GIL.
    for i in range(n):  # driver-loop: UNPACK exact scalar storage through public C slots
        var raw = p3_item(value, i, tuple_input)
        if p3_type_word(raw) != expected:
            return None
        if captured.wide:
            var word = api[].PyFloat_AsDouble(raw)
            if word == -1.0 and api[].PyErr_Occurred():
                raise cpy.unsafe_get_error()
            captured.values[i] = bitcast[DType.uint64](word)
        else:
            overflow[0] = 0
            var word = api[].PyLong_AsLongLongAndOverflow(raw, Int(overflow.unsafe_ptr()))
            if overflow[0] != 0:
                return None  # same arbitrary-size-int path as the former array conversion refusal
            if word == -1 and api[].PyErr_Occurred():
                raise cpy.unsafe_get_error()
            captured.values[i] = bitcast[DType.uint64](word)
    _ = overflow
    _ = expected_type
    _ = value
    return captured^
