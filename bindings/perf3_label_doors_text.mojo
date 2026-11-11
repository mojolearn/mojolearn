# SPDX-License-Identifier: Apache-2.0
# Exact-str label test shared by the label doors (lane perf3-label-doors,
# 2026-11-06). The native str-label reader (bindings/str_labels_native.mojo)
# takes any str subclass; the trees and GP encoders' ORDER RULE paths refuse
# or sort a subclass by its own hooks. A door hands a flat label list to that
# reader only when every label is exactly a str, so its classes, codes and
# refusals are the ORDER RULE's.
from std.python import Python, PythonObject
from bindings.label_sequence_source import label_source_count, label_source_raw
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type
from bindings.p3_labels_metrics_native import p3_item, p3_type_word


def labels_all_exact_str(flat: PythonObject) raises -> Bool:
    var pins = List[PythonObject]()
    return _labels_all_exact_str_source[False](flat, pins)


def labels_all_exact_str(flat: List[PythonObject]) raises -> Bool:
    return _labels_all_exact_str_source[True](PythonObject(None), flat)


def _labels_all_exact_str_source[native: Bool](flat: PythonObject, pins: List[PythonObject]) raises -> Bool:
    """True for a nonempty list whose every label is exactly a str, read by
    class pointer through the list's C storage (no label boxed into a Python
    call), stopping at the first other label."""
    var b = native_loaded_module("builtins")
    var n = label_source_count[native](flat, pins)
    if n == 0:
        return False
    comptime if not native:
        if py_type(flat) is not b.list:
            return False
    var str_word = Int(b.str._obj_ptr)
    var str_type = b.str
    for i in range(n):  # driver-loop: exact str label classes by pointer, one C read per label
        # p3-labels-metrics: PyList_GetItem from the process-wide table (it
        # was resolved by name, dlsym, on every call).
        if p3_type_word(label_source_raw[native](flat, pins, i)) != str_word:
            return False
    _ = pins
    _ = flat  # keep the list alive through the borrowed item reads above
    _ = str_type
    return True
