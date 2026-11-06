# SPDX-License-Identifier: Apache-2.0
# Exact-str label test shared by the label doors (lane perf3-label-doors,
# 2026-11-06). The native str-label reader (bindings/str_labels_native.mojo)
# takes any str subclass; the trees and GP encoders' ORDER RULE paths refuse
# or sort a subclass by its own hooks. A door hands a flat label list to that
# reader only when every label is exactly a str, so its classes, codes and
# refusals are the ORDER RULE's.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type


def labels_all_exact_str(flat: PythonObject) raises -> Bool:
    """True for a nonempty list whose every label is exactly a str, read by
    class pointer through the list's C storage (no label boxed into a Python
    call), stopping at the first other label."""
    var b = native_loaded_module("builtins")
    var n = len(flat)
    if n == 0 or py_type(flat) is not b.list:
        return False
    ref cpy = Python().cpython()
    var get_item = cpy.lib.get_function[PyObjectPtr]("PyList_GetItem")
    var str_type = b.str
    for i in range(n):  # driver-loop: exact str label classes by pointer, one C read per label
        var raw = get_item(flat._obj_ptr, i)
        if not raw:
            raise cpy.unsafe_get_error()
        var item_type = cpy.PyObject_Type(raw)
        var same = item_type == str_type._obj_ptr
        cpy.Py_DecRef(item_type)
        if not same:
            return False
    _ = flat  # keep the list alive through the borrowed item reads above
    _ = str_type
    return True
