# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The float-list target fast path of the neighbor and hierarchy doors
(p3-neighbors-cluster, 2026-11-06). An exact flat list or tuple of exact
floats (`labels_list_scalar_kind` == LABELS_LIST_FLOAT) took the label walk:
four or five Python-object calls per label. Its every item is read here
borrowed from the container's C storage with one `PyFloat_AsDouble`, and the
walk's tests run natively (core/p3nc_scan.mojo), in the same order with the
same verdicts."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type
from bindings.label_input_sources import labels_list_scalar_kind, LABELS_LIST_FLOAT
from core.p3nc_scan import p3nc_real_labels

comptime P3NC_LIST_NOT_TAKEN = -1


def p3nc_float_list_verdict(source: PythonObject, classifier: Bool) raises -> Int:
    """P3NC_LIST_NOT_TAKEN unless `source` is an exact flat list or tuple of
    exact floats; else `p3nc_real_labels`' verdict on its items."""
    if labels_list_scalar_kind(source) != LABELS_LIST_FLOAT:
        return P3NC_LIST_NOT_TAKEN
    var is_tuple = py_type(source) is native_loaded_module("builtins").tuple
    var n = len(source)
    var values = List[Float64](length=n, fill=0.0)
    ref cpy = Python().cpython()
    var get_item = cpy.lib.get_function[PyObjectPtr]("PyTuple_GetItem" if is_tuple else "PyList_GetItem")
    var as_double = cpy.lib.get_function[Float64]("PyFloat_AsDouble")
    for i in range(n):  # driver-loop: exact float items read borrowed, one C read per label
        var raw = get_item(source._obj_ptr, i)
        if not raw:
            raise cpy.unsafe_get_error()
        values[i] = as_double(raw)
    _ = source  # keep the container alive through the borrowed item reads above
    return p3nc_real_labels(values, classifier)
