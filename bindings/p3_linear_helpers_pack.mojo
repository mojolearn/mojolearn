# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Native publication helpers for the linear-model owners (lane p3-linear).

`py_list_select` builds the object-label result list of a classifier's
predict in one PyList_New and direct slot stores: each slot is a new
reference to the stored class object, exactly what `list.append(labels[i])`
published, without a bound-method lookup and call per predicted row.
"""
from std.ffi import c_int
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from core.host_tree_impl import host_label_codes_in_range


def py_list_select(labels: PythonObject, indices: List[Int]) raises -> PythonObject:
    """[labels[indices[0]], labels[indices[1]], ...] as a new Python list.

    `labels` is read once per class (k item reads), never once per row; the
    indices must already be checked in range of `len(labels)`."""
    var k = len(labels)
    var table = List[PythonObject]()
    for c in range(k):  # driver-loop: over the fitted classes (model structure)
        table.append(labels[c])
    if not host_label_codes_in_range(indices, k):
        raise Error("predicted class index exceeds the fitted class table")
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyList_New")(len(indices))
    if not raw:
        raise cpy.unsafe_get_error()
    # NULL-filled until complete; the owner releases initialized slots on failure.
    var out = PythonObject(from_owned=raw)
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    for i in range(len(indices)):  # cpu-route: object-label output, one stored class reference per predicted row
        var item = table[indices[i]]._obj_ptr
        # SetItem steals a reference: hand it a new one to the stored class.
        cpy.Py_IncRef(item)
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    _ = table^
    return out
