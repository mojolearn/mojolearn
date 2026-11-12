# SPDX-License-Identifier: Apache-2.0
# Borrowed reads of exact list / tuple items and class words through the
# process-wide CPython table: the two primitives every native reader of a
# Python container uses (bindings/p3_labels_metrics_native.mojo re-exports
# them as p3_type_word / p3_item). No loops here.
# The PACK half of the former `bindings/cpython_borrowed.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/cpython_borrowed_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table


def borrowed_item(container: PythonObject, index: Int, is_tuple: Bool) raises -> PyObjectPtr:
    """A borrowed item of an exact list or tuple (the caller holds the
    container and runs no Python code while it uses the item)."""
    ref cpy = Python().cpython()
    var raw = cpython_table()[].PyTuple_GetItem(container._obj_ptr, index) if is_tuple else cpython_table()[].PyList_GetItem(container._obj_ptr, index)
    if not raw:
        raise cpy.unsafe_get_error()
    return raw
