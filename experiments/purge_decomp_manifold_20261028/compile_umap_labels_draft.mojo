# SPDX-License-Identifier: Apache-2.0
# Compile-only root for the unwired native UMAP categorical label draft
# (bindings/umap_labels_native.mojo). Not called by any product path.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from bindings.umap_labels_native import umap_categorical


@export
def purge_umap_labels_draft_root(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args)
        var result = umap_categorical(call[0], Int(py=call[1]), call[2])
        return result[0].steal_data()
    except error:
        return raise_python_exception(error)
