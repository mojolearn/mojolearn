# SPDX-License-Identifier: Apache-2.0
# Borrowed reads of exact list / tuple items and class words through the
# process-wide CPython table: the two primitives every native reader of a
# Python container uses (bindings/p3_labels_metrics_native.mojo re-exports
# them as p3_type_word / p3_item). No loops here.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table


def borrowed_type_word(raw: PyObjectPtr) raises -> Int:
    """The class pointer of a live object as a word (the reference
    PyObject_Type returns is dropped at once; the object keeps its class
    alive)."""
    ref cpy = Python().cpython()
    var cls = cpy.PyObject_Type(raw)
    var word = Int(cls)
    cpy.Py_DecRef(cls)
    return word
