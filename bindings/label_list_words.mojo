# SPDX-License-Identifier: Apache-2.0
"""The UNPACK of an exact list or tuple of exact ints, bools or floats into
one native word buffer (lane i3-move-data-loops, 2026-11-09): one
`array.array(typecode, values)` construction, CPython's own C conversion of
every item (`PyLong_AsLongLong` for 'q', `PyFloat_AsDouble` for 'd', the C
reads the per-label loops of lanes P2/P3 made), then the buffer's address.
The words are read and coded in core/label_words_impl.mojo. No loop here.

The caller has already checked that every item is an exact builtin int,
bool or float (`labels_list_scalar_kind`), so no item's Python method runs.
"""
from std.python import PythonObject
from bindings.py_balanced import py_call
from bindings.native_loaded_module import native_loaded_module


def label_list_store(values: PythonObject, typecode: String) raises -> PythonObject:
    """`array.array(typecode, values)`, or None when an item does not convert
    (an int outside int64 for 'q'): the caller's object path then decides,
    as the per-item conversion's cleared error did."""
    try:
        return py_call(native_loaded_module("array").array, PythonObject(typecode), values)
    except:
        return PythonObject(None)


def label_store_address(store: PythonObject) raises -> Int:
    """The first word's address of an `array.array` (its `buffer_info()`)."""
    return Int(py=py_call(store.buffer_info)[0])
