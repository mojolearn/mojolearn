# SPDX-License-Identifier: Apache-2.0
"""Borrowed CPython exception references owned by the active ABI call."""
from std.python import Python
from std.python._cpython import PyObjectPtr


def causal_borrowed_item(mapping: PyObjectPtr, key: String) raises -> PyObjectPtr:
    ref cpy = Python().cpython()
    # DecodeUTF8 returns a new reference; GetItemWithError returns a borrowed
    # reference. Only the temporary key is ours to release. The dictionary is
    # retained by the Python call's args tuple for this entire native call.
    var name = cpy.PyUnicode_DecodeUTF8(key)
    if not name:
        raise cpy.unsafe_get_error()
    var value = cpy.PyDict_GetItemWithError(mapping, name)
    cpy.Py_DecRef(name)
    if not value:
        raise Error("model exception boundary lacks " + key)
    return value


def causal_set_exception(error: Error, cls: PyObjectPtr) -> PyObjectPtr:
    # PyErr_SetString borrows its exception type. The calling CPython frame
    # owns args and the context/class reachable through it; no transfer or
    # manual decrement of the class is performed here.
    # https://docs.python.org/3/c-api/exceptions.html#c.PyErr_SetString
    ref cpy = Python().cpython()
    var message = String(error)
    cpy.PyErr_SetString(cls, message.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin())
    return PyObjectPtr()
