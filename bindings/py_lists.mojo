# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Publish completed native scalar Lists directly as Python lists.

No intermediate NumPy array, buffer metadata lookup, memcpy or tolist method
is needed. These loops only box completed results: f32 promotes exactly to a
Python double, signed/unsigned integers use their corresponding 64-bit API.
No input Python protocols or numerical algorithm execute here.
"""
from std.ffi import c_int
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr


def _py_result_list(count: Int) raises -> PythonObject:
    # Keep the NULL-filled list private until complete. Its PythonObject owner
    # releases every initialized slot on failure; CPython skips NULL slots.
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyList_New")(count)
    if not raw:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=raw)


def py_list_f64(values: List[Float64]) raises -> PythonObject:
    var out = _py_result_list(len(values))
    ref cpy = Python().cpython()
    var box = cpy.lib.get_function[PyObjectPtr]("PyFloat_FromDouble")
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    for i in range(len(values)):  # driver-loop: over completed native scalar result slots for Python publication
        var item = box(Float64(values[i]))
        if not item:
            raise cpy.unsafe_get_error()
        # SetItem steals the new reference on BOTH success and failure.
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    _ = values
    return out


def py_list_f32(values: List[Float32]) raises -> PythonObject:
    var out = _py_result_list(len(values))
    ref cpy = Python().cpython()
    var box = cpy.lib.get_function[PyObjectPtr]("PyFloat_FromDouble")
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    for i in range(len(values)):  # driver-loop: over completed native scalar result slots for Python publication
        var item = box(Float64(values[i]))
        if not item:
            raise cpy.unsafe_get_error()
        # SetItem steals the new reference on BOTH success and failure.
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    _ = values
    return out


def py_list_i32(values: List[Int32]) raises -> PythonObject:
    var out = _py_result_list(len(values))
    ref cpy = Python().cpython()
    var box = cpy.lib.get_function[PyObjectPtr]("PyLong_FromLongLong")
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    for i in range(len(values)):  # driver-loop: over completed native scalar result slots for Python publication
        var item = box(Int64(values[i]))
        if not item:
            raise cpy.unsafe_get_error()
        # SetItem steals the new reference on BOTH success and failure.
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    _ = values
    return out


def py_list_i64(values: List[Int64]) raises -> PythonObject:
    var out = _py_result_list(len(values))
    ref cpy = Python().cpython()
    var box = cpy.lib.get_function[PyObjectPtr]("PyLong_FromLongLong")
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    for i in range(len(values)):  # driver-loop: over completed native scalar result slots for Python publication
        var item = box(Int64(values[i]))
        if not item:
            raise cpy.unsafe_get_error()
        # SetItem steals the new reference on BOTH success and failure.
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    _ = values
    return out


def py_list_u32(values: List[UInt32]) raises -> PythonObject:
    var out = _py_result_list(len(values))
    ref cpy = Python().cpython()
    var box = cpy.lib.get_function[PyObjectPtr]("PyLong_FromUnsignedLongLong")
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    for i in range(len(values)):  # driver-loop: over completed native scalar result slots for Python publication
        var item = box(UInt64(values[i]))
        if not item:
            raise cpy.unsafe_get_error()
        # SetItem steals the new reference on BOTH success and failure.
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    _ = values
    return out


def py_list_u64(values: List[UInt64]) raises -> PythonObject:
    var out = _py_result_list(len(values))
    ref cpy = Python().cpython()
    var box = cpy.lib.get_function[PyObjectPtr]("PyLong_FromUnsignedLongLong")
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    for i in range(len(values)):  # driver-loop: over completed native scalar result slots for Python publication
        var item = box(UInt64(values[i]))
        if not item:
            raise cpy.unsafe_get_error()
        # SetItem steals the new reference on BOTH success and failure.
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    _ = values
    return out


def py_list_int(values: List[Int]) raises -> PythonObject:
    var out = _py_result_list(len(values))
    ref cpy = Python().cpython()
    var box = cpy.lib.get_function[PyObjectPtr]("PyLong_FromLongLong")
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    for i in range(len(values)):  # driver-loop: over completed native scalar result slots for Python publication
        var item = box(Int64(values[i]))
        if not item:
            raise cpy.unsafe_get_error()
        # SetItem steals the new reference on BOTH success and failure.
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    _ = values
    return out
