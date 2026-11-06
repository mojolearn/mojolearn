# SPDX-License-Identifier: Apache-2.0
# Shared result builders for the byte LM, Mamba and Samba bindings: a Python
# list of ints published in one pass through the CPython C API (PyList_New,
# PyLong_FromLongLong, PyList_SetItem), with no per-item `list.append` call.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.ffi import c_int


def byte_int_list_from_i32(address: Int, count: Int) raises -> PythonObject:
    """A new list of `count` Python ints read from native int32 storage."""
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyList_New")(count)
    if not raw:
        raise cpy.unsafe_get_error()
    # The list owns itself at once; unset NULL slots are safe to destroy.
    var out = PythonObject(from_owned=raw)
    var box_int = cpy.lib.get_function[PyObjectPtr]("PyLong_FromLongLong")
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    var source = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(count):  # cpu-route: publish completed native token ids as Python integers
        var item = box_int(Int64(source[i]))
        if not item:
            raise cpy.unsafe_get_error()
        # SetItem steals the new reference on success and failure.
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    return out


def byte_int_list_from_i64(address: Int, count: Int) raises -> PythonObject:
    """A new list of `count` Python ints read from native int64 storage."""
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyList_New")(count)
    if not raw:
        raise cpy.unsafe_get_error()
    var out = PythonObject(from_owned=raw)
    var box_int = cpy.lib.get_function[PyObjectPtr]("PyLong_FromLongLong")
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    var source = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(count):  # cpu-route: publish completed native token ids as Python integers
        var item = box_int(source[i])
        if not item:
            raise cpy.unsafe_get_error()
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    return out
