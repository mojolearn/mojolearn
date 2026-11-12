# SPDX-License-Identifier: Apache-2.0
# Public CPython buffer/allocator calls for already-admitted native storage.
# These acquire owned references without looking up mutable builtins fields.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table


def native_buffer_view(value: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    var result = cpython_table()[].PyMemoryView_FromObject(value._obj_ptr)
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


def native_storage_allocate(cls: PythonObject) raises -> PythonObject:
    # The connected Array class uses object.__new__ and its ordinary heap
    # allocation. Allocate that layout directly; no Python __new__ runs.
    ref cpy = Python().cpython()
    var result = cpython_table()[].PyType_GenericAlloc(cls._obj_ptr, Int(0))
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)
