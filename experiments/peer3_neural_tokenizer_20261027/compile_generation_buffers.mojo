# SPDX-License-Identifier: Apache-2.0
# Compile-only allocation, admission and retained generation-buffer operations.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.generation_buffers import (
    generation_initialize_binding, generation_array, generation_array_snapshot,
    generation_buffer_view, generation_address, generation_ids, generation_copy,
    generation_mutable,
)


@export
def peer3_generation_buffers_compile(op: Int, pointer: PyObjectPtr) abi("C") -> Int:
    try:
        var value = PythonObject(from_owned=Python().cpython().Py_NewRef(pointer))
        var keep = List[PythonObject]()
        var copies = List[PythonObject]()
        if op == 0:
            _ = generation_initialize_binding(value)
        elif op == 1:
            _ = generation_array([1, 1])
        elif op == 2:
            _ = generation_array_snapshot(value)
        elif op == 3:
            _ = generation_buffer_view(value)
        elif op == 4:
            return generation_address(value, keep)
        elif op == 5:
            _ = generation_ids(value, 256)
        elif op == 6:
            _ = generation_copy(value)
        else:
            return generation_mutable(value, [1, 1], keep, copies)
        return 0
    except:
        return -1
