# SPDX-License-Identifier: Apache-2.0
# Compile-only export of the actual neighbor public owner; never execute it.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from bindings.neighbors_native import neighbors_operation


@export
def peer1_radius_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        return neighbors_operation(args[0], String(args[1]), args[2], args[3]).steal_data()
    except error:
        return raise_python_exception(error)
