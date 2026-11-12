# SPDX-License-Identifier: Apache-2.0
# Compile actual linalg buffer, low-bit, decomposition and result metadata owners.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from bindings.linalg_ops_native import linalg_ops_operation

@export
def linalg_buffer_boundary_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        return linalg_ops_operation(args[0], String(args[1]), args[2], args[3]).steal_data()
    except error:
        return raise_python_exception(error)
