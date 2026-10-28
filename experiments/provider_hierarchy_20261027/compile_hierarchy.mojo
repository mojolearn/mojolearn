# SPDX-License-Identifier: Apache-2.0
# Compile-only export of the actual hierarchy public owner; never execute it.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from bindings.cfb_agglo import cfb_agglo_operation


@export
def peer1_hierarchy_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        return cfb_agglo_operation(args[0], String(args[1]), args[2], args[3]).steal_data()
    except error:
        return raise_python_exception(error)
