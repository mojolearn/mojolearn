# SPDX-License-Identifier: Apache-2.0
# Compile-only forwarding entry to the actual native family admission helper.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from providers.python.classical_family_admission_pyapi import classical_family_admit


@export
def classical_family_admission_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        return classical_family_admit(args[0]).steal_data()
    except error:
        return raise_python_exception(error)
