# SPDX-License-Identifier: Apache-2.0
# Compile-only actual DBSCAN/KDE lifecycle and public attribute owner entries.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from providers.python.classical_family_admission_pyapi import classical_family_admit
from providers.python.density_public_lifecycle_pyapi import density_public_operation
from bindings.public_attributes_native import public_attribute_c

@export
def density_public_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        _ = classical_family_admit(args[0])
        return density_public_operation(args[0], String(args[1]), args[2], args[3]).steal_data()
    except error:
        return raise_python_exception(error)

@export
def density_attribute_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    return public_attribute_c(py_self, packed)
