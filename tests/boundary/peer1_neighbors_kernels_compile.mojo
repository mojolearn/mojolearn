# SPDX-License-Identifier: Apache-2.0
# Compile-only retention of the real neighbor/kernel public native owners.
# This is not a runtime test and never constructs a model or starts a kernel.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from bindings.neighbors_native import neighbors_operation
from bindings.kernel_methods_native import kernel_methods_operation
from bindings.neighbor_query_native import neighbor_query_c, neighbor_query_attribute_c
from bindings.public_attributes_native import public_attribute_c
from providers.python.classical_family_admission_pyapi import classical_family_admit


@export
def peer1_family_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        _ = classical_family_admit(args[0])
        var operation = String(args[1])
        if operation.startswith("family_km_"):
            return kernel_methods_operation(args[0], operation, args[2], args[3]).steal_data()
        return neighbors_operation(args[0], operation, args[2], args[3]).steal_data()
    except error:
        return raise_python_exception(error)


@export
def peer1_query_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    return neighbor_query_c(py_self, packed)


@export
def peer1_query_attribute_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    return neighbor_query_attribute_c(py_self, packed)


@export
def peer1_public_attribute_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    return public_attribute_c(py_self, packed)
