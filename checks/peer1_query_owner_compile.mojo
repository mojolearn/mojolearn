# SPDX-License-Identifier: Apache-2.0
# Compile-only split of actual native entries; no product operation is run.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from bindings.neighbor_query_native import neighbor_query_c, neighbor_query_attribute_c

@export
def peer1_query_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    return neighbor_query_c(py_self, packed)

@export
def peer1_query_attribute_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    return neighbor_query_attribute_c(py_self, packed)
