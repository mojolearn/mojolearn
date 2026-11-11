# SPDX-License-Identifier: Apache-2.0
# Compile-only split of actual native entries; no product operation is run.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from bindings.public_attributes_native import public_attribute_c

@export
def peer1_attribute_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    return public_attribute_c(py_self, packed)
