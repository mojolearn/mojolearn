# SPDX-License-Identifier: Apache-2.0
# Compile the actual family dispatcher (SVC/SVR/linear/OCSVM lifecycle, spectral,
# shared classical label admission) and the installed public attribute owner.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from bindings.family_native import family_operation
from bindings.public_attributes_native import public_attribute_c

@export
def svm_public_lifecycle_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        return family_operation(args[0], String(args[1]), args[2], args[3]).steal_data()
    except error:
        return raise_python_exception(error)

@export
def svm_public_attribute_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    return public_attribute_c(py_self, packed)
