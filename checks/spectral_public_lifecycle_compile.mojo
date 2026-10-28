# SPDX-License-Identifier: Apache-2.0
# Compile the actual cluster_fit_b family route that dispatches the spectral public lifecycle.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from bindings.cluster_fit_b_native import cluster_fit_b_operation

@export
def spectral_public_lifecycle_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        return cluster_fit_b_operation(args[0], String(args[1]), args[2], args[3]).steal_data()
    except error:
        return raise_python_exception(error)
