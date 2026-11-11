# SPDX-License-Identifier: Apache-2.0
# Compile-only split of actual native entries; no product operation is run.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from bindings.kernel_methods_native import kernel_methods_operation
from providers.python.classical_family_admission_pyapi import classical_family_admit

@export
def peer1_kernel_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        _ = classical_family_admit(args[0])
        return kernel_methods_operation(args[0], String(args[1]), args[2], args[3]).steal_data()
    except error:
        return raise_python_exception(error)
