# SPDX-License-Identifier: Apache-2.0
# Compile-only root for the true public Cholesky owner and shared admission.
from std.os import abort
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder, raise_python_exception
from providers.python.classical_family_admission_pyapi import classical_family_admit
from bindings.linalg_native import linalg_operation


def cholesky_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        _ = classical_family_admit(args[0])
        return linalg_operation(args[0], String(args[1]), args[2], args[3]).steal_data()
    except problem:
        return raise_python_exception(problem)


@export
def PyInit__mojolearn_cholesky_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_cholesky_compile")
        module.def_py_c_function(cholesky_c, "cholesky")
        return module.finalize()
    except problem:
        abort(String(problem))
