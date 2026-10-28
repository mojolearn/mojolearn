# SPDX-License-Identifier: Apache-2.0
# Compile Linalg metadata admission, lifecycle, dispatch and result ownership.
from std.os import abort
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder, raise_python_exception
from providers.python.classical_family_admission_pyapi import classical_family_admit
from bindings.linalg_ops_decomp import lin_decomp_operation


def linalg_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        return lin_decomp_operation(String(args[1]), args[2], args[3]).steal_data()
    except problem:
        return raise_python_exception(problem)


@export
def PyInit__mojolearn_linalg_constructor_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_linalg_constructor_compile")
        module.def_py_c_function(linalg_c, "linalg")
        return module.finalize()
    except problem:
        abort(String(problem))
