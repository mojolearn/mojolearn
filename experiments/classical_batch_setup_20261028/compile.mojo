# SPDX-License-Identifier: Apache-2.0
# Compile-only actual public HDBSCAN and regression metric provider roots.
from std.os import abort
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder, raise_python_exception
from checks.hdbscan_public_boundary_compile import hdbscan_c
from bindings.metrics_reg_native import metrics_reg_operation


def regression_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        return metrics_reg_operation(String(args[0]), args[1], args[2]).steal_data()
    except problem:
        return raise_python_exception(problem)


@export
def PyInit__classical_batch_setup() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_classical_batch_setup")
        module.def_py_c_function(hdbscan_c, "hdbscan")
        module.def_py_c_function(regression_c, "regression")
        return module.finalize()
    except problem:
        abort(String(problem))
