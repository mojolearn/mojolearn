# SPDX-License-Identifier: Apache-2.0
# Compile-only roots for SVM state, coefficient, and installed attribute owners.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from providers.python.classical_family_admission_pyapi import classical_family_admit
from providers.python.svc_public_state_pyapi import svc_public_state_operation
from providers.python.estimator_binding_pyapi import estimator_binding
from bindings.svc_kernel_native import svc_kernel_operation
from bindings.public_attributes_native import public_attribute_c

@export
def svm_state_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        _ = classical_family_admit(args[0])
        return svc_public_state_operation(args[0], String(args[1]), args[2], args[3]).steal_data()
    except error:
        return raise_python_exception(error)

@export
def svm_coef_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        _ = classical_family_admit(args[0])
        var module = estimator_binding(args[0], "_mojolearn_svm", args[3])
        return svc_kernel_operation(args[0], String(args[1]), Python.tuple(module), args[3]).steal_data()
    except error:
        return raise_python_exception(error)

@export
def svm_attribute_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    return public_attribute_c(py_self, packed)
