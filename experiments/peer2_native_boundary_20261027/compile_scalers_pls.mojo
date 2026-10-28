# SPDX-License-Identifier: Apache-2.0
# Focused real public entries. A successful build qualifies compilation only.
from std.python._cpython import PyObjectPtr
from bindings.scaler_native import scaler_fit_c
from bindings.scaler_lifecycle_native import scaler_lifecycle_c
from bindings.minmax_native import minmax_lifecycle_c
from bindings.pls_native import pls_c


@export
def peer2_scaler_fit(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return scaler_fit_c(py_self, args_ptr)


@export
def peer2_scaler_lifecycle(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return scaler_lifecycle_c(py_self, args_ptr)


@export
def peer2_minmax_lifecycle(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return minmax_lifecycle_c(py_self, args_ptr)


@export
def peer2_pls(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return pls_c(py_self, args_ptr)
