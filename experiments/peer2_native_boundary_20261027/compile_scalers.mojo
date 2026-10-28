# SPDX-License-Identifier: Apache-2.0
# Focused public scaler entries; compile status is independent of PLS.
from std.python._cpython import PyObjectPtr
from bindings.scaler_native import scaler_fit_c
from bindings.scaler_lifecycle_native import scaler_lifecycle_c
from bindings.minmax_native import minmax_lifecycle_c


@export
def peer2_scaler_fit(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return scaler_fit_c(py_self, args_ptr)


@export
def peer2_scaler_lifecycle(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return scaler_lifecycle_c(py_self, args_ptr)


@export
def peer2_minmax_lifecycle(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return minmax_lifecycle_c(py_self, args_ptr)
