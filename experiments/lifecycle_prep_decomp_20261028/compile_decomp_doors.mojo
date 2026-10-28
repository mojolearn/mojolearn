# SPDX-License-Identifier: Apache-2.0
# Compile-only root: the five decomposition doors whose public lifecycle runs
# natively (bindings/decomp_public_lifecycle.mojo). Not runtime qualification.
from std.python._cpython import PyObjectPtr
from bindings.decomp_native import decomp_c
from bindings.ipca_native import ipca_c
from bindings.dict_learning_native import dict_learning_c
from bindings.mcd_native import mcd_c
from bindings.als_native import als_c


@export
def lpd_decomp_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return decomp_c(py_self, args_ptr)


@export
def lpd_ipca_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return ipca_c(py_self, args_ptr)


@export
def lpd_dict_learning_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return dict_learning_c(py_self, args_ptr)


@export
def lpd_mcd_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return mcd_c(py_self, args_ptr)


@export
def lpd_als_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return als_c(py_self, args_ptr)
