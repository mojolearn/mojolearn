# SPDX-License-Identifier: Apache-2.0
# Compile the real constructor and reset entries, including UMAP admission.
from std.python._cpython import PyObjectPtr
from bindings.full_reset_constructor_native import full_reset_constructor_c
from bindings.full_reset_params_native import full_reset_params_c


@export
def peer2_constructor(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return full_reset_constructor_c(py_self, args_ptr)


@export
def peer2_reset(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return full_reset_params_c(py_self, args_ptr)
