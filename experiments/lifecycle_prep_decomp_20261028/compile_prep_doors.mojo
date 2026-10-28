# SPDX-License-Identifier: Apache-2.0
# Compile-only root for the prep lifecycle doors: prep_stateless_c, prep_owner_c
# (prep_encode_c and the label encoders) and prep_classify_c. No runtime claimed.
from std.python._cpython import PyObjectPtr
from bindings.prep_stateless_native import prep_stateless_c
from bindings.prep_owner_native import prep_owner_c
from bindings.prep_classify_native import prep_classify_c


@export
def lifecycle_prep_stateless(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return prep_stateless_c(py_self, args_ptr)


@export
def lifecycle_prep_owner(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return prep_owner_c(py_self, args_ptr)


@export
def lifecycle_prep_classify(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return prep_classify_c(py_self, args_ptr)
