# SPDX-License-Identifier: Apache-2.0
# Compile-only root for the TruncatedSVD native public door (bindings/tsvd_native.mojo tsvd_c).
from std.python._cpython import PyObjectPtr
from bindings.tsvd_native import tsvd_c

@export
def lifecycle_tsvd_compile(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return tsvd_c(py_self, args_ptr)
