# SPDX-License-Identifier: Apache-2.0
from std.python._cpython import PyObjectPtr
from bindings.pls_native import pls_c


@export
def peer2_pls(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return pls_c(py_self, args_ptr)
