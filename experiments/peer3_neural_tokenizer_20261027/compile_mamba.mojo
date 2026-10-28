# SPDX-License-Identifier: Apache-2.0
# Small compile-only public boundary roots. Do not execute these raw doors.
from std.python._cpython import PyObjectPtr
from bindings.mamba_public_native import mamba_native_c


@export
def peer3_mamba_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return mamba_native_c(py_self, args)
