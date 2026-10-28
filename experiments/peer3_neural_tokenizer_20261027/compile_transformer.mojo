# SPDX-License-Identifier: Apache-2.0
# Small compile-only public boundary roots. Do not execute these raw doors.
from std.python._cpython import PyObjectPtr
from bindings.transformer_public_native import transformer_public_c


@export
def peer3_transformer_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return transformer_public_c(py_self, args)
