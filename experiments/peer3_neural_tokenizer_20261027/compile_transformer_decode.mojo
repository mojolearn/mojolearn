# SPDX-License-Identifier: Apache-2.0
# Compile-only exact Transformer decode door; never invoke with unvalidated pointers.
from std.python._cpython import PyObjectPtr
from providers.python.transformer_decode_pyapi import transformer_decode_native_c


@export
def peer3_transformer_decode_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return transformer_decode_native_c(py_self, args)
