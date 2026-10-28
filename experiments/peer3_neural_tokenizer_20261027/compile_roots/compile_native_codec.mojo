# SPDX-License-Identifier: Apache-2.0
# Compile-only closure: dynamic codec_c reaches initialize, pack and unpack.
# Compilation is not wire-format or runtime qualification.
from std.python._cpython import PyObjectPtr
from bindings.native_codec import codec_c

@export
def peer3_native_codec(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return codec_c(py_self, args)
