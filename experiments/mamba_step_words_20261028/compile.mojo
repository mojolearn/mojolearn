# Compile-only full public closure after native plan ownership extraction.
from std.python._cpython import PyObjectPtr
from bindings.mamba_public_native import mamba_native_c


@export
def mamba_public_compile(self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return mamba_native_c(self, args)
