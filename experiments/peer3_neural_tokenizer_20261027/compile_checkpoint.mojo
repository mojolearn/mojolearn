# SPDX-License-Identifier: Apache-2.0
# Compile-only native public safetensors/checkpoint/configuration entry points.
from std.python._cpython import PyObjectPtr
from bindings.causal_checkpoint import checkpoint_c


@export
def peer3_checkpoint_compile(op: Int, py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    comptime for index in range(20):
        if op == index:
            return checkpoint_c[index](py_self, args)
    return PyObjectPtr()
