# SPDX-License-Identifier: Apache-2.0
# Compile-only whole public model constructor/load/execution/state boundary.
from std.python._cpython import PyObjectPtr
from bindings.causal_lm_construct import causal_model_c
from bindings.causal_lm_public import causal_public_c


@export
def peer3_causal_model_compile(op: Int, py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    if op == 18:
        return causal_model_c[0](py_self, args)
    if op == 19:
        return causal_model_c[1](py_self, args)
    comptime for index in range(18):
        if op == index:
            return causal_public_c[index](py_self, args)
    return PyObjectPtr()
