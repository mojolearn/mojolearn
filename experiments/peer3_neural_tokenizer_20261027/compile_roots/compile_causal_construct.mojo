# SPDX-License-Identifier: Apache-2.0
# Compile-only complete native construction/load/read-weights public roots.
from std.python._cpython import PyObjectPtr
from bindings.causal_lm_construct import causal_model_c

@export
def peer3_causal_construct_0(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_model_c[0](py_self, args)

@export
def peer3_causal_construct_1(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_model_c[1](py_self, args)

@export
def peer3_causal_construct_2(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_model_c[2](py_self, args)

