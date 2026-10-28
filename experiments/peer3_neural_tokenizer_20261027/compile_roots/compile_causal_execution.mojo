# SPDX-License-Identifier: Apache-2.0
# Compile-only complete native CausalLM public operation closures.
from std.python._cpython import PyObjectPtr
from bindings.causal_lm_public import causal_public_c

@export
def peer3_compile_causal_execution_1(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[1](py_self, args)

@export
def peer3_compile_causal_execution_2(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[2](py_self, args)

@export
def peer3_compile_causal_execution_3(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[3](py_self, args)

@export
def peer3_compile_causal_execution_10(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[10](py_self, args)

@export
def peer3_compile_causal_execution_13(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[13](py_self, args)

@export
def peer3_compile_causal_execution_14(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[14](py_self, args)

