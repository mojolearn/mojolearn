# SPDX-License-Identifier: Apache-2.0
# Compile-only complete native CausalLM public operation closures.
from std.python._cpython import PyObjectPtr
from bindings.causal_lm_public import causal_public_c

@export
def peer3_compile_causal_generation_4(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[4](py_self, args)

@export
def peer3_compile_causal_generation_5(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[5](py_self, args)

@export
def peer3_compile_causal_generation_6(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[6](py_self, args)

@export
def peer3_compile_causal_generation_7(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[7](py_self, args)

@export
def peer3_compile_causal_generation_8(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[8](py_self, args)

@export
def peer3_compile_causal_generation_9(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[9](py_self, args)

@export
def peer3_compile_causal_generation_11(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[11](py_self, args)

@export
def peer3_compile_causal_generation_12(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[12](py_self, args)

