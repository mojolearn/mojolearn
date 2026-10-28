# SPDX-License-Identifier: Apache-2.0
# Compile-only native CausalLM state construction and representation boundaries.
from std.python._cpython import PyObjectPtr
from bindings.causal_lm_public import causal_public_c

@export
def peer3_causal_metadata_15(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[15](py_self, args)

@export
def peer3_causal_metadata_16(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[16](py_self, args)

@export
def peer3_causal_metadata_17(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[17](py_self, args)

