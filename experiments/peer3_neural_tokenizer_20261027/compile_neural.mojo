# SPDX-License-Identifier: Apache-2.0
# Small compile-only public boundary roots. Do not execute these raw doors.
from std.python._cpython import PyObjectPtr
from bindings.neural_block_public import neural_block_c


@export
def peer3_neural_0_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return neural_block_c[0](py_self, args)


@export
def peer3_neural_1_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return neural_block_c[1](py_self, args)


@export
def peer3_neural_2_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return neural_block_c[2](py_self, args)


@export
def peer3_neural_3_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return neural_block_c[3](py_self, args)


@export
def peer3_neural_4_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return neural_block_c[4](py_self, args)


@export
def peer3_neural_5_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return neural_block_c[5](py_self, args)


@export
def peer3_neural_6_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return neural_block_c[6](py_self, args)


@export
def peer3_neural_7_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return neural_block_c[7](py_self, args)
