# SPDX-License-Identifier: Apache-2.0
# Compile-only root (lane decode-resident-20261028); never invoke.
# Covers the byte-LM resident train step MOJOLEARN_IDN_TRAIN_ONE_WAIT changes: the session step
# binding (byte_session_step -> byte_train_step_resident -> _byte_step_device) and the native
# descriptor operation (byte_owner_step), which share the step body.
from std.python._cpython import PyObjectPtr
from std.python import PythonObject
from bindings._mojolearn_byte_lm import byte_lm_session_step_binding
from training.byte_lm_native_dispatch import byte_lm_native_operation


@export
def drio_compile_byte_lm_operation(op: Int, a0: Int, a1: Int, a2: Int) abi("C") -> Int:
    try:
        return byte_lm_native_operation(op, [a0, a1, a2], [a0, a1, a2, a2], [Float64(1e-3)])
    except:
        return -1


@export
def drio_compile_byte_lm_step(s: PyObjectPtr, x: PyObjectPtr, y: PyObjectPtr, z: PyObjectPtr) abi("C") -> Int:
    try:
        _ = byte_lm_session_step_binding(
            PythonObject(from_borrowed=s), PythonObject(from_borrowed=x),
            PythonObject(from_borrowed=y), PythonObject(from_borrowed=z))
        return 0
    except:
        return -1
