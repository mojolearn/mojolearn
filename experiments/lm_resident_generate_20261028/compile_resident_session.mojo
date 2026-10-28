# SPDX-License-Identifier: Apache-2.0
# Compile-only root (lane lm-resident-generate-20261028); never invoke.
from std.python._cpython import PyObjectPtr
from std.python import PythonObject
from transformer.native_lm_resident_dispatch import transformer_lm_resident_operation
from bindings._mojolearn_transformer import (
    causal_lm_session_open_binding, causal_lm_session_open_int15_binding,
    causal_lm_session_run_binding, causal_lm_session_forward_binding,
)


@export
def lmrg_compile_resident_operation(op: Int, a0: Int, a1: Int, a2: Int) abi("C") -> Int:
    try:
        return transformer_lm_resident_operation(op, [a0, a1, a2], [a0, a1, a2, a2], [Float64(1e-5)])
    except:
        return -1


@export
def lmrg_compile_session_bindings(which: Int, s: PyObjectPtr, x: PyObjectPtr, y: PyObjectPtr, z: PyObjectPtr) abi("C") -> Int:
    try:
        var so = PythonObject(from_borrowed=s)
        var xo = PythonObject(from_borrowed=x)
        var yo = PythonObject(from_borrowed=y)
        var zo = PythonObject(from_borrowed=z)
        if which == 0:
            _ = causal_lm_session_open_binding(so, xo, yo)
        elif which == 1:
            _ = causal_lm_session_open_int15_binding(so, xo, yo)
        elif which == 2:
            _ = causal_lm_session_run_binding(so, xo, yo, zo)
        else:
            _ = causal_lm_session_forward_binding(so, xo, yo, zo)
        return 0
    except:
        return -1
