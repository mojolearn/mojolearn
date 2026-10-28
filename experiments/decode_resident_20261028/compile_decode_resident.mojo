# SPDX-License-Identifier: Apache-2.0
# Compile-only root (lane decode-resident-20261028); never invoke.
# Covers the resident decode step closures MOJOLEARN_IDN_DECODE_RESIDENT_IO
# changes: the transformer decode session (open, open_int15, step through the
# Python bindings) and the Mamba-2 session step. transformer/native_decode_dispatch.mojo is left
# out: on main cbc14776d7 it fails to parse with Mojo 1.0.0 (ed45d567) at :49 (a conditional over
# two list literals of different lengths), independent of this lane.
from std.python._cpython import PyObjectPtr
from std.python import PythonObject
from bindings._mojolearn_transformer import (
    transformer_decode_session_open_binding,
    transformer_decode_session_open_int15_binding,
    transformer_decode_session_step_binding,
)
from bindings._mojolearn_mamba import (
    mamba2_session_open_binding,
    mamba2_session_step_binding,
)


@export
def drio_compile_session_bindings(which: Int, s: PyObjectPtr, x: PyObjectPtr, y: PyObjectPtr) abi("C") -> Int:
    try:
        var so = PythonObject(from_borrowed=s)
        var xo = PythonObject(from_borrowed=x)
        var yo = PythonObject(from_borrowed=y)
        if which == 0:
            _ = transformer_decode_session_open_binding(so, xo, yo)
        elif which == 1:
            _ = transformer_decode_session_open_int15_binding(so, xo, yo)
        elif which == 2:
            _ = transformer_decode_session_step_binding(so, xo, yo)
        elif which == 3:
            _ = mamba2_session_open_binding(so, xo, yo)
        else:
            _ = mamba2_session_step_binding(so, xo)
        return 0
    except:
        return -1
