# SPDX-License-Identifier: Apache-2.0
"""Host model and admission for NN64; no GPU imports or Python data work."""
from std.math import isfinite
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
from gemm.host.neural_gemm import gemm_host_rows
from gemm.contract import OP_NT

comptime _FP = MutPointer[Float32, MutUntrackedOrigin]


def nn_mlp_session_shape(inputs: List[Int], rows: List[Int], in_width: Int, hidden: Int, out_width: Int) raises -> Int:
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("neural MLP sessions: IDENTICAL required")
    if len(inputs) != len(rows) or len(rows) < 1 or in_width < 1 or hidden < 1 or out_width < 1:
        raise Error("neural MLP sessions: invalid metadata")
    var limit = 2147483647
    if max(in_width, max(hidden, out_width)) > limit or hidden > limit // in_width or out_width > limit // hidden:
        raise Error("neural MLP sessions: weights exceed index range")
    var total = 0
    for i in range(len(rows)):
        if rows[i] < 0 or rows[i] > limit - total or (rows[i] > 0 and inputs[i] <= 0):
            raise Error("neural MLP sessions: invalid session rows/address")
        total += rows[i]
    if total > limit // max(in_width, max(hidden, out_width)):
        raise Error("neural MLP sessions: activation exceeds index range")
    return total


def _read_finite(p: _FP, n: Int, label: String) raises -> List[Float32]:
    var values = List[Float32](capacity=n)
    for i in range(n):
        var value = p[i]
        if not isfinite(value):
            raise Error("neural MLP sessions: nonfinite " + label)
        values.append(value)
    return values^


def nn_mlp_sessions_host(inputs: List[Int], rows: List[Int], w1: _FP, b1: _FP,
    w2: _FP, b2: _FP, output: _FP, in_width: Int, hidden: Int, out_width: Int,
) raises -> Int:
    var total = nn_mlp_session_shape(inputs, rows, in_width, hidden, out_width)
    var weight1 = _read_finite(w1, hidden * in_width, "weights")
    var bias1 = _read_finite(b1, hidden, "weights")
    var weight2 = _read_finite(w2, out_width * hidden, "weights")
    var bias2 = _read_finite(b2, out_width, "weights")
    var x = List[Float32](capacity=total * in_width)
    for session in range(len(rows)):
        var part = _read_finite(_FP(unsafe_from_address=inputs[session]), rows[session] * in_width, "input")
        for i in range(len(part)):
            x.append(part[i])
    if total == 0:
        return 0
    var a = gemm_host_rows(x, weight1, OP_NT, total, hidden, in_width)
    for i in range(total * hidden):
        var value = ftz(identical_mul_add(Float32(1), ftz(a[i]), ftz(bias1[i % hidden])))
        a[i] = Float32(0) if value <= Float32(0) else value
    var y = gemm_host_rows(a, weight2, OP_NT, total, out_width, hidden)
    for i in range(total * out_width):
        y[i] = ftz(identical_mul_add(Float32(1), ftz(y[i]), ftz(bias2[i % out_width])))
        if not isfinite(y[i]):
            raise Error("neural MLP sessions: nonfinite logits")
    # All admission/output status completes before exposing any session result.
    for i in range(total * out_width):
        output[i] = y[i]
    return total * out_width
