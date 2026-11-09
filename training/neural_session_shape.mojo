# SPDX-License-Identifier: Apache-2.0
"""Session address and extent admission shared by native device/host bindings."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


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

