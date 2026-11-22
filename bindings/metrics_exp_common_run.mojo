# SPDX-License-Identifier: Apache-2.0
# RUN pieces shared by the classification, ranking and clustering metric
# doors: typed only, no Python import (tools/python_free_dirs.sh). Lane
# metric-programs-2 (2026-11-12): split from metrics_exp_common.mojo.
from std.memory import bitcast
from std.math import sqrt

comptime MX_NONE = -1


def mx_nan() -> Float64:
    return bitcast[DType.float64](UInt64(0x7FF8000000000000))


def mx_inf() -> Float64:
    return bitcast[DType.float64](UInt64(0x7FF0000000000000))


def mx_isnan(x: Float64) -> Bool:
    return x != x


def mx_isinf(x: Float64) -> Bool:
    return x == x and x - x != 0.0


def mx_class_code(kind: Int, item: Int) -> String:
    """The `array.array` code holding one class item of a label buffer."""
    if kind == 1:  # signed
        if item == 1:
            return "b"
        if item == 2:
            return "h"
        if item == 4:
            return "i"
        return "q"
    if item == 1:
        return "B"
    if item == 2:
        return "H"
    if item == 4:
        return "I"
    return "Q"


# Moved verbatim from bindings/metrics_exp_cluster_run.mojo (lane bindings-inline, 2026-11-22).
# RUN part of bindings/metrics_exp_cluster.mojo: typed Mojo only (no Python
# imports). The binary64 epilogue of NMI and AMI.

def _generalized_average(U: Float64, V: Float64, method: Int) raises -> Float64:
    if method == 0:
        return V if V < U else U
    if method == 1:
        var x = U * V
        if x < 0.0:
            raise Error("math domain error")
        return sqrt(x)
    if method == 2:
        return (U + V) / 2
    return V if V > U else U


def nmi_finish(mi: Float64, entropy_true: Float64, entropy_pred: Float64, method: Int) raises -> Float64:
    """RUN: NMI from the program's MI and the two entropies (lane
    entangled-programs: moved out of the door, same order of operations)."""
    if mi == 0:
        return 0.0
    return mi / _generalized_average(entropy_true, entropy_pred, method)


def ami_finish(mi: Float64, emi: Float64, entropy_true: Float64, entropy_pred: Float64, method: Int) raises -> Float64:
    """RUN: AMI's binary64 epilogue (the eps guards on numerator and
    denominator), moved out of the door unchanged."""
    var norm = _generalized_average(entropy_true, entropy_pred, method)
    var eps = 2.220446049250313e-16
    var den = norm - emi
    if den < 0:
        den = -eps if -eps < den else den
    else:
        den = eps if eps > den else den
    var num = mi - emi
    if num < 0:
        num = -eps if -eps < num else num
    else:
        num = eps if eps > num else num
    return num / den


def contingency_dtype_code(name: String) -> Int:
    """RUN: the requested dtype's cast code for `mexp_cast_i64` (0 int32,
    1 float64, 2 float32), -1 to keep the int64 counts."""
    if name == "int32":
        return 0
    if name == "float64":
        return 1
    if name == "float32":
        return 2
    return -1
