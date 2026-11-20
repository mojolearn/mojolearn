# SPDX-License-Identifier: Apache-2.0
# RUN pieces shared by the classification, ranking and clustering metric
# doors: typed only, no Python import (tools/python_free_dirs.sh). Lane
# metric-programs-2 (2026-11-12): split from metrics_exp_common.mojo.
from std.memory import bitcast

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
