# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column of x_neighbors/kapprox_dev.mojo: the same items in a plain
loop, so the CPU binding exports the same names over the same address
contract. HOST ONLY. The Python layer never takes the kapprox device path on
this binding (`x_neighbors_kapprox_fast()` is 0 here)."""
from std.python import PythonObject
from x_neighbors.items import FP, IP, skew_transform_item
from x_neighbors.kapprox_items import (
    kapprox_check_item,
    kapprox_achi2_item,
    kapprox_skew_fit_item,
    kapprox_skew_log_item,
)


def kapprox_fast_binding() raises -> PythonObject:
    """The host column never takes the device fit/transform."""
    return PythonObject(0)


def op_kapprox_check(x: Int, flag: Int, n: Int, d: Int, strict: Int, floor: Float32) raises:
    var p_x = FP(unsafe_from_address=x)
    var p_flag = IP(unsafe_from_address=flag)
    for t in range(n * d):
        kapprox_check_item(t, p_x, p_flag, n, d, strict, floor)


def op_kapprox_achi2(x: Int, res: Int, flag: Int, n: Int, d: Int, steps: Int, interval: Float32) raises:
    var p_x = FP(unsafe_from_address=x)
    var p_res = FP(unsafe_from_address=res)
    var p_flag = IP(unsafe_from_address=flag)
    for t in range(n * d):
        kapprox_achi2_item(t, p_x, p_res, p_flag, n, d, steps, interval)


def op_kapprox_skew_fit(w: Int, off: Int, d: Int, nc: Int, seed: Int) raises:
    var p_w = FP(unsafe_from_address=w)
    var p_off = FP(unsafe_from_address=off)
    for t in range(d * nc + nc):
        kapprox_skew_fit_item(t, p_w, p_off, d, nc, seed)


def op_kapprox_skew_transform(x: Int, w: Int, off: Int, res: Int, flag: Int, n: Int, d: Int, nc: Int, skew: Float32) raises:
    var s_lx = List[Float32](length=(n * d) if (n * d) > 0 else 1, fill=Float32(0))
    var p_x = FP(unsafe_from_address=x)
    var p_w = FP(unsafe_from_address=w)
    var p_off = FP(unsafe_from_address=off)
    var p_res = FP(unsafe_from_address=res)
    var p_flag = IP(unsafe_from_address=flag)
    var p_lx = FP(unsafe_from_address=Int(s_lx.unsafe_ptr()))
    for t in range(n * d):
        kapprox_skew_log_item(t, p_x, p_lx, p_flag, skew)
    for t in range(n * nc):
        skew_transform_item(t, p_lx, p_w, p_off, p_res, n, d, nc)
    _ = s_lx^
