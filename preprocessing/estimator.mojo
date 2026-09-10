# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
from std.math import isfinite
from max.gpu.host import DeviceContext
from metrics.checks.device_io import upload_f32
from preprocessing.minmax import minmax_fit, minmax_transform


def validate_dimensions(n: Int, d: Int, lower: Float32, upper: Float32) raises:
    if n <= 0 or d <= 0 or d > 2147483647 or n > 2147483647 // d:
        raise Error("MinMaxScaler: positive dimensions with n*d<=Int32.max required")
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise Error("MinMaxScaler: finite increasing Float32 feature range required")


def finite_values(values: List[Float32]) raises:
    for value in values:
        if not isfinite(value):
            raise Error("MinMaxScaler: nonfinite input or Float32 arithmetic overflow")


def minmax_fit_host(x: List[Float32], n: Int, d: Int, lower: Float32, upper: Float32) raises -> List[Float32]:
    validate_dimensions(n,d,lower,upper)
    if len(x) < n*d:
        raise Error("MinMaxScaler: short input")
    finite_values(x)
    var ctx = DeviceContext()
    var dx = upload_f32(ctx,x)
    var result = minmax_fit(ctx,dx,n,d,lower,upper)
    _ = dx^
    _ = ctx^
    finite_values(result)
    for c in range(d):
        if result[3*d+c] <= 0:
            raise Error("MinMaxScaler: Float32 scale underflow")
    return result^


def minmax_transform_host(
    x: List[Float32], scale: List[Float32], offset: List[Float32], n: Int, d: Int,
    inverse: Int, clip: Int, lower: Float32, upper: Float32,
) raises -> List[Float32]:
    validate_dimensions(n,d,lower,upper)
    if len(x) < n*d or len(scale) < d or len(offset) < d or inverse < 0 or inverse > 1 or clip < 0 or clip > 1:
        raise Error("MinMaxScaler: invalid transform parameters")
    finite_values(x)
    finite_values(scale)
    finite_values(offset)
    for c in range(d):
        if scale[c] <= 0:
            raise Error("MinMaxScaler: scale must be positive")
    var ctx = DeviceContext()
    var dx = upload_f32(ctx,x)
    var ds = upload_f32(ctx,scale)
    var dm = upload_f32(ctx,offset)
    var result = minmax_transform(ctx,dx,ds,dm,n,d,inverse,clip,lower,upper)
    _ = dm^
    _ = ds^
    _ = dx^
    _ = ctx^
    finite_values(result)
    return result^
