# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
from std.math import isfinite
from max.gpu.host import DeviceContext
from metrics.checks.device_io import upload_f32
from preprocessing.minmax import minmax_fit, minmax_transform
from preprocessing.standard import standard_fit, standard_transform


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


def validate_standard(n: Int, d: Int, with_mean: Int, with_std: Int) raises:
    if n <= 0 or d <= 0 or d > 2147483647 or n > 2147483647 // d:
        raise Error("StandardScaler: positive dimensions with n*d<=Int32.max required")
    if with_mean < 0 or with_mean > 1 or with_std < 0 or with_std > 1:
        raise Error("StandardScaler: flags must be 0 or 1")


def standard_finite(values: List[Float32]) raises:
    for value in values:
        if not isfinite(value):
            raise Error("StandardScaler: nonfinite input or Float32 arithmetic overflow")


def standard_fit_host(x: List[Float32], n: Int, d: Int, with_mean: Int, with_std: Int) raises -> List[Float32]:
    validate_standard(n,d,with_mean,with_std)
    if len(x) < n*d:
        raise Error("StandardScaler: short input")
    standard_finite(x)
    var ctx = DeviceContext()
    var dx = upload_f32(ctx,x)
    var result = standard_fit(ctx,dx,n,d,with_mean,with_std)
    _ = dx^
    _ = ctx^
    standard_finite(result)
    for c in range(d):
        if result[d+c] < 0 or result[2*d+c] <= 0:
            raise Error("StandardScaler: invalid variance or scale")
    return result^


def standard_transform_host(
    x: List[Float32], mean: List[Float32], scale: List[Float32], n: Int, d: Int,
    inverse: Int, with_mean: Int, with_std: Int,
) raises -> List[Float32]:
    validate_standard(n,d,with_mean,with_std)
    if len(x) < n*d or len(mean) < d or len(scale) < d or inverse < 0 or inverse > 1:
        raise Error("StandardScaler: invalid transform parameters")
    standard_finite(x)
    if with_mean != 0:
        standard_finite(mean)
    if with_std != 0:
        standard_finite(scale)
        for c in range(d):
            if scale[c] <= 0:
                raise Error("StandardScaler: scale must be positive")
    var ctx = DeviceContext()
    var dx = upload_f32(ctx,x)
    var dm = upload_f32(ctx,mean)
    var ds = upload_f32(ctx,scale)
    var result = standard_transform(ctx,dx,dm,ds,n,d,inverse,with_mean,with_std)
    _ = ds^
    _ = dm^
    _ = dx^
    _ = ctx^
    standard_finite(result)
    return result^
