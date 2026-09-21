# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
from std.math import isfinite
from max.gpu.host import DeviceContext
from metrics.checks.device_io import download_f32, upload_f32
from core.device_scan import device_first_nonfinite
from preprocessing.minmax import minmax_fit, minmax_transform, minmax_transform_into
from preprocessing.standard import standard_fit, standard_fit_transform_into, standard_transform, standard_transform_into


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


def standard_fit_transform_host_into[stats_origin: MutOrigin, out_origin: MutOrigin](
    mut x: List[Float32], stats_output: MutPointer[Float32, stats_origin],
    output: MutPointer[Float32, out_origin], n: Int, d: Int,
    with_mean: Int, with_std: Int,
) raises:
    """One-copy/one-upload StandardScaler fit followed by transform."""
    validate_standard(n,d,with_mean,with_std)
    if len(x) < n*d:
        raise Error("StandardScaler: short input")
    standard_finite(x)
    var ctx = DeviceContext()
    var dx = upload_f32(ctx,x)
    var dstats = ctx.enqueue_create_buffer[DType.float32](3*d)
    var dout = ctx.enqueue_create_buffer[DType.float32](n*d)
    standard_fit_transform_into(ctx,dx,dstats,dout,n,d,with_mean,with_std)
    if device_first_nonfinite(ctx,dout,n*d) >= 0:
        raise Error("StandardScaler: nonfinite input or Float32 arithmetic overflow")
    var result = download_f32(ctx,dstats,3*d)
    standard_finite(result)
    for c in range(d):
        if result[d+c] < 0 or result[2*d+c] <= 0:
            raise Error("StandardScaler: invalid variance or scale")
    for i in range(3*d):
        stats_output.unsafe_store(i,result[i])
    ctx.enqueue_copy(dst_ptr=output,src_buf=dout)
    ctx.synchronize()
    _ = dout^; _ = dstats^; _ = dx^; _ = ctx^


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


def standard_transform_host_into[out_origin: MutOrigin, //](
    mut x: List[Float32], mut mean: List[Float32], mut scale: List[Float32],
    output: MutPointer[Float32, out_origin], n: Int, d: Int, inverse: Int,
    with_mean: Int, with_std: Int,
) raises:
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
    var dout = ctx.enqueue_create_buffer[DType.float32](n*d)
    standard_transform_into(ctx,dx,dm,ds,dout,n,d,inverse,with_mean,with_std)
    if device_first_nonfinite(ctx,dout,n*d) >= 0:
        raise Error("StandardScaler: nonfinite input or Float32 arithmetic overflow")
    ctx.enqueue_copy(dst_ptr=output,src_buf=dout)
    ctx.synchronize()
    _ = dout^; _ = ds^; _ = dm^; _ = dx^; _ = ctx^


def minmax_transform_host_into[out_origin: MutOrigin, //](
    mut x: List[Float32], mut scale: List[Float32], mut offset: List[Float32],
    output: MutPointer[Float32, out_origin], n: Int, d: Int, inverse: Int,
    clip: Int, lower: Float32, upper: Float32,
) raises:
    validate_dimensions(n,d,lower,upper)
    if len(x) < n*d or len(scale) < d or len(offset) < d or inverse < 0 or inverse > 1 or clip < 0 or clip > 1:
        raise Error("MinMaxScaler: invalid transform parameters")
    finite_values(x); finite_values(scale); finite_values(offset)
    for c in range(d):
        if scale[c] <= 0:
            raise Error("MinMaxScaler: scale must be positive")
    var ctx = DeviceContext()
    var dx = upload_f32(ctx,x)
    var ds = upload_f32(ctx,scale)
    var dm = upload_f32(ctx,offset)
    var dout = ctx.enqueue_create_buffer[DType.float32](n*d)
    minmax_transform_into(ctx,dx,ds,dm,dout,n,d,inverse,clip,lower,upper)
    if device_first_nonfinite(ctx,dout,n*d) >= 0:
        raise Error("MinMaxScaler: nonfinite input or Float32 arithmetic overflow")
    ctx.enqueue_copy(dst_ptr=output,src_buf=dout)
    ctx.synchronize()
    _ = dout^; _ = dm^; _ = ds^; _ = dx^; _ = ctx^
