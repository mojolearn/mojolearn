# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""A1: independent Float64 reference plus bitwise fixed-tree/launch checks."""
from std.math import sqrt, isfinite
from std.memory import bitcast
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div, portable_sqrtf, numeric_mode_name
from metrics.checks.device_io import upload_f32
from metrics.impl.regression_errors import regression_error
from metrics.estimator import regression_error_host


def close(got: Float32, want: Float64) raises:
    if not isfinite(got) or abs(Float64(got) - want) > 2e-5 * max(abs(want), 1.0):
        raise Error("regression error differs from independent Float64 reference")


def model[absolute: Bool](y: List[Float32], p: List[Float32]) -> Float32:
    # Independent host spelling of the specified 256-slot halving tree.
    var total = Float32(0)
    for base in range(0, len(y), 256):
        var values = List[Float32]()
        for j in range(256):
            var term = Float32(0)
            if base+j < len(y):
                var d = ftz(ftz(y[base+j]) - ftz(p[base+j]))
                comptime if absolute:
                    term = abs(d)
                else:
                    term = ftz(d*d)
            values.append(term)
        var distance = 128
        while distance:
            for j in range(distance):
                values[j] = ftz(values[j] + values[j+distance])
            distance //= 2
        total = ftz(total + values[0])
    return ftz(identical_div(total, Float32(len(y))))


def exercise[absolute: Bool, root: Bool](ctx: DeviceContext, y: List[Float32], p: List[Float32]) raises:
    var padded_y = y.copy()
    var padded_p = p.copy()
    for _ in range(37):
        padded_y.append(bitcast[DType.float32](UInt32(0x7fc00000)))
        padded_p.append(bitcast[DType.float32](UInt32(0x7fc00000)))
    var dy = upload_f32(ctx, padded_y)
    var dp = upload_f32(ctx, padded_p)
    var got = regression_error[absolute, root](ctx, dy, dp, len(y))
    var again = regression_error[absolute, root](ctx, dy, dp, len(y))
    if bitcast[DType.uint32](got) != bitcast[DType.uint32](again):
        raise Error("repeat changed regression metric bits")
    var reference = Float64(0)
    for j in range(len(y)):
        var d = Float64(y[j]) - Float64(p[j])
        comptime if absolute:
            reference += abs(d)
        else:
            reference += d*d
    reference /= Float64(len(y))
    comptime if root:
        reference = sqrt(reference)
    close(got, reference)
    var alternate = regression_error[absolute, root, 64](ctx, dy, dp, len(y), 3, 2)
    close(alternate, reference)
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var want = model[absolute](y,p)
        comptime if root:
            want = portable_sqrtf(want)
        if bitcast[DType.uint32](got) != bitcast[DType.uint32](want):
            raise Error("IDENTICAL differs from independently spelled tree")
        if bitcast[DType.uint32](got) != bitcast[DType.uint32](alternate):
            raise Error("IDENTICAL changed across block/grid shapes")
    print("metric", absolute, root, "n", len(y), "bits", bitcast[DType.uint32](got))
    _ = dy^
    _ = dp^


def main() raises:
    print("numeric_mode", numeric_mode_name())
    var ctx = DeviceContext()
    var y: List[Float32] = [1,-2,3,-4]
    var p: List[Float32] = [0,0,0,0]
    exercise[False,False](ctx,y,p)
    exercise[True,False](ctx,y,p)
    exercise[False,True](ctx,y,p)
    close(regression_error_host(y,p,4), 7.5)
    close(regression_error_host[True](y,p,4), 2.5)
    var large = List[Float32]()
    var pred = List[Float32]()
    for i in range(4099):
        large.append(Float32((i*137)%1021-510) / Float32(16))
        pred.append(Float32((i*73)%503-251) / Float32(32))
    large[4098] = 4096
    exercise[False,False](ctx,large,pred)
    exercise[True,False](ctx,large,pred)
    exercise[False,True](ctx,large,pred)
    # Perfect extreme inputs must subtract to zero rather than overflow.
    var huge: List[Float32] = [bitcast[DType.float32](UInt32(0x7f7fffff))]
    if bitcast[DType.uint32](regression_error_host(huge,huge,1)) != 0:
        raise Error("identical extreme inputs should have zero error")
    var opposite: List[Float32] = [-huge[0]]
    if bitcast[DType.uint32](regression_error_host(huge,opposite,1)) != 0x7f800000 or bitcast[DType.uint32](regression_error_host[True](huge,opposite,1)) != 0x7f800000 or bitcast[DType.uint32](regression_error_host[False,True](huge,opposite,1)) != 0x7f800000:
        raise Error("residual overflow must give positive infinity")
    var many = List[Float32]()
    var zeros = List[Float32]()
    for _ in range(1024):
        many.append(bitcast[DType.float32](UInt32(0x7b800000))) # 2^120
        zeros.append(0)
    if bitcast[DType.uint32](regression_error_host[True](many,zeros,1024)) != 0x7f800000:
        raise Error("Float32 partial sum overflow must remain infinity")
    var square_overflow: List[Float32] = [Float32(1e20)]
    var single_zero: List[Float32] = [Float32(0)]
    if bitcast[DType.uint32](regression_error_host(square_overflow,single_zero,1)) != 0x7f800000 or bitcast[DType.uint32](regression_error_host[False,True](square_overflow,single_zero,1)) != 0x7f800000:
        raise Error("square overflow must remain positive infinity for MSE/RMSE")
    var signed_zero: List[Float32] = [bitcast[DType.float32](UInt32(0x80000000))]
    if bitcast[DType.uint32](regression_error_host[True](signed_zero,single_zero,1)) != 0 or bitcast[DType.uint32](regression_error_host(signed_zero,single_zero,1)) != 0 or bitcast[DType.uint32](regression_error_host[False,True](signed_zero,single_zero,1)) != 0:
        raise Error("signed zero residual must produce positive zero")
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var sub: List[Float32] = [bitcast[DType.float32](UInt32(0x007fffff))]
        var neg: List[Float32] = [-sub[0]]
        if bitcast[DType.uint32](regression_error_host[True](sub,neg,1)) != 0:
            raise Error("operand FTZ seam must produce positive zero")
        var tiny: List[Float32] = [Float32(1e-20)]
        var zero: List[Float32] = [Float32(0)]
        if bitcast[DType.uint32](regression_error_host(tiny,zero,1)) != 0:
            raise Error("squared residual FTZ seam must produce positive zero")
    var rejected = False
    try:
        _ = regression_error_host(y,p,0)
    except:
        rejected = True
    if not rejected:
        raise Error("empty input accepted")
    rejected = False
    try:
        var bad: List[Float32] = [bitcast[DType.float32](UInt32(0x7f800000))]
        _ = regression_error_host(bad,huge,1)
    except:
        rejected = True
    if not rejected:
        raise Error("nonfinite input accepted")
    print("PASS regression errors: numeric, repeat, shapes, tails, overflow and validation")
    _ = ctx^
