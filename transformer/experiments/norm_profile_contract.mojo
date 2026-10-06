# SPDX-License-Identifier: Apache-2.0
"""Pure host/device NN24 scalar contract; no device runtime dependencies."""
from std.sys.compile import is_defined
from checks.numerics import (GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div, identical_mul, identical_mul_add, identical_rsqrt)

comptime NN24_NORM_LANES8 = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN24_NORM_LANES8"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NN24_LANES = 8 if NN24_NORM_LANES8 else 1


def _merge[LANES: Int](var sums: SIMD[DType.float32, LANES]) -> Float32:
    """Fixed adjacent-pair tree; absent logical lanes contribute their +0 seed."""
    comptime assert LANES == 1 or LANES == 8
    comptime for level in range(3 if LANES == 8 else 0):
        comptime for lane in range(LANES >> (level + 1)):
            sums[lane] = ftz(sums[2 * lane] + sums[2 * lane + 1])
    return sums[0]


def _sum[LANES: Int](
    x: MutPointer[Float32, MutAnyOrigin], base: Int, width: Int,
) -> Float32:
    var sums = SIMD[DType.float32, LANES](0.0)
    # Round-robin logical membership i%LANES is part of the new profile.
    comptime for lane in range(LANES):
        var j = lane
        while j < width:
            sums[lane] = ftz(ftz(sums[lane]) + ftz(x.unsafe_load(base + j)))
            j += LANES
    return _merge[LANES](sums)


def _square[LANES: Int](
    x: MutPointer[Float32, MutAnyOrigin], base: Int, width: Int,
    mean: Float32,
) -> Float32:
    var sums = SIMD[DType.float32, LANES](0.0)
    comptime for lane in range(LANES):
        var j = lane
        while j < width:
            var dev = ftz(ftz(x.unsafe_load(base + j)) - mean)
            sums[lane] = ftz(identical_mul_add(dev, dev, sums[lane]))
            j += LANES
    return _merge[LANES](sums)


def norm_profile_forward_row[LANES: Int, LAYER: Bool](
    output: MutPointer[Float32, MutAnyOrigin],
    sums: MutPointer[Float32, MutAnyOrigin],
    means: MutPointer[Float32, MutAnyOrigin],
    rstds: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    bias: MutPointer[Float32, MutAnyOrigin],
    row: Int, width: Int, eps: Float32, has_bias: Bool,
):
    """Host-callable and device-callable, same graph; centered variance."""
    var base = row * width
    var mean = Float32(0.0)
    comptime if LAYER:
        mean = ftz(identical_div(_sum[LANES](x, base, width), Float32(width)))
    var ss = _square[LANES](x, base, width, mean)
    var variance = ftz(identical_div(ss, Float32(width)))
    var rstd = ftz(identical_rsqrt(ftz(variance + eps)))
    sums.unsafe_store(row, ss)
    means.unsafe_store(row, mean)
    rstds.unsafe_store(row, rstd)
    for j in range(width):
        var dev = ftz(ftz(x.unsafe_load(base + j)) - mean)
        var y = ftz(identical_mul(ftz(weight.unsafe_load(j)), ftz(identical_mul(dev, rstd))))
        if has_bias:
            y = ftz(ftz(y) + ftz(bias.unsafe_load(j)))
        output.unsafe_store(base + j, y)


def norm_profile_backward_row[LANES: Int, LAYER: Bool](
    dx: MutPointer[Float32, MutAnyOrigin],
    weight_products: MutPointer[Float32, MutAnyOrigin],
    bias_products: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    means: MutPointer[Float32, MutAnyOrigin],
    rstds: MutPointer[Float32, MutAnyOrigin],
    row: Int, width: Int,
):
    """Defined derivative graph for NN24; rounded FP operations are not AD'd.

    c=sum(dh*dev), dvariance=(-.5*c*rstd^3)/width. RMS uses
    dh*rstd + (2*dev)*dvariance. LayerNorm additionally subtracts the
    canonical mean of these centered-input gradients. This intentionally
    preserves the centered-variance definition rather than raw moments.
    Parameter products are emitted; their across-row reduction stays an
    explicit second kernel with a fixed ascending row chain.
    """
    var base = row * width
    var mean = means.unsafe_load(row)
    var rstd = rstds.unsafe_load(row)
    var cs = SIMD[DType.float32, LANES](0.0)
    comptime for lane in range(LANES):
        var j = lane
        while j < width:
            var dev = ftz(ftz(x.unsafe_load(base + j)) - mean)
            var dh = ftz(identical_mul(ftz(dy.unsafe_load(base + j)), ftz(weight.unsafe_load(j))))
            cs[lane] = ftz(identical_mul_add(dh, dev, cs[lane]))
            j += LANES
    var c = _merge[LANES](cs)
    var r2 = ftz(identical_mul(rstd, rstd))
    var r3 = ftz(identical_mul(r2, rstd))
    var da = ftz(identical_mul(Float32(-0.5), ftz(identical_mul(c, r3))))
    var dv = ftz(identical_div(da, Float32(width)))
    for j in range(width):
        var dev = ftz(ftz(x.unsafe_load(base + j)) - mean)
        var gy = ftz(dy.unsafe_load(base + j))
        var dh = ftz(identical_mul(gy, ftz(weight.unsafe_load(j))))
        var direct = ftz(identical_mul(dh, rstd))
        var square = ftz(identical_mul(ftz(identical_mul(Float32(2.0), dev)), dv))
        dx.unsafe_store(base + j, ftz(direct + square))
        weight_products.unsafe_store(base + j, ftz(identical_mul(gy, ftz(identical_mul(dev, rstd)))))
        bias_products.unsafe_store(base + j, gy)
    comptime if LAYER:
        var dmean = ftz(identical_div(_sum[LANES](dx, base, width), Float32(width)))
        for j in range(width):
            dx.unsafe_store(base + j, ftz(ftz(dx.unsafe_load(base + j)) - dmean))


def norm_profile_parameter_column(
    dw: MutPointer[Float32, MutAnyOrigin], db: MutPointer[Float32, MutAnyOrigin],
    wp: MutPointer[Float32, MutAnyOrigin], bp: MutPointer[Float32, MutAnyOrigin],
    rows: Int, width: Int, j: Int,
):
    var aw = Float32(0.0)
    var ab = Float32(0.0)
    for row in range(rows):
        aw = ftz(ftz(aw) + ftz(wp.unsafe_load(row * width + j)))
        ab = ftz(ftz(ab) + ftz(bp.unsafe_load(row * width + j)))
    dw.unsafe_store(j, aw)
    db.unsafe_store(j, ab)


def norm_profile_host_forward[LAYER: Bool](
    output: MutPointer[Float32, MutAnyOrigin], sums: MutPointer[Float32, MutAnyOrigin],
    means: MutPointer[Float32, MutAnyOrigin], rstds: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin], weight: MutPointer[Float32, MutAnyOrigin],
    bias: MutPointer[Float32, MutAnyOrigin], rows: Int, width: Int,
    eps: Float32, has_bias: Bool,
):
    for row in range(rows):
        norm_profile_forward_row[NN24_LANES, LAYER](output, sums, means, rstds, x, weight, bias, row, width, eps, has_bias)


def norm_profile_host_backward[LAYER: Bool](
    dx: MutPointer[Float32, MutAnyOrigin], wp: MutPointer[Float32, MutAnyOrigin],
    bp: MutPointer[Float32, MutAnyOrigin], dw: MutPointer[Float32, MutAnyOrigin],
    db: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin], weight: MutPointer[Float32, MutAnyOrigin],
    means: MutPointer[Float32, MutAnyOrigin], rstds: MutPointer[Float32, MutAnyOrigin],
    rows: Int, width: Int,
):
    for row in range(rows):
        norm_profile_backward_row[NN24_LANES, LAYER](dx, wp, bp, x, dy, weight, means, rstds, row, width)
    for j in range(width):
        norm_profile_parameter_column(dw, db, wp, bp, rows, width, j)


def norm_profile_dot[LANES: Int](
    a: MutPointer[Float32, MutAnyOrigin], b: MutPointer[Float32, MutAnyOrigin],
    a_base: Int, b_base: Int, width: Int,
) -> Float32:
    """NN24 same logical lane membership for every forward/backward column."""
    var sums = SIMD[DType.float32, LANES](0.0)
    comptime for lane in range(LANES):
        var j = lane
        while j < width:
            sums[lane] = ftz(identical_mul_add(ftz(a.unsafe_load(a_base + j)),
                ftz(b.unsafe_load(b_base + j)), sums[lane]))
            j += LANES
    return _merge[LANES](sums)
