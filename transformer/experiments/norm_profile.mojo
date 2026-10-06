# SPDX-License-Identifier: Apache-2.0
"""NN24 fixed logical lanes, standalone full forward/backward component.

SOURCE DRAFT ONLY. The component's host and GPU entrypoints share arithmetic
helpers. It is deliberately not dispatched inside a model until every norm
caller, derivative and checkpoint profile has migrated together. Decode and
prefill use the same row helper; logical lanes depend on the version, never
vendor, subgroup, row count or device occupancy. Existing transformer and
sequence norm profiles remain unchanged. Eight lanes are an ILP candidate,
not a demonstrated optimum. No compilation/identity/quality/timing was run.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import (
    GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div,
    identical_mul, identical_mul_add, identical_rsqrt,
)

comptime NN24_NORM_LANES8 = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN24_NORM_LANES8"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NN24_LANES = 8 if NN24_NORM_LANES8 else 1


def _merge[LANES: Int](var sums: SIMD[DType.float32, LANES]) -> Float32:
    """Adjacent pairs, no artificial +0 terms at a row's tail."""
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
    out: MutPointer[Float32, MutAnyOrigin],
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
        out.unsafe_store(base + j, y)


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


def norm_profile_forward_kernel[LAYER: Bool](
    out: MutPointer[Float32, MutAnyOrigin], sums: MutPointer[Float32, MutAnyOrigin],
    means: MutPointer[Float32, MutAnyOrigin], rstds: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin], weight: MutPointer[Float32, MutAnyOrigin],
    bias: MutPointer[Float32, MutAnyOrigin], rows_in: Int32, width_in: Int32,
    eps: Float32, has_bias: Bool,
):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row < Int(rows_in):
        norm_profile_forward_row[NN24_LANES, LAYER](out, sums, means, rstds, x, weight, bias, row, Int(width_in), eps, has_bias)


def norm_profile_backward_kernel[LAYER: Bool](
    dx: MutPointer[Float32, MutAnyOrigin], wp: MutPointer[Float32, MutAnyOrigin],
    bp: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin], weight: MutPointer[Float32, MutAnyOrigin],
    means: MutPointer[Float32, MutAnyOrigin], rstds: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32, width_in: Int32,
):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row < Int(rows_in):
        norm_profile_backward_row[NN24_LANES, LAYER](dx, wp, bp, x, dy, weight, means, rstds, row, Int(width_in))


def norm_profile_parameter_kernel(
    dw: MutPointer[Float32, MutAnyOrigin], db: MutPointer[Float32, MutAnyOrigin],
    wp: MutPointer[Float32, MutAnyOrigin], bp: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32, width_in: Int32,
):
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(width_in):
        norm_profile_parameter_column(dw, db, wp, bp, Int(rows_in), Int(width_in), j)


def norm_profile_host_forward[LAYER: Bool](
    out: MutPointer[Float32, MutAnyOrigin], sums: MutPointer[Float32, MutAnyOrigin],
    means: MutPointer[Float32, MutAnyOrigin], rstds: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin], weight: MutPointer[Float32, MutAnyOrigin],
    bias: MutPointer[Float32, MutAnyOrigin], rows: Int, width: Int,
    eps: Float32, has_bias: Bool,
):
    for row in range(rows):
        norm_profile_forward_row[NN24_LANES, LAYER](out, sums, means, rstds, x, weight, bias, row, width, eps, has_bias)


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


def enqueue_norm_profile_forward[LAYER: Bool](
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32], mut sums: DeviceBuffer[DType.float32],
    mut means: DeviceBuffer[DType.float32], mut rstds: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32], mut weight: DeviceBuffer[DType.float32],
    mut bias: DeviceBuffer[DType.float32], rows: Int, width: Int,
    eps: Float32, has_bias: Bool,
) raises:
    if rows <= 0 or width <= 0:
        raise Error("NN24: positive row count and width required")
    if len(out) < rows * width or len(x) < rows * width or len(weight) < width:
        raise Error("NN24: short forward operand")
    if len(sums) < rows or len(means) < rows or len(rstds) < rows:
        raise Error("NN24: short row state")
    if has_bias and len(bias) < width:
        raise Error("NN24: short bias")
    ctx.enqueue_function[norm_profile_forward_kernel[LAYER]](
        out.unsafe_ptr(), sums.unsafe_ptr(), means.unsafe_ptr(), rstds.unsafe_ptr(),
        x.unsafe_ptr(), weight.unsafe_ptr(), bias.unsafe_ptr(),
        Int32(rows), Int32(width), eps, has_bias,
        grid_dim=((rows + 63) // 64, 1, 1), block_dim=(64, 1, 1),
    )


def enqueue_norm_profile_backward[LAYER: Bool](
    ctx: DeviceContext,
    mut dx: DeviceBuffer[DType.float32], mut wp: DeviceBuffer[DType.float32],
    mut bp: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.float32],
    mut db: DeviceBuffer[DType.float32], mut x: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32], mut weight: DeviceBuffer[DType.float32],
    mut means: DeviceBuffer[DType.float32], mut rstds: DeviceBuffer[DType.float32],
    rows: Int, width: Int,
) raises:
    if rows <= 0 or width <= 0:
        raise Error("NN24: positive row count and width required")
    if len(dx) < rows * width or len(wp) < rows * width or len(bp) < rows * width or len(x) < rows * width or len(dy) < rows * width:
        raise Error("NN24: short backward cell buffer")
    if len(dw) < width or len(db) < width or len(weight) < width or len(means) < rows or len(rstds) < rows:
        raise Error("NN24: short backward row/parameter buffer")
    ctx.enqueue_function[norm_profile_backward_kernel[LAYER]](
        dx.unsafe_ptr(), wp.unsafe_ptr(), bp.unsafe_ptr(), x.unsafe_ptr(),
        dy.unsafe_ptr(), weight.unsafe_ptr(), means.unsafe_ptr(), rstds.unsafe_ptr(),
        Int32(rows), Int32(width), grid_dim=((rows + 63) // 64, 1, 1), block_dim=(64, 1, 1),
    )
    # In-order queue establishes completion of all row products before the
    # parameter kernel. Caller owns every buffer through synchronization.
    ctx.enqueue_function[norm_profile_parameter_kernel](
        dw.unsafe_ptr(), db.unsafe_ptr(), wp.unsafe_ptr(), bp.unsafe_ptr(),
        Int32(rows), Int32(width), grid_dim=((width + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
