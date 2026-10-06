# SPDX-License-Identifier: Apache-2.0
"""NN24 device launchers for the common scalar normalization profile.

Model RMS forward/backward, residual RMS and RMS/LayerNorm inference select
norm_profile_contract helpers coherently. Extended-options training still
refuses at the existing public API. Default OFF; no compilation, identity,
quality or timing has been run. Standalone centered-LN backward is an optional
component, not a claim of new public LayerNorm training support.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import (
    GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div,
    identical_mul, identical_mul_add, identical_rsqrt,
)

from transformer.experiments.norm_profile_contract import (
    NN24_NORM_LANES8, NN24_LANES, _merge, _sum, _square, norm_profile_dot,
    norm_profile_forward_row, norm_profile_backward_row, norm_profile_parameter_column,
    norm_profile_host_forward, norm_profile_host_backward,
)

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


