# SPDX-License-Identifier: Apache-2.0
"""Compile-only probe for the first Mamba-2 SSD backward kernel."""

from max.gpu.host import DeviceContext

from mamba.impl.mamba_ssm.ops.mamba2_ssd_backward import (
    Mamba2SSDBackwardState,
    Mamba2SSDScaleReduction,
    Mamba2SSDDiscretizeBackward,
    mamba2_reduce_scale_product_into,
    mamba2_reverse_cumsum_and_da_into,
    mamba2_ydiag_xd_and_partial_dt_into,
    mamba2_reverse_chunk_state_into,
    mamba2_s18_direct_dpass_into,
)


def main() raises:
    # Compile-shape witness: B=2, T=513, H=1, C=3, Q=256, P=64, N=128.
    # Every mutable input is a distinct allocation so borrow checking here
    # represents the production launcher rather than an artificial alias.
    var ctx = DeviceContext()
    var out = Mamba2SSDBackwardState(ctx, 2, 3, 1)
    var reduction = Mamba2SSDScaleReduction(ctx, 2, 3, 1, 256)
    var discretize = Mamba2SSDDiscretizeBackward(ctx, 2, 513, 1)
    var d_y = ctx.enqueue_create_buffer[DType.float32](2 * 513 * 1 * 64)
    var cb_g = ctx.enqueue_create_buffer[DType.float32](2 * 3 * 256 * 256)
    var seg_l = ctx.enqueue_create_buffer[DType.float32](2 * 3 * 1 * 256 * 256)
    var xbc = ctx.enqueue_create_buffer[DType.float32](2 * 513 * 40)
    var dt = ctx.enqueue_create_buffer[DType.float32](2 * 513 * 1)
    var dtraw = ctx.enqueue_create_buffer[DType.float32](2 * 513 * 1)
    var dt_bias = ctx.enqueue_create_buffer[DType.float32](1)
    var dacs = ctx.enqueue_create_buffer[DType.float32](2 * 1 * 3 * 256)
    var pass_states = ctx.enqueue_create_buffer[DType.float32](
        2 * 3 * 1 * 64 * 128
    )
    var d_final = ctx.enqueue_create_buffer[DType.float32](2 * 1 * 64 * 128)
    mamba2_s18_direct_dpass_into(
        ctx, out, d_y, xbc, pass_states, dacs, 2, 513, 1, 32, 40, 3, 256
    )
    mamba2_reverse_chunk_state_into(
        ctx, out, d_final, pass_states, dacs, 2, 1, 3, 256
    )
    mamba2_reduce_scale_product_into(
        ctx, reduction, out, dacs, 2, 3, 1, 256
    )
    mamba2_reverse_cumsum_and_da_into(
        ctx, discretize, reduction.d_dacs_total, dt, dt_bias,
        2, 513, 1, 3, 256
    )
    mamba2_ydiag_xd_and_partial_dt_into(
        ctx, discretize, d_y, cb_g, seg_l, xbc, dt, dtraw, dt_bias,
        2, 513, 1, 32, 40, 3, 256, 0.0, 1.0
    )
    print("mamba2 SSD backward reverse-chunk recurrence compile probe")
