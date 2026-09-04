# SPDX-License-Identifier: Apache-2.0
"""Implemented tail of the Mamba-2 block backward.

This module deliberately claims only the residual/output-projection seam:

    residual.out = input.x + out_proj(gnorm.out)

Given ``d_residual``, it computes ``d_gnorm`` and ``d_out_proj_weight`` with
the certified GEMM-v1 backward routes. The residual contribution to ``d_x``
is exactly ``d_residual`` and remains caller-owned; the incomplete upstream
path is not silently added here. No SSD, gated norm, convolution, input
projection, or block-norm derivative is claimed.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from mamba.checks.mamba2_backward import (
    PROJ2_OUT,
    mamba2_backward_proj_a_into,
    mamba2_backward_proj_a_workspace_max_floats,
    mamba2_backward_proj_b_into,
    mamba2_backward_proj_b_workspace_max_floats,
)
from mamba.checks.mamba2_fixture import Mamba2Dims


struct Mamba2BackwardTail(Movable):
    """Caller-owned outputs for the implemented tail seam."""

    var d_gnorm: DeviceBuffer[DType.float32]  # [M, d_inner]
    var d_w_out: DeviceBuffer[DType.float32]  # [d_model, d_inner]
    var workspace: DeviceBuffer[DType.float32]

    def __init__(
        out self, ctx: DeviceContext, dims: Mamba2Dims, m: Int
    ) raises:
        var ng = m * dims.d_inner
        if ng < 1:
            ng = 1
        var nw = dims.d_model * dims.d_inner
        if nw < 1:
            nw = 1
        self.d_gnorm = ctx.enqueue_create_buffer[DType.float32](ng)
        self.d_w_out = ctx.enqueue_create_buffer[DType.float32](nw)
        var wa = mamba2_backward_proj_a_workspace_max_floats(
            PROJ2_OUT, dims, m
        )
        var wb = mamba2_backward_proj_b_workspace_max_floats(
            PROJ2_OUT, dims, m
        )
        var ws = wa
        if wb > ws:
            ws = wb
        if ws < 1:
            ws = 1
        self.workspace = ctx.enqueue_create_buffer[DType.float32](ws)


def mamba2_backward_tail_into(
    ctx: DeviceContext,
    mut out: Mamba2BackwardTail,
    mut d_residual: DeviceBuffer[DType.float32],
    mut gnorm_out: DeviceBuffer[DType.float32],
    mut w_out: DeviceBuffer[DType.float32],
    dims: Mamba2Dims,
    m: Int,
) raises:
    """Enqueue the two exact derivatives of the output projection.

    ``out_proj`` is ``OP_NT`` at ``[M, d_model, d_inner]``. These launchers
    derive both transpose routes from that forward call, so this composer
    introduces no independent shape table or floating-point order.
    """
    mamba2_backward_proj_a_into(
        ctx,
        out.d_gnorm,
        d_residual,
        w_out,
        out.workspace,
        PROJ2_OUT,
        dims,
        m,
    )
    mamba2_backward_proj_b_into(
        ctx,
        out.d_w_out,
        d_residual,
        gnorm_out,
        out.workspace,
        PROJ2_OUT,
        dims,
        m,
    )
