# SPDX-License-Identifier: Apache-2.0
"""Synchronous Mamba-2 zero-state IDENTICAL prefill VJP.

Launch order extracted from checks/mamba2_backward_tail_dump.mojo.
The original diagnostic driver remains an independent byte-comparison gate.
Python/native API qualification is required separately from historical dumps.
No incoming cache or final-state cotangent is accepted.
"""
from std.memory import bitcast

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from mamba.checks.mamba2_fixture import (
    M2_D_STATE,
    M2_HEADDIM,
)
from mamba.impl.modules.mamba2 import (
    Mamba2DeviceStages,
    Mamba2DeviceWeights,
    allocate_inference_cache,
    mamba2_block_forward,
)
from mamba.impl.modules.mamba2_backward import (
    Mamba2BackwardTail,
    mamba2_backward_d_skip_into,
    mamba2_backward_gnorm_into,
    mamba2_backward_silu_gate_into,
    mamba2_backward_tail_into,
    mamba2_backward_input_projection_into,
    mamba2_backward_block_norm_into,
)
from mamba.impl.modules.ssd_minimal import m2_q_eff
from mamba.impl.ops.mamba2_ssd_backward import (
    Mamba2SSDBackwardState,
    Mamba2SSDDiscretizeBackward,
    Mamba2ConvBackward,
    Mamba2SSDScaleReduction,
    mamba2_reduce_scale_product_into,
    mamba2_reverse_cumsum_and_da_into,
    mamba2_ydiag_xd_and_partial_dt_into,
    mamba2_reverse_chunk_state_into,
    mamba2_cstate_ddecay_into,
    mamba2_postconv_merge_into,
    mamba2_conv_backward_prefill_into,
    mamba2_s18_direct_dpass_into,
)
from mamba.impl.modeling.modeling_mamba import (
    mamba_download,
    mamba_upload,
    mamba_zeros,
)

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from mamba.checks.mamba2_fixture import Mamba2Weights


struct Mamba2PrefillGradients(Movable):
    var x: List[Float32]
    var block_norm_weight: List[Float32]
    var in_proj_weight: List[Float32]
    var conv1d_weight: List[Float32]
    var conv1d_bias: List[Float32]
    var dt_bias: List[Float32]
    var A_log: List[Float32]
    var D: List[Float32]
    var norm_weight: List[Float32]
    var out_proj_weight: List[Float32]

    def __init__(out self):
        self.x = List[Float32]()
        self.block_norm_weight = List[Float32]()
        self.in_proj_weight = List[Float32]()
        self.conv1d_weight = List[Float32]()
        self.conv1d_bias = List[Float32]()
        self.dt_bias = List[Float32]()
        self.A_log = List[Float32]()
        self.D = List[Float32]()
        self.norm_weight = List[Float32]()
        self.out_proj_weight = List[Float32]()


def mamba2_prefill_backward(
    weights: Mamba2Weights,
    input: List[Float32],
    grad_output: List[Float32],
    b: Int,
    l: Int,
    dt_lo: Float32,
    dt_hi: Float32,
) raises -> Mamba2PrefillGradients:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("mamba2 backward: only IDENTICAL zero-state prefill is implemented")
    if b <= 0 or l <= 0:
        raise Error("mamba2 backward: B and L must be positive")
    if len(input) != b * l * weights.dims.d_model or len(grad_output) != len(input):
        raise Error("mamba2 backward: input and grad_output lengths must equal B*L*d_model")
    for i in range(len(grad_output)):
        var bits = bitcast[DType.uint32](grad_output[i])
        if (bits & UInt32(0x7f800000)) == UInt32(0x7f800000):
            raise Error("mamba2 backward: non-finite grad_output at flat index " + String(i))
    var dims = weights.dims.copy()
    var m = b * l
    var ctx = DeviceContext()
    var dweights = Mamba2DeviceWeights(ctx, weights)
    var state = allocate_inference_cache(ctx, b, dims)
    var stages = Mamba2DeviceStages(ctx, b, l, 0, dims)
    var x = mamba_upload(ctx, input)
    var trace = IdentityTrace.disabled()
    mamba2_block_forward(
        ctx,
        stages,
        state,
        dweights,
        x,
        b,
        l,
        dt_lo,
        dt_hi,
        trace,
        String("m2.backward.tail"),
    )

    var d_residual = mamba_upload(
        ctx, grad_output
    )
    var tail = Mamba2BackwardTail(ctx, dims, m)
    mamba2_backward_tail_into(
        ctx,
        tail,
        d_residual,
        stages.gnorm_out,
        dweights.w_out,
        dims,
        m,
    )
    mamba2_backward_gnorm_into(
        ctx,
        tail,
        stages.gnorm_gate,
        stages.gnorm_sumsq,
        dweights.gnorm_w,
        dims,
        m,
    )
    mamba2_backward_silu_gate_into(
        ctx, tail, stages.skip_out, stages.in_proj, dims, m
    )
    mamba2_backward_d_skip_into(
        ctx, tail, stages.silu_out, dweights.d_skip, dims, m
    )
    var ssd_backward = Mamba2SSDBackwardState(
        ctx, b, stages.nc, dims.nheads
    )
    mamba2_s18_direct_dpass_into(
        ctx,
        ssd_backward,
        tail.d_scan,
        stages.xbc_work,
        stages.pass_states,
        stages.dacs,
        b,
        stages.t_work,
        dims.nheads,
        dims.d_inner,
        dims.conv_dim(),
        stages.nc,
        m2_q_eff(),
    )
    # The scalar objective depends only on residual.out, not h_last, so the
    # final-state cotangent is explicitly zero at this partial boundary.
    var d_final = mamba_zeros(
        ctx, b * dims.nheads * M2_HEADDIM * M2_D_STATE
    )
    mamba2_reverse_chunk_state_into(
        ctx,
        ssd_backward,
        d_final,
        stages.pass_states,
        stages.dacs,
        b,
        dims.nheads,
        stages.nc,
        m2_q_eff(),
    )
    mamba2_cstate_ddecay_into(
        ctx, ssd_backward, stages.xd_work, stages.xbc_work, b,
        stages.t_work, dims.nheads, dims.d_inner, dims.conv_dim(),
        stages.nc, m2_q_eff(),
    )
    var scale_reduction = Mamba2SSDScaleReduction(
        ctx, b, stages.nc, dims.nheads, m2_q_eff()
    )
    mamba2_reduce_scale_product_into(
        ctx,
        scale_reduction,
        ssd_backward,
        stages.dacs,
        stages.decay,
        b,
        stages.nc,
        dims.nheads,
        m2_q_eff(),
    )
    var discretize_backward = Mamba2SSDDiscretizeBackward(
        ctx, b, stages.t_work, dims.nheads
    )
    mamba2_reverse_cumsum_and_da_into(
        ctx, discretize_backward, scale_reduction.d_dacs_total,
        stages.dt_work, stages.a_out, b, stages.t_work,
        dims.nheads, stages.nc, m2_q_eff(),
    )
    mamba2_ydiag_xd_and_partial_dt_into(
        ctx, discretize_backward, tail.d_scan, stages.cb_g, stages.seg_l,
        stages.xd_work,
        stages.xbc_work, stages.dt_work, stages.a_out, stages.dtraw_work, dweights.dt_bias,
        ssd_backward.d_cstate, stages.decay,
        b, stages.t_work, dims.nheads, dims.d_inner,
        dims.conv_dim(), stages.nc, m2_q_eff(), dt_lo, dt_hi,
    )
    mamba2_postconv_merge_into(
        ctx, discretize_backward, ssd_backward, tail.d_x_from_d,
        b, stages.t_work, dims.nheads,
    )
    var conv_backward = Mamba2ConvBackward(ctx, b, l, dims.conv_dim())
    mamba2_conv_backward_prefill_into(
        ctx, conv_backward, discretize_backward, stages.conv_out,
        stages.in_proj, dweights.conv_w, b, l, dims.d_inner,
        dims.conv_dim(), dims.d_in_proj(), stages.q0,
    )
    mamba2_backward_input_projection_into(
        ctx, tail, conv_backward.d_in_xbc, discretize_backward.d_dtraw,
        stages.norm_out, dweights.w_in, dims, m,
    )
    mamba2_backward_block_norm_into(
        ctx, tail, d_residual, x, stages.norm_sumsq,
        dweights.norm_w, dims, m,
    )
    ctx.synchronize()

    var result = Mamba2PrefillGradients()
    result.x = mamba_download(ctx, tail.d_block_x, m*dims.d_model)
    result.block_norm_weight = mamba_download(ctx, tail.d_block_w, dims.d_model)
    result.in_proj_weight = mamba_download(ctx, tail.d_w_in, dims.d_in_proj()*dims.d_model)
    result.conv1d_weight = mamba_download(ctx, conv_backward.d_w, dims.conv_dim()*4)
    result.conv1d_bias = mamba_download(ctx, conv_backward.d_b, dims.conv_dim())
    result.dt_bias = mamba_download(ctx, discretize_backward.d_dt_bias, dims.nheads)
    result.A_log = mamba_download(ctx, discretize_backward.d_a_log, dims.nheads)
    result.D = mamba_download(ctx, tail.d_d, dims.nheads)
    result.norm_weight = mamba_download(ctx, tail.d_gnorm_w, dims.d_inner)
    result.out_proj_weight = mamba_download(ctx, tail.d_w_out, dims.d_model*dims.d_inner)

    # Explicit ownership extends all GPU buffers through synchronization.
    _ = conv_backward^
    _ = discretize_backward^
    _ = scale_reduction^
    _ = d_final^
    _ = ssd_backward^
    _ = tail^
    _ = d_residual^
    _ = x^
    _ = stages^
    _ = state^
    _ = dweights^
    _ = ctx^
    return result^
