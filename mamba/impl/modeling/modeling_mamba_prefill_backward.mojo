# SPDX-License-Identifier: Apache-2.0
"""Shared synchronous Mamba-1 zero-state prefill VJP.

Extracted from checks/mamba_backward_device_tail_dump.mojo (2026-09-06).
The certified launch/reduction order and post-sync convolution layout copy
are retained unchanged. The driver and Python binding use this same pass.
"""
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from mamba.checks.mamba_backward import (
    PROJ_OUT,
    PROJ_DT,
    PROJ_IN,
    PROJ_X,
    RED_CONV_BIAS,
    RED_CONV_W_TAP0,
    RED_CONV_W_TAP1,
    RED_CONV_W_TAP2,
    RED_CONV_W_TAP3,
    RED_D,
    RED_DT_BIAS,
    RED_NORM_W,
    mamba_backward_proj_a_into,
    mamba_backward_proj_b_into,
    mamba_backward_reduce_into,
    mamba_reduction_tap,
    mamba_backward_workspace_max_floats,
)
from mamba.checks.mamba_fixture import (
    D_CONV,
    D_STATE,
    MambaDims,
    MambaWeights,
)
from mamba.impl.modeling.modeling_mamba import (
    MambaDeviceStages,
    MambaDeviceState,
    MambaDeviceWeights,
    mamba_block_forward,
    mamba_download,
    mamba_upload,
    mamba_zeros,
)
from mamba.impl.modeling.modeling_mamba_backward import (
    mamba_bwd_concat_xp_into,
    mamba_bwd_concat_p_into,
    mamba_bwd_conv_tap_product_into,
    mamba_bwd_ddtp_into,
    mamba_bwd_dhin_into,
    mamba_bwd_du_join_into,
    mamba_bwd_gate_into,
    mamba_bwd_norm_into,
)
from mamba.impl.ops.selective_scan_backward import (
    bwd_da_partial_floats,
    bwd_dh_floats,
    bwd_h_checkpoint_floats,
    mamba_bwd_da_into,
    mamba_bwd_da_log_into,
    mamba_bwd_dbc_into,
    selective_scan_bwd_scan_into,
    selective_scan_checkpoint_fn,
)


struct Mamba1PrefillGradients(Movable):
    """Independent host arrays; field order preserves the native dump inventory."""
    var out_proj_weight: List[Float32]
    var D: List[Float32]
    var A_log: List[Float32]
    var dt_proj_weight: List[Float32]
    var dt_proj_bias: List[Float32]
    var x_proj_weight: List[Float32]
    var conv1d_weight: List[Float32]
    var conv1d_bias: List[Float32]
    var in_proj_weight: List[Float32]
    var norm_weight: List[Float32]
    var x: List[Float32]
    var stage_dB: List[Float32]
    var stage_dC: List[Float32]

    def __init__(out self):
        self.out_proj_weight = List[Float32]()
        self.D = List[Float32]()
        self.A_log = List[Float32]()
        self.dt_proj_weight = List[Float32]()
        self.dt_proj_bias = List[Float32]()
        self.x_proj_weight = List[Float32]()
        self.conv1d_weight = List[Float32]()
        self.conv1d_bias = List[Float32]()
        self.in_proj_weight = List[Float32]()
        self.norm_weight = List[Float32]()
        self.x = List[Float32]()
        self.stage_dB = List[Float32]()
        self.stage_dC = List[Float32]()


def _ones(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(1.0))
    return out^


def mamba1_prefill_backward(
    weights: MambaWeights,
    input: List[Float32],
    grad_output: List[Float32],
    b: Int,
    l: Int,
) raises -> Mamba1PrefillGradients:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("mamba1 backward: only IDENTICAL zero-state prefill is implemented")
    if b <= 0 or l <= 0:
        raise Error("mamba1 backward: B and L must be positive")
    if len(input) != b * l * weights.dims.d_model or len(grad_output) != len(input):
        raise Error("mamba1 backward: input and grad_output lengths must equal B*L*d_model")
    for i in range(len(grad_output)):
        var bits = bitcast[DType.uint32](grad_output[i])
        if (bits & UInt32(0x7f800000)) == UInt32(0x7f800000):
            raise Error("mamba1 backward: non-finite grad_output at flat index " + String(i))
    var dims = weights.dims.copy()
    var m = b * l
    var ctx = DeviceContext()
    var dweights = MambaDeviceWeights(ctx, weights)
    # The forward mutates its recurrent state. Backward T2 needs the state
    # entering that call, so retain a distinct zero-state allocation.
    var state_in = MambaDeviceState(ctx, b, dims)
    var state = MambaDeviceState(ctx, b, dims)
    var stages = MambaDeviceStages(ctx, b, l, dims)
    var x = mamba_upload(ctx, input)
    var trace = IdentityTrace.disabled()
    mamba_block_forward(
        ctx, stages, state, dweights, x, b, l, trace,
        String("m1.backward.tail"),
    )

    var dres = mamba_upload(ctx, grad_output)
    var workspace = mamba_zeros(
        ctx, mamba_backward_workspace_max_floats(dims, m)
    )
    var dg = mamba_zeros(ctx, m * dims.d_inner)
    var dw_out = mamba_zeros(ctx, dims.d_model * dims.d_inner)
    mamba_backward_proj_a_into(
        ctx, dg, dres, dweights.w_out, workspace, PROJ_OUT, dims, m
    )
    mamba_backward_proj_b_into(
        ctx, dw_out, dres, stages.gate_out, workspace, PROJ_OUT, dims, m
    )

    var dsk = mamba_zeros(ctx, m * dims.d_inner)
    var dz = mamba_zeros(ctx, m * dims.d_inner)
    var du_d = mamba_zeros(ctx, m * dims.d_inner)
    var product_d = mamba_zeros(ctx, m * dims.d_inner)
    mamba_bwd_gate_into(
        ctx, dsk, dz, du_d, product_d, dg, stages.skip_out, stages.silu_out,
        stages.in_proj, dweights.d_skip, m, dims.d_inner,
    )

    # T2, T1, B17/B18 and T3: reconstruct every forward state, walk the
    # recurrence backward, then contract the channel-shared B/C gradients.
    var h_ckpt = mamba_zeros(
        ctx, bwd_h_checkpoint_floats(b, l, dims.d_inner)
    )
    selective_scan_checkpoint_fn(
        ctx, h_ckpt, state_in.h, stages.silu_out, stages.softplus_out,
        stages.a_out, stages.b_mat, b, l, dims.d_inner,
    )
    var dh = mamba_zeros(
        ctx, bwd_dh_floats(b, l, dims.d_inner)
    )
    var du_s = mamba_zeros(ctx, m * dims.d_inner)
    var ddelta = mamba_zeros(ctx, m * dims.d_inner)
    var scan_w = mamba_zeros(ctx, m * dims.d_inner)
    selective_scan_bwd_scan_into(
        ctx, dh, du_s, ddelta, scan_w, dsk, stages.c_mat, stages.b_mat,
        stages.softplus_out, stages.a_out, stages.silu_out, h_ckpt,
        b, l, dims.d_inner,
    )
    var dcm = mamba_zeros(ctx, m * D_STATE)
    var dbm = mamba_zeros(ctx, m * D_STATE)
    mamba_bwd_dbc_into(
        ctx, dcm, dbm, dsk, h_ckpt, scan_w, dh,
        b, l, dims.d_inner,
    )

    # T4/T5 and B21: batch-private dA partials, pinned batch fold, then the
    # derivative through A = -exp(A_log). This produces the third complete
    # external parameter gradient owned by this composed segment.
    var da = mamba_zeros(ctx, dims.d_inner * D_STATE)
    var da_partial = mamba_zeros(
        ctx, bwd_da_partial_floats(b, dims.d_inner)
    )
    mamba_bwd_da_into(
        ctx, da, da_partial, dh, stages.softplus_out, stages.a_out,
        stages.silu_out, stages.b_mat, h_ckpt,
        b, l, dims.d_inner,
    )
    var da_log = mamba_zeros(ctx, dims.d_inner * D_STATE)
    mamba_bwd_da_log_into(ctx, da_log, da, stages.a_out, dims.d_inner)

    # B22-B29: softplus derivative, dt projection, inverse x_proj split,
    # x-projection gradients, and the three-way join at the scan input.
    var ddtp = mamba_zeros(ctx, m * dims.d_inner)
    mamba_bwd_ddtp_into(
        ctx, ddtp, ddelta, stages.dt_proj, dweights.b_dt, m, dims.d_inner
    )
    var ddtl = mamba_zeros(ctx, m * dims.dt_rank)
    var dw_dt = mamba_zeros(ctx, dims.d_inner * dims.dt_rank)
    mamba_backward_proj_a_into(
        ctx, ddtl, ddtp, dweights.w_dt, workspace, PROJ_DT, dims, m
    )
    mamba_backward_proj_b_into(
        ctx, dw_dt, ddtp, stages.dt_low, workspace, PROJ_DT, dims, m
    )
    var db_dt = mamba_zeros(ctx, dims.d_inner)
    var ones = mamba_upload(ctx, _ones(m))
    mamba_backward_reduce_into(
        ctx, db_dt, ddtp, ones, workspace, RED_DT_BIAS, dims, m
    )
    var dxp = mamba_zeros(ctx, m * dims.x_proj_rows())
    mamba_bwd_concat_xp_into(ctx, dxp, ddtl, dbm, dcm, m, dims)
    var du_x = mamba_zeros(ctx, m * dims.d_inner)
    var dw_x = mamba_zeros(ctx, dims.x_proj_rows() * dims.d_inner)
    mamba_backward_proj_a_into(
        ctx, du_x, dxp, dweights.w_x, workspace, PROJ_X, dims, m
    )
    mamba_backward_proj_b_into(
        ctx, dw_x, dxp, stages.silu_out, workspace, PROJ_X, dims, m
    )
    var du = mamba_zeros(ctx, m * dims.d_inner)
    var dconv = mamba_zeros(ctx, m * dims.d_inner)
    mamba_bwd_du_join_into(
        ctx, du, dconv, du_d, du_s, du_x, stages.conv_out, m, dims.d_inner
    )

    # B31-B33: causal-convolution input gradient plus bias and four weight
    # tap reductions. Each tap's arithmetic stays on device. The final four
    # contiguous columns are interleaved on the host after synchronization;
    # that operation is a pure layout copy and introduces no numeric seam.
    var dhin = mamba_zeros(ctx, m * dims.d_inner)
    mamba_bwd_dhin_into(
        ctx, dhin, dconv, dweights.conv_w,
        b, l, dims.d_inner,
    )
    var dconv_bias = mamba_zeros(ctx, dims.d_inner)
    mamba_backward_reduce_into(
        ctx, dconv_bias, dconv, ones, workspace,
        RED_CONV_BIAS, dims, m,
    )
    var tap_product0 = mamba_zeros(ctx, m * dims.d_inner)
    var tap_product1 = mamba_zeros(ctx, m * dims.d_inner)
    var tap_product2 = mamba_zeros(ctx, m * dims.d_inner)
    var tap_product3 = mamba_zeros(ctx, m * dims.d_inner)
    var tap_grad0 = mamba_zeros(ctx, dims.d_inner)
    var tap_grad1 = mamba_zeros(ctx, dims.d_inner)
    var tap_grad2 = mamba_zeros(ctx, dims.d_inner)
    var tap_grad3 = mamba_zeros(ctx, dims.d_inner)
    mamba_bwd_conv_tap_product_into(
        ctx, tap_product0, dconv, stages.in_proj, state_in.conv_win,
        b, l, dims.d_inner, 0,
    )
    mamba_backward_reduce_into(
        ctx, tap_grad0, tap_product0, ones, workspace,
        RED_CONV_W_TAP0, dims, m,
    )
    mamba_bwd_conv_tap_product_into(
        ctx, tap_product1, dconv, stages.in_proj, state_in.conv_win,
        b, l, dims.d_inner, 1,
    )
    mamba_backward_reduce_into(
        ctx, tap_grad1, tap_product1, ones, workspace,
        RED_CONV_W_TAP1, dims, m,
    )
    mamba_bwd_conv_tap_product_into(
        ctx, tap_product2, dconv, stages.in_proj, state_in.conv_win,
        b, l, dims.d_inner, 2,
    )
    mamba_backward_reduce_into(
        ctx, tap_grad2, tap_product2, ones, workspace,
        RED_CONV_W_TAP2, dims, m,
    )
    mamba_bwd_conv_tap_product_into(
        ctx, tap_product3, dconv, stages.in_proj, state_in.conv_win,
        b, l, dims.d_inner, 3,
    )
    mamba_backward_reduce_into(
        ctx, tap_grad3, tap_product3, ones, workspace,
        RED_CONV_W_TAP3, dims, m,
    )

    # B34-B42: join the two in-projection halves, route both projection
    # derivatives, then run the fused RMSNorm/residual backward and its
    # deterministic token reduction. `dx` is the full block input gradient.
    var dp = mamba_zeros(ctx, m * 2 * dims.d_inner)
    mamba_bwd_concat_p_into(ctx, dp, dhin, dz, m, dims.d_inner)
    var dnrm = mamba_zeros(ctx, m * dims.d_model)
    var dw_in = mamba_zeros(ctx, 2 * dims.d_inner * dims.d_model)
    mamba_backward_proj_a_into(
        ctx, dnrm, dp, dweights.w_in, workspace, PROJ_IN, dims, m
    )
    mamba_backward_proj_b_into(
        ctx, dw_in, dp, stages.norm_out, workspace, PROJ_IN, dims, m
    )
    var dx_full = mamba_zeros(ctx, m * dims.d_model)
    var drstd = mamba_zeros(ctx, m)
    var norm_product = mamba_zeros(ctx, m * dims.d_model)
    mamba_bwd_norm_into(
        ctx, dx_full, drstd, norm_product, dres, dnrm, x,
        stages.norm_out, dweights.norm_w, stages.norm_sumsq,
        m, dims.d_model,
    )
    var dw_norm = mamba_zeros(ctx, dims.d_model)
    mamba_backward_reduce_into(
        ctx, dw_norm, norm_product, ones, workspace, RED_NORM_W, dims, m
    )

    var dd_skip = mamba_zeros(ctx, dims.d_inner)
    mamba_backward_reduce_into(
        ctx, dd_skip, product_d, ones, workspace, RED_D, dims, m
    )
    ctx.synchronize()

    var tg0 = mamba_download(ctx, tap_grad0, dims.d_inner)
    var tg1 = mamba_download(ctx, tap_grad1, dims.d_inner)
    var tg2 = mamba_download(ctx, tap_grad2, dims.d_inner)
    var tg3 = mamba_download(ctx, tap_grad3, dims.d_inner)
    var dconv_weight = List[Float32]()
    for _ in range(dims.d_inner * D_CONV):
        dconv_weight.append(Float32(0.0))
    for d in range(dims.d_inner):
        dconv_weight[
            d * D_CONV + mamba_reduction_tap(RED_CONV_W_TAP0)
        ] = tg0[d]
        dconv_weight[
            d * D_CONV + mamba_reduction_tap(RED_CONV_W_TAP1)
        ] = tg1[d]
        dconv_weight[
            d * D_CONV + mamba_reduction_tap(RED_CONV_W_TAP2)
        ] = tg2[d]
        dconv_weight[
            d * D_CONV + mamba_reduction_tap(RED_CONV_W_TAP3)
        ] = tg3[d]

    var result = Mamba1PrefillGradients()
    result.out_proj_weight = mamba_download(ctx, dw_out, dims.d_model * dims.d_inner)
    result.D = mamba_download(ctx, dd_skip, dims.d_inner)
    result.A_log = mamba_download(ctx, da_log, dims.d_inner * D_STATE)
    result.dt_proj_weight = mamba_download(ctx, dw_dt, dims.d_inner * dims.dt_rank)
    result.dt_proj_bias = mamba_download(ctx, db_dt, dims.d_inner)
    result.x_proj_weight = mamba_download(ctx, dw_x, dims.x_proj_rows() * dims.d_inner)
    result.conv1d_weight = dconv_weight.copy()
    result.conv1d_bias = mamba_download(ctx, dconv_bias, dims.d_inner)
    result.in_proj_weight = mamba_download(ctx, dw_in, 2 * dims.d_inner * dims.d_model)
    result.norm_weight = mamba_download(ctx, dw_norm, dims.d_model)
    result.x = mamba_download(ctx, dx_full, m * dims.d_model)
    result.stage_dB = mamba_download(ctx, dbm, m * D_STATE)
    result.stage_dC = mamba_download(ctx, dcm, m * D_STATE)
    _ = dconv_weight^
    _ = tg3^
    _ = tg2^
    _ = tg1^
    _ = tg0^
    _ = dw_norm^
    _ = norm_product^
    _ = drstd^
    _ = dx_full^
    _ = dw_in^
    _ = dnrm^
    _ = dp^
    _ = tap_grad3^
    _ = tap_grad2^
    _ = tap_grad1^
    _ = tap_grad0^
    _ = tap_product3^
    _ = tap_product2^
    _ = tap_product1^
    _ = tap_product0^
    _ = dconv_bias^
    _ = dhin^
    _ = dconv^
    _ = du^
    _ = dw_x^
    _ = du_x^
    _ = dxp^
    _ = db_dt^
    _ = dw_dt^
    _ = ddtl^
    _ = ddtp^
    _ = da_log^
    _ = da_partial^
    _ = da^
    _ = dbm^
    _ = dcm^
    _ = scan_w^
    _ = ddelta^
    _ = du_s^
    _ = dh^
    _ = h_ckpt^
    _ = dd_skip^
    _ = ones^
    _ = product_d^
    _ = du_d^
    _ = dz^
    _ = dsk^
    _ = dw_out^
    _ = dg^
    _ = workspace^
    _ = dres^
    _ = stages^
    _ = state^
    _ = state_in^
    _ = dweights^
    _ = x^
    _ = ctx^
    return result^
