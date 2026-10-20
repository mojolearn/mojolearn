# SPDX-License-Identifier: Apache-2.0
"""The CPU column's Mamba-3 zero-state IDENTICAL prefill VJP: the hand-chained
launch walk, kept ONLY as the input `tools/mamba_host_gen.py` restates for the
host (`mamba/host/gen/mamba3_prefill_chain.mojo`, served by
`bindings/_mojolearn_mamba_host.mojo`). The device never runs it: the GPU
binding and the Samba train step assemble the same VJP on the tape
(autograd/rules_mamba3.mojo), whose rules launch these same kernels in this
same order. No device module imports this file (lane ag-mamba3-tape,
2026-10-19: the device copy of this walk is tombstoned in
autograd/kernels/mamba3_prefill.mojo).
No incoming cache or final-state cotangent is accepted.
"""
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns

from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojoNeuralMambaContextIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoNeuralMambaContextFast"


from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.identity_trace import IdentityTrace
from autograd.kernels.mamba3_proj import (
    PROJ3_IN, PROJ3_OUT,
    RED3_D, RED3_NORM_W,
    RED3_DT_BIAS,
    RED3_BNORM_W, RED3_CNORM_W, RED3_B_BIAS, RED3_C_BIAS,
    mamba3_backward_ones_floats,
    mamba3_backward_proj_a_into,
    mamba3_backward_proj_b_into,
    mamba3_backward_reduce_into,
    mamba3_backward_workspace_max_floats,
)
from mamba.checks.mamba3_fixture import (
    M3_CHUNK_SIZE,
    M3_D_STATE,
    M3_HEADDIM,
    M3_NUM_ROPE_ANGLES,
)
from autograd.kernels.mamba_refusal import mamba_refuse_grad_output
from mamba.impl.modules.mamba3 import (
    Mamba3DeviceStages,
    Mamba3DeviceWeights,
    allocate_inference_cache,
    mamba3_block_forward,
)
from autograd.kernels.mamba3 import (
    mamba3_backward_gate_skip_into,
    mamba3_backward_qkdot_into,
    mamba3_backward_s16_s15_into,
    mamba3_backward_join_rotary_into,
    mamba3_backward_join_current_into,
    mamba3_backward_angle_into,
    mamba3_angle_suffix_sums_cells,
    mamba3_backward_dt_partial_into,
    mamba3_backward_seg_adt_into,
    mamba3_backward_adt_product_into,
    mamba3_backward_s17_state_into,
    mamba3_backward_s17_operands_into,
    mamba3_backward_join_s16_s17_into,
    mamba3_backward_s15_only_into,
    mamba3_backward_rotary_only_into,
    mamba3_backward_dacs_to_adt_into,
    mamba3_backward_join_two_into,
    mamba3_backward_a_heavy_tail_into,
    mamba3_backward_bcnorm_into,
    mamba3_backward_pack_in_proj_into,
    mamba3_backward_block_norm_into,
)
from mamba.impl.modeling.modeling_mamba import (
    mamba_download,
    mamba_upload,
    mamba_zeros,
)
from mamba.checks.mamba3_fixture import Mamba3Weights, Mamba3Dims

from autograd.kernels.mamba3 import (
    AFN_M3_BWD_CHUNK,
    afn_m3_theta_chunks,
    mamba3_afn_backward_angle_into,
)
# the result types and the scratch helpers stay with the device tree
from autograd.kernels.mamba3_prefill import (
    IDN_M3_BWD_SKIP_DEAD,
    Mamba3DeviceGradients,
    Mamba3PrefillGradients,
    _AfnM3Arena,
    _m3_scratch,
    _mtick,
)


def mamba3_prefill_backward(
    weights: Mamba3Weights,
    input: List[Float32],
    grad_output: List[Float32],
    b: Int,
    l: Int,
    var ctx_in: Optional[DeviceContext] = None,
) raises -> Mamba3PrefillGradients:
    if b <= 0 or l <= 0:
        raise Error("mamba3 backward: B and L must be positive")
    if len(input) != b * l * weights.dims.d_model or len(grad_output) != len(input):
        raise Error("mamba3 backward: input and grad_output lengths must equal B*L*d_model")
    var dims = weights.dims.copy()
    var m = b * l
    # lane/neural-apple2 (2026-09-28): the binding passes its process-lifetime
    # context (core/neural_context.mojo); a fresh context per call meant a
    # new Metal queue and a pipeline compile of every kernel on every call.
    # Direct callers also reuse that context when none is supplied.
    var ctx: DeviceContext
    if ctx_in:
        ctx = ctx_in.take()
    else:
        ctx = process_ctx[_DEVCTX_SLOT]()
    # lane/neural-apple2: MOJOLEARN_MAMBA_TIMING=1 prints a wall per stage
    # (a synchronize around each; measurement only, off by default).
    var ton = String(getenv("MOJOLEARN_MAMBA_TIMING")) != ""
    var tk = Int(perf_counter_ns())
    var device_weights = Mamba3DeviceWeights(ctx, weights)
    var state = allocate_inference_cache(ctx, b, dims)
    var stages = Mamba3DeviceStages(ctx, b, l, 0, dims)
    var x = mamba_upload(ctx, input)
    var trace = IdentityTrace.disabled()
    mamba3_block_forward(
        ctx, stages, state, device_weights, x, b, l,
        trace, String("m3.backward.tail"),
    )
    _mtick(ctx, ton, tk, "mamba3_block_forward")

    # residual.out = x + out_proj, hence the objective cotangent reaches the
    # projection unchanged; the pass continues through every public leaf.
    var d_output = mamba_upload(
        ctx, grad_output
    )
    # cpu3-seq: the grad_output finite refusal runs on the uploaded buffer
    # (same predicate by bits, same first flat index and message), not as a
    # host walk over the List.
    mamba_refuse_grad_output(ctx, String("mamba3"), d_output, grad_output)
    var gradients = mamba3_prefill_backward_on(
        ctx, device_weights, stages, x, d_output, b, l, dims, ton, tk
    )

    var result = Mamba3PrefillGradients()
    result.x = mamba_download(ctx, gradients.x, m*dims.d_model)
    result.block_norm_weight = mamba_download(ctx, gradients.block_norm_weight, dims.d_model)
    result.in_proj_weight = mamba_download(ctx, gradients.in_proj_weight, dims.d_in_proj()*dims.d_model)
    result.dt_bias = mamba_download(ctx, gradients.dt_bias, dims.nheads)
    result.B_norm_weight = mamba_download(ctx, gradients.B_norm_weight, M3_D_STATE)
    result.C_norm_weight = mamba_download(ctx, gradients.C_norm_weight, M3_D_STATE)
    result.B_bias = mamba_download(ctx, gradients.B_bias, dims.nheads*M3_D_STATE)
    result.C_bias = mamba_download(ctx, gradients.C_bias, dims.nheads*M3_D_STATE)
    result.D = mamba_download(ctx, gradients.D, dims.nheads)
    result.out_proj_weight = mamba_download(ctx, gradients.out_proj_weight, dims.d_model*dims.d_inner)
    _mtick(ctx, ton, tk, "downloads")

    # Explicit ownership extends all GPU buffers through synchronization.
    _ = gradients^
    _ = d_output^
    _ = x^
    _ = stages^
    _ = state^
    _ = device_weights^
    _ = ctx^
    return result^


def mamba3_prefill_backward_on(
    ctx: DeviceContext,
    mut device_weights: Mamba3DeviceWeights,
    mut stages: Mamba3DeviceStages,
    mut x: DeviceBuffer[DType.float32],
    mut d_output: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    dims: Mamba3Dims,
    ton: Bool,
    mut tk: Int,
) raises -> Mamba3DeviceGradients:
    """The VJP's launches on device-resident operands: `stages` and `x` are
    the forward's (recorded by `mamba3_block_forward` on `device_weights`
    from the zero state), `d_output` the objective cotangent. Every
    launch and its order are `mamba3_prefill_backward`'s, which now calls
    this; the session binding (lane/neural-net-experiment) calls it with
    the stages its last forward recorded, so a training step's backward
    skips the forward recompute when x and the weights are byte for byte
    the forward's. Synchronized on return; the gradients stay on the
    device."""
    var m = b * l
    # lane afn-samba: the arena bracket and the angle chains' chunk sums,
    # declared here and initialized only under their defines.
    var afn_arena = _AfnM3Arena()
    var afn_own = afn_arena.own()
    # lane/neural-apple2: every scratch below is filled and used on the one
    # in-order `ctx` and kept alive to the final synchronize (the explicit
    # last uses at the end), so its allocation needs no wait of its own.
    var d_gate = _m3_scratch(afn_own, ctx, m * dims.d_inner)
    var d_weight = mamba_zeros[False](ctx, dims.d_model * dims.d_inner)
    var workspace = mamba_zeros(
        ctx, mamba3_backward_workspace_max_floats(dims, m)
    )
    mamba3_backward_proj_a_into(
        ctx, d_gate, d_output, device_weights.w_out, workspace,
        PROJ3_OUT, dims, m,
    )
    _mtick(ctx, ton, tk, "proj_a")
    mamba3_backward_proj_b_into(
        ctx, d_weight, d_output, stages.gate_out, workspace,
        PROJ3_OUT, dims, m,
    )
    _mtick(ctx, ton, tk, "proj_b")
    var tail_cells = m * dims.d_inner
    var head_cells = m * dims.nheads
    var d_skip = _m3_scratch(afn_own, ctx, tail_cells)
    var d_z = _m3_scratch(afn_own, ctx, tail_cells)
    var d_v = _m3_scratch(afn_own, ctx, tail_cells)
    var d_qkdot = _m3_scratch(afn_own, ctx, head_cells)
    var d_d_product = _m3_scratch(afn_own, ctx, head_cells)
    var d_d = mamba_zeros[False](ctx, dims.nheads)
    var ones = _m3_scratch(afn_own, ctx, mamba3_backward_ones_floats(m))
    ones.enqueue_fill(Float32(1.0))
    mamba3_backward_gate_skip_into(
        ctx, d_skip, d_z, d_v, d_qkdot, d_d_product, d_gate,
        stages.skip_out, stages.qkdot, stages.in_proj,
        device_weights.d_skip, dims, m,
    )
    _mtick(ctx, ton, tk, "gate_skip")
    mamba3_backward_reduce_into(
        ctx, d_d, d_d_product, ones, workspace, RED3_D, dims, m
    )
    _mtick(ctx, ton, tk, "reduce")
    var qk_cells = m * dims.nheads * M3_D_STATE
    var d_b_qk = _m3_scratch(afn_own, ctx, qk_cells)
    var d_c_qk = _m3_scratch(afn_own, ctx, qk_cells)
    var d_b_bias_qk = _m3_scratch(afn_own, ctx, qk_cells)
    var d_c_bias_qk = _m3_scratch(afn_own, ctx, qk_cells)
    var d_gamma_qk = _m3_scratch(afn_own, ctx, head_cells)
    var d_dt_qk = _m3_scratch(afn_own, ctx, head_cells)
    var d_trap_qk = _m3_scratch(afn_own, ctx, head_cells)
    mamba3_backward_qkdot_into(
        ctx, d_b_qk, d_c_qk, d_b_bias_qk, d_c_bias_qk, d_gamma_qk,
        d_dt_qk, d_trap_qk,
        d_qkdot, stages.bcnorm_b, stages.bcnorm_c, device_weights.b_bias,
        device_weights.c_bias, stages.gamma_work, stages.dt_work,
        stages.sig_work, dims, m,
    )
    _mtick(ctx, ton, tk, "qkdot")
    var state_cells = m * dims.nheads * M3_D_STATE
    var d_q_s16 = _m3_scratch(afn_own, ctx, state_cells)
    var d_ks_s16 = _m3_scratch(afn_own, ctx, state_cells)
    var d_v_s16 = _m3_scratch(afn_own, ctx, tail_cells)
    var d_krot_s15 = _m3_scratch(afn_own, ctx, state_cells)
    var d_scale_s15 = _m3_scratch(afn_own, ctx, head_cells)
    mamba3_backward_s16_s15_into(
        ctx, d_q_s16, d_ks_s16, d_v_s16, d_krot_s15, d_scale_s15,
        d_skip, stages.rotq_work, stages.kscale_work, stages.v_work,
        stages.seg_l, stages.rotk_work, stages.scale_work,
        b, l, dims, M3_CHUNK_SIZE,
    )
    _mtick(ctx, ton, tk, "s16_s15")
    var d_value_total = _m3_scratch(afn_own, ctx, tail_cells)
    var d_gamma_scale = _m3_scratch(afn_own, ctx, head_cells)
    var d_beta_scale = _m3_scratch(afn_own, ctx, head_cells)
    var d_qraw_rot = _m3_scratch(afn_own, ctx, state_cells)
    var d_kraw_rot = _m3_scratch(afn_own, ctx, state_cells)
    var d_theta_rot = _m3_scratch(afn_own, ctx, m * dims.nheads * M3_NUM_ROPE_ANGLES)
    mamba3_backward_join_rotary_into(
        ctx, d_value_total, d_gamma_scale, d_beta_scale, d_qraw_rot,
        d_kraw_rot, d_theta_rot, d_v, d_v_s16, d_scale_s15,
        d_q_s16, d_krot_s15, stages.bcnorm_b, stages.bcnorm_c,
        device_weights.b_bias, device_weights.c_bias, stages.theta_out,
        m, dims,
    )
    _mtick(ctx, ton, tk, "join_rotary")
    var d_b_total=_m3_scratch(afn_own, ctx,state_cells);var d_c_total=_m3_scratch(afn_own, ctx,state_cells)
    var d_gamma_total=_m3_scratch(afn_own, ctx,head_cells);var d_dt_total=_m3_scratch(afn_own, ctx,head_cells);var d_trap_total=_m3_scratch(afn_own, ctx,head_cells)
    # lane nr-mamba IDN_M3_BWD_SKIP_DEAD: the pre-join pass below (join
    # current, angle, dt partial, dt_bias reduce, and the adt product after
    # seg_adt) feeds no returned gradient: the join pass recomputes each from
    # the joined d_theta / d_q / d_k, and only d_adt_seg (seg_adt) and
    # d_value_total (join_rotary) cross over. Skipping it moves no bit.
    comptime if not IDN_M3_BWD_SKIP_DEAD:
        mamba3_backward_join_current_into(ctx,d_b_total,d_c_total,d_gamma_total,d_dt_total,d_trap_total,d_b_qk,d_c_qk,d_kraw_rot,d_qraw_rot,d_gamma_qk,d_gamma_scale,d_dt_qk,d_trap_qk,d_beta_scale,stages.dt_work,stages.sig_work,b,l,dims)
        _mtick(ctx, ton, tk, "join_current")
    var d_angle_rate=_m3_scratch(afn_own, ctx,m*dims.nheads*M3_NUM_ROPE_ANGLES)
    var d_angle_raw=_m3_scratch(afn_own, ctx,m*M3_NUM_ROPE_ANGLES)
    var d_dt_angle=_m3_scratch(afn_own, ctx,head_cells)
    comptime if not IDN_M3_BWD_SKIP_DEAD:
        comptime if AFN_M3_BWD_CHUNK:
            var afn_sums = _m3_scratch(afn_own, ctx, b * dims.nheads * M3_NUM_ROPE_ANGLES * afn_m3_theta_chunks(l))
            mamba3_afn_backward_angle_into(ctx,d_angle_rate,d_angle_raw,d_dt_angle,d_theta_rot,stages.dt_work,stages.in_proj,afn_sums,b,l,dims)
            afn_arena.keep.append(afn_sums^)
        else:
            var sums_rot=_m3_scratch(afn_own, ctx,mamba3_angle_suffix_sums_cells(b,l,dims.nheads))
            mamba3_backward_angle_into(ctx,d_angle_rate,d_angle_raw,d_dt_angle,d_theta_rot,stages.dt_work,stages.in_proj,b,l,dims,sums_rot)
            afn_arena.keep.append(sums_rot^)
        _mtick(ctx, ton, tk, "angle")
    var d_dt_available=_m3_scratch(afn_own, ctx,head_cells);var d_dt_raw=_m3_scratch(afn_own, ctx,head_cells);var d_dt_bias_rows=_m3_scratch(afn_own, ctx,head_cells);var d_dt_bias=_m3_scratch(afn_own, ctx,dims.nheads)
    comptime if not IDN_M3_BWD_SKIP_DEAD:
        mamba3_backward_dt_partial_into(ctx,d_dt_available,d_dt_raw,d_dt_bias_rows,d_dt_total,d_dt_angle,stages.in_proj,device_weights.dt_bias,m,dims)
        _mtick(ctx, ton, tk, "dt_partial")
        mamba3_backward_reduce_into(ctx,d_dt_bias,d_dt_bias_rows,ones,workspace,RED3_DT_BIAS,dims,m)
        _mtick(ctx, ton, tk, "reduce")
    var seg_cells=b*stages.nc*dims.nheads*M3_CHUNK_SIZE*M3_CHUNK_SIZE
    var d_seg=_m3_scratch(afn_own, ctx,seg_cells);var d_adt_seg=_m3_scratch(afn_own, ctx,head_cells)
    mamba3_backward_seg_adt_into(ctx,d_seg,d_adt_seg,d_skip,stages.rotq_work,stages.kscale_work,stages.v_work,stages.seg_l,b,l,dims,M3_CHUNK_SIZE)
    _mtick(ctx, ton, tk, "seg_adt")
    var d_a_seg=_m3_scratch(afn_own, ctx,head_cells);var d_dt_seg=_m3_scratch(afn_own, ctx,head_cells);var d_dt_with_seg=_m3_scratch(afn_own, ctx,head_cells)
    comptime if not IDN_M3_BWD_SKIP_DEAD:
        mamba3_backward_adt_product_into(ctx,d_a_seg,d_dt_seg,d_dt_with_seg,d_adt_seg,stages.a_out,stages.dt_out,d_dt_available,head_cells)
        _mtick(ctx, ton, tk, "adt_product")
    var chunk_state_cells=b*stages.nc*dims.nheads*M3_HEADDIM*M3_D_STATE
    var initial_state_cells=b*dims.nheads*M3_HEADDIM*M3_D_STATE
    var d_state_direct=_m3_scratch(afn_own, ctx,chunk_state_cells);var d_state_total=_m3_scratch(afn_own, ctx,chunk_state_cells);var d_initial_state=_m3_scratch(afn_own, ctx,initial_state_cells)
    mamba3_backward_s17_state_into(ctx,d_state_direct,d_state_total,d_initial_state,d_skip,stages.rotq_work,stages.dacs,b,l,dims,M3_CHUNK_SIZE)
    _mtick(ctx, ton, tk, "s17_state")
    var d_q17=_m3_scratch(afn_own, ctx,state_cells);var d_dacs_read17=_m3_scratch(afn_own, ctx,head_cells);var d_k17=_m3_scratch(afn_own, ctx,state_cells);var d_v17=_m3_scratch(afn_own, ctx,tail_cells);var d_dacs_rec17=_m3_scratch(afn_own, ctx,head_cells)
    mamba3_backward_s17_operands_into(ctx,d_q17,d_dacs_read17,d_k17,d_v17,d_dacs_rec17,d_skip,stages.rotq_work,stages.kscale_work,stages.v_work,stages.dacs,stages.pass_states,d_state_total,b,l,dims,M3_CHUNK_SIZE)
    _mtick(ctx, ton, tk, "s17_operands")
    var d_q_join=_m3_scratch(afn_own, ctx,state_cells);var d_k_join=_m3_scratch(afn_own, ctx,state_cells);var d_v_join=_m3_scratch(afn_own, ctx,tail_cells);var d_dacs_join=_m3_scratch(afn_own, ctx,head_cells)
    mamba3_backward_join_s16_s17_into(ctx,d_q_join,d_k_join,d_v_join,d_dacs_join,d_q_s16,d_q17,d_ks_s16,d_k17,d_value_total,d_v17,d_dacs_read17,d_dacs_rec17,state_cells,tail_cells,head_cells)
    _mtick(ctx, ton, tk, "join_s16_s17")
    var d_krot_join=_m3_scratch(afn_own, ctx,state_cells);var d_scale_join=_m3_scratch(afn_own, ctx,head_cells)
    mamba3_backward_s15_only_into(ctx,d_krot_join,d_scale_join,d_k_join,stages.rotk_work,stages.scale_work,head_cells)
    _mtick(ctx, ton, tk, "s15_only")
    var d_qraw_join=_m3_scratch(afn_own, ctx,state_cells);var d_kraw_join=_m3_scratch(afn_own, ctx,state_cells);var d_theta_join=_m3_scratch(afn_own, ctx,m*dims.nheads*M3_NUM_ROPE_ANGLES)
    mamba3_backward_rotary_only_into(ctx,d_qraw_join,d_kraw_join,d_theta_join,d_q_join,d_krot_join,stages.bcnorm_b,stages.bcnorm_c,device_weights.b_bias,device_weights.c_bias,stages.theta_out,m*dims.nheads*(M3_D_STATE//2),dims.nheads)
    _mtick(ctx, ton, tk, "rotary_only")
    var d_adt_from_dacs=_m3_scratch(afn_own, ctx,head_cells);var d_adt_join=_m3_scratch(afn_own, ctx,head_cells)
    mamba3_backward_dacs_to_adt_into(ctx,d_adt_from_dacs,d_dacs_join,b,l,dims.nheads,M3_CHUNK_SIZE)
    _mtick(ctx, ton, tk, "dacs_to_adt")
    mamba3_backward_join_two_into(ctx,d_adt_join,d_adt_seg,d_adt_from_dacs,head_cells)
    _mtick(ctx, ton, tk, "join_two")
    var d_a_join=_m3_scratch(afn_own, ctx,head_cells);var d_dt_join_adt=_m3_scratch(afn_own, ctx,head_cells);var d_dt_join_scratch=_m3_scratch(afn_own, ctx,head_cells);var zero_dt=_m3_scratch(afn_own, ctx,head_cells)
    mamba3_backward_adt_product_into(ctx,d_a_join,d_dt_join_adt,d_dt_join_scratch,d_adt_join,stages.a_out,stages.dt_out,zero_dt,head_cells)
    _mtick(ctx, ton, tk, "adt_product")
    var d_a_raw_join=_m3_scratch(afn_own, ctx,head_cells)
    mamba3_backward_a_heavy_tail_into(ctx,d_a_raw_join,d_a_join,stages.in_proj,m,dims)
    _mtick(ctx, ton, tk, "a_heavy_tail")
    var d_b_join=_m3_scratch(afn_own, ctx,state_cells);var d_c_join=_m3_scratch(afn_own, ctx,state_cells);var d_gamma_join=_m3_scratch(afn_own, ctx,head_cells);var d_dt_join_current=_m3_scratch(afn_own, ctx,head_cells);var d_trap_join=_m3_scratch(afn_own, ctx,head_cells);var d_beta_join=_m3_scratch(afn_own, ctx,head_cells)
    ctx.enqueue_copy(dst_buf=d_beta_join, src_buf=d_scale_join)
    mamba3_backward_join_current_into(ctx,d_b_join,d_c_join,d_gamma_join,d_dt_join_current,d_trap_join,d_b_qk,d_c_qk,d_kraw_join,d_qraw_join,d_gamma_qk,d_scale_join,d_dt_qk,d_trap_qk,d_beta_join,stages.dt_work,stages.sig_work,b,l,dims)
    _mtick(ctx, ton, tk, "join_current")
    var d_angle_rate_join=_m3_scratch(afn_own, ctx,m*dims.nheads*M3_NUM_ROPE_ANGLES);var d_angle_raw_join=_m3_scratch(afn_own, ctx,m*M3_NUM_ROPE_ANGLES);var d_dt_angle_join=_m3_scratch(afn_own, ctx,head_cells)
    comptime if AFN_M3_BWD_CHUNK:
        var afn_sums = _m3_scratch(afn_own, ctx, b * dims.nheads * M3_NUM_ROPE_ANGLES * afn_m3_theta_chunks(l))
        mamba3_afn_backward_angle_into(ctx,d_angle_rate_join,d_angle_raw_join,d_dt_angle_join,d_theta_join,stages.dt_work,stages.in_proj,afn_sums,b,l,dims)
        afn_arena.keep.append(afn_sums^)
    else:
        var sums_join=_m3_scratch(afn_own, ctx,mamba3_angle_suffix_sums_cells(b,l,dims.nheads))
        mamba3_backward_angle_into(ctx,d_angle_rate_join,d_angle_raw_join,d_dt_angle_join,d_theta_join,stages.dt_work,stages.in_proj,b,l,dims,sums_join)
        afn_arena.keep.append(sums_join^)
    _mtick(ctx, ton, tk, "angle")
    var d_dt_join_base=_m3_scratch(afn_own, ctx,head_cells);mamba3_backward_join_two_into(ctx,d_dt_join_base,d_dt_join_current,d_dt_join_adt,head_cells)
    _mtick(ctx, ton, tk, "join_two")
    var d_dt_join_available=_m3_scratch(afn_own, ctx,head_cells);var d_dt_raw_join=_m3_scratch(afn_own, ctx,head_cells);var d_dt_bias_rows_join=_m3_scratch(afn_own, ctx,head_cells);var d_dt_bias_join=mamba_zeros[False](ctx,dims.nheads)
    mamba3_backward_dt_partial_into(ctx,d_dt_join_available,d_dt_raw_join,d_dt_bias_rows_join,d_dt_join_base,d_dt_angle_join,stages.in_proj,device_weights.dt_bias,m,dims)
    _mtick(ctx, ton, tk, "dt_partial")
    mamba3_backward_reduce_into(ctx,d_dt_bias_join,d_dt_bias_rows_join,ones,workspace,RED3_DT_BIAS,dims,m)
    _mtick(ctx, ton, tk, "reduce")
    var d_b_raw=_m3_scratch(afn_own, ctx,m*M3_D_STATE);var d_c_raw=_m3_scratch(afn_own, ctx,m*M3_D_STATE);var d_bw_rows=_m3_scratch(afn_own, ctx,m*M3_D_STATE);var d_cw_rows=_m3_scratch(afn_own, ctx,m*M3_D_STATE)
    mamba3_backward_bcnorm_into(ctx,d_b_raw,d_bw_rows,d_b_join,stages.in_proj,device_weights.bnorm_w,m,dims,dims.col_b())
    _mtick(ctx, ton, tk, "bcnorm")
    mamba3_backward_bcnorm_into(ctx,d_c_raw,d_cw_rows,d_c_join,stages.in_proj,device_weights.cnorm_w,m,dims,dims.col_c())
    _mtick(ctx, ton, tk, "bcnorm")
    var d_bw=mamba_zeros[False](ctx,M3_D_STATE);var d_cw=mamba_zeros[False](ctx,M3_D_STATE);var d_bb=mamba_zeros[False](ctx,dims.nheads*M3_D_STATE);var d_cb=mamba_zeros[False](ctx,dims.nheads*M3_D_STATE)
    mamba3_backward_reduce_into(ctx,d_bw,d_bw_rows,ones,workspace,RED3_BNORM_W,dims,m);mamba3_backward_reduce_into(ctx,d_cw,d_cw_rows,ones,workspace,RED3_CNORM_W,dims,m)
    _mtick(ctx, ton, tk, "reduce")
    mamba3_backward_reduce_into(ctx,d_bb,d_b_join,ones,workspace,RED3_B_BIAS,dims,m);mamba3_backward_reduce_into(ctx,d_cb,d_c_join,ones,workspace,RED3_C_BIAS,dims,m)
    _mtick(ctx, ton, tk, "reduce")
    var d_in_proj=_m3_scratch(afn_own, ctx,m*dims.d_in_proj());var d_norm=_m3_scratch(afn_own, ctx,m*dims.d_model);var d_w_in=mamba_zeros[False](ctx,dims.d_in_proj()*dims.d_model)
    mamba3_backward_pack_in_proj_into(ctx,d_in_proj,d_z,d_v_join,d_b_raw,d_c_raw,d_dt_raw_join,d_a_raw_join,d_trap_join,d_angle_raw_join,m,dims)
    _mtick(ctx, ton, tk, "pack_in_proj")
    mamba3_backward_proj_a_into(ctx,d_norm,d_in_proj,device_weights.w_in,workspace,PROJ3_IN,dims,m)
    _mtick(ctx, ton, tk, "proj_a")
    mamba3_backward_proj_b_into(ctx,d_w_in,d_in_proj,stages.norm_out,workspace,PROJ3_IN,dims,m)
    _mtick(ctx, ton, tk, "proj_b")
    var d_x=mamba_zeros[False](ctx,m*dims.d_model);var d_norm_w_rows=_m3_scratch(afn_own, ctx,m*dims.d_model);var d_norm_w=mamba_zeros[False](ctx,dims.d_model)
    mamba3_backward_block_norm_into(ctx,d_x,d_norm_w_rows,d_norm,d_output,x,stages.norm_sumsq,device_weights.norm_w,m,dims)
    _mtick(ctx, ton, tk, "block_norm")
    mamba3_backward_reduce_into(ctx,d_norm_w,d_norm_w_rows,ones,workspace,RED3_NORM_W,dims,m)
    _mtick(ctx, ton, tk, "reduce")
    ctx.synchronize()
    var gradients = Mamba3DeviceGradients(
        d_x^, d_norm_w^, d_w_in^, d_dt_bias_join^, d_bw^, d_cw^, d_bb^, d_cb^,
        d_d^, d_weight^,
    )
    # Explicit ownership extends all GPU buffers through synchronization.
    _ = d_norm_w_rows^
    _ = d_norm^
    _ = d_in_proj^
    _ = d_cw_rows^
    _ = d_bw_rows^
    _ = d_c_raw^
    _ = d_b_raw^
    _ = d_dt_bias_rows_join^
    _ = d_dt_raw_join^
    _ = d_dt_join_available^
    _ = d_dt_join_base^
    _ = d_dt_angle_join^
    _ = d_angle_raw_join^
    _ = d_angle_rate_join^
    _ = d_beta_join^
    _ = d_trap_join^
    _ = d_dt_join_current^
    _ = d_gamma_join^
    _ = d_c_join^
    _ = d_b_join^
    _ = d_a_raw_join^
    _ = zero_dt^
    _ = d_dt_join_scratch^
    _ = d_dt_join_adt^
    _ = d_a_join^
    _ = d_adt_join^
    _ = d_adt_from_dacs^
    _ = d_theta_join^
    _ = d_kraw_join^
    _ = d_qraw_join^
    _ = d_scale_join^
    _ = d_krot_join^
    _ = d_dacs_join^
    _ = d_v_join^
    _ = d_k_join^
    _ = d_q_join^
    _ = d_dacs_rec17^
    _ = d_v17^
    _ = d_k17^
    _ = d_dacs_read17^
    _ = d_q17^
    _ = d_initial_state^
    _ = d_state_total^
    _ = d_state_direct^
    _ = d_dt_with_seg^
    _ = d_dt_seg^
    _ = d_a_seg^
    _ = d_adt_seg^
    _ = d_seg^
    _ = d_dt_bias^
    _ = d_dt_bias_rows^
    _ = d_dt_raw^
    _ = d_dt_available^
    _ = d_dt_angle^
    _ = d_angle_raw^
    _ = d_angle_rate^
    _ = d_trap_total^
    _ = d_dt_total^
    _ = d_gamma_total^
    _ = d_c_total^
    _ = d_b_total^
    _ = d_theta_rot^
    _ = d_kraw_rot^
    _ = d_qraw_rot^
    _ = d_beta_scale^
    _ = d_gamma_scale^
    _ = d_value_total^
    _ = d_scale_s15^
    _ = d_krot_s15^
    _ = d_v_s16^
    _ = d_ks_s16^
    _ = d_q_s16^
    _ = d_trap_qk^
    _ = d_dt_qk^
    _ = d_gamma_qk^
    _ = d_c_bias_qk^
    _ = d_b_bias_qk^
    _ = d_c_qk^
    _ = d_b_qk^
    _ = ones^
    _ = d_d_product^
    _ = d_qkdot^
    _ = d_v^
    _ = d_z^
    _ = d_skip^
    _ = workspace^
    _ = d_gate^
    afn_arena.close()
    return gradients^
