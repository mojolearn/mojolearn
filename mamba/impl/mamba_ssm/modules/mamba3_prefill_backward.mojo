# SPDX-License-Identifier: Apache-2.0
"""Synchronous Mamba-3 zero-state IDENTICAL prefill VJP.

Launch order extracted from checks/mamba3_backward_tail_dump.mojo.
The original diagnostic driver remains an independent byte-comparison gate.
Python/native API qualification is required separately from historical dumps.
No incoming cache or final-state cotangent is accepted.
"""
from std.memory import bitcast

from max.gpu.host import DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.identity_trace import IdentityTrace
from mamba.checks.mamba3_backward import (
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
from mamba.impl.mamba_ssm.modules.mamba3 import (
    Mamba3DeviceStages,
    Mamba3DeviceWeights,
    allocate_inference_cache,
    mamba3_block_forward,
)
from mamba.impl.mamba_ssm.modules.mamba3_backward import (
    mamba3_backward_gate_skip_into,
    mamba3_backward_qkdot_into,
    mamba3_backward_s16_s15_into,
    mamba3_backward_join_rotary_into,
    mamba3_backward_join_current_into,
    mamba3_backward_angle_into,
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
from mamba.impl.transformers.models.mamba.modeling_mamba import (
    mamba_download,
    mamba_upload,
    mamba_zeros,
)
from mamba.checks.mamba3_fixture import Mamba3Weights


struct Mamba3PrefillGradients(Movable):
    var x: List[Float32]
    var block_norm_weight: List[Float32]
    var in_proj_weight: List[Float32]
    var dt_bias: List[Float32]
    var B_norm_weight: List[Float32]
    var C_norm_weight: List[Float32]
    var B_bias: List[Float32]
    var C_bias: List[Float32]
    var D: List[Float32]
    var out_proj_weight: List[Float32]

    def __init__(out self):
        self.x = List[Float32]()
        self.block_norm_weight = List[Float32]()
        self.in_proj_weight = List[Float32]()
        self.dt_bias = List[Float32]()
        self.B_norm_weight = List[Float32]()
        self.C_norm_weight = List[Float32]()
        self.B_bias = List[Float32]()
        self.C_bias = List[Float32]()
        self.D = List[Float32]()
        self.out_proj_weight = List[Float32]()


def mamba3_prefill_backward(
    weights: Mamba3Weights,
    input: List[Float32],
    grad_output: List[Float32],
    b: Int,
    l: Int,
) raises -> Mamba3PrefillGradients:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("mamba3 backward: only IDENTICAL zero-state prefill is implemented")
    if b <= 0 or l <= 0:
        raise Error("mamba3 backward: B and L must be positive")
    if len(input) != b * l * weights.dims.d_model or len(grad_output) != len(input):
        raise Error("mamba3 backward: input and grad_output lengths must equal B*L*d_model")
    for i in range(len(grad_output)):
        var bits = bitcast[DType.uint32](grad_output[i])
        if (bits & UInt32(0x7f800000)) == UInt32(0x7f800000):
            raise Error("mamba3 backward: non-finite grad_output at flat index " + String(i))
    var dims = weights.dims.copy()
    var m = b * l
    var ctx = DeviceContext()
    var device_weights = Mamba3DeviceWeights(ctx, weights)
    var state = allocate_inference_cache(ctx, b, dims)
    var stages = Mamba3DeviceStages(ctx, b, l, 0, dims)
    var x = mamba_upload(ctx, input)
    var trace = IdentityTrace.disabled()
    mamba3_block_forward(
        ctx, stages, state, device_weights, x, b, l,
        trace, String("m3.backward.tail"),
    )

    # residual.out = x + out_proj, hence the objective cotangent reaches the
    # projection unchanged; the pass continues through every public leaf.
    var d_output = mamba_upload(
        ctx, grad_output
    )
    var d_gate = mamba_zeros(ctx, m * dims.d_inner)
    var d_weight = mamba_zeros(ctx, dims.d_model * dims.d_inner)
    var workspace = mamba_zeros(
        ctx, mamba3_backward_workspace_max_floats(dims, m)
    )
    mamba3_backward_proj_a_into(
        ctx, d_gate, d_output, device_weights.w_out, workspace,
        PROJ3_OUT, dims, m,
    )
    mamba3_backward_proj_b_into(
        ctx, d_weight, d_output, stages.gate_out, workspace,
        PROJ3_OUT, dims, m,
    )
    var tail_cells = m * dims.d_inner
    var head_cells = m * dims.nheads
    var d_skip = mamba_zeros(ctx, tail_cells)
    var d_z = mamba_zeros(ctx, tail_cells)
    var d_v = mamba_zeros(ctx, tail_cells)
    var d_qkdot = mamba_zeros(ctx, head_cells)
    var d_d_product = mamba_zeros(ctx, head_cells)
    var d_d = mamba_zeros(ctx, dims.nheads)
    var ones = mamba_zeros(ctx, mamba3_backward_ones_floats(m))
    ones.enqueue_fill(Float32(1.0))
    mamba3_backward_gate_skip_into(
        ctx, d_skip, d_z, d_v, d_qkdot, d_d_product, d_gate,
        stages.skip_out, stages.qkdot, stages.in_proj,
        device_weights.d_skip, dims, m,
    )
    mamba3_backward_reduce_into(
        ctx, d_d, d_d_product, ones, workspace, RED3_D, dims, m
    )
    var qk_cells = m * dims.nheads * M3_D_STATE
    var d_b_qk = mamba_zeros(ctx, qk_cells)
    var d_c_qk = mamba_zeros(ctx, qk_cells)
    var d_b_bias_qk = mamba_zeros(ctx, qk_cells)
    var d_c_bias_qk = mamba_zeros(ctx, qk_cells)
    var d_gamma_qk = mamba_zeros(ctx, head_cells)
    var d_dt_qk = mamba_zeros(ctx, head_cells)
    var d_trap_qk = mamba_zeros(ctx, head_cells)
    mamba3_backward_qkdot_into(
        ctx, d_b_qk, d_c_qk, d_b_bias_qk, d_c_bias_qk, d_gamma_qk,
        d_dt_qk, d_trap_qk,
        d_qkdot, stages.bcnorm_b, stages.bcnorm_c, device_weights.b_bias,
        device_weights.c_bias, stages.gamma_work, stages.dt_work,
        stages.sig_work, dims, m,
    )
    var state_cells = m * dims.nheads * M3_D_STATE
    var d_q_s16 = mamba_zeros(ctx, state_cells)
    var d_ks_s16 = mamba_zeros(ctx, state_cells)
    var d_v_s16 = mamba_zeros(ctx, tail_cells)
    var d_krot_s15 = mamba_zeros(ctx, state_cells)
    var d_scale_s15 = mamba_zeros(ctx, head_cells)
    mamba3_backward_s16_s15_into(
        ctx, d_q_s16, d_ks_s16, d_v_s16, d_krot_s15, d_scale_s15,
        d_skip, stages.rotq_work, stages.kscale_work, stages.v_work,
        stages.seg_l, stages.rotk_work, stages.scale_work,
        b, l, dims, M3_CHUNK_SIZE,
    )
    var d_value_total = mamba_zeros(ctx, tail_cells)
    var d_gamma_scale = mamba_zeros(ctx, head_cells)
    var d_beta_scale = mamba_zeros(ctx, head_cells)
    var d_qraw_rot = mamba_zeros(ctx, state_cells)
    var d_kraw_rot = mamba_zeros(ctx, state_cells)
    var d_theta_rot = mamba_zeros(ctx, m * dims.nheads * M3_NUM_ROPE_ANGLES)
    mamba3_backward_join_rotary_into(
        ctx, d_value_total, d_gamma_scale, d_beta_scale, d_qraw_rot,
        d_kraw_rot, d_theta_rot, d_v, d_v_s16, d_scale_s15,
        d_q_s16, d_krot_s15, stages.bcnorm_b, stages.bcnorm_c,
        device_weights.b_bias, device_weights.c_bias, stages.theta_out,
        m, dims,
    )
    var d_b_total=mamba_zeros(ctx,state_cells);var d_c_total=mamba_zeros(ctx,state_cells)
    var d_gamma_total=mamba_zeros(ctx,head_cells);var d_dt_total=mamba_zeros(ctx,head_cells);var d_trap_total=mamba_zeros(ctx,head_cells)
    mamba3_backward_join_current_into(ctx,d_b_total,d_c_total,d_gamma_total,d_dt_total,d_trap_total,d_b_qk,d_c_qk,d_kraw_rot,d_qraw_rot,d_gamma_qk,d_gamma_scale,d_dt_qk,d_trap_qk,d_beta_scale,stages.dt_work,stages.sig_work,b,l,dims)
    var d_angle_rate=mamba_zeros(ctx,m*dims.nheads*M3_NUM_ROPE_ANGLES)
    var d_angle_raw=mamba_zeros(ctx,m*M3_NUM_ROPE_ANGLES)
    var d_dt_angle=mamba_zeros(ctx,head_cells)
    mamba3_backward_angle_into(ctx,d_angle_rate,d_angle_raw,d_dt_angle,d_theta_rot,stages.dt_work,stages.in_proj,b,l,dims)
    var d_dt_available=mamba_zeros(ctx,head_cells);var d_dt_raw=mamba_zeros(ctx,head_cells);var d_dt_bias_rows=mamba_zeros(ctx,head_cells);var d_dt_bias=mamba_zeros(ctx,dims.nheads)
    mamba3_backward_dt_partial_into(ctx,d_dt_available,d_dt_raw,d_dt_bias_rows,d_dt_total,d_dt_angle,stages.in_proj,device_weights.dt_bias,m,dims)
    mamba3_backward_reduce_into(ctx,d_dt_bias,d_dt_bias_rows,ones,workspace,RED3_DT_BIAS,dims,m)
    var seg_cells=b*stages.nc*dims.nheads*M3_CHUNK_SIZE*M3_CHUNK_SIZE
    var d_seg=mamba_zeros(ctx,seg_cells);var d_adt_seg=mamba_zeros(ctx,head_cells)
    mamba3_backward_seg_adt_into(ctx,d_seg,d_adt_seg,d_skip,stages.rotq_work,stages.kscale_work,stages.v_work,stages.seg_l,b,l,dims,M3_CHUNK_SIZE)
    var d_a_seg=mamba_zeros(ctx,head_cells);var d_dt_seg=mamba_zeros(ctx,head_cells);var d_dt_with_seg=mamba_zeros(ctx,head_cells)
    mamba3_backward_adt_product_into(ctx,d_a_seg,d_dt_seg,d_dt_with_seg,d_adt_seg,stages.a_out,stages.dt_out,d_dt_available,head_cells)
    var chunk_state_cells=b*stages.nc*dims.nheads*M3_HEADDIM*M3_D_STATE
    var initial_state_cells=b*dims.nheads*M3_HEADDIM*M3_D_STATE
    var d_state_direct=mamba_zeros(ctx,chunk_state_cells);var d_state_total=mamba_zeros(ctx,chunk_state_cells);var d_initial_state=mamba_zeros(ctx,initial_state_cells)
    mamba3_backward_s17_state_into(ctx,d_state_direct,d_state_total,d_initial_state,d_skip,stages.rotq_work,stages.dacs,b,l,dims,M3_CHUNK_SIZE)
    var d_q17=mamba_zeros(ctx,state_cells);var d_dacs_read17=mamba_zeros(ctx,head_cells);var d_k17=mamba_zeros(ctx,state_cells);var d_v17=mamba_zeros(ctx,tail_cells);var d_dacs_rec17=mamba_zeros(ctx,head_cells)
    mamba3_backward_s17_operands_into(ctx,d_q17,d_dacs_read17,d_k17,d_v17,d_dacs_rec17,d_skip,stages.rotq_work,stages.kscale_work,stages.v_work,stages.dacs,stages.pass_states,d_state_total,b,l,dims,M3_CHUNK_SIZE)
    var d_q_join=mamba_zeros(ctx,state_cells);var d_k_join=mamba_zeros(ctx,state_cells);var d_v_join=mamba_zeros(ctx,tail_cells);var d_dacs_join=mamba_zeros(ctx,head_cells)
    mamba3_backward_join_s16_s17_into(ctx,d_q_join,d_k_join,d_v_join,d_dacs_join,d_q_s16,d_q17,d_ks_s16,d_k17,d_value_total,d_v17,d_dacs_read17,d_dacs_rec17,state_cells,tail_cells,head_cells)
    var d_krot_join=mamba_zeros(ctx,state_cells);var d_scale_join=mamba_zeros(ctx,head_cells)
    mamba3_backward_s15_only_into(ctx,d_krot_join,d_scale_join,d_k_join,stages.rotk_work,stages.scale_work,head_cells)
    var d_qraw_join=mamba_zeros(ctx,state_cells);var d_kraw_join=mamba_zeros(ctx,state_cells);var d_theta_join=mamba_zeros(ctx,m*dims.nheads*M3_NUM_ROPE_ANGLES)
    mamba3_backward_rotary_only_into(ctx,d_qraw_join,d_kraw_join,d_theta_join,d_q_join,d_krot_join,stages.bcnorm_b,stages.bcnorm_c,device_weights.b_bias,device_weights.c_bias,stages.theta_out,m*dims.nheads*(M3_D_STATE//2),dims.nheads)
    var d_adt_from_dacs=mamba_zeros(ctx,head_cells);var d_adt_join=mamba_zeros(ctx,head_cells)
    mamba3_backward_dacs_to_adt_into(ctx,d_adt_from_dacs,d_dacs_join,b,l,dims.nheads,M3_CHUNK_SIZE)
    mamba3_backward_join_two_into(ctx,d_adt_join,d_adt_seg,d_adt_from_dacs,head_cells)
    var d_a_join=mamba_zeros(ctx,head_cells);var d_dt_join_adt=mamba_zeros(ctx,head_cells);var d_dt_join_scratch=mamba_zeros(ctx,head_cells);var zero_dt=mamba_zeros(ctx,head_cells)
    mamba3_backward_adt_product_into(ctx,d_a_join,d_dt_join_adt,d_dt_join_scratch,d_adt_join,stages.a_out,stages.dt_out,zero_dt,head_cells)
    var d_a_raw_join=mamba_zeros(ctx,head_cells)
    mamba3_backward_a_heavy_tail_into(ctx,d_a_raw_join,d_a_join,stages.in_proj,m,dims)
    var d_b_join=mamba_zeros(ctx,state_cells);var d_c_join=mamba_zeros(ctx,state_cells);var d_gamma_join=mamba_zeros(ctx,head_cells);var d_dt_join_current=mamba_zeros(ctx,head_cells);var d_trap_join=mamba_zeros(ctx,head_cells);var d_beta_join=mamba_zeros(ctx,head_cells)
    ctx.enqueue_copy(dst_buf=d_beta_join, src_buf=d_scale_join)
    mamba3_backward_join_current_into(ctx,d_b_join,d_c_join,d_gamma_join,d_dt_join_current,d_trap_join,d_b_qk,d_c_qk,d_kraw_join,d_qraw_join,d_gamma_qk,d_scale_join,d_dt_qk,d_trap_qk,d_beta_join,stages.dt_work,stages.sig_work,b,l,dims)
    var d_angle_rate_join=mamba_zeros(ctx,m*dims.nheads*M3_NUM_ROPE_ANGLES);var d_angle_raw_join=mamba_zeros(ctx,m*M3_NUM_ROPE_ANGLES);var d_dt_angle_join=mamba_zeros(ctx,head_cells)
    mamba3_backward_angle_into(ctx,d_angle_rate_join,d_angle_raw_join,d_dt_angle_join,d_theta_join,stages.dt_work,stages.in_proj,b,l,dims)
    var d_dt_join_base=mamba_zeros(ctx,head_cells);mamba3_backward_join_two_into(ctx,d_dt_join_base,d_dt_join_current,d_dt_join_adt,head_cells)
    var d_dt_join_available=mamba_zeros(ctx,head_cells);var d_dt_raw_join=mamba_zeros(ctx,head_cells);var d_dt_bias_rows_join=mamba_zeros(ctx,head_cells);var d_dt_bias_join=mamba_zeros(ctx,dims.nheads)
    mamba3_backward_dt_partial_into(ctx,d_dt_join_available,d_dt_raw_join,d_dt_bias_rows_join,d_dt_join_base,d_dt_angle_join,stages.in_proj,device_weights.dt_bias,m,dims)
    mamba3_backward_reduce_into(ctx,d_dt_bias_join,d_dt_bias_rows_join,ones,workspace,RED3_DT_BIAS,dims,m)
    var d_b_raw=mamba_zeros(ctx,m*M3_D_STATE);var d_c_raw=mamba_zeros(ctx,m*M3_D_STATE);var d_bw_rows=mamba_zeros(ctx,m*M3_D_STATE);var d_cw_rows=mamba_zeros(ctx,m*M3_D_STATE)
    mamba3_backward_bcnorm_into(ctx,d_b_raw,d_bw_rows,d_b_join,stages.in_proj,device_weights.bnorm_w,m,dims,dims.col_b())
    mamba3_backward_bcnorm_into(ctx,d_c_raw,d_cw_rows,d_c_join,stages.in_proj,device_weights.cnorm_w,m,dims,dims.col_c())
    var d_bw=mamba_zeros(ctx,M3_D_STATE);var d_cw=mamba_zeros(ctx,M3_D_STATE);var d_bb=mamba_zeros(ctx,dims.nheads*M3_D_STATE);var d_cb=mamba_zeros(ctx,dims.nheads*M3_D_STATE)
    mamba3_backward_reduce_into(ctx,d_bw,d_bw_rows,ones,workspace,RED3_BNORM_W,dims,m);mamba3_backward_reduce_into(ctx,d_cw,d_cw_rows,ones,workspace,RED3_CNORM_W,dims,m)
    mamba3_backward_reduce_into(ctx,d_bb,d_b_join,ones,workspace,RED3_B_BIAS,dims,m);mamba3_backward_reduce_into(ctx,d_cb,d_c_join,ones,workspace,RED3_C_BIAS,dims,m)
    var d_in_proj=mamba_zeros(ctx,m*dims.d_in_proj());var d_norm=mamba_zeros(ctx,m*dims.d_model);var d_w_in=mamba_zeros(ctx,dims.d_in_proj()*dims.d_model)
    mamba3_backward_pack_in_proj_into(ctx,d_in_proj,d_z,d_v_join,d_b_raw,d_c_raw,d_dt_raw_join,d_a_raw_join,d_trap_join,d_angle_raw_join,m,dims)
    mamba3_backward_proj_a_into(ctx,d_norm,d_in_proj,device_weights.w_in,workspace,PROJ3_IN,dims,m)
    mamba3_backward_proj_b_into(ctx,d_w_in,d_in_proj,stages.norm_out,workspace,PROJ3_IN,dims,m)
    var d_x=mamba_zeros(ctx,m*dims.d_model);var d_norm_w_rows=mamba_zeros(ctx,m*dims.d_model);var d_norm_w=mamba_zeros(ctx,dims.d_model)
    mamba3_backward_block_norm_into(ctx,d_x,d_norm_w_rows,d_norm,d_output,x,stages.norm_sumsq,device_weights.norm_w,m,dims)
    mamba3_backward_reduce_into(ctx,d_norm_w,d_norm_w_rows,ones,workspace,RED3_NORM_W,dims,m)
    ctx.synchronize()

    var result = Mamba3PrefillGradients()
    result.x = mamba_download(ctx, d_x, m*dims.d_model)
    result.block_norm_weight = mamba_download(ctx, d_norm_w, dims.d_model)
    result.in_proj_weight = mamba_download(ctx, d_w_in, dims.d_in_proj()*dims.d_model)
    result.dt_bias = mamba_download(ctx, d_dt_bias_join, dims.nheads)
    result.B_norm_weight = mamba_download(ctx, d_bw, M3_D_STATE)
    result.C_norm_weight = mamba_download(ctx, d_cw, M3_D_STATE)
    result.B_bias = mamba_download(ctx, d_bb, dims.nheads*M3_D_STATE)
    result.C_bias = mamba_download(ctx, d_cb, dims.nheads*M3_D_STATE)
    result.D = mamba_download(ctx, d_d, dims.nheads)
    result.out_proj_weight = mamba_download(ctx, d_weight, dims.d_model*dims.d_inner)

    # Explicit ownership extends all GPU buffers through synchronization.
    _ = d_norm_w^
    _ = d_norm_w_rows^
    _ = d_x^
    _ = d_w_in^
    _ = d_norm^
    _ = d_in_proj^
    _ = d_cb^
    _ = d_bb^
    _ = d_cw^
    _ = d_bw^
    _ = d_cw_rows^
    _ = d_bw_rows^
    _ = d_c_raw^
    _ = d_b_raw^
    _ = d_dt_bias_join^
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
    _ = d_d^
    _ = d_d_product^
    _ = d_qkdot^
    _ = d_v^
    _ = d_z^
    _ = d_skip^
    _ = workspace^
    _ = d_weight^
    _ = d_gate^
    _ = d_output^
    _ = x^
    _ = stages^
    _ = state^
    _ = device_weights^
    _ = ctx^
    return result^
