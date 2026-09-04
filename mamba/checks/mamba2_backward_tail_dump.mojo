# SPDX-License-Identifier: Apache-2.0
"""Dump the implemented Mamba-2 backward tail for the external oracle.

This runner emits ONE comparable parameter gradient and claims ONE seam. It
must not be called a whole-block backward gate. The selected fixture and the
dense dyadic output cotangent exactly match ``tools/mamba_gradient_oracle.py``.

Usage, from the repository root (the directory must already exist):

    MOJOLEARN_MAMBA_GRAD_DUMP=/tmp/mamba2-mojo \
      mojo run -I . mamba/checks/mamba2_backward_tail_dump.mojo

Set ``MOJOLEARN_MAMBA2_GRAD_CASE=m2_base_b1_l257_d64`` for the dedicated
two-chunk S17 recurrence witness.  No other alternate case is accepted.
"""

from std.memory import bitcast
from std.os import getenv

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from mamba.checks.mamba2_fixture import (
    M2_D_STATE,
    M2_HEADDIM,
    m2_case_weights,
    m2_case_x,
    m2_corpus_case,
)
from mamba.impl.mamba_ssm.modules.mamba2 import (
    Mamba2DeviceStages,
    Mamba2DeviceWeights,
    allocate_inference_cache,
    mamba2_block_forward,
)
from mamba.impl.mamba_ssm.modules.mamba2_backward import (
    Mamba2BackwardTail,
    mamba2_backward_d_skip_into,
    mamba2_backward_gnorm_into,
    mamba2_backward_silu_gate_into,
    mamba2_backward_tail_into,
    mamba2_backward_input_projection_into,
    mamba2_backward_block_norm_into,
)
from mamba.impl.mamba_ssm.modules.ssd_minimal import m2_q_eff
from mamba.impl.mamba_ssm.ops.mamba2_ssd_backward import (
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
from mamba.impl.transformers.models.mamba.modeling_mamba import (
    mamba_download,
    mamba_upload,
    mamba_zeros,
)


def _objective_cotangent(n: Int) -> List[Float32]:
    """Derivative of signed_dyadic_weight_v1, exactly representable in f32."""
    var out = List[Float32]()
    for i in range(n):
        var numerator = (i * 37 + 11) % 31 - 15
        if numerator == 0:
            numerator = 1
        out.append(Float32(numerator) / 16.0)
    return out^


def _write_f32(path: String, values: List[Float32]) raises:
    var bytes = List[UInt8]()
    for i in range(len(values)):
        var u = bitcast[DType.uint32](values[i])
        bytes.append(UInt8(Int(u & UInt32(0xFF))))
        bytes.append(UInt8(Int((u >> UInt32(8)) & UInt32(0xFF))))
        bytes.append(UInt8(Int((u >> UInt32(16)) & UInt32(0xFF))))
        bytes.append(UInt8(Int((u >> UInt32(24)) & UInt32(0xFF))))
    with open(path, "w") as fh:
        fh.write_bytes(Span(bytes))


def main() raises:
    # Default remains the inexpensive tail fixture.  The one supported
    # alternate crosses Q=256 exactly once and is the S17 carry witness.
    var case_k = 1
    var case_name = String("m2_base_b2_l4_d32")
    var requested_case = String(getenv("MOJOLEARN_MAMBA2_GRAD_CASE"))
    if requested_case != "":
        if requested_case != "m2_base_b1_l257_d64":
            raise Error(
                "MOJOLEARN_MAMBA2_GRAD_CASE supports only "
                "m2_base_b1_l257_d64"
            )
        case_k = 4
        case_name = requested_case
    var fixture = m2_corpus_case(case_k)
    var weights = m2_case_weights(case_k)
    var dims = weights.dims.copy()
    var m = fixture.b * fixture.l
    var dump_dir = String(getenv("MOJOLEARN_MAMBA_GRAD_DUMP"))
    if dump_dir == "":
        raise Error(
            "set MOJOLEARN_MAMBA_GRAD_DUMP to an existing output directory"
        )

    var ctx = DeviceContext()
    var dweights = Mamba2DeviceWeights(ctx, weights)
    var state = allocate_inference_cache(ctx, fixture.b, dims)
    var stages = Mamba2DeviceStages(ctx, fixture.b, fixture.l, 0, dims)
    var x = mamba_upload(ctx, m2_case_x(case_k))
    var trace = IdentityTrace.disabled()
    mamba2_block_forward(
        ctx,
        stages,
        state,
        dweights,
        x,
        fixture.b,
        fixture.l,
        fixture.dt_lo,
        fixture.dt_hi,
        trace,
        String("m2.backward.tail"),
    )

    var d_residual = mamba_upload(
        ctx, _objective_cotangent(m * dims.d_model)
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
        ctx, fixture.b, stages.nc, dims.nheads
    )
    mamba2_s18_direct_dpass_into(
        ctx,
        ssd_backward,
        tail.d_scan,
        stages.xbc_work,
        stages.pass_states,
        stages.dacs,
        fixture.b,
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
        ctx, fixture.b * dims.nheads * M2_HEADDIM * M2_D_STATE
    )
    mamba2_reverse_chunk_state_into(
        ctx,
        ssd_backward,
        d_final,
        stages.pass_states,
        stages.dacs,
        fixture.b,
        dims.nheads,
        stages.nc,
        m2_q_eff(),
    )
    mamba2_cstate_ddecay_into(
        ctx, ssd_backward, stages.xd_work, stages.xbc_work, fixture.b,
        stages.t_work, dims.nheads, dims.d_inner, dims.conv_dim(),
        stages.nc, m2_q_eff(),
    )
    var scale_reduction = Mamba2SSDScaleReduction(
        ctx, fixture.b, stages.nc, dims.nheads, m2_q_eff()
    )
    mamba2_reduce_scale_product_into(
        ctx,
        scale_reduction,
        ssd_backward,
        stages.dacs,
        stages.decay,
        fixture.b,
        stages.nc,
        dims.nheads,
        m2_q_eff(),
    )
    var discretize_backward = Mamba2SSDDiscretizeBackward(
        ctx, fixture.b, stages.t_work, dims.nheads
    )
    mamba2_reverse_cumsum_and_da_into(
        ctx, discretize_backward, scale_reduction.d_dacs_total,
        stages.dt_work, stages.a_out, fixture.b, stages.t_work,
        dims.nheads, stages.nc, m2_q_eff(),
    )
    mamba2_ydiag_xd_and_partial_dt_into(
        ctx, discretize_backward, tail.d_scan, stages.cb_g, stages.seg_l,
        stages.xd_work,
        stages.xbc_work, stages.dt_work, stages.a_out, stages.dtraw_work, dweights.dt_bias,
        ssd_backward.d_cstate, stages.decay,
        fixture.b, stages.t_work, dims.nheads, dims.d_inner,
        dims.conv_dim(), stages.nc, m2_q_eff(), fixture.dt_lo, fixture.dt_hi,
    )
    mamba2_postconv_merge_into(
        ctx, discretize_backward, ssd_backward, tail.d_x_from_d,
        fixture.b, stages.t_work, dims.nheads,
    )
    var conv_backward = Mamba2ConvBackward(ctx, fixture.b, fixture.l, dims.conv_dim())
    mamba2_conv_backward_prefill_into(
        ctx, conv_backward, discretize_backward, stages.conv_out,
        stages.in_proj, dweights.conv_w, fixture.b, fixture.l, dims.d_inner,
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
    _write_f32(
        dump_dir + "/grad.out_proj.weight.f32",
        mamba_download(ctx, tail.d_w_out, dims.d_model * dims.d_inner),
    )
    _write_f32(
        dump_dir + "/grad.stage.gnorm.out.f32",
        mamba_download(ctx, tail.d_gnorm, m * dims.d_inner),
    )
    _write_f32(
        dump_dir + "/grad.stage.gnorm.gate.f32",
        mamba_download(ctx, tail.d_gate, m * dims.d_inner),
    )
    _write_f32(
        dump_dir + "/grad.norm.weight.f32",
        mamba_download(ctx, tail.d_gnorm_w, dims.d_inner),
    )
    _write_f32(
        dump_dir + "/grad.stage.skip.out.f32",
        mamba_download(ctx, tail.d_skip, m * dims.d_inner),
    )
    _write_f32(
        dump_dir + "/grad.stage.in_proj.z.f32",
        mamba_download(ctx, tail.d_z, m * dims.d_inner),
    )
    _write_f32(
        dump_dir + "/grad.stage.scan.y.f32",
        mamba_download(ctx, tail.d_scan, m * dims.d_inner),
    )
    _write_f32(
        dump_dir + "/grad.partial.silu.x.from_D.f32",
        mamba_download(ctx, tail.d_x_from_d, m * dims.d_inner),
    )
    _write_f32(
        dump_dir + "/grad.D.f32",
        mamba_download(ctx, tail.d_d, dims.nheads),
    )
    _write_f32(
        dump_dir + "/grad.stage.pass.states.direct.f32",
        mamba_download(
            ctx,
            ssd_backward.direct_d_pass,
            fixture.b
            * stages.nc
            * dims.nheads
            * M2_HEADDIM
            * M2_D_STATE,
        ),
    )
    var state_cells = (
        fixture.b * stages.nc * dims.nheads * M2_HEADDIM * M2_D_STATE
    )
    _write_f32(
        dump_dir + "/grad.stage.pass.states.total.f32",
        mamba_download(ctx, ssd_backward.d_pass, state_cells),
    )
    _write_f32(
        dump_dir + "/grad.stage.cstate.out.f32",
        mamba_download(ctx, ssd_backward.d_cstate, state_cells),
    )
    _write_f32(
        dump_dir + "/grad.stage.scale.product.f32",
        mamba_download(ctx, ssd_backward.d_scale_product, state_cells),
    )
    _write_f32(
        dump_dir + "/grad.partial.decay.from_cstate.f32",
        mamba_download(ctx, ssd_backward.d_decay_cstate,
            fixture.b * dims.nheads * stages.nc * m2_q_eff()),
    )
    _write_f32(
        dump_dir + "/grad.stage.initial_state.f32",
        mamba_download(
            ctx,
            ssd_backward.d_initial,
            fixture.b * dims.nheads * M2_HEADDIM * M2_D_STATE,
        ),
    )
    _write_f32(
        dump_dir + "/grad.partial.dacs.from_state.f32",
        mamba_download(
            ctx,
            scale_reduction.d_dacs_state,
            fixture.b * dims.nheads * stages.nc * m2_q_eff(),
        ),
    )
    _write_f32(
        dump_dir + "/grad.partial.dacs.from_decay.f32",
        mamba_download(ctx, scale_reduction.d_dacs_decay,
            fixture.b * dims.nheads * stages.nc * m2_q_eff()),
    )
    _write_f32(
        dump_dir + "/grad.partial.C.from_yoff.f32",
        mamba_download(
            ctx, ssd_backward.d_c_yoff,
            fixture.b * stages.t_work * M2_D_STATE,
        ),
    )
    _write_f32(
        dump_dir + "/grad.partial.dacs.from_yoff.f32",
        mamba_download(
            ctx, ssd_backward.d_dacs_yoff,
            fixture.b * dims.nheads * stages.nc * m2_q_eff(),
        ),
    )
    _write_f32(
        dump_dir + "/grad.partial.dacs.total.f32",
        mamba_download(
            ctx, scale_reduction.d_dacs_total,
            fixture.b * dims.nheads * stages.nc * m2_q_eff(),
        ),
    )
    _write_f32(
        dump_dir + "/grad.partial.da.total.f32",
        mamba_download(
            ctx, discretize_backward.d_da_total,
            fixture.b * stages.t_work * dims.nheads,
        ),
    )
    _write_f32(
        dump_dir + "/grad.partial.da.from_seg.f32",
        mamba_download(
            ctx, discretize_backward.d_da_seg,
            fixture.b * stages.t_work * dims.nheads,
        ),
    )
    _write_f32(
        dump_dir + "/grad.partial.A.from_da.f32",
        mamba_download(ctx, discretize_backward.d_a, dims.nheads),
    )
    _write_f32(
        dump_dir + "/grad.partial.A_log.from_current_ssd.f32",
        mamba_download(ctx, discretize_backward.d_a_log, dims.nheads),
    )
    _write_f32(
        dump_dir + "/grad.partial.dt.from_da.f32",
        mamba_download(
            ctx, discretize_backward.d_dt,
            fixture.b * stages.t_work * dims.nheads,
        ),
    )
    _write_f32(
        dump_dir + "/grad.partial.xd.from_ydiag.f32",
        mamba_download(
            ctx, discretize_backward.d_xd_ydiag,
            fixture.b * stages.t_work * dims.d_inner,
        ),
    )
    _write_f32(
        dump_dir + "/grad.partial.x.from_xd.f32",
        mamba_download(
            ctx, discretize_backward.d_x_from_xd,
            fixture.b * stages.t_work * dims.d_inner,
        ),
    )
    _write_f32(
        dump_dir + "/grad.partial.dt.from_xd.f32",
        mamba_download(
            ctx, discretize_backward.d_dt_from_xd,
            fixture.b * stages.t_work * dims.nheads,
        ),
    )
    _write_f32(
        dump_dir + "/grad.partial.dt.merged.f32",
        mamba_download(
            ctx, discretize_backward.d_dt_merged,
            fixture.b * stages.t_work * dims.nheads,
        ),
    )
    _write_f32(
        dump_dir + "/grad.partial.cb.G.from_ydiag.f32",
        mamba_download(ctx, discretize_backward.d_cb_ydiag,
            fixture.b * stages.nc * m2_q_eff() * m2_q_eff()),
    )
    _write_f32(
        dump_dir + "/grad.partial.seg.L.from_ydiag.f32",
        mamba_download(ctx, discretize_backward.d_seg_ydiag,
            fixture.b * stages.nc * dims.nheads * m2_q_eff() * m2_q_eff()),
    )
    _write_f32(
        dump_dir + "/grad.partial.B.from_cb.f32",
        mamba_download(ctx, discretize_backward.d_b_cb,
            fixture.b * stages.t_work * M2_D_STATE),
    )
    _write_f32(dump_dir + "/grad.partial.xd.from_cstate.f32",
        mamba_download(ctx, discretize_backward.d_xd_cstate, fixture.b*stages.t_work*dims.d_inner))
    _write_f32(dump_dir + "/grad.partial.xd.total.f32",
        mamba_download(ctx, discretize_backward.d_xd_total, fixture.b*stages.t_work*dims.d_inner))
    _write_f32(dump_dir + "/grad.partial.B.from_cstate.f32",
        mamba_download(ctx, discretize_backward.d_b_cstate, fixture.b*stages.t_work*M2_D_STATE))
    _write_f32(dump_dir + "/grad.partial.B.total.f32",
        mamba_download(ctx, discretize_backward.d_b_total, fixture.b*stages.t_work*M2_D_STATE))
    _write_f32(
        dump_dir + "/grad.partial.C.from_cb.f32",
        mamba_download(ctx, discretize_backward.d_c_cb,
            fixture.b * stages.t_work * M2_D_STATE),
    )
    _write_f32(dump_dir + "/grad.partial.C.total.f32",
        mamba_download(ctx, discretize_backward.d_c_total, fixture.b*stages.t_work*M2_D_STATE))
    _write_f32(dump_dir + "/grad.partial.silu.x.total.f32",
        mamba_download(ctx, discretize_backward.d_x_total, fixture.b*stages.t_work*dims.d_inner))
    _write_f32(
        dump_dir + "/grad.partial.dt_raw.merged.f32",
        mamba_download(
            ctx, discretize_backward.d_dtraw,
            fixture.b * stages.t_work * dims.nheads,
        ),
    )
    _write_f32(dump_dir + "/grad.partial.conv.input.f32", mamba_download(ctx, conv_backward.d_in_xbc, m*dims.conv_dim()))
    _write_f32(dump_dir + "/grad.conv1d.weight.f32", mamba_download(ctx, conv_backward.d_w, dims.conv_dim()*4))
    _write_f32(dump_dir + "/grad.conv1d.bias.f32", mamba_download(ctx, conv_backward.d_b, dims.conv_dim()))
    _write_f32(dump_dir + "/grad.partial.in_proj.packed.f32", mamba_download(ctx, tail.d_in_proj, m*dims.d_in_proj()))
    _write_f32(dump_dir + "/grad.stage.norm.out.f32", mamba_download(ctx, tail.d_norm, m*dims.d_model))
    _write_f32(dump_dir + "/grad.in_proj.weight.f32", mamba_download(ctx, tail.d_w_in, dims.d_in_proj()*dims.d_model))
    _write_f32(dump_dir + "/grad.x.f32", mamba_download(ctx, tail.d_block_x, m*dims.d_model))
    _write_f32(dump_dir + "/grad.block_norm.weight.f32", mamba_download(ctx, tail.d_block_w, dims.d_model))
    _write_f32(
        dump_dir + "/grad.partial.dt_bias.merged.f32",
        mamba_download(ctx, discretize_backward.d_dt_bias, dims.nheads),
    )
    with open(dump_dir + "/dump_manifest.json", "w") as fh:
        fh.write(
            "{\"schema\":\"mojolearn.mamba.gradient-dump.v1\","
            "\"family\":\"mamba2\",\"case\":\""
            + case_name
            + "\","
            "\"objective\":\"signed_dyadic_weight_v1\","
            "\"tensors\":[\"out_proj.weight\",\"stage.gnorm.out\","
            "\"stage.gnorm.gate\",\"norm.weight\",\"stage.skip.out\","
            "\"stage.in_proj.z\",\"stage.scan.y\","
            "\"partial.silu.x.from_D\",\"D\","
            "\"stage.pass.states.direct\",\"stage.pass.states.total\","
            "\"stage.cstate.out\",\"stage.initial_state\","
            "\"stage.scale.product\",\"partial.decay.from_cstate\","
            "\"partial.dacs.from_state\",\"partial.dacs.from_decay\","
            "\"partial.C.from_yoff\",\"partial.dacs.from_yoff\","
            "\"partial.dacs.total\",\"partial.da.from_seg\",\"partial.da.total\","
            "\"partial.A.from_da\",\"partial.A_log.from_current_ssd\","
            "\"partial.dt.from_da\","
            "\"partial.xd.from_ydiag\",\"partial.x.from_xd\","
            "\"partial.cb.G.from_ydiag\",\"partial.seg.L.from_ydiag\","
            "\"partial.B.from_cb\",\"partial.C.from_cb\","
            "\"partial.xd.from_cstate\",\"partial.xd.total\","
            "\"partial.B.from_cstate\",\"partial.B.total\","
            "\"partial.C.total\",\"partial.silu.x.total\","
            "\"partial.conv.input\",\"conv1d.weight\",\"conv1d.bias\","
            "\"partial.in_proj.packed\",\"stage.norm.out\",\"in_proj.weight\","
            "\"x\",\"block_norm.weight\","
            "\"partial.dt.from_xd\",\"partial.dt.merged\","
            "\"partial.dt_raw.merged\",\"partial.dt_bias.merged\"]}\n"
        )
    print(
        "MAMBA2 BACKWARD PARTIAL: emitted output projection + gated RMSNorm"
        " + SiLU gate + D-skip + S18/S17 pass-state gradients;"
        " intra-chunk SSD, convolution, input projection, block norm, and"
        " full dx remain"
    )
    _ = tail^
    _ = ssd_backward^
    _ = d_final^
    _ = scale_reduction^
    _ = discretize_backward^
    _ = conv_backward^
    _ = d_residual^
    _ = stages^
    _ = state^
    _ = dweights^
    _ = x^
