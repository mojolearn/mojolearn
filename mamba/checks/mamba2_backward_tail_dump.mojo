# SPDX-License-Identifier: Apache-2.0
"""Dump the implemented Mamba-2 backward tail for the external oracle.

This runner emits ONE comparable parameter gradient and claims ONE seam. It
must not be called a whole-block backward gate. The selected fixture and the
dense dyadic output cotangent exactly match ``tools/mamba_gradient_oracle.py``.

Usage, from the repository root (the directory must already exist):

    MOJOLEARN_MAMBA_GRAD_DUMP=/tmp/mamba2-mojo \
      mojo run -I . mamba/checks/mamba2_backward_tail_dump.mojo
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
)
from mamba.impl.mamba_ssm.modules.ssd_minimal import m2_q_eff
from mamba.impl.mamba_ssm.ops.mamba2_ssd_backward import (
    Mamba2SSDBackwardState,
    Mamba2SSDScaleReduction,
    mamba2_reduce_scale_product_into,
    mamba2_reverse_chunk_state_into,
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
    # This is the gradient oracle's default Mamba-2 case.
    comptime case_k = 1  # m2_base_b2_l4_d32
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
    var scale_reduction = Mamba2SSDScaleReduction(
        ctx, fixture.b, stages.nc, dims.nheads, m2_q_eff()
    )
    mamba2_reduce_scale_product_into(
        ctx,
        scale_reduction,
        ssd_backward,
        stages.dacs,
        fixture.b,
        stages.nc,
        dims.nheads,
        m2_q_eff(),
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
    with open(dump_dir + "/dump_manifest.json", "w") as fh:
        fh.write(
            "{\"schema\":\"mojolearn.mamba.gradient-dump.v1\","
            "\"family\":\"mamba2\",\"case\":\"m2_base_b2_l4_d32\","
            "\"objective\":\"signed_dyadic_weight_v1\","
            "\"tensors\":[\"out_proj.weight\",\"stage.gnorm.out\","
            "\"stage.gnorm.gate\",\"norm.weight\",\"stage.skip.out\","
            "\"stage.in_proj.z\",\"stage.scan.y\","
            "\"partial.silu.x.from_D\",\"D\","
            "\"stage.pass.states.direct\",\"stage.pass.states.total\","
            "\"stage.cstate.out\",\"stage.initial_state\","
            "\"stage.scale.product\",\"partial.dacs.from_state\"]}\n"
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
    _ = d_residual^
    _ = stages^
    _ = state^
    _ = dweights^
    _ = x^
