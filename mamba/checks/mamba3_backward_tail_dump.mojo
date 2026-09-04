# SPDX-License-Identifier: Apache-2.0
"""Dump the implemented Mamba-3 output-projection backward boundary."""

from std.memory import bitcast
from std.os import getenv

from max.gpu.host import DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.identity_trace import IdentityTrace
from mamba.checks.mamba3_backward import (
    PROJ3_OUT,
    RED3_D,
    mamba3_backward_ones_floats,
    mamba3_backward_proj_a_into,
    mamba3_backward_proj_b_into,
    mamba3_backward_reduce_into,
    mamba3_backward_workspace_max_floats,
)
from mamba.checks.mamba3_fixture import (
    m3_case_weights,
    m3_case_x,
    m3_corpus_case,
)
from mamba.impl.mamba_ssm.modules.mamba3 import (
    Mamba3DeviceStages,
    Mamba3DeviceWeights,
    allocate_inference_cache,
    mamba3_block_forward,
)
from mamba.impl.mamba_ssm.modules.mamba3_backward import (
    mamba3_backward_gate_skip_into,
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
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error(
            "mamba3_backward_tail_dump is the pinned IDENTICAL boundary; "
            + "FAST backward is not implemented and must not be mislabeled"
        )
    var output = String(getenv("MOJOLEARN_MAMBA_GRAD_DUMP"))
    if output == "":
        raise Error("set MOJOLEARN_MAMBA_GRAD_DUMP to an existing directory")

    var case_k = 1
    var fixture = m3_corpus_case(case_k)
    var weights = m3_case_weights(case_k)
    var dims = weights.dims.copy()
    var m = fixture.b * fixture.l
    var ctx = DeviceContext()
    var device_weights = Mamba3DeviceWeights(ctx, weights)
    var state = allocate_inference_cache(ctx, fixture.b, dims)
    var stages = Mamba3DeviceStages(ctx, fixture.b, fixture.l, 0, dims)
    var x = mamba_upload(ctx, m3_case_x(case_k))
    var trace = IdentityTrace.disabled()
    mamba3_block_forward(
        ctx, stages, state, device_weights, x, fixture.b, fixture.l,
        trace, String("m3.backward.tail"),
    )

    # residual.out = x + out_proj, hence the objective cotangent reaches the
    # projection unchanged.  This boundary intentionally stops at gate.out.
    var d_output = mamba_upload(
        ctx, _objective_cotangent(m * dims.d_model)
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
    ctx.synchronize()

    _write_f32(
        output + "/grad.stage.gate.out.f32",
        mamba_download(ctx, d_gate, m * dims.d_inner),
    )
    _write_f32(
        output + "/grad.out_proj.weight.f32",
        mamba_download(ctx, d_weight, dims.d_model * dims.d_inner),
    )
    _write_f32(output + "/grad.stage.skip.out.f32", mamba_download(ctx, d_skip, tail_cells))
    _write_f32(output + "/grad.stage.in_proj.z.f32", mamba_download(ctx, d_z, tail_cells))
    _write_f32(output + "/grad.partial.in_proj.x.from_skip.f32", mamba_download(ctx, d_v, tail_cells))
    _write_f32(output + "/grad.stage.qkdot.out.f32", mamba_download(ctx, d_qkdot, head_cells))
    _write_f32(output + "/grad.D.f32", mamba_download(ctx, d_d, dims.nheads))
    with open(output + "/dump_manifest.json", "w") as fh:
        fh.write(
            "{\"schema\":\"mojolearn.mamba.gradient-dump.v1\","
            + "\"family\":\"mamba3\",\"case\":\"m3_base_b2_l4_d32\","
            + "\"objective\":\"signed_dyadic_weight_v1\","
            + "\"mode\":\"partial-output-gate-skip-tail\","
            + "\"tensors\":[\"stage.gate.out\",\"out_proj.weight\","
            + "\"stage.skip.out\",\"stage.in_proj.z\","
            + "\"partial.in_proj.x.from_skip\",\"stage.qkdot.out\",\"D\"]}\n"
        )

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
