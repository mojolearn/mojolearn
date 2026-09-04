# SPDX-License-Identifier: Apache-2.0
"""Dump the implemented Mamba-3 output-projection backward boundary."""

from std.memory import bitcast
from std.os import getenv

from max.gpu.host import DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.identity_trace import IdentityTrace
from mamba.checks.mamba3_backward import (
    PROJ3_OUT,
    mamba3_backward_proj_a_into,
    mamba3_backward_proj_b_into,
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
    ctx.synchronize()

    _write_f32(
        output + "/grad.stage.gate.out.f32",
        mamba_download(ctx, d_gate, m * dims.d_inner),
    )
    _write_f32(
        output + "/grad.out_proj.weight.f32",
        mamba_download(ctx, d_weight, dims.d_model * dims.d_inner),
    )
    with open(output + "/dump_manifest.json", "w") as fh:
        fh.write(
            "{\"schema\":\"mojolearn.mamba.gradient-dump.v1\","
            + "\"family\":\"mamba3\",\"case\":\"m3_base_b2_l4_d32\","
            + "\"objective\":\"signed_dyadic_weight_v1\","
            + "\"mode\":\"partial-output-projection-tail\","
            + "\"tensors\":[\"stage.gate.out\",\"out_proj.weight\"]}\n"
        )

    _ = workspace^
    _ = d_weight^
    _ = d_gate^
    _ = d_output^
    _ = x^
    _ = stages^
    _ = state^
    _ = device_weights^
