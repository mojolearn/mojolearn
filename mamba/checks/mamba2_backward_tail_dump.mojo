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
    mamba2_backward_tail_into,
)
from mamba.impl.transformers.models.mamba.modeling_mamba import (
    mamba_download,
    mamba_upload,
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
    ctx.synchronize()
    _write_f32(
        dump_dir + "/grad.out_proj.weight.f32",
        mamba_download(ctx, tail.d_w_out, dims.d_model * dims.d_inner),
    )
    _write_f32(
        dump_dir + "/grad.stage.gnorm.out.f32",
        mamba_download(ctx, tail.d_gnorm, m * dims.d_inner),
    )
    with open(dump_dir + "/dump_manifest.json", "w") as fh:
        fh.write(
            "{\"schema\":\"mojolearn.mamba.gradient-dump.v1\","
            "\"family\":\"mamba2\",\"case\":\"m2_base_b2_l4_d32\","
            "\"objective\":\"signed_dyadic_weight_v1\","
            "\"tensors\":[\"out_proj.weight\",\"stage.gnorm.out\"]}\n"
        )
    print(
        "MAMBA2 BACKWARD PARTIAL: emitted d_gnorm and out_proj.weight; SSD, gated"
        " norm, convolution, input projection, block norm, and full dx remain"
    )
    _ = tail^
    _ = d_residual^
    _ = stages^
    _ = state^
    _ = dweights^
    _ = x^
