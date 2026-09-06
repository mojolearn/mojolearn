# SPDX-License-Identifier: Apache-2.0
"""Dump complete Mamba-1 public-prefill gradients for named corpus fixtures."""

from std.memory import bitcast
from std.os import getenv

from mamba.checks.mamba_fixture import MambaDims, corpus_case, corpus_case_weights, corpus_case_x
from mamba.impl.transformers.models.mamba.modeling_mamba_prefill_backward import mamba1_prefill_backward


def _objective_cotangent(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var numerator = (i * 37 + 11) % 31 - 15
        if numerator == 0:
            numerator = 1
        out.append(Float32(numerator) / Float32(16.0))
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
    var case_k = 1
    var case_name = String("base_b2_l4_d8")
    var requested_case = String(getenv("MOJOLEARN_MAMBA1_GRAD_CASE"))
    if requested_case != "":
        if requested_case != "base_b1_l64_d8":
            raise Error("MOJOLEARN_MAMBA1_GRAD_CASE supports only base_b1_l64_d8")
        case_k = 3
        case_name = requested_case
    var fixture = corpus_case(case_k)
    var weights = corpus_case_weights(case_k)
    var dims = MambaDims.of(fixture.d_model)
    var m = fixture.b * fixture.l
    var output = String(getenv("MOJOLEARN_MAMBA_GRAD_DUMP"))
    if output == "":
        raise Error("set MOJOLEARN_MAMBA_GRAD_DUMP to an existing directory")

    var gradients = mamba1_prefill_backward(
        weights, corpus_case_x(case_k), _objective_cotangent(m * dims.d_model),
        fixture.b, fixture.l,
    )

    _write_f32(
        output + "/grad.out_proj.weight.f32",
        gradients.out_proj_weight,
    )
    _write_f32(
        output + "/grad.D.f32",
        gradients.D,
    )
    _write_f32(
        output + "/grad.A_log.f32",
        gradients.A_log,
    )
    _write_f32(
        output + "/grad.dt_proj.weight.f32",
        gradients.dt_proj_weight,
    )
    _write_f32(
        output + "/grad.dt_proj.bias.f32",
        gradients.dt_proj_bias,
    )
    _write_f32(
        output + "/grad.x_proj.weight.f32",
        gradients.x_proj_weight,
    )
    _write_f32(output + "/grad.conv1d.weight.f32", gradients.conv1d_weight)
    _write_f32(
        output + "/grad.conv1d.bias.f32",
        gradients.conv1d_bias,
    )
    _write_f32(
        output + "/grad.in_proj.weight.f32",
        gradients.in_proj_weight,
    )
    _write_f32(
        output + "/grad.norm.weight.f32",
        gradients.norm_weight,
    )
    _write_f32(
        output + "/grad.x.f32",
        gradients.x,
    )
    _write_f32(
        output + "/stage.dB.f32", gradients.stage_dB
    )
    _write_f32(
        output + "/stage.dC.f32", gradients.stage_dC
    )
    with open(output + "/dump_manifest.json", "w") as fh:
        fh.write(
            "{\"schema\":\"mojolearn.mamba.gradient-dump.v1\","
            "\"family\":\"mamba1\",\"case\":\"" + case_name + "\","
            "\"objective\":\"signed_dyadic_weight_v1\","
            "\"producer\":\"mamba1-device-whole-pass-v1\","
            "\"partial\":false,"
            "\"public_prefill_leaves\":[\"x\",\"norm.weight\","
            "\"in_proj.weight\",\"conv1d.weight\",\"conv1d.bias\","
            "\"out_proj.weight\",\"D\",\"A_log\",\"dt_proj.weight\","
            "\"dt_proj.bias\",\"x_proj.weight\"],"
            "\"diagnostics\":[\"stage.dB\",\"stage.dC\"],"
            "\"tensors\":[\"x\",\"norm.weight\","
            "\"in_proj.weight\",\"conv1d.weight\",\"conv1d.bias\","
            "\"out_proj.weight\",\"D\","
            "\"A_log\",\"dt_proj.weight\",\"dt_proj.bias\","
            "\"x_proj.weight\",\"stage.dB\",\"stage.dC\"]}\n"
        )
    print(
        "MAMBA1 BACKWARD DEVICE WHOLE PASS: emitted all 11 external-oracle"
        " gradients plus diagnostic dB/dC"
    )
