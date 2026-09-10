# SPDX-License-Identifier: Apache-2.0
"""Mamba-1 reference decode gate plus the public device decode route."""
from std.sys import argv
from std.memory import bitcast
from core.identity_trace import IdentityTrace
from max.gpu.host import DeviceContext
from mamba.impl.modules.mamba_simple import check_reference_decode as reference_gate
from mamba.checks.mamba_check import (
    clause_d, run_step, stage_names, stage_kind, token_slice, compare_stage, KIND_TOKEN,
)
from mamba.impl.modules.mamba_simple import allocate_inference_cache, mamba_step
from mamba.impl.modeling.modeling_mamba import (
    MambaDeviceWeights, MambaDeviceStages, mamba_download,
)
from mamba.checks.mamba_fixture import corpus_case, corpus_case_weights, corpus_case_x


def check_device_continuation(ctx: DeviceContext) raises:
    var fixture = corpus_case(6)  # B=2, L=16, d_model=16; window rolls.
    var w = corpus_case_weights(6)
    var x = corpus_case_x(6)
    var b = fixture.b
    var l = fixture.l
    var dm = fixture.d_model
    var dims = w.dims.copy()
    var dw = MambaDeviceWeights(ctx, w)
    var full_state = allocate_inference_cache(ctx, b, dims)
    var cache = allocate_inference_cache(ctx, b, dims)
    var zeros = mamba_download(ctx, cache.conv_win, len(cache.conv_win))
    var zeros_h = mamba_download(ctx, cache.h, len(cache.h))
    for value in zeros:
        if bitcast[DType.uint32](value) != 0:
            raise Error("decode cache conv window must start at +0")
    for value in zeros_h:
        if bitcast[DType.uint32](value) != 0:
            raise Error("decode cache h must start at +0")
    var trace = IdentityTrace.disabled()
    var full = run_step(ctx, dw, full_state, x, b, l, dims, trace, "full")
    var prefix = List[Float32]()
    var token = List[Float32]()
    for row in range(b):
        for t in range(l - 1):
            for j in range(dm):
                prefix.append(x[(row * l + t) * dm + j])
        for j in range(dm):
            token.append(x[(row * l + l - 1) * dm + j])
    _ = run_step(ctx, dw, cache, prefix, b, l - 1, dims, trace, "prefix")
    var step = run_step(ctx, dw, cache, token, b, 1, dims, trace, "step", True)
    var names = stage_names()
    for i in range(len(names)):
        if stage_kind(i) == KIND_TOKEN:
            for row in range(b):
                var expected = token_slice(full[i], i, row * l + l - 1, dims)
                var actual = token_slice(step[i], i, row, dims)
                if compare_stage(names[i], expected, actual, False).n_diff != 0:
                    raise Error("batched prefill/decode stage mismatch: " + names[i])
        elif compare_stage(names[i], full[i], step[i], False).n_diff != 0:
            raise Error("batched prefill/decode state mismatch: " + names[i])
    var stages = MambaDeviceStages(ctx, b, 1, dims)
    for delta in range(-1, 2, 2):
        var bad = ctx.enqueue_create_buffer[DType.float32](b * dm + delta)
        var refused = False
        try:
            mamba_step(ctx, stages, cache, dw, bad, b, trace, "invalid")
        except e:
            refused = String(e).find("exactly one token") >= 0
        if not refused:
            raise Error("decode must refuse wrong input length")
        _ = bad^
    var refused_cache = False
    try:
        var invalid_cache = allocate_inference_cache(ctx, 0, dims)
        _ = invalid_cache^
    except e:
        refused_cache = String(e).find("batch_size must be positive") >= 0
    if not refused_cache:
        raise Error("decode cache must refuse nonpositive batch size")
    print("Mamba-1 B2 continuation, zero cache, input/cache refusals PASS")
    _ = stages^
    _ = cache^
    _ = full_state^
    _ = dw^


def main() raises:
    reference_gate()
    # The reference negative controls intentionally corrupt host stages;
    # a successful detection does not authorize a normal device verdict.
    for arg in argv():
        if arg == "sabotage" or arg == "sabotage-window":
            return
    var ctx = DeviceContext()
    var cases: List[Int] = [0, 3, 7, 9, 12]
    for case_k in cases:
        var fixture = corpus_case(case_k)
        var weights = corpus_case_weights(case_k)
        var x = corpus_case_x(case_k)
        clause_d(ctx, weights, x, fixture.l, weights.dims)
    check_device_continuation(ctx)
    print("Mamba-1 device decode PASS: 5 cases, every stage equals prefill")
