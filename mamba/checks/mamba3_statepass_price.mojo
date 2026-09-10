# SPDX-License-Identifier: Apache-2.0
"""IDENTICAL-only block timer for the S20 execution-plan A/B.

Build twice, with and without MOJOLEARN_MAMBA3_LEGACY_STATEPASS. State and
scratch allocation are outside the timed call; input refusal, projections,
all core stages, output and synchronization are inside. Each sample starts
with a fresh prefill state. This is not the September 7 torch fixture.
"""
from std.memory import bitcast
from std.time import perf_counter_ns
from std.sys import argv
from max.gpu.host import DeviceContext
from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from mamba.checks.mamba_fixture import corpus_tensor
from mamba.checks.mamba3_fixture import m3_case_weights, m3_case_seed, M3_TID_X
from mamba.impl.mamba_ssm.modules.mamba3 import (
    Mamba3DeviceWeights, Mamba3DeviceStages, allocate_inference_cache,
    mamba3_block_forward,
)
from mamba.impl.transformers.models.mamba.modeling_mamba import mamba_upload, mamba_download


def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var args = argv()
    var l = 512
    if len(args) > 1:
        l = Int(args[1])
    var ctx = DeviceContext()
    var w = m3_case_weights(2)
    var dims = w.dims.copy()
    var x = corpus_tensor(m3_case_seed(2), M3_TID_X, l * dims.d_model, -2.0, 2.0)
    var dw = Mamba3DeviceWeights(ctx, w)
    var dx = mamba_upload(ctx, x)
    var trace = IdentityTrace.disabled()
    for r in range(7):
        var state = allocate_inference_cache(ctx, 1, dims)
        var stages = Mamba3DeviceStages(ctx, 1, l, 0, dims)
        ctx.synchronize()
        var start = perf_counter_ns()
        mamba3_block_forward(ctx, stages, state, dw, dx, 1, l, trace, String("price"))
        ctx.synchronize()
        var elapsed = Float64(perf_counter_ns() - start) / 1e6
        var values = mamba_download(ctx, stages.residual_out, l * dims.d_model)
        var bits = UInt64(1469598103934665603)
        for i in range(len(values)):
            bits = (bits ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)
        print("M3_BLOCK", l, r, elapsed, bits)
