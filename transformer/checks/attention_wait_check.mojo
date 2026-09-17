# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Two executions of the B2 grouped-head base fixture, not a full sweep.

Counters enabled, phase timing disabled. The first execution has tracing off;
the second writes all 37 backward stages for cross-build card comparison.
"""
from std.os import getenv
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from core.step_phase import step_counts_now
from core.identity_trace import IdentityTrace
from transformer.impl.llama.modeling_llama import LlamaDims, LlamaDeviceStages
from transformer.checks.transformer_backward import LlamaBackwardStages, bwd_attention_weight_grad
from transformer.checks.transformer_backward_check import clause_a_case


def check_head_budget(ctx: DeviceContext) raises:
    var dims = LlamaDims(32, 2, 1, 16, 64)
    var fwd = LlamaDeviceStages(ctx, 2, 4, 4, dims)
    var bwd = LlamaBackwardStages(ctx, 2, 4, 4, dims)
    var before = step_counts_now()
    bwd_attention_weight_grad(ctx, bwd, fwd, 2, 4, 4, dims)
    var after = step_counts_now()
    print("head_loop waits", after.syncs - before.syncs,
          "launches", after.launches - before.launches)
    if after.syncs - before.syncs != 1:
        raise Error("head loop must complete with one wait")
    if after.launches - before.launches != 16:
        raise Error("head loop must enqueue all four operations for each of four heads")
    _ = bwd^
    _ = fwd^


def main() raises:
    comptime assert is_defined["MOJOLEARN_NUMERIC_IDENTICAL"](), "IDENTICAL required"
    comptime assert is_defined["MOJOLEARN_STEP_PHASE_TIMERS"](), "counters required"
    if String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != "":
        raise Error("disable phase timing: it adds waits")
    var path = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if path == "":
        raise Error("set MOJOLEARN_IDENTITY_TRACE to a fresh card path")
    var ctx = DeviceContext()
    for repeat in range(2):
        check_head_budget(ctx)
        var trace = IdentityTrace.disabled()
        if repeat == 1:
            trace = IdentityTrace.to_path(path)
        var before = step_counts_now()
        var start = perf_counter_ns()
        # Base fixture 1: B=2, L=4, two query heads sharing one KV head.
        var result = clause_a_case(ctx, 1, trace, "bounded")
        var elapsed = Float64(perf_counter_ns() - start) / 1000000.0
        var after = step_counts_now()
        if result.n_moved != 0:
            raise Error("backward stage differs from oracle: " + result.first)
        print("execution", repeat + 1, "waits", after.syncs - before.syncs,
              "launches", after.launches - before.launches, "ms", elapsed,
              "oracle_cells", result.cells)
    print("PASS: head-loop wait budget and all 37 stages against oracle, two executions")
