# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""One base fixture, two fits: backward scratch bits and completion budget.

Enable MOJOLEARN_NUMERIC_IDENTICAL and MOJOLEARN_STEP_PHASE_TIMERS at build
time, but leave MOJOLEARN_TRANSFORMER_TIMING unset (it adds timer waits).
"""
from std.memory import bitcast
from std.os import getenv
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from core.step_phase import step_counts_now
from core.identity_trace import IdentityTrace
from transformer.impl.llama.modeling_llama import LlamaDims
from transformer.checks.transformer_backward import LlamaBackwardStages, _download
from transformer.checks.transformer_backward_check import clause_a_case


def expect_fill(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32],
                expected: UInt32, name: String) raises:
    var host = _download(ctx, buf, len(buf))
    for i in range(len(host)):
        if bitcast[DType.uint32](host[i]) != expected:
            raise Error(name + " wrong initialization at " + String(i))


def check_init(ctx: DeviceContext, lean: Bool) raises:
    var before = step_counts_now()
    var start = perf_counter_ns()
    var stages = LlamaBackwardStages(ctx, 1, 4, 4, LlamaDims(32, 2, 2, 16, 64), lean=lean)
    var elapsed = Float64(perf_counter_ns() - start) / 1000000.0
    var after = step_counts_now()
    print("initialization lean", lean, "waits", after.syncs - before.syncs,
          "launches", after.launches - before.launches, "ms", elapsed)
    expect_fill(ctx, stages.in_d_residual2, UInt32(0), "in_d_residual2")
    expect_fill(ctx, stages.d_down_proj_out, UInt32(0), "d_down_proj_out")
    expect_fill(ctx, stages.d_mlp_gated, UInt32(0), "d_mlp_gated")
    expect_fill(ctx, stages.dw_down, UInt32(0), "dw_down")
    expect_fill(ctx, stages.d_silu_out, UInt32(0), "d_silu_out")
    expect_fill(ctx, stages.d_up_proj_out, UInt32(0), "d_up_proj_out")
    expect_fill(ctx, stages.d_gate_proj_out, UInt32(0), "d_gate_proj_out")
    expect_fill(ctx, stages.dw_gate, UInt32(0), "dw_gate")
    expect_fill(ctx, stages.dw_up, UInt32(0), "dw_up")
    expect_fill(ctx, stages.d_norm2_out, UInt32(0), "d_norm2_out")
    expect_fill(ctx, stages.norm2_dot, UInt32(0), "norm2_dot")
    expect_fill(ctx, stages.dw_norm2, UInt32(0), "dw_norm2")
    expect_fill(ctx, stages.norm2_dx, UInt32(0), "norm2_dx")
    expect_fill(ctx, stages.d_residual1, UInt32(0), "d_residual1")
    expect_fill(ctx, stages.d_o_proj_out, UInt32(0), "d_o_proj_out")
    expect_fill(ctx, stages.d_attn_ctx, UInt32(0), "d_attn_ctx")
    expect_fill(ctx, stages.dw_o, UInt32(0), "dw_o")
    expect_fill(ctx, stages.d_attn_weights, UInt32(0), "d_attn_weights")
    expect_fill(ctx, stages.attn_zdot, UInt32(0), "attn_zdot")
    expect_fill(ctx, stages.d_attn_masked, UInt32(0), "d_attn_masked")
    expect_fill(ctx, stages.d_attn_scores, UInt32(0), "d_attn_scores")
    expect_fill(ctx, stages.d_qk_cell, UInt32(0), "d_qk_cell")
    expect_fill(ctx, stages.d_q_rope, UInt32(0), "d_q_rope")
    expect_fill(ctx, stages.d_k_cache, UInt32(0), "d_k_cache")
    expect_fill(ctx, stages.d_v_cache, UInt32(0), "d_v_cache")
    expect_fill(ctx, stages.d_k_rope, UInt32(0), "d_k_rope")
    expect_fill(ctx, stages.d_v_proj_out, UInt32(0), "d_v_proj_out")
    expect_fill(ctx, stages.d_q_proj_out, UInt32(0), "d_q_proj_out")
    expect_fill(ctx, stages.d_k_proj_out, UInt32(0), "d_k_proj_out")
    expect_fill(ctx, stages.dw_q, UInt32(0), "dw_q")
    expect_fill(ctx, stages.dw_k, UInt32(0), "dw_k")
    expect_fill(ctx, stages.dw_v, UInt32(0), "dw_v")
    expect_fill(ctx, stages.d_norm1_out, UInt32(0), "d_norm1_out")
    expect_fill(ctx, stages.norm1_dot, UInt32(0), "norm1_dot")
    expect_fill(ctx, stages.dw_norm1, UInt32(0), "dw_norm1")
    expect_fill(ctx, stages.norm1_dx, UInt32(0), "norm1_dx")
    expect_fill(ctx, stages.d_x, UInt32(0), "d_x")
    expect_fill(ctx, stages.dh, UInt32(0), "dh")
    expect_fill(ctx, stages.dprod, UInt32(0), "dprod")
    expect_fill(ctx, stages.rstd, UInt32(0), "rstd")
    expect_fill(ctx, stages.dvcoef, UInt32(0), "dvcoef")
    expect_fill(ctx, stages.ones, UInt32(0x3F800000), "ones")
    expect_fill(ctx, stages.tmp0, UInt32(0), "tmp0")
    expect_fill(ctx, stages.tmp1, UInt32(0), "tmp1")
    expect_fill(ctx, stages.tmp2, UInt32(0), "tmp2")
    expect_fill(ctx, stages.head_a, UInt32(0), "head_a")
    expect_fill(ctx, stages.head_b, UInt32(0), "head_b")
    expect_fill(ctx, stages.head_c, UInt32(0), "head_c")
    _ = stages^
    if after.syncs - before.syncs != 1:
        raise Error("backward initialization must complete with one wait")


def main() raises:
    comptime assert is_defined["MOJOLEARN_NUMERIC_IDENTICAL"](), "IDENTICAL required"
    comptime assert is_defined["MOJOLEARN_STEP_PHASE_TIMERS"](), "counters required"
    if String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != "":
        raise Error("disable phase timing: it adds waits")
    var ctx = DeviceContext()
    for repeat in range(2):
        check_init(ctx, lean=repeat == 1)
        var trace = IdentityTrace.disabled()
        var before = step_counts_now()
        var start = perf_counter_ns()
        var verdict = clause_a_case(ctx, 0, trace, "bounded")
        var elapsed = Float64(perf_counter_ns() - start) / 1000000.0
        var after = step_counts_now()
        if verdict.n_moved != 0:
            raise Error("backward stage differs from oracle: " + verdict.first)
        print("fit", repeat + 1, "waits", after.syncs - before.syncs,
              "launches", after.launches - before.launches, "ms", elapsed,
              "oracle_cells", verdict.cells)
    print("PASS: every scratch fill and all 37 backward stages, two base fits")
