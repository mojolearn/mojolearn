# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Bounded readback diagnostic: one B2 fixture, two checked executions."""
from std.memory import bitcast
from std.os import getenv
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from core.identity_trace import IdentityTrace
from core.step_phase import step_counts_now
from transformer.impl.llama.modeling_llama import LlamaDims
from transformer.checks.transformer_fixture import fixture_case, fixture_dims
from transformer.checks.transformer_backward import LlamaBackwardStages
from transformer.checks.transformer_backward_check import (
    backward_device_dump, bwd_stage_cells, clause_a_case,
)


def check_readback(ctx: DeviceContext) raises:
    var dims = fixture_dims(fixture_case(1))
    # Capacity exceeds logical S: trailing attention storage must not leak
    # into the dump. Two distant stages distinguish packed offsets and order.
    var bst = LlamaBackwardStages(ctx, 2, 4, 8, LlamaDims(32, 2, 1, 16, 64))
    bst.d_attn_weights.enqueue_fill(Float32(-17.0))
    bst.d_x.enqueue_fill(Float32(41.0))
    bst.norm1_dot.enqueue_fill(bitcast[DType.float32](UInt32(0x80000000)))
    var before = step_counts_now()
    var dump = backward_device_dump(ctx, bst, dims, 2, 4, 4)
    var after = step_counts_now()
    print("readback waits", after.syncs - before.syncs,
          "host_allocs", after.host_allocs - before.host_allocs,
          "copies", after.d2h - before.d2h)
    if after.syncs - before.syncs != 2 or after.host_allocs - before.host_allocs != 1:
        raise Error("readback must use two waits and one host allocation")
    if after.d2h - before.d2h != 37 or after.launches != before.launches:
        raise Error("readback must copy all 37 stages without kernels")
    if len(dump) != 37:
        raise Error("readback lost a stage")
    for stage in range(37):
        if len(dump[stage]) != bwd_stage_cells(stage, dims, 2, 4, 4):
            raise Error("readback returned allocation capacity instead of logical cells")
        var expected = UInt32(0)
        if stage == 17:
            expected = bitcast[DType.uint32](Float32(-17.0))
        if stage == 33:
            expected = UInt32(0x80000000)
        if stage == 36:
            expected = bitcast[DType.uint32](Float32(41.0))
        for j in range(len(dump[stage])):
            if bitcast[DType.uint32](dump[stage][j]) != expected:
                raise Error("readback changed bits or stage order")
    _ = bst^


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
        check_readback(ctx)
        var trace = IdentityTrace.disabled()
        if repeat == 1:
            trace = IdentityTrace.to_path(path)
        var before = step_counts_now()
        var start = perf_counter_ns()
        var result = clause_a_case(ctx, 1, trace, "bounded")
        var elapsed = Float64(perf_counter_ns() - start) / 1000000.0
        var after = step_counts_now()
        if result.n_moved != 0:
            raise Error("backward stage differs from oracle: " + result.first)
        print("execution", repeat + 1, "waits", after.syncs - before.syncs,
              "launches", after.launches - before.launches, "ms", elapsed,
              "oracle_cells", result.cells,
              "host_allocs", after.host_allocs - before.host_allocs,
              "copies", after.d2h - before.d2h)
    print("PASS: logical readback bits and all 37 oracle stages, two executions")
