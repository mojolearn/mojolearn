# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Bounded forward readback check: one B2 fixture, two executions."""
from std.memory import bitcast
from std.os import getenv
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from core.identity_trace import IdentityTrace
from core.step_phase import step_counts_now
from transformer.impl.llama.modeling_llama import (
    LlamaDims, LlamaDeviceStages, LlamaRopeTable, _upload, _download,
)
from transformer.checks.transformer_fixture import fixture_case, fixture_dims, ROPE_THETA
from transformer.checks.transformer_check import device_dump, clause_a_case


def check_readback(ctx: DeviceContext) raises:
    var dims = fixture_dims(fixture_case(1))
    var ldims = LlamaDims(32, 2, 1, 16, 64)
    # Capacity 8, logical S 4: attention and cache tails must be excluded.
    var stages = LlamaDeviceStages(ctx, 2, 4, 8, ldims)
    var rope = LlamaRopeTable(ctx, ldims, ROPE_THETA, dims.rope_positions)
    var dx = _upload(ctx, List[Float32](length=256, fill=Float32(9.0)))
    # Independent single-buffer reads verify the separately owned rotary data.
    var rotary = List[List[Float32]]()
    rotary.append(_download(ctx, rope.inv_freq, 8))
    rotary.append(_download(ctx, rope.cos, rope.p_max * 8))
    rotary.append(_download(ctx, rope.sin, rope.p_max * 8))
    stages.scores.enqueue_fill(Float32(-17.0))
    stages.residual2.enqueue_fill(Float32(41.0))
    stages.norm1_sumsq.enqueue_fill(bitcast[DType.float32](UInt32(0x80000000)))
    var before = step_counts_now()
    var dump = device_dump(ctx, stages, rope, dx, 2, 4, 4, dims)
    var after = step_counts_now()
    print("readback waits", after.syncs - before.syncs,
          "host_allocs", after.host_allocs - before.host_allocs,
          "copies", after.d2h - before.d2h)
    if after.syncs - before.syncs != 2 or after.host_allocs - before.host_allocs != 1:
        raise Error("readback must use two waits and one host allocation")
    if after.d2h - before.d2h != 30 or after.launches != before.launches:
        raise Error("readback must copy all 30 stages without kernels")
    if len(dump) != 30:
        raise Error("readback lost a stage")
    for stage in range(30):
        if stage == 11 or stage == 12:
            if len(dump[stage]) != 128:
                raise Error("cache dump included unused capacity")
        if stage == 13 or stage == 14 or stage == 16 or stage == 18:
            if len(dump[stage]) != 64:
                raise Error("attention dump included unused capacity")
        var expected = UInt32(0)
        if stage == 0:
            expected = bitcast[DType.uint32](Float32(9.0))
        if stage == 1:
            expected = UInt32(0x80000000)
        if stage == 13:
            expected = bitcast[DType.uint32](Float32(-17.0))
        if stage == 29:
            expected = bitcast[DType.uint32](Float32(41.0))
        if stage >= 6 and stage <= 8:
            if len(dump[stage]) != len(rotary[stage - 6]):
                raise Error("rotary dump length changed")
        for j in range(len(dump[stage])):
            if stage >= 6 and stage <= 8:
                expected = bitcast[DType.uint32](rotary[stage - 6][j])
            if bitcast[DType.uint32](dump[stage][j]) != expected:
                raise Error("readback changed bits or stage order")
    _ = dx^
    _ = rope^
    _ = stages^


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
            raise Error("forward stage differs from oracle: " + result.first)
        print("execution", repeat + 1, "waits", after.syncs - before.syncs,
              "launches", after.launches - before.launches, "ms", elapsed,
              "oracle_cells", result.cells,
              "host_allocs", after.host_allocs - before.host_allocs,
              "copies", after.d2h - before.d2h)
    print("PASS: logical readback bits and all 30 oracle stages, two executions")
