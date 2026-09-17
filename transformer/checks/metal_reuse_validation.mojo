# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""One selected fixture, two forward/backward executions on retained buffers."""
from std.memory import bitcast
from std.os import getenv
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from core.identity_trace import IdentityTrace
from core.step_phase import step_counts_now
from transformer.checks.transformer_fixture import (
    fixture_case, fixture_dims, fixture_weights, fixture_x, ScorePlant,
    RMS_EPS, ROPE_THETA,
)
from transformer.impl.llama.modeling_llama import (
    LlamaDeviceWeights, LlamaDeviceStages, LlamaKVCache, LlamaRopeTable,
    _upload, PLANT_AT_NONE, llama_decoder_layer_forward_planted,
)
from transformer.checks.transformer_backward import (
    LlamaBackwardStages, llama_decoder_layer_backward,
)
from transformer.checks.transformer_check import (
    llama_dims_of, device_dump, run_host_case, compare_dumps, count_moved,
    total_cells,
)
from transformer.checks.transformer_backward_check import (
    bwd_d_out, DOUT_PLANT_NONE, backward_device_dump, run_host_backward,
    compare_dumps as compare_backward, count_moved as count_backward,
)


def changed(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return True
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return True
    return False


def report_staging(label: String, dump: List[List[Float32]]):
    var largest = 0
    for i in range(len(dump)):
        largest = max(largest, len(dump[i]))
    print("staging", label, "bytes", total_cells(dump) * 4,
          "previous_largest_stage_bytes", largest * 4)


def main() raises:
    comptime assert is_defined["MOJOLEARN_NUMERIC_IDENTICAL"](), "IDENTICAL required"
    comptime assert is_defined["MOJOLEARN_STEP_PHASE_TIMERS"](), "counters required"
    if String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != "":
        raise Error("disable phase timing: it adds waits")
    var selected = String(getenv("MOJOLEARN_VALIDATION_FIXTURE"))
    var k = 1
    if selected == "batch":
        k = 3
    elif selected == "head":
        k = 5
    elif selected == "long":
        k = 7
    elif selected != "base":
        raise Error("select exactly one fixture: base, batch, head, long")
    var path = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if path == "":
        raise Error("set a fresh MOJOLEARN_IDENTITY_TRACE path")
    var c = fixture_case(k)
    var dims = fixture_dims(c)
    var ldims = llama_dims_of(dims)
    var w = fixture_weights(c)
    var x = fixture_x(c)
    var dy = bwd_d_out(c, dims, DOUT_PLANT_NONE)
    var ctx = DeviceContext()
    # Spare capacity exercises logical sub-buffer reads on every chosen shape.
    var capacity = c.l + 4
    var dw = LlamaDeviceWeights(ctx, ldims, RMS_EPS, w.norm1_w, w.norm2_w,
        w.w_q, w.w_k, w.w_v, w.w_o, w.w_gate, w.w_up, w.w_down)
    var kv = LlamaKVCache(ctx, c.b, ldims, capacity)
    var rope = LlamaRopeTable(ctx, ldims, ROPE_THETA, dims.rope_positions)
    var stages = LlamaDeviceStages(ctx, c.b, c.l, capacity, ldims)
    var bst = LlamaBackwardStages(ctx, c.b, c.l, capacity, ldims)
    var last_output = List[Float32]()
    var last_dx = List[Float32]()
    print("fixture", c.name, "B", c.b, "L", c.l, "head_dim", c.head_dim,
          "capacity", capacity)
    for repeat in range(2):
        if repeat == 1:
            x[0] += Float32(0.25)
            dy[0] += Float32(0.5)
        var host_f = run_host_case(c, dims, w, x, ScorePlant.none())
        var host_b = run_host_backward(c, dims, w, x, dy, c.b, c.l)
        var dx = _upload(ctx, x)
        # Restart at position zero but reuse all allocated caches and scratch.
        kv.s = 0
        var trace = IdentityTrace.disabled()
        if repeat == 1:
            trace = IdentityTrace.to_path(path)
        var counts = step_counts_now()
        var start = perf_counter_ns()
        llama_decoder_layer_forward_planted(ctx, stages, kv, rope, dw, dx,
            c.b, c.l, 0, PLANT_AT_NONE, List[Int](), List[UInt32](),
            trace, "reuse.fwd", materialize=True)
        var got_f = device_dump(ctx, stages, rope, dx, c.b, c.l, c.l, dims)
        llama_decoder_layer_backward(ctx, bst, stages, dw, rope.cos, rope.sin,
            dx, dy, c.b, c.l, 0, trace, "reuse.bwd", materialize=True)
        var got_b = backward_device_dump(ctx, bst, dims, c.b, c.l, c.l)
        var ms = Float64(perf_counter_ns() - start) / 1000000.0
        var after = step_counts_now()
        if count_moved(compare_dumps(host_f, got_f, True)) != 0:
            raise Error("forward oracle mismatch")
        if count_backward(compare_backward(host_b, got_b, True)) != 0:
            raise Error("backward oracle mismatch")
        if repeat == 1:
            if not changed(last_output, got_f[29]) or not changed(last_dx, got_b[36]):
                raise Error("changed input failed to change reused forward/backward output")
        last_output = got_f[29].copy()
        last_dx = got_b[36].copy()
        report_staging("forward", got_f)
        report_staging("backward", got_b)
        print("execution", repeat + 1, "forward_cells", total_cells(got_f),
              "backward_cells", total_cells(got_b), "waits", after.syncs - counts.syncs,
              "launches", after.launches - counts.launches, "device_and_dump_ms", ms)
        _ = dx^
    _ = bst^
    _ = stages^
    _ = rope^
    _ = kv^
    _ = dw^
    print("PASS: 67 stages match oracle twice; changed inputs reach reused outputs")
