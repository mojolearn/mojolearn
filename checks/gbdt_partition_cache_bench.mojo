# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""IDENTICAL Lossguide fit-only A/B. Args: rows; five warmups, three timings."""
from std.sys import argv
from std.time import perf_counter_ns
from std.memory import bitcast
from max.gpu.host import DeviceContext
from checks.gbdt_partition_cache_check import fold, tree_hash
from checks.numerics import numeric_mode_name
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper_depthwise import INCREMENTAL_PART_STATS
from gbdt.train import train, predict_floats
from std.testing import assert_equal


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: gbdt_partition_cache_bench <rows>")
    var rows = Int(args[1])
    if rows < 128:
        raise Error("rows must be at least 128")
    print("numeric_mode", numeric_mode_name())
    print("incremental", INCREMENTAL_PART_STATS, "rows", rows)
    var ctx = DeviceContext()
    var columns = 8
    var x = List[Float32]()
    var y = List[Float32]()
    for f in range(columns):
        for r in range(rows):
            var bits = UInt32(r * 2654435761 + f * 40503 + 0x1234567)
            bits ^= bits << 13
            bits ^= bits >> 17
            bits ^= bits << 5
            x.append(Float32(bits % 65536) / 32768 - 1)
    for r in range(rows):
        y.append(x[rows + r] * x[rows + r] if x[r] > 0 else x[2 * rows + r] + x[3 * rows + r] * 0.1)
    var expected = UInt64(0)
    for repeat in range(8):
        ctx.synchronize()
        var start = perf_counter_ns()
        var model = train(ctx, x, y, rows, columns, border_count=32,
            n_estimators=5, max_depth=10, grow_policy="Lossguide", max_leaves=31,
            loss="RMSE", random_seed=202109)
        ctx.synchronize()
        var elapsed = Float64(perf_counter_ns() - start) / 1e6
        var h = UInt64(14695981039346656037)
        fold(h, bitcast[DType.uint64](model.model.bias))
        for tree in model.model.non_symmetric_models:
            tree_hash(h, tree)
        for value in model.losses:
            fold(h, bitcast[DType.uint64](value))
        var predictions = predict_floats(ctx, model, x, rows)
        for value in predictions:
            fold(h, UInt64(bitcast[DType.uint32](value)))
        if repeat == 0:
            expected = h
        else:
            assert_equal(h, expected)
        print("fingerprint", h)
        if repeat >= 5:
            print("fit_ms", elapsed)
        else:
            print("warmup_ms", elapsed)
    _ = ctx^
