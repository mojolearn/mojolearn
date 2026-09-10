# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Full-model native symmetric fit A/B for MOJOLEARN_2031_SYM_RIDX_SPLITS.

Build baseline and candidate in each numeric mode, run under the timing
lock, and compare every fingerprint line between the two binaries. A
warmup precedes configurable measured fits; prediction materialization is untimed.
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns
from gbdt.train import train, predict_floats
from checks.numerics import numeric_mode_name
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import SYM_RIDX_SPLITS_2031


def option(name: String, fallback: Int) raises -> Int:
    var value = getenv(name)
    return Int(value) if value.byte_length() > 0 else fallback


def main() raises:
    var ctx = DeviceContext()
    var n = option("MOJOLEARN_SYM_ROWS", 65537)
    var f = option("MOJOLEARN_SYM_COLS", 8)
    var depth = option("MOJOLEARN_SYM_DEPTH", 6)
    var trees = option("MOJOLEARN_SYM_TREES", 12)
    var borders = option("MOJOLEARN_SYM_BORDERS", 32)
    var reps = option("MOJOLEARN_SYM_REPS", 3)
    var loss = getenv("MOJOLEARN_SYM_LOSS")
    if loss.byte_length() == 0:
        loss = "Logloss"
    if n < 1 or f < 7 or reps < 0:
        raise Error("need positive rows, >=7 columns and nonnegative reps")
    var x = List[Float32]()
    var y = List[Float32]()
    for feat in range(f):
        for r in range(n):
            var h = UInt32(r * 2654435761 + feat * 40503 + 0x2545F491)
            h ^= h << 13
            h ^= h >> 17
            h ^= h << 5
            x.append(Float32(h % 65536) / 32768 - 1)
    for r in range(n):
        var score = x[r] + x[3 * n + r] * 0.75 - x[6 * n + r] * 0.5
        y.append(score if loss == "RMSE" else (Float32(1) if score > 0 else Float32(0)))
    print("numeric_mode", numeric_mode_name())
    print("ridx", SYM_RIDX_SPLITS_2031, "rows", n, "cols", f, "depth", depth, "trees", trees, "borders", borders, "loss", loss)
    print("device", ctx.name())
    for rep in range(reps + 1):
        ctx.synchronize()
        var started = perf_counter_ns()
        var model = train(ctx, x, y, n, f, border_count=borders,
            n_estimators=trees, max_depth=depth, learning_rate=Float32(0.2),
            loss=loss, leaf_estimation_iterations=1 if loss == "RMSE" else 10)
        ctx.synchronize()
        var elapsed = Float64(perf_counter_ns() - started) / 1.0e6
        var predictions = predict_floats(ctx, model, x, n)
        var fingerprint = UInt64(14695981039346656037)
        for p in predictions:
            fingerprint = (fingerprint ^ UInt64(bitcast[DType.uint32](p))) * UInt64(1099511628211)
        for tree in model.model.weak_models:
            for split in tree.structure.splits:
                fingerprint = (fingerprint ^ UInt64(split.feature_id)) * UInt64(1099511628211)
                fingerprint = (fingerprint ^ UInt64(split.bin_idx)) * UInt64(1099511628211)
                fingerprint = (fingerprint ^ UInt64(split.split_type)) * UInt64(1099511628211)
            for p in tree.leaf_values:
                fingerprint = (fingerprint ^ UInt64(bitcast[DType.uint32](p))) * UInt64(1099511628211)
        fingerprint = (fingerprint ^ bitcast[DType.uint64](model.model.bias)) * UInt64(1099511628211)
        for feature_borders in model.borders:
            fingerprint = (fingerprint ^ UInt64(len(feature_borders))) * UInt64(1099511628211)
            for border in feature_borders:
                fingerprint = (fingerprint ^ UInt64(bitcast[DType.uint32](border))) * UInt64(1099511628211)
        for count in model.fold_counts:
            fingerprint = (fingerprint ^ UInt64(count)) * UInt64(1099511628211)
        for p in model.losses:
            fingerprint = (fingerprint ^ bitcast[DType.uint64](p)) * UInt64(1099511628211)
        print("fingerprint", rep, fingerprint)
        if rep > 0:
            print("fit_ms", elapsed)
