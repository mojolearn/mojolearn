# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Native fit A/B workload for MOJOLEARN_2030_FUSED_EST_MOVE.

Build baseline and candidate in each numeric mode, run under the timing
lock, and compare every fingerprint line between the two binaries. A
warmup precedes five measured fits; prediction materialization is untimed.
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns
from gbdt.train import train, predict_floats
from checks.numerics import numeric_mode_name
from gbdt.methods.leaves_estimation.pointwise_oracle import FUSED_EST_MOVE_2030


def main() raises:
    var ctx = DeviceContext()
    var n = 65537
    var rows_env = getenv("MOJOLEARN_GBDT_AB_ROWS")
    if rows_env.byte_length() > 0:
        n = Int(rows_env)
    if n <= 0:
        raise Error("MOJOLEARN_GBDT_AB_ROWS must be positive")
    var f = 8
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
        y.append(Float32(1) if x[r] + x[3 * n + r] * 0.75 > x[6 * n + r] * 0.5 else Float32(0))
    print("numeric_mode", numeric_mode_name())
    print("fused", FUSED_EST_MOVE_2030, "rows", n)
    for rep in range(6):
        ctx.synchronize()
        var started = perf_counter_ns()
        var model = train(ctx, x, y, n, f, border_count=32,
            n_estimators=20, max_depth=6, learning_rate=Float32(0.2),
            loss="Logloss", leaf_estimation_iterations=10)
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
        for p in model.losses:
            fingerprint = (fingerprint ^ bitcast[DType.uint64](p)) * UInt64(1099511628211)
        print("fingerprint", rep, fingerprint)
        if rep > 0:
            print("fit_ms", elapsed)
