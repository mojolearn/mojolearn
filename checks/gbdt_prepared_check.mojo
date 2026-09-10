# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Prepared-vs-ordinary complete model, prediction and Float64 loss gate."""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.time import perf_counter_ns
from std.os import getenv
from gbdt.prepared import prepare_numeric_dataset
from gbdt.train import train, predict_floats
from gbdt.models.model_text import model_text
from checks.numerics import numeric_mode_name
from gbdt.gpu_util.kernel.bootstrap import BOOTSTRAP_KERNEL_POISSON


def main() raises:
    var ctx = DeviceContext()
    if getenv("MOJOLEARN_PREPARED_BENCH").byte_length() > 0:
        benchmark(ctx)
        return
    var n = 2053
    var f = 4
    var x = List[Float32]()
    var y = List[Float32]()
    var w = List[Float32]()
    for c in range(f):
        for r in range(n):
            var h = UInt32(r * 2654435761 + c * 40503 + 12345)
            h ^= h >> 13
            x.append(Float32(h % 511) / 256 - 1)
    # NaN treatment, constant feature, ragged sample count.
    x[n + 7] = Float32(0.0) / Float32(0.0)
    for r in range(n):
        x[3 * n + r] = 2
        y.append(Float32(1) if x[r] + x[2 * n + r] > 0 else Float32(0))
        w.append(Float32(r % 5) / 3)
    print("numeric_mode", numeric_mode_name())
    var cases = 0
    var caps: List[Int] = [0, 257]
    var policies: List[String] = ["SymmetricTree", "Depthwise", "Lossguide"]
    var losses: List[String] = ["RMSE", "Logloss", "CrossEntropy"]
    for cap in caps:
        for weighted in range(2):
            var weights = w.copy() if weighted else List[Float32]()
            var pool = prepare_numeric_dataset(ctx, x, y, n, f,
                border_count=15, border_build_max_samples=cap,
                random_seed=UInt64(42), sample_weight=weights)
            for policy in policies:
                for loss in losses:
                    var reference = train(ctx, x, y, n, f, border_count=15,
                        border_build_max_samples=cap, n_estimators=3,
                        max_depth=3, grow_policy=policy, loss=loss,
                        sample_weight=weights, random_seed=UInt64(42))
                    var expected = model_text(reference)
                    var predictions = predict_floats(ctx, reference, x, n)
                    for rep in range(2):
                        var got = pool.fit(n_estimators=3, max_depth=3,
                            grow_policy=policy, loss=loss, random_seed=UInt64(42))
                        if model_text(got) != expected:
                            raise Error("prepared model mismatch " + policy + " " + loss)
                        if len(got.losses) != len(reference.losses):
                            raise Error("prepared loss length mismatch")
                        for i in range(len(got.losses)):
                            if bitcast[DType.uint64](got.losses[i]) != bitcast[DType.uint64](reference.losses[i]):
                                raise Error("prepared Float64 loss mismatch")
                        var p = predict_floats(ctx, got, x, n)
                        for i in range(len(p)):
                            if bitcast[DType.uint32](p[i]) != bitcast[DType.uint32](predictions[i]):
                                raise Error("prepared prediction mismatch")
                        cases += 1
    # Preparation takes an owned snapshot, including labels and weights.
    var snapshot = prepare_numeric_dataset(ctx, x, y, n, f, border_count=15,
        border_build_max_samples=257, sample_weight=w)
    var before = snapshot.fit(n_estimators=2, max_depth=3, grow_policy="Lossguide")
    for r in range(n):
        x[r] = -99
        y[r] = 17
        w[r] = 0
    var after = snapshot.fit(n_estimators=2, max_depth=3, grow_policy="Lossguide")
    if model_text(before) != model_text(after):
        raise Error("caller mutation changed prepared data")
    var invalid_rates: List[Float32] = [-1, 0, Float32(0.0) / Float32(0.0)]
    for rate in invalid_rates:
        var refused = False
        try:
            var invalid = snapshot.fit(n_estimators=1, max_depth=2,
                bootstrap_type=BOOTSTRAP_KERNEL_POISSON, bootstrap_param=rate)
        except:
            refused = True
        if not refused:
            raise Error("invalid Poisson rate was accepted")
    print("PASS prepared comparisons", cases, "owned snapshot and invalid bootstrap")


def benchmark(ctx: DeviceContext) raises:
    var n = 65537
    var f = 8
    var x = List[Float32]()
    var y = List[Float32]()
    for c in range(f):
        for r in range(n):
            var h = UInt32(r * 2654435761 + c * 40503 + 12345)
            h ^= h >> 13
            x.append(Float32(h % 65521) / 32768 - 1)
    for r in range(n):
        y.append(x[r] + x[3 * n + r] * Float32(0.75))
    ctx.synchronize()
    var t0 = perf_counter_ns()
    var pool = prepare_numeric_dataset(ctx, x, y, n, f, border_count=32,
        border_build_max_samples=4097)
    ctx.synchronize()
    print("numeric_mode", numeric_mode_name(), "prepare_ms",
          Float64(perf_counter_ns() - t0) / 1e6)
    var expected = String("")
    # Three warmup pairs, then six pairs with reversed order every pair.
    for rep in range(9):
        for k in range(2):
            var arm = (k + rep) % 2
            ctx.synchronize()
            var start = perf_counter_ns()
            var tm = train(ctx, x, y, n, f, border_count=32,
                border_build_max_samples=4097, n_estimators=5,
                max_depth=5, grow_policy="Lossguide") if arm == 0 else pool.fit(
                    n_estimators=5, max_depth=5, grow_policy="Lossguide")
            ctx.synchronize()
            var elapsed = Float64(perf_counter_ns() - start) / 1e6
            var text = model_text(tm)
            if expected.byte_length() == 0:
                expected = text
            elif text != expected:
                raise Error("prepared benchmark model mismatch")
            if rep >= 3:
                print("fit_ms", arm, rep, elapsed)
    print("PASS benchmark full-model equivalence")
