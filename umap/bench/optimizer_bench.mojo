# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from umap.optimizer import optimize_layout_identical
from umap.optimizer_fast import optimize_layout_fast


def main() raises:
    var ctx = DeviceContext()
    var n = 2048
    var initial = List[Float32]()
    var weights = List[Float32]()
    initial.resize(n * 2, Float32(0.0))
    weights.resize(n * n, Float32(0.0))
    for i in range(n):
        initial[2 * i] = Float32(i % 31) / Float32(31.0)
        initial[2 * i + 1] = Float32(i % 17) / Float32(17.0)
        for offset in range(1, 8):
            var j = (i + offset) % n
            weights[i * n + j] = Float32(1.0)
            weights[j * n + i] = Float32(1.0)
    var t0 = perf_counter_ns()
    _ = optimize_layout_identical(initial, weights, n, 2, 10)
    var serial_ms = Float64(perf_counter_ns() - t0) / Float64(1.0e6)
    t0 = perf_counter_ns()
    _ = optimize_layout_fast(
        ctx, initial, weights, n, 2, 10, Float32(1.0), 5,
        Float32(1.0), Float32(1.57694346), Float32(0.89506088), UInt64(0),
    )
    var fast_ms = Float64(perf_counter_ns() - t0) / Float64(1.0e6)
    var speedup = serial_ms / fast_ms
    print("umap optimizer serial_ms=" + String(serial_ms))
    print("umap optimizer fast_ms=" + String(fast_ms))
    print("umap optimizer speedup=" + String(speedup))
    if not (speedup > Float64(1.0)):
        raise Error("UMAP FAST optimizer did not beat the serial reference")
