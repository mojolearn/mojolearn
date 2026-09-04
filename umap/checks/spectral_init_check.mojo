# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from max.gpu.host import DeviceContext
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from umap.spectral_init import spectral_initialize_weights


def main() raises:
    var ctx = DeviceContext()
    var n = 10
    var weights = List[Float32]()
    weights.resize(n * n, Float32(0.0))
    for i in range(n - 1):
        weights[i * n + i + 1] = Float32(1.0)
        weights[(i + 1) * n + i] = Float32(1.0)
    var first = spectral_initialize_weights(ctx, weights, n, 3, 3, UInt64(7))
    var second = spectral_initialize_weights(ctx, weights, n, 3, 3, UInt64(7))
    if len(first) != 30 or len(second) != 30:
        raise Error("UMAP spectral initialization shape mismatch")
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        for i in range(30):
            if bitcast[DType.uint32](first[i]) != bitcast[DType.uint32](
                second[i]
            ):
                raise Error("UMAP seeded spectral initialization changed bits")
    for c in range(3):
        var peak = Float32(0.0)
        for i in range(n):
            var value = first[i * 3 + c]
            var magnitude = value if value >= Float32(0.0) else -value
            if magnitude > peak:
                peak = magnitude
        var scale_error = peak - Float32(10.0)
        if scale_error < Float32(0.0):
            scale_error = -scale_error
        if scale_error > Float32(1.0e-5):
            raise Error("UMAP spectral component was not scaled to ten")
    print("UMAP spectral initialization PASS")
