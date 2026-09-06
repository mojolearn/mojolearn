# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Serial semantic reference for UMAP attractive/repulsive optimization."""

from std.math import isfinite, pow
from std.memory import bitcast
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from umap.optimizer_fast import optimize_layout_fast


comptime UMAP_GRAD_CLIP = Float32(4.0)


def _finite(v: Float32) -> Bool:
    var bits = bitcast[DType.uint32](v)
    return ((bits >> UInt32(23)) & UInt32(0xFF)) != UInt32(0xFF)


def _splitmix64(value: UInt64) -> UInt64:
    var z = value + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _clip(value: Float32) -> Float32:
    if value > UMAP_GRAD_CLIP:
        return UMAP_GRAD_CLIP
    if value < -UMAP_GRAD_CLIP:
        return -UMAP_GRAD_CLIP
    return value


def optimize_layout_identical(
    initial_embedding: List[Float32],
    weights: List[Float32],
    n_samples: Int,
    n_components: Int,
    n_epochs: Int,
    initial_learning_rate: Float32 = Float32(1.0),
    negative_sample_rate: Int = 5,
    repulsion_strength: Float32 = Float32(1.0),
    a: Float32 = Float32(1.57694346),
    b: Float32 = Float32(0.89506088),
    seed: UInt64 = UInt64(0),
) raises -> List[Float32]:
    """Optimize in a fully specified serial order.

    Positive edges are the nonzero dense entries in row-major order. Edge
    `(head, tail)` is eligible at epoch `e` when
    `floor((e+1)*weight/max_weight) > floor(e*weight/max_weight)`. Every
    eligible edge draws exactly `negative_sample_rate` vertices from a
    SplitMix64 counter keyed by `(seed, epoch, edge ordinal, negative slot)`.
    Attractive updates move both endpoints; repulsive updates move only the
    head, matching UMAP's `move_other=True` layout semantics. Each coordinate
    update rounds through Float32 in program order. This is IDENTICAL's
    conflict-free reference. FAST uses the separately gated Jacobi launcher.
    """
    if n_samples < 2 or (n_components != 2 and n_components != 3):
        raise Error("UMAP optimizer supports at least two samples in 2D/3D")
    if len(initial_embedding) != n_samples * n_components or len(
        weights
    ) != n_samples * n_samples:
        raise Error("UMAP optimizer input shape mismatch")
    if not isfinite(initial_learning_rate) or not isfinite(repulsion_strength) or (
        not isfinite(a) or not isfinite(b)
    ):
        raise Error("UMAP optimizer scalar parameters must be finite")
    if n_epochs < 1 or not (initial_learning_rate > Float32(0.0)):
        raise Error("UMAP optimizer needs positive epochs and learning rate")
    if negative_sample_rate < 0 or repulsion_strength < Float32(0.0):
        raise Error("UMAP optimizer negative sampling parameters are invalid")
    if not (a > Float32(0.0)) or not (b > Float32(0.0)):
        raise Error("UMAP optimizer curve parameters must be positive")
    var max_weight = Float32(0.0)
    for i in range(n_samples * n_samples):
        if not _finite(weights[i]) or weights[i] < Float32(0.0):
            raise Error("UMAP optimizer graph weight is invalid")
        if weights[i] > max_weight:
            max_weight = weights[i]
    if not (max_weight > Float32(0.0)):
        raise Error("UMAP optimizer graph has no positive edges")
    var embedding = initial_embedding.copy()
    for i in range(len(embedding)):
        if not _finite(embedding[i]):
            raise Error("UMAP optimizer initialization is not finite")
    for epoch in range(n_epochs):
        var alpha = initial_learning_rate * Float32(
            Float64(n_epochs - epoch) / Float64(n_epochs)
        )
        var edge_ordinal = 0
        for head in range(n_samples):
            for tail in range(n_samples):
                var weight = weights[head * n_samples + tail]
                if head == tail or not (weight > Float32(0.0)):
                    continue
                var scaled = Float64(weight) / Float64(max_weight)
                var before = Int(Float64(epoch) * scaled)
                var after = Int(Float64(epoch + 1) * scaled)
                if after <= before:
                    edge_ordinal += 1
                    continue
                var dist_squared = Float32(0.0)
                for c in range(n_components):
                    var delta = embedding[head * n_components + c] - (
                        embedding[tail * n_components + c]
                    )
                    dist_squared += delta * delta
                if dist_squared > Float32(0.0):
                    var dist_pow = Float32(pow(Float64(dist_squared), Float64(b)))
                    var coeff = -Float32(2.0) * a * b * (
                        dist_pow / dist_squared
                    ) / (a * dist_pow + Float32(1.0))
                    for c in range(n_components):
                        var delta = embedding[head * n_components + c] - (
                            embedding[tail * n_components + c]
                        )
                        var update = alpha * _clip(coeff * delta)
                        embedding[head * n_components + c] += update
                        embedding[tail * n_components + c] -= update
                for negative in range(negative_sample_rate):
                    var counter = (
                        seed
                        ^ (UInt64(epoch) * UInt64(0xD1B54A32D192ED03))
                        ^ (UInt64(edge_ordinal) * UInt64(0x94D049BB133111EB))
                        ^ UInt64(negative)
                    )
                    var other = Int(_splitmix64(counter) % UInt64(n_samples))
                    if other == head:
                        continue
                    var neg_dist = Float32(0.0)
                    for c in range(n_components):
                        var delta = embedding[head * n_components + c] - (
                            embedding[other * n_components + c]
                        )
                        neg_dist += delta * delta
                    if neg_dist > Float32(0.0):
                        var neg_pow = Float32(pow(Float64(neg_dist), Float64(b)))
                        var coeff = Float32(2.0) * repulsion_strength * b / (
                            (Float32(0.001) + neg_dist)
                            * (a * neg_pow + Float32(1.0))
                        )
                        for c in range(n_components):
                            var delta = embedding[head * n_components + c] - (
                                embedding[other * n_components + c]
                            )
                            embedding[head * n_components + c] += (
                                alpha * _clip(coeff * delta)
                            )
                edge_ordinal += 1
    return embedding^


def optimize_layout(
    ctx: DeviceContext,
    initial_embedding: List[Float32],
    weights: List[Float32],
    n_samples: Int,
    n_components: Int,
    n_epochs: Int,
    initial_learning_rate: Float32 = Float32(1.0),
    negative_sample_rate: Int = 5,
    repulsion_strength: Float32 = Float32(1.0),
    a: Float32 = Float32(1.57694346),
    b: Float32 = Float32(0.89506088),
    seed: UInt64 = UInt64(0),
) raises -> List[Float32]:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return optimize_layout_identical(
            initial_embedding, weights, n_samples, n_components, n_epochs,
            initial_learning_rate, negative_sample_rate, repulsion_strength,
            a, b, seed,
        )
    # Kernel launch and graph-upload overhead dominates small layouts on the
    # currently supported devices.  Keep FAST on the serial reference below
    # the measured crossover instead of making the "fast" API slower.
    if n_samples < 1024:
        return optimize_layout_identical(
            initial_embedding, weights, n_samples, n_components, n_epochs,
            initial_learning_rate, negative_sample_rate, repulsion_strength,
            a, b, seed,
        )
    return optimize_layout_fast(
        ctx, initial_embedding, weights, n_samples, n_components, n_epochs,
        initial_learning_rate, negative_sample_rate, repulsion_strength,
        a, b, seed,
    )
