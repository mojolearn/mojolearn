# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Deterministic first UMAP slice: smooth k-NN distances and fuzzy graph."""

from std.math import exp, log2
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def _finite(v: Float32) -> Bool:
    var bits = bitcast[DType.uint32](v)
    return ((bits >> UInt32(23)) & UInt32(0xFF)) != UInt32(0xFF)


struct FuzzySimplicialGraph(Copyable, Movable):
    var n_samples: Int
    var n_neighbors: Int
    var rhos: List[Float32]
    var sigmas: List[Float32]
    var directed: List[Float32]
    var weights: List[Float32]

    def __init__(
        out self,
        n_samples: Int,
        n_neighbors: Int,
        var rhos: List[Float32],
        var sigmas: List[Float32],
        var directed: List[Float32],
        var weights: List[Float32],
    ):
        self.n_samples = n_samples
        self.n_neighbors = n_neighbors
        self.rhos = rhos^
        self.sigmas = sigmas^
        self.directed = directed^
        self.weights = weights^


def _membership_sum(
    distances: List[Float32], row: Int, k: Int, rho: Float64, sigma: Float64
) -> Float64:
    var total = Float64(0.0)
    for j in range(1, k):
        var d = Float64(distances[row * k + j]) - rho
        total += Float64(1.0) if d <= 0.0 else exp(-d / sigma)
    return total


def _sigma_upper_bound(
    distances: List[Float32], row: Int, k: Int, rho: Float64, target: Float64
) raises -> Float64:
    var hi = Float64(1.0)
    while _membership_sum(distances, row, k, rho, hi) < target:
        hi *= 2.0
        if hi > Float64(1.0e20):
            raise Error("UMAP sigma search did not bracket its target")
    return hi


@no_inline
def _sigma_identical(
    distances: List[Float32], row: Int, k: Int, rho: Float64, target: Float64
) raises -> Float64:
    """Fixed runtime loop: ordered and deliberately not compiler-unrolled."""
    var lo = Float64(0.0)
    var hi = _sigma_upper_bound(distances, row, k, rho, target)
    var sigma = hi
    var step = 0
    while step < 64:
        sigma = (lo + hi) * 0.5
        var value = _membership_sum(distances, row, k, rho, sigma)
        if value > target:
            hi = sigma
        else:
            lo = sigma
        step += 1
    return sigma


@no_inline
def _sigma_fast(
    distances: List[Float32], row: Int, k: Int, rho: Float64, target: Float64
) raises -> Float64:
    var lo = Float64(0.0)
    var hi = _sigma_upper_bound(distances, row, k, rho, target)
    var sigma = hi
    var step = 0
    while step < 64:
        sigma = (lo + hi) * 0.5
        var value = _membership_sum(distances, row, k, rho, sigma)
        var error = value - target
        if error < 0.0:
            error = -error
        if error <= Float64(1.0e-5):
            return sigma
        if value > target:
            hi = sigma
        else:
            lo = sigma
        step += 1
    return sigma


def fuzzy_simplicial_graph(
    knn_indices: List[UInt32],
    knn_distances: List[Float32],
    n_samples: Int,
    n_neighbors: Int,
    set_op_mix_ratio: Float32 = Float32(1.0),
) raises -> FuzzySimplicialGraph:
    """Build UMAP's directed memberships and fuzzy union as dense rows.

    Neighbours must be distance-sorted and include self in slot zero. In
    `NUMERIC_IDENTICAL`, sigma search always takes 64 ordered Float64 steps.
    `NUMERIC_FAST` exits once the membership sum is within 1e-5 of target.
    Graph accumulation is host row order in both modes; later GPU sparse
    assembly may parallelize only the FAST arm.
    """
    if n_samples < 2 or n_neighbors < 2 or n_neighbors > n_samples:
        raise Error("invalid UMAP k-NN graph shape")
    if len(knn_indices) != n_samples * n_neighbors or len(
        knn_distances
    ) != n_samples * n_neighbors:
        raise Error("UMAP k-NN arrays do not match their shape")
    if not _finite(set_op_mix_ratio):
        raise Error("UMAP set operation mix ratio must be finite")
    if set_op_mix_ratio < Float32(0.0) or set_op_mix_ratio > Float32(1.0):
        raise Error("UMAP set operation mix ratio must be in [0, 1]")
    var rhos = List[Float32]()
    var sigmas = List[Float32]()
    rhos.resize(n_samples, Float32(0.0))
    sigmas.resize(n_samples, Float32(0.0))
    var target = log2(Float64(n_neighbors))
    for i in range(n_samples):
        if Int(knn_indices[i * n_neighbors]) != i:
            raise Error("UMAP expects self in k-NN slot zero")
        var previous = Float32(-1.0)
        var rho = Float64(0.0)
        for j in range(n_neighbors):
            var d = knn_distances[i * n_neighbors + j]
            if not _finite(d) or d < Float32(0.0) or (
                j > 0 and d < previous
            ):
                raise Error("UMAP k-NN distances must be finite and sorted")
            previous = d
            if rho == 0.0 and d > Float32(0.0):
                rho = Float64(d)
        rhos[i] = Float32(rho)
        var sigma: Float64
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            sigma = _sigma_identical(
                knn_distances, i, n_neighbors, rho, target
            )
        else:
            sigma = _sigma_fast(knn_distances, i, n_neighbors, rho, target)
        sigmas[i] = Float32(sigma)
    var directed = List[Float32]()
    directed.resize(n_samples * n_samples, Float32(0.0))
    for i in range(n_samples):
        for j in range(1, n_neighbors):
            var dst = Int(knn_indices[i * n_neighbors + j])
            if dst < 0 or dst >= n_samples or dst == i:
                raise Error("UMAP k-NN index is invalid or repeats self")
            var delta = knn_distances[i * n_neighbors + j] - rhos[i]
            var value = Float32(1.0)
            if delta > Float32(0.0):
                value = Float32(exp(-Float64(delta) / Float64(sigmas[i])))
            var at = i * n_samples + dst
            if value > directed[at]:
                directed[at] = value
    var weights = List[Float32]()
    weights.resize(n_samples * n_samples, Float32(0.0))
    for i in range(n_samples):
        for j in range(n_samples):
            var a = directed[i * n_samples + j]
            var b = directed[j * n_samples + i]
            var union = a + b - a * b
            var intersection = a * b
            weights[i * n_samples + j] = (
                set_op_mix_ratio * union
                + (Float32(1.0) - set_op_mix_ratio) * intersection
            )
    return FuzzySimplicialGraph(
        n_samples, n_neighbors, rhos^, sigmas^, directed^, weights^
    )
