# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""UMAP spectral initialization over the shipped cuVS/Lanczos path."""

from max.gpu.host import DeviceContext
from std.memory import bitcast

from core.identity_trace import IdentityTrace
from spectral.impl.sparse.coo import CooGraph
from spectral.impl.spectral.spectral_embedding import (
    MLSpectralEmbeddingParams,
    transform_connectivity,
)
from umap.graph import FuzzySimplicialGraph


def _finite(v: Float32) -> Bool:
    var bits = bitcast[DType.uint32](v)
    return ((bits >> UInt32(23)) & UInt32(0xFF)) != UInt32(0xFF)


def _weights_to_coo(
    weights: List[Float32], n_samples: Int
) raises -> CooGraph:
    if len(weights) != n_samples * n_samples:
        raise Error("UMAP spectral graph has the wrong dense shape")
    var rows = List[Int32]()
    var cols = List[Int32]()
    var vals = List[Float32]()
    for i in range(n_samples):
        for j in range(n_samples):
            var value = weights[i * n_samples + j]
            if not _finite(value) or value < Float32(0.0):
                raise Error("UMAP spectral graph contains an invalid weight")
            if i != j and value > Float32(0.0):
                rows.append(Int32(i))
                cols.append(Int32(j))
                vals.append(value)
    if len(vals) == 0:
        raise Error("UMAP spectral graph has no edges")
    return CooGraph(n_samples, rows^, cols^, vals^)


def spectral_initialize_weights(
    ctx: DeviceContext,
    weights: List[Float32],
    n_samples: Int,
    n_components: Int,
    n_neighbors: Int,
    seed: UInt64,
) raises -> List[Float32]:
    """Return row-major 2D/3D initialization, canonically signed and scaled.

    The eigensolver and normalized Laplacian are the repository's shipped
    spectral implementation. IDENTICAL inherits its pinned reductions and
    seeded Lanczos path. FAST inherits its faster numeric kernels. This host
    post-pass is ordered in both modes: the largest-magnitude entry in each
    column is made positive and each column is scaled to max magnitude 10.
    """
    if n_components != 2 and n_components != 3:
        raise Error("UMAP spectral initialization supports only 2D or 3D")
    # The shipped Lanczos configuration uses k=n_components+1 and requires
    # ncv=n-k > k+1 at these small shapes.
    if n_samples < 2 * n_components + 4:
        raise Error("UMAP spectral initialization has too few samples")
    var graph = _weights_to_coo(weights, n_samples)
    return spectral_initialize_coo(
        ctx, graph^, n_samples, n_components, n_neighbors, seed
    )


def spectral_initialize_coo(
    ctx: DeviceContext,
    var graph: CooGraph,
    n_samples: Int,
    n_components: Int,
    n_neighbors: Int,
    seed: UInt64,
) raises -> List[Float32]:
    """Shared solver/post-pass for dense-converted and CSR-origin COO.

    Caller supplies the same positive edges in row-major order. Lanczos
    uses ncv=min(n-k,max(2k+1,20)); UMAP2D/3D basis size is at most20*n,
    not n*n. This changes neither the solver nor its arithmetic.
    """
    if n_components != 2 and n_components != 3:
        raise Error("UMAP spectral initialization supports only 2D or 3D")
    if n_samples < 2 * n_components + 4:
        raise Error("UMAP spectral initialization has too few samples")
    if graph.n != n_samples:
        raise Error("UMAP spectral COO shape disagrees with n_samples")
    var config = MLSpectralEmbeddingParams(
        n_components=n_components + 1,
        n_neighbors=n_neighbors,
        norm_laplacian=True,
        drop_first=True,
        has_seed=True,
        seed=seed,
    )
    var embedding = List[Float32]()
    var trace = IdentityTrace.disabled()
    var n_out = transform_connectivity(
        ctx, config, graph^, embedding, trace
    )
    if n_out != n_components or len(embedding) != n_samples * n_components:
        raise Error("UMAP spectral solver returned the wrong shape")
    for c in range(n_components):
        var pivot = 0
        var peak = Float32(0.0)
        for i in range(n_samples):
            var value = embedding[i * n_components + c]
            if not _finite(value):
                raise Error("UMAP spectral solver returned a non-finite value")
            var magnitude = value if value >= Float32(0.0) else -value
            if magnitude > peak:
                peak = magnitude
                pivot = i
        if not (peak > Float32(0.0)):
            raise Error("UMAP spectral solver returned a zero component")
        var scale = Float32(10.0) / peak
        if embedding[pivot * n_components + c] < Float32(0.0):
            scale = -scale
        for i in range(n_samples):
            embedding[i * n_components + c] *= scale
    return embedding^


def spectral_initialize(
    ctx: DeviceContext,
    graph: FuzzySimplicialGraph,
    n_components: Int,
    seed: UInt64,
) raises -> List[Float32]:
    return spectral_initialize_weights(
        ctx, graph.weights, graph.n_samples, n_components,
        graph.n_neighbors, seed,
    )
