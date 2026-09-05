# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CSR UMAP candidate: exact neighbors, sparse graph, same solver/optimizer.

Not wired into the public estimator yet. For fixed k and 2D/3D output,
fuzzy graph and Lanczos basis storage are linear in n (ncv<=20). Exact
brute-force neighbor work remains quadratic in computation; its bounded
query-tile workspace is O(tile*n), not an n*n distance matrix. No whole-
process memory or speed claim follows without main-lane measurements.
"""

from max.gpu.host import DeviceContext
from std.math import isfinite
from neighbors.estimator import knn_search
from umap.sparse_graph import SparseFuzzySimplicialGraph, sparse_fuzzy_simplicial_graph
from umap.curve import fit_umap_curve
from umap.sparse_optimizer import optimize_sparse_layout, validate_sparse_weights
from umap.params import UMAPParams
from umap.spectral_init import spectral_initialize_coo
from spectral.impl.sparse.coo import CooGraph


def sparse_fuzzy_graph_from_data(
    ctx: DeviceContext,
    x_rowmajor: List[Float32],
    n_samples: Int,
    n_features: Int,
    params: UMAPParams,
) raises -> SparseFuzzySimplicialGraph:
    """Reuse mojolearn's exact k-NN surface, then build UMAP memberships."""
    params.validate(n_samples)
    if n_features < 1 or len(x_rowmajor) != n_samples * n_features:
        raise Error("UMAP input does not match its declared shape")
    for i in range(len(x_rowmajor)):
        if not isfinite(x_rowmajor[i]):
            raise Error("UMAP input coordinates must be finite")
    var hx = ctx.enqueue_create_host_buffer[DType.float32](len(x_rowmajor))
    for i in range(len(x_rowmajor)):
        hx.unsafe_ptr().unsafe_store(i, x_rowmajor[i])
    var hd = ctx.enqueue_create_host_buffer[DType.float32](
        n_samples * params.n_neighbors
    )
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](
        n_samples * params.n_neighbors
    )
    _ = knn_search(
        ctx, hx.unsafe_ptr(), n_samples, hx.unsafe_ptr(), n_samples,
        n_features, params.n_neighbors, hd.unsafe_ptr(), hi.unsafe_ptr(),
    )
    var distances = List[Float32]()
    var indices = List[UInt32]()
    for i in range(n_samples * params.n_neighbors):
        distances.append(hd.unsafe_ptr().unsafe_load(i))
        indices.append(hi.unsafe_ptr().unsafe_load(i))
    _ = hx^
    _ = hd^
    _ = hi^
    return sparse_fuzzy_simplicial_graph(
        indices^, distances^, n_samples, params.n_neighbors,
        params.set_op_mix_ratio,
    )



def sparse_spectral_initialize(
    ctx: DeviceContext, graph: SparseFuzzySimplicialGraph,
    n_components: Int, seed: UInt64,
) raises -> List[Float32]:
    """Feed the exact dense adapter's positive row-major COO to its solver."""
    _ = validate_sparse_weights(graph)
    var rows = List[Int32]()
    var cols = List[Int32]()
    var vals = List[Float32]()
    for row in range(graph.n_samples):
        for edge in range(graph.offsets[row], graph.offsets[row + 1]):
            var value = graph.values[edge]
            if value > Float32(0.0):
                rows.append(Int32(row))
                cols.append(Int32(graph.indices[edge]))
                vals.append(value)
    if len(vals) == 0:
        raise Error("UMAP spectral graph has no edges")
    var coo = CooGraph(graph.n_samples, rows^, cols^, vals^)
    return spectral_initialize_coo(
        ctx, coo^, graph.n_samples, n_components, graph.n_neighbors, seed
    )


def sparse_fit_transform(
    ctx: DeviceContext,
    x_rowmajor: List[Float32],
    n_samples: Int,
    n_features: Int,
    params: UMAPParams,
) raises -> List[Float32]:
    """Exact k-NN → fuzzy graph → spectral init → serial UMAP optimizer.

    FAST uses a conflict-free GPU Jacobi optimizer with one owner per output
    row and an epoch snapshot; it is tolerance-compared, not bit-compared, to
    the serial update trajectory. IDENTICAL retains the k-NN/eigensolver
    cross-vendor contracts and the optimizer's stable host update order.
    """
    params.validate(n_samples)
    if n_features < 1 or len(x_rowmajor) != n_samples * n_features:
        raise Error("UMAP input does not match its declared shape")
    if params.n_components != 2 and params.n_components != 3:
        raise Error("UMAP fit_transform currently supports only 2D or 3D")
    if n_samples < 2 * params.n_components + 4:
        raise Error("UMAP fit_transform has too few samples for spectral init")
    var graph = sparse_fuzzy_graph_from_data(
        ctx, x_rowmajor, n_samples, n_features, params
    )
    var initial = sparse_spectral_initialize(
        ctx, graph.copy(), params.n_components, params.random_seed
    )
    var epochs = params.n_epochs
    if epochs == 0:
        epochs = 200
    var curve = fit_umap_curve(params.min_dist, params.spread)
    return optimize_sparse_layout(
        ctx, initial^, graph, n_samples, params.n_components, epochs,
        a=curve.a, b=curve.b,
        seed=params.random_seed,
    )
