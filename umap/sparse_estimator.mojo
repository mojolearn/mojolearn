# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CSR UMAP candidate: exact neighbors, sparse graph, same solver/optimizer.

Used by the source public estimator. For fixed k and 2D/3D output,
fuzzy graph and Lanczos basis storage are linear in n (ncv<=20). Exact
brute-force neighbor work remains quadratic in computation; its bounded
query-tile workspace is O(tile*n), not an n*n distance matrix. No whole-
process memory or speed claim follows without main-lane measurements.
"""

from bindings.hostptr import copy_f32
from max.gpu.host import DeviceContext
from neighbors.estimator import knn_search_resident
from umap.sparse_graph import (
    SparseFuzzySimplicialGraph,
    categorical_intersection,
    general_intersection,
    sparse_fuzzy_simplicial_graph_device,
    ug_device_all_finite,
)
from umap.sparse_optimizer import optimize_sparse_layout
from umap.optimizer_identical_device import umap_positive_coo_device
from umap.params import UMAPParams, UMAP_MAX_COMPONENTS
from umap.spectral_init import spectral_initialize_device_coo


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
    # The input goes up once (a bulk copy through pinned memory), is checked
    # for finiteness on the device (one word back) and is the k-NN's
    # resident index (lane cpu3-neighbors, 2026-10-04: the host walked every
    # coordinate twice, a finiteness test and an element-by-element store).
    var nx = len(x_rowmajor)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](nx)
    var hd = ctx.enqueue_create_host_buffer[DType.float32](
        n_samples * params.n_neighbors
    )
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](
        n_samples * params.n_neighbors
    )
    ctx.synchronize()
    copy_f32(x_rowmajor.unsafe_ptr(), hx.unsafe_ptr(), nx)
    var d_x = ctx.enqueue_create_buffer[DType.float32](nx)
    ctx.enqueue_copy(dst_buf=d_x, src_ptr=hx.unsafe_ptr())
    if not ug_device_all_finite(ctx, d_x, nx):
        raise Error("UMAP input coordinates must be finite")
    if params.metric == -1:
        _ = knn_search_resident(
            ctx, d_x, hx.unsafe_ptr(), n_samples, hx.unsafe_ptr(), n_samples,
            n_features, params.n_neighbors, hd.unsafe_ptr(), hi.unsafe_ptr(),
        )
    else:
        # the metric option (lane/algos-decomp, 2026-09-27): the kNN's own
        # metric arms, the knn-<metric> lanes' contracts
        _ = knn_search_resident(
            ctx, d_x, hx.unsafe_ptr(), n_samples, hx.unsafe_ptr(), n_samples,
            n_features, params.n_neighbors, hd.unsafe_ptr(), hi.unsafe_ptr(),
            metric=params.metric, metric_arg=params.metric_arg,
        )
    _ = d_x^
    # The k-NN's rows go up once; the self-first adapter and the whole graph
    # build run on the device (`sparse_fuzzy_simplicial_graph_device`).
    var nk = n_samples * params.n_neighbors
    var d_dist = ctx.enqueue_create_buffer[DType.float32](nk)
    var d_idx = ctx.enqueue_create_buffer[DType.uint32](nk)
    ctx.enqueue_copy(dst_buf=d_dist, src_ptr=hd.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_idx, src_ptr=hi.unsafe_ptr())
    var graph = sparse_fuzzy_simplicial_graph_device(
        ctx, d_idx, d_dist, n_samples, params.n_neighbors,
        params.set_op_mix_ratio, params.local_connectivity, canonicalize=True,
    )
    _ = hx^
    _ = hd^
    _ = hi^
    _ = d_dist^
    _ = d_idx^
    return graph^



def sparse_spectral_initialize(
    ctx: DeviceContext, graph: SparseFuzzySimplicialGraph,
    n_components: Int, seed: UInt64,
) raises -> List[Float32]:
    """Feed the exact dense adapter's positive row-major COO to its solver.
    The graph's validation and the COO run on the device and the COO stays
    there (`umap_positive_coo_device`, lane cpu3-neighbors)."""
    var coo = umap_positive_coo_device(
        ctx, graph.offsets, graph.indices, graph.values, graph.n_samples
    )
    var nnz = coo.nnz
    # shared handles (DeviceBuffer copies retain the same allocation, no data
    # copy): moving fields out of coo is refused (box-run-2 compile fix)
    var res = spectral_initialize_device_coo(
        ctx, graph.n_samples, nnz, coo.rows.copy(), coo.cols.copy(), coo.vals.copy(),
        n_components, graph.n_neighbors, seed,
    )
    _ = coo^
    return res^


def supervise_graph(
    ctx: DeviceContext, graph: SparseFuzzySimplicialGraph, target: List[Float32], n_samples: Int,
    target_kind: Int, target_dims: Int, target_n_neighbors: Int, target_weight: Float32, seed: UInt64,
) raises -> SparseFuzzySimplicialGraph:
    """umap-learn's supervised graph (DEVIATION 5324): target_kind 1 is
    'categorical' (labels, -1 unknown; far_dist 2.5 / (1 - target_weight),
    1e12 at weight 1), 2 is a continuous target under the euclidean metric
    (its own fuzzy graph, local connectivity 1 and mix 1, intersected at
    target_weight). Both end with reset_local_connectivity."""
    if target_kind == 1:
        var far = Float64(1.0e12)
        if target_weight < Float32(1.0):
            far = Float64(2.5) * (Float64(1.0) / (Float64(1.0) - Float64(target_weight)))
        return categorical_intersection(ctx, graph, target, far)
    var tp = UMAPParams(n_neighbors=target_n_neighbors, n_components=2, random_seed=seed)
    var tgraph = sparse_fuzzy_graph_from_data(ctx, target, n_samples, target_dims, tp)
    return general_intersection(ctx, graph, tgraph, target_weight)


def sparse_fit_transform(
    ctx: DeviceContext,
    x_rowmajor: List[Float32],
    n_samples: Int,
    n_features: Int,
    params: UMAPParams,
    initial_given: List[Float32] = List[Float32](),
    target: List[Float32] = List[Float32](),
    target_kind: Int = 0,
    target_dims: Int = 1,
    target_n_neighbors: Int = 0,
    target_weight: Float32 = Float32(0.5),
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
    if params.n_components < 1 or params.n_components > UMAP_MAX_COMPONENTS:
        raise Error("UMAP fit_transform supports 1 to 32 output dimensions")
    var given = len(initial_given) > 0
    if given and len(initial_given) != n_samples * params.n_components:
        raise Error("UMAP initial embedding does not match n_samples x n_components")
    if not given and n_samples < 2 * params.n_components + 4:
        raise Error("UMAP fit_transform has too few samples for spectral init")
    var graph = sparse_fuzzy_graph_from_data(
        ctx, x_rowmajor, n_samples, n_features, params
    )
    if target_kind != 0:
        var tk = target_n_neighbors if target_n_neighbors > 0 else params.n_neighbors
        graph = supervise_graph(
            ctx, graph, target, n_samples, target_kind, target_dims, tk, target_weight, params.random_seed,
        )
    var initial: List[Float32]
    if given:
        initial = initial_given.copy()
    else:
        initial = sparse_spectral_initialize(
            ctx, graph.copy(), params.n_components, params.random_seed
        )
    var epochs = params.n_epochs
    if epochs == 0:
        epochs = 200
    var ab = params.curve()
    return optimize_sparse_layout(
        ctx, initial^, graph, n_samples, params.n_components, epochs,
        initial_learning_rate=params.learning_rate,
        negative_sample_rate=params.negative_sample_rate,
        repulsion_strength=params.repulsion_strength,
        a=ab[0], b=ab[1],
        seed=params.random_seed,
    )
