# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CSR UMAP candidate: exact neighbors, sparse graph, same solver/optimizer.

Used by the source public estimator. For fixed k and 2D/3D output,
fuzzy graph and Lanczos basis storage are linear in n (ncv<=20). Exact
brute-force neighbor work remains quadratic in computation; its bounded
query-tile workspace is O(tile*n), not an n*n distance matrix. No whole-
process memory or speed claim follows without main-lane measurements.
"""

from core.hostptr import copy_f32
from max.gpu.host import DeviceBuffer, DeviceContext
from neighbors.estimator import knn_self_search_slot
from umap.sparse_graph import (
    SparseFuzzySimplicialGraph,
    categorical_intersection,
    general_intersection,
    sparse_fuzzy_simplicial_graph_device,
    ug_device_all_finite,
)
from umap.sparse_optimizer import optimize_sparse_layout
from umap.optimizer_identical_device import umap_positive_coo_device
from umap.params import UMAPParams, UMAP_MAX_COMPONENTS, UMAP_NEIGHBORS_IVF
from umap.ivf_neighbors import umap_ivf_self_knn_device
from umap.graph_out import UmapFitOut, umap_graph_capacity, umap_publish_graph, umap_publish_curve
from umap.spectral_init import spectral_initialize_device_coo


def sparse_fuzzy_graph_from_data(
    ctx: DeviceContext,
    x_rowmajor: List[Float32],
    n_samples: Int,
    n_features: Int,
    params: UMAPParams,
) raises -> SparseFuzzySimplicialGraph:
    """Reuse mojolearn's exact k-NN surface, then build UMAP memberships.
    The input goes up once (a bulk copy through pinned memory) and the rest
    is `sparse_fuzzy_graph_from_device` on that one buffer (lane
    cpu3-neighbors, 2026-10-04: the host walked every coordinate twice;
    lane vdf-scope-cluster7, 2026-11-24: the k-NN's queries are the uploaded
    rows copied on the device, where they used to go up a second time)."""
    params.validate(n_samples)
    if n_features < 1 or len(x_rowmajor) != n_samples * n_features:
        raise Error("UMAP input does not match its declared shape")
    var nx = len(x_rowmajor)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](nx)
    ctx.synchronize()
    copy_f32(x_rowmajor.unsafe_ptr(), hx.unsafe_ptr(), nx)
    var d_x = ctx.enqueue_create_buffer[DType.float32](nx)
    ctx.enqueue_copy(dst_buf=d_x, src_ptr=hx.unsafe_ptr())
    var graph = sparse_fuzzy_graph_from_device(ctx, d_x, n_samples, n_features, params)
    _ = hx^
    _ = d_x^
    return graph^


def sparse_fuzzy_graph_from_device(
    ctx: DeviceContext,
    mut d_x: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_features: Int,
    params: UMAPParams,
) raises -> SparseFuzzySimplicialGraph:
    """`sparse_fuzzy_graph_from_data` over rows ALREADY ON THE DEVICE (`d_x`
    holds at least `n_samples x n_features` row-major float32 on `ctx`: the
    host form's upload, or a model-selection fold slot of the metrics
    store, lane vdf-scope-cluster7). Checked finite on the device (one word
    back), the k-NN self-search over the rows with its queries copied on the
    device (`knn_self_search_slot`: the same plan, refusals, search and
    order on the same bytes; the cosine zero-row refusal a device
    reduction), then the graph build on the device. X never comes home."""
    params.validate(n_samples)
    if n_features < 1:
        raise Error("UMAP input does not match its declared shape")
    var nx = n_samples * n_features
    var nk = n_samples * params.n_neighbors
    var head = d_x.create_sub_buffer[DType.float32](0, nx)
    if params.neighbors == UMAP_NEIGHBORS_IVF:
        # lane sc-ann-1-umap: the IVF graph's self-first form on the device
        # (umap/ivf_neighbors.mojo); False (the exact plan, a short probe
        # pool) falls through to the exact search below, the same bits as
        # neighbors='exact'.
        if not ug_device_all_finite(ctx, head, nx):
            raise Error("UMAP input coordinates must be finite")
        var v_dist = ctx.enqueue_create_buffer[DType.float32](nk)
        var v_idx = ctx.enqueue_create_buffer[DType.uint32](nk)
        if umap_ivf_self_knn_device(ctx, head, n_samples, n_features, params, v_dist, v_idx):
            var ivf_graph = sparse_fuzzy_simplicial_graph_device(
                ctx, v_idx, v_dist, n_samples, params.n_neighbors,
                params.set_op_mix_ratio, params.local_connectivity, canonicalize=True,
            )
            _ = v_dist^
            _ = v_idx^
            _ = head^
            _ = d_x.unsafe_ptr()  # the rows outlive the search's launches
            return ivf_graph^
        _ = v_dist^
        _ = v_idx^
    var hd = ctx.enqueue_create_host_buffer[DType.float32](nk)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](nk)
    ctx.synchronize()
    if not ug_device_all_finite(ctx, head, nx):
        raise Error("UMAP input coordinates must be finite")
    if params.metric == -1:
        _ = knn_self_search_slot(
            ctx, head, n_samples, n_features, params.n_neighbors, hd.unsafe_ptr(), hi.unsafe_ptr(),
        )
    else:
        # the metric option (lane/algos-decomp, 2026-09-27): the kNN's own
        # metric arms, the knn-<metric> lanes' contracts
        _ = knn_self_search_slot(
            ctx, head, n_samples, n_features, params.n_neighbors, hd.unsafe_ptr(), hi.unsafe_ptr(),
            metric=params.metric, metric_arg=params.metric_arg,
        )
    # The k-NN's rows go up once; the self-first adapter and the whole graph
    # build run on the device (`sparse_fuzzy_simplicial_graph_device`).
    var d_dist = ctx.enqueue_create_buffer[DType.float32](nk)
    var d_idx = ctx.enqueue_create_buffer[DType.uint32](nk)
    ctx.enqueue_copy(dst_buf=d_dist, src_ptr=hd.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_idx, src_ptr=hi.unsafe_ptr())
    var graph = sparse_fuzzy_simplicial_graph_device(
        ctx, d_idx, d_dist, n_samples, params.n_neighbors,
        params.set_op_mix_ratio, params.local_connectivity, canonicalize=True,
    )
    _ = hd^
    _ = hi^
    _ = d_dist^
    _ = d_idx^
    _ = head^
    _ = d_x.unsafe_ptr()  # the rows outlive the search's launches
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


def _sparse_fit_checks(n_samples: Int, params: UMAPParams, initial_given: List[Float32]) raises:
    """`sparse_fit_transform`'s refusals after the shape test, in its order."""
    if params.n_components < 1 or params.n_components > UMAP_MAX_COMPONENTS:
        raise Error("UMAP fit_transform supports 1 to 32 output dimensions")
    var given = len(initial_given) > 0
    if given and len(initial_given) != n_samples * params.n_components:
        raise Error("UMAP initial embedding does not match n_samples x n_components")
    if not given and n_samples < 2 * params.n_components + 4:
        raise Error("UMAP fit_transform has too few samples for spectral init")


def _sparse_fit_layout(
    ctx: DeviceContext,
    var graph: SparseFuzzySimplicialGraph,
    n_samples: Int,
    params: UMAPParams,
    initial_given: List[Float32],
    target: List[Float32],
    target_kind: Int,
    target_dims: Int,
    target_n_neighbors: Int,
    target_weight: Float32,
    fit_out: UmapFitOut,
) raises -> List[Float32]:
    """Everything of the fit after the data's graph: the supervised
    intersection, the start, the optimizer (one body for the host-input and
    the device-row forms). `fit_out` (lane sc-ann-1-umap): the finished
    graph (after the supervised intersection, umap-learn's `graph_`) and
    the curve's (a, b) into the caller's arrays when asked
    (umap/graph_out.mojo; addresses 0 skip them)."""
    var tk = target_n_neighbors if target_n_neighbors > 0 else params.n_neighbors
    if target_kind != 0:
        graph = supervise_graph(
            ctx, graph, target, n_samples, target_kind, target_dims, tk, target_weight, params.random_seed,
        )
    umap_publish_graph(graph, umap_graph_capacity(n_samples, params.n_neighbors, target_kind, tk), fit_out)
    var initial: List[Float32]
    if len(initial_given) > 0:
        initial = initial_given.copy()
    else:
        initial = sparse_spectral_initialize(
            ctx, graph.copy(), params.n_components, params.random_seed
        )
    var epochs = params.n_epochs
    if epochs == 0:
        epochs = 200
    var ab = params.curve()
    umap_publish_curve(ab[0], ab[1], fit_out)
    return optimize_sparse_layout(
        ctx, initial^, graph, n_samples, params.n_components, epochs,
        initial_learning_rate=params.learning_rate,
        negative_sample_rate=params.negative_sample_rate,
        repulsion_strength=params.repulsion_strength,
        a=ab[0], b=ab[1],
        seed=params.random_seed,
    )


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
    out_indptr: Int = 0,
    out_indices: Int = 0,
    out_data: Int = 0,
    out_ab: Int = 0,
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
    _sparse_fit_checks(n_samples, params, initial_given)
    var graph = sparse_fuzzy_graph_from_data(
        ctx, x_rowmajor, n_samples, n_features, params
    )
    return _sparse_fit_layout(ctx, graph^, n_samples, params, initial_given, target, target_kind, target_dims,
        target_n_neighbors, target_weight, UmapFitOut(out_indptr, out_indices, out_data, out_ab))


def sparse_fit_transform_device(
    ctx: DeviceContext,
    mut d_x: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_features: Int,
    params: UMAPParams,
    initial_given: List[Float32] = List[Float32](),
    target: List[Float32] = List[Float32](),
    target_kind: Int = 0,
    target_dims: Int = 1,
    target_n_neighbors: Int = 0,
    target_weight: Float32 = Float32(0.5),
    out_indptr: Int = 0,
    out_indices: Int = 0,
    out_data: Int = 0,
    out_ab: Int = 0,
) raises -> List[Float32]:
    """`sparse_fit_transform` over rows ALREADY ON THE DEVICE (lane
    vdf-scope-cluster7: a model-selection fold slot of the metrics store,
    `d_x` at least `n_samples x n_features` on `ctx`). The same refusals in
    the same order, `sparse_fuzzy_graph_from_device` (the host form's own
    graph body), then the same layout: the same bits, and X never comes
    home."""
    params.validate(n_samples)
    if n_features < 1:
        raise Error("UMAP input does not match its declared shape")
    _sparse_fit_checks(n_samples, params, initial_given)
    var graph = sparse_fuzzy_graph_from_device(ctx, d_x, n_samples, n_features, params)
    return _sparse_fit_layout(ctx, graph^, n_samples, params, initial_given, target, target_kind, target_dims,
        target_n_neighbors, target_weight, UmapFitOut(out_indptr, out_indices, out_data, out_ab))
