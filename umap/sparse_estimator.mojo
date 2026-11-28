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
from umap.ivf_neighbors import umap_ivf_self_knn_device, umap_knn_strip_kernel
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
    reduction), then the graph build on the device. X never comes home.
    The body is `sparse_fuzzy_graph_knn_device` over k-NN buffers of its own."""
    var nk = n_samples * params.n_neighbors
    var d_dist = ctx.enqueue_create_buffer[DType.float32](max(nk, 1))
    var d_idx = ctx.enqueue_create_buffer[DType.uint32](max(nk, 1))
    var graph = sparse_fuzzy_graph_knn_device(ctx, d_x, n_samples, n_features, params, d_dist, d_idx)
    _ = d_dist^
    _ = d_idx^
    return graph^


def sparse_fuzzy_graph_knn_device(
    ctx: DeviceContext,
    mut d_x: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_features: Int,
    params: UMAPParams,
    mut d_dist: DeviceBuffer[DType.float32],
    mut d_idx: DeviceBuffer[DType.uint32],
) raises -> SparseFuzzySimplicialGraph:
    """`sparse_fuzzy_graph_from_device` with the caller's n x n_neighbors
    k-NN buffers: on return they hold the self-first euclidean (or metric)
    k-NN the graph was built from (the build's canonicalize pass runs in
    place on them), which `mojolearn.neighbors_graph` returns as its
    distances (lane sc-ann-1-umap)."""
    params.validate(n_samples)
    if n_features < 1:
        raise Error("UMAP input does not match its declared shape")
    var nx = n_samples * n_features
    var nk = n_samples * params.n_neighbors
    if len(d_dist) < nk or len(d_idx) < nk:
        raise Error("UMAP k-NN buffers do not match n_samples x n_neighbors")
    var head = d_x.create_sub_buffer[DType.float32](0, nx)
    if params.neighbors == UMAP_NEIGHBORS_IVF:
        # lane sc-ann-1-umap: the IVF graph's self-first form on the device
        # (umap/ivf_neighbors.mojo); False (the exact plan, a short probe
        # pool) falls through to the exact search below, the same bits as
        # neighbors='exact'.
        if not ug_device_all_finite(ctx, head, nx):
            raise Error("UMAP input coordinates must be finite")
        if umap_ivf_self_knn_device(ctx, head, n_samples, n_features, params, d_dist, d_idx):
            var ivf_graph = sparse_fuzzy_simplicial_graph_device(
                ctx, d_idx, d_dist, n_samples, params.n_neighbors,
                params.set_op_mix_ratio, params.local_connectivity, canonicalize=True,
            )
            _ = head^
            _ = d_x.unsafe_ptr()  # the rows outlive the search's launches
            return ivf_graph^
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
    var dist_head = d_dist.create_sub_buffer[DType.float32](0, nk)
    var idx_head = d_idx.create_sub_buffer[DType.uint32](0, nk)
    ctx.enqueue_copy(dst_buf=dist_head, src_ptr=hd.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=idx_head, src_ptr=hi.unsafe_ptr())
    var graph = sparse_fuzzy_simplicial_graph_device(
        ctx, d_idx, d_dist, n_samples, params.n_neighbors,
        params.set_op_mix_ratio, params.local_connectivity, canonicalize=True,
    )
    _ = hd^
    _ = hi^
    _ = dist_head^
    _ = idx_head^
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


def _fitted_graph(
    ctx: DeviceContext,
    var graph: SparseFuzzySimplicialGraph,
    n_samples: Int,
    params: UMAPParams,
    target: List[Float32],
    target_kind: Int,
    target_dims: Int,
    target_n_neighbors: Int,
    target_weight: Float32,
    fit_out: UmapFitOut,
) raises -> SparseFuzzySimplicialGraph:
    """The data's graph after the supervised intersection (umap-learn's
    `graph_`), published into `fit_out` when its `indptr` is asked: the one
    body of the fit and of `umap_fit_graph`."""
    var tk = target_n_neighbors if target_n_neighbors > 0 else params.n_neighbors
    if target_kind != 0:
        graph = supervise_graph(
            ctx, graph, target, n_samples, target_kind, target_dims, tk, target_weight, params.random_seed,
        )
    umap_publish_graph(graph, umap_graph_capacity(n_samples, params.n_neighbors, target_kind, tk), fit_out)
    return graph^


def umap_fit_graph(
    ctx: DeviceContext,
    x_rowmajor: List[Float32],
    n_samples: Int,
    n_features: Int,
    params: UMAPParams,
    target: List[Float32],
    target_kind: Int,
    target_dims: Int,
    target_n_neighbors: Int,
    target_weight: Float32,
    dst: UmapFitOut,
) raises -> Int:
    """UMAP_OP_FIT_GRAPH (`UMAP.graph_`, lane sc-leftovers-1): the graph
    `sparse_fit_transform` builds from the same words (the same graph body
    and `_fitted_graph`), into the caller's arrays, without the start or
    the layout. The fit no longer publishes it, so a fit makes no graph
    arrays; the graph is built again only when `graph_` is read. Returns
    the edge count."""
    params.validate(n_samples)
    if n_features < 1 or len(x_rowmajor) != n_samples * n_features:
        raise Error("UMAP input does not match its declared shape")
    var graph = sparse_fuzzy_graph_from_data(ctx, x_rowmajor, n_samples, n_features, params)
    graph = _fitted_graph(ctx, graph^, n_samples, params, target, target_kind, target_dims, target_n_neighbors,
                          target_weight, dst)
    return len(graph.indices)


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
    graph = _fitted_graph(ctx, graph^, n_samples, params, target, target_kind, target_dims, target_n_neighbors,
                          target_weight, fit_out)
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


def umap_neighbors_graph_device(
    ctx: DeviceContext,
    mut d_x: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_features: Int,
    params: UMAPParams,
    dist_indptr: Int,
    dist_indices: Int,
    dist_data: Int,
    conn: UmapFitOut,
) raises -> Int:
    """`mojolearn.neighbors_graph` on the device (lane sc-ann-1-umap): UMAP's
    own graph body (`sparse_fuzzy_graph_knn_device`, exact or IVF), its
    fuzzy simplicial set into the caller's `conn` arrays (the same CSR as
    `UMAP.graph_`, capacity `umap_graph_capacity(n, k, 0, k)`), and the
    k-NN it was built from, self dropped, as the distances CSR: one strip
    launch, three downloads into the caller's n + 1 int64, n (k - 1) int32
    and n (k - 1) float32 words. Returns the connectivities' edge count."""
    var n = n_samples
    var k = params.n_neighbors
    var nk = n * k
    var d_dist = ctx.enqueue_create_buffer[DType.float32](max(nk, 1))
    var d_idx = ctx.enqueue_create_buffer[DType.uint32](max(nk, 1))
    var graph = sparse_fuzzy_graph_knn_device(ctx, d_x, n, n_features, params, d_dist, d_idx)
    var kept = n * (k - 1)
    var s_ptr = ctx.enqueue_create_buffer[DType.int64](n + 1)
    var s_idx = ctx.enqueue_create_buffer[DType.int32](max(kept, 1))
    var s_val = ctx.enqueue_create_buffer[DType.float32](max(kept, 1))
    ctx.enqueue_function[umap_knn_strip_kernel](
        d_dist.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), d_idx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        s_ptr.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), s_idx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        s_val.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), Int32(k),
        grid_dim=(n + 1 + 255) // 256, block_dim=256,
    )
    var ptr_head = s_ptr.create_sub_buffer[DType.int64](0, n + 1)
    ctx.enqueue_copy(dst_ptr=MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=dist_indptr), src_buf=ptr_head)
    if kept > 0:
        var idx_head = s_idx.create_sub_buffer[DType.int32](0, kept)
        var val_head = s_val.create_sub_buffer[DType.float32](0, kept)
        ctx.enqueue_copy(dst_ptr=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=dist_indices),
                         src_buf=idx_head)
        ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dist_data),
                         src_buf=val_head)
        ctx.synchronize()
        _ = idx_head^
        _ = val_head^
    ctx.synchronize()
    umap_publish_graph(graph, umap_graph_capacity(n, k, 0, k), conn)
    var nnz = len(graph.indices)
    _ = ptr_head^
    _ = s_ptr^
    _ = s_idx^
    _ = s_val^
    _ = d_dist^
    _ = d_idx^
    _ = d_x.unsafe_ptr()
    return nnz
