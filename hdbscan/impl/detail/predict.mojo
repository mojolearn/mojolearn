# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`approximate_predict`: the label and probability of new points under a
fitted HDBSCAN.

Reference: `cuml-v26.08.00/cpp/src/hdbscan/detail/predict.cuh`
(`_find_neighbor_and_lambda` `:44-83`, `_find_cluster_and_probability`
`:103-142`, `_compute_knn_and_nearest_neighbor` `:146-197`,
`approximate_predict` `:220-262`) and its kernels,
`detail/kernels/predict.cuh` (`min_mutual_reachability_kernel` `:13-44`,
`cluster_probability_kernel` `:46-86`). The Python entry is cuML's
`hdbscan.pyx:1264` (`approximate_predict`). Steps run in the reference
order, on the device, vendor-agnostic: no branch on a vendor anywhere.

THE PASS
  1. k-NN of every query against the training rows at
     `neighborhood = (min_samples - 1) * 2` (`:161`), through
     `neighbors/estimator.mojo::knn_search_device_resident`, the identical
     k-NN's statements with the index and queries on the device and the
     result left there (lane vdf-misc4; was `knn_search_traced`).
  2. the query core distance, slot `min_samples - 1` of each sorted row
     (`reachability.mojo::core_distances`, the fit's own kernel).
  3. the nearest neighbor in mutual reachability space and its lambda
     (`min_mutual_reachability_kernel`, `prediction_lambda_kernel`).
  4. the label of that neighbor, kept only when the query's lambda is past
     the birth of the neighbor's selected cluster (or the cluster is the
     root), and the probability `min(death, lambda) / death`
     (`cluster_probability_kernel`).
Every kernel is one thread per query row, one pass over that row's
neighbors, no fold across rows and no atomic, so a row's answer cannot see
which other rows were in the call. That is what the identity harness's
batch part holds the public call to.

======================================================================
DEVIATION BLOCK -- DEVIATION 1615. THE NEAREST MUTUAL REACHABILITY
NEIGHBOR ON A TIE IS THE FIRST IN (distance, index) ORDER.
======================================================================
WHAT THEIRS DOES. `min_mutual_reachability_kernel` keeps the FIRST
neighbor, in k-NN slot order, whose mutual reachability distance is
strictly smaller than the best so far (`kernels/predict.cuh:35`). Their
slot order is `select_k`'s, sorted by distance only, so among equal
distances the slot order is arrival order (IDENTITY_PATHS row 11).

WHAT OURS DOES. The same strict `>` scan over slots that
`knn_search_traced` sorted on `(distance, index)` (DEVIATION 1602), so a
tie in mutual reachability distance resolves to the smallest distance and
then the smallest training index. That is theirs with the arrival order
pinned. The tie is not a corner case here: whenever a core distance
dominates, many neighbors share one mutual reachability distance, and
the chosen neighbor's LABEL is an output.

THE DIVISIONS go through `identical_div` (row 49's seam, DEVIATION 740):
`1 / min_mr_dist` (`predict.cuh:78-81`) and `min(death, lambda) / death`
(`kernels/predict.cuh:74-77`). The labels are Int32, the dtype of this
estimator's `labels_`; theirs are int64 (`hdbscan.pyx:1316`).
======================================================================
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import identical_div
from core.identity_trace import IdentityTrace
from hdbscan.checks.hdbscan_sabotage import HDB_SAB_NONE
from hdbscan.impl.detail.reachability import (
    core_distances,
    knn_inds_to_i32_kernel,
)
from hdbscan.impl.prediction_data import (
    PredictionData,
    prediction_neighborhood,
    refuse_nonfinite_queries,
)
from neighbors.estimator import knn_search_device_resident
from hdbscan.impl.detail.resident import HdbscanResident


comptime PREDICT_TPB = 256
"""Their `int tpb = 256` template default (`predict.cuh:44,103,146,220`)."""

comptime PREDICT_FLOAT32_MAX = Float32(3.4028234663852886e38)


def min_mutual_reachability_kernel(
    input_core_dists: MutPointer[Float32, MutAnyOrigin],
    prediction_core_dists: MutPointer[Float32, MutAnyOrigin],
    pairwise_dists: MutPointer[Float32, MutAnyOrigin],
    neighbor_indices: MutPointer[Int32, MutAnyOrigin],
    n_prediction_points: Int32,
    neighborhood_in: Int32,
    min_mr_dists: MutPointer[Float32, MutAnyOrigin],
    min_mr_indices: MutPointer[Int32, MutAnyOrigin],
):
    """`kernels/predict.cuh:13-44`, line for line. DEVIATION 1615."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_prediction_points):
        return
    var neighborhood = Int(neighborhood_in)
    var min_mr_dist = PREDICT_FLOAT32_MAX
    var min_mr_ind = Int32(-1)
    for i in range(neighborhood):
        var slot = idx * neighborhood + i
        var mr_dist = prediction_core_dists.unsafe_load(idx)
        var nb = Int(neighbor_indices.unsafe_load(slot))
        var nb_core = input_core_dists.unsafe_load(nb)
        if nb_core > mr_dist:
            mr_dist = nb_core
        var pd = pairwise_dists.unsafe_load(slot)
        if pd > mr_dist:
            mr_dist = pd
        if min_mr_dist > mr_dist:
            min_mr_dist = mr_dist
            min_mr_ind = Int32(nb)
    min_mr_dists.unsafe_store(idx, min_mr_dist)
    min_mr_indices.unsafe_store(idx, min_mr_ind)


def prediction_lambda_kernel(
    min_mr_dists: MutPointer[Float32, MutAnyOrigin],
    prediction_lambdas: MutPointer[Float32, MutAnyOrigin],
    n_prediction_points: Int32,
):
    """`predict.cuh:75-82`, the `map_offset` body:
    `dist > 0 ? 1 / dist : numeric_limits<float>::max()`."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_prediction_points):
        return
    var dist = min_mr_dists.unsafe_load(idx)
    if dist > Float32(0.0):
        prediction_lambdas.unsafe_store(idx, identical_div(Float32(1.0), dist))
    else:
        prediction_lambdas.unsafe_store(idx, PREDICT_FLOAT32_MAX)


def cluster_probability_kernel(
    min_mr_indices: MutPointer[Int32, MutAnyOrigin],
    prediction_lambdas: MutPointer[Float32, MutAnyOrigin],
    index_into_children: MutPointer[Int32, MutAnyOrigin],
    labels: MutPointer[Int32, MutAnyOrigin],
    deaths: MutPointer[Float32, MutAnyOrigin],
    selected_clusters: MutPointer[Int32, MutAnyOrigin],
    lambdas: MutPointer[Float32, MutAnyOrigin],
    n_leaves_in: Int32,
    n_prediction_points: Int32,
    predicted_labels: MutPointer[Int32, MutAnyOrigin],
    cluster_probabilities: MutPointer[Float32, MutAnyOrigin],
):
    """`kernels/predict.cuh:46-86`, line for line."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_prediction_points):
        return
    var n_leaves = n_leaves_in
    var cluster_label = labels.unsafe_load(Int(min_mr_indices.unsafe_load(idx)))
    var pl = prediction_lambdas.unsafe_load(idx)
    var out_label = Int32(-1)
    if cluster_label >= Int32(0):
        var sel = selected_clusters.unsafe_load(Int(cluster_label))
        if sel > n_leaves:
            var birth = lambdas.unsafe_load(
                Int(index_into_children.unsafe_load(Int(sel)))
            )
            if birth < pl:
                out_label = cluster_label
        elif sel == n_leaves:
            out_label = cluster_label
    predicted_labels.unsafe_store(idx, out_label)
    if out_label >= Int32(0):
        var sel2 = selected_clusters.unsafe_load(Int(cluster_label))
        var max_lambda = deaths.unsafe_load(Int(sel2 - n_leaves))
        if max_lambda > Float32(0.0):
            var num = max_lambda if max_lambda < pl else pl
            cluster_probabilities.unsafe_store(idx, identical_div(num, max_lambda))
        else:
            cluster_probabilities.unsafe_store(idx, Float32(1.0))
    else:
        cluster_probabilities.unsafe_store(idx, Float32(0.0))


struct PredictOutput(Movable):
    var labels: List[Int32]
    var probabilities: List[Float32]
    var min_mr_indices: List[Int32]
    var prediction_lambdas: List[Float32]

    def __init__(
        out self,
        var labels: List[Int32],
        var probabilities: List[Float32],
        var min_mr_indices: List[Int32],
        var prediction_lambdas: List[Float32],
    ):
        self.labels = labels^
        self.probabilities = probabilities^
        self.min_mr_indices = min_mr_indices^
        self.prediction_lambdas = prediction_lambdas^


struct PredictDevice(Movable):
    """`approximate_predict`'s four per-query results LEFT ON THE DEVICE
    (lane vdf-misc4): `membership_vector`'s soft pass reads the first two
    there, `approximate_predict` brings the last two home once."""

    var min_mr_indices: DeviceBuffer[DType.int32]
    var prediction_lambdas: DeviceBuffer[DType.float32]
    var labels: DeviceBuffer[DType.int32]
    var probabilities: DeviceBuffer[DType.float32]

    def __init__(
        out self,
        var min_mr_indices: DeviceBuffer[DType.int32],
        var prediction_lambdas: DeviceBuffer[DType.float32],
        var labels: DeviceBuffer[DType.int32],
        var probabilities: DeviceBuffer[DType.float32],
    ):
        self.min_mr_indices = min_mr_indices^
        self.prediction_lambdas = prediction_lambdas^
        self.labels = labels^
        self.probabilities = probabilities^


def approximate_predict_device(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    res: HdbscanResident,
    mut queries: DeviceBuffer[DType.float32],
    nq: Int,
    neighborhood: Int,
    min_samples: Int,
    tpb: Int = PREDICT_TPB,
) raises -> PredictDevice:
    """`predict.cuh:220-262` after the refusals, on the fitted snapshot
    `res` and the `nq x res.n` queries ALREADY ON THE DEVICE (lane
    vdf-misc4, 2026-11-24): the k-NN reads the snapshot's training rows as
    its index and leaves its result on the device
    (`knn_search_device_resident`; was `knn_search_traced`, whose result
    went to the host and back between two device steps), the cast kernel,
    `core_distances` and the three predict kernels are the same launches on
    the same values, and nothing comes home. One wait at the end."""
    var cells = nq * neighborhood
    var knn_dists = ctx.enqueue_create_buffer[DType.float32](cells)
    var idx_u32 = ctx.enqueue_create_buffer[DType.uint32](cells)
    var knn_inds = ctx.enqueue_create_buffer[DType.int32](cells)
    var prediction_core_dists = ctx.enqueue_create_buffer[DType.float32](nq)
    var min_mr_dists = ctx.enqueue_create_buffer[DType.float32](nq)
    var min_mr_inds = ctx.enqueue_create_buffer[DType.int32](nq)
    var prediction_lambdas = ctx.enqueue_create_buffer[DType.float32](nq)
    var out_labels = ctx.enqueue_create_buffer[DType.int32](nq)
    var out_probs = ctx.enqueue_create_buffer[DType.float32](nq)
    # The snapshot's buffers as local handles (reference counted; the memory
    # is the snapshot's), so a launch takes a pointer from a `var`.
    var xb = res.x.copy()
    var core_buf = res.core.copy()
    var labels_buf = res.labels.copy()
    var iic_buf = res.iic.copy()
    var deaths_buf = res.deaths.copy()
    var sel_buf = res.selected.copy()
    var lambdas_buf = res.lambdas.copy()

    # `:163-177` perform knn, device in and device out
    _ = knn_search_device_resident(
        ctx, trace, xb, res.m, queries, nq, res.n, neighborhood, knn_dists, idx_u32, True,
    )
    # the indices are narrowed to Int32 by the fit's kernel
    ctx.enqueue_function[knn_inds_to_i32_kernel](
        knn_inds.unsafe_ptr(), idx_u32.unsafe_ptr(), Int32(cells),
        grid_dim=((cells + tpb - 1) // tpb, 1, 1), block_dim=(tpb, 1, 1),
    )

    # `:181-186` Slice core distances, slot min_samples - 1 (its own one wait)
    core_distances(
        ctx, knn_dists, min_samples, neighborhood, nq, prediction_core_dists,
        tpb, HDB_SAB_NONE,
    )

    # `:44-83` _find_neighbor_and_lambda, `:103-142` _find_cluster_and_probability
    var blocks = (nq + tpb - 1) // tpb
    ctx.enqueue_function[min_mutual_reachability_kernel](
        core_buf.unsafe_ptr(),
        prediction_core_dists.unsafe_ptr(),
        knn_dists.unsafe_ptr(),
        knn_inds.unsafe_ptr(),
        Int32(nq),
        Int32(neighborhood),
        min_mr_dists.unsafe_ptr(),
        min_mr_inds.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.enqueue_function[prediction_lambda_kernel](
        min_mr_dists.unsafe_ptr(),
        prediction_lambdas.unsafe_ptr(),
        Int32(nq),
        grid_dim=(blocks, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.enqueue_function[cluster_probability_kernel](
        min_mr_inds.unsafe_ptr(),
        prediction_lambdas.unsafe_ptr(),
        iic_buf.unsafe_ptr(),
        labels_buf.unsafe_ptr(),
        deaths_buf.unsafe_ptr(),
        sel_buf.unsafe_ptr(),
        lambdas_buf.unsafe_ptr(),
        Int32(res.m),
        Int32(nq),
        out_labels.unsafe_ptr(),
        out_probs.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    if trace.enabled:
        trace.record_device(ctx, "hdbscan.predict.min_mr_inds", min_mr_inds, nq)
        trace.record_device(ctx, "hdbscan.predict.lambdas", prediction_lambdas, nq)
        trace.record_device(ctx, "hdbscan.predict.labels", out_labels, nq)
        trace.record_device(ctx, "hdbscan.predict.probabilities", out_probs, nq)

    # [[mojo-buffer-freed-at-last-use]]: every buffer outlives the queue.
    _ = knn_dists^
    _ = idx_u32^
    _ = knn_inds^
    _ = prediction_core_dists^
    _ = min_mr_dists^
    _ = xb^
    _ = core_buf^
    _ = labels_buf^
    _ = iic_buf^
    _ = deaths_buf^
    _ = sel_buf^
    _ = lambdas_buf^
    return PredictDevice(min_mr_inds^, prediction_lambdas^, out_labels^, out_probs^)


def upload_queries(ctx: DeviceContext, address: Int, rows: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
    """`rows x n` float32 query words at the host `address`, uploaded once
    into an exact-length buffer (the k-NN reads it as exactly that)."""
    var q = ctx.enqueue_create_buffer[DType.float32](rows * n)
    ctx.enqueue_copy(dst_buf=q, src_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=address))
    return q^


def approximate_predict_snapshot(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    res: HdbscanResident,
    queries_address: Int,
    nq: Int,
    min_samples: Int,
    labels_out: Int,
    probs_out: Int,
    tpb: Int = PREDICT_TPB,
) raises:
    """`approximate_predict` on a fitted snapshot (lane vdf-misc4): the
    `nq` query rows at the host `queries_address` uploaded once, the device
    body, and the labels and probabilities copied once each to the host
    addresses `labels_out` / `probs_out` (one wait)."""
    var neighborhood = prediction_neighborhood(min_samples, res.m)
    var q = upload_queries(ctx, queries_address, nq, res.n)
    var near = approximate_predict_device(ctx, trace, res, q, nq, neighborhood, min_samples, tpb)
    ctx.enqueue_copy(dst_ptr=MutPointer[Int32, MutAnyOrigin](unsafe_from_address=labels_out), src_buf=near.labels)
    ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=probs_out), src_buf=near.probabilities)
    ctx.synchronize()
    _ = q^
    _ = near^


def approximate_predict(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    x_host: List[Float32],
    m: Int,
    n: Int,
    input_core_dists: List[Float32],
    labels: List[Int32],
    tree_lambdas: List[Float32],
    pd: PredictionData,
    queries: List[Float32],
    n_prediction_points: Int,
    min_samples: Int,
    tpb: Int = PREDICT_TPB,
) raises -> PredictOutput:
    """`predict.cuh:220-262`. `x_host` is the training matrix (`m x n`,
    row-major), `input_core_dists` and `labels` the fit's, `min_samples`
    the estimator's parameter as the user gave it (theirs:
    `clusterer.min_samples or clusterer.min_cluster_size`,
    `hdbscan.pyx:1322`), NOT the fit's `+ 1` k. The host lists are taken
    into a temporary fitted snapshot and the one device body runs
    (`approximate_predict_device`, lane vdf-misc4)."""
    if n_prediction_points < 1:
        raise Error(
            "hdbscan.approximate_predict: points_to_predict has no rows;"
            " refused by name"
        )
    if (
        len(queries) < n_prediction_points * n or len(x_host) < m * n
        or len(input_core_dists) < m or len(labels) < m
        or len(tree_lambdas) < pd.n_edges or len(pd.deaths) < pd.n_clusters
        or len(pd.selected_clusters) < pd.n_selected_clusters
        or len(pd.index_into_children) < pd.n_edges + 1
    ):
        raise Error(
            "hdbscan.approximate_predict: a buffer is shorter than its shape;"
            " refused by name"
        )
    refuse_nonfinite_queries(queries, n_prediction_points, n)
    var neighborhood = prediction_neighborhood(min_samples, m)
    var nq = n_prediction_points

    trace.header(
        "hdbscan/impl/detail/predict.mojo approximate_predict n_rows="
        + String(m) + " n_cols=" + String(n) + " n_prediction_points="
        + String(nq) + " min_samples=" + String(min_samples)
        + " neighborhood=" + String(neighborhood)
    )
    var res = HdbscanResident(
        ctx, Int(x_host.unsafe_ptr()), m, n, Int(input_core_dists.unsafe_ptr()), Int(labels.unsafe_ptr()),
        0, Int(tree_lambdas.unsafe_ptr()), pd.n_edges, Int(pd.deaths.unsafe_ptr()), pd.n_clusters,
        Int(pd.selected_clusters.unsafe_ptr()), pd.n_selected_clusters,
        Int(pd.index_into_children.unsafe_ptr()), 0, 0, 0,
    )
    var q = upload_queries(ctx, Int(queries.unsafe_ptr()), nq, n)
    var near = approximate_predict_device(ctx, trace, res, q, nq, neighborhood, min_samples, tpb)
    var h_inds = List[Int32](unsafe_uninit_length=nq)
    var h_lam = List[Float32](unsafe_uninit_length=nq)
    var h_labels = List[Int32](unsafe_uninit_length=nq)
    var h_probs = List[Float32](unsafe_uninit_length=nq)
    ctx.enqueue_copy(dst_ptr=MutPointer[Int32, MutAnyOrigin](unsafe_from_address=Int(h_inds.unsafe_ptr())),
                     src_buf=near.min_mr_indices)
    ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(h_lam.unsafe_ptr())),
                     src_buf=near.prediction_lambdas)
    ctx.enqueue_copy(dst_ptr=MutPointer[Int32, MutAnyOrigin](unsafe_from_address=Int(h_labels.unsafe_ptr())),
                     src_buf=near.labels)
    ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(h_probs.unsafe_ptr())),
                     src_buf=near.probabilities)
    ctx.synchronize()
    _ = q^
    _ = near^
    _ = res^
    return PredictOutput(h_labels^, h_probs^, h_inds^, h_lam^)
