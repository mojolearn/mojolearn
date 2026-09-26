# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`SpectralClustering.predict` on the device (lane/spectral-predict,
2026-09-15; DEVIATION 2860, new capability).

The rule, its reference (Bengio et al., NIPS 2003; Fowlkes et al., TPAMI
2004), its threshold and its tie rules are stated once, in
`spectral/host/spectral_predict_host.mojo`, which is also the CPU spelling.
This file launches steps 1 to 4 on the device: the fit's own `knn_search`
for the nearest-neighbors affinity, the host slot builder (integer sorting
only), one `nystrom_kernel` thread per query (the degree fold, the square
root and the projection, each fold over the slots in ascending slot order
seeded `+0.0`), then `kmeans_predict`, the fit's final assignment pass. No
fold crosses queries, so a row's label does not depend on the batch.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from checks.numerics import (
    ftz,
    identical_div,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)
from cluster.estimator import kmeans_predict
from cluster.impl.kmeans_params import METRIC_L2_EXPANDED
from neighbors.estimator import knn_search
from spectral.checks.device_io import download_f32, upload_f32, upload_i32
from spectral.host.spectral_predict_host import (
    SPECTRAL_AFFINITY_NEAREST_NEIGHBORS,
    SpectralPrediction,
    SpectralPredictionState,
    SpectralSlots,
    spectral_predict_check_state,
    spectral_predict_mu,
    spectral_predict_validate,
    spectral_slots_from_dense,
    spectral_slots_from_knn,
)

comptime SPECTRAL_PREDICT_TPB = 64


def nystrom_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    diag: MutPointer[Float32, MutAnyOrigin],
    vecs: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    width_in: Int32,
    k_in: Int32,
    nq_in: Int32,
):
    """Steps 2 and 3 for query `q` (`host_spectral_nystrom`, statement for
    statement). `vecs` is `n_train x k` row-major; `dst` `n_queries x k`."""
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if q >= Int(nq_in):
        return
    var w = Int(width_in)
    var k = Int(k_in)
    var deg = Float32(0.0)
    for s in range(w):
        if cols.unsafe_load(q * w + s) >= Int32(0):
            deg = ftz(deg + vals.unsafe_load(q * w + s))
    var sd = ftz(identical_sqrt(deg))
    if sd == Float32(0.0):
        sd = Float32(1.0)
    for c in range(k):
        var acc = Float32(0.0)
        for s in range(w):
            var j = Int(cols.unsafe_load(q * w + s))
            if j >= 0:
                var kt = identical_div(
                    vals.unsafe_load(q * w + s), identical_mul(sd, diag.unsafe_load(j))
                )
                acc = ftz(identical_mul_add(kt, vecs.unsafe_load(j * k + c), acc))
        dst.unsafe_store(q * k + c, identical_div(identical_div(acc, mu.unsafe_load(c)), sd))


def spectral_predict_device(
    ctx: DeviceContext,
    input: List[Float32],
    train_x: List[Float32],
    n_train: Int,
    n_queries: Int,
    n_features: Int,
    n_components: Int,
    n_clusters: Int,
    n_neighbors: Int,
    affinity: Int,
    state: SpectralPredictionState,
) raises -> SpectralPrediction:
    """`host_spectral_predict` with the k-NN, the projection and the
    assignment on the device."""
    spectral_predict_validate(
        n_train, n_queries, n_features, n_components, n_clusters, n_neighbors, affinity
    )
    spectral_predict_check_state(state, n_train, n_components, n_clusters)
    var mu = spectral_predict_mu(state.eigenvalues)
    var k = n_components
    var slots: SpectralSlots
    if affinity == SPECTRAL_AFFINITY_NEAREST_NEIGHBORS:
        if len(input) < n_queries * n_features or len(train_x) < n_train * n_features:
            raise Error("spectral_predict: a buffer is shorter than its shape; refused by name")
        # The search goes through buffers the runtime made, exactly as the
        # fit's `create_connectivity_graph` stages it: the rows and queries in
        # host buffers, the outputs in host buffers, then copied out.
        var nnz = n_queries * n_neighbors
        var h_index = ctx.enqueue_create_host_buffer[DType.float32](n_train * n_features)
        var h_queries = ctx.enqueue_create_host_buffer[DType.float32](n_queries * n_features)
        var h_dist = ctx.enqueue_create_host_buffer[DType.float32](nnz)
        var h_idx = ctx.enqueue_create_host_buffer[DType.uint32](nnz)
        ctx.synchronize()
        for i in range(n_train * n_features):
            h_index.unsafe_ptr().unsafe_store(i, train_x[i])
        for i in range(n_queries * n_features):
            h_queries.unsafe_ptr().unsafe_store(i, input[i])
        _ = knn_search(
            ctx,
            h_index.unsafe_ptr(),
            n_train,
            h_queries.unsafe_ptr(),
            n_queries,
            n_features,
            n_neighbors,
            h_dist.unsafe_ptr(),
            h_idx.unsafe_ptr(),
            True,
        )
        var idx = List[UInt32](capacity=nnz)
        for e in range(nnz):
            var j = h_idx.unsafe_ptr().unsafe_load(e)
            if Int(j) >= n_train:
                raise Error(
                    "spectral_predict: the k-NN search returned training index " + String(j)
                    + " for n_train=" + String(n_train) + "; refused by name"
                )
            idx.append(j)
        _ = h_index^
        _ = h_queries^
        _ = h_dist^
        _ = h_idx^
        slots = spectral_slots_from_knn(idx, n_queries, n_neighbors)
    else:
        if len(input) < n_queries * n_train:
            raise Error("spectral_predict: a buffer is shorter than its shape; refused by name")
        slots = spectral_slots_from_dense(input, n_queries, n_train)
    var d_cols = upload_i32(ctx, slots.cols)
    var d_vals = upload_f32(ctx, slots.vals)
    var d_diag = upload_f32(ctx, state.diag)
    var d_vecs = upload_f32(ctx, state.eigenvectors)
    var d_mu = upload_f32(ctx, mu)
    var d_out = ctx.enqueue_create_buffer[DType.float32](n_queries * k)
    ctx.enqueue_memset(d_out, Float32(0.0))
    ctx.synchronize()
    comptime tpb = SPECTRAL_PREDICT_TPB
    ctx.enqueue_function[nystrom_kernel](
        d_out.unsafe_ptr(),
        d_cols.unsafe_ptr(),
        d_vals.unsafe_ptr(),
        d_diag.unsafe_ptr(),
        d_vecs.unsafe_ptr(),
        d_mu.unsafe_ptr(),
        Int32(slots.width),
        Int32(k),
        Int32(n_queries),
        grid_dim=((n_queries + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    var emb = download_f32(ctx, d_out, n_queries * k)
    _ = d_cols^
    _ = d_vals^
    _ = d_diag^
    _ = d_vecs^
    _ = d_mu^
    _ = d_out^
    var u_labels = List[UInt32](length=n_queries, fill=UInt32(0))
    kmeans_predict(
        ctx,
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(emb.unsafe_ptr())),
        n_queries,
        k,
        n_clusters,
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(state.centroids.unsafe_ptr())),
        MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(u_labels.unsafe_ptr())),
        METRIC_L2_EXPANDED,
    )
    var labels = List[Int32](capacity=n_queries)
    for i in range(n_queries):
        labels.append(Int32(u_labels[i]))
    return SpectralPrediction(labels^, emb^)
