# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`SpectralClustering.predict` on the device (lane/spectral-predict,
2026-09-15; DEVIATION 2860, new capability).

The rule, its reference (Bengio et al., NIPS 2003; Fowlkes et al., TPAMI
2004), its threshold and its tie rules are stated once, in
`spectral/host/spectral_predict_host.mojo`, which is also the CPU spelling.
This file runs steps 1 to 4 on the device: the fit's own `knn_search` for
the nearest-neighbors affinity, the slots (one thread per query sorts its k
neighbor indices; the dense affinity is one thread per cell), one
`nystrom_kernel` thread per query (the degree fold, the square root and the
projection, each fold over the slots in ascending slot order seeded `+0.0`),
then `kmeans_predict_device`, the fit's final assignment pass. `mu` is the
k-sized `spectral_predict_mu`. No fold crosses queries, so a row's label
does not depend on the batch. The refusals keep the host's order and
wording: `mu` first, then the slots (the first offending query, found by an
integer `Atomic.min`, read once before the assignment).
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import (
    ftz,
    identical_div,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)
from cluster.estimator import kmeans_predict_device
from cluster.impl.kmeans_params import METRIC_L2_EXPANDED
from core.device_zero import enqueue_fill
from neighbors.estimator import knn_search
from spectral.checks.device_io import download_f32, download_i32, upload_f32
from spectral.impl.spectral_predict_common import (
    SPECTRAL_AFFINITY_NEAREST_NEIGHBORS,
    SPECTRAL_PREDICT_ONE_WAY_EDGE,
    SpectralPrediction,
    SpectralPredictionState,
    spectral_affinity_refused,
    spectral_predict_check_state,
    spectral_predict_mu,
    spectral_predict_validate,
)

comptime SPECTRAL_PREDICT_TPB = 64
comptime _SP_NONE = Int32(0x7FFFFFFF)


def spectral_slots_knn_kernel(
    idx: MutPointer[UInt32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    bad: MutPointer[Int32, MutAnyOrigin],
    n_train_in: Int32,
    k_in: Int32,
    nq_in: Int32,
):
    """`spectral_slots_from_knn` for query `q`: its k neighbor indices in
    ascending order (the same insertion sort on distinct integers), every
    value the one-way edge weight. An index past the training rows folds `q`
    into `bad` (integer min: the first such query)."""
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if q >= Int(nq_in):
        return
    var k = Int(k_in)
    var base = q * k
    for s in range(k):
        var j = idx.unsafe_load(base + s)
        if Int(j) >= Int(n_train_in):
            _ = Atomic.min(bad, Int32(q))
        var v = Int32(j)
        var pos = s
        while pos > 0 and cols.unsafe_load(base + pos - 1) > v:
            cols.unsafe_store(base + pos, cols.unsafe_load(base + pos - 1))
            pos -= 1
        cols.unsafe_store(base + pos, v)
        vals.unsafe_store(base + s, SPECTRAL_PREDICT_ONE_WAY_EDGE)


def spectral_slots_dense_kernel(
    affinity: MutPointer[Float32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    bad: MutPointer[Int32, MutAnyOrigin],
    n_train_in: Int32,
    nq_in: Int32,
):
    """`spectral_slots_from_dense`, one thread per cell `(q, j)`: slot `j`
    is training row `j` (the values are the affinity itself); a refused
    value (`spectral_affinity_refused`, a bit test) folds `q` into `bad`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var w = Int(n_train_in)
    if i >= Int(nq_in) * w:
        return
    var q = i // w
    var j = i - q * w
    if spectral_affinity_refused(affinity.unsafe_load(i)):
        _ = Atomic.min(bad, Int32(q))
    cols.unsafe_store(i, Int32(j))


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
    """`host_spectral_predict` on the device (module docstring)."""
    spectral_predict_validate(
        n_train, n_queries, n_features, n_components, n_clusters, n_neighbors, affinity
    )
    spectral_predict_check_state(state, n_train, n_components, n_clusters)
    var mu = spectral_predict_mu(state.eigenvalues)
    var k = n_components
    comptime tpb = SPECTRAL_PREDICT_TPB
    var qgrid = (n_queries + tpb - 1) // tpb
    var bad = ctx.enqueue_create_buffer[DType.int32](1)
    enqueue_fill[DType.int32](ctx, bad, _SP_NONE)
    var width: Int
    var d_cols: DeviceBuffer[DType.int32]
    var d_vals: DeviceBuffer[DType.float32]
    var h_idx = ctx.enqueue_create_host_buffer[DType.uint32](1)
    if affinity == SPECTRAL_AFFINITY_NEAREST_NEIGHBORS:
        if len(input) < n_queries * n_features or len(train_x) < n_train * n_features:
            raise Error("spectral_predict: a buffer is shorter than its shape; refused by name")
        # The fit's own search, reading the caller's rows where they are.
        var nnz = n_queries * n_neighbors
        var h_dist = ctx.enqueue_create_host_buffer[DType.float32](nnz)
        h_idx = ctx.enqueue_create_host_buffer[DType.uint32](nnz)
        ctx.synchronize()
        _ = knn_search(
            ctx,
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(train_x.unsafe_ptr())),
            n_train,
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(input.unsafe_ptr())),
            n_queries,
            n_features,
            n_neighbors,
            h_dist.unsafe_ptr(),
            h_idx.unsafe_ptr(),
            True,
        )
        var d_idx = ctx.enqueue_create_buffer[DType.uint32](nnz)
        ctx.enqueue_copy(dst_buf=d_idx, src_ptr=h_idx.unsafe_ptr())
        width = n_neighbors
        d_cols = ctx.enqueue_create_buffer[DType.int32](nnz)
        d_vals = ctx.enqueue_create_buffer[DType.float32](nnz)
        ctx.enqueue_function[spectral_slots_knn_kernel](
            d_idx.unsafe_ptr(),
            d_cols.unsafe_ptr(),
            d_vals.unsafe_ptr(),
            bad.unsafe_ptr(),
            Int32(n_train),
            Int32(n_neighbors),
            Int32(n_queries),
            grid_dim=(qgrid, 1, 1),
            block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        _ = d_idx^
        _ = h_dist^
    else:
        if len(input) < n_queries * n_train:
            raise Error("spectral_predict: a buffer is shorter than its shape; refused by name")
        var cells = n_queries * n_train
        width = n_train
        d_vals = ctx.enqueue_create_buffer[DType.float32](cells)
        ctx.enqueue_copy(dst_buf=d_vals, src_ptr=input.unsafe_ptr())
        d_cols = ctx.enqueue_create_buffer[DType.int32](cells)
        ctx.enqueue_function[spectral_slots_dense_kernel](
            d_vals.unsafe_ptr(),
            d_cols.unsafe_ptr(),
            bad.unsafe_ptr(),
            Int32(n_train),
            Int32(n_queries),
            grid_dim=((cells + tpb - 1) // tpb, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    var d_diag = upload_f32(ctx, state.diag)
    var d_vecs = upload_f32(ctx, state.eigenvectors)
    var d_mu = upload_f32(ctx, mu)
    var d_cent = upload_f32(ctx, state.centroids)
    var d_out = ctx.enqueue_create_buffer[DType.float32](n_queries * k)
    var d_labels = ctx.enqueue_create_buffer[DType.uint32](n_queries)
    ctx.enqueue_function[nystrom_kernel](
        d_out.unsafe_ptr(),
        d_cols.unsafe_ptr(),
        d_vals.unsafe_ptr(),
        d_diag.unsafe_ptr(),
        d_vecs.unsafe_ptr(),
        d_mu.unsafe_ptr(),
        Int32(width),
        Int32(k),
        Int32(n_queries),
        grid_dim=(qgrid, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    # The slot refusal (the first offending query), before the assignment.
    var first_bad = download_i32(ctx, bad, 1)[0]
    if first_bad != _SP_NONE:
        _spectral_refuse_slots(
            Int(first_bad), affinity, input, h_idx.unsafe_ptr(), n_train, n_neighbors
        )
    kmeans_predict_device(
        ctx, d_out, n_queries, k, n_clusters, d_cent, d_labels, METRIC_L2_EXPANDED
    )
    var emb = download_f32(ctx, d_out, n_queries * k)
    var u_labels = List[UInt32](length=n_queries, fill=UInt32(0))
    ctx.enqueue_copy(dst_ptr=u_labels.unsafe_ptr(), src_buf=d_labels)
    ctx.synchronize()
    _ = bad^
    _ = h_idx^
    _ = d_cols^
    _ = d_vals^
    _ = d_diag^
    _ = d_vecs^
    _ = d_mu^
    _ = d_cent^
    _ = d_out^
    _ = d_labels^
    var labels = List[Int32](capacity=n_queries)
    for i in range(n_queries):
        labels.append(Int32(u_labels[i]))
    return SpectralPrediction(labels^, emb^)


def _spectral_refuse_slots(
    q: Int,
    affinity: Int,
    input: List[Float32],
    h_idx: MutPointer[UInt32, MutUntrackedOrigin],
    n_train: Int,
    n_neighbors: Int,
) raises:
    """The host's slot refusal for query `q`, in its wording: only the
    offending query's own row is read (the error path)."""
    if affinity == SPECTRAL_AFFINITY_NEAREST_NEIGHBORS:
        for s in range(n_neighbors):
            var j = h_idx[q * n_neighbors + s]
            if Int(j) >= n_train:
                raise Error(
                    "spectral_predict: the k-NN search returned training index " + String(j)
                    + " for n_train=" + String(n_train) + "; refused by name"
                )
    else:
        for j in range(n_train):
            if spectral_affinity_refused(input[q * n_train + j]):
                raise Error(
                    "spectral_predict: the affinity to the training rows has a non-finite or"
                    " negative value at (" + String(q) + ", " + String(j) + "); refused by name"
                )
    raise Error("spectral_predict: query " + String(q) + " was refused by its slots")
