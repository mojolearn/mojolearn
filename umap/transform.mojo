# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Dense query-to-training UMAP transform, on the device.

Reference structure: umap-learn umap_.py smooth_knn_dist, transform and
init_graph_transform (https://github.com/lmcinnes/umap/blob/master/umap/umap_.py).
Training coordinates never move. The graph is bipartite, represented by
query*k indices/weights, without a fuzzy union or query-query edges.
With supported local_connectivity=1, transform rho is zero. Sigma search
skips slot zero; memberships include it, including a zero-distance edge.

DEVICE ROUTE (lane cpu4-umap, 2026-10-04). Until this lane the whole
transform after the k-NN ran on the host in Float64 (memberships, the
initialization, every refinement epoch), a host route inside a GPU
transform. Now: the inputs go up once and are checked for finiteness on
the device; the k-NN runs over the uploaded training block
(`knn_search_resident`); its n x k result goes up once; ONE launch (one
thread per query row) computes the memberships, the starting coordinates,
the per-edge schedule ratios and the row keys; then one launch per epoch
(one thread per row, in place: rows are independent) refines; the result
comes back once. Refusals are flags on the device, read back as a handful
of words at three points, and raised with the old messages.

Numerical contract: the per-row statements are `umap/transform_rows.mojo`,
shared with the host column (`umap/host/umap_oracle.mojo::host_umap_transform`),
so every column computes the same words: binary64 steps in the portable soft
arithmetic (`checks/soft_f64.mojo`; the Apple GPU has no float64), float32
steps through the pinned seams under the flush model. Refinement uses this
repository's SplitMix64 counter and floor-difference schedule, not umap-learn's
RNG and epochs-per-sample. Query batching does NOT change results: the sigma
floor's mean, the edge-weight scale and the negative-sample counter are all
per row, and the refinement epoch count does not read the request size, so a
batch of N is the concatenation of N batches of one, bitwise
(`umap/checks/batch_determinism_check.mojo`, `batch_epoch_cliff_check.mojo`).

Residual (neighbors family): `knn_search_resident` writes its result to host
pointers, so the n x k distances and indices make one host round trip, and
the queries are uploaded twice (once for the finiteness check, once by the
k-NN). A k-NN entry that leaves its result on the device removes both.
"""
# DEVIATION 2486: bulk host staging; stream/lifetime boundaries unchanged.
from bindings.hostptr import copy_f32
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from std.memory import bitcast
from neighbors.estimator import knn_search_resident
from umap.params import UMAPParams
from umap.transform_rows import (
    TR_EMBED_NONFINITE,
    TR_F32P,
    TR_FLAGS,
    TR_OK,
    TR_OUT_NONFINITE,
    TR_QUERY_NONFINITE,
    TR_TRAIN_NONFINITE,
    TR_U32P,
    TR_U64P,
    tr_alpha,
    tr_finite,
    tr_neg2ab,
    tr_raise,
    tr_rep2b,
    tr_row_initialize,
    tr_row_memberships,
    tr_row_refine_epoch,
    tr_row_refine_prep,
    tr_target,
)


comptime TR_TPB = 128
comptime _TR_I32P = MutPointer[Int32, MutAnyOrigin]


@always_inline
def _tr_gid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _tr_blocks(n: Int) -> Int:
    return max(1, (n + TR_TPB - 1) // TR_TPB)


def umap_transform_nonfinite_kernel(v: TR_F32P, flags: _TR_I32P, slot: Int32, n_in: Int32):
    """`flags[slot] = 1` on a NaN or infinity (every writer stores the same 1)."""
    var i = _tr_gid()
    if i >= Int(n_in):
        return
    if not tr_finite(v[i]):
        flags[Int(slot)] = Int32(1)


def umap_transform_prepare_kernel(
    dist: TR_F32P, idx: TR_U32P, weights: TR_F32P, emb: TR_F32P, coords: TR_F32P,
    scaled: TR_U64P, keys: TR_U64P, flags: _TR_I32P,
    rows_in: Int32, k_in: Int32, n_train_in: Int32, comps_in: Int32, target: UInt64,
):
    """One thread per query row: memberships, starting coordinates, schedule
    ratios and row key (`transform_rows.mojo`); a refusal sets its flag."""
    var row = _tr_gid()
    if row >= Int(rows_in):
        return
    var k = Int(k_in)
    var status = tr_row_memberships(dist, weights, row, k, target)
    if status == TR_OK:
        status = tr_row_initialize(idx, weights, emb, coords, row, k, Int(n_train_in), Int(comps_in))
    if status == TR_OK:
        status = tr_row_refine_prep(idx, weights, scaled, keys, row, k)
    if status != TR_OK:
        flags[status] = Int32(1)


def umap_transform_epoch_kernel(
    coords: TR_F32P, emb: TR_F32P, idx: TR_U32P, scaled: TR_U64P, keys: TR_U64P,
    flags: _TR_I32P, rows_in: Int32, k_in: Int32, comps_in: Int32, n_train_in: Int32,
    epoch_in: Int32, alpha: Float32, a: Float32, b_word: UInt64,
    neg2ab: Float32, rep2b: Float32, seed: UInt64, negative_rate_in: Int32,
):
    """One refinement epoch, one thread per query row, in place."""
    var row = _tr_gid()
    if row >= Int(rows_in):
        return
    var epoch = Int(epoch_in)
    var status = tr_row_refine_epoch(
        coords, emb, idx, scaled, keys[row], row, Int(k_in), Int(comps_in),
        Int(n_train_in), epoch, epoch, alpha, a, b_word, neg2ab, rep2b, seed,
        Int(negative_rate_in),
    )
    if status != TR_OK:
        flags[status] = Int32(1)


def _tr_check(ctx: DeviceContext, flags: DeviceBuffer[DType.int32]) raises:
    """The refusal flags back (TR_FLAGS words); the lowest set one raises."""
    var words = List[Int32](length=TR_FLAGS, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=words.unsafe_ptr(), src_buf=flags)
    ctx.synchronize()
    comptime for s in range(TR_FLAGS):
        if words[s] != Int32(0):
            tr_raise(s)


def _tr_flag_nonfinite(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int,
    mut flags: DeviceBuffer[DType.int32], slot: Int,
) raises:
    if n > 0:
        ctx.enqueue_function[umap_transform_nonfinite_kernel](
            buf.unsafe_ptr(), flags.unsafe_ptr(), Int32(slot), Int32(n),
            grid_dim=_tr_blocks(n), block_dim=TR_TPB,
        )


def transform(
    ctx: DeviceContext, training_data: List[Float32], training_embedding: List[Float32],
    queries: List[Float32], n_train: Int, n_queries: Int, n_features: Int,
    params: UMAPParams,
) raises -> List[Float32]:
    params.validate(n_train)
    if n_queries < 1 or n_features < 1 or (params.n_components < 1 or params.n_components > 32):
        raise Error("UMAP transform supports nonempty dense queries and 1 to 32 output dimensions")
    if len(training_data) != n_train * n_features or len(queries) != n_queries * n_features or len(training_embedding) != n_train * params.n_components:
        raise Error("UMAP transform input shape mismatch")
    var k = params.n_neighbors
    var comps = params.n_components
    if n_queries < 1 or n_train < 2 or k < 2 or k > n_train:
        raise Error("UMAP transform initialization dimensions are unsupported")
    if n_train >= 2147483647 or n_queries * k >= 2147483647 or n_queries * comps >= 2147483647:
        raise Error("UMAP transform: sizes are past the Int32 launch arguments")

    # The inputs up once (bulk copies through pinned staging, which the
    # k-NN's host-pointer interface also reads); finiteness checked on the
    # device.
    var n_x = len(training_data)
    var n_q = len(queries)
    var n_e = len(training_embedding)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n_x)
    var hq = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var he = ctx.enqueue_create_host_buffer[DType.float32](n_e)
    var hd = ctx.enqueue_create_host_buffer[DType.float32](n_queries * k)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_queries * k)
    ctx.synchronize()
    copy_f32(training_data.unsafe_ptr(), hx.unsafe_ptr(), n_x)
    copy_f32(queries.unsafe_ptr(), hq.unsafe_ptr(), n_q)
    copy_f32(training_embedding.unsafe_ptr(), he.unsafe_ptr(), n_e)
    var flags = ctx.enqueue_create_buffer[DType.int32](TR_FLAGS)
    ctx.enqueue_memset(flags, Int32(0))
    var dx = ctx.enqueue_create_buffer[DType.float32](n_x)
    var dq = ctx.enqueue_create_buffer[DType.float32](n_q)
    var de = ctx.enqueue_create_buffer[DType.float32](n_e)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dq, src_ptr=hq.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=de, src_ptr=he.unsafe_ptr())
    _tr_flag_nonfinite(ctx, dx, n_x, flags, TR_TRAIN_NONFINITE)
    _tr_flag_nonfinite(ctx, dq, n_q, flags, TR_QUERY_NONFINITE)
    _tr_flag_nonfinite(ctx, de, n_e, flags, TR_EMBED_NONFINITE)
    _tr_check(ctx, flags)
    _ = dq^
    _ = he^

    # The k-NN over the uploaded training block (host query pointer and
    # host result pointers: the neighbors family's interface).
    if params.metric == -1:
        _ = knn_search_resident(ctx, dx, hx.unsafe_ptr(), n_train, hq.unsafe_ptr(), n_queries,
                                n_features, k, hd.unsafe_ptr(), hi.unsafe_ptr())
    else:
        _ = knn_search_resident(ctx, dx, hx.unsafe_ptr(), n_train, hq.unsafe_ptr(), n_queries,
                                n_features, k, hd.unsafe_ptr(), hi.unsafe_ptr(),
                                metric=params.metric, metric_arg=params.metric_arg)
    ctx.synchronize()
    var edges = n_queries * k
    var dd = ctx.enqueue_create_buffer[DType.float32](edges)
    var di = ctx.enqueue_create_buffer[DType.uint32](edges)
    ctx.enqueue_copy(dst_buf=dd, src_ptr=hd.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=di, src_ptr=hi.unsafe_ptr())
    _ = dx^
    _ = hx^
    _ = hq^

    # Memberships, starting coordinates, schedule ratios, row keys.
    var dw = ctx.enqueue_create_buffer[DType.float32](edges)
    var dc = ctx.enqueue_create_buffer[DType.float32](n_queries * comps)
    var ds = ctx.enqueue_create_buffer[DType.uint64](edges)
    var dk = ctx.enqueue_create_buffer[DType.uint64](n_queries)
    var rg = _tr_blocks(n_queries)
    ctx.enqueue_function[umap_transform_prepare_kernel](
        dd.unsafe_ptr(), di.unsafe_ptr(), dw.unsafe_ptr(), de.unsafe_ptr(), dc.unsafe_ptr(),
        ds.unsafe_ptr(), dk.unsafe_ptr(), flags.unsafe_ptr(),
        Int32(n_queries), Int32(k), Int32(n_train), Int32(comps), tr_target(k),
        grid_dim=rg, block_dim=TR_TPB,
    )
    _tr_check(ctx, flags)
    _ = hd^
    _ = hi^
    _ = dd^

    var epochs = max(1, params.n_epochs // 3)
    if params.n_epochs == 0:
        # BATCH INVARIANCE. This read `100 if n_queries <= 10000 else 30`, so
        # one extra row in a request of ten thousand cut every other row's
        # refinement from 100 epochs to 30 and moved a row by 1.36 on a map
        # whose clusters sit about 11 apart, while the same one-row growth at
        # 9,999 moved no bit at all. The count no longer reads the request
        # size. The cost of that was measured, not asserted.
        epochs = 100
    var ab = params.curve()
    var a = ab[0]
    var b = ab[1]
    var lr = params.learning_rate
    var gamma = params.repulsion_strength
    var rate = params.negative_sample_rate
    if not isfinite(a) or not isfinite(b) or a <= Float32(0) or b <= Float32(0):
        raise Error("UMAP transform refinement requires positive finite parameters")
    if not isfinite(lr) or lr <= Float32(0) or (
        not isfinite(gamma) or gamma < Float32(0)
    ) or rate < 0 or rate > 2147483647:
        raise Error("UMAP transform optimizer controls are invalid")
    var b_word = bitcast[DType.uint64](Float64(b))
    var neg2ab = tr_neg2ab(a, b)
    var rep2b = tr_rep2b(gamma, b)
    for epoch in range(epochs):
        ctx.enqueue_function[umap_transform_epoch_kernel](
            dc.unsafe_ptr(), de.unsafe_ptr(), di.unsafe_ptr(), ds.unsafe_ptr(), dk.unsafe_ptr(),
            flags.unsafe_ptr(), Int32(n_queries), Int32(k), Int32(comps), Int32(n_train),
            Int32(epoch), tr_alpha(epoch, epochs, lr), a, b_word, neg2ab, rep2b,
            params.random_seed, Int32(rate),
            grid_dim=rg, block_dim=TR_TPB,
        )
    _tr_flag_nonfinite(ctx, dc, n_queries * comps, flags, TR_OUT_NONFINITE)
    _tr_check(ctx, flags)

    # The result back once.
    var result = List[Float32](length=n_queries * comps, fill=Float32(0.0))
    ctx.enqueue_copy(dst_ptr=result.unsafe_ptr(), src_buf=dc)
    ctx.synchronize()
    _ = flags^
    _ = de^
    _ = di^
    _ = dw^
    _ = dc^
    _ = ds^
    _ = dk^
    return result^
