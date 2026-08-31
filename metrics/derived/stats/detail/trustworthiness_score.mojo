# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RAFT `cpp/include/raft/stats/detail/trustworthiness_score.cuh` (ebf9268;
cuVS `cpp/src/stats/detail/trustworthiness_score.cuh` is the same text and
is what cuML 26.08 links through `trustworthiness.cu`).

THEIRS (:122-207), `distance_type = L2SqrtUnexpanded` (the one cuML
instantiates, trustworthiness.cu:25-33):

    emb_ind = brute_force_knn(X_embedded, k = n_neighbors + 1)              (:81-108 run_knn)
    for each batch of rows of X:
        X_dist = pairwise_distance(X[batch], X, L2SqrtUnexpanded)
        X_ind  = sort_cols_per_row(X_dist)                                  a per-row key-value radix sort
        lookup_table[sample, X_ind[sample, rank]] = rank                    (:29-39 build_lookup_table)
        compute_rank (:52-69): for each (sample, one of its k+1 embedded neighbors):
            r = lookup_table[sample, emb_nn]; tmp = r - (n_neighbors + 1) + 1
            if tmp > 0: atomicAdd(rank, double(tmp))
    t = 1 - (2 / ((n * n_neighbors) * (2n - 3 n_neighbors - 1))) * t        (:204)

sklearn `trustworthiness(X, X_embedded, n_neighbors=5)`: the same sum
(`ranks = inverted_index[..., ind_X_embedded] - n_neighbors; t = sum(ranks
[ranks > 0])`, self excluded from both neighborhoods -- RAFT's k+1 includes
self at rank 0, which contributes `0 - k <= 0`, nothing) and the same
closed form.

THE SUM IS INTEGER (their `double` accumulator holds integers below 2^53
exactly), so it is order-free and identity-safe; the one float op is the
host's closed form, correctly rounded everywhere. What is NOT integer is
the two neighbor structures it reads: the embedded k-NN and the rank of
every embedded neighbor in the original space, both decided by Float32
distances and their ties.

=========================================================================
DEVIATION 655 (metrics lane, 2026-08-23): THE RANKS ARE COUNTED, NOT
SORTED, AND THE EMBEDDED k-NN IS THIS REPOSITORY'S `knn_search`.
=========================================================================
(1) THEIRS sorts every row of X's full n x n distance matrix
(`sort_cols_per_row`, a stable radix sort over a `batchSize x n` tile,
then a lookup table) and reads k+1 entries of it. OURS computes, per row
i and per embedded neighbor e, `rank(e) = #{j : d(i,j) < d(i,e)} + #{j :
d(i,j) == d(i,e) and j < e}` -- the position a STABLE ascending sort of
`d(i, .)` keyed by the Float32 bits gives column e (equal keys keep
column order, `cub::DeviceSegmentedRadixSort` is stable), so it is the
SAME INTEGER as their lookup, with O(n k) work per row instead of a sort
and a materialized lookup table. The distance `d(i, .)` is
`metrics/original/pinned_distance.mojo`'s one-thread unexpanded L2
(exactly `L2SqrtUnexpanded`, through `identical_mul_add`/`ftz`/
`identical_sqrt`), so the comparison compares the same bits on every
vendor. (2) THEIRS calls `brute_force_knn` on X_embedded with
`L2SqrtUnexpanded`; OURS calls `neighbors/estimator.mojo::knn_search`,
the repository's one k-NN entry (cuVS's fused EXPANDED L2 arm under
FAST, the pinned expanded tile under IDENTICAL, IDENTITY_PATHS row 24),
with `k = n_neighbors + 1`. Expanded and unexpanded distances can order
two NEAR-TIED embedded neighbors differently in the last bit; the rank
sum then moves by an integer. Measured on the hashed fixture the neighbor
sets agree with a Float64 host k-NN on every row (trustworthiness_check.
mojo reports the count); a second k-NN kernel for the unexpanded metric
would be neighbors/'s to add and is named in the README's hand-off. The
rank sum lands as one Int32 partial per row (<= (k+1) n), summed on the
host in Int64 (DEVIATION 652's reason: no 64-bit atomic on Apple).
`n_neighbors + 1 > TRUST_MAX_K` is refused by name (a threadgroup slab
holds the k+1 neighbor distances).
"""

from std.atomic import Atomic
from std.gpu import thread_idx
from std.math import ceildiv
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from core.identity_trace import IdentityTrace
from metrics.original.pinned_distance import l2sqrt_unexpanded
from metrics.original.pinned_sum import (
    PINNED_SUM_TPB,
    linear_block_id,
    physical_block_count,
)
from neighbors.estimator import knn_search_traced


#: The threadgroup slab for the k+1 embedded-neighbor distances and ranks.
comptime TRUST_MAX_K = 256


def trust_rank_kernel[
    block_size: Int
](
    x: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    m_in: Int32,
    emb_ind: MutPointer[UInt32, MutAnyOrigin],
    k_plus_1_in: Int32,
    row_partials: MutPointer[Int32, MutAnyOrigin],
):
    """`build_lookup_table` + `compute_rank` for one row per block, by
    counting (DEVIATION 655). `row_partials[i] = sum over the row's k+1
    embedded neighbors of max(0, rank - n_neighbors)`."""
    var tid = Int(thread_idx.x)
    var n = Int(n_in)
    var m = Int(m_in)
    var k1 = Int(k_plus_1_in)
    var d_e = stack_allocation[
        TRUST_MAX_K, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var e_idx = stack_allocation[
        TRUST_MAX_K, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var counts = stack_allocation[
        TRUST_MAX_K, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var row = linear_block_id()
    while row < n:
        # the k+1 embedded neighbors of `row` and their distances in X
        var e = tid
        while e < k1:
            var idx = Int(emb_ind.unsafe_load(row * k1 + e))
            e_idx[unsafe_offset = e] = Int32(idx)
            d_e[unsafe_offset = e] = l2sqrt_unexpanded(x, row, idx, m)
            counts[unsafe_offset = e] = Int32(0)
            e += block_size
        barrier()
        # every thread walks j = tid, tid + block_size, ... and counts
        var j = tid
        while j < n:
            var dj = l2sqrt_unexpanded(x, row, j, m)
            for q in range(k1):
                var de = d_e[unsafe_offset = q]
                var ei = Int(e_idx[unsafe_offset = q])
                # IDENTITY_PATHS row 39: a positional compare on distances
                # that are `>= +0.0`, never `-0.0`, never NaN on a finite X
                # (pinned_distance.mojo's header); `+inf` ties with `+inf`
                # and the index decides, as in their stable sort.
                if dj < de or (dj == de and j < ei):
                    _ = Atomic.fetch_add(counts.unsafe_offset(q), Int32(1))
            j += block_size
        barrier()
        if tid == 0:
            var total = Int32(0)
            for q in range(k1):
                # `tmp = r - n_neighbors + 1` with their n_neighbors = k+1
                var tmp = counts[unsafe_offset = q] - Int32(k1) + Int32(1)
                if tmp > Int32(0):
                    total += tmp
            row_partials.unsafe_store(row, total)
        barrier()
        row += physical_block_count()


def trustworthiness_rank_sum(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut x_host: List[Float32],
    mut x_embedded_host: List[Float32],
    n: Int,
    m: Int,
    d: Int,
    n_neighbors: Int,
    block_size_is_64: Bool,
    grid_x_override: Int,
) raises -> Tuple[Int64, List[UInt32]]:
    """The integer half: `(rank sum, emb_ind)`; exposed so the check can
    gate it EXACTLY and see the neighbor structure it rests on. `trace`
    is handed to `knn_search_traced` so the k-NN's `knn.*` stages land in
    the SAME card as the metric's (one `seq` per file; the DEVIATION 518
    lesson), then `trust.emb_ind` and `trust.rank_sum` are recorded."""
    if n_neighbors < 1:
        raise Error("trustworthiness: n_neighbors must be >= 1")
    # cuML's own surface (`python/cuml/cuml/metrics/trustworthiness.pyx:
    # 114`): `if n_neighbors < 1 or 2 * n_neighbors >= n_samples: raise
    # ValueError`. Mirrored here so the closed form's denominator `(n k)
    # (2n - 3k - 1)` is always > 0 (with 2k < n and n >= 3 it is >= n k /
    # 2): no `2 / 0`, no `0 * inf`, no NaN reaches the recorded scalar
    # (IDENTITY_PATHS row 39). This subsumes the older `n_neighbors + 1 >
    # n` refusal.
    if 2 * n_neighbors >= n:
        raise Error(
            "trustworthiness: n_neighbors ("
            + String(n_neighbors)
            + ") must be >= 1 and < n_samples / 2; n_samples is "
            + String(n)
            + " (cuML trustworthiness.pyx:114)"
        )
    if n_neighbors + 1 > TRUST_MAX_K:
        raise Error(
            "trustworthiness: n_neighbors + 1 = "
            + String(n_neighbors + 1)
            + " exceeds TRUST_MAX_K = "
            + String(TRUST_MAX_K)
            + " (refused by name)"
        )
    var k1 = n_neighbors + 1
    # run_knn (:81-108): brute_force_knn(X_embedded, X_embedded, k+1).
    # knn_search's boundary is MutUntrackedOrigin host pointers, which a
    # host buffer provides (neighbors/original/estimator_check.mojo does
    # the same).
    var h_emb = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var h_dist = ctx.enqueue_create_host_buffer[DType.float32](n * k1)
    var h_idx = ctx.enqueue_create_host_buffer[DType.uint32](n * k1)
    ctx.synchronize()
    for i in range(n * d):
        h_emb.unsafe_ptr().unsafe_store(i, x_embedded_host[i])
    _ = knn_search_traced(
        ctx,
        trace,
        h_emb.unsafe_ptr(),
        n,
        h_emb.unsafe_ptr(),
        n,
        d,
        k1,
        h_dist.unsafe_ptr(),
        h_idx.unsafe_ptr(),
        True,
    )
    var out_idx = List[UInt32]()
    for i in range(n * k1):
        out_idx.append(h_idx.unsafe_ptr().unsafe_load(i))
    var x_dev = ctx.enqueue_create_buffer[DType.float32](n * m)
    ctx.enqueue_copy(dst_buf=x_dev, src_ptr=x_host.unsafe_ptr())
    var emb = ctx.enqueue_create_buffer[DType.uint32](n * k1)
    ctx.enqueue_copy(dst_buf=emb, src_ptr=out_idx.unsafe_ptr())
    var partials = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()
    var gx = n if grid_x_override <= 0 else grid_x_override
    var gy = ceildiv(n, gx)
    if block_size_is_64:
        ctx.enqueue_function[trust_rank_kernel[64]](
            x_dev.unsafe_ptr(), Int32(n), Int32(m), emb.unsafe_ptr(),
            Int32(k1), partials.unsafe_ptr(),
            grid_dim=(gx, gy, 1), block_dim=(64, 1, 1),
        )
    else:
        ctx.enqueue_function[trust_rank_kernel[PINNED_SUM_TPB]](
            x_dev.unsafe_ptr(), Int32(n), Int32(m), emb.unsafe_ptr(),
            Int32(k1), partials.unsafe_ptr(),
            grid_dim=(gx, gy, 1), block_dim=(PINNED_SUM_TPB, 1, 1),
        )
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=partials)
    ctx.synchronize()
    var total = Int64(0)
    for i in range(n):
        total += Int64(h.unsafe_ptr().unsafe_load(i))
    if trace.enabled:
        var tmp_idx = out_idx.copy()
        trace.record_host("trust.emb_ind", tmp_idx.unsafe_ptr(), n * k1)
        _ = tmp_idx^
        var one = List[Int64]()
        one.append(total)
        trace.record_host("trust.rank_sum", one.unsafe_ptr(), 1)
        _ = one^
    _ = h^
    _ = partials^
    _ = emb^
    _ = x_dev^
    _ = h_emb^
    _ = h_dist^
    _ = h_idx^
    return (total, out_idx^)


def trustworthiness_score(
    ctx: DeviceContext,
    mut x_host: List[Float32],
    mut x_embedded_host: List[Float32],
    n: Int,
    m: Int,
    d: Int,
    n_neighbors: Int,
    batch_size: Int = 512,
) raises -> Float64:
    """`trustworthiness_score<math_t, L2SqrtUnexpanded>(h, X, X_embedded, n,
    m, d, n_neighbors, batchSize)` (:122-207), with a trace read from the
    environment (`ols_fit` / `ols_fit_traced`'s split, DEVIATION 517)."""
    var trace = IdentityTrace()
    return trustworthiness_score_traced(
        ctx, trace, x_host, x_embedded_host, n, m, d, n_neighbors, batch_size
    )


def trustworthiness_score_traced(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut x_host: List[Float32],
    mut x_embedded_host: List[Float32],
    n: Int,
    m: Int,
    d: Int,
    n_neighbors: Int,
    batch_size: Int = 512,
) raises -> Float64:
    """The traced form: `knn.*` (neighbors'), `trust.emb_ind`, `trust.
    rank_sum` land in `trace`. `batchSize` is their memory tiling of the
    X distance rows and is validated (>= 1) and otherwise scheduling here
    (no tile is materialized)."""
    if batch_size < 1:
        raise Error("trustworthiness: batchSize must be >= 1")
    var r = trustworthiness_rank_sum(
        ctx, trace, x_host, x_embedded_host, n, m, d, n_neighbors, False, 0
    )
    var t = Float64(r[0])
    # (:204), in double
    var nn = Float64(n)
    var kk = Float64(n_neighbors)
    return 1.0 - ((2.0 / ((nn * kk) * ((2.0 * nn) - (3.0 * kk) - 1.0))) * t)
