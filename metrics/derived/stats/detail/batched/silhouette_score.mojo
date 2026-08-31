# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RAFT `cpp/include/raft/stats/detail/batched/silhouette_score.cuh`
(ebf9268; cuVS `cpp/src/stats/detail/batched/silhouette_score.cuh` is the
same text and is what cuML 26.08 links). THIS IS THE PATH cuML's Python
takes: `silhouette_score.pyx::_silhouette_coeff` calls the batched entry
with `chunksize = 40000` for every `silhouette_score` / `silhouette_samples`
call (PORTING_RULES 0b-i), with `metric = L2SqrtUnexpanded` for the default
`'euclidean'`.

THEIRS (:167-266):

    ASSERT(n_labels >= 2 && n_labels <= n_rows - 1)
    cluster_counts = countLabels(y)                                    int[n_labels]
    a = 0 (n_rows);  b = fill_b_kernel (n_rows x n_labels):             (:35-64)
        b[i, c] = (c == y[i] || counts[c] == 0) ? (counts[y[i]] == 1 ? 0 : MAX) : 0
    for every (chunk_i, chunk_j) tile of rows:
        distances = pairwise_distance(X[chunk_i], X[chunk_j], metric)   cuVS, L2SqrtUnexpanded
        compute_chunked_a_b_kernel (:74-107), one thread per (i, j), i != j:
            if counts[y[i]] == 1: return
            if y[j] == y[i]: atomicAdd(&a[i],            d(i,j) / (counts[y[j]] - 1))
            else:            atomicAdd(&b[i * L + y[j]], d(i,j) /  counts[y[j]])
    b[i] = reduce<min_op>(b[i, :], init MAX)                           (:248-260)
    a[i] = SilOp(a[i], b[i])                                           (:262-263)
    return thrust::reduce(a) / n_rows                                  (:265)

sklearn `silhouette_samples`: a(i) = mean distance to the other members of
its cluster, b(i) = min over other clusters of the mean distance, s =
(b - a) / max(a, b), singleton clusters score 0; `silhouette_score` is the
mean. The same numbers (RAFT divides each term before summing; sklearn
sums then divides -- last bits differ).

=========================================================================
DEVIATION 654 (metrics lane, 2026-08-23): THE FLOAT `atomicAdd` INTO a AND
b IS REPLACED BY ONE FIXED TREE PER (ROW, CLUSTER), AND THE DISTANCE TILE
BY THE ONE-THREAD UNEXPANDED FORMULA.
=========================================================================
THEIRS lands `d(i,j)/count` into `a[i]` and `b[i, c]` by float atomicAdd
from every (i, j) thread of every tile: an ARRIVAL order, a different sum
run to run and vendor to vendor, and `chunksize` changes which threads
race. OURS is row-owned: one block per row `i` walks `j` ascending in
`PINNED_SUM_W` chunks, each thread computes `d(i,j)` through
`metrics/original/pinned_distance.mojo` (one thread per cell, ascending
features, `identical_mul_add`, `ftz`, `identical_sqrt`) and, per cluster
`c`, the terms `d/denom_c` (0 where `y[j] != c` or `j == i`) are folded by
`virtual_block_sum` (DEVIATION 653's slab tree, independent of the
block) and the chunk totals are added ascending by thread 0 into `a` (for
`c == y[i]`) or `b[i, c]`. The sum for (i, c) is therefore a pure
function of `n_rows` and of which rows carry label c -- "n and the
cluster sizes" -- never of the grid, the block, or `chunksize`. The min
over clusters and `SilOp` run in the same thread (a selection and an
elementwise op: no numeric change from RAFT's separate `reduce` and
`binaryOp`), the mean is DEVIATION 653's tree over the scores. Under FAST
the per-chunk fold is `block.sum` and the distance helpers are the naive
chain; FAST is a report.

THE MIN OVER CLUSTERS has no tie-break to state for the VALUE (a min is
a selection; two equal b's give the same b) and the `+0.0/-0.0` and NaN
hazards are absent (silhouette_score.mojo's header; IDENTITY_PATHS row
39); it is written ascending with first-index-wins, a strict `<` compare
and never a hardware `min`, so a future per-sample `argmin` output would
be pinned too. The `SilOp` that follows carries DEVIATION 656 (a NaN
quotient from an `inf` distance is +0.0, sklearn's `nan_to_num`).

`chunk` (cuML's `chunksize`) is ACCEPTED and validated (`1 <= chunk`) and
changes no bit: ours materializes no distance tile, so there is nothing
for it to size; it is recorded in NOT_IMPLEMENTED.tsv as honored-as-scheduling.
`metric` other than `L2SqrtUnexpanded` (cuML 'euclidean' / 'l2') is
REFUSED BY NAME.
"""

from std.gpu import thread_idx
from std.math import ceildiv
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.original.pinned_distance import l2sqrt_unexpanded
from metrics.original.pinned_sum import (
    PINNED_SUM_TPB,
    PINNED_SUM_W,
    chunk_count,
    host_fold_partials,
    linear_block_id,
    physical_block_count,
    virtual_block_sum,
)
from metrics.derived.stats.detail.scores import sum_chunks_kernel
from metrics.derived.stats.detail.silhouette_score import count_labels, sil_op
from original.numerics import ftz


#: `ML::distance::DistanceType` (cuml/common/distance_type.hpp): the one
#: arm cuML's default takes. Every other value is refused by name.
comptime DISTANCE_L2_SQRT_UNEXPANDED = 5

#: `std::numeric_limits<value_t>::max()` for Float32 (:59, :253).
comptime FLOAT32_MAX_BITS: UInt32 = 0x7F7FFFFF


@always_inline
def _float32_max() -> Float32:
    return bitcast[DType.float32](FLOAT32_MAX_BITS)


def silhouette_rows_kernel[
    block_size: Int
](
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Int32, MutAnyOrigin],
    cluster_counts: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    n_labels_in: Int32,
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    scores: MutPointer[Float32, MutAnyOrigin],
):
    """`fill_b_kernel` + `compute_chunked_a_b_kernel` + the min `reduce` +
    the `SilOp` `binaryOp`, row-owned (DEVIATION 654). One block serves
    rows `linear_block_id(), += physical_block_count()`."""
    comptime R = PINNED_SUM_W // block_size
    var tid = Int(thread_idx.x)
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var n_labels = Int(n_labels_in)
    var chunks = chunk_count(n_rows)
    var row = linear_block_id()
    while row < n_rows:
        var row_cluster = Int(y.unsafe_load(row))
        var row_count = Int(cluster_counts.unsafe_load(row_cluster))
        var singleton = row_count == 1
        # fill_b_kernel (:35-64) and `thrust::fill(a, 0)` (:202), this row
        if tid == 0:
            a.unsafe_store(row, Float32(0.0))
            for c in range(n_labels):
                var cc = Int(cluster_counts.unsafe_load(c))
                var v = Float32(0.0)
                if c == row_cluster or cc == 0:
                    v = Float32(0.0) if singleton else _float32_max()
                b.unsafe_store(row * n_labels + c, v)
        # compute_chunked_a_b_kernel (:74-107): `if counts[y[i]] == 1 return`
        if not singleton:
            for chunk in range(chunks):
                var d = SIMD[DType.float32, R](0.0)
                var lab = SIMD[DType.int32, R](-1)
                comptime for r in range(R):
                    var j = chunk * PINNED_SUM_W + tid + r * block_size
                    if j < n_rows and j != row:
                        d[r] = l2sqrt_unexpanded(x, row, j, n_cols)
                        lab[r] = y.unsafe_load(j)
                for c in range(n_labels):
                    var cc = Int(cluster_counts.unsafe_load(c))
                    # `/ (col_cluster_counts - 1)` for the own cluster (:102),
                    # `/ col_cluster_counts` otherwise (:105)
                    var denom = Float32(cc - 1) if c == row_cluster else Float32(cc)
                    var vals = SIMD[DType.float32, R](0.0)
                    comptime for r in range(R):
                        if lab[r] == Int32(c):
                            vals[r] = ftz(d[r] / denom)
                    var part = virtual_block_sum[block_size](vals)
                    if tid == 0:
                        if c == row_cluster:
                            a.unsafe_store(row, ftz(a.unsafe_load(row) + ftz(part)))
                        else:
                            var idx = row * n_labels + c
                            b.unsafe_store(idx, ftz(b.unsafe_load(idx) + ftz(part)))
        # reduce<min_op>(b[i, :], init MAX) (:248-260), then SilOp (:262-263)
        # IDENTITY_PATHS row 39: a POSITIONAL fold (strict `<`, ascending c,
        # first index wins), never a hardware `min`. Its candidates are
        # `FLT_MAX` (own / empty slot), `+0.0`, or a `+0.0`-seeded sum of
        # nonnegative terms: no `-0.0` (x + y is -0.0 only when both are),
        # no NaN (no `inf - inf` in an all-nonnegative sum), so the winner's
        # VALUE is the same whichever vendor runs it. silhouette_score.mojo's
        # header carries the proof.
        if tid == 0:
            var bmin = _float32_max()
            for c in range(n_labels):
                var v = b.unsafe_load(row * n_labels + c)
                if v < bmin:
                    bmin = v
            scores.unsafe_store(row, sil_op(a.unsafe_load(row), bmin))
        row += physical_block_count()


def silhouette_score(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut y: DeviceBuffer[DType.int32],
    n_labels: Int,
    mut scores: DeviceBuffer[DType.float32],
    chunk: Int,
    metric: Int,
) raises -> Float32:
    """`batched::detail::silhouette_score(X, n_rows, n_cols, y, n_labels,
    scores, chunk, metric)` (:167-266) at the default launch. `scores`
    receives the per-sample coefficients (cuML's `silhouette_samples`)."""
    return silhouette_score_launch[PINNED_SUM_TPB](
        ctx, x, n_rows, n_cols, y, n_labels, scores, chunk, metric, 0
    )


def silhouette_score_launch[
    block_size: Int
](
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut y: DeviceBuffer[DType.int32],
    n_labels: Int,
    mut scores: DeviceBuffer[DType.float32],
    chunk: Int,
    metric: Int,
    grid_x_override: Int,
) raises -> Float32:
    """The launch-parameterized form for the invariance gate (block size,
    and a 2-D grid of `grid_x_override` columns when nonzero)."""
    # ASSERT(n_labels >= 2 && n_labels <= (n_rows - 1)) (:178-179)
    if not (n_labels >= 2 and n_labels <= n_rows - 1):
        raise Error(
            "silhouette_score: silhouette Score not defined for the given"
            " number of labels (n_labels="
            + String(n_labels)
            + ", n_rows="
            + String(n_rows)
            + ")"
        )
    if n_cols <= 0:
        raise Error("silhouette_score: n_cols must be positive")
    if metric != DISTANCE_L2_SQRT_UNEXPANDED:
        raise Error(
            "silhouette_score: metric "
            + String(metric)
            + " is refused; only DistanceType::L2SqrtUnexpanded (5, cuML"
            " 'euclidean'/'l2') is ported (NOT_IMPLEMENTED.tsv)"
        )
    if chunk < 1:
        raise Error(
            "silhouette_score: chunk (cuML chunksize) must be >= 1, got "
            + String(chunk)
        )
    if len(scores) < n_rows:
        raise Error("silhouette_score: scores holds fewer than n_rows floats")
    # get_cluster_counts (:109-124) -> countLabels
    var cluster_counts = ctx.enqueue_create_buffer[DType.int32](n_labels)
    count_labels(ctx, y, cluster_counts, n_rows, n_labels)
    var a = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var b = ctx.enqueue_create_buffer[DType.float32](n_rows * n_labels)
    var gx = n_rows if grid_x_override <= 0 else grid_x_override
    var gy = ceildiv(n_rows, gx)
    ctx.enqueue_function[silhouette_rows_kernel[block_size]](
        x.unsafe_ptr(),
        y.unsafe_ptr(),
        cluster_counts.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        Int32(n_labels),
        a.unsafe_ptr(),
        b.unsafe_ptr(),
        scores.unsafe_ptr(),
        grid_dim=(gx, gy, 1),
        block_dim=(block_size, 1, 1),
    )
    ctx.synchronize()
    # `thrust::reduce(a_ptr, a_ptr + n_rows, 0) / n_rows` (:265): the
    # DEVIATION 653 tree over the scores, then one division.
    var chunks = chunk_count(n_rows)
    var sgx = chunks if grid_x_override <= 0 else grid_x_override
    var sgy = ceildiv(chunks, sgx)
    var partials = ctx.enqueue_create_buffer[DType.float32](chunks)
    ctx.enqueue_function[sum_chunks_kernel[block_size]](
        scores.unsafe_ptr(),
        Int32(n_rows),
        partials.unsafe_ptr(),
        grid_dim=(sgx, sgy, 1),
        block_dim=(block_size, 1, 1),
    )
    var h = ctx.enqueue_create_host_buffer[DType.float32](chunks)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=partials)
    ctx.synchronize()
    var lst = List[Float32]()
    for c in range(chunks):
        lst.append(h.unsafe_ptr().unsafe_load(c))
    var total = host_fold_partials(lst, chunks)
    _ = h^
    _ = partials^
    _ = a^
    _ = b^
    _ = cluster_counts^
    return ftz(total / Float32(n_rows))
