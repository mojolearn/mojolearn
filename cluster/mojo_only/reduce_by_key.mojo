"""Weighted sums per cluster, which is the whole update half of Lloyd.

NOT A PORT of cuVS. Their two calls are
`raft::linalg::reduce_rows_by_key` and `raft::linalg::reduce_cols_by_key`
(`detail/kmeans_common.cuh::compute_centroid_adjustments`), and RAFT is a
separate library this tree does not mirror file for file. The CALL SITES,
their order, and their accumulate-versus-reset semantics are theirs and are
copied in `cluster/ported/cluster/detail/kmeans_common.mojo`.

THE ACCUMULATOR IS THE INTERESTING PART, AND IT IS NOT A NEW PROBLEM
--------------------------------------------------------------------
Summing rows into their cluster is a scatter-add: many rows land on one
centroid and the order they arrive in is not fixed. RAFT does it with float
arithmetic on NVIDIA. Metal has no float atomic add, which is the exact wall
`PORTING.md 7` describes for the histogram flush, so this file reuses the
answer already built and verified for that: **`mojo_only/fixed_point.mojo`**.

**The overflow bound transfers word for word, with one noun changed.** That
file's argument is "any leaf's rows are a subset of all rows, so a scale that
keeps the global sum of magnitudes inside `Int32` keeps every partial sum
inside `Int32`". Here it is "any CLUSTER's rows are a subset of all rows",
and the rest is identical. One host pass over the weighted data per fit
yields a scale that makes overflow impossible for every cluster at every
iteration.

Two things follow, and the second is why this file was worth writing before
any optimization:

1. Cluster sums are order-independent, so two runs agree, and Metal, CUDA and
   HIP agree. The centroid update is exactly reproducible, for free, at the
   cost of one quantization per element.
2. **This answers the question `PLAN.md` said k-means existed to answer.**
   The shared substrate is genuinely shared: the fixed-point accumulator was
   built for histograms and needed no change to serve a clustering algorithm
   that has no histogram in it. `core/` is not secretly tree-shaped, and the
   evidence is that this file imports it and adds nothing.

It also gives `mojo_only/fixed_point.mojo` its first reader. `UNWIRED.md`
listed it as verified in isolation and used by no kernel.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation


comptime REDUCE_BY_KEY_TPB = 128


def accumulate_centroid_sums_kernel(
    sums_i32: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    labels: MutPointer[UInt32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_features_in: Int32,
    scale_in: Float32,
):
    """`reduce_rows_by_key`, as a quantized scatter-add.

    Grid x strides rows, threads stride features, so consecutive threads
    write consecutive features of one cluster and the scatter is coalesced
    within a row.
    """
    var n_rows = Int(n_rows_in)
    var n_features = Int(n_features_in)

    var row = Int(block_idx.x)
    while row < n_rows:
        var label = Int(labels.unsafe_load(row))
        var w = weights.unsafe_load(row)
        var base_in = row * n_features
        var base_out = label * n_features

        var f = Int(thread_idx.x)
        while f < n_features:
            var q = Int32(x.unsafe_load(base_in + f) * w * scale_in)
            _ = Atomic.fetch_add(sums_i32.unsafe_offset(base_out + f), q)
            f += REDUCE_BY_KEY_TPB

        row += Int(grid_dim.x)


def accumulate_weight_per_cluster_kernel(
    weight_i32: MutPointer[Int32, MutAnyOrigin],
    labels: MutPointer[UInt32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    scale_in: Float32,
):
    """`reduce_cols_by_key` over a single row of weights.

    Their call passes `n_rows = 1` and the weight vector as the matrix
    (`kmeans_common.cuh`, the `reduce_cols_by_key` call), so this is the
    denominator of the centroid update and the empty-cluster test in one
    array.
    """
    var n_rows = Int(n_rows_in)
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row < n_rows:
        var label = Int(labels.unsafe_load(row))
        var q = Int32(weights.unsafe_load(row) * scale_in)
        _ = Atomic.fetch_add(weight_i32.unsafe_offset(label), q)


def finalize_centroids_kernel(
    new_centroids: MutPointer[Float32, MutAnyOrigin],
    old_centroids: MutPointer[Float32, MutAnyOrigin],
    sums_i32: MutPointer[Int32, MutAnyOrigin],
    weight_i32: MutPointer[Int32, MutAnyOrigin],
    n_clusters_in: Int32,
    n_features_in: Int32,
    sum_scale_in: Float32,
    weight_scale_in: Float32,
):
    """`finalize_centroids` (`kmeans_common.cuh`), both arms.

    Theirs is a `div_checkzero_op` followed by a `gather_if` that copies the
    OLD centroid back wherever the cluster weight is zero. Fused here into
    one pass because the two are a single branch per cell and their split
    exists only because RAFT has no primitive for the pair.

    **Keeping the old centroid for an empty cluster is a real decision, not a
    fallback.** scikit-learn instead relocates an empty cluster onto the
    point furthest from its centroid. Two implementations that disagree here
    diverge permanently on the same seed, so a validation harness has to
    compare inertia and not centroid identity. See `cluster/tools/
    sklearn_reference.py`.
    """
    var n_clusters = Int(n_clusters_in)
    var n_features = Int(n_features_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < n_clusters * n_features:
        var cluster = idx // n_features
        var w = Float32(weight_i32.unsafe_load(cluster)) / weight_scale_in
        if w == Float32(0.0):
            new_centroids.unsafe_store(idx, old_centroids.unsafe_load(idx))
        else:
            var s = Float32(sums_i32.unsafe_load(idx)) / sum_scale_in
            new_centroids.unsafe_store(idx, s / w)


def centroid_shift_kernel(
    out_partial: MutPointer[Float32, MutAnyOrigin],
    old_centroids: MutPointer[Float32, MutAnyOrigin],
    new_centroids: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`compute_centroid_shift` (`kmeans_common.cuh`), `sum((old - new)^2)`.

    One block, one partial per block; the host takes the total over at most a
    handful of blocks, the same shape `compute_scores.mojo` uses for its
    argmax. `n_clusters * n_features` is small, so this is not on any hot
    path and does not deserve a two-stage reduction.
    """
    var n = Int(n_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < n:
        var d = old_centroids.unsafe_load(idx) - new_centroids.unsafe_load(idx)
        out_partial.unsafe_store(idx, d * d)


def sum_partials_kernel(
    out_partial: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    use_b_in: Int32,
):
    """One partial sum per block, host adds the handful of partials.

    Serves two call sites in the Lloyd loop: the weighted clustering cost
    (`a` = per-row distance, `b` = per-row weight) and the centroid shift
    (`a` = squared differences, `use_b` = 0).

    **This is a float sum and therefore NUMERIC.** The block count changes
    the order and moves the last bits of inertia, which is exactly the trap
    `mojo_only/numerics.mojo` is about: a grid size looks like scheduling and
    is not. It does not affect WHICH cluster a point joins, only the reported
    cost and the convergence ratio, so a fit can converge one iteration
    earlier or later on a different backend while producing the same
    assignment. Say that plainly rather than claiming determinism this path
    does not have.
    """
    var n = Int(n_in)
    var tid = Int(thread_idx.x)

    var acc = Float32(0.0)
    var i = Int(block_idx.x) * REDUCE_BY_KEY_TPB + tid
    var stride = Int(grid_dim.x) * REDUCE_BY_KEY_TPB
    while i < n:
        var v = a.unsafe_load(i)
        if use_b_in != 0:
            v = v * b.unsafe_load(i)
        acc += v
        i += stride

    var s = stack_allocation[
        REDUCE_BY_KEY_TPB,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    s[tid] = acc
    barrier()

    var half = REDUCE_BY_KEY_TPB // 2
    while half > 0:
        if tid < half:
            s[tid] = s[tid] + s[tid + half]
        barrier()
        half //= 2

    if tid == 0:
        out_partial.unsafe_store(Int(block_idx.x), s[0])


def zero_i32_kernel(
    a: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`raft::matrix::fill(0)` on the two accumulators, every iteration."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < Int(n_in):
        a.unsafe_store(idx, Int32(0))


def copy_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """Device to device copy of the centroid buffer.

    Exists because `std::swap(cur_centroids_ptr, new_centroids_ptr)`
    (`detail/kmeans.cuh:907`) has no counterpart here: theirs swaps two raw
    pointers on the host and ours are `DeviceBuffer` values. A kernel is the
    honest way to move 'k x d' floats, and 'k x d' is small.

    It is also the second time this tree has needed one: see `UNWIRED.md` on
    `enqueue_copy(dst_buf=, src_ptr=device)` being a silent no-op.
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < Int(n_in):
        dst.unsafe_store(idx, src.unsafe_load(idx))
