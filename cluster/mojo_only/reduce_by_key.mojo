"""Weighted sums per cluster, which is the whole update half of Lloyd.

NOT A PORT of cuVS. Their two calls are
`raft::linalg::reduce_rows_by_key` and `raft::linalg::reduce_cols_by_key`
(`raft/linalg/reduce_rows_by_key.cuh` and `reduce_cols_by_key.cuh`, called
from `update_centroids`, `detail/kmeans.cuh:300-318`), and RAFT is a
separate library this tree does not mirror file for file. The CALL SITES,
their order, and their accumulate-versus-reset semantics are theirs and are
copied in `cluster/ported/cluster/detail/kmeans_common.mojo`.

THE ACCUMULATOR IS THE INTERESTING PART, AND IT IS NOT A NEW PROBLEM
--------------------------------------------------------------------
Summing rows into their cluster is a scatter-add: many rows land on one
centroid and the order they arrive in is not fixed. RAFT does it with float
atomics on NVIDIA. This file quantizes into `Int32` through
**`mojo_only/fixed_point.mojo`** instead, and the reason is REPRODUCIBILITY,
not capability.

**A float `atomicAdd` does work on Metal.** Probed on this M4: 1024 threads
each adding 1.0 return exactly 1024.0. Earlier notes here, and the ones that
described this as a hardware wall, were wrong, and a design resting on a false
constraint has to be re-argued rather than annotated. The argument that
survives is point 1 below: float atomics sum in whatever order the hardware
delivers, so the same fit gives different last bits run to run and device to
device, while an integer accumulator is exactly order-independent. That is
worth one quantization per element here.

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

from mojo_only.kernel_matrix import (
    K_LIB_REDUCE_BY_KEY,
    TARGET_COLUMN,
    lib_block_size_for,
)


from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier
from std.memory import stack_allocation


# READ FROM THE MATRIX, not restated here. `mojo_only/kernel_matrix.mojo`
# owns every tunable in this tree; changing TARGET_COLUMN there rebuilds
# this kernel for another vendor with no edit in this file.
comptime REDUCE_BY_KEY_TPB = lib_block_size_for[
    K_LIB_REDUCE_BY_KEY, TARGET_COLUMN
]()


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

    **The atomic scatter-add itself is faithful, not a substitution gap.**
    RAFT's `reduce_rows_by_key.cuh:300` also lands its contributions with a
    global atomic add rather than a vendor segmented reduce, so there is
    nothing here to swap `cub::DeviceSegmentedReduce` into (and that is
    NOT FOUND anyway; see VENDOR_LIBRARIES.md). The only thing we change is
    the accumulator TYPE, and the module docstring says why: not because
    Metal lacks a float atomic, which it does not, but because an integer
    accumulator is order-independent and a float one is not.

    The indexing is theirs too: a FLAT grid-stride over `n_rows * n_features`
    cells, matching their `gid = threadIdx.x + blockDim.x * blockIdx.x`
    (`reduce_rows_by_key.cuh:292`). The earlier shape here strided rows on
    grid x and features on the threads, which idles `REDUCE_BY_KEY_TPB -
    n_features` threads of every block whenever `n_features < 128` and reads
    `x` in short runs. Flat indexing keeps all 128 threads on 128 consecutive
    elements of `x`.

    That change cannot move a number: every `(row, feature)` cell is still
    visited exactly once, and the accumulator is fixed point, so the sum is
    order-independent by construction. Bit-identical, which is the only
    reason it was safe to make without a measurement.
    """
    var n_rows = Int(n_rows_in)
    var n_features = Int(n_features_in)
    var total = n_rows * n_features

    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while gid < total:
        var row = gid // n_features
        var f = gid - row * n_features

        var label = Int(labels.unsafe_load(row))
        var w = weights.unsafe_load(row)
        var q = Int32(x.unsafe_load(gid) * w * scale_in)
        _ = Atomic.fetch_add(sums_i32.unsafe_offset(label * n_features + f), q)

        gid += stride


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


comptime SUM_MODE_PLAIN = 0
comptime SUM_MODE_PRODUCT = 1
comptime SUM_MODE_SQDIFF = 2


def sum_partials_kernel(
    out_partial: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    mode_in: Int32,
):
    """One partial sum per block; a second kernel folds the partials.

    The MAP IS INSIDE THE REDUCTION, which is theirs. Both call sites in the
    Lloyd loop are a `map`-then-`sum` in cuVS and neither materializes the
    mapped array:

    - the centroid shift is `raft::linalg::mapThenSumReduce(sqrdNorm, ...,
      raft::sqdiff_op{}, ..., centroids, newCentroids)`
      (`detail/kmeans.cuh:453-459`) -> `SUM_MODE_SQDIFF`;
    - the weighted cluster cost is `computeClusterCost(..., raft::value_op{},
      raft::add_op{})` over `minClusterAndDistance` whose `.value` was already
      multiplied by the sample weight (`:516-535`) -> `SUM_MODE_PRODUCT`.

    An earlier version of this port wrote the squared differences to a
    `n_clusters * n_features` scratch buffer with a separate
    `centroid_shift_kernel` and then summed it. That kernel is gone: it was a
    launch cuVS does not have.

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
    var mode = Int(mode_in)

    var acc = Float32(0.0)
    var i = Int(block_idx.x) * REDUCE_BY_KEY_TPB + tid
    var stride = Int(grid_dim.x) * REDUCE_BY_KEY_TPB
    while i < n:
        var v = a.unsafe_load(i)
        if mode == SUM_MODE_PRODUCT:
            v = v * b.unsafe_load(i)
        elif mode == SUM_MODE_SQDIFF:
            var d = v - b.unsafe_load(i)
            v = d * d
        acc += v
        i += stride

    # `cub::BlockReduce`'s counterpart from
    # `max.gpu.primitives.block`. The hand-written shared-memory tree
    # reduction this replaced is gone: same arithmetic, one call, and
    # the reduction shape is Modular's to tune rather than ours to
    # guess. See VENDOR_LIBRARIES.md.
    var s0 = block_sum[block_size=REDUCE_BY_KEY_TPB](acc)
    if tid == 0:
        out_partial.unsafe_store(Int(block_idx.x), s0)


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

    This is theirs: cuVS moves the freshly finalized centroids back over the
    working set with a device-to-device `raft::copy`
    (`detail/kmeans.cuh:465-466`), after the shift has been measured between
    the two buffers and before the next iteration reads them. It is a copy in
    their code too, not a pointer swap. A kernel is how a `DeviceBuffer` to
    `DeviceBuffer` copy is spelled here, and 'k x d' is small.

    It is also the second time this tree has needed one: see `UNWIRED.md` on
    `enqueue_copy(dst_buf=, src_ptr=device)` being a silent no-op.
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < Int(n_in):
        dst.unsafe_store(idx, src.unsafe_load(idx))


def finish_sum_kernel(
    out_scalar: MutPointer[Float32, MutAnyOrigin],
    partials: MutPointer[Float32, MutAnyOrigin],
    n_blocks_in: Int32,
):
    """Second stage: fold the block partials into ONE device scalar.

    Exists so the Lloyd loop never has to bring a sum to the host. The first
    version of this port summed the partials in a host loop, which cost a
    drain and a transfer per iteration for a number the host only needed in
    order to make a decision the DEVICE can make. See
    `HOST_AND_DEVICE.md`.

    One block, because `n_blocks` is at most 256 by construction.
    """
    var n_blocks = Int(n_blocks_in)
    var tid = Int(thread_idx.x)

    var acc = Float32(0.0)
    var i = tid
    while i < n_blocks:
        acc += partials.unsafe_load(i)
        i += REDUCE_BY_KEY_TPB

    # `cub::BlockReduce`'s counterpart from
    # `max.gpu.primitives.block`. The hand-written shared-memory tree
    # reduction this replaced is gone: same arithmetic, one call, and
    # the reduction shape is Modular's to tune rather than ours to
    # guess. See VENDOR_LIBRARIES.md.
    var s0 = block_sum[block_size=REDUCE_BY_KEY_TPB](acc)
    if tid == 0:
        out_scalar.unsafe_store(0, s0)
