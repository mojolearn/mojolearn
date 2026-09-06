# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Weighted sums per cluster, which is the whole update half of Lloyd.

NOT A PORT of cuVS. Their two calls are
`raft::linalg::reduce_rows_by_key` and `raft::linalg::reduce_cols_by_key`
(`raft/linalg/reduce_rows_by_key.cuh` and `reduce_cols_by_key.cuh`, called
from `update_centroids`, `detail/kmeans.cuh:300-318`), and RAFT is a
separate library this tree does not mirror file for file. The CALL SITES,
their order, and their accumulate-versus-reset semantics are theirs and are
copied in `cluster/impl/cluster/detail/kmeans_common.mojo`.

THE ACCUMULATOR IS THE INTERESTING PART, AND IT IS NOT A NEW PROBLEM
--------------------------------------------------------------------
Summing rows into their cluster is a scatter-add: many rows land on one
centroid and the order they arrive in is not fixed. RAFT does it with float
atomics on NVIDIA. This file quantizes into `Int32` through
**`checks/fixed_point.mojo`** instead, and the reason is REPRODUCIBILITY,
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

It also gives `checks/fixed_point.mojo` its first reader. `archive/plans/UNWIRED.md`
listed it as verified in isolation and used by no kernel.
"""

from checks.kernel_matrix import (
    K_LIB_REDUCE_BY_KEY,
    TARGET_COLUMN,
    column_shared_limit,
    lib_block_size_for,
)
from checks.hardware_matrix import gpu_cores_for, max_active_blocks_for

from cluster.impl.distance.fused_distance_nn.simt_kernel import (
    fused_veclen_for,
)

from max.gpu.host import DeviceBuffer, DeviceContext
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.memory import AddressSpace
from core.pinned_reduce import pinned_block_sum
from checks.numerics import ftz
from max.gpu.sync import barrier
from std.memory import stack_allocation


# READ FROM THE MATRIX, not restated here. `checks/kernel_matrix.mojo`
# owns every tunable in this tree; changing TARGET_COLUMN there rebuilds
# this kernel for another vendor with no edit in this file.
comptime REDUCE_BY_KEY_TPB = lib_block_size_for[
    K_LIB_REDUCE_BY_KEY, TARGET_COLUMN
]()

#: Threadgroup-private accumulator cells (Int32) for the privatized arm.
#: RAFT sizes its shared cache DYNAMICALLY at launch and guards the arm with
#: "the cache fits in shared memory" -- `cache_size <= 49152ull`, the full
#: default CUDA shared budget (`raft/linalg/detail/reduce_cols_by_key.cuh:
#: 125-126`). Mojo's `stack_allocation` is comptime-sized, so the cap is
#: pinned here instead, from the matrix's per-block shared limit: HALF the
#: target's budget, in Int32 cells (4,096 cells / 16 KB on the Apple
#: column's 32 KB). Half rather than all of it so the launch never sits on
#: the validity wall; k-means' shipped shape (k=64 x d=32 = 2,048 cells,
#: 8 KB) fits with headroom, exactly as the SCOREBOARD's privatization case
#: priced it.
comptime PRIVATE_ACC_CELLS = column_shared_limit(TARGET_COLUMN) // 8

#: RAFT's "input is large enough to be worth the flush" guard, copied:
#: `nrows * ncols >= IdxType{8192}` (`reduce_cols_by_key.cuh:126`, with
#: their own comment: cached is slightly slower for small inputs, orders of
#: magnitude faster for large ones).
comptime PRIVATE_MIN_WORK = 8192


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

    This mirrors the arm THEIR dispatch takes at k-means' nkeys: the
    small-nkeys kernel is gated `nkeys <= 4`, everything else falls to
    `sum_rows_by_key_large_nkeys_rowmajor` (`raft/linalg/detail/
    reduce_rows_by_key.cuh:354-363`). Measured contention-bound at the 4M
    x 32 k=64 shape, so the SHIPPED dispatch (`launch_accumulate_centroid_
    sums`) now prefers the bit-identical privatized kernel below and keeps
    this one as the guard's fallback arm.

    **The atomic scatter-add itself is faithful, not a substitution gap.**
    RAFT's `reduce_rows_by_key.cuh:287` (`raft::myAtomicAdd`) also lands its
    contributions with a
    global atomic add rather than a vendor segmented reduce, so there is
    nothing here to swap `cub::DeviceSegmentedReduce` into (and that is
    NOT FOUND anyway; see archive/reference/VENDOR_LIBRARIES.md). The only thing we change is
    the accumulator TYPE, and the module docstring says why: not because
    Metal lacks a float atomic, which it does not, but because an integer
    accumulator is order-independent and a float one is not.

    The indexing is theirs too: a FLAT grid-stride over `n_rows * n_features`
    cells, matching their `gid = threadIdx.x + blockDim.x * blockIdx.x`
    (`reduce_rows_by_key.cuh:280`, in
    `sum_rows_by_key_large_nkeys_kernel_rowmajor`). The earlier shape here strided rows on
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
    """`reduce_cols_by_key` over a single row of weights -- their DIRECT arm
    (`reduce_cols_by_key_direct_kernel`, `raft/linalg/detail/
    reduce_cols_by_key.cuh:35-47`).

    Their call passes `nrows = 1` and the weight vector as the matrix
    (`cuvs/src/cluster/detail/kmeans.cuh:312-318`), so this is the
    denominator of the centroid update and the empty-cluster test in one
    array. NOTE: at the fit's shape their dispatch takes the CACHED arm, not
    this one; see `accumulate_weight_per_cluster_privatized_kernel`.
    """
    var n_rows = Int(n_rows_in)
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row < n_rows:
        var label = Int(labels.unsafe_load(row))
        var q = Int32(weights.unsafe_load(row) * scale_in)
        _ = Atomic.fetch_add(weight_i32.unsafe_offset(label), q)


def accumulate_centroid_sums_privatized_kernel[
    veclen: Int
](
    sums_i32: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    labels: MutPointer[UInt32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_features_in: Int32,
    n_clusters_in: Int32,
    scale_in: Float32,
):
    """The SMEM-privatized arm: block-local partials, one flush per block.

    DEVIATION, and a measured one. RAFT's `reduce_rows_by_key` dispatch at
    nkeys=64 takes `sum_rows_by_key_large_nkeys_kernel_rowmajor` -- the
    small-nkeys arm is gated `nkeys <= 4` (`raft/linalg/detail/
    reduce_rows_by_key.cuh:354-363`) -- which is exactly the direct global
    scatter-add `accumulate_centroid_sums_kernel` above mirrors. Measured
    here (SCOREBOARD_2026-08-19 item 1, 4M x 32, k=64) that arm is
    contention-bound at 56 ms against a 15-20 ms traffic floor: 128M global
    atomics landing on 2,048 cells.

    The privatization STRUCTURE is still theirs, not invented: RAFT ships it
    twice in the same two files, as `sum_rows_by_key_large_nkeys_kernel_
    colmajor` (`reduce_rows_by_key.cuh:196-242`: `__shared__ local_sums`,
    accumulate with shared atomics, flush non-zero cells to global atomics)
    and as `reduce_cols_by_key_cached_kernel` (`reduce_cols_by_key.cuh:
    51-81`, same three phases) -- the arm THEIR `reduce_cols_by_key`
    dispatch takes for this fit's WEIGHT reduction. This kernel applies that
    scheme to the row-major sums, walking the direct arm's flat cell
    order `veclen` cells per thread step.

    THE `veclen`-WIDE X READ IS A SECOND DELIBERATE DEVIATION BEYOND
    UPSTREAM (archive/reference/PORTING.md 46). RAFT's rowmajor kernel reads ONE element per
    thread (`SumsT val = d_A[j + lda * i]`, `reduce_rows_by_key.cuh:285`;
    no `TxN_t`/`ldg` anywhere in that file) and loses nothing by it on
    NVIDIA, where a warp's 32 scalar reads coalesce into full
    transactions. On this Apple device the same scalar-to-vector swap was
    MEASURED 3x on the assignment kernel (63 -> 21 ms/iter, re-verdict
    2026-08-20), so the hardware premise upstream's scalar reads rest on
    does not hold here. The instantiation contract is the SAME ladder the
    assignment port dispatches on (`fused_veclen_for`, their
    `fused_distance_nn-inl.cuh:107-110` selection, fed x's base address
    for both pointer terms): it guarantees `n_features % veclen == 0`, so
    a chunk of `veclen` cells never straddles a row (one label and one
    weight read serve the whole chunk), and a `4 * veclen`-byte aligned
    load that starts in bounds ends in bounds. `veclen = 1` IS the old
    scalar body, and it is the arm the dispatch takes whenever the ladder
    says the row length or alignment forbids wider. A vector load returns
    the same bits as `veclen` scalar loads and each lane's quantization
    is the identical scalar fp32 expression, so bit-identity is untouched
    -- proven by `check_privatized_accumulate` (veclen=4 pinned) and
    `check_accumulate_veclen_dispatch` (veclen=1 at d=33, veclen=2 at
    d=34), each held bitwise to the direct scatter-add oracle.

    DETERMINISM IS PRESERVED, BY THE SAME ARGUMENT THAT PICKED Int32.
    The addends are the identical quantized `Int32` values the direct kernel
    forms, and Int32 addition (two's-complement, wrapping) is associative
    and commutative. Grouping them into per-block partials and adding the
    partials atomically is a re-association of the same multiset of adds, so
    the totals are BIT-IDENTICAL to the direct kernel's and to themselves
    run to run, at any grid size, under any scheduling. A float accumulator
    would lose exactly this property, which is why privatizing it would have
    needed a numerics argument and this needs one comment.

    Skipping zero partials in the flush is theirs (`reduce_rows_by_key.cuh:
    235`, `reduce_cols_by_key.cuh:78`) and is safe under the same argument:
    adding zero is the identity, in integers exactly.

    Requires `n_clusters * n_features <= PRIVATE_ACC_CELLS`; the host
    dispatch in `launch_accumulate_centroid_sums` guards it, as theirs does.
    """
    var n_rows = Int(n_rows_in)
    var n_features = Int(n_features_in)
    var used = Int(n_clusters_in) * n_features
    var total = n_rows * n_features
    var tid = Int(thread_idx.x)

    var priv = stack_allocation[
        PRIVATE_ACC_CELLS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    # `for (local_key = threadIdx.x; ...) local_sums[local_key] = 0.0;`
    var z = tid
    while z < used:
        priv.unsafe_store(z, Int32(0))
        z += Int(block_dim.x)
    barrier()

    # The direct arm's flat grid-stride, walked `veclen` cells per step:
    # ONE `SIMD[float32, veclen]` x read and one label/weight read per
    # chunk (`n_features % veclen == 0` is the dispatch contract, so a
    # chunk never straddles a row), then `veclen` atomics landing in the
    # block's own threadgroup memory (their `raft::myAtomicAdd(
    # &local_sums[...])`, shared-memory atomic, `reduce_rows_by_key.cuh:227`).
    var chunks = total // veclen
    var gid = Int(block_idx.x) * Int(block_dim.x) + tid
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while gid < chunks:
        var base = gid * veclen
        var row = base // n_features
        var f = base - row * n_features
        var label = Int(labels.unsafe_load(row))
        var w = weights.unsafe_load(row)
        var xv = x.unsafe_load[width=veclen](base)
        comptime for j in range(veclen):
            var q = Int32(xv[j] * w * scale_in)
            _ = Atomic.fetch_add(
                priv.unsafe_offset(label * n_features + f + j), q
            )
        gid += stride
    barrier()

    # One flush per block: `used` global atomics instead of one per element.
    var c = tid
    while c < used:
        var v = priv.unsafe_load(c)
        if v != Int32(0):
            _ = Atomic.fetch_add(sums_i32.unsafe_offset(c), v)
        c += Int(block_dim.x)


def accumulate_weight_per_cluster_privatized_kernel(
    weight_i32: MutPointer[Int32, MutAnyOrigin],
    labels: MutPointer[UInt32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_clusters_in: Int32,
    scale_in: Float32,
):
    """The weight denominator through the SAME privatization -- and for this
    one it is not a deviation at all: the upstream call IS
    `raft::linalg::reduce_cols_by_key` (`cuvs/src/cluster/detail/
    kmeans.cuh:312-318`, nrows=1, ncols=n_samples, nkeys=n_clusters), whose
    dispatch at this fit's shape (cache 4*k bytes <= 49152, work
    n_samples >= 8192) takes `reduce_cols_by_key_cached_kernel`
    (`reduce_cols_by_key.cuh:125-133`) -- the shared-memory arm. The direct
    `accumulate_weight_per_cluster_kernel` above mirrored their OTHER arm.

    Bit-identical to it anyway: same quantized Int32 addends, associative
    and commutative, re-associated per block. See the sums kernel's comment.
    """
    var n_rows = Int(n_rows_in)
    var used = Int(n_clusters_in)
    var tid = Int(thread_idx.x)

    var priv = stack_allocation[
        PRIVATE_ACC_CELLS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var z = tid
    while z < used:
        priv.unsafe_store(z, Int32(0))
        z += Int(block_dim.x)
    barrier()

    var row = Int(block_idx.x) * Int(block_dim.x) + tid
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while row < n_rows:
        var label = Int(labels.unsafe_load(row))
        var q = Int32(weights.unsafe_load(row) * scale_in)
        _ = Atomic.fetch_add(priv.unsafe_offset(label), q)
        row += stride
    barrier()

    var c = tid
    while c < used:
        var v = priv.unsafe_load(c)
        if v != Int32(0):
            _ = Atomic.fetch_add(weight_i32.unsafe_offset(c), v)
        c += Int(block_dim.x)


def accumulate_grid_blocks(work_items: Int, smem_bytes: Int) raises -> Int:
    """Grid for the accumulate kernels, FROM THE MATRIX, replacing the magic
    `min(1024, ...)` cap the launch site used to carry.

    RAFT sizes the cached arm's grid from a runtime occupancy query:
    `target_nblks = 4 * getMultiProcessorCount(); nblks = min(target_nblks,
    ceildiv(work, TPB))` (`reduce_cols_by_key.cuh:127-131`). Metal exposes
    no such query through Mojo, so the two factors are read from
    `checks/hardware_matrix.mojo`'s target column the way every other
    launch in this tree reads them: cores x resident blocks per core, then
    capped by the work. Their literal `4` is their occupancy guess for their
    kernel; ours is the table's computed one (`max_active_blocks_for`), so
    the same call is honest per vendor column instead of pinned to CUDA's.

    For the DIRECT sums arm RAFT launches one thread per cell with no cap
    (`reduce_rows_by_key.cuh:304-307`). That kernel here grid-strides
    instead -- a recorded launch-shape deviation, same arithmetic and same
    atomic targets -- and this function now sizes that grid too (pass
    `smem_bytes = 0`), so no launch below restates a hardware number. The
    direct WEIGHT kernel is one-thread-per-row like theirs and keeps its
    ceildiv grid.
    """
    var per_core = max_active_blocks_for[TARGET_COLUMN](
        REDUCE_BY_KEY_TPB, smem_bytes
    )
    var target = gpu_cores_for[TARGET_COLUMN]() * per_core
    var by_work = (work_items + REDUCE_BY_KEY_TPB - 1) // REDUCE_BY_KEY_TPB
    var blocks = target if by_work > target else by_work
    if blocks < 1:
        blocks = 1
    return blocks


def _enqueue_privatized_sums[
    veclen: Int
](
    ctx: DeviceContext,
    mut sums_i32: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    mut labels: DeviceBuffer[DType.uint32],
    mut weights: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    sum_scale: Float32,
) raises:
    """One privatized-arm launch, bound at a comptime `veclen`. Exists so
    `launch_accumulate_centroid_sums`' runtime ladder result can meet the
    kernel's comptime parameter without repeating the argument list three
    times (the shape `min_cluster_distance_compute.mojo`'s `_launch_fused`
    uses for the same reason)."""
    comptime kern = accumulate_centroid_sums_privatized_kernel[veclen]
    ctx.enqueue_function[kern](
        sums_i32.unsafe_ptr(),
        x.unsafe_ptr(),
        labels.unsafe_ptr(),
        weights.unsafe_ptr(),
        Int32(n_samples),
        Int32(n_features),
        Int32(n_clusters),
        sum_scale,
        grid_dim=(
            accumulate_grid_blocks(
                (n_samples * n_features) // veclen, PRIVATE_ACC_CELLS * 4
            ),
            1,
            1,
        ),
        block_dim=(REDUCE_BY_KEY_TPB, 1, 1),
    )


def launch_accumulate_centroid_sums(
    ctx: DeviceContext,
    mut sums_i32: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    mut labels: DeviceBuffer[DType.uint32],
    mut weights: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    sum_scale: Float32,
) raises:
    """The dispatch, shaped like RAFT's (`reduce_cols_by_key.cuh:125-139`):
    privatized when the block-local cache fits and the input is large enough
    to amortize the flush, direct scatter-add otherwise. The privatized
    arm's X read width comes from the same selection ladder the assignment
    launcher dispatches on (`fused_veclen_for`, fed x's address for both
    pointer terms because this kernel reads one matrix; archive/reference/PORTING.md 46).
    Both arms -- and every `veclen` instantiation of the privatized one --
    produce bit-identical Int32 totals (see the kernels), so this selector
    is SCHEDULING: it can change the time, never the model.
    """
    var cells = n_samples * n_features
    if (
        n_clusters * n_features <= PRIVATE_ACC_CELLS
        and cells >= PRIVATE_MIN_WORK
    ):
        var vl = fused_veclen_for(
            n_features, Int(x.unsafe_ptr()), Int(x.unsafe_ptr())
        )
        if vl == 4:
            _enqueue_privatized_sums[4](
                ctx, sums_i32, x, labels, weights,
                n_samples, n_features, n_clusters, sum_scale,
            )
        elif vl == 2:
            _enqueue_privatized_sums[2](
                ctx, sums_i32, x, labels, weights,
                n_samples, n_features, n_clusters, sum_scale,
            )
        else:
            _enqueue_privatized_sums[1](
                ctx, sums_i32, x, labels, weights,
                n_samples, n_features, n_clusters, sum_scale,
            )
    else:
        ctx.enqueue_function[accumulate_centroid_sums_kernel](
            sums_i32.unsafe_ptr(),
            x.unsafe_ptr(),
            labels.unsafe_ptr(),
            weights.unsafe_ptr(),
            Int32(n_samples),
            Int32(n_features),
            sum_scale,
            grid_dim=(accumulate_grid_blocks(cells, 0), 1, 1),
            block_dim=(REDUCE_BY_KEY_TPB, 1, 1),
        )


def launch_accumulate_weight_per_cluster(
    ctx: DeviceContext,
    mut weight_i32: DeviceBuffer[DType.int32],
    mut labels: DeviceBuffer[DType.uint32],
    mut weights: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_clusters: Int,
    weight_scale: Float32,
) raises:
    """Same dispatch for the denominator; upstream's own is
    `reduce_cols_by_key` at nrows=1, so the guard terms are `n_clusters`
    cells and `n_samples` work items."""
    if n_clusters <= PRIVATE_ACC_CELLS and n_samples >= PRIVATE_MIN_WORK:
        ctx.enqueue_function[accumulate_weight_per_cluster_privatized_kernel](
            weight_i32.unsafe_ptr(),
            labels.unsafe_ptr(),
            weights.unsafe_ptr(),
            Int32(n_samples),
            Int32(n_clusters),
            weight_scale,
            grid_dim=(
                accumulate_grid_blocks(n_samples, PRIVATE_ACC_CELLS * 4),
                1,
                1,
            ),
            block_dim=(REDUCE_BY_KEY_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[accumulate_weight_per_cluster_kernel](
            weight_i32.unsafe_ptr(),
            labels.unsafe_ptr(),
            weights.unsafe_ptr(),
            Int32(n_samples),
            weight_scale,
            grid_dim=(
                (n_samples + REDUCE_BY_KEY_TPB - 1) // REDUCE_BY_KEY_TPB,
                1,
                1,
            ),
            block_dim=(REDUCE_BY_KEY_TPB, 1, 1),
        )


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
        # Two quotients and a divide, every one of them a seam another
        # kernel reads: flushed under IDENTICAL (IDENTITY_PATHS row 10).
        # The DIVIDENDS are exact integers, so nothing here can differ
        # between vendors except through the denormal policy -- which is
        # precisely what a centroid of a near-empty cluster can produce.
        var w = ftz(Float32(weight_i32.unsafe_load(cluster)) / weight_scale_in)
        if w == Float32(0.0):
            new_centroids.unsafe_store(idx, old_centroids.unsafe_load(idx))
        else:
            var s = ftz(Float32(sums_i32.unsafe_load(idx)) / sum_scale_in)
            new_centroids.unsafe_store(idx, ftz(s / w))


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
    `checks/numerics.mojo` is about: a grid size looks like scheduling and
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
            v = ftz(ftz(v) * ftz(b.unsafe_load(i)))
        elif mode == SUM_MODE_SQDIFF:
            var d = ftz(ftz(v) - ftz(b.unsafe_load(i)))
            v = ftz(d * d)
        acc = ftz(acc + v)
        i += stride

    # `cub::BlockReduce`'s counterpart from
    # `max.gpu.primitives.block`. The hand-written shared-memory tree
    # reduction this replaced is gone: same arithmetic, one call, and
    # the reduction shape is Modular's to tune rather than ours to
    # guess. See archive/reference/VENDOR_LIBRARIES.md.
    var s0 = pinned_block_sum[REDUCE_BY_KEY_TPB](acc)
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

    It is also the second time this tree has needed one: see `archive/plans/UNWIRED.md` on
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
    `archive/reference/HOST_AND_DEVICE.md`.

    One block, because `n_blocks` is at most 256 by construction.
    """
    var n_blocks = Int(n_blocks_in)
    var tid = Int(thread_idx.x)

    var acc = Float32(0.0)
    var i = tid
    while i < n_blocks:
        acc = ftz(acc + partials.unsafe_load(i))
        i += REDUCE_BY_KEY_TPB

    # `cub::BlockReduce`'s counterpart from
    # `max.gpu.primitives.block`. The hand-written shared-memory tree
    # reduction this replaced is gone: same arithmetic, one call, and
    # the reduction shape is Modular's to tune rather than ours to
    # guess. See archive/reference/VENDOR_LIBRARIES.md.
    var s0 = pinned_block_sum[REDUCE_BY_KEY_TPB](acc)
    if tid == 0:
        out_scalar.unsafe_store(0, s0)
