# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""Distance and 1-nearest-neighbor, GEMM first and reduction second.

PORT OF the non-fused arm of `minClusterAndDistanceCompute`,
`cuvs/src/cluster/detail/kmeans_common.cuh:450-491`, at cuVS `94c2819`.
(There is no `unfused_distance_nn.cuh` in cuVS.)
Transliterated. Do not improve.

cuVS has two paths to "which centroid is nearest, and how far". The FUSED one
keeps the distance tile in registers and never writes the `n x k` matrix; the
UNFUSED one calls a pairwise-distance kernel and then reduces its output.
**Their dispatch takes the fused arm for L2Expanded and L2SqrtExpanded and the
unfused arm for everything else** -- `is_fused` at `kmeans_common.cuh:378-379`
is that whole decision, a metric test with no hardware term in it.

k-means's default metric is L2Expanded, so their default fit takes the FUSED
arm, and so does this tree: `distance/fused_distance_nn/simt_kernel.mojo` is a
port of their SIMT fused kernel
(`src/distance/detail/fused_distance_nn/simt_kernel.cuh`) and it is what
`min_cluster_and_distance_compute` launches. **This file is the OTHER arm.**
It is kept reachable so the two can be differentially tested, and because it
is the arm their dispatch takes for metrics this tree does not yet fit.

(cuVS also has a CUTLASS specialization of the fused arm, under
`src/distance/detail/fused_distance_nn/`, which is CUDA and has no Metal
counterpart. Their SIMT kernel is the portable one and is the one ported.)

**What survives the port is the whole reason this is fast**: the expanded
identity `||x-c||^2 = ||x||^2 + ||c||^2 - 2 x.c`. That turns every pairwise
distance into one matrix product plus two vectors of precomputed norms, which
is why k-means on a GPU is a GEMM problem and not a distance problem.

Two consequences of the identity, both of them theirs and both load-bearing:

1. GEMM round-off can make the expanded value slightly NEGATIVE for a point
   sitting on its own centroid, so the kernel clamps to zero before any
   square root (`src/distance/detail/distance_ops/l2_exp.cuh:124-135`). Not
   defensive coding; without it `sqrt` returns NaN on the easiest input in
   the dataset.

   DIVERGENCE, NOT FIXED HERE. Their expression has a SECOND factor this
   port does not have: `!((val * val < get_clamp_precision<DataT, AccT>()) *
   (regxn[i] == regyn[j]))`, which forces an exact zero when the two norms
   are bit-equal and the residual is below a type-dependent precision floor
   (`l2_exp.cuh:132-134`). It is the self-neighbor case. For k-means it is
   nearly unreachable, because a point's norm is not bit-equal to a
   centroid's; for a self-join k-NN, where every row meets itself, it is the
   normal case. Both kernels here clamp on sign alone.
2. The reduction is an ARGMIN over a total order, not a sum. Ties break on
   the lower key, so the answer does not depend on the reduction tree, the
   block size, or the number of blocks. **This is the rare reduction in this
   repository that is exactly reproducible without costing anything**, and it
   is why the whole assignment step can sit in the SCHEDULING half of
   `mojo_only/numerics.mojo`'s table rather than the numeric half. The GEMM
   that feeds it is a different story, see `mojo_only/gemm.mojo`.
"""

from mojo_only.kernel_matrix import (
    K_LIB_FUSED_DISTANCE_NN,
    TARGET_COLUMN,
    lib_block_size_for,
    lib_reduce_lanes_for,
)


from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.primitives.warp import shuffle_xor
from std.math import sqrt
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation

from mojo_only.numerics import ftz, identical_mul_add


# `P::Nthreads` for their SIMT fused kernel is 256
# (`fused_distance_nn/simt_kernel.cuh:70`, `__launch_bounds__(P::Nthreads, 2)`,
# with the policy from `linalg/contractions.cuh`); this arm uses 128, and
# `blocks = m`,
# so the grid is one block per row of X and the block strides the centroids.
# READ FROM THE MATRIX, not restated here. `mojo_only/kernel_matrix.mojo`
# owns every tunable in this tree; changing TARGET_COLUMN there rebuilds
# this kernel for another vendor with no edit in this file.
comptime REDUCE_MIN_TPB = lib_block_size_for[
    K_LIB_FUSED_DISTANCE_NN, TARGET_COLUMN
]()

# The width of one shuffle group in the block argmin below. `shuffle_xor`
# with an offset under this value flips only the low bits of the lane id, so
# it never leaves an aligned group of this size. 32 is the lane width on
# NVIDIA and on Apple; on AMD's 64-wide wavefront each aligned half reduces
# independently and the cross-group stage picks both halves up, so the only
# assumption is that the hardware lane width is a MULTIPLE of 32, never that
# it equals 32.
# READ FROM THE MATRIX, not restated here. `mojo_only/kernel_matrix.mojo`
# owns every tunable in this tree; changing TARGET_COLUMN there rebuilds
# this kernel for another vendor with no edit in this file.
comptime REDUCE_MIN_LANES = lib_reduce_lanes_for[TARGET_COLUMN, False]()
comptime REDUCE_MIN_GROUPS = REDUCE_MIN_TPB // REDUCE_MIN_LANES

# `max_val<float>()` and `max_val<IdxT>()` at `:71-72`, the identity elements
# of the argmin.
comptime FLOAT32_MAX = Float32(3.4028234663852886e38)
comptime INDEX_MAX = UInt32(0xFFFFFFFF)

# `DistanceType` codes, matching `cluster/kmeans_params.mojo`.
comptime METRIC_L2_EXPANDED = 0
comptime METRIC_L2_SQRT_EXPANDED = 1
comptime METRIC_COSINE_EXPANDED = 2


def reduce_min_kernel(
    out_key: MutPointer[UInt32, MutAnyOrigin],
    out_value: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    x_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    metric_in: Int32,
    is_sqrt_in: Int32,
    init_out_buffer_in: Int32,
    key_base_in: Int32,
):
    """`reduce_min_kernel`, copied.

    One block per row, `TPB` threads striding the centroid axis, then a block
    argmin. Distance is FORMED here rather than by the GEMM: the GEMM only
    produced `z = x . y^T` and the norms turn that into a distance one element
    at a time, which is why their comment says the reduction cannot use a
    stock RAFT reduce.

    DEVIATIONS, both mechanical:

    - `cub::BlockReduce<KVType, TPB>` becomes a warp-shuffle butterfly
      followed by a one-slot-per-warp shared merge. **This used to be a
      whole-block shared-memory tree, justified by a claim that Mojo has no
      warp primitives. That claim was false** (PORTING.md 2). CUB's default
      algorithm is `BLOCK_REDUCE_WARP_REDUCTIONS`
      (`cub/block/block_reduce.cuh:238`), documented at `:120-135` as a
      warp-synchronous reduction inside each warp followed by a propagation
      phase over the per-warp aggregates. So the two stages below are THEIRS
      rather than a substitute for them, and the shared footprint drops from
      `TPB` pairs to `TPB / 32` pairs.
    - `raft::KeyValuePair` becomes two output pointers. Mojo can express the
      struct, but a kernel parameter that is a struct of mixed types is a
      launch risk this tree has already been bitten by (PORTING.md 9), and
      the split costs one extra store per block.

    Neither one costs fidelity, and the reason is worth stating precisely
    (PORTING.md 14). Unlike the histogram scan (PORTING.md 8) this reducer is
    a min over a TOTAL order, `(value, then key ascending)`, which makes it
    associative, commutative AND idempotent. The shared tree, the butterfly,
    and CUB's own raking reduction therefore all return the same pair. The
    tie-break on the lower key is what makes that true; without it the block
    shape would pick the winner. It is load-bearing, not tidiness.

    LAUNCH CONTRACT: `block_dim` must be exactly `REDUCE_MIN_TPB`. Their
    `cub::BlockReduce<KVType, TPB>` carries the same requirement, and so did
    the tree this replaced: a short block leaves shared slots unwritten.

    `key_base` is ours and is not a deviation in behavior: their caller adds
    the centroid-tile offset AFTERWARD, in a `raft::linalg::map` over the
    whole output (`kmeans_common.cuh:405-410`, a `thrust::fill` of
    `(0, FLT_MAX)` before the tile loop). Folding it into
    the kernel removes a full-length pass over the output per centroid tile
    and cannot change which centroid wins, because it is the same constant
    added to every key in the tile.
    """
    var n = Int(n_in)
    var metric = Int(metric_in)
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var x_norm_row = x_norm.unsafe_load(row)

    var thread_min_value = FLOAT32_MAX
    var thread_min_key = INDEX_MAX

    var col = tid
    while col < n:
        var dist = Float32(0.0)

        if metric == METRIC_COSINE_EXPANDED:
            # Guard against zero-norm vectors to avoid inf/NaN from division
            # by zero. Theirs, `:84-86`.
            var denom = x_norm_row * y_norm.unsafe_load(col)
            if denom <= Float32(0.0):
                denom = Float32(1.0)
            dist = Float32(1.0) - (z.unsafe_load(row * n + col) / denom)
        else:
            # The same pinned multiply-add the FUSED twin's epilog uses
            # (IDENTITY_PATHS row 9), so the two arms differ only where
            # their inputs do -- this one's `z` is a VENDOR MATMUL, which
            # is why the unfused arm is refused under IDENTICAL and kept
            # for differential testing. See the module docstring.
            dist = ftz(
                identical_mul_add(
                    Float32(-2.0),
                    ftz(z.unsafe_load(row * n + col)),
                    ftz(ftz(x_norm_row) + ftz(y_norm.unsafe_load(col))),
                )
            )
            # GEMM round-off can produce slightly negative expanded
            # distances; clamp to zero. Theirs,
            # `src/distance/detail/distance_ops/l2_exp.cuh:132`, minus the
            # self-neighbor factor -- see the module docstring.
            if dist <= Float32(0.0):
                dist = Float32(0.0)

        # Strict `<`, so within a thread the LOWEST column wins a tie. Half
        # of their `Reducer`'s total order lives here.
        if dist < thread_min_value:
            thread_min_value = dist
            thread_min_key = UInt32(col)

        col += REDUCE_MIN_TPB

    # --- STAGE 1, the warp butterfly. CUB's `BLOCK_REDUCE_WARP_REDUCTIONS`
    # first half, and the same primitive their fused twin uses in
    # `simt_kernel.cuh:125-126`. `Reducer::operator()` at `:44-49` verbatim:
    #     (a.value < b.value) || (a.value == b.value && a.key < b.key)
    # Every lane of the group ends holding the group's winner, which is what
    # lets one leader per group write stage 2 with no extra broadcast.
    #
    # All REDUCE_MIN_TPB threads reach these shuffles unconditionally. That
    # is required: a lane that skips a full-mask shuffle hangs the rest.
    var lane_offset = 1
    while lane_offset < REDUCE_MIN_LANES:
        var other_value = shuffle_xor(thread_min_value, UInt32(lane_offset))
        var other_key = shuffle_xor(thread_min_key, UInt32(lane_offset))
        var takes_other = other_value < thread_min_value or (
            other_value == thread_min_value and other_key < thread_min_key
        )
        if takes_other:
            thread_min_value = other_value
            thread_min_key = other_key
        lane_offset *= 2

    # --- STAGE 2, the merge over the per-group aggregates. Four pairs, not
    # 128, because stage 1 already collapsed each group.
    var s_value = stack_allocation[
        REDUCE_MIN_GROUPS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_key = stack_allocation[
        REDUCE_MIN_GROUPS,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    if tid % REDUCE_MIN_LANES == 0:
        s_value[tid // REDUCE_MIN_LANES] = thread_min_value
        s_key[tid // REDUCE_MIN_LANES] = thread_min_key
    barrier()

    if tid == 0:
        var best_value = s_value[0]
        var best_group_key = s_key[0]
        # Same total order again, so the group count cannot pick the winner.
        for g in range(1, REDUCE_MIN_GROUPS):
            var group_value = s_value[g]
            var group_key = s_key[g]
            var takes_group = group_value < best_value or (
                group_value == best_value and group_key < best_group_key
            )
            if takes_group:
                best_value = group_value
                best_group_key = group_key
        var best_key = best_group_key + UInt32(key_base_in)
        if is_sqrt_in != 0:
            if best_value <= Float32(0.0):
                best_value = Float32(0.0)
            best_value = sqrt(best_value)
        if init_out_buffer_in != 0:
            out_key.unsafe_store(row, best_key)
            out_value.unsafe_store(row, best_value)
        else:
            # The merge arm, `:107-113`. Same total order as both reduction
            # stages above, so tiling the centroids cannot change the winner.
            var prev_value = out_value.unsafe_load(row)
            var prev_key = out_key.unsafe_load(row)
            var takes_new = best_value < prev_value or (
                best_value == prev_value and best_key < prev_key
            )
            if takes_new:
                out_key.unsafe_store(row, best_key)
                out_value.unsafe_store(row, best_value)
