"""`cub::BlockReduce<T, TPB>::Sum`: the block-wide sum cuML's RF counts with.

PORT OF `cub/cub/block/block_reduce.cuh` and its default algorithm
`cub/cub/block/specializations/block_reduce_warp_reductions.cuh`
(`BLOCK_REDUCE_WARP_REDUCTIONS`, `block_reduce.cuh:291`) at NVIDIA/cccl
`d10a88a945caa4ea63dd2a909cf789c6dbe085a4`, cloned read-only into
`~/CascadeProjects/upstream/cccl` for this lane.

CUB IS OPEN SOURCE, SO IT IS A PORT TARGET, NOT A VENDOR CALL. The rule
that says "substitute the platform primitive for a vendor call" applies to
cuBLAS and cuSOLVER, whose kernels nobody can read. `block_reduce.cuh` can
be read, so it is transcribed. `max.gpu.primitives.block.sum` was NOT
called in its place: it is addition over a scalar with no say in the fold
ORDER, and the fold order is the whole content of
`ApplyWarpAggregates` below.

## The call site this exists for

`cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels_impl.cuh`
at rapidsai/cuml `v26.08.00`
(`265b9da6a0e75dbef071a3168398b993a5ff6f0e`), `countLocalLeftKernel`:

    using BlockReduce = cub::BlockReduce<std::int64_t, TPB>;   // :62
    __shared__ typename BlockReduce::TempStorage temp_storage; // :63
    ...
    const auto block_count = BlockReduce(temp_storage).Sum(thread_count);  // :78
    if (threadIdx.x == 0 && block_count > 0) {                            // :79
      atomicAdd(reinterpret_cast<unsigned long long*>(&splits[nid].local_nLeft),
                static_cast<unsigned long long>(block_count));            // :80-81
    }

One `std::int64_t` per thread, one block-wide total, valid in thread 0,
flushed with an atomic. Both halves are here: the reduction and the flush.

## What CUB's default `Sum` actually does, in order

`BlockReduce::Sum(T input)` (`block_reduce.cuh:592`) calls
`InternalBlockReduce(temp_storage).template Sum<true>(input, BLOCK_THREADS)`
-- a FULL TILE. `BlockReduceWarpReductions::Sum<FullTile>`
(`block_reduce_warp_reductions.cuh:187-200`) then:

  1. reduces inside each warp (`WarpReduceInternal`, i.e. the shuffle
     reduction),
  2. `ApplyWarpAggregates` (`:145-171`): lane 0 of each warp stores its
     aggregate to `temp_storage.warp_aggregates[warp_id]`,
     `__syncthreads()`, and then **thread 0 alone** folds warps
     `1 .. warps-1` into its own aggregate IN ASCENDING WARP ORDER.

That ascending, single-threaded fold is why the result is deterministic
and why it is transcribed rather than replaced by a tree or an atomic.
CUB ships an atomic variant right beside it
(`ApplyWarpAggregatesNonDeterministic`, `:112-135`) and the default
algorithm does not select it.

# =========================================================================
# DEVIATION BLOCK
#
# DEVIATION 124. THE WARP WIDTH IS QUERIED, NOT CUB'S HARDCODED 32.
#
# THEIRS: `BlockReduceWarpReductions` sizes itself from `warp_threads`
# (`block_reduce_warp_reductions.cuh:60-64`), which is
# `CUB_PTX_WARP_THREADS` (`cub/util_arch.cuh:105`), i.e. 32; and
# `raft::WarpSize` is a literal 32
# (`raft/util/cuda_dev_essentials.cuh:83`), used across cuML's own
# decision-tree code (`split.cuh:210,236-237`, `builder.cuh:536`,
# `builder_kernels_impl.cuh:365`). CUB and RAFT compile for one vendor, so
# a literal is correct there.
#
# OURS: `WARP_SIZE` from `std.gpu`, which is 32 on NVIDIA and Apple, 64 on
# AMD CDNA and 32 on AMD RDNA. `WARPS` therefore differs per vendor, and so
# does the NUMBER OF PARTIAL SUMS thread 0 folds and the number of lanes the
# warp stage folds.
#
# PRICE, stated rather than waved through:
#   * For an INTEGER element type the answer is BIT-IDENTICAL on every
#     vendor, because integer addition is associative and commutative and
#     no partial sum can overflow that would not overflow in any other
#     grouping. cuML's only instantiation is `std::int64_t`
#     (`builder_kernels_impl.cuh:62`), so the shipped path pays nothing.
#   * For a FLOAT element type it would NOT be identical: 128 threads fold
#     as 4 warp aggregates on Apple and 2 on CDNA, which is a different
#     summation tree. No float instantiation is reachable from this port
#     today. If one is added it needs a NUMERIC row in
#     `mojo_only/kernel_matrix.mojo` pinning the lane width, exactly as
#     `replication_lanes` is pinned to 32 there for the histogram. Declared,
#     not taken.
#
# The alternative -- transcribing 32 -- was REFUSED: it is the precise
# defect `kernel_matrix.mojo` was written to prevent, and on AMD CDNA it
# would size `warp_aggregates` for twice as many warps as exist and fold
# uninitialised threadgroup memory into the total.
#
# DEVIATION 125. THE FLUSH ATOMIC IS 32-BIT, NOT 64-BIT.
#
# THEIRS: `atomicAdd` on an `unsigned long long*` aliasing
# `Split::local_nLeft`, which is `std::int64_t` (`split.cuh:51`).
#
# OURS: `Atomic.fetch_add` on an `Int32`. NOT A CHOICE. MEASURED this
# session: a 64-bit `Atomic.fetch_add` on this target is a hard COMPILE
# error -- "Atomic operation is not supported for this type on Apple GPU"
# -- so the 64-bit form cannot be built at all, let alone run.
#
# PRICE: ZERO, and the claim is checked in their source rather than
# asserted. `local_nLeft` counts rows of ONE node's instance range that go
# left. That range is a subrange of `Dataset::row_ids`, which holds
# `n_sampled_rows` entries, and `n_sampled_rows` is `IdxT`
# (`dataset.h:32`). Every explicit instantiation of the RF builder fixes
# `IdxT = int`: `kernels/node-split.cu` instantiates
# `launchNodeSplitKernel<float,int,int,TPB_DEFAULT>` and three siblings,
# and `classification-*.cu` / `regression-*.cu` / `weighted-*.cu` each open
# with `using IdxT = int;`. So `0 <= local_nLeft <= n_sampled_rows <=
# INT_MAX` and 32 bits are EXACT, not truncating. The int64 in their struct
# is headroom they do not use, not a width they need.
#
# WHAT IS STILL OPEN, because a priced deviation is not a solved one: the
# REDUCTION accumulator here stays 64-bit (`block_reduce_sum` is generic
# over the dtype and `countLocalLeftKernel` instantiates it at int64), so
# only the ATOMIC narrows. A caller that wants the int64 total off the
# device without an atomic must reduce per block and sum the block results,
# which is one extra kernel; that path is NOT written here because cuML
# does not need it.
# =========================================================================
"""

from std.atomic import Atomic, Ordering
from std.gpu import thread_idx, lane_id, WARP_SIZE
from std.gpu.primitives import warp
from std.math import ceildiv
from std.memory import stack_allocation
from std.sys.info import size_of
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from mojo_only.kernel_matrix import TARGET_COLUMN, column_shared_limit


def block_reduce_sum[
    dtype: DType, TPB: Int, sabotage: Int = 0
](input: Scalar[dtype]) -> Scalar[dtype]:
    """`cub::BlockReduce<T, TPB>::Sum(input)`, `block_reduce.cuh:592`.

    `sabotage` is a CHECK HOOK and 0 is the only value a caller may pass.
    A digest cannot tell a working change from a no-op, so the cross-warp
    fold has a switch (`sabotage == 1` returns warp 0's aggregate alone)
    that the check flips to prove its comparison has teeth. It is here
    rather than in a copy of this function because sabotaging a copy
    proves nothing about this one.

    The return value is VALID ONLY IN THREAD 0, exactly as theirs is
    ("The return value is only valid for *thread*<sub>0</sub>",
    `block_reduce_warp_reductions.cuh:177-179`). Every thread must reach
    this call: it contains a `barrier()`.

    `TPB` is their `BLOCK_THREADS` and must equal the launched
    `block_dim.x`; CUB has the same requirement and states it as
    "the thread block size in threads" (`block_reduce.cuh:283`).
    """
    # `warps = ceil_div(threads_per_block, warp_threads)`,
    # `block_reduce_warp_reductions.cuh:60`. See DEVIATION 124 for why the
    # divisor is queried.
    comptime WARPS = ceildiv(TPB, WARP_SIZE)

    # The whole threadgroup claim of this collective is
    # `warp_aggregates[warps]` (`:75-76`). Their `block_prefix` member
    # (`:78`) belongs to the prefix-callback form, which cuML never calls,
    # so it is not allocated.
    comptime assert (
        WARPS * size_of[Scalar[dtype]]() <= column_shared_limit(TARGET_COLUMN)
    ), "block_reduce warp_aggregates exceeds the vendor threadgroup budget"

    var warp_aggregates = stack_allocation[
        WARPS,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()

    # Step 1: "Warp reduction in every warp"
    # (`block_reduce_warp_reductions.cuh:199-200`). `warp.sum` is Mojo's
    # spelling of the `__shfl_*_sync` reduction CUB's `WarpReduce`
    # internal is built from -- a language-level intrinsic, not a library
    # standing in for the algorithm. It differs from theirs in one
    # harmless way: Mojo broadcasts the result to every lane where CUB
    # leaves it valid only in lane 0. Only lane 0's copy is read below.
    var warp_aggregate = warp.sum(input)

    # Step 2: `ApplyWarpAggregates` (`:145-171`). "Share lane aggregates".
    if lane_id() == 0:
        warp_aggregates[unsafe_offset = Int(thread_idx.x) // WARP_SIZE] = (
            warp_aggregate
        )

    barrier()

    # "Update total aggregate in warp 0, lane 0" (`:158-170`). ONE thread,
    # ASCENDING warp order. `FullTile` is True at this entry point
    # (`block_reduce.cuh:592` passes `<true>`), so their
    # `if (FullTile || ...)` guard is unconditionally taken and is not
    # transcribed.
    comptime if sabotage == 1:
        # SABOTAGE: warps 1.. never reach the total. Thread 0 returns its
        # own warp's aggregate, which is right whenever the block is one
        # warp wide and wrong the moment it is not.
        pass
    else:
        if Int(thread_idx.x) == 0:
            for warp_idx in range(1, WARPS):
                var addend = warp_aggregates[unsafe_offset=warp_idx]
                warp_aggregate = warp_aggregate + addend

    return warp_aggregate


def block_flush_count_i32(
    splits_local_nleft: MutPointer[Int32, MutAnyOrigin],
    nid: Int,
    block_count: Int64,
):
    """`countLocalLeftKernel`'s flush, `builder_kernels_impl.cuh:79-81`.

        if (threadIdx.x == 0 && block_count > 0) {
          atomicAdd(reinterpret_cast<unsigned long long*>(&splits[nid].local_nLeft),
                    static_cast<unsigned long long>(block_count));
        }

    `nid` is theirs (`workload_info[blockIdx.x].nodeid`, `:65`) and is a
    parameter here rather than folded into the pointer because their
    address is `&splits[nid].local_nLeft` -- an INDEXED address, and the
    indexing is what makes concurrent blocks of different nodes hit
    different lines.

    Their `block_count > 0` guard is theirs and is kept: it is what keeps
    a block whose whole tile is invalid from touching the line at all.

    See DEVIATION 125 for why the target is `Int32` and why that is exact.
    """
    if Int(thread_idx.x) == 0 and block_count > Int64(0):
        # Upstream `atomicAdd` is relaxed; the non-Apple default here is
        # seq_cst. Fences only -- the total cannot move.
        _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
            splits_local_nleft.unsafe_offset(nid), Int32(block_count)
        )
