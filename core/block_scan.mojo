# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`cub::BlockScan<BinT, TPB>::InclusiveSum`, and cuML's `pdf_to_cdf` loop.

PORT OF `cub/cub/block/block_scan.cuh` and its default algorithm
`cub/cub/block/specializations/block_scan_warp_scans.cuh`, over
`cub/cub/warp/specializations/warp_scan_smem.cuh`, at NVIDIA/cccl
`d10a88a945caa4ea63dd2a909cf789c6dbe085a4` (cloned read-only into
`~/CascadeProjects/upstream/cccl` for this lane), plus the caller
`cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels_impl.cuh
:259-282` and the element type
`cpp/src/decisiontree/batched-levelalgo/bins.cuh` at rapidsai/cuml
`v26.08.00` (`265b9da6a0e75dbef071a3168398b993a5ff6f0e`).

CUB IS OPEN, SO IT IS PORTED. `max.gpu.primitives.block.prefix_sum` was
not called in its place, and here the reason is not a preference: it is
addition over a SCALAR and takes no operator, while this scan's element is
one of cuML's four bin STRUCTS. There is nothing to substitute.

## The call site

    template <typename BinT, typename IdxT, int TPB>
    DI BinT pdf_to_cdf(BinT* histogram, IdxT n_bins)          // :263
    {
      typedef cub::BlockScan<BinT, TPB> BlockScan;            // :267
      __shared__ typename BlockScan::TempStorage temp_storage;// :268
      BinT total_aggregate = BinT();                          // :271
      for (IdxT tix = threadIdx.x;
           tix < raft::ceildiv(n_bins, TPB) * TPB;
           tix += blockDim.x) {                               // :273
        BinT result; BinT block_aggregate;
        BinT element = tix < n_bins ? histogram[tix] : BinT();// :276
        BlockScan(temp_storage).InclusiveSum(element, result, block_aggregate); // :277
        __syncthreads();                                      // :278
        if (tix < n_bins) { histogram[tix] = result + total_aggregate; } // :279
        total_aggregate += block_aggregate;                   // :280
      }
      return total_aggregate;                                 // :282
    }

Three things in that loop decide the port's shape and all three are
transcribed rather than paraphrased:

  * the loop bound is `ceildiv(n_bins, TPB) * TPB`, so EVERY thread enters
    EVERY iteration even when its element is past `n_bins`. A collective
    with a barrier inside cannot be entered by a subset of the block, and
    that is why they pad instead of returning early.
  * one `TempStorage` is REUSED across iterations, which is what the
    `__syncthreads()` at `:278` is for -- not for the histogram write.
  * the carry is applied to the OUTPUT (`result + total_aggregate`), not
    fed into the next scan as a seed. CUB ships a prefix-callback form
    (`BlockScanWarpScans::InclusiveScan` with `BlockPrefixCallbackOp`,
    `block_scan_warp_scans.cuh:486-508`) that would do it inside the
    collective; cuML does not use it, so this file does not port it.

## `BinT` is not a scalar

`bins.cuh` defines four: `ClassificationBin` (one `unsigned long long`),
`WeightedClassificationBin` (+ a `double`), `RegressionBin` (a `double`
and a count), `WeightedRegressionBin` (three fields). Each has

    HDI BinT& operator+=(const BinT& b) { <fieldwise add>; return *this; }
    HDI BinT  operator+(BinT b) const   { b += *this; return b; }

and a default constructor that zeroes. `BlockScanElement` below is exactly
that pair of requirements: `zero()` is their default constructor and
`plus()` is their `operator+`. THE ELEMENT TYPE IS THE CALLER'S PROBLEM,
including the fact that this device has no float64 and their `double`
fields cannot be carried at their width -- that is DEVIATION 101 in
`ensemble/decisiontree/batched_levelalgo/bins.mojo`, not this file's.

# =========================================================================
# DEVIATION BLOCK
#
# DEVIATION 126. THE WARP SCAN TAKES CUB'S OWN `HAS_IDENTITY == false`
# ARM, UNPADDED, AND SYNCS AT BLOCK SCOPE.
#
# THEIRS: `WarpScanSmem` allocates `LOGICAL_WARP_THREADS +
# HALF_WARP_THREADS` slots per warp (`warp_scan_smem.cuh:66`) and, when the
# operator has an identity, preloads the pad with it so `ScanStep` can read
# `temp_storage[HALF_WARP_THREADS + lane_id - OFFSET]` with NO predicate
# (`:113-120`, guarded by `HAS_IDENTITY ||`). It also fences with
# `__syncwarp(member_mask)` between the store and the load (`:111`, `:120`).
#
# OURS: `WARP_SIZE` slots per warp, no pad, and CUB's OTHER arm --
# `lane_id >= OFFSET` -- taken unconditionally; the fence is `barrier()`,
# Mojo's `__syncthreads`, because Mojo 1.0 exposes no lane-scoped fence and
# this repository does not invent one.
#
# PRICE, both directions, and the first one is the point:
#   * NO VALUE CHANGES, AND AT cuML'S OWN INSTANTIATION THIS IS NOT EVEN A
#     DEVIATION. CUB selects between the two arms with
#     `cuda::has_identity_element_v<ScanOp, T>` (`:186`), which is FALSE
#     for a user-defined `BinT` under `cuda::std::plus<>`. So the arm this
#     file transcribes is the arm CUB ITSELF COMPILES for every one of the
#     four bins. The padded arm is reachable only for built-in scalars,
#     which cuML never scans here.
#   * THREADGROUP MEMORY SAVED: `TPB / 2` elements, the pad CUB carries.
#   * THREADS SYNCHRONISED THAT NEED NOT BE: a block barrier where a warp
#     fence would do, `log2(WARP_SIZE)` times per scan plus once for the
#     aggregates. Every thread of the block executes the same number of
#     steps -- the step count is comptime and the loop has no data-
#     dependent exit -- so the stronger barrier is always reached by every
#     thread and cannot deadlock. It is a SCHEDULING cost, not a numeric
#     one, and it is not measured here: timing is out of scope for this
#     lane.
#
# DEVIATION 124 (declared in `core/block_reduce.mojo`, and it applies
# unchanged to this file). CUB's `WARP_THREADS` is a hardcoded 32; ours is
# `WARP_SIZE`. For this scan the consequence is sharper than for the
# reduce, because the fold in `ComputeWarpPrefix` is over WARPS partials
# and WARPS is `ceildiv(TPB, WARP_SIZE)`: at TPB 128 that is 4 groups on
# Apple/NVIDIA/RDNA and 2 on AMD CDNA. For an integer bin
# (`ClassificationBin`) the sum is identical on all four. For a bin with a
# FLOAT field (`RegressionBin::label_sum`, once it is carried as Float32
# here) IT IS NOT, and that is a cross-vendor numeric difference this
# repository refuses to leave implicit: a float-field bin needs a NUMERIC
# row in `checks/kernel_matrix.mojo` pinning the lane group, the way
# `replication_lanes` is already pinned to 32 there. DECLARED AND NOT
# TAKEN, because no float-field bin is wired to this scan yet -- and it is
# recorded as an OPEN item rather than a solved one.
# =========================================================================
"""

from std.gpu import thread_idx, lane_id, WARP_SIZE
from std.math import ceildiv
from std.memory import stack_allocation
from std.sys.info import size_of
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import TARGET_COLUMN, column_shared_limit


trait BlockScanElement(TrivialRegisterPassable):
    """`BinT`: `bins.cuh`'s default constructor and `operator+`.

    Nothing else is required, because `cub::BlockScan::InclusiveSum` uses
    nothing else -- it is `InclusiveScan` under `cuda::std::plus<>`
    (`block_scan.cuh`, the `InclusiveSum` family) and the identity is only
    ever produced by the caller's `BinT()` at `:271` and `:276`.
    """

    @staticmethod
    def zero() -> Self:
        """Their `HDI BinT() : <fields>(0) {}`, e.g. `bins.cuh:29`."""
        ...

    def plus(self, rhs: Self) -> Self:
        """Their `HDI BinT operator+(BinT b) const`, e.g. `bins.cuh:47-51`.

        `a.plus(b)` is their `a + b`. Their body is `b += *this; return b;`
        -- fieldwise addition, so the operand order is immaterial to the
        value for all four bins; it is preserved anyway so the two files
        diff.
        """
        ...


@fieldwise_init
struct BlockScanResult[T: BlockScanElement](TrivialRegisterPassable):
    """Their two out-parameters, `result` and `block_aggregate`
    (`builder_kernels_impl.cuh:274-275`), returned together because Mojo
    has no `&` out-parameter.
    """

    var result: Self.T
    """`InclusiveSum`'s per-thread inclusive output."""
    var block_aggregate: Self.T
    """"block-wide aggregate of all inputs", valid in EVERY thread
    (`block_scan_warp_scans.cuh:436-438`)."""


def block_inclusive_sum[
    T: BlockScanElement, TPB: Int, sabotage: Int = 0
](input: T) -> BlockScanResult[T]:
    """`cub::BlockScan<T, TPB>(temp).InclusiveSum(input, result, aggregate)`.

    The body is `BlockScanWarpScans::InclusiveScan(input, inclusive_output,
    scan_op, block_aggregate)` (`block_scan_warp_scans.cuh:449-462`):

        WarpScanT(temp_storage.warp_scan[warp_id]).InclusiveScan(input, inclusive_output, scan_op);
        T warp_prefix = ComputeWarpPrefix(scan_op, inclusive_output, block_aggregate);
        if (warp_id != 0) { inclusive_output = scan_op(warp_prefix, inclusive_output); }

    EVERY THREAD OF THE BLOCK MUST CALL THIS -- it contains barriers, and
    that is why their caller pads its loop to a multiple of `TPB` instead
    of letting short threads leave.

    `TPB` must equal the launched `block_dim.x`.

    `sabotage` is a CHECK HOOK and 0 is the only value a caller may pass;
    `sabotage == 1` drops the cross-warp prefix, which leaves every warp's
    first `WARP_SIZE` results correct and everything after them wrong --
    the failure a total-only comparison cannot see.
    """
    comptime WARPS = ceildiv(TPB, WARP_SIZE)

    # `warp_id((WARPS == 1) ? 0 : linear_tid / WARP_THREADS)`,
    # `block_scan_warp_scans.cuh:101`. A short final warp would never fire
    # the `lane_id == WARP_SIZE - 1` store below and would fold an
    # uninitialised aggregate, so the divisibility CUB assumes is asserted
    # rather than assumed. cuML launches this at TPB_DEFAULT = 128
    # (`builder_kernels_impl.cuh:34`).
    comptime assert (
        TPB % WARP_SIZE == 0
    ), "block_inclusive_sum needs TPB to be a whole number of warps"

    # Their `_TempStorage` (`block_scan_warp_scans.cuh:69-79`) is
    # `warp_aggregates[WARPS]` + `warp_scan[WARPS]` + `block_prefix`. The
    # `block_prefix` member serves only the prefix-callback form, which
    # cuML does not call, so it is not allocated. See DEVIATION 126 for
    # the missing pad on `warp_scan`.
    comptime SCAN_BYTES = (TPB + WARPS) * size_of[T]()
    comptime assert (
        SCAN_BYTES <= column_shared_limit(TARGET_COLUMN)
    ), "block_inclusive_sum temp storage exceeds the vendor threadgroup budget"

    var warp_scan = stack_allocation[
        TPB, T, address_space = AddressSpace.SHARED
    ]()
    var warp_aggregates = stack_allocation[
        WARPS, T, address_space = AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var warp_id = 0 if WARPS == 1 else tid // WARP_SIZE

    # ---- WarpScanSmem::InclusiveScan (`warp_scan_smem.cuh:175-188`) ----
    # `inclusive_output = input; ScanStep<false>(inclusive_output, op, 0)`,
    # then `ScanStep` (`:104-121`) for OFFSET = 1, 2, 4, ... < WARP_SIZE:
    #
    #     store partial; syncwarp;
    #     if (lane_id >= OFFSET) partial = scan_op(temp[lane - OFFSET], partial);
    #     syncwarp;
    #
    # NOTE THE OPERAND ORDER: the ADDEND is on the LEFT and the running
    # partial on the RIGHT (`:117`). That is the direction a prefix scan
    # needs and it is copied, not re-derived.
    var partial = input
    var offset = 1
    while offset < WARP_SIZE:
        warp_scan[unsafe_offset=tid] = partial
        barrier()
        if lane >= offset:
            var addend = warp_scan[unsafe_offset = tid - offset]
            partial = addend.plus(partial)
        barrier()
        offset *= 2

    # ---- ComputeWarpPrefix (`block_scan_warp_scans.cuh:165-195`) ----
    # "Last lane in each warp shares its warp-aggregate", `__syncthreads()`,
    # then `block_aggregate = temp_storage.warp_aggregates[0]` and
    # `ApplyWarpAggregates` unrolled over WARP = 1 .. WARPS-1:
    #
    #     if (warp_id == WARP) warp_prefix = block_aggregate;
    #     block_aggregate = scan_op(block_aggregate, warp_aggregates[WARP]);
    #
    # so `warp_prefix` for warp W is the fold of aggregates [0, W) and
    # `block_aggregate` ends as the fold of all of them, IN ASCENDING WARP
    # ORDER, in every thread. Warp 0's `warp_prefix` is left invalid by
    # CUB and is never applied.
    if lane == WARP_SIZE - 1:
        warp_aggregates[unsafe_offset=warp_id] = partial

    barrier()

    var block_aggregate = warp_aggregates[unsafe_offset=0]
    var warp_prefix = T.zero()
    for w in range(1, WARPS):
        if warp_id == w:
            warp_prefix = block_aggregate
        var addend = warp_aggregates[unsafe_offset=w]
        block_aggregate = block_aggregate.plus(addend)

    # "Apply warp prefix to our lane's partial" (`:457-461`).
    comptime if sabotage == 1:
        # SABOTAGE: no warp past warp 0 learns what came before it.
        pass
    else:
        if warp_id != 0:
            partial = warp_prefix.plus(partial)

    return BlockScanResult[T](partial, block_aggregate)


def pdf_to_cdf[
    T: BlockScanElement,
    TPB: Int,
    address_space: AddressSpace = AddressSpace.GENERIC,
    sabotage: Int = 0,
](
    histogram: MutPointer[T, MutAnyOrigin, address_space=address_space],
    n_bins: Int32,
) -> T:
    """`ML::DT::pdf_to_cdf<BinT, IdxT, TPB>`,
    `builder_kernels_impl.cuh:259-283`, transcribed line for line.

    "For every threadblock, converts a pdf-histogram to a cdf-histogram
    inplace using inclusive block-sum-scan." Returns their
    `total_aggregate`.

    `address_space` carries their `BinT*`, which at the histogram call
    site points into `extern __shared__ char smem[]` and at the
    global-memory fallback into device memory
    (`use_global_memory_histogram`, `builder_kernels_impl.cuh:294`). Both
    of their arms are therefore reachable from one instantiation here.

    `sabotage` is a CHECK HOOK and 0 is the only value a caller may pass:
    1 is forwarded to `block_inclusive_sum`, 2 drops the CHUNK carry
    (`total_aggregate`) so every chunk of `TPB` bins restarts from zero --
    which leaves the first `TPB` bins right and the rest wrong, and leaves
    a check that only looks at bin 0 or at the returned total blind.
    """
    # `BinT total_aggregate = BinT();` (`:271`).
    var total_aggregate = T.zero()

    # `for (IdxT tix = threadIdx.x; tix < ceildiv(n_bins, TPB) * TPB;
    #      tix += blockDim.x)` (`:273`). The bound is PADDED to a whole
    # number of blocks on purpose; see the module docstring.
    var padded = ceildiv(Int(n_bins), TPB) * TPB
    var tix = Int(thread_idx.x)
    while tix < padded:
        # `BinT element = tix < n_bins ? histogram[tix] : BinT();` (`:276`)
        var element = T.zero()
        if tix < Int(n_bins):
            element = histogram[unsafe_offset=tix]

        comptime inner = 1 if sabotage == 1 else 0
        var scan = block_inclusive_sum[T, TPB, inner](element)

        # `__syncthreads();` (`:278`) -- for the reuse of one TempStorage
        # across iterations, not for the store below.
        barrier()

        # `if (tix < n_bins) histogram[tix] = result + total_aggregate;`
        if tix < Int(n_bins):
            histogram[unsafe_offset=tix] = scan.result.plus(total_aggregate)

        # `total_aggregate += block_aggregate;` (`:280`)
        comptime if sabotage == 2:
            # SABOTAGE: the chunk carry never accumulates.
            pass
        else:
            total_aggregate = total_aggregate.plus(scan.block_aggregate)

        tix += TPB

    return total_aggregate
