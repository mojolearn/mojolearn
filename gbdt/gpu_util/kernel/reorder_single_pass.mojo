# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Single-pass stable one-bit partition: their LARGE-leaf sort path.

PORT OF THE DESIGN behind `SortByFlagsInLeaf`'s big arm
(`catboost/cuda/methods/greedy_subsets_searcher/kernel/split_points.cu:741`):

    if (part.Size > FastSortSize()) {          // FastSortSize() == 500000
        cub::DeviceRadixSort::SortPairs(..., 0, 1, stream);
    } else {
        SortWithoutCub(leafId, ...);
    }

`reorder_one_bit.mojo` ports the small arm and records (its `FAST_SORT_SIZE`
banner) that the big arm was a gap: we had no device sort to take, so the
block-scan partition ran at EVERY size. The key is ONE BIT, so the sort was
never the point -- a stable one-bit partition is a prefix scan plus a
scatter -- and the modern single-pass formulation of that scan is CUB's
DECOUPLED LOOKBACK, the machinery `cub::DeviceScan` itself runs:

  * NVIDIA/cccl `d10a88a9`, `cub/cub/agent/single_pass_scan_operators.cuh`:
    `ScanTileState<T, true>` packs a status word and a value into ONE machine
    word written with release stores and read with acquire loads
    (`SetPartial` `:746`, `SetInclusive` `:733`, `WaitForValid` `:763`);
    `TilePrefixCallbackOp::lookback` (`:1257-1303`) publishes PARTIAL, walks
    predecessors accumulating until an INCLUSIVE is found, then publishes
    INCLUSIVE.
  * `cub/cub/device/dispatch/dispatch_scan.cuh`: `DeviceScanInitKernel`
    zeroes the tile statuses, then ONE `DeviceScanKernel` does the whole
    scan -- the shape this file's launcher mirrors as two memsets plus one
    kernel.
  * XGBoost (dmlc/xgboost, `src/tree/gpu_hist/row_partitioner.cuh:98-201`)
    runs its row partition as exactly this: `SortPositionBatch` feeds a
    per-row go-left flag through `cub::DispatchScan` and SCATTERS DURING THE
    SCAN via a transform-output iterator (`WriteResultsFunctor`).

## Why the scatter here is NOT XGBoost's scatter

Their `WriteResultsFunctor` (`row_partitioner.cuh:118-139`) places left rows
stably from `segment.begin` but places right rows REVERSED from
`segment.end` backwards, precisely so the scatter never needs the segment's
total left count. CatBoost's partition -- and `launch_stable_partition`,
which this file must match BIT FOR BIT -- keeps BOTH sides stable, and a
stable ones side needs `total_zeros` before the first one can be placed. So
this kernel publishes the leaf total through the same packed-word protocol
and the scatter tiles wait on it; the reversed-right shortcut was READ AND
DECLINED, not missed.

## Tile ordering: rocPRIM's ticket, not CUB's blockIdx

CUB assigns `tile_idx = start_tile + blockIdx.x` (`agent_scan.cuh:412`) and
leans on NVIDIA's scheduler starting blocks in nondecreasing order. AMD's
rocPRIM implements the same lookback with an ORDERED BLOCK ID -- a global
atomic ticket -- because HIP does not promise that order. This port routes
to BOTH vendors through one row, so it takes the ticket. The ticket is also
what makes the scatter's wait sound; see the deadlock argument on the
kernel.

# =========================================================================
# DEVIATION BLOCK -- DEVIATION 1907
#
# THEIRS: above `FastSortSize()` == 500,000 rows CatBoost leaves the
# per-leaf partition to `cub::DeviceRadixSort::SortPairs` on the one flag
# bit (`split_points.cu:737-741`). Below it, `SortWithoutCub` -- already
# ported as `reorder_one_bit.mojo` and, batched per level, as
# `launch_stable_partition` in `kernel/split_points.mojo`.
#
# OURS BEFORE THIS FILE: the 3-launch block-scan partition at every size.
# Its phase 2 walks a leaf's per-chunk totals with ONE block -- `size/512`
# dependent steps -- so the partition's critical path grows linearly with
# the largest leaf, and the number of levels holding a >500k leaf grows
# with the dataset. MEASURED: the FAST speed lane's H100 ladder has the
# 5M rung 1.09x BEHIND catboost-gpu after the 1M/2M rungs flipped to wins
# (bench/results/fast_speed/2026-08-26-nvidia-HIGGS-ladder-2M.md and the
# 5M leg it cites); this path's launch-and-pass structure at large leaves
# is the recorded remaining cause.
#
# OURS NOW: on a FAST NVIDIA or AMD column, a level whose leaf bound is
# above the transcribed 500,000-row threshold runs ONE kernel (plus two
# scratch memsets): decoupled-lookback scan of the zero flags fused with
# the placement scatter, tickets ordering the tiles. Everything else --
# Apple under FAST, every column under IDENTICAL, levels at or below the
# threshold -- keeps `launch_stable_partition` BYTE FOR BYTE. The routing
# row is `reorder_single_pass_for` (`checks/kernel_matrix.mojo`).
#
# WHY THE ROW EXCLUDES APPLE: decoupled lookback spins on another block's
# published word, which is safe only where co-resident blocks have an
# INDEPENDENT FORWARD-PROGRESS guarantee. CUDA and HIP provide it (CUB and
# rocPRIM shipping this machinery is the vendors' own statement); Metal
# promises nothing of the shape, and a Metal kernel that spins does not
# get interrupted (`random_gen.mojo:105` learned this the hard way).
# DEVIATION 115a (`core/scan_by_key.mojo`) declined the lookback for that
# reason ACROSS ALL COLUMNS; this row narrows the decline to the columns
# whose vendors decline it themselves, which is exactly what a kernel-
# matrix row is for. Never an inline vendor `if`: the kernel below is
# vendor-agnostic and the ROW decides who launches it.
#
# WHAT IS BITWISE, AND WHY BY CONSTRUCTION: the scatter body below is
# `partition_place_kernel`'s chunk body, copied verbatim, and its three
# inputs are exact integers -- a chunk's `zeros_before` (the predecessor
# tile's inclusive zero count), the leaf's `n_zeros`, and the in-block
# rank from the same `prefix_sum` collective. Integer addition is
# associative, so the lookback's accumulation order cannot change any of
# them (CUB grew `lookback_stable_reduction_order` for FLOAT scans, where
# it can). Same integers into the same placement arithmetic is the same
# `gather_map` and `sorted_flags`, byte for byte: the SAME PERMUTATION,
# ones stable at `total_zeros` exactly as the 3-launch path leaves them.
# No tie-order subtlety survives, because no tie is ever broken by
# anything but the row index, in both formulations.
#
# THE PRICE:
#   * A SECOND CODE PATH for one kernel family, alive only on FAST
#     NVIDIA/AMD above the threshold -- the exact price DEVIATION 1906
#     paid for the 64-lane histogram route, carried for the same reason.
#   * The packed status word spends 2 bits on the flag, so a leaf above
#     2^30 rows cannot be described and the launcher FALLS BACK to the
#     3-launch path (`SP_MAX_DESCRIBABLE_ROWS`). CatBoost's own `ui32`
#     sizes give out at 2^32; ours gives out at a quarter of that, on a
#     path that only exists above 500k rows.
#   * The lookback walk is SERIAL ON THREAD 0, not CUB's warp-windowed
#     `ProcessWindow` (32 predecessors per step behind `__any_sync`).
#     The walk's length is bounded by the resident-block window (tickets
#     are only ever held by resident blocks), a few hundred loads worst
#     case; the window version is the recorded next step if the A/B says
#     the spin shows.
#   * Scratch is ALIASED, not grown: tile statuses live in `chunk_zeros`,
#     the packed leaf total and the ticket in `chunk_offsets[leaf, 0..1]`,
#     and the plain totals still land in `leaf_zeros` exactly as the
#     3-launch path leaves them. Same buffers, same geometry, same
#     signature -- which is what makes the wiring one line.
#
# A/B: the orchestrator measures the 5M rung intra-leg (FAST NVIDIA,
# routed vs. `launch_stable_partition` at the same commit); the row is
# where the default flips if the number says so.
#
# WIRING (one line per call site, owned by other lanes, NOT edited here):
# replace `launch_stable_partition(` with
# `launch_stable_partition_routed[HIST_BUILD_MODE == NUMERIC_IDENTICAL](`
# -- same arguments -- at
#   gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:1295
#   gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:3525
#   gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:1655
# (the single-leaf helper at greedy_search_helper.mojo:645 may take it
# too; it never exceeds one leaf so the threshold decides). Until wired,
# the path is callable and dead, and `archive/plans/UNWIRED.md`'s owner should list it
# if a level ships without the wiring.
# =========================================================================
"""

from std.atomic import Atomic, Ordering
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import broadcast as block_broadcast
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.sync import barrier

from gbdt.gpu_util.kernel.reorder_one_bit import FAST_SORT_SIZE
from gbdt.methods.greedy_subsets_searcher.kernel.split_points import (
    PARTITION_BLOCK,
    launch_stable_partition,
    split_points_grid_x,
)
from checks.kernel_matrix import TARGET_COLUMN, reorder_single_pass_for

#: `ScanTileStatus` (`single_pass_scan_operators.cuh:97-103`), renumbered so
#: that NOT-YET-PROCESSED is 0 and a memset is the whole init kernel. CUB
#: numbers INVALID 99 and reserves 0 for OOB padding tiles; the padding
#: exists for the warp-windowed lookback's negative indices, and the serial
#: walk below never reads past tile 0, so there is nothing to pad.
comptime SP_TILE_INVALID = 0
comptime SP_TILE_PARTIAL = 1
comptime SP_TILE_INCLUSIVE = 2

#: The packed word: 2 status bits above 30 value bits, so one 32-bit
#: release store publishes flag and count together -- `ScanTileState`'s
#: single-word specialization (`single_pass_scan_operators.cuh:599`), which
#: packs into whatever word one atomic access covers. 30 bits bound the
#: describable leaf; above it the launcher falls back rather than wraps.
comptime SP_STATUS_SHIFT = 30
comptime SP_VALUE_MASK = (1 << SP_STATUS_SHIFT) - 1
comptime SP_MAX_DESCRIBABLE_ROWS = 1 << SP_STATUS_SHIFT


@always_inline
def _sp_pack(status: Int32, value: Int32) -> UInt32:
    """One descriptor word: status in the top 2 bits, count below."""
    return (UInt32(Int(status)) << SP_STATUS_SHIFT) | UInt32(Int(value))


def single_pass_partition_kernel[
    sabotage: Int = 0
](
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt8, MutAnyOrigin],
    tile_status: MutPointer[UInt32, MutAnyOrigin],
    leaf_ctrl: MutPointer[UInt32, MutAnyOrigin],
    leaf_total_zeros: MutPointer[UInt32, MutAnyOrigin],
    gather_map: MutPointer[UInt32, MutAnyOrigin],
    sorted_flags: MutPointer[UInt8, MutAnyOrigin],
    max_chunks_in: Int32,
):
    """The whole partition of a level, one launch: scan tiles then scatter
    tiles, ordered by a per-leaf ticket.

    Grid y is the LEAF, as in every phase of `launch_stable_partition`.
    Within a leaf, a block loops taking tickets from
    `leaf_ctrl[leaf, 1]`: tickets `[0, n_chunks)` are SCAN tiles (count
    zeros, publish through the lookback protocol), tickets
    `[n_chunks, 2 * n_chunks)` are SCATTER tiles (wait for the leaf total
    and the predecessor's inclusive count, then place rows with
    `partition_place_kernel`'s exact arithmetic), and a ticket past
    `2 * n_chunks` ends the block.

    WHY THIS CANNOT DEADLOCK (on a column with independent forward
    progress, which is what the routing row asserts): a ticket is only
    ever held by a RESIDENT, RUNNING block, and each block finishes its
    ticket before taking another. A scan tile's lookback waits only on
    LOWER scan tickets; the lowest unfinished one has all predecessors
    finished and progresses, so the scan front always completes -- the
    induction CUB's own protocol rests on. A scatter ticket exists only
    after every scan ticket of its leaf was issued (one counter, issued in
    order), so the tiles a scatter tile waits on are held by blocks that
    progress by the same induction. Spinning scatter blocks occupy
    resources but never resources the scan front needs to EXIST -- its
    blocks already hold their tickets.

    `sabotage` is a CHECK HOOK, 0 in all real calls, same contract as
    `scan_by_key.mojo`'s: each value leaves the totals right and the
    placement wrong, the failure a count check cannot see. 1 stops the
    lookback at the first published predecessor as if PARTIAL were
    INCLUSIVE; 2 drops every scatter tile's inter-chunk carry; 3 ignores
    the published leaf total and seats the ones at the leaf base as if it
    held no zeros.
    """
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var leaf_id = Int(leaves.unsafe_load(leaf_slot))
    var offset = Int(part_offset.unsafe_load(leaf_id))
    var size = Int(part_size.unsafe_load(leaf_id))
    var n_chunks = (size + PARTITION_BLOCK - 1) // PARTITION_BLOCK
    var tid = Int(thread_idx.x)

    var status_base = leaf_slot * max_chunks
    # `leaf_ctrl[leaf, 0]` is the packed leaf total, `[leaf, 1]` the ticket.
    var ctrl_base = leaf_slot * max_chunks

    var s_ticket = stack_allocation[
        1, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var s_zeros_before = stack_allocation[
        1, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var s_total = stack_allocation[
        1, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    while True:
        # Guards the shared slots against the previous iteration's readers,
        # and is the first barrier of the loop so every thread reaches it.
        barrier()
        if tid == 0:
            var t32 = Atomic.fetch_add[ordering = Ordering.RELAXED](
                leaf_ctrl.unsafe_offset(ctrl_base + 1), UInt32(1)
            )
            s_ticket[unsafe_offset=0] = Int32(Int(t32))
        barrier()
        var t = Int(s_ticket[unsafe_offset=0])

        if t >= 2 * n_chunks:
            # An EMPTY leaf still gets its total written, as phase 2 of the
            # 3-launch path writes it (`partition_scan_chunks_kernel`'s
            # `carry` is 0 there). Ticket 0 exists exactly once per leaf.
            if n_chunks == 0 and t == 0 and tid == 0:
                leaf_total_zeros.unsafe_store(leaf_slot, UInt32(0))
            return

        if t < n_chunks:
            # ---- SCAN TILE `t`: count zeros, publish, look back --------
            # The count is `partition_count_chunks_kernel`'s, verbatim: a
            # thread past `size` contributes 0, so the last thread's
            # inclusive prefix is the tile total whether or not the tile
            # is full.
            var i = t * PARTITION_BLOCK + tid
            var v = Int32(0)
            if i < size:
                if flags.unsafe_load(offset + i) == UInt8(0):
                    v = Int32(1)
            var inc = block_prefix_sum[
                block_size=PARTITION_BLOCK, exclusive=False
            ](v)
            var agg = block_broadcast[block_size=PARTITION_BLOCK](
                inc, src_thread = PARTITION_BLOCK - 1
            )

            if tid == 0:
                # `TilePrefixCallbackOp::lookback`
                # (`single_pass_scan_operators.cuh:1257-1303`): publish
                # PARTIAL, accumulate predecessors until an INCLUSIVE,
                # publish INCLUSIVE. Tile 0 has no predecessor and
                # publishes INCLUSIVE at once, which is `DeviceScanKernel`
                # treating tile 0 as its own prefix.
                var exclusive = Int32(0)
                if t > 0:
                    Atomic.store[ordering = Ordering.RELEASE](
                        tile_status.unsafe_offset(status_base + t),
                        _sp_pack(Int32(SP_TILE_PARTIAL), agg),
                    )
                    var p = t - 1
                    while True:
                        var w = Atomic.load[ordering = Ordering.ACQUIRE](
                            tile_status.unsafe_offset(status_base + p)
                        )
                        var st = Int(w >> SP_STATUS_SHIFT)
                        if st == SP_TILE_INVALID:
                            continue
                        exclusive += Int32(Int(w & SP_VALUE_MASK))
                        comptime if sabotage == 1:
                            # SABOTAGE: the first published predecessor
                            # ends the walk as if it were INCLUSIVE, so a
                            # PARTIAL chain loses everything before it.
                            break
                        if st == SP_TILE_INCLUSIVE:
                            break
                        p -= 1
                var inclusive = exclusive + agg
                Atomic.store[ordering = Ordering.RELEASE](
                    tile_status.unsafe_offset(status_base + t),
                    _sp_pack(Int32(SP_TILE_INCLUSIVE), inclusive),
                )
                if t == n_chunks - 1:
                    # The leaf total: the plain word the 3-launch path
                    # leaves in `leaf_zeros`, and the packed word the
                    # scatter tiles wait on.
                    leaf_total_zeros.unsafe_store(
                        leaf_slot, UInt32(Int(inclusive))
                    )
                    Atomic.store[ordering = Ordering.RELEASE](
                        leaf_ctrl.unsafe_offset(ctrl_base),
                        _sp_pack(Int32(SP_TILE_INCLUSIVE), inclusive),
                    )
        else:
            # ---- SCATTER TILE `s`: place rows ---------------------------
            var s = t - n_chunks
            if tid == 0:
                var total = Int32(0)
                while True:
                    var w = Atomic.load[ordering = Ordering.ACQUIRE](
                        leaf_ctrl.unsafe_offset(ctrl_base)
                    )
                    if Int(w >> SP_STATUS_SHIFT) == SP_TILE_INCLUSIVE:
                        total = Int32(Int(w & SP_VALUE_MASK))
                        break
                comptime if sabotage == 3:
                    # SABOTAGE: the ones are seated at the leaf base as if
                    # it held no zeros; the zeros side stays right, every
                    # write stays inside the leaf, and the two sides
                    # collide -- placement wrong, totals untouched.
                    total = Int32(0)
                var zb = Int32(0)
                if s > 0:
                    # `chunk_zero_offsets[s]` of the 3-launch path is the
                    # exclusive scan of per-chunk zero counts, which IS the
                    # predecessor tile's inclusive count: same integer.
                    while True:
                        var w = Atomic.load[ordering = Ordering.ACQUIRE](
                            tile_status.unsafe_offset(status_base + s - 1)
                        )
                        if Int(w >> SP_STATUS_SHIFT) == SP_TILE_INCLUSIVE:
                            zb = Int32(Int(w & SP_VALUE_MASK))
                            break
                comptime if sabotage == 2:
                    # SABOTAGE: no scatter tile carries the chunks before
                    # it. Totals unchanged, placement wrong past chunk 0.
                    zb = Int32(0)
                s_total[unsafe_offset=0] = total
                s_zeros_before[unsafe_offset=0] = zb
            barrier()
            var n_zeros = Int(s_total[unsafe_offset=0])
            var zeros_before = Int(s_zeros_before[unsafe_offset=0])

            # `partition_place_kernel`'s chunk body, verbatim -- including
            # the `rank_one == tid - rank_zero` subtraction its docstring
            # derives -- so the destination arithmetic cannot drift from
            # the path it must match byte for byte.
            var i = s * PARTITION_BLOCK + tid
            var elems_before = s * PARTITION_BLOCK
            if elems_before > size:
                elems_before = size
            var ones_before = elems_before - zeros_before

            var in_range = i < size
            var is_zero = Int32(0)
            if in_range:
                if flags.unsafe_load(offset + i) == UInt8(0):
                    is_zero = Int32(1)

            var inc_zero = block_prefix_sum[
                block_size=PARTITION_BLOCK, exclusive=False
            ](is_zero)
            var rank_zero = Int(inc_zero) - Int(is_zero)
            var rank_one = tid - rank_zero

            if in_range:
                var dst = 0
                if is_zero == Int32(1):
                    dst = zeros_before + rank_zero
                else:
                    dst = n_zeros + ones_before + rank_one
                gather_map.unsafe_store(offset + dst, UInt32(i))
                sorted_flags.unsafe_store(
                    offset + dst,
                    UInt8(0) if is_zero == Int32(1) else UInt8(1),
                )


def launch_single_pass_partition[
    sabotage: Int = 0
](
    ctx: DeviceContext,
    n_leaf_slots: Int,
    max_leaf_rows: Int,
    mut leaves: DeviceBuffer[DType.uint32],
    mut part_offset: DeviceBuffer[DType.uint32],
    mut part_size: DeviceBuffer[DType.uint32],
    mut flags: DeviceBuffer[DType.uint8],
    mut chunk_zeros: DeviceBuffer[DType.uint32],
    mut chunk_offsets: DeviceBuffer[DType.uint32],
    mut leaf_zeros: DeviceBuffer[DType.uint32],
    mut gather_map: DeviceBuffer[DType.uint32],
    mut sorted_flags: DeviceBuffer[DType.uint8],
    sm_count: Int = -1,
) raises:
    """`DispatchScan::Invoke` (`dispatch_scan.cuh`): init the tile
    descriptors, then one kernel. The init is two memsets because
    `SP_TILE_INVALID` is 0 by construction.

    SAME SIGNATURE AND SAME SCRATCH as `launch_stable_partition`, reused
    under different names: `chunk_zeros` holds the packed tile
    descriptors, `chunk_offsets[leaf, 0]` the packed leaf total and
    `chunk_offsets[leaf, 1]` the ticket, and `leaf_zeros` receives the
    plain per-leaf totals exactly as the 3-launch path leaves them. The
    memsets cover the WHOLE buffers, which may exceed this level's
    `max_chunks` geometry; extra words are never read.

    `sm_count > 0` machine-sizes the grid with the same
    `split_points_grid_x` target the 3-launch path takes; blocks loop on
    tickets, so an undersized grid still covers every tile and an
    oversized one exits on its first ticket.
    """
    if n_leaf_slots <= 0:
        return
    var max_chunks = (max_leaf_rows + PARTITION_BLOCK - 1) // PARTITION_BLOCK
    if max_chunks < 1:
        max_chunks = 1
    if max_chunks < 2:
        # The ticket lives in `chunk_offsets[leaf, 1]`; a one-chunk
        # geometry has no slot for it. Unreachable through the routed
        # entry (the threshold is 500k rows) and refused rather than
        # silently misaddressed for a direct caller.
        raise Error(
            "single-pass partition needs max_chunks >= 2 for its"
            " control words; call launch_stable_partition instead"
        )
    if max_leaf_rows >= SP_MAX_DESCRIBABLE_ROWS:
        raise Error(
            "single-pass partition packs counts into 30 bits; a leaf"
            " bound at or above 2^30 rows must take"
            " launch_stable_partition"
        )

    ctx.enqueue_memset(chunk_zeros, UInt32(0))
    ctx.enqueue_memset(chunk_offsets, UInt32(0))

    var chunk_grid = max_chunks
    if sm_count > 0:
        var target = split_points_grid_x(n_leaf_slots, sm_count)
        if target < chunk_grid:
            chunk_grid = target

    comptime kernel = single_pass_partition_kernel[sabotage]
    ctx.enqueue_function[kernel](
        leaves.unsafe_ptr(),
        part_offset.unsafe_ptr(),
        part_size.unsafe_ptr(),
        flags.unsafe_ptr(),
        chunk_zeros.unsafe_ptr(),
        chunk_offsets.unsafe_ptr(),
        leaf_zeros.unsafe_ptr(),
        gather_map.unsafe_ptr(),
        sorted_flags.unsafe_ptr(),
        Int32(max_chunks),
        grid_dim=(chunk_grid, n_leaf_slots, 1),
        block_dim=(PARTITION_BLOCK, 1, 1),
    )


def launch_stable_partition_routed[
    identical: Bool
](
    ctx: DeviceContext,
    n_leaf_slots: Int,
    max_leaf_rows: Int,
    mut leaves: DeviceBuffer[DType.uint32],
    mut part_offset: DeviceBuffer[DType.uint32],
    mut part_size: DeviceBuffer[DType.uint32],
    mut flags: DeviceBuffer[DType.uint8],
    mut chunk_zeros: DeviceBuffer[DType.uint32],
    mut chunk_offsets: DeviceBuffer[DType.uint32],
    mut leaf_zeros: DeviceBuffer[DType.uint32],
    mut gather_map: DeviceBuffer[DType.uint32],
    mut sorted_flags: DeviceBuffer[DType.uint8],
    sm_count: Int = -1,
) raises:
    """`SortByFlagsInLeaf`'s dispatch (`split_points.cu:737-741`),
    transcribed: above `FastSortSize()` take the single-pass machinery,
    otherwise the path that already ran. Drop-in for
    `launch_stable_partition` -- same arguments -- with `identical` bound
    to `HIST_BUILD_MODE == NUMERIC_IDENTICAL` at the call site, the same
    binding every DEVIATION 1906 consultation makes.

    Theirs branches PER LEAF, from the host, because their partition sizes
    live on the host; ours are device-resident, so the branch is per LEVEL
    on the host-known leaf bound. A routed level's sub-threshold leaves
    take the single-pass kernel too -- same permutation by construction,
    see the file banner -- and a level whose BOUND is under the threshold
    keeps the 3-launch path byte for byte, which preserves the gate the
    orchestrator's Apple byte-compare runs.
    """
    comptime single_pass = reorder_single_pass_for[TARGET_COLUMN, identical]()

    comptime if single_pass:
        if (
            max_leaf_rows > FAST_SORT_SIZE
            and max_leaf_rows < SP_MAX_DESCRIBABLE_ROWS
        ):
            launch_single_pass_partition(
                ctx, n_leaf_slots, max_leaf_rows, leaves, part_offset,
                part_size, flags, chunk_zeros, chunk_offsets, leaf_zeros,
                gather_map, sorted_flags, sm_count=sm_count,
            )
            return

    launch_stable_partition(
        ctx, n_leaf_slots, max_leaf_rows, leaves, part_offset, part_size,
        flags, chunk_zeros, chunk_offsets, leaf_zeros, gather_map,
        sorted_flags, sm_count=sm_count,
    )
