"""DEVIATION 1902: per-leaf stat totals read THROUGH the row index.

The ridx-only twin of `gpu_util/partitions_reduce.mojo`. Under DEVIATION
1902 the stat planes stay in the order the objective wrote them for the
life of the fit -- only `row_index` is permuted at a split -- so a
partition range `[offset, offset + size)` no longer addresses its rows
directly and the per-leaf reduce must gather:

    OLD (permuted plane):   v += stats[stat * line + offset + i]
    NEW (stationary plane): v += stats[stat * line + row_index[offset + i]]

Same value bits at the same loop iteration on the same thread (the
transcription-exactness invariant on `split_points_ridx.mojo`'s banner),
same `pinned_block_sum` fold, same phase-2 finish: the per-leaf totals are
byte-for-byte the old path's. The indirection is one extra `UInt32` load
per row, on a pass that DEVIATION 1901's FAST arm already cut to twice per
tree (the root seed, which needs no gather because nothing has split yet
and is not routed here, and the end-of-tree leaf-value sweep, which does).

PHASE 1 IS A TRANSCRIPTION OF `partition_stats_partial_kernel`, one load
changed. It is a second body for that kernel and the drift surface is
priced deliberately: threading a comptime arm through the shared kernel
would change its runtime signature (the `row_index` pointer has to travel)
and re-plumb every existing enqueue in a file two other lanes call into,
against a one-changed-line loop of ~30 lines. The cross-reference is
written on BOTH bodies; an edit to either loop must visit the other.
Phase 2 and the chunk-count formula are IMPORTED, not copied -- the pinned
`partition_stats_chunks` stays the one place the grid formula lives
(IDENTITY_PATHS row 7's argument), and the finish kernel reads partials,
not stats, so it needs no gather.

FAST-ARM ONLY by routing (`ridx_only_splits_for` returns False under
IDENTICAL), never by a mode test here: this file compiles the same in both
modes and the ROUTE decides who calls it.
"""

from std.gpu import block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.device_attribute import DeviceAttribute

from gbdt.gpu_util.partitions_reduce import (
    STATS_BLOCK,
    partition_stats_chunks,
    partition_stats_finish_kernel,
)
from gbdt.targets.kernel.pointwise_targets import pinned_block_sum


def partition_stats_partial_gather_kernel(
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    line_size_in: Int32,
    partials: MutPointer[Float32, MutAnyOrigin],
    max_chunks_in: Int32,
):
    """PHASE 1 with the stat load gathered through `row_index`.

    TRANSCRIPTION of `partition_stats_partial_kernel`
    (`gpu_util/partitions_reduce.mojo`) -- grid, stripe, fold and store are
    ITS, unchanged; the single difference is the load marked below. An edit
    to that kernel's loop must be mirrored here and vice versa (the file
    banner carries why the body is doubled rather than parameterized).

    `row_index[offset + i]` is a document id in `[0, n_rows)` and
    `line_size >= n_rows`, so the gathered address stays inside the plane
    the contiguous read stayed inside.
    """
    var line_size = Int(line_size_in)
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var leaf_id = Int(leaves.unsafe_load(leaf_slot))
    var stat = Int(block_idx.z)
    var n_stats = Int(grid_dim.z)

    var offset = Int(part_offset.unsafe_load(leaf_id))
    var size = Int(part_size.unsafe_load(leaf_id))

    var chunk = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var stride = Int(grid_dim.x) * STATS_BLOCK
    var v = Float32(0.0)
    var i = chunk * STATS_BLOCK + tid
    while i < size:
        # THE ONE CHANGED LINE (DEVIATION 1902): position -> row -> stat.
        # The twin reads `stats[stat * line_size + offset + i]`.
        var row = Int(row_index.unsafe_load(offset + i))
        v += stats.unsafe_load(stat * line_size + row)
        i += stride
    # IDENTITY_PATHS row 8, the same fold as the twin's -- the fold shape
    # must not differ between the two bodies or the totals stop being
    # comparable bit for bit.
    var total = pinned_block_sum[STATS_BLOCK](v)
    if tid == 0:
        partials.unsafe_store(
            (leaf_slot * n_stats + stat) * max_chunks + chunk, total
        )


def compute_partition_stats_gather(
    ctx: DeviceContext,
    n_leaf_slots: Int,
    max_leaf_rows: Int,
    n_stats: Int,
    line_size: Int,
    mut leaves: DeviceBuffer[DType.uint32],
    mut part_offset: DeviceBuffer[DType.uint32],
    mut part_size: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut partials: DeviceBuffer[DType.float32],
    mut out_stats: DeviceBuffer[DType.float32],
    sm_count: Int = -1,
) raises:
    """`compute_partition_stats` with the gathered phase 1.

    Same signature plus `row_index`, same scratch, same grid: the chunk
    count comes from the SAME `partition_stats_chunks` both readers of the
    formula already go through, so the `partials` layout and the launch
    cannot disagree, and phase 2 is the shared `partition_stats_finish_kernel`
    -- partials in, totals out, no row data touched.
    """
    _ = max_leaf_rows
    var sm = sm_count
    if sm < 0:
        sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var max_chunks = partition_stats_chunks(sm, n_stats)

    ctx.enqueue_function[partition_stats_partial_gather_kernel](
        leaves.unsafe_ptr(),
        part_offset.unsafe_ptr(),
        part_size.unsafe_ptr(),
        stats.unsafe_ptr(),
        row_index.unsafe_ptr(),
        Int32(line_size),
        partials.unsafe_ptr(),
        Int32(max_chunks),
        grid_dim=(max_chunks, n_leaf_slots, n_stats),
        block_dim=(STATS_BLOCK, 1, 1),
    )
    ctx.enqueue_function[partition_stats_finish_kernel](
        leaves.unsafe_ptr(),
        partials.unsafe_ptr(),
        out_stats.unsafe_ptr(),
        Int32(max_chunks),
        grid_dim=(1, n_leaf_slots, n_stats),
        block_dim=(STATS_BLOCK, 1, 1),
    )
