# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CTR estimation order on the device, one thread per position (lane
cpu4-gbdt; seeded from `random_state` by lane cat-ctr, 2026-11-21). The
order itself is defined in `gbdt/ctrs/ctr_order.mojo`, which the host
column shares."""

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.ctrs.index_wrapper import CTR_INDEX_MASK
from gbdt.data.group_layout import device_group_layout
from gbdt.gpu_util.kernel.scan import SCAN_BLOCK, launch_scan_vector_u32
from gbdt.gpu_util.kernel.transform import launch_gather_with_mask_u32
from gbdt.ctrs.ctr_order import (
    learn_load_key,
    learn_order_row,
    learn_permutation_key,
    short_block_slot,
)

comptime CTR_ORDER_BLOCK = 256


def ctr_order_kernel(
    dst: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
    key: UInt64,
    load_key: UInt64,
    permuted: Int32,
    block_size: Int32,
    short_slot: Int32,
):
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        dst.unsafe_store(
            i,
            UInt32(
                learn_order_row(
                    i, n, Int(block_size), permuted != Int32(0), key,
                    Int(short_slot), load_key,
                )
            ),
        )
        i += stride


def launch_ctr_estimation_order(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.uint32],
    n_rows: Int,
    permutation_id: Int,
    random_seed: UInt64,
    block_size: Int = 1,
) raises:
    """Write permutation `permutation_id`'s CTR estimation order of a fit
    seeded `random_seed` into `dst[0:n_rows]` (`order[i]` = the original
    row at position `i`, the direction `TCtrBinBuilderGpu` consumes).

    `block_size` is 1 for a Plain fit. An Ordered fit passes its fold
    permutation block (`ordered_permutation_block_size`), so the order is
    the one its folds walk (`ordered_permutations`, the same
    `learn_order_row` per position): a row's CTR then reads exactly the
    rows before it in the permutation its fold models are trained on."""
    if n_rows <= 0:
        return
    var key = learn_permutation_key(random_seed, permutation_id, block_size)
    # the slot of the short last block, a host function of (rows, block,
    # key) only; unread for permutation 0
    var short_slot = -1
    if permutation_id != 0:
        short_slot = short_block_slot(n_rows, block_size, key)
    var blocks = min(
        (n_rows + CTR_ORDER_BLOCK - 1) // CTR_ORDER_BLOCK, 65535
    )
    ctx.enqueue_function[ctr_order_kernel](
        dst.unsafe_ptr(),
        Int32(n_rows),
        key,
        learn_load_key(random_seed),
        Int32(0) if permutation_id == 0 else Int32(1),
        Int32(block_size),
        Int32(short_slot),
        grid_dim=(blocks, 1, 1),
        block_dim=(CTR_ORDER_BLOCK, 1, 1),
    )


def expand_query_order_kernel(
    ordered_groups: MutPointer[UInt32, MutAnyOrigin], ends: MutPointer[UInt32, MutAnyOrigin],
    offsets: MutPointer[UInt32, MutAnyOrigin], count: Int32, rows: Int32,
    order: MutPointer[UInt32, MutAnyOrigin], ids: MutPointer[UInt32, MutAnyOrigin],
):
    var pos = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if pos >= Int(rows):
        return
    var lo = 0
    var hi = Int(count)
    while lo < hi:
        var mid = (lo + hi) // 2
        if Int(ends.unsafe_load(mid)) <= pos:
            lo = mid + 1
        else:
            hi = mid
    var group = Int(ordered_groups.unsafe_load(lo))
    var begin = 0 if lo == 0 else Int(ends.unsafe_load(lo - 1))
    var row = Int(offsets.unsafe_load(group)) + pos - begin
    order.unsafe_store(pos, UInt32(row))
    ids.unsafe_store(row, UInt32(group))


def launch_grouped_ctr_order(
    ctx: DeviceContext, mut rows_order: DeviceBuffer[DType.uint32],
    sizes: List[UInt32], n_rows: Int, seed: UInt64,
) raises -> DeviceBuffer[DType.uint32]:
    """Shuffle whole queries with the same seed as the host. Rows keep their order inside a query."""
    var groups = device_group_layout(ctx, sizes, n_rows, "group_id: group ", "group_id: the group sizes cover ")
    var n = len(sizes)
    var shuffled = ctx.enqueue_create_buffer[DType.uint32](n)
    var lengths = ctx.enqueue_create_buffer[DType.uint32](n)
    var ends = ctx.enqueue_create_buffer[DType.uint32](n)
    var scratch = ctx.enqueue_create_buffer[DType.uint32]((n + SCAN_BLOCK - 1) // SCAN_BLOCK)
    var ids = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    try:
        launch_ctr_estimation_order(ctx, shuffled, n, 0, seed)
        launch_gather_with_mask_u32(ctx, lengths, groups.sizes, shuffled, n, UInt32(0xFFFFFFFF))
        launch_scan_vector_u32(ctx, n, True, lengths, ends, scratch)
        ctx.enqueue_function[expand_query_order_kernel](
            shuffled.unsafe_ptr(), ends.unsafe_ptr(), groups.offsets.unsafe_ptr(), Int32(n), Int32(n_rows),
            rows_order.unsafe_ptr(), ids.unsafe_ptr(), grid_dim=(n_rows + CTR_ORDER_BLOCK - 1) // CTR_ORDER_BLOCK,
            block_dim=CTR_ORDER_BLOCK,
        )
        ctx.synchronize()
    except err:
        ctx.synchronize()
        _ = len(groups.offsets) + len(shuffled) + len(lengths) + len(ends) + len(scratch)
        raise err
    _ = groups^
    _ = shuffled^
    _ = lengths^
    _ = ends^
    _ = scratch^
    return ids^


def grouped_ctr_anchor_kernel(
    sorted_rows: MutPointer[UInt32, MutAnyOrigin], codes: MutPointer[UInt32, MutAnyOrigin],
    query: MutPointer[UInt32, MutAnyOrigin], anchors: MutPointer[UInt32, MutAnyOrigin], n: Int32,
):
    """The stable category sort puts every (category, query) in one run.
    Its first row's prefix excludes *all* labels from that query."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n):
        return
    var row = Int(sorted_rows.unsafe_load(i) & CTR_INDEX_MASK)
    var code = codes.unsafe_load(row)
    var q = query.unsafe_load(row)
    var lo = 0
    var hi = i
    while lo < hi:
        var mid = (lo + hi) // 2
        var other = Int(sorted_rows.unsafe_load(mid) & CTR_INDEX_MASK)
        if codes.unsafe_load(other) == code and query.unsafe_load(other) == q:
            hi = mid
        else:
            lo = mid + 1
    anchors.unsafe_store(row, sorted_rows.unsafe_load(lo) & CTR_INDEX_MASK)
