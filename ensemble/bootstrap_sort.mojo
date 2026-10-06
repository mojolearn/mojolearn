# SPDX-License-Identifier: Apache-2.0
"""Shared exact bootstrap locality sort for RF and ET T09.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
Extracted incumbent radix schedule; nonnegative Int32 IDs, stable even passes.
"""
from std.math import ceildiv as _ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext
from core.launch_clock import log_launch_ctx
from core.segmented_sort import SORT_BLOCK, seg_add_block_carry_kernel, seg_reorder_one_bit_kernel, enqueue_seg_scan_block_sums, seg_scan_key_bit_kernel

def sort_passes_for(n_rows_bound: Int) -> Int:
    """DEVIATION 2010: the pass count, a pure host function so the check
    can hold it. Just enough bits to cover keys in `[0, n_rows_bound)`,
    rounded UP TO EVEN so the ping-pong parity leaves the answer in the
    caller's buffer (the same argument `core/segmented_sort` records for
    its fixed 32). Passes beyond the top live bit are stable identity
    partitions (every key's bit is 0), so rounding up is correct by
    construction, never just convenient."""
    var nb = 1
    while (1 << nb) < n_rows_bound:
        nb += 1
    if nb % 2 == 1:
        nb += 1
    return nb


def sort_selected_rows[
    sabotage: Int = 0
](
    ctx: DeviceContext,
    mut rows: DeviceBuffer[DType.int32],
    n: Int,
    n_rows_bound: Int,
    mut keys_scratch: DeviceBuffer[DType.uint32],
    mut offsets: DeviceBuffer[DType.int32],
    mut block_sums: DeviceBuffer[DType.int32],
) raises:
    """DEVIATION 2010's sort: LSD one-bit radix, ascending, on the first
    `n` Int32 entries of `rows` (all in `[0, n_rows_bound)`, so no
    twiddle: non-negative Int32 order raw-bit unsigned order).

    The four pass kernels are `core/segmented_sort`'s, launched with ONE
    segment -- the driver is duplicated here rather than calling
    `segmented_sort_keys_f32` because that entry twiddles float keys and
    runs all 32 bits; the kernels themselves are imported, not copied.
    STABILITY per pass is what makes the LSD loop a sort (their
    `seg_reorder_one_bit_kernel` docstring); the pass count is
    `sort_passes_for` (even), so the sorted keys end in `rows` itself.

    `sabotage` is a CHECK HOOK and 0 is the only value a caller may
    pass; 1 drops the LAST TWO passes (parity kept even), so any two
    keys differing in the top live bits keep their input order -- the
    output is provably NOT ascending on a fixture that spans those bits,
    which is how the check proves the sort is reached (the FOREST cannot
    prove it: any row order yields the same forest, which is the whole
    identity argument)."""
    if n <= 1:
        return
    var n_passes = sort_passes_for(n_rows_bound)
    comptime if sabotage == 1:
        n_passes -= 2
        if n_passes < 0:
            n_passes = 0
    var blocks_wide = _ceildiv(n, SORT_BLOCK)
    # One origin type on both ping-pong sides, so the per-pass ternary
    # below is well-typed; the kernels take MutAnyOrigin.
    var rows_u32 = (
        rows.unsafe_ptr()
        .unsafe_origin_cast[MutAnyOrigin]()
        .unsafe_bitcast[UInt32]()
    )
    var keys_u32 = keys_scratch.unsafe_ptr().unsafe_origin_cast[
        MutAnyOrigin
    ]()
    var offsets_p = offsets.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var block_sums_p = block_sums.unsafe_ptr().unsafe_origin_cast[
        MutAnyOrigin
    ]()
    var bit = 0
    while bit < n_passes:
        # Ping-pong: even bit reads `rows`, writes scratch; odd bit
        # reads scratch, writes `rows`. Even pass count -> `rows` holds
        # the answer.
        var src = rows_u32 if bit % 2 == 0 else keys_u32
        var dst = keys_u32 if bit % 2 == 0 else rows_u32
        log_launch_ctx(ctx, "rows_sort_scan_bit")
        ctx.enqueue_function[seg_scan_key_bit_kernel](
            src,
            Int32(bit),
            Int32(n),
            Int32(blocks_wide),
            offsets_p,
            block_sums_p,
            grid_dim=(blocks_wide, 1, 1),
            block_dim=(SORT_BLOCK, 1, 1),
        )
        log_launch_ctx(ctx, "rows_sort_block_sums")
        enqueue_seg_scan_block_sums(ctx, block_sums_p, n, blocks_wide, 1)
        log_launch_ctx(ctx, "rows_sort_carry")
        ctx.enqueue_function[seg_add_block_carry_kernel](
            offsets_p,
            block_sums_p,
            Int32(n),
            Int32(blocks_wide),
            grid_dim=(blocks_wide, 1, 1),
            block_dim=(SORT_BLOCK, 1, 1),
        )
        log_launch_ctx(ctx, "rows_sort_reorder")
        ctx.enqueue_function[seg_reorder_one_bit_kernel](
            src,
            offsets_p,
            Int32(bit),
            Int32(n),
            dst,
            grid_dim=(blocks_wide, 1, 1),
            block_dim=(SORT_BLOCK, 1, 1),
        )
        bit += 1
    # NO synchronize -- the pass kernels ride the in-order queue exactly
    # as the sampler's own launches do, and every reader of
    # `selected_rows_` is enqueued after this on the same queue. Mojo
    # frees at LAST USE: the buffers are the CALLER's struct fields
    # (RowSampler), alive past every launch by construction.


