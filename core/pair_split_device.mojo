# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""cpu3-bindings (2026-10-04): a caller's interleaved (winner, loser) uint32
pairs split into two lists on the device.

The PairLogit pairs cross the Python boundary as one row-major n x 2 block;
the boosting code takes winners and losers as separate lists. The split
used to be a host loop over every pair; here the block goes up once, one
thread per pair writes both halves, and the two halves come back. Pure
moves: no arithmetic, so no bit can differ on any vendor.
"""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

comptime PAIR_SPLIT_TPB = 256


def pair_split_kernel(
    src: MutPointer[UInt32, MutAnyOrigin],
    win: MutPointer[UInt32, MutAnyOrigin],
    lose: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
):
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if q >= Int(n_in):
        return
    win.unsafe_store(q, src.unsafe_load(2 * q))
    lose.unsafe_store(q, src.unsafe_load(2 * q + 1))


def device_split_pairs(
    ctx: DeviceContext, src: MutPointer[UInt32, MutUntrackedOrigin], n: Int,
    mut winners: List[UInt32], mut losers: List[UInt32],
) raises:
    """`winners[q] = src[2q]`, `losers[q] = src[2q + 1]` for q < n."""
    winners = List[UInt32](length=max(n, 0), fill=UInt32(0))
    losers = List[UInt32](length=max(n, 0), fill=UInt32(0))
    if n <= 0:
        return
    if n > 1073741823:
        raise Error("device_split_pairs: pair count exceeds Int32")
    var d_src = ctx.enqueue_create_buffer[DType.uint32](2 * n)
    var d_win = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_lose = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_buf=d_src, src_ptr=src.unsafe_origin_cast[MutAnyOrigin]())
    ctx.enqueue_function[pair_split_kernel](
        d_src.unsafe_ptr(), d_win.unsafe_ptr(), d_lose.unsafe_ptr(), Int32(n),
        grid_dim=((n + PAIR_SPLIT_TPB - 1) // PAIR_SPLIT_TPB, 1, 1),
        block_dim=(PAIR_SPLIT_TPB, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=winners.unsafe_ptr(), src_buf=d_win)
    ctx.enqueue_copy(dst_ptr=losers.unsafe_ptr(), src_buf=d_lose)
    ctx.synchronize()
    _ = d_src^
    _ = d_win^
    _ = d_lose^
