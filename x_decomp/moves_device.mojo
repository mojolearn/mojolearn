# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`move` (x_decomp/moves.mojo) on device-resident matrices (lane
apple-fast-py2mojo-decomp, 2026-10-03): one thread per moved element, the
same source and destination index functions as the host loop, so the copies
are the host's exactly. GPU binding only."""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from x_decomp.cells import F32Ptr
from x_decomp.device import TPB, _blocks
from x_decomp.moves import move_dst, move_src


def move_kernel(
    op: Int32, src: F32Ptr, idx: F32Ptr, dst: F32Ptr, count: Int32, a1: Int32, a2: Int32, a3: Int32,
    ist: Int32, ioff: Int32,
):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(count):
        var s = move_src(Int(op), t, idx, Int(a1), Int(a2), Int(ist), Int(ioff))
        var d = move_dst(Int(op), t, Int(a1), Int(a2), Int(a3))
        if s >= 0:
            dst.unsafe_store(d, src.unsafe_load(s))
        else:
            dst.unsafe_store(d, Float32(0))


def launch_move(
    ctx: DeviceContext, op: Int, src: F32Ptr, idx: F32Ptr, dst: F32Ptr, count: Int, a1: Int, a2: Int, a3: Int,
    ist: Int, ioff: Int,
) raises:
    if count <= 0:
        return
    ctx.enqueue_function[move_kernel](
        Int32(op), src, idx, dst, Int32(count), Int32(a1), Int32(a2), Int32(a3), Int32(ist), Int32(ioff),
        grid_dim=_blocks(count), block_dim=TPB,
    )
