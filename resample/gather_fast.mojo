# SPDX-License-Identifier: Apache-2.0
"""Opt-in Apple FAST row gather, recovered independently of old resample bundle."""
from std.gpu import block_idx, block_dim, thread_idx


def gather_rows_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin], count_in: Int32, d_in: Int32,
):
    var count = Int(count_in)
    var d = Int(d_in)
    var o = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if o < count * d:
        var r = o // d
        dst.unsafe_store(o, src.unsafe_load(Int(rows.unsafe_load(r)) * d + o % d))

# A 32-column by 8-row output tile reads each selected row index once.
# Adjacent lanes read contiguous features; width tails are masked. This is
# a bandwidth/reuse experiment, never an input-width dispatch shortcut.
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier


def gather_rows_tiled_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin], count_in: Int32, d_in: Int32,
):
    var tid = Int(thread_idx.x)
    var local_row = tid // 32
    var row = Int(block_idx.y) * 8 + local_row
    var col = Int(block_idx.x) * 32 + tid % 32
    var selected = stack_allocation[8, Int32, address_space=AddressSpace.SHARED]()
    if tid < 8:
        var output_row = Int(block_idx.y) * 8 + tid
        selected[tid] = rows.unsafe_load(output_row) if output_row < Int(count_in) else Int32(0)
    barrier()
    if row < Int(count_in) and col < Int(d_in):
        dst.unsafe_store(row * Int(d_in) + col, src.unsafe_load(Int(selected[local_row]) * Int(d_in) + col))
