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


# Stable total-order merge passes over device-generated uint64 draw keys.
# Every position scatters to its unique rank in one merged pair; kernel
# boundaries preserve visibility. No host key/index materialization.
def permutation_positions_kernel(rows: MutPointer[Int32, MutAnyOrigin], n_in: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        rows[i] = Int32(i)

def permutation_merge_kernel(
    dst: MutPointer[Int32, MutAnyOrigin], src: MutPointer[Int32, MutAnyOrigin],
    keys: MutPointer[UInt64, MutAnyOrigin], n_in: Int32, width_in: Int32,
):
    var n = Int(n_in)
    var width = Int(width_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var base = (i // (2 * width)) * (2 * width)
    var mid = min(base + width, n)
    var end = min(base + 2 * width, n)
    var other_lo = mid if i < mid else base
    var other_hi = end if i < mid else mid
    var own_lo = base if i < mid else mid
    var position = Int(src[i])
    var key = keys[position]
    var lo = other_lo
    var hi = other_hi
    while lo < hi:
        var probe = (lo + hi) // 2
        var opponent = Int(src[probe])
        var opposite = keys[opponent]
        if opposite < key or (opposite == key and opponent < position):
            lo = probe + 1
        else:
            hi = probe
    dst[base + (i - own_lo) + (lo - other_lo)] = Int32(position)


# C11 deterministic counter draws at the output consumer, preserving the
# exact utils_draw_kernel index (key, replicate=0, output row). No index
# storage, no pointer-based cache and no mutable generator state.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
from resample.checks.index_map import draw_row_index, key_join


def classical_draw_gather_kernel(
    dst: MutPointer[Float32, MutAnyOrigin], src: MutPointer[Float32, MutAnyOrigin],
    key_lo: Int32, key_hi: Int32, n_in: Int32, count_in: Int32, d_in: Int32,
):
    var o = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    var d = Int(d_in)
    if o < Int(count_in)*d:
        var row = o//d
        var selected = Int(draw_row_index(key_join(key_lo, key_hi), 0, row, n_in))
        dst.unsafe_store(o, src.unsafe_load(selected*d+o%d))
