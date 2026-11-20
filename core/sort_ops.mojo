# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device sort as an executor step op (lane n7-device-linalg-ops,
2026-11-08; docs/plans/EXECUTOR_N7_OPS.md). Host twin: core/sort_ops_host.mojo.

`sort_op_f32` returns the ORDER of n float32 keys: order[r] = the input
index of the r-th key. The order is the unique stable order of the keys'
sortable bits (`sort_key_bits`: the float order with -0.0 before +0.0,
negative-signed NaNs first and positive-signed NaNs last; descending
inverts the bits, so ties still keep input order). It is computed by
`core/fast_radix_sort.mojo`'s stable LSD radix sort of (key bits, index)
pairs, the sort the SVM working set takes on every GPU (12 launches for
the one-bit sort's 128, svm/impl/workingset.mojo FAST_WS_SORT). A stable
sort by the full key is ONE permutation, so every column, and the host twin
(the same tiles and passes), returns the same words. Integer work only.

`sort_op_gather_rows_f32` applies an order to the rows of a matrix (the
eigen solves' reorder, a sorted copy of the keys with cols = 1).

Gate: word 0 of `gate` (the loop's done word, or `step_gate_open_word`).
The caller's outputs are written only while it is open; the radix passes
run on the op's own scratch either way (a stable re-sort of stale scratch
changes nothing the caller sees). Nothing is read back or drained.
"""
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from core.fast_radix_sort import fast_radix_sort_pairs_u32, frs_counts_len
from core.step_gate import StepI32

comptime SO_TPB = 256

comptime _F = MutPointer[Float32, MutAnyOrigin]
comptime _U = MutPointer[UInt32, MutAnyOrigin]


@always_inline
def _so_blocks(count: Int) -> Int:
    return (count + SO_TPB - 1) // SO_TPB if count > 0 else 1


@always_inline
def sort_key_bits(v: Float32, descending: Bool) -> UInt32:
    """The sortable key of a float32: unsigned order is ascending float
    order (svm/checks/device_select.mojo `twiddle_keys_kernel`'s map:
    negatives bit-inverted, non-negatives with the sign bit set); inverted
    for a descending order."""
    var bits = bitcast[DType.uint32](v)
    var key: UInt32
    if bits & UInt32(0x80000000) != UInt32(0):
        key = ~bits
    else:
        key = bits | UInt32(0x80000000)
    if descending:
        key = ~key
    return key


def so_key_kernel(gate: StepI32, f: _F, keys: _U, vals: _U, n_in: Int32, desc_in: Int32):
    """keys[i] = sort_key_bits(f[i]), vals[i] = i, while the gate is open."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    keys.unsafe_store(i, sort_key_bits(f.unsafe_load(i), desc_in != Int32(0)))
    vals.unsafe_store(i, UInt32(i))


def so_order_kernel(gate: StepI32, vals: _U, order: StepI32, n_in: Int32):
    """order[r] = the sorted pairs' index r, while the gate is open."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(n_in):
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    order.unsafe_store(r, Int32(Int(vals.unsafe_load(r))))


def so_gather_rows_kernel(gate: StepI32, src: _F, order: StepI32, dst: _F, rows_in: Int32, cols_in: Int32):
    """dst[r, c] = src[order[r], c] while the gate is open (a bit copy)."""
    var cols = Int(cols_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(rows_in) * cols:
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    var r = t // cols
    var c = t - r * cols
    dst.unsafe_store(t, src.unsafe_load(Int(order.unsafe_load(r)) * cols + c))


struct SortScratch(Movable):
    """The radix sort's pairs, ping-pong buffers and digit counts for up to
    `capacity` keys. Allocate once per fit, outside the loop."""

    var keys: DeviceBuffer[DType.uint32]
    var vals: DeviceBuffer[DType.uint32]
    var tkeys: DeviceBuffer[DType.uint32]
    var tvals: DeviceBuffer[DType.uint32]
    var counts: DeviceBuffer[DType.int32]
    var capacity: Int

    def __init__(out self, ctx: DeviceContext, capacity: Int) raises:
        var cap = max(capacity, 1)
        self.capacity = cap
        self.keys = ctx.enqueue_create_buffer[DType.uint32](cap)
        self.vals = ctx.enqueue_create_buffer[DType.uint32](cap)
        self.tkeys = ctx.enqueue_create_buffer[DType.uint32](cap)
        self.tvals = ctx.enqueue_create_buffer[DType.uint32](cap)
        self.counts = ctx.enqueue_create_buffer[DType.int32](frs_counts_len(cap))


def sort_op_f32(
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut keys: DeviceBuffer[DType.float32],
    mut order: DeviceBuffer[DType.int32],
    n: Int,
    descending: Bool,
    mut scratch: SortScratch,
) raises:
    """order (n Int32) = the stable order of keys[0, n) (read only),
    ascending or descending. Enqueued only."""
    if n <= 0:
        return
    if n > scratch.capacity or len(keys) < n or len(order) < n:
        raise Error("sort_op_f32: n=" + String(n) + " exceeds the scratch capacity " + String(scratch.capacity) + " or a buffer")
    var gp = StepI32(unsafe_from_address=Int(gate.unsafe_ptr()))
    ctx.enqueue_function[so_key_kernel](
        gp, keys.unsafe_ptr(), scratch.keys.unsafe_ptr(), scratch.vals.unsafe_ptr(), Int32(n),
        Int32(1) if descending else Int32(0), grid_dim=_so_blocks(n), block_dim=SO_TPB,
    )
    fast_radix_sort_pairs_u32(ctx, n, scratch.keys, scratch.vals, scratch.tkeys, scratch.tvals, scratch.counts)
    ctx.enqueue_function[so_order_kernel](
        gp, scratch.vals.unsafe_ptr(), order.unsafe_ptr(), Int32(n), grid_dim=_so_blocks(n), block_dim=SO_TPB,
    )


def sort_op_gather_rows_f32(
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut src: DeviceBuffer[DType.float32],
    mut order: DeviceBuffer[DType.int32],
    mut dst: DeviceBuffer[DType.float32],
    rows: Int,
    cols: Int,
) raises:
    """dst (rows x cols) = the rows of src in `order` (dst must not alias
    src). Enqueued only."""
    if rows <= 0 or cols <= 0:
        return
    if len(order) < rows or len(dst) < rows * cols:
        raise Error("sort_op_gather_rows_f32: a buffer is shorter than " + String(rows) + " x " + String(cols))
    ctx.enqueue_function[so_gather_rows_kernel](
        StepI32(unsafe_from_address=Int(gate.unsafe_ptr())), src.unsafe_ptr(), order.unsafe_ptr(), dst.unsafe_ptr(),
        Int32(rows), Int32(cols), grid_dim=_so_blocks(rows * cols), block_dim=SO_TPB,
    )
