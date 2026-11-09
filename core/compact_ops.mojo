# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Stream compaction with a device-side count as an executor step op (lane
n7-device-linalg-ops, 2026-11-08; docs/plans/EXECUTOR_N7_OPS.md). Host twin:
core/compact_ops_host.mojo.

`cub::DeviceSelect::Flagged`, ORDER-PRESERVING: out[k] is the k-th flagged
element in index order and count[0] the number flagged, both on the
device. The scan is the three-pass one svm/checks/device_select.mojo
`SelectScratch` runs (gbdt/gpu_util/kernel/reorder_one_bit.mojo: block scan
of the flag bytes, the block sums' scan, the carry), then the count and a
scatter. Integer work only: a pure function of the flag bytes on every
column. Nothing comes back: a consumer sizes its grids by the capacity and
reads count[0] on the device (`compact_count_bound`), so the SMO working
set's drain per outer iteration goes away.

Flags are bytes 0 / 1 (`compact_flag_*_kernel` build them). Gate: word 0 of
`gate` (the loop's done word, or `step_gate_open_word`); out and count are
written only while it is open, the scan writes only the scratch.
"""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.step_gate import StepI32
from gbdt.gpu_util.kernel.reorder_one_bit import (
    REORDER_BLOCK,
    add_block_carry_kernel,
    block_scan_flags_kernel,
    scan_block_sums_kernel,
)

comptime CO_TPB = 256

comptime _F = MutPointer[Float32, MutAnyOrigin]
comptime _B = MutPointer[UInt8, MutAnyOrigin]


@always_inline
def _co_blocks(count: Int) -> Int:
    return (count + CO_TPB - 1) // CO_TPB if count > 0 else 1


@always_inline
def compact_count_bound(count: StepI32, t: Int) -> Bool:
    """Inside a consumer kernel launched over the capacity: thread t has an
    element (t < count[0])."""
    return t < Int(count.unsafe_load(0))


def compact_count_kernel(gate: StepI32, count: StepI32, offsets: StepI32, flags: _B, n_in: Int32):
    """count[0] = offsets[n - 1] + flags[n - 1] (the thread at n - 1), while
    the gate is open."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i != n - 1:
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    count.unsafe_store(0, offsets.unsafe_load(i) + Int32(Int(flags.unsafe_load(i)) & 1))


def compact_zero_count_kernel(gate: StepI32, count: StepI32):
    """count[0] = 0 for an empty input, while the gate is open."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    count.unsafe_store(0, Int32(0))


def compact_index_kernel(gate: StepI32, flags: _B, offsets: StepI32, dst: StepI32, n_in: Int32):
    """dst[offsets[i]] = i for flagged i, while the gate is open."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    if flags.unsafe_load(i) != UInt8(0):
        dst.unsafe_store(Int(offsets.unsafe_load(i)), Int32(i))


def compact_i32_kernel(gate: StepI32, src: StepI32, flags: _B, offsets: StepI32, dst: StepI32, n_in: Int32):
    """dst[offsets[i]] = src[i] for flagged i, while the gate is open."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    if flags.unsafe_load(i) != UInt8(0):
        dst.unsafe_store(Int(offsets.unsafe_load(i)), src.unsafe_load(i))


def compact_f32_kernel(gate: StepI32, src: _F, flags: _B, offsets: StepI32, dst: _F, n_in: Int32):
    """dst[offsets[i]] = src[i] for flagged i, while the gate is open."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    if gate.unsafe_load(0) != Int32(0):
        return
    if flags.unsafe_load(i) != UInt8(0):
        dst.unsafe_store(Int(offsets.unsafe_load(i)), src.unsafe_load(i))


def compact_flag_nonzero_kernel(src: _F, flags: _B, n_in: Int32):
    """flags[i] = 1 where src[i] != 0 (NaN counts as nonzero), else 0."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    flags.unsafe_store(i, UInt8(1) if src.unsafe_load(i) != Float32(0.0) else UInt8(0))


struct CompactScratch(Movable):
    """The scan's exclusive offsets and block sums for up to `capacity`
    elements. Allocate once per fit, outside the loop."""

    var offsets: DeviceBuffer[DType.int32]
    var block_sums: DeviceBuffer[DType.int32]
    var capacity: Int

    def __init__(out self, ctx: DeviceContext, capacity: Int) raises:
        var cap = max(capacity, 1)
        self.capacity = cap
        self.offsets = ctx.enqueue_create_buffer[DType.int32](cap)
        self.block_sums = ctx.enqueue_create_buffer[DType.int32]((cap + REORDER_BLOCK - 1) // REORDER_BLOCK)


def compact_scan_enqueue(
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut flags: DeviceBuffer[DType.uint8],
    n: Int,
    mut scratch: CompactScratch,
    mut count: DeviceBuffer[DType.int32],
) raises:
    """The exclusive prefix sum of flags[0, n) into scratch.offsets and the
    count into count[0] (gated). Enqueued only. svm/checks/device_select.mojo
    `SelectScratch.scan` runs this, then its one read."""
    if n > scratch.capacity:
        raise Error("compact_scan_enqueue: n=" + String(n) + " exceeds the scratch capacity " + String(scratch.capacity))
    var gp = StepI32(unsafe_from_address=Int(gate.unsafe_ptr()))
    if n <= 0:
        ctx.enqueue_function[compact_zero_count_kernel](  # small-launch(one word): an empty compaction's count
            gp, count.unsafe_ptr(), grid_dim=(1, 1, 1), block_dim=(1, 1, 1)
        )
        return
    var n_blocks = (n + REORDER_BLOCK - 1) // REORDER_BLOCK
    ctx.enqueue_function[block_scan_flags_kernel](
        flags.unsafe_ptr(), Int32(0), Int32(n), scratch.offsets.unsafe_ptr(), scratch.block_sums.unsafe_ptr(),
        grid_dim=n_blocks, block_dim=REORDER_BLOCK,
    )
    ctx.enqueue_function[scan_block_sums_kernel](  # small-launch(n / 512 block sums): the serial scan of the block totals
        scratch.block_sums.unsafe_ptr(), Int32(n_blocks), grid_dim=1, block_dim=1,
    )
    ctx.enqueue_function[add_block_carry_kernel](
        scratch.offsets.unsafe_ptr(), scratch.block_sums.unsafe_ptr(), Int32(n),
        grid_dim=n_blocks, block_dim=REORDER_BLOCK,
    )
    ctx.enqueue_function[compact_count_kernel](
        gp, count.unsafe_ptr(), scratch.offsets.unsafe_ptr(), flags.unsafe_ptr(), Int32(n),
        grid_dim=_co_blocks(n), block_dim=CO_TPB,
    )


def compact_op_indices(
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut flags: DeviceBuffer[DType.uint8],
    n: Int,
    mut scratch: CompactScratch,
    mut dst: DeviceBuffer[DType.int32],
    mut count: DeviceBuffer[DType.int32],
) raises:
    """dst[0, count) = the flagged indices of [0, n) ascending; count[0] on
    the device. Enqueued only."""
    compact_scan_enqueue(ctx, gate, flags, n, scratch, count)
    if n > 0:
        ctx.enqueue_function[compact_index_kernel](
            StepI32(unsafe_from_address=Int(gate.unsafe_ptr())), flags.unsafe_ptr(), scratch.offsets.unsafe_ptr(),
            dst.unsafe_ptr(), Int32(n), grid_dim=_co_blocks(n), block_dim=CO_TPB,
        )


def compact_op_i32(
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut src: DeviceBuffer[DType.int32],
    mut flags: DeviceBuffer[DType.uint8],
    n: Int,
    mut scratch: CompactScratch,
    mut dst: DeviceBuffer[DType.int32],
    mut count: DeviceBuffer[DType.int32],
) raises:
    """dst[0, count) = src's flagged elements in index order. Enqueued only."""
    compact_scan_enqueue(ctx, gate, flags, n, scratch, count)
    if n > 0:
        ctx.enqueue_function[compact_i32_kernel](
            StepI32(unsafe_from_address=Int(gate.unsafe_ptr())), src.unsafe_ptr(), flags.unsafe_ptr(),
            scratch.offsets.unsafe_ptr(), dst.unsafe_ptr(), Int32(n), grid_dim=_co_blocks(n), block_dim=CO_TPB,
        )


def compact_op_f32(
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut src: DeviceBuffer[DType.float32],
    mut flags: DeviceBuffer[DType.uint8],
    n: Int,
    mut scratch: CompactScratch,
    mut dst: DeviceBuffer[DType.float32],
    mut count: DeviceBuffer[DType.int32],
) raises:
    """dst[0, count) = src's flagged elements in index order. Enqueued only."""
    compact_scan_enqueue(ctx, gate, flags, n, scratch, count)
    if n > 0:
        ctx.enqueue_function[compact_f32_kernel](
            StepI32(unsafe_from_address=Int(gate.unsafe_ptr())), src.unsafe_ptr(), flags.unsafe_ptr(),
            scratch.offsets.unsafe_ptr(), dst.unsafe_ptr(), Int32(n), grid_dim=_co_blocks(n), block_dim=CO_TPB,
        )
