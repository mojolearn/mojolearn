# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 2514, step 1 (2026-09-11): the device refusal scans, in one
place any lane may import.

`NONFINITE_NONE`, `nonfinite_partial_kernel` and `device_first_nonfinite`
MOVED here from `transformer/impl/llama/fused_attention.mojo` (which
re-imports them, so every existing caller resolves as before) because the
training lane must not import the transformer implementation
(`DESIGN_lm_device_owned_step_2026-09-11.md` section 9 item 2). The
kernel, its geometry (256 threads, at most 512 blocks) and the host fold
are the ones that were already on an IDENTICAL path
(`modeling_llama._refuse_nonfinite_device`, measured 2026-09-09); nothing
about them changed in the move.

WHAT A SCAN IS, AND WHY IT IS NOT A FOLD THAT FEEDS AN OUTPUT. Each
kernel reads a Float32 buffer and writes ONE `Int32` per block: the
smallest flat index in that block's grid-stride slice at which the
predicate holds, or `NONFINITE_NONE`. The host takes the minimum over the
partials. An integer minimum is exact and order-free, so "first index"
means THE SMALLEST INDEX on every vendor and at every grid size, and no
float is produced anywhere. These are bounds in the sense of
`device_absmax`'s docstring: nothing here reaches a card.

THE TWO PREDICATES, BOTH BY BITS (contract section 8, row 49: never by a
compare, because Metal flushes compare operands):

    nonfinite   `(bits & 0x7FFFFFFF) >= 0x7F800000`    NaN or infinity
    negative    `(bits & 0x80000000) != 0 and (bits & 0x7FFFFFFF) != 0`
                which equals `x < 0` for every non-NaN float; `-0.0` is
                NOT negative (as `x < 0` says), and a NaN with the sign
                bit set is reported. A caller that wants `x < 0` on a
                buffer that may hold NaN runs the finite scan first, as
                every host loop this replaces did (`byte_validate_state`).

`device_classify_nonfinite` is the tail of `_refuse_nonfinite_device`:
one 4 B readback of the offending element so the same message the host
oracle builds ("NaN in" or "infinity in") can be built from a device
result. It is the only download these helpers perform.

`DeviceScanScratch` holds the partials buffer and its pinned host mirror
so a step that runs ten scans allocates nothing per scan and waits twice
per scan instead of four times. The free functions keep the allocate-per-
call form for callers that scan once.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
# DEVIATION 2630: the step phase timers and counters (core/step_phase.mojo;
# compiled only under -D MOJOLEARN_STEP_PHASE_TIMERS=1).
from core.step_phase import (
    step_count_d2h,
    step_count_device_alloc,
    step_count_host_alloc,
    step_count_launch,
    step_count_sync,
)
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier


comptime SCAN_TPB = 256
"""Threads per block. The value `fused_attention.ABSMAX_TPB` had when the
kernel lived there; the block-level fold below is a halving tree over it."""
comptime SCAN_BLOCKS = 512
"""The grid cap, `fused_attention.ABSMAX_BLOCKS`'s value. Above this many
blocks' worth of elements every thread grid-strides."""

comptime NONFINITE_NONE: Int32 = 2147483647
"""The partial a block writes when its slice holds no hit. Larger than any
index a `DeviceBuffer` can hold (`Int32` indices), so the minimum over the
partials is the first hit or this."""


def _scan_blocks(n: Int) -> Int:
    """Blocks for `n` elements: one per `SCAN_TPB`, capped at `SCAN_BLOCKS`."""
    var blocks = (n + SCAN_TPB - 1) // SCAN_TPB
    if blocks > SCAN_BLOCKS:
        blocks = SCAN_BLOCKS
    return blocks


def nonfinite_partial_kernel(
    part: MutPointer[Int32, MutAnyOrigin],
    buf: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """One partial per block: the SMALLEST flat index at which `|bits| >=
    0x7F800000` (NaN or infinity, BY BITS, contract section 8), or
    `NONFINITE_NONE`. The host takes the minimum, so the index reported is
    the first one, exactly as the host loop it replaces reported it."""
    var n = Int(n_in)
    var red = stack_allocation[
        SCAN_TPB,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x) * SCAN_TPB
    var i = Int(block_idx.x) * SCAN_TPB + tid
    var best = NONFINITE_NONE
    while i < n:
        var au = bitcast[DType.uint32](buf.unsafe_load(i)) & UInt32(0x7FFFFFFF)
        if au >= UInt32(0x7F800000):
            best = Int32(i)
            break
        i += stride
    red.unsafe_store(tid, best)
    barrier()
    var active = SCAN_TPB // 2
    while active > 0:
        if tid < active:
            var o = red.unsafe_load(tid + active)
            if o < red.unsafe_load(tid):
                red.unsafe_store(tid, o)
        barrier()
        active = active // 2
    if tid == 0:
        part.unsafe_store(Int(block_idx.x), red.unsafe_load(0))


def negative_partial_kernel(
    part: MutPointer[Int32, MutAnyOrigin],
    buf: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`nonfinite_partial_kernel` with the predicate `(bits & 0x80000000)
    != 0 and (bits & 0x7FFFFFFF) != 0`, which is `x < 0` for every non-NaN
    float (design 2.3). Same partial shape, same fold, so "first" means
    the smallest index here too."""
    var n = Int(n_in)
    var red = stack_allocation[
        SCAN_TPB,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x) * SCAN_TPB
    var i = Int(block_idx.x) * SCAN_TPB + tid
    var best = NONFINITE_NONE
    while i < n:
        var bits = bitcast[DType.uint32](buf.unsafe_load(i))
        if (bits & UInt32(0x80000000)) != UInt32(0) and (
            bits & UInt32(0x7FFFFFFF)
        ) != UInt32(0):
            best = Int32(i)
            break
        i += stride
    red.unsafe_store(tid, best)
    barrier()
    var active = SCAN_TPB // 2
    while active > 0:
        if tid < active:
            var o = red.unsafe_load(tid + active)
            if o < red.unsafe_load(tid):
                red.unsafe_store(tid, o)
        barrier()
        active = active // 2
    if tid == 0:
        part.unsafe_store(Int(block_idx.x), red.unsafe_load(0))


def _fold_partials(mut host: HostBuffer[DType.int32], blocks: Int) -> Int:
    """The host half of every scan: the minimum over the block partials, in
    ascending block order (the order the moved code used; an integer
    minimum does not depend on it). -1 when no block hit."""
    var best = NONFINITE_NONE
    for i in range(blocks):
        var v = host.unsafe_ptr().unsafe_load(i)
        if v < best:
            best = v
    if best == NONFINITE_NONE:
        return -1
    return Int(best)


def device_first_nonfinite(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> Int:
    """The first flat index of a NaN or infinity in `buf[0:n]`, or -1.
    ONE read of the buffer on the device, where the host loop it replaces
    downloaded the buffer and walked it (65 ms for one block input at the
    Samba shape, measured 2026-09-09)."""
    if n <= 0:
        return -1
    var blocks = _scan_blocks(n)
    step_count_device_alloc()
    var part = ctx.enqueue_create_buffer[DType.int32](blocks)
    step_count_sync()
    ctx.synchronize()
    step_count_launch()
    ctx.enqueue_function[nonfinite_partial_kernel](
        part.unsafe_ptr(),
        buf.unsafe_ptr(),
        Int32(n),
        grid_dim=(blocks, 1, 1),
        block_dim=(SCAN_TPB, 1, 1),
    )
    step_count_sync()
    ctx.synchronize()
    step_count_host_alloc()
    var host = ctx.enqueue_create_host_buffer[DType.int32](blocks)
    step_count_sync()
    ctx.synchronize()
    step_count_d2h()
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=part)
    step_count_sync()
    ctx.synchronize()
    var best = _fold_partials(host, blocks)
    _ = host^
    _ = part^
    return best


def device_first_negative(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> Int:
    """The first flat index `i < n` with `buf[i] < 0` (by bits: sign set
    and magnitude nonzero), or -1. Same launch shape and same host fold as
    `device_first_nonfinite`. Run the finite scan first on a buffer that
    may hold NaN; see the module docstring."""
    if n <= 0:
        return -1
    var blocks = _scan_blocks(n)
    step_count_device_alloc()
    var part = ctx.enqueue_create_buffer[DType.int32](blocks)
    step_count_sync()
    ctx.synchronize()
    step_count_launch()
    ctx.enqueue_function[negative_partial_kernel](
        part.unsafe_ptr(),
        buf.unsafe_ptr(),
        Int32(n),
        grid_dim=(blocks, 1, 1),
        block_dim=(SCAN_TPB, 1, 1),
    )
    step_count_sync()
    ctx.synchronize()
    step_count_host_alloc()
    var host = ctx.enqueue_create_host_buffer[DType.int32](blocks)
    step_count_sync()
    ctx.synchronize()
    step_count_d2h()
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=part)
    step_count_sync()
    ctx.synchronize()
    var best = _fold_partials(host, blocks)
    _ = host^
    _ = part^
    return best


def device_classify_nonfinite(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], idx: Int
) raises -> Bool:
    """True if `buf[idx]` is a NaN, False if it is an infinity (either
    sign). One 4 B readback, the tail of
    `modeling_llama._refuse_nonfinite_device`; call it only with an index
    `device_first_nonfinite` returned. A finite element (a caller bug) is
    reported as an infinity rather than raising, so the refusal it feeds
    still fires."""
    var one = buf.create_sub_buffer[DType.float32](idx, 1)
    step_count_host_alloc()
    var host = ctx.enqueue_create_host_buffer[DType.float32](1)
    step_count_sync()
    ctx.synchronize()
    step_count_d2h()
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=one)
    step_count_sync()
    ctx.synchronize()
    var v = host.unsafe_ptr().unsafe_load(0)
    _ = host^
    _ = one^
    var au = bitcast[DType.uint32](v) & UInt32(0x7FFFFFFF)
    return au > UInt32(0x7F800000)


struct DeviceScanScratch(Movable):
    """The partials buffer and its pinned host mirror, allocated ONCE, for
    a caller that scans many buffers per step (design section 0: ten scans
    per LM step). Each scan is one launch, one 2 KB copy and two waits.
    The answers are the free functions' answers: same kernels, same
    geometry, same fold.

    `[[mojo-buffer-freed-at-last-use]]`: both buffers are fields, alive as
    long as the scratch is, so nothing here hands out a pointer to a
    buffer that could be dead at the wait."""

    var part: DeviceBuffer[DType.int32]
    var host: HostBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext) raises:
        step_count_device_alloc()
        self.part = ctx.enqueue_create_buffer[DType.int32](SCAN_BLOCKS)
        step_count_host_alloc()
        self.host = ctx.enqueue_create_host_buffer[DType.int32](SCAN_BLOCKS)
        step_count_sync()
        ctx.synchronize()

    def _finish(mut self, ctx: DeviceContext, blocks: Int) raises -> Int:
        step_count_sync()
        ctx.synchronize()
        var view = self.part.create_sub_buffer[DType.int32](0, blocks)
        step_count_d2h()
        ctx.enqueue_copy(dst_ptr=self.host.unsafe_ptr(), src_buf=view)
        step_count_sync()
        ctx.synchronize()
        _ = view^
        return _fold_partials(self.host, blocks)

    def first_nonfinite(
        mut self,
        ctx: DeviceContext,
        mut buf: DeviceBuffer[DType.float32],
        n: Int,
    ) raises -> Int:
        """`device_first_nonfinite` through this scratch."""
        if n <= 0:
            return -1
        var blocks = _scan_blocks(n)
        step_count_launch()
        ctx.enqueue_function[nonfinite_partial_kernel](
            self.part.unsafe_ptr(),
            buf.unsafe_ptr(),
            Int32(n),
            grid_dim=(blocks, 1, 1),
            block_dim=(SCAN_TPB, 1, 1),
        )
        return self._finish(ctx, blocks)

    def first_negative(
        mut self,
        ctx: DeviceContext,
        mut buf: DeviceBuffer[DType.float32],
        n: Int,
    ) raises -> Int:
        """`device_first_negative` through this scratch."""
        if n <= 0:
            return -1
        var blocks = _scan_blocks(n)
        step_count_launch()
        ctx.enqueue_function[negative_partial_kernel](
            self.part.unsafe_ptr(),
            buf.unsafe_ptr(),
            Int32(n),
            grid_dim=(blocks, 1, 1),
            block_dim=(SCAN_TPB, 1, 1),
        )
        return self._finish(ctx, blocks)
