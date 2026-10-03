# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device folds that replace "download, then loop over n on the host".

Lane cgr4-download-loop (2026-10-03). Every helper here reads a device
buffer once, in parallel, and leaves at most `SCAN_BLOCKS` (512) partials
for the host to combine: a fixed, size-capped fold, never a pass over n.
The launch shape is `core/device_scan.mojo`'s (256 threads, one block per
256 elements, at most 512 blocks, grid-stride above that).

INTEGER FOLDS (`device_sum_i32`, `device_count_nonzero_i32`,
`device_count_equal_i32`, `device_first_nonneg_i32`) are exact and
order-free: they return the value the host loop they replace returned, on
every vendor, so they change no bits.

THE FLOAT SUM (`device_sum_f32_fixed`) has ONE fold order, a pure function
of n, the same on NVIDIA, AMD, Apple and the host column:

  1. blocks = min(ceil(n / 256), 512); stride = blocks * 256.
  2. thread t of block b adds, in ascending k, the elements
     `b * 256 + t + k * stride` into a register that starts at 0.0.
  3. the block folds its 256 registers with the halving tree
     `red[t] += red[t + s]`, s = 128, 64, ..., 1.
  4. the host adds the block partials in ascending block order onto 0.0.

`host_sum_f32_fixed` is that order on the host, for the CPU-only column.
Every add is flushed (`checks.numerics.ftz`), so a subnormal is zero on
every vendor; no multiply appears, so no fused multiply-add can change a bit.
"""

from std.gpu import block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz
from core.device_scan import SCAN_TPB, SCAN_BLOCKS, NONFINITE_NONE

comptime _I32P = MutPointer[Int32, MutAnyOrigin]
comptime _I64P = MutPointer[Int64, MutAnyOrigin]
comptime _F32P = MutPointer[Float32, MutAnyOrigin]


def fold_blocks(n: Int) -> Int:
    """Blocks for `n` elements: one per `SCAN_TPB`, capped at `SCAN_BLOCKS`."""
    var blocks = (n + SCAN_TPB - 1) // SCAN_TPB
    if blocks > SCAN_BLOCKS:
        blocks = SCAN_BLOCKS
    if blocks < 1:
        blocks = 1
    return blocks


# ------------------------------------------------------------- kernels ----


def _sum_i32_kernel(part: _I64P, buf: _I32P, n_in: Int32, mode: Int32, ref_v: Int32):
    """One Int64 partial per block. mode 0: the sum of the values; mode 1:
    the count of nonzero values; mode 2: the count of values equal to
    `ref_v`. Integer, so exact in any order."""
    var n = Int(n_in)
    var red = stack_allocation[SCAN_TPB, Scalar[DType.int64], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x) * SCAN_TPB
    var i = Int(block_idx.x) * SCAN_TPB + tid
    var acc = Int64(0)
    while i < n:
        var v = buf.unsafe_load(i)
        if mode == 0:
            acc += Int64(v)
        elif mode == 1:
            if v != Int32(0):
                acc += 1
        else:
            if v == ref_v:
                acc += 1
        i += stride
    red.unsafe_store(tid, acc)
    barrier()
    var active = SCAN_TPB // 2
    while active > 0:
        if tid < active:
            red.unsafe_store(tid, red.unsafe_load(tid) + red.unsafe_load(tid + active))
        barrier()
        active = active // 2
    if tid == 0:
        part.unsafe_store(Int(block_idx.x), red.unsafe_load(0))


def _first_nonneg_i32_kernel(part: _I32P, buf: _I32P, n_in: Int32):
    """One partial per block: the smallest index with `buf[i] >= 0`, or
    `NONFINITE_NONE`. An integer minimum, so the first index everywhere."""
    var n = Int(n_in)
    var red = stack_allocation[SCAN_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x) * SCAN_TPB
    var i = Int(block_idx.x) * SCAN_TPB + tid
    var best = NONFINITE_NONE
    while i < n:
        if buf.unsafe_load(i) >= Int32(0):
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


def _sum_f32_kernel(part: _F32P, buf: _F32P, n_in: Int32):
    """Steps 2 and 3 of the module docstring's fold."""
    var n = Int(n_in)
    var red = stack_allocation[SCAN_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x) * SCAN_TPB
    var i = Int(block_idx.x) * SCAN_TPB + tid
    var acc = Float32(0.0)
    while i < n:
        acc = ftz(acc + buf.unsafe_load(i))
        i += stride
    red.unsafe_store(tid, acc)
    barrier()
    var active = SCAN_TPB // 2
    while active > 0:
        if tid < active:
            red.unsafe_store(tid, ftz(red.unsafe_load(tid) + red.unsafe_load(tid + active)))
        barrier()
        active = active // 2
    if tid == 0:
        part.unsafe_store(Int(block_idx.x), red.unsafe_load(0))


def _mark_kernel(flags: _I32P, keys: _I32P, n_in: Int32, mask: Int32, m_in: Int32):
    """flags[keys[i] & mask] = 1 for every i (keys outside [0, m) skipped).
    Every writer stores the same 1, so the race is benign and exact."""
    var i = Int(block_idx.x) * SCAN_TPB + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var key = Int(keys.unsafe_load(i) & mask)
    if key >= 0 and key < Int(m_in):
        flags.unsafe_store(key, Int32(1))


def _zero_i32_kernel(buf: _I32P, n_in: Int32):
    var i = Int(block_idx.x) * SCAN_TPB + Int(thread_idx.x)
    if i < Int(n_in):
        buf.unsafe_store(i, Int32(0))


# ------------------------------------------------------------ host side ----


def _fold_i64(ctx: DeviceContext, mut part: DeviceBuffer[DType.int64], blocks: Int) raises -> Int64:
    var host = ctx.enqueue_create_host_buffer[DType.int64](blocks)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=part)
    # the host reads `host` next: the one wait this fold needs
    ctx.synchronize()
    var total = Int64(0)
    for b in range(blocks):
        total += host.unsafe_ptr().unsafe_load(b)
    _ = host^
    return total


def _i32_fold(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], n: Int, mode: Int32, ref_v: Int32
) raises -> Int64:
    if n <= 0:
        return Int64(0)
    var blocks = fold_blocks(n)
    var part = ctx.enqueue_create_buffer[DType.int64](blocks)
    ctx.enqueue_function[_sum_i32_kernel](
        part.unsafe_ptr(), buf.unsafe_ptr(), Int32(n), mode, ref_v,
        grid_dim=(blocks, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    var total = _fold_i64(ctx, part, blocks)
    _ = part^
    return total


def device_sum_i32(ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], n: Int) raises -> Int64:
    """The exact Int64 sum of `buf[0:n]`."""
    return _i32_fold(ctx, buf, n, Int32(0), Int32(0))


def device_count_nonzero_i32(ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], n: Int) raises -> Int:
    """How many of `buf[0:n]` are nonzero."""
    return Int(_i32_fold(ctx, buf, n, Int32(1), Int32(0)))


def device_count_equal_i32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], n: Int, value: Int32
) raises -> Int:
    """How many of `buf[0:n]` equal `value`."""
    return Int(_i32_fold(ctx, buf, n, Int32(2), value))


def device_first_nonneg_i32(ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], n: Int) raises -> Int:
    """The first index `i < n` with `buf[i] >= 0`, or -1."""
    if n <= 0:
        return -1
    var blocks = fold_blocks(n)
    var part = ctx.enqueue_create_buffer[DType.int32](blocks)
    ctx.enqueue_function[_first_nonneg_i32_kernel](
        part.unsafe_ptr(), buf.unsafe_ptr(), Int32(n),
        grid_dim=(blocks, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    var host = ctx.enqueue_create_host_buffer[DType.int32](blocks)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=part)
    ctx.synchronize()
    var best = NONFINITE_NONE
    for b in range(blocks):
        var v = host.unsafe_ptr().unsafe_load(b)
        if v < best:
            best = v
    _ = host^
    _ = part^
    return -1 if best == NONFINITE_NONE else Int(best)


def device_count_distinct_keys(
    ctx: DeviceContext, mut keys: DeviceBuffer[DType.int32], n: Int, m: Int, mask: Int32
) raises -> Int:
    """How many distinct values `keys[i] & mask` take over `i < n`, for keys
    known to lie in [0, m): one flag per key value, set in parallel, then
    counted. Exact."""
    if n <= 0 or m <= 0:
        return 0
    var flags = ctx.enqueue_create_buffer[DType.int32](m)
    ctx.enqueue_function[_zero_i32_kernel](
        flags.unsafe_ptr(), Int32(m),
        grid_dim=((m + SCAN_TPB - 1) // SCAN_TPB, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    ctx.enqueue_function[_mark_kernel](
        flags.unsafe_ptr(), keys.unsafe_ptr(), Int32(n), mask, Int32(m),
        grid_dim=((n + SCAN_TPB - 1) // SCAN_TPB, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    var c = device_count_nonzero_i32(ctx, flags, m)
    _ = flags^
    return c


def device_sum_f32_fixed(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> Float32:
    """The module docstring's fixed-order Float32 sum of `buf[0:n]`."""
    if n <= 0:
        return Float32(0.0)
    var blocks = fold_blocks(n)
    var part = ctx.enqueue_create_buffer[DType.float32](blocks)
    ctx.enqueue_function[_sum_f32_kernel](
        part.unsafe_ptr(), buf.unsafe_ptr(), Int32(n),
        grid_dim=(blocks, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    var host = ctx.enqueue_create_host_buffer[DType.float32](blocks)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=part)
    ctx.synchronize()
    var total = Float32(0.0)
    for b in range(blocks):
        total = ftz(total + host.unsafe_ptr().unsafe_load(b))
    _ = host^
    _ = part^
    return total


def host_sum_f32_fixed(x: List[Float32], n: Int) -> Float32:
    """`device_sum_f32_fixed`'s fold order on the host (the CPU-only column)."""
    if n <= 0:
        return Float32(0.0)
    var blocks = fold_blocks(n)
    var stride = blocks * SCAN_TPB
    var red = List[Float32](length=SCAN_TPB, fill=Float32(0.0))
    var total = Float32(0.0)
    for b in range(blocks):
        for t in range(SCAN_TPB):
            var acc = Float32(0.0)
            var i = b * SCAN_TPB + t
            while i < n:
                acc = ftz(acc + x[i])
                i += stride
            red[t] = acc
        var active = SCAN_TPB // 2
        while active > 0:
            for t in range(active):
                red[t] = ftz(red[t] + red[t + active])
            active = active // 2
        total = ftz(total + red[0])
    return total
