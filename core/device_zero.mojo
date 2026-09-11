# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 2501 (2026-09-10): zero a device span with a KERNEL, not
`enqueue_memset`.

Measured on the Apple M4 under MAX 26.5 (`bench/results/rf_fast_mac_2026-09-10/`):
a lone `enqueue_memset` costs the host 12 us for 4 KB and 41 us for 4 MB,
but a memset placed BETWEEN two kernel launches costs 110 us more than the
three enqueues alone, and the RandomForest builder's per-round histogram
memset (up to 42 MB, 14,640 of them per 100-tree HIGGS 1M fit) averaged
650 us of host time each: 9.5 s of a 14.4 s fit, with the histogram and
best-split launches beside it at 10 us apiece. A copy between launches
pays nothing extra and neither does a kernel launch, so the fill moves
into a kernel. Zeros are zeros: the bytes written are the same in every
tier and on every vendor, so this changes no fingerprint. On CUDA a launch
and a memset are both a few microseconds; the helper is used everywhere
so the launch sequence is one sequence (`[[always-gpu-agnostic]]`).

Two kernels: a grid-stride SIMD store over the 16-byte body and a scalar
byte kernel for the tail, so any byte count and any 4-byte-aligned start
is covered. Device buffers from `enqueue_create_buffer` and the builders'
carved prefixes are 16-byte aligned or better; a start that is not 4-byte
aligned takes the byte kernel for the whole span.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv

comptime ZERO_TPB = 256
comptime ZERO_MAX_BLOCKS = 2048
comptime ZERO_LANES = 4
"""Four uint32 lanes, 16 bytes, per thread per grid stride."""


def zero_words_kernel(dst: MutPointer[UInt32, MutAnyOrigin], n_vec: Int32):
    """`n_vec` 16-byte groups of zeros, grid-stride."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var z = SIMD[DType.uint32, ZERO_LANES](0)
    while i < Int(n_vec):
        dst.unsafe_store[width=ZERO_LANES](i * ZERO_LANES, z)
        i += stride


def zero_bytes_kernel(dst: MutPointer[UInt8, MutAnyOrigin], n: Int32):
    """`n` bytes of zeros, one per thread, grid-stride."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        dst.unsafe_store(i, UInt8(0))
        i += stride


def _blocks_for(items: Int, per_thread: Int = 1) -> Int:
    var b = ceildiv(items, ZERO_TPB * per_thread)
    if b > ZERO_MAX_BLOCKS:
        return ZERO_MAX_BLOCKS
    if b < 1:
        return 1
    return b


def enqueue_zero_bytes[
    origin: MutOrigin
](ctx: DeviceContext, dst: MutPointer[UInt8, origin], nbytes: Int) raises:
    """Enqueue zeros over `[dst, dst + nbytes)` on `ctx`'s queue. Nothing
    is enqueued for `nbytes <= 0`. A negative count is a caller bug and is
    treated as zero rather than read."""
    if nbytes <= 0:
        return
    var addr = Int(dst)
    var p8 = dst.unsafe_origin_cast[MutAnyOrigin]()
    if addr % 4 != 0:
        ctx.enqueue_function[zero_bytes_kernel](
            p8, Int32(nbytes),
            grid_dim=_blocks_for(nbytes), block_dim=ZERO_TPB,
        )
        return
    var n_vec = nbytes // 16
    if n_vec > 0:
        ctx.enqueue_function[zero_words_kernel](
            p8.unsafe_bitcast[UInt32](), Int32(n_vec),
            grid_dim=_blocks_for(n_vec, 4), block_dim=ZERO_TPB,
        )
    var tail = nbytes - n_vec * 16
    if tail > 0:
        ctx.enqueue_function[zero_bytes_kernel](
            p8 + n_vec * 16, Int32(tail),
            grid_dim=1, block_dim=ZERO_TPB,
        )


def enqueue_zero_buffer[
    dt: DType
](ctx: DeviceContext, buf: DeviceBuffer[dt]) raises:
    """`enqueue_zero_bytes` over a whole device buffer."""
    var nbytes = len(buf) * dt.size_of()
    enqueue_zero_bytes(
        ctx, buf.unsafe_ptr().unsafe_bitcast[UInt8]().unsafe_origin_cast[MutAnyOrigin](), nbytes
    )
