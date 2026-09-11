# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 2512 (2026-09-10): zero a device span with a KERNEL, not
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
from std.sys.compile import is_defined
from std.sys.info import size_of

comptime ZERO_TPB = 256
comptime ZERO_MAX_BLOCKS = 2048
comptime ZERO_LANES = 4
"""Four uint32 lanes, 16 bytes, per thread per grid stride."""

comptime MEMSET_FILL = is_defined["MOJOLEARN_2560_MEMSET_FILL"]()
"""DEVIATION 2560 (2026-09-11): opt-out switch for `enqueue_fill` only.
`-D MOJOLEARN_2560_MEMSET_FILL=1` restores `ctx.enqueue_memset` at every
gbdt fill site (the pre-89cb86ee launch sequence) so the section-9 A/B on
taxi and Istella-S runs from one source. The RF builder's
`enqueue_zero_bytes` is not switched. Both sides are held to the same
bytes by `pixi run check-device-zero` and `check-device-zero-memset`."""


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
    var nbytes = len(buf) * size_of[Scalar[dt]]()
    # The buffer is borrowed immutably here; the device memory behind it is
    # what the kernel writes, addressed as the caller's memset would.
    var p8 = MutPointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(buf.unsafe_ptr())
    )
    enqueue_zero_bytes(ctx, p8, nbytes)


def fill_kernel[dt: DType](
    dst: MutPointer[Scalar[dt], MutAnyOrigin], n: Int32, value: Scalar[dt]
):
    """`n` copies of `value`, one per thread, grid-stride."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        dst.unsafe_store(i, value)
        i += stride


def _all_zero_bits[dt: DType](value: Scalar[dt]) -> Bool:
    """True when every byte of `value` is 0x00. DEVIATION 2561: `value ==
    0` is also true for a float -0.0, whose bytes are not zero, so a -0.0
    fill routed to the zero kernel wrote +0.0 where the memset it replaced
    wrote -0.0. The test is on the bytes, as a memset's result is."""
    var v = value
    var p = MutPointer(to=v).unsafe_bitcast[UInt8]()
    for i in range(size_of[Scalar[dt]]()):
        if p[unsafe_offset=i] != UInt8(0):
            return False
    return True


def enqueue_fill[
    dt: DType
](ctx: DeviceContext, mut buf: DeviceBuffer[dt], value: Scalar[dt]) raises:
    """`enqueue_memset(buf, value)` as a kernel launch (DEVIATION 2512):
    the same bytes, none of Metal's memset-between-launches host cost. An
    all-zero-bytes value goes through the SIMD zero kernel (DEVIATION
    2561); any other value, -0.0 and NaN included, through the scalar
    fill. Nothing is enqueued for an empty buffer. `buf` is `mut` because
    the DEVIATION 2560 arm hands it to `enqueue_memset`, which every
    caller in the repository passes a mutable buffer."""
    comptime if MEMSET_FILL:
        ctx.enqueue_memset(buf, value)
    else:
        var n = len(buf)
        if n <= 0:
            return
        if _all_zero_bits[dt](value):
            enqueue_zero_buffer(ctx, buf)
            return
        var p = MutPointer[Scalar[dt], MutAnyOrigin](
            unsafe_from_address=Int(buf.unsafe_ptr())
        )
        ctx.enqueue_function[fill_kernel[dt]](
            p, Int32(n), value,
            grid_dim=_blocks_for(n), block_dim=ZERO_TPB,
        )
