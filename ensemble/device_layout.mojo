# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The forest fits' X upload, on the device (cpu-gpu-cleanup t-forest).

Replaces `ensemble/host_layout.mojo` on the GPU path. That module moved the
caller's float32 X into a pinned stage across the host thread pool,
transposing it there when the caller lent a ROW-major block, and scanned it
for NaN on host threads before the fit. Here the caller's bytes go to the
device exactly as they are (one host-pointer copy, no host pass over the
cells), the column-major plane is written by `core/column_stats.mojo`'s tiled
`transpose_kernel`, and the NaN refusal is a grid-wide device scan whose only
host step is reading back one flag word.

BITS. Both kernels move or test data; neither performs float arithmetic. The
transpose stores `dst[c * n_rows + r] = src[r * n_cols + c]` for every cell,
the exact words the host transpose staged, so the device plane is the plane
the old path uploaded. The NaN scan's flag is an idempotent store of the
constant 1 (every writer writes the same word), so its result does not depend
on thread order, grid shape or vendor.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from core.column_stats import CUDA_MAX_GRID_YZ, TRANSPOSE_TILE, transpose_kernel

comptime FOREST_SCAN_TPB = 256
"""Threads per block of the NaN scan."""

comptime FOREST_SCAN_MAX_BLOCKS = 65535
"""Grid cap of the NaN scan; the kernel strides over the rest."""


def forest_nan_scan_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    n: Int64,
    flag: MutPointer[Int32, MutAnyOrigin],
):
    """`flag[0] = 1` when any of `src[0:n]` is a NaN (exponent all ones and a
    nonzero mantissa: `(bits & 0x7FFFFFFF) > 0x7F800000`, the host scan's
    test). +-inf is not refused: it bins and partitions consistently.
    Grid-stride; the store is an idempotent constant, so no atomic."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var total = Int(n)
    var bad = False
    while i < total:
        var bits = bitcast[DType.uint32](src.unsafe_load(i))
        if (bits & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000):
            bad = True
        i += stride
    if bad:
        flag.unsafe_store(0, Int32(1))


def upload_forest_x(
    ctx: DeviceContext,
    src: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_cols: Int,
    row_major: Bool,
) raises -> DeviceBuffer[DType.float32]:
    """The column-major `n_rows x n_cols` device plane of a borrowed float32
    block. `row_major` says the block is the caller's C-order rows (then it is
    copied raw and transposed on the device); otherwise it is already
    column-major and is copied as it is. Synchronous: the borrow ends when
    this returns."""
    var n = n_rows * n_cols
    var dx = ctx.enqueue_create_buffer[DType.float32](max(1, n))
    if n <= 0:
        return dx^
    if row_major:
        var raw = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.enqueue_copy(dst_buf=raw, src_ptr=src)
        ctx.enqueue_function[transpose_kernel](
            dx.unsafe_ptr(),
            raw.unsafe_ptr(),
            Int32(n_rows),
            Int32(n_cols),
            grid_dim=(
                ceildiv(n_cols, TRANSPOSE_TILE),
                min(ceildiv(n_rows, TRANSPOSE_TILE), CUDA_MAX_GRID_YZ),
                1,
            ),
            block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
        )
        ctx.synchronize()
        _ = raw^
    else:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=src)
        ctx.synchronize()
    return dx^


def device_has_nan_f32(
    ctx: DeviceContext, x: DeviceBuffer[DType.float32], n: Int
) raises -> Bool:
    """True when any of the first `n` values of `x` is a NaN, scanned on the
    device (`forest_nan_scan_kernel`). The one host step is the flag word's
    readback."""
    if n <= 0:
        return False
    var flag = ctx.enqueue_create_buffer[DType.int32](1)
    flag.enqueue_fill(Int32(0))
    var blocks = min(ceildiv(n, FOREST_SCAN_TPB), FOREST_SCAN_MAX_BLOCKS)
    ctx.enqueue_function[forest_nan_scan_kernel](
        rebind[MutPointer[Float32, MutAnyOrigin]](x.unsafe_ptr()),
        Int64(n),
        flag.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(FOREST_SCAN_TPB, 1, 1),
    )
    var hflag = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=hflag, src_buf=flag)
    ctx.synchronize()
    var bad = hflag.unsafe_ptr().unsafe_load(0) != Int32(0)
    _ = flag^
    _ = hflag^
    return bad
