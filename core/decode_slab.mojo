# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""N tokens per native call on a resident decode session (decode plan D2).

A session's `decode(x, n)` uploads the caller's (B, N, width) input slab once,
runs its one-token step body N times on the device and downloads one
(B, N, width) output slab at the end. The step body reads a (B, width) token
buffer and writes a (B, width) output buffer, so token t moves between the
slab and those buffers with one strided row copy on the device:

    dst[r * dst_stride + dst_offset + j] = src[r * src_stride + src_offset + j]
    for r < rows, j < width

It is a copy: no arithmetic, so the step's bytes are the bytes of N separate
`step` calls. One thread per element; a launch covers rows * width elements.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext


comptime DECODE_SLAB_TPB = 256
"""Threads per block of the row copy."""


def decode_slab_rows_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    width_in: Int32,
    dst_stride_in: Int32,
    dst_offset_in: Int32,
    src_stride_in: Int32,
    src_offset_in: Int32,
):
    var rows = Int(rows_in)
    var width = Int(width_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= rows * width:
        return
    var r = cell // width
    var j = cell - r * width
    dst.unsafe_store(
        r * Int(dst_stride_in) + Int(dst_offset_in) + j,
        src.unsafe_load(r * Int(src_stride_in) + Int(src_offset_in) + j),
    )


def decode_slab_check(rows: Int, tokens: Int, width: Int, what: String) raises:
    """Refuse a slab the Int32 launch words cannot index."""
    if rows < 1 or tokens < 1 or width < 1:
        raise Error(what + ": B, n_tokens and the width must be positive")
    if rows * tokens * width >= 2147483647:
        raise Error(what + ": the (B, n_tokens, width) slab exceeds 2**31 - 1 elements; decode it in shorter calls")


def decode_slab_rows(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.float32],
    dst_stride: Int,
    dst_offset: Int,
    mut src: DeviceBuffer[DType.float32],
    src_stride: Int,
    src_offset: Int,
    rows: Int,
    width: Int,
) raises:
    """Enqueue one strided row copy (no wait)."""
    var n = rows * width
    var grid = (n + DECODE_SLAB_TPB - 1) // DECODE_SLAB_TPB
    if grid < 1:
        grid = 1
    ctx.enqueue_function[decode_slab_rows_kernel](
        dst.unsafe_ptr(),
        src.unsafe_ptr(),
        Int32(rows),
        Int32(width),
        Int32(dst_stride),
        Int32(dst_offset),
        Int32(src_stride),
        Int32(src_offset),
        grid_dim=(grid, 1, 1),
        block_dim=(DECODE_SLAB_TPB, 1, 1),
    )


def decode_slab_take(
    ctx: DeviceContext,
    mut token: DeviceBuffer[DType.float32],
    mut slab: DeviceBuffer[DType.float32],
    rows: Int,
    tokens: Int,
    width: Int,
    t: Int,
) raises:
    """Token `t` of a (rows, tokens, width) slab into a (rows, width) buffer."""
    decode_slab_rows(ctx, token, width, 0, slab, tokens * width, t * width, rows, width)


def decode_slab_put(
    ctx: DeviceContext,
    mut slab: DeviceBuffer[DType.float32],
    mut token: DeviceBuffer[DType.float32],
    rows: Int,
    tokens: Int,
    width: Int,
    t: Int,
) raises:
    """A (rows, width) buffer into token `t` of a (rows, tokens, width) slab."""
    decode_slab_rows(ctx, slab, tokens * width, t * width, token, width, 0, rows, width)


def decode_slab_ids_kernel(
    dst: MutPointer[Int32, MutAnyOrigin],
    src: MutPointer[Int32, MutAnyOrigin],
    rows_in: Int32,
    tokens_in: Int32,
    t_in: Int32,
):
    """Column `t` of a (rows, tokens) id slab into a (rows,) id buffer."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(rows_in):
        return
    dst.unsafe_store(r, src.unsafe_load(r * Int(tokens_in) + Int(t_in)))


def decode_slab_take_ids(
    ctx: DeviceContext,
    mut token: DeviceBuffer[DType.int32],
    mut slab: DeviceBuffer[DType.int32],
    rows: Int,
    tokens: Int,
    t: Int,
) raises:
    """Token `t` of a (rows, tokens) id slab into a (rows,) buffer (no wait)."""
    var grid = (rows + DECODE_SLAB_TPB - 1) // DECODE_SLAB_TPB
    if grid < 1:
        grid = 1
    ctx.enqueue_function[decode_slab_ids_kernel](
        token.unsafe_ptr(),
        slab.unsafe_ptr(),
        Int32(rows),
        Int32(tokens),
        Int32(t),
        grid_dim=(grid, 1, 1),
        block_dim=(DECODE_SLAB_TPB, 1, 1),
    )
