# SPDX-License-Identifier: Apache-2.0
"""T26: bounded leaf batches over the canonical no-sort row-chunk graph.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
"""
from std.atomic import Atomic
from std.gpu import block_idx, block_dim, thread_idx
from std.math import isfinite
from max.gpu.host import DeviceContext, DeviceBuffer
from checks.numerics import identical_mul_add
from core.device_zero import enqueue_fill
from gbdt.methods.leaves_estimation.tree_t26_units import (
    T26_SCRATCH_BYTES, T26F, T26B, t26_partial, t26_add, t26_leaf, t26_chunks,
)


def _partial_kernel(bins: T26B, y: T26F, cursor: T26F, n: Int32,
                    chunks: Int32, first: Int32, count: Int32, dst: T26F):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(chunks) * Int(count):
        var p = t26_partial(Int(first) + i // Int(chunks), i % Int(chunks), Int(n), bins, y, cursor)
        dst[2 * i] = p[0]
        dst[2 * i + 1] = p[1]


def _merge_kernel(dst: T26F, chunks: Int32, count: Int32, step_in: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(step_in)
    var pairs = (Int(chunks) + 2 * step - 1) // (2 * step)
    if i < pairs * Int(count):
        var leaf = i // pairs
        var c = (i % pairs) * 2 * step
        if c + step < Int(chunks):
            var left = 2 * (leaf * Int(chunks) + c)
            var right = left + 2 * step
            dst[left] = t26_add(dst[left], dst[right])
            dst[left + 1] = t26_add(dst[left + 1], dst[right + 1])


def _leaf_kernel(part: T26F, chunks: Int32, first: Int32, count: Int32,
                 lam: Float32, values: T26F, bad: T26B):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        var at = 2 * i * Int(chunks)
        var value = t26_leaf(part[at], part[at + 1], lam)
        if not isfinite(part[at]) or not isfinite(part[at + 1]) or not isfinite(value):
            # Every publisher writes the same failure flag after the queued
            # zero fill. This needs an atomic store, not read-modify-write.
            Atomic.store(bad, UInt32(1))
        values[Int(first) + i] = value


def _apply_kernel(bins: T26B, values: T26F, cursor: T26F, n: Int32, rate: Float32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        cursor[i] = identical_mul_add(values[Int(bins[i])], rate, cursor[i])


def t26_estimate_apply(
    ctx: DeviceContext, mut bins: DeviceBuffer[DType.uint32],
    mut y: DeviceBuffer[DType.float32], mut cursor: DeviceBuffer[DType.float32],
    n: Int, leaves: Int, lam: Float32, rate: Float32,
) raises -> List[Float32]:
    var chunks = t26_chunks(n, leaves, lam)
    # At most 64 MiB of chunk statistics; batch size changes scheduling only.
    # Host and device share the exact ABI and per-leaf scratch refusal above.
    var batch = min(leaves, max(1, T26_SCRATCH_BYTES // (8 * chunks)))
    var part = ctx.enqueue_create_buffer[DType.float32](2 * chunks * batch)
    var values = ctx.enqueue_create_buffer[DType.float32](leaves)
    var bad = ctx.enqueue_create_buffer[DType.uint32](1)
    enqueue_fill(ctx, bad, UInt32(0))
    var first = 0
    while first < leaves:
        var count = min(batch, leaves - first)
        ctx.enqueue_function[_partial_kernel](bins.unsafe_ptr(), y.unsafe_ptr(), cursor.unsafe_ptr(),
            Int32(n), Int32(chunks), Int32(first), Int32(count), part.unsafe_ptr(),
            grid_dim=(chunks * count + 255) // 256, block_dim=256)
        var step = 1
        while step < chunks:
            var pairs = (chunks + 2 * step - 1) // (2 * step)
            ctx.enqueue_function[_merge_kernel](part.unsafe_ptr(), Int32(chunks), Int32(count), Int32(step),
                grid_dim=(pairs * count + 255) // 256, block_dim=256)
            step *= 2
        ctx.enqueue_function[_leaf_kernel](part.unsafe_ptr(), Int32(chunks), Int32(first), Int32(count),
            lam, values.unsafe_ptr(), bad.unsafe_ptr(), grid_dim=(count + 255) // 256, block_dim=256)
        first += count
    var hv = ctx.enqueue_create_host_buffer[DType.float32](leaves)
    var hb = ctx.enqueue_create_host_buffer[DType.uint32](1)
    ctx.enqueue_copy(dst_buf=hb, src_buf=bad)
    ctx.enqueue_copy(dst_buf=hv, src_buf=values)
    ctx.synchronize()
    if hb[0] != UInt32(0):
        raise Error("T26 refuses nonfinite leaf statistics or estimates")
    ctx.enqueue_function[_apply_kernel](bins.unsafe_ptr(), values.unsafe_ptr(), cursor.unsafe_ptr(),
        Int32(n), rate, grid_dim=(n + 255) // 256, block_dim=256)
    ctx.synchronize()
    var result = List[Float32](capacity=leaves)
    for leaf in range(leaves):
        result.append(hv[leaf])
    _ = part^
    _ = values^
    _ = bad^
    _ = hv^
    _ = hb^
    return result^
