# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE fam2-prep-metrics (2026-10-04): the device kernels and entries of the
label metrics' binary64 epilogue. The arithmetic, the fold order and the
switch (IDN_METRIC_EPI) are sf_epilogue_core.mojo's; read its header."""
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.soft_f64 import SF64_ZERO, sf64_add, sf64_div
from metrics.impl.stats.detail.sf_epilogue_core import (
    IDN_METRIC_EPI, EPI_CH, sf64_from_i64, entropy_term, mi_term, n_c_two_i, ari_value, sf_to_f64,
)

comptime EPI_TPB = 256
comptime I32P = MutPointer[Int32, MutAnyOrigin]
comptime I64P = MutPointer[Int64, MutAnyOrigin]
comptime U64P = MutPointer[UInt64, MutAnyOrigin]


# ---------------------------------------------------------------- device kernels
def entropy_part_kernel(part: U64P, counts: I32P, k: Int32, size: Int32, nch: Int32):
    """One thread per chunk: PART[c] = the chunk's entropy terms added
    ascending from zero."""
    var c = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    if c < Int(nch):
        var lo = c * EPI_CH
        var hi = min(lo + EPI_CH, Int(k))
        var acc = SF64_ZERO
        for i in range(lo, hi):
            acc = sf64_add(acc, entropy_term(Int(counts.unsafe_load(i)), Int(size)))
        part.unsafe_store(c, acc)


def cm_sums_kernel(a: I64P, b: I64P, rs: I64P, cm: I32P, k: Int32):
    """One thread per row (t < k) or column (t >= k) of the k x k
    contingency matrix: A[i] = the row sum, RS[i] = the row's sum of
    nCTwo(c_ij), B[j] = the column sum. Exact integers."""
    var t = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    var kk = Int(k)
    if t < kk:
        var s = 0
        var r = 0
        for j in range(kk):
            var v = Int(cm.unsafe_load(t * kk + j))
            s += v
            r += n_c_two_i(v)
        a.unsafe_store(t, Int64(s))
        rs.unsafe_store(t, Int64(r))
    elif t < 2 * kk:
        var j = t - kk
        var s = 0
        for i in range(kk):
            s += Int(cm.unsafe_load(i * kk + j))
        b.unsafe_store(j, Int64(s))


def mi_part_kernel(part: U64P, cm: I32P, a: I64P, b: I64P, k: Int32, size: Int32, nch: Int32):
    """One thread per chunk of EPI_CH row-major cells: PART[c] = the chunk's
    MI terms added ascending from zero."""
    var c = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    if c < Int(nch):
        var kk = Int(k)
        var lo = c * EPI_CH
        var hi = min(lo + EPI_CH, kk * kk)
        var acc = SF64_ZERO
        for t in range(lo, hi):
            var i = t // kk
            var j = t - i * kk
            var ab = Int(a.unsafe_load(i)) * Int(b.unsafe_load(j))
            acc = sf64_add(acc, mi_term(Int(cm.unsafe_load(t)), ab, Int(size)))
        part.unsafe_store(c, acc)


def epi_final_kernel(out: U64P, part: U64P, nch: Int32, size: Int32, divide: Int32):
    """One thread: OUT[0] = the chunk sums added ascending from zero,
    divided by size when `divide` (MI)."""
    var t = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    if t == 0:
        var acc = SF64_ZERO
        for c in range(Int(nch)):
            acc = sf64_add(acc, part.unsafe_load(c))
        if divide != Int32(0):
            acc = sf64_div(acc, sf64_from_i64(Int(size)))
        out.unsafe_store(0, acc)


def ari_final_kernel(out: U64P, a: I64P, b: I64P, rs: I64P, k: Int32, size: Int32):
    """One thread: the three pair-count sums over the k rows / columns
    (exact Int64, any order), then `ari_value`."""
    var t = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    if t == 0:
        var n2 = 0
        var a2 = 0
        var b2 = 0
        for i in range(Int(k)):
            n2 += Int(rs.unsafe_load(i))
            a2 += n_c_two_i(Int(a.unsafe_load(i)))
            b2 += n_c_two_i(Int(b.unsafe_load(i)))
        out.unsafe_store(0, ari_value(n2, a2, b2, Int(size)))


# ---------------------------------------------------------------- device entries
def _read_word(ctx: DeviceContext, mut out: DeviceBuffer[DType.uint64]) raises -> UInt64:
    """The one binary64 word a label metric brings back."""
    var h = ctx.enqueue_create_host_buffer[DType.uint64](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=out)
    ctx.synchronize()
    var bits = h.unsafe_ptr().unsafe_load(0)
    _ = h^
    return bits


def entropy_epilogue_device(
    ctx: DeviceContext, mut bins: DeviceBuffer[DType.int32], k: Int, size: Int
) raises -> Float64:
    """The entropy of the device histogram `bins[0:k]` of `size` labels."""
    var nch = ceildiv(k, EPI_CH)
    var part = ctx.enqueue_create_buffer[DType.uint64](nch)
    var out = ctx.enqueue_create_buffer[DType.uint64](1)
    ctx.enqueue_function[entropy_part_kernel](
        part.unsafe_ptr(), bins.unsafe_ptr(), Int32(k), Int32(size), Int32(nch),
        grid_dim=(ceildiv(nch, EPI_TPB), 1, 1), block_dim=(EPI_TPB, 1, 1),
    )
    ctx.enqueue_function[epi_final_kernel](
        out.unsafe_ptr(), part.unsafe_ptr(), Int32(nch), Int32(size), Int32(0),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
    var bits = _read_word(ctx, out)
    _ = part^
    _ = out^
    return sf_to_f64(bits)


def mi_epilogue_device(
    ctx: DeviceContext, mut cm: DeviceBuffer[DType.int32], k: Int, size: Int
) raises -> Float64:
    """Mutual information (nats) of the device contingency matrix `cm`
    (k x k row-major ints) of `size` label pairs."""
    var a = ctx.enqueue_create_buffer[DType.int64](k)
    var b = ctx.enqueue_create_buffer[DType.int64](k)
    var rs = ctx.enqueue_create_buffer[DType.int64](k)
    var nch = ceildiv(k * k, EPI_CH)
    var part = ctx.enqueue_create_buffer[DType.uint64](nch)
    var out = ctx.enqueue_create_buffer[DType.uint64](1)
    ctx.enqueue_function[cm_sums_kernel](
        a.unsafe_ptr(), b.unsafe_ptr(), rs.unsafe_ptr(), cm.unsafe_ptr(), Int32(k),
        grid_dim=(ceildiv(2 * k, EPI_TPB), 1, 1), block_dim=(EPI_TPB, 1, 1),
    )
    ctx.enqueue_function[mi_part_kernel](
        part.unsafe_ptr(), cm.unsafe_ptr(), a.unsafe_ptr(), b.unsafe_ptr(), Int32(k), Int32(size), Int32(nch),
        grid_dim=(ceildiv(nch, EPI_TPB), 1, 1), block_dim=(EPI_TPB, 1, 1),
    )
    ctx.enqueue_function[epi_final_kernel](
        out.unsafe_ptr(), part.unsafe_ptr(), Int32(nch), Int32(size), Int32(1),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
    var bits = _read_word(ctx, out)
    _ = a^
    _ = b^
    _ = rs^
    _ = part^
    _ = out^
    return sf_to_f64(bits)


def ari_epilogue_device(
    ctx: DeviceContext, mut cm: DeviceBuffer[DType.int32], k: Int, size: Int
) raises -> Float64:
    """The adjusted Rand index of the device contingency matrix `cm`."""
    var a = ctx.enqueue_create_buffer[DType.int64](k)
    var b = ctx.enqueue_create_buffer[DType.int64](k)
    var rs = ctx.enqueue_create_buffer[DType.int64](k)
    var out = ctx.enqueue_create_buffer[DType.uint64](1)
    ctx.enqueue_function[cm_sums_kernel](
        a.unsafe_ptr(), b.unsafe_ptr(), rs.unsafe_ptr(), cm.unsafe_ptr(), Int32(k),
        grid_dim=(ceildiv(2 * k, EPI_TPB), 1, 1), block_dim=(EPI_TPB, 1, 1),
    )
    ctx.enqueue_function[ari_final_kernel](
        out.unsafe_ptr(), a.unsafe_ptr(), b.unsafe_ptr(), rs.unsafe_ptr(), Int32(k), Int32(size),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
    var bits = _read_word(ctx, out)
    _ = a^
    _ = b^
    _ = rs^
    _ = out^
    return sf_to_f64(bits)
