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


def sf_level_kernel(
    buf: U64P, src_off: Int32, n_src: Int32, dst_off: Int32, n_dst: Int32, size: Int32, divide: Int32
):
    """One level of the fold: one thread per chunk of EPI_CH source sums,
    BUF[dst_off + c] = BUF[src_off + c*EPI_CH ..] added ascending from zero.
    The last level (n_dst == 1) divides by size when `divide` (MI)."""
    var c = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    if c < Int(n_dst):
        var lo = c * EPI_CH
        var hi = min(lo + EPI_CH, Int(n_src))
        var acc = SF64_ZERO
        for i in range(lo, hi):
            acc = sf64_add(acc, buf.unsafe_load(Int(src_off) + i))
        if divide != Int32(0) and Int(n_dst) == 1:
            acc = sf64_div(acc, sf64_from_i64(Int(size)))
        buf.unsafe_store(Int(dst_off) + c, acc)


def ari_part_kernel(tri: I64P, a: I64P, b: I64P, rs: I64P, k: Int32, nch: Int32):
    """One thread per chunk of EPI_CH rows / columns: TRI[3c ..] = the
    chunk's sums of RS, nCTwo(A) and nCTwo(B) (exact Int64, any order)."""
    var c = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    if c < Int(nch):
        var lo = c * EPI_CH
        var hi = min(lo + EPI_CH, Int(k))
        var n2 = 0
        var a2 = 0
        var b2 = 0
        for i in range(lo, hi):
            n2 += Int(rs.unsafe_load(i))
            a2 += n_c_two_i(Int(a.unsafe_load(i)))
            b2 += n_c_two_i(Int(b.unsafe_load(i)))
        tri.unsafe_store(3 * c, Int64(n2))
        tri.unsafe_store(3 * c + 1, Int64(a2))
        tri.unsafe_store(3 * c + 2, Int64(b2))


def ari_level_kernel(
    output: U64P, tri: I64P, src_off: Int32, n_src: Int32, dst_off: Int32, n_dst: Int32, size: Int32
):
    """One level of the integer fold over triples (offsets count triples);
    the last level (n_dst == 1) also writes OUT[0] = `ari_value`."""
    var c = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    if c < Int(n_dst):
        var lo = c * EPI_CH
        var hi = min(lo + EPI_CH, Int(n_src))
        var n2 = 0
        var a2 = 0
        var b2 = 0
        for i in range(lo, hi):
            var at = 3 * (Int(src_off) + i)
            n2 += Int(tri.unsafe_load(at))
            a2 += Int(tri.unsafe_load(at + 1))
            b2 += Int(tri.unsafe_load(at + 2))
        var to = 3 * (Int(dst_off) + c)
        tri.unsafe_store(to, Int64(n2))
        tri.unsafe_store(to + 1, Int64(a2))
        tri.unsafe_store(to + 2, Int64(b2))
        if Int(n_dst) == 1:
            output.unsafe_store(0, ari_value(n2, a2, b2, Int(size)))


# ---------------------------------------------------------------- device entries
def fold_words(nch: Int) -> Int:
    """Words a fold buffer needs: the nch chunk sums and every level above."""
    return nch + nch // 32 + 8


def _read_word(ctx: DeviceContext, mut buf: DeviceBuffer[DType.uint64], at: Int) raises -> UInt64:
    """The one binary64 word a label metric brings back."""
    var h = ctx.enqueue_create_host_buffer[DType.uint64](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf.create_sub_buffer[DType.uint64](at, 1))
    ctx.synchronize()
    var bits = h.unsafe_ptr().unsafe_load(0)
    _ = h^
    return bits


def _fold_levels(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.uint64], nch: Int, size: Int, divide: Bool
) raises -> UInt64:
    """The levels of the fold over BUF[0:nch] (sf_epilogue_core.mojo
    `sf_fold_list`'s order: at least one level, then until one sum is
    left), and the final word read back."""
    var src_off = 0
    var n = nch
    var dst_off = nch
    if n <= 0:
        # lane/review-fixes: an empty fold would never reach one sum
        raise Error("sf_epilogue: the fold needs at least one partial")
    while True:
        var nd = ceildiv(n, EPI_CH)
        ctx.enqueue_function[sf_level_kernel](
            buf.unsafe_ptr(), Int32(src_off), Int32(n), Int32(dst_off), Int32(nd), Int32(size),
            Int32(1) if divide else Int32(0),
            grid_dim=(ceildiv(nd, EPI_TPB), 1, 1), block_dim=(EPI_TPB, 1, 1),
        )
        src_off = dst_off
        dst_off += nd
        n = nd
        if n == 1:
            break
    return _read_word(ctx, buf, src_off)


def entropy_epilogue_device(
    ctx: DeviceContext, mut bins: DeviceBuffer[DType.int32], k: Int, size: Int
) raises -> Float64:
    """The entropy of the device histogram `bins[0:k]` of `size` labels."""
    var nch = ceildiv(k, EPI_CH)
    var buf = ctx.enqueue_create_buffer[DType.uint64](fold_words(nch))
    ctx.enqueue_function[entropy_part_kernel](
        buf.unsafe_ptr(), bins.unsafe_ptr(), Int32(k), Int32(size), Int32(nch),
        grid_dim=(ceildiv(nch, EPI_TPB), 1, 1), block_dim=(EPI_TPB, 1, 1),
    )
    var bits = _fold_levels(ctx, buf, nch, size, False)
    _ = buf^
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
    var buf = ctx.enqueue_create_buffer[DType.uint64](fold_words(nch))
    ctx.enqueue_function[cm_sums_kernel](
        a.unsafe_ptr(), b.unsafe_ptr(), rs.unsafe_ptr(), cm.unsafe_ptr(), Int32(k),
        grid_dim=(ceildiv(2 * k, EPI_TPB), 1, 1), block_dim=(EPI_TPB, 1, 1),
    )
    ctx.enqueue_function[mi_part_kernel](
        buf.unsafe_ptr(), cm.unsafe_ptr(), a.unsafe_ptr(), b.unsafe_ptr(), Int32(k), Int32(size), Int32(nch),
        grid_dim=(ceildiv(nch, EPI_TPB), 1, 1), block_dim=(EPI_TPB, 1, 1),
    )
    var bits = _fold_levels(ctx, buf, nch, size, True)
    _ = a^
    _ = b^
    _ = rs^
    _ = buf^
    return sf_to_f64(bits)


def ari_epilogue_device(
    ctx: DeviceContext, mut cm: DeviceBuffer[DType.int32], k: Int, size: Int
) raises -> Float64:
    """The adjusted Rand index of the device contingency matrix `cm`."""
    var a = ctx.enqueue_create_buffer[DType.int64](k)
    var b = ctx.enqueue_create_buffer[DType.int64](k)
    var rs = ctx.enqueue_create_buffer[DType.int64](k)
    var nch = ceildiv(k, EPI_CH)
    var tri = ctx.enqueue_create_buffer[DType.int64](3 * fold_words(nch))
    var out = ctx.enqueue_create_buffer[DType.uint64](1)
    ctx.enqueue_function[cm_sums_kernel](
        a.unsafe_ptr(), b.unsafe_ptr(), rs.unsafe_ptr(), cm.unsafe_ptr(), Int32(k),
        grid_dim=(ceildiv(2 * k, EPI_TPB), 1, 1), block_dim=(EPI_TPB, 1, 1),
    )
    ctx.enqueue_function[ari_part_kernel](
        tri.unsafe_ptr(), a.unsafe_ptr(), b.unsafe_ptr(), rs.unsafe_ptr(), Int32(k), Int32(nch),
        grid_dim=(ceildiv(nch, EPI_TPB), 1, 1), block_dim=(EPI_TPB, 1, 1),
    )
    var src_off = 0
    var n = nch
    var dst_off = nch
    if n <= 0:
        # lane/review-fixes: an empty fold would never reach one sum
        raise Error("sf_epilogue: the fold needs at least one partial")
    while True:
        var nd = ceildiv(n, EPI_CH)
        ctx.enqueue_function[ari_level_kernel](
            out.unsafe_ptr(), tri.unsafe_ptr(), Int32(src_off), Int32(n), Int32(dst_off), Int32(nd), Int32(size),
            grid_dim=(ceildiv(nd, EPI_TPB), 1, 1), block_dim=(EPI_TPB, 1, 1),
        )
        src_off = dst_off
        dst_off += nd
        n = nd
        if n == 1:
            break
    var bits = _read_word(ctx, out, 0)
    _ = a^
    _ = b^
    _ = rs^
    _ = tri^
    _ = out^
    return sf_to_f64(bits)
