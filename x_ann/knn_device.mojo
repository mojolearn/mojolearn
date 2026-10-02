# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The exact k-NN graph of t-SNE and CAGRA on the device, tiled (lane
ann-apple, 2026-09-28). SAME BITS as one thread per row running
`ts_knn_cell`:

  * one thread per row i, as before; the candidate rows j arrive in tiles of
    KTJ rows staged in threadgroup memory, tiles and rows ascending, so row
    i offers its candidates in the cell's order j = 0, 1, ..., n - 1;
  * the staged values are `ftz(x)`, and row i's own values are `ftz(x)` in
    registers: the cell's `ftz(ftz(x_i) - ftz(x_j))` on the same words
    (ftz is idempotent);
  * the fold runs over a comptime width MAXD >= d with zero padding past d:
    a padded step is `fma(+0, +0, acc)`, which returns acc unchanged (acc is
    a sum of squares from +0, never -0), so the sum is the cell's ascending
    fused fold over c < d;
  * each candidate goes through the cell's own `ts_knn_offer` (DEVIATION
    5810); a candidate that cannot enter a full list is skipped first by the
    same `ts_knn_beats` test against a register copy of the list's last
    entry.

Rows wider than 64 features take the untiled cell."""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_NVIDIA, COLUMN_AMD

from checks.numerics import ftz, identical_mul_add
from x_ann.tsne_core import F32P, I32P, ts_ftz_nonneg, ts_knn_beats, ts_knn_cell, ts_knn_offer
from x_ann.fast_env import FAST_KNN_BIGD

#: rows per threadgroup (one thread each) and candidate rows per tile
comptime KTB = 128
comptime KTJ = 64
comptime TPB = 64


def knn_cell_kernel(n: Int32, x: F32P, d: Int32, nn: Int32, nn_d: F32P, nn_i: I32P):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        ts_knn_cell(i, x, Int(n), Int(d), Int(nn), nn_d, nn_i)


def knn_tiled_kernel[MAXD: Int](n: Int32, x: F32P, d: Int32, nn: Int32, nn_d: F32P, nn_i: I32P):
    var t = Int(thread_idx.x)
    var nr = Int(n)
    var dd = Int(d)
    var k = Int(nn)
    var i = Int(block_idx.x) * KTB + t
    var tile = stack_allocation[KTJ * MAXD, Scalar[DType.float32], alignment=16, address_space=AddressSpace.SHARED]()
    var live = i < nr
    var xi = InlineArray[Float32, MAXD](fill=Float32(0.0))
    if live:
        comptime for c in range(MAXD):
            if c < dd:
                xi[c] = ftz(x.unsafe_load(i * dd + c))
    var base = i * k
    var filled = 0
    var ld = Float32(0.0)
    var li = 0
    var j0 = 0
    while j0 < nr:
        for e in range(t, KTJ * MAXD, KTB):
            var r = e // MAXD
            var c = e % MAXD
            var v = Float32(0.0)
            if j0 + r < nr and c < dd:
                v = ftz(x.unsafe_load((j0 + r) * dd + c))
            tile[e] = v
        barrier()
        if live:
            var jn = KTJ if nr - j0 > KTJ else nr - j0
            for r in range(jn):
                var j = j0 + r
                if j == i:
                    continue
                var acc = Float32(0.0)
                # the staged row four floats per load; the fold is the same
                # statements in the same order, lane by lane
                # lane ann-apple2: acc, a sum of squares from +0, is flushed
                # by `ts_ftz_nonneg` (the same word). A skip of the padded
                # groups was measured slower (m4pro-b 1790604321939) and is
                # not taken.
                comptime for c4 in range(MAXD // 4):
                    var tv = tile.unsafe_load[width=4, alignment=16](r * MAXD + 4 * c4)
                    comptime for u in range(4):
                        var diff = ftz(xi[4 * c4 + u] - tv[u])
                        acc = ts_ftz_nonneg(identical_mul_add(diff, diff, acc))
                if filled == k and not ts_knn_beats(acc, j, ld, li):
                    continue
                filled = ts_knn_offer(acc, j, base, k, filled, nn_d, nn_i)
                if filled == k:
                    ld = nn_d.unsafe_load(base + k - 1)
                    li = Int(nn_i.unsafe_load(base + k - 1))
        barrier()
        j0 += KTJ


#: the chunked arm (rows wider than 64 features): candidate rows per tile
#: and features per staged chunk
comptime KBJ = 32
comptime KBC = 32


def knn_tiled_bigd_kernel(n: Int32, x: F32P, d: Int32, nn: Int32, nn_d: F32P, nn_i: I32P):
    """FAST on Apple, `-D MOJOLEARN_ANN_FAST_KNN_BIGD=1` (lane/apple-fast-ann,
    2026-10-02): the tiled k-NN for rows wider than 64 features. One thread
    per row i (KTB per threadgroup); the candidate rows arrive in tiles of
    KBJ rows, each tile staged KBC features at a time (a KBJ x KBC slab of
    ftz(x) in threadgroup memory), and thread i keeps KBJ running sums, one
    per candidate, folding feature c ascending across the chunks: the
    cell's ascending fused fold on the same words, then the cell's own
    `ts_knn_offer` per candidate in j order. Cause: rows wider than 64
    features took `knn_cell_kernel` (`knn_enqueue`), one thread per row
    reading every candidate row from device memory with nothing staged,
    400,000 x 400,000 x 220 loads at CAGRA's Istella build. Same bits
    expected (the same statements in the same order)."""
    var t = Int(thread_idx.x)
    var nr = Int(n)
    var dd = Int(d)
    var k = Int(nn)
    var i = Int(block_idx.x) * KTB + t
    var slab = stack_allocation[KBJ * KBC, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var live = i < nr
    var base = i * k
    var filled = 0
    var ld = Float32(0.0)
    var li = 0
    var acc = InlineArray[Float32, KBJ](fill=Float32(0.0))
    var j0 = 0
    while j0 < nr:
        var jn = KBJ if nr - j0 > KBJ else nr - j0
        comptime for r in range(KBJ):
            acc[r] = Float32(0.0)
        var c0 = 0
        while c0 < dd:
            var cn = KBC if dd - c0 > KBC else dd - c0
            for e in range(t, KBJ * KBC, KTB):
                var r = e // KBC
                var c = e % KBC
                var v = Float32(0.0)
                if r < jn and c < cn:
                    v = ftz(x.unsafe_load((j0 + r) * dd + c0 + c))
                slab[e] = v
            barrier()
            if live:
                for c in range(cn):
                    var xi = ftz(x.unsafe_load(i * dd + c0 + c))
                    comptime for r in range(KBJ):
                        var diff = ftz(xi - slab[r * KBC + c])
                        acc[r] = ts_ftz_nonneg(identical_mul_add(diff, diff, acc[r]))
            barrier()
            c0 += KBC
        if live:
            for r in range(jn):
                var j = j0 + r
                if j == i:
                    continue
                var a = acc[r]
                if filled == k and not ts_knn_beats(a, j, ld, li):
                    continue
                filled = ts_knn_offer(a, j, base, k, filled, nn_d, nn_i)
                if filled == k:
                    ld = nn_d.unsafe_load(base + k - 1)
                    li = Int(nn_i.unsafe_load(base + k - 1))
        j0 += KBJ


# lane/gap-nv-classical2: rows wider than 64 features on NVIDIA and AMD.
# `knn_cell_kernel` (one thread per row re-reading both rows from global
# memory) ran istella's 20k x 220 graph. `knn_wide_kernel` computes a
# KW_TI x KW_TJ block of squared distances with both sides staged KW_KC
# features at a time and 4 x 4 cells per thread, each cell the cell's chain
# `ts_ftz_nonneg(fma(diff, diff, acc))`, diff = ftz(ftz(x_i) - ftz(x_j)),
# c ascending over exactly d; then row i's owner offers the tile's
# candidates in ascending j through the same `ts_knn_beats` /
# `ts_knn_offer`. The same bits. -D MOJOLEARN_KNN_WIDE_OFF=1 restores the
# cell kernel.
comptime KNN_WIDE = (
    (TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD)
    and not is_defined["MOJOLEARN_KNN_WIDE_OFF"]()
)
comptime KW_TI = 64
comptime KW_TJ = 64
comptime KW_KC = 16
comptime KW_TX = 16
comptime KW_TY = 16
comptime KW_RI = KW_TI // KW_TY
comptime KW_RJ = KW_TJ // KW_TX
comptime KW_TPB = KW_TX * KW_TY
comptime KW_DS = KW_TJ + 1


def knn_wide_kernel(n: Int32, x: F32P, d: Int32, nn: Int32, nn_d: F32P, nn_i: I32P):
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * KW_TX + tx
    var nr = Int(n)
    var dd = Int(d)
    var k = Int(nn)
    var i0 = Int(block_idx.x) * KW_TI
    var a_s = stack_allocation[KW_KC * KW_TI, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var b_s = stack_allocation[KW_KC * KW_TJ, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var dist_s = stack_allocation[KW_TI * KW_DS, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var i = i0 + tid
    var owner = tid < KW_TI and i < nr
    var base = i * k
    var filled = 0
    var ld = Float32(0.0)
    var li = 0
    var j0 = 0
    while j0 < nr:
        var acc = InlineArray[Float32, KW_RI * KW_RJ](fill=Float32(0.0))
        var k0 = 0
        while k0 < dd:
            comptime for q in range(KW_KC * KW_TI // KW_TPB):
                var e = tid + q * KW_TPB
                var ii = e // KW_KC
                var kk = e - ii * KW_KC
                var c = k0 + kk
                var v = Float32(0.0)
                if c < dd and i0 + ii < nr:
                    v = ftz(x.unsafe_load((i0 + ii) * dd + c))
                a_s[kk * KW_TI + ii] = v
            comptime for q in range(KW_KC * KW_TJ // KW_TPB):
                var e = tid + q * KW_TPB
                var jj = e // KW_KC
                var kk = e - jj * KW_KC
                var c = k0 + kk
                var v = Float32(0.0)
                if c < dd and j0 + jj < nr:
                    v = ftz(x.unsafe_load((j0 + jj) * dd + c))
                b_s[kk * KW_TJ + jj] = v
            barrier()
            var kmax = dd - k0
            if kmax > KW_KC:
                kmax = KW_KC
            for kk in range(kmax):
                var av = InlineArray[Float32, KW_RI](fill=Float32(0.0))
                var bv = InlineArray[Float32, KW_RJ](fill=Float32(0.0))
                comptime for r in range(KW_RI):
                    av[r] = a_s[kk * KW_TI + ty + r * KW_TY]
                comptime for c in range(KW_RJ):
                    bv[c] = b_s[kk * KW_TJ + tx + c * KW_TX]
                comptime for r in range(KW_RI):
                    comptime for c in range(KW_RJ):
                        var diff = ftz(av[r] - bv[c])
                        acc[r * KW_RJ + c] = ts_ftz_nonneg(identical_mul_add(diff, diff, acc[r * KW_RJ + c]))
            barrier()
            k0 += KW_KC
        comptime for r in range(KW_RI):
            comptime for c in range(KW_RJ):
                dist_s[(ty + r * KW_TY) * KW_DS + tx + c * KW_TX] = acc[r * KW_RJ + c]
        barrier()
        if owner:
            var jn = KW_TJ if nr - j0 > KW_TJ else nr - j0
            for r in range(jn):
                var j = j0 + r
                if j == i:
                    continue
                var dv = dist_s[tid * KW_DS + r]
                if filled == k and not ts_knn_beats(dv, j, ld, li):
                    continue
                filled = ts_knn_offer(dv, j, base, k, filled, nn_d, nn_i)
                if filled == k:
                    ld = nn_d.unsafe_load(base + k - 1)
                    li = Int(nn_i.unsafe_load(base + k - 1))
        barrier()
        j0 += KW_TJ


def knn_enqueue(
    ctx: DeviceContext, mut dx: DeviceBuffer[DType.float32], n: Int, d: Int, nn: Int,
    mut dnd: DeviceBuffer[DType.float32], mut dni: DeviceBuffer[DType.int32],
) raises:
    """Enqueue the k-NN graph of the n x d rows in dx (no sync)."""
    var blocks = (n + KTB - 1) // KTB
    # lane/apple-fast-ann: the chunked arm for wide rows, by build define
    comptime if FAST_KNN_BIGD:
        if d > 64:
            ctx.enqueue_function[knn_tiled_bigd_kernel](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn),
                                                        dnd.unsafe_ptr(), dni.unsafe_ptr(), grid_dim=blocks,
                                                        block_dim=KTB)
            return
    # lane ann-apple2: the fold width is d rounded up to a multiple of 4 up
    # to 32 (fewer padded steps; a padded step is the identity, see above),
    # then 48 and 64
    comptime for w4 in range(1, 9):
        if d <= 4 * w4 and d > 4 * (w4 - 1):
            ctx.enqueue_function[knn_tiled_kernel[4 * w4]](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn),
                                                            dnd.unsafe_ptr(), dni.unsafe_ptr(), grid_dim=blocks,
                                                            block_dim=KTB)
            return
    if d <= 48:
        ctx.enqueue_function[knn_tiled_kernel[48]](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn), dnd.unsafe_ptr(),
                                                    dni.unsafe_ptr(), grid_dim=blocks, block_dim=KTB)
    elif d <= 64:
        ctx.enqueue_function[knn_tiled_kernel[64]](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn), dnd.unsafe_ptr(),
                                                    dni.unsafe_ptr(), grid_dim=blocks, block_dim=KTB)
    else:
        comptime if KNN_WIDE:
            ctx.enqueue_function[knn_wide_kernel](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn), dnd.unsafe_ptr(),
                                                  dni.unsafe_ptr(), grid_dim=(n + KW_TI - 1) // KW_TI,
                                                  block_dim=(KW_TX, KW_TY, 1))
            return
        ctx.enqueue_function[knn_cell_kernel](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn), dnd.unsafe_ptr(),
                                              dni.unsafe_ptr(), grid_dim=(n + TPB - 1) // TPB, block_dim=TPB)
