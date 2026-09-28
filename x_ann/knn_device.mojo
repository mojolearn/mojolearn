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

from checks.numerics import ftz, identical_mul_add
from x_ann.tsne_core import F32P, I32P, ts_ftz_nonneg, ts_knn_beats, ts_knn_cell, ts_knn_offer

#: rows per threadgroup (one thread each) and candidate rows per tile
comptime KTB = 256
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


def knn_enqueue(
    ctx: DeviceContext, mut dx: DeviceBuffer[DType.float32], n: Int, d: Int, nn: Int,
    mut dnd: DeviceBuffer[DType.float32], mut dni: DeviceBuffer[DType.int32],
) raises:
    """Enqueue the k-NN graph of the n x d rows in dx (no sync)."""
    var blocks = (n + KTB - 1) // KTB
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
        ctx.enqueue_function[knn_cell_kernel](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn), dnd.unsafe_ptr(),
                                              dni.unsafe_ptr(), grid_dim=(n + TPB - 1) // TPB, block_dim=TPB)
