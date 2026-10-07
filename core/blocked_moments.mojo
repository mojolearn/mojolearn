# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Blocked column moments on the device (lane classical-decomp, 2026-10-07).

Row-parallel replacements for the classical IDENTICAL switches that ran one
GPU thread per output cell over every row (C01 mean, C04 PCA covariance,
C23 centered panels): each leaf of rows is one block, every cell of the leaf
is filled in parallel, and the leaves fold in the fixed binary-counter tree
of `core/blocked_moments_ops.mojo`, one parallel launch per level. The host
column is `core/blocked_moments_host.mojo` (and, for the panel Gram, the
per-cell reference cells it reproduces bit for bit).

No warp shuffles, no atomics, no hardware-sized geometry: the bits are a
function of (n, d) and the profile constants only, so NVIDIA and AMD agree.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from core.blocked_moments_ops import (
    BM_GRAM_MEAN_CHAINS,
    BM_MAX_LEVELS,
    BM_ROW_TILE,
    BM_TILE,
    BM_TPB,
    bm_add,
    bm_chan_m2,
    bm_chan_mean,
    bm_div,
    bm_fma,
    bm_leaf_count,
    bm_mean_leaf_rows,
    bm_node_rows,
    bm_onepass_leaf_rows,
    bm_sub,
    bm_sub_chains,
    bm_tile_pair,
)
from checks.numerics import ftz

comptime BmPtr = MutPointer[Float32, MutAnyOrigin]


@always_inline
def _bp(buf: DeviceBuffer[DType.float32]) -> BmPtr:
    return BmPtr(unsafe_from_address=Int(buf.unsafe_ptr()))


# ---------------------------------------------------------------------------
# Leaf kernels
# ---------------------------------------------------------------------------


def bm_leaf_colsum_kernel(part: BmPtr, x: BmPtr, n_in: Int32, d_in: Int32, leaf_in: Int32):
    """Block `k` = leaf `k`: `part[k * d + j]` = the leaf's column-`j` sum.
    Thread (s, j), `j` fastest, chains rows `r0 + s, r0 + s + S, ...`; the S
    chains of a column are then added in ascending `s`."""
    var n = Int(n_in)
    var d = Int(d_in)
    var leaf = Int(leaf_in)
    var k = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var r0 = k * leaf
    var r1 = min(n, r0 + leaf)
    var sh = stack_allocation[BM_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    if d <= BM_TPB:
        var chains = bm_sub_chains(d)
        var s = t // d
        var j = t - s * d
        var acc = Float32(0.0)
        if s < chains:
            var r = r0 + s
            while r < r1:
                acc = bm_add(acc, x.unsafe_load(r * d + j))
                r += chains
        sh[t] = acc
        barrier()
        if t < d:
            var tot = sh[t]
            for q in range(1, chains):
                tot = bm_add(tot, sh[q * d + t])
            part.unsafe_store(k * d + t, tot)
    else:
        var j = t
        while j < d:
            var acc = Float32(0.0)
            for r in range(r0, r1):
                acc = bm_add(acc, x.unsafe_load(r * d + j))
            part.unsafe_store(k * d + j, acc)
            j += BM_TPB


def bm_leaf_gram_kernel(
    part: BmPtr, leaf_mean: BmPtr, x: BmPtr, center: BmPtr,
    n_in: Int32, d_in: Int32, leaf_in: Int32, tiles_in: Int32, own_center: Int32,
):
    """Block (k, tp): leaf `k`, upper tile pair `tp` = (I, J). Thread (a, b)
    owns cell (I*16 + a, J*16 + b) and chains the leaf's centered products
    `(x_ri - c_i)(x_rj - c_j)` over ascending rows, staged BM_ROW_TILE rows
    at a time. `own_center == 0`: `c = center` (one vector, every leaf).
    `own_center != 0`: `c` = the leaf's own column means (BM_GRAM_MEAN_CHAINS
    interleaved chains added in order, then one division by the leaf's rows),
    written to `leaf_mean[k * d + i]` by the diagonal tile pairs.
    `part[k * d * d + i * d + j]` (and its mirror) holds the cell."""
    var n = Int(n_in)
    var d = Int(d_in)
    var leaf = Int(leaf_in)
    var k = Int(block_idx.x)
    var pair = bm_tile_pair(Int(block_idx.y), Int(tiles_in))
    var ti = pair[0]
    var tj = pair[1]
    var t = Int(thread_idx.x)
    var a = t // BM_TILE
    var b = t - a * BM_TILE
    var r0 = k * leaf
    var r1 = min(n, r0 + leaf)
    var ci = stack_allocation[BM_TILE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var cj = stack_allocation[BM_TILE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sums = stack_allocation[BM_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var xa = stack_allocation[BM_ROW_TILE * BM_TILE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var xb = stack_allocation[BM_ROW_TILE * BM_TILE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    if own_center != 0:
        comptime W = 2 * BM_TILE
        var u = t % W
        var s = t // W
        var col = ti * BM_TILE + u if u < BM_TILE else tj * BM_TILE + (u - BM_TILE)
        var acc = Float32(0.0)
        if col < d:
            var r = r0 + s
            while r < r1:
                acc = bm_add(acc, x.unsafe_load(r * d + col))
                r += BM_GRAM_MEAN_CHAINS
        sums[t] = acc
        barrier()
        if t < W:
            var tot = sums[t]
            for q in range(1, BM_GRAM_MEAN_CHAINS):
                tot = bm_add(tot, sums[q * W + t])
            var mean = bm_div(tot, Float32(r1 - r0))
            if t < BM_TILE:
                ci[t] = mean
            else:
                cj[t - BM_TILE] = mean
        barrier()
        if ti == tj and t < BM_TILE and ti * BM_TILE + t < d:
            leaf_mean.unsafe_store(k * d + ti * BM_TILE + t, ci[t])
    else:
        if t < BM_TILE:
            var gi = ti * BM_TILE + t
            ci[t] = ftz(center.unsafe_load(gi)) if gi < d else Float32(0.0)
        elif t < 2 * BM_TILE:
            var gj = tj * BM_TILE + (t - BM_TILE)
            cj[t - BM_TILE] = ftz(center.unsafe_load(gj)) if gj < d else Float32(0.0)
        barrier()
    var acc = Float32(0.0)
    var rt = r0
    while rt < r1:
        comptime for h in range(BM_ROW_TILE * BM_TILE // BM_TPB):
            var slot = t + h * BM_TPB
            var lr = slot // BM_TILE
            var lc = slot - lr * BM_TILE
            var rr = rt + lr
            var ga = ti * BM_TILE + lc
            var gb = tj * BM_TILE + lc
            var va = Float32(0.0)
            var vb = Float32(0.0)
            if rr < r1 and ga < d:
                va = bm_sub(x.unsafe_load(rr * d + ga), ci[lc])
            if rr < r1 and gb < d:
                vb = bm_sub(x.unsafe_load(rr * d + gb), cj[lc])
            xa[slot] = va
            xb[slot] = vb
        barrier()
        var here = min(BM_ROW_TILE, r1 - rt)
        for q in range(here):
            acc = bm_fma(xa[q * BM_TILE + a], xb[q * BM_TILE + b], acc)
        barrier()
        rt += BM_ROW_TILE
    var i = ti * BM_TILE + a
    var j = tj * BM_TILE + b
    if i < d and j < d:
        var base = k * d * d
        part.unsafe_store(base + i * d + j, acc)
        if ti != tj:
            part.unsafe_store(base + j * d + i, acc)


@always_inline
def _tsvd_value(x: BmPtr, v: BmPtr, r: Int, d: Int, j: Int) -> Float32:
    """Column `j` of [X | X V^T] at row `r`; a projection is one ascending
    flushed multiply-add chain over the features."""
    if j < d:
        return ftz(x.unsafe_load(r * d + j))
    var c = j - d
    var acc = Float32(0.0)
    for f in range(d):
        acc = bm_fma(ftz(x.unsafe_load(r * d + f)), ftz(v.unsafe_load(c * d + f)), acc)
    return acc


def bm_tsvd_leaf_kernel(
    leaf_mean: BmPtr, leaf_m2: BmPtr, x: BmPtr, v: BmPtr,
    n_in: Int32, d_in: Int32, nc_in: Int32, leaf_in: Int32,
):
    """Block `k` = leaf `k` of [X | X V^T] (m = d + nc columns, the
    projection formed on the fly, never stored): per column the leaf mean,
    then the leaf's centered sum of squares, in the colsum kernel's
    (s, j) chains. One read of the leaf for the sums, one for the squares."""
    var n = Int(n_in)
    var d = Int(d_in)
    var m = d + Int(nc_in)
    var leaf = Int(leaf_in)
    var k = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var r0 = k * leaf
    var r1 = min(n, r0 + leaf)
    var cnt = Float32(r1 - r0)
    var sh = stack_allocation[BM_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var mu = stack_allocation[BM_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    if m <= BM_TPB:
        var chains = bm_sub_chains(m)
        var s = t // m
        var j = t - s * m
        var acc = Float32(0.0)
        if s < chains:
            var r = r0 + s
            while r < r1:
                acc = bm_add(acc, _tsvd_value(x, v, r, d, j))
                r += chains
        sh[t] = acc
        barrier()
        if t < m:
            var tot = sh[t]
            for q in range(1, chains):
                tot = bm_add(tot, sh[q * m + t])
            mu[t] = bm_div(tot, cnt)
        barrier()
        var acc2 = Float32(0.0)
        if s < chains:
            var mj = mu[j]
            var r = r0 + s
            while r < r1:
                var c = bm_sub(_tsvd_value(x, v, r, d, j), mj)
                acc2 = bm_fma(c, c, acc2)
                r += chains
        barrier()
        sh[t] = acc2
        barrier()
        if t < m:
            var tot = sh[t]
            for q in range(1, chains):
                tot = bm_add(tot, sh[q * m + t])
            leaf_mean.unsafe_store(k * m + t, mu[t])
            leaf_m2.unsafe_store(k * m + t, tot)
    else:
        var j = t
        while j < m:
            var acc = Float32(0.0)
            for r in range(r0, r1):
                acc = bm_add(acc, _tsvd_value(x, v, r, d, j))
            var mj = bm_div(acc, cnt)
            var acc2 = Float32(0.0)
            for r in range(r0, r1):
                var c = bm_sub(_tsvd_value(x, v, r, d, j), mj)
                acc2 = bm_fma(c, c, acc2)
            leaf_mean.unsafe_store(k * m + j, mj)
            leaf_m2.unsafe_store(k * m + j, acc2)
            j += BM_TPB


# ---------------------------------------------------------------------------
# Binary-counter folds
# ---------------------------------------------------------------------------


def bm_sum_level_kernel(
    dst: BmPtr, src: BmPtr, rem: BmPtr, count_in: Int32, cells_in: Int32, level_in: Int32,
):
    """One fold level: node p = src[2p] + src[2p + 1]; an odd last node is
    this level's remainder."""
    var count = Int(count_in)
    var cells = Int(cells_in)
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var q = (count + 1) // 2
    if g >= q * cells:
        return
    var p = g // cells
    var c = g - p * cells
    if 2 * p + 1 < count:
        dst.unsafe_store(p * cells + c, bm_add(src.unsafe_load(2 * p * cells + c), src.unsafe_load((2 * p + 1) * cells + c)))
    else:
        rem.unsafe_store(Int(level_in) * cells + c, src.unsafe_load(2 * p * cells + c))


def bm_sum_final_kernel(out: BmPtr, rem: BmPtr, cells_in: Int32, mask_in: Int32, divisor_in: Int32):
    """Remainders right to left (`rem[L] + acc`, ascending L); then `/ divisor`
    when `divisor > 0`."""
    var cells = Int(cells_in)
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= cells:
        return
    var mask = Int(mask_in)
    var acc = Float32(0.0)
    var have = False
    for lv in range(BM_MAX_LEVELS):
        if (mask >> lv) & 1 != 0:
            var r = rem.unsafe_load(lv * cells + c)
            acc = bm_add(r, acc) if have else r
            have = True
    if Int(divisor_in) > 0:
        acc = bm_div(acc, Float32(Int(divisor_in)))
    out.unsafe_store(c, acc)


@always_inline
def _cell_ij[diag: Bool](e: Int, m: Int) -> Tuple[Int, Int]:
    comptime if diag:
        return (e, e)
    var i = e // m
    return (i, e - i * m)


def bm_chan_level_kernel[diag: Bool](
    dst_m2: BmPtr, dst_mean: BmPtr, src_m2: BmPtr, src_mean: BmPtr, rem_m2: BmPtr, rem_mean: BmPtr,
    count_in: Int32, m_in: Int32, cells_in: Int32, level_in: Int32, n_in: Int32, leaf_in: Int32,
):
    """One Chan fold level: node p = merge(src[2p], src[2p + 1]) over every
    cell's M2 and every column's mean (each from the pre-merge means); an odd
    last node is this level's remainder."""
    var count = Int(count_in)
    var m = Int(m_in)
    var cells = Int(cells_in)
    var level = Int(level_in)
    var width = cells + m
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var q = (count + 1) // 2
    if g >= q * width:
        return
    var p = g // width
    var e = g - p * width
    var lf = 2 * p
    var rt = lf + 1
    if rt < count:
        var na = bm_node_rows(Int(n_in), Int(leaf_in), level, lf)
        var nb = bm_node_rows(Int(n_in), Int(leaf_in), level, rt)
        if e < cells:
            var ij = _cell_ij[diag](e, m)
            dst_m2.unsafe_store(p * cells + e, bm_chan_m2(
                src_m2.unsafe_load(lf * cells + e), src_m2.unsafe_load(rt * cells + e),
                src_mean.unsafe_load(lf * m + ij[0]), src_mean.unsafe_load(rt * m + ij[0]),
                src_mean.unsafe_load(lf * m + ij[1]), src_mean.unsafe_load(rt * m + ij[1]),
                na, nb,
            ))
        else:
            var col = e - cells
            dst_mean.unsafe_store(p * m + col, bm_chan_mean(
                src_mean.unsafe_load(lf * m + col), src_mean.unsafe_load(rt * m + col), na, nb,
            ))
    else:
        if e < cells:
            rem_m2.unsafe_store(level * cells + e, src_m2.unsafe_load(lf * cells + e))
        else:
            var col = e - cells
            rem_mean.unsafe_store(level * m + col, src_mean.unsafe_load(lf * m + col))


def bm_chan_final_kernel[diag: Bool](
    out_m2: BmPtr, out_mean: BmPtr, rem_m2: BmPtr, rem_mean: BmPtr,
    m_in: Int32, cells_in: Int32, mask_in: Int32, leaves_in: Int32, n_in: Int32, leaf_in: Int32,
    divisor_in: Int32,
):
    """Thread per cell: the remainders merged right to left, (rows, mean_i,
    mean_j, M2) carried; `out_m2 = M2 / divisor`; diagonal cells also write
    the merged mean."""
    var m = Int(m_in)
    var cells = Int(cells_in)
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= cells:
        return
    var ij = _cell_ij[diag](c, m)
    var i = ij[0]
    var j = ij[1]
    var mask = Int(mask_in)
    var leaves = Int(leaves_in)
    var acc = Float32(0.0)
    var mi = Float32(0.0)
    var mj = Float32(0.0)
    var rows = 0
    var have = False
    for lv in range(BM_MAX_LEVELS):
        if (mask >> lv) & 1 != 0:
            var node = (leaves >> lv) - 1
            var nr = bm_node_rows(Int(n_in), Int(leaf_in), lv, node)
            var r2 = rem_m2.unsafe_load(lv * cells + c)
            var ri = rem_mean.unsafe_load(lv * m + i)
            var rj = rem_mean.unsafe_load(lv * m + j)
            if have:
                acc = bm_chan_m2(r2, acc, ri, mi, rj, mj, nr, rows)
                mi = bm_chan_mean(ri, mi, nr, rows)
                mj = bm_chan_mean(rj, mj, nr, rows)
                rows += nr
            else:
                acc = r2
                mi = ri
                mj = rj
                rows = nr
                have = True
    out_m2.unsafe_store(c, bm_div(acc, Float32(Int(divisor_in))))
    if i == j:
        out_mean.unsafe_store(i, mi)


# ---------------------------------------------------------------------------
# Launchers. Each allocates its workspace and SYNCHRONIZES before release.
# ---------------------------------------------------------------------------


def _grid(total: Int) -> Int:
    return max(1, (total + BM_TPB - 1) // BM_TPB)


def bm_sum_fold(
    ctx: DeviceContext, mut part: DeviceBuffer[DType.float32], leaves: Int, cells: Int,
    out: BmPtr, divisor: Int,
) raises:
    """`out[c]` = the binary-counter fold of `part[k * cells + c]` over
    `leaves` leaves (then `/ divisor` when `divisor > 0`). `part` is
    overwritten."""
    var ws = ctx.enqueue_create_buffer[DType.float32](max(1, (leaves // 2) * cells))
    var rem = ctx.enqueue_create_buffer[DType.float32](BM_MAX_LEVELS * max(cells, 1))
    var count = leaves
    var level = 0
    var mask = 0
    var flip = False
    while count > 0:
        var src = _bp(ws) if flip else _bp(part)
        var dst = _bp(part) if flip else _bp(ws)
        ctx.enqueue_function[bm_sum_level_kernel](
            dst, src, _bp(rem), Int32(count), Int32(cells), Int32(level),
            grid_dim=_grid(((count + 1) // 2) * cells), block_dim=BM_TPB,
        )
        if count % 2 == 1:
            mask |= 1 << level
        count //= 2
        level += 1
        flip = not flip
    ctx.enqueue_function[bm_sum_final_kernel](
        out, _bp(rem), Int32(cells), Int32(mask), Int32(divisor),
        grid_dim=_grid(cells), block_dim=BM_TPB,
    )
    ctx.synchronize()
    _ = ws^
    _ = rem^


def bm_chan_fold[diag: Bool](
    ctx: DeviceContext, mut m2: DeviceBuffer[DType.float32], mut mean: DeviceBuffer[DType.float32],
    leaves: Int, m: Int, n: Int, leaf: Int, out_m2: BmPtr, out_mean: BmPtr, divisor: Int,
) raises:
    """Chan fold of per-leaf (mean[k * m + col], M2[k * cells + cell]) over
    `leaves` leaves: `out_m2 = M2 / divisor`, `out_mean` the merged means.
    `diag`: cells = m, cell c is (c, c); else cells = m * m. Inputs are
    overwritten."""
    var cells = m if diag else m * m
    var half = max(1, leaves // 2)
    var ws2 = ctx.enqueue_create_buffer[DType.float32](half * cells)
    var wsm = ctx.enqueue_create_buffer[DType.float32](half * m)
    var rem2 = ctx.enqueue_create_buffer[DType.float32](BM_MAX_LEVELS * cells)
    var remm = ctx.enqueue_create_buffer[DType.float32](BM_MAX_LEVELS * m)
    var count = leaves
    var level = 0
    var mask = 0
    var flip = False
    comptime level_k = bm_chan_level_kernel[diag]
    comptime final_k = bm_chan_final_kernel[diag]
    while count > 0:
        var s2 = _bp(ws2) if flip else _bp(m2)
        var sm = _bp(wsm) if flip else _bp(mean)
        var d2 = _bp(m2) if flip else _bp(ws2)
        var dm = _bp(mean) if flip else _bp(wsm)
        ctx.enqueue_function[level_k](
            d2, dm, s2, sm, _bp(rem2), _bp(remm),
            Int32(count), Int32(m), Int32(cells), Int32(level), Int32(n), Int32(leaf),
            grid_dim=_grid(((count + 1) // 2) * (cells + m)), block_dim=BM_TPB,
        )
        if count % 2 == 1:
            mask |= 1 << level
        count //= 2
        level += 1
        flip = not flip
    ctx.enqueue_function[final_k](
        out_m2, out_mean, _bp(rem2), _bp(remm),
        Int32(m), Int32(cells), Int32(mask), Int32(leaves), Int32(n), Int32(leaf), Int32(divisor),
        grid_dim=_grid(cells), block_dim=BM_TPB,
    )
    ctx.synchronize()
    _ = ws2^
    _ = wsm^
    _ = rem2^
    _ = remm^


def bm_column_mean(ctx: DeviceContext, mu: BmPtr, x: BmPtr, n: Int, d: Int) raises:
    """C01_MEAN: column means of row-major `x` (n x d): leaf column sums
    (`bm_mean_leaf_rows`), the binary-counter fold, one division by n."""
    if n < 1 or d < 1:
        return
    var leaf = bm_mean_leaf_rows(n)
    var leaves = bm_leaf_count(n, leaf)
    var part = ctx.enqueue_create_buffer[DType.float32](leaves * d)
    ctx.enqueue_function[bm_leaf_colsum_kernel](
        _bp(part), x, Int32(n), Int32(d), Int32(leaf),
        grid_dim=leaves, block_dim=BM_TPB,
    )
    bm_sum_fold(ctx, part, leaves, d, mu, n)
    _ = part^


def bm_centered_gram_panels(
    ctx: DeviceContext, out: BmPtr, x: BmPtr, center: BmPtr, n: Int, d: Int, leaf: Int,
) raises:
    """`out[i * d + j]` = sum over rows of (x_ri - c_i)(x_rj - c_j), the
    centered (unscaled) Gram around one given center, in leaves of `leaf`
    rows folded by the binary counter: the value of the per-cell reference
    cells (`x_decomp/classical_cells.mojo::centered_gram_cell` at leaf 256,
    `core/classical_centered.mojo::centered_gram_v1_cell` at
    `contract_leaf_size(n)`), computed row-parallel."""
    if d < 1:
        return
    var cells = d * d
    var leaves = bm_leaf_count(n, leaf)
    if leaves == 0:
        # no rows: every cell is +0.0 (the reference cells' empty answer);
        # the final kernel with an empty mask writes exactly that
        ctx.enqueue_function[bm_sum_final_kernel](
            out, out, Int32(cells), Int32(0), Int32(0),
            grid_dim=_grid(cells), block_dim=BM_TPB,
        )
        ctx.synchronize()
        return
    var tiles = (d + BM_TILE - 1) // BM_TILE
    var part = ctx.enqueue_create_buffer[DType.float32](leaves * cells)
    ctx.enqueue_function[bm_leaf_gram_kernel](
        _bp(part), _bp(part), x, center,
        Int32(n), Int32(d), Int32(leaf), Int32(tiles), Int32(0),
        grid_dim=(leaves, tiles * (tiles + 1) // 2, 1), block_dim=BM_TPB,
    )
    bm_sum_fold(ctx, part, leaves, cells, out, 0)
    _ = part^


def bm_onepass_covariance(
    ctx: DeviceContext, mu: BmPtr, cov: BmPtr, x: BmPtr, n: Int, d: Int,
) raises:
    """C23 (PCA): the mean and the (n - 1)-scaled covariance in one blocked
    read: each leaf centers on its own means, and the leaves merge by Chan's
    update in the binary-counter order. Needs n >= 2."""
    if d < 1 or n < 2:
        return
    var cells = d * d
    var leaf = bm_onepass_leaf_rows(n, cells)
    var leaves = bm_leaf_count(n, leaf)
    var tiles = (d + BM_TILE - 1) // BM_TILE
    var part = ctx.enqueue_create_buffer[DType.float32](leaves * cells)
    var means = ctx.enqueue_create_buffer[DType.float32](leaves * d)
    ctx.enqueue_function[bm_leaf_gram_kernel](
        _bp(part), _bp(means), x, x,
        Int32(n), Int32(d), Int32(leaf), Int32(tiles), Int32(1),
        grid_dim=(leaves, tiles * (tiles + 1) // 2, 1), block_dim=BM_TPB,
    )
    bm_chan_fold[False](ctx, part, means, leaves, d, n, leaf, cov, mu, n - 1)
    _ = part^
    _ = means^


def bm_tsvd_variances(
    ctx: DeviceContext, var_out: BmPtr, x: BmPtr, v: BmPtr, n: Int, d: Int, nc: Int,
) raises:
    """TSVD_FUSED_STATS: `var_out[0:d]` = np.var(X, axis=0) and
    `var_out[d:d+nc]` = np.var(X V^T, axis=0) (ddof 0), the projection formed
    inside the leaf kernel, the moments merged by Chan's update."""
    var m = d + nc
    if m < 1 or n < 1:
        return
    var leaf = bm_onepass_leaf_rows(n, m)
    var leaves = bm_leaf_count(n, leaf)
    var lmean = ctx.enqueue_create_buffer[DType.float32](leaves * m)
    var lm2 = ctx.enqueue_create_buffer[DType.float32](leaves * m)
    var outm = ctx.enqueue_create_buffer[DType.float32](m)
    ctx.enqueue_function[bm_tsvd_leaf_kernel](
        _bp(lmean), _bp(lm2), x, v, Int32(n), Int32(d), Int32(nc), Int32(leaf),
        grid_dim=leaves, block_dim=BM_TPB,
    )
    bm_chan_fold[True](ctx, lm2, lmean, leaves, m, n, leaf, var_out, _bp(outm), n)
    _ = lmean^
    _ = lm2^
    _ = outm^
