# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`X^T dZ` with the pinned fold of `xty_kernel` / `xtdz_multi_kernel`, read
row-coalesced (lane apple-identical-steps, 2026-09-26). Apple, IDENTICAL.

THE CONTRACT BEING KEPT. `core/column_stats.mojo::xty_kernel` (C == 1) and
`glm/impl/qn/glm_softmax.mojo::xtdz_multi_kernel` (C > 1) define output cell
`b = c + C * j` as: for each `tid in [0, STATS_TPB)`, the chain
`acc = identical_mul_add(x[r * D + j], dz[c + C * r], acc)` over
`r = tid, tid + STATS_TPB, ...` ascending from `acc = 0.0`, then
`ftz(pinned_block_sum[STATS_TPB](acc))` over the `STATS_TPB` partials.

WHAT MOVES. Only which hardware thread runs which chain, and when its loads
are issued. Those kernels run one block per cell, so the threads of a SIMD
group read one column of X at a stride of D floats: every load is its own
cache line, and X is re-read from memory once per cell. Measured on the M4:
`xty` 1,000,000 x 220 = 99.8 ms for 880 MB of X, `xtdz` 500,000 x 54 x 7 =
170.6 ms. Here pass 1 gives each block `S` whole residues `tid` and every
cell, cells fastest, so a SIMD group reads consecutive floats of one row
and all the blocks together sweep X once, front to back; each chain's loads
are issued `STRIDED_UNROLL` rows ahead (`core/strided_walk.mojo`). Pass 1
stores the `STATS_TPB` partials of every cell; pass 2 is the unchanged fold,
one `STATS_TPB` block per cell, every thread reaching it with its own
chain's value. Same chains, same fold, same bits.

`-D MOJOLEARN_APPLE_STEP_UNROLL_OFF` turns this off with the unroll; the
caller then launches the one-block-per-cell kernel.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz
from core.column_stats import STATS_TPB, column_mean_kernel, xty_kernel
from core.host_tile_fold import IDN_XTY_TILED, XTY_TILE_ROWS, xty_tiles
from core.pinned_reduce import pinned_block_sum
from core.strided_walk import (
    APPLE_IDENTICAL_STEP_UNROLL,
    STRIDED_UNROLL,
    strided_ftz_sum,
    strided_mul_add,
)


#: Pass 1 needs `cells <= XTDZ_CO_MAX_CELLS` so one block holds every cell of
#: one residue.
comptime XTDZ_CO_MAX_CELLS = 1024
#: Target threads per pass-1 block; `S = max(1, this // cells)` residues.
comptime XTDZ_CO_BLOCK_TARGET = 256


def xtdz_coalesced_applies(d: Int, c: Int) -> Bool:
    comptime if not APPLE_IDENTICAL_STEP_UNROLL:
        return False
    return d >= 1 and c >= 1 and d * c <= XTDZ_CO_MAX_CELLS


def xtdz_coalesced_workspace_floats(d: Int, c: Int) -> Int:
    return d * c * STATS_TPB


def xtdz_partial_kernel(
    partial: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    dz: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    d_in: Int32,
    c_in: Int32,
    s_in: Int32,
):
    """Pass 1: thread `(s, b)`, `b` fastest, runs chain `(b, tid)` for
    `tid = block * S + s` and stores it at `partial[b * STATS_TPB + tid]`."""
    var D = Int(d_in)
    var C = Int(c_in)
    var cells = D * C
    var local = Int(thread_idx.x)
    var s = local // cells
    var b = local - s * cells
    var tid = Int(block_idx.x) * Int(s_in) + s
    if s >= Int(s_in) or tid >= STATS_TPB:
        return
    var c = b % C
    var j = b // C
    var acc = strided_mul_add[STATS_TPB](x, D, j, dz, C, c, Int(n_rows_in), tid)
    partial.unsafe_store(b * STATS_TPB + tid, acc)


def xtdz_fold_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    partial: MutPointer[Float32, MutAnyOrigin],
):
    """Pass 2: block `b`, `STATS_TPB` threads, the pinned fold of
    `xty_kernel` over the chains pass 1 stored."""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var acc = partial.unsafe_load(b * STATS_TPB + tid)
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        out_v.unsafe_store(b, s0)


def xtdz_coalesced(
    ctx: DeviceContext,
    mut out_v: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut dz: DeviceBuffer[DType.float32],
    mut partial: DeviceBuffer[DType.float32],
    n_rows: Int,
    d: Int,
    c: Int,
) raises:
    """`out_v[c + C * j] = X^T dZ` bit for bit as `xty_kernel` (`c == 1`) or
    `xtdz_multi_kernel`. `partial` holds at least
    `xtdz_coalesced_workspace_floats(d, c)` floats. Asynchronous; call only
    where `xtdz_coalesced_applies(d, c)`."""
    var cells = d * c
    var s = max(1, XTDZ_CO_BLOCK_TARGET // cells)
    ctx.enqueue_function[xtdz_partial_kernel](
        partial.unsafe_ptr(), x.unsafe_ptr(), dz.unsafe_ptr(),
        Int32(n_rows), Int32(d), Int32(c), Int32(s),
        grid_dim=((STATS_TPB + s - 1) // s, 1, 1),
        block_dim=(s * cells, 1, 1),
    )
    ctx.enqueue_function[xtdz_fold_kernel](
        out_v.unsafe_ptr(), partial.unsafe_ptr(),
        grid_dim=(cells, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )


# ===========================================================================
# IDN_XTY_TILED (lane fam2-shared, 2026-10-04): `xty_launch` and
# `column_mean_launch` in the row-tile-then-fold order of
# `core/host_tile_fold.mojo` (IDENTICAL, every GPU and the host column). The
# forms above and `xty_kernel` / `column_mean_kernel` give each output column
# STATS_TPB chains, each walking n / STATS_TPB rows one add after another: the
# chain length grows with n and at most STATS_TPB * D threads work. Here pass 1
# runs one chain per (tile, column), XTY_TILE_ROWS rows long whatever n is,
# columns fastest so a SIMD group reads consecutive floats of one row and X is
# read once; pass 2 folds each column's tile partials (lane t takes tiles t,
# t + STATS_TPB, ..., then the pinned halving tree). BITS MOVE, on NVIDIA,
# AMD, Apple and the host column together (`host_xty_tiled`,
# `host_column_mean_tiled`). `-D MOJOLEARN_IDN_XTY_TILED_OFF` or the master
# `-D MOJOLEARN_IDN_ALL_OFF` restores the old order (device and host).
# ===========================================================================

comptime XTY_TILE_TPB = 256


def xty_tile_partial_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    tiles_in: Int32,
    is_mean: Int32,
):
    """Pass 1: thread `(k, j)`, `j` fastest, runs column `j`'s chain over
    tile `k`'s rows ascending from 0.0 (`identical_mul_add(x, y, acc)`, or
    `ftz(acc + x)` when `is_mean`; `y` is then unread) and stores
    `part[j * tiles + k]`."""
    var n = Int(n_in)
    var D = Int(d_in)
    var tiles = Int(tiles_in)
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var k = gid // D
    var j = gid - k * D
    if k >= tiles:
        return
    var r0 = k * XTY_TILE_ROWS
    var r1 = min(n, r0 + XTY_TILE_ROWS)
    var acc = Float32(0.0)
    if is_mean != 0:
        acc = strided_ftz_sum[1](x, D, j, r1, r0, Float32(0.0))
    else:
        acc = strided_mul_add[1](x, D, j, y, 1, 0, r1, r0)
    part.unsafe_store(j * tiles + k, acc)


def xty_tile_fold_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    tiles_in: Int32,
    count_in: Int32,
    is_mean: Int32,
):
    """Pass 2: block `j`, `STATS_TPB` lanes, the pinned fold of column `j`'s
    tile partials; the mean divides by `count_in` (`ftz(s0 / n)`, the
    quotient `column_mean_kernel` forms). Every lane reaches the fold."""
    var tiles = Int(tiles_in)
    var j = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var acc = strided_ftz_sum[STATS_TPB](part, 1, j * tiles, tiles, tid, Float32(0.0))
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        if is_mean != 0:
            var m = ftz(s0 / Float32(Int(count_in)))
            out_v.unsafe_store(j, m)
        else:
            out_v.unsafe_store(j, s0)


def xty_tiled(
    ctx: DeviceContext,
    out_p: MutPointer[Float32, MutAnyOrigin],
    x_p: MutPointer[Float32, MutAnyOrigin],
    y_p: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int,
    n_cols: Int,
    is_mean: Bool,
) raises:
    """`out = A^T y` (or the column means of `x`; pass `x`'s pointer again
    as `y_p`, it is not read) in the tile order. Allocates its `tiles * n_cols`
    partials here and SYNCHRONIZES before releasing them."""
    var tiles = xty_tiles(n_rows)
    var cells = tiles * n_cols
    var ws = ctx.enqueue_create_buffer[DType.float32](max(cells, 1))
    var mean_flag = Int32(1) if is_mean else Int32(0)
    ctx.enqueue_function[xty_tile_partial_kernel](
        ws.unsafe_ptr(), x_p, y_p,
        Int32(n_rows), Int32(n_cols), Int32(tiles), mean_flag,
        grid_dim=((cells + XTY_TILE_TPB - 1) // XTY_TILE_TPB, 1, 1),
        block_dim=(XTY_TILE_TPB, 1, 1),
    )
    ctx.enqueue_function[xty_tile_fold_kernel](  # small-launch(count_in: the mean divisor only): folds xty_tiles(n) tile partials per column, never walks n
        out_p, ws.unsafe_ptr(), Int32(tiles), Int32(n_rows), mean_flag,
        grid_dim=(n_cols, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = ws^


def xty_launch(
    ctx: DeviceContext,
    mut out_v: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
) raises:
    """`out_v = A^T b`: `xty_kernel`'s value on every column. On Apple
    IDENTICAL it is the coalesced two-pass form with a workspace allocated
    here, and it SYNCHRONIZES before releasing it; everywhere else it is the
    one-block-per-column launch, asynchronous as before. IDN_XTY_TILED
    (IDENTICAL, default): the tile order of `xty_tiled`, every width."""
    comptime if IDN_XTY_TILED:
        if n_rows >= 1 and n_cols >= 1:
            xty_tiled(
                ctx, out_v.unsafe_ptr(), x.unsafe_ptr(), y.unsafe_ptr(),
                n_rows, n_cols, False,
            )
            return
    if xtdz_coalesced_applies(n_cols, 1):
        var ws = ctx.enqueue_create_buffer[DType.float32](
            xtdz_coalesced_workspace_floats(n_cols, 1)
        )
        xtdz_coalesced(ctx, out_v, x, y, ws, n_rows, n_cols, 1)
        ctx.synchronize()
        _ = ws^
        return
    ctx.enqueue_function[xty_kernel](
        out_v.unsafe_ptr(),
        x.unsafe_ptr(),
        y.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )


# ===========================================================================
# `column_mean_kernel`, read row-coalesced (lane/gap-classical-nv,
# 2026-10-02). That kernel runs one block per column, each SIMD group reading
# one column at a stride of D floats (PCA's mean over 4M rows: 11 blocks on
# a 142-SM L40S at taxi's width). Pass 1 here runs the SAME per-thread chain
# `acc += x[r * D + j]` over `r = tid, tid + STATS_TPB, ...` ascending from
# 0.0, cells fastest so a SIMD group reads consecutive floats of one row,
# loads issued STRIDED_UNROLL rows ahead; pass 2 is the kernel's unchanged
# tail, `ftz(pinned_block_sum)` then `ftz(s0 / n_rows)`. Same chains, same
# fold, same quotient, same bits. Taken where `xtdz_coalesced_applies(D, 1)`;
# `-D MOJOLEARN_APPLE_STEP_UNROLL_OFF` keeps the one-block-per-column kernel.
# ===========================================================================


def mean_partial_kernel(
    partial: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    d_in: Int32,
    s_in: Int32,
):
    """Pass 1: thread `(s, j)`, `j` fastest, runs column `j`'s chain `tid =
    block * S + s` and stores it at `partial[j * STATS_TPB + tid]`."""
    var D = Int(d_in)
    var local = Int(thread_idx.x)
    var s = local // D
    var j = local - s * D
    var tid = Int(block_idx.x) * Int(s_in) + s
    if s >= Int(s_in) or tid >= STATS_TPB:
        return
    var n = Int(n_rows_in)
    comptime U = STRIDED_UNROLL
    comptime ST = STATS_TPB
    var acc = Float32(0.0)
    var r = tid
    while r + (U - 1) * ST < n:
        var t = InlineArray[Float32, U](fill=Float32(0.0))
        comptime for u in range(U):
            t[u] = x.unsafe_load((r + u * ST) * D + j)
        comptime for u in range(U):
            acc += t[u]
        r += U * ST
    while r < n:
        acc += x.unsafe_load(r * D + j)
        r += ST
    partial.unsafe_store(j * STATS_TPB + tid, acc)


def mean_fold_kernel(
    mu: MutPointer[Float32, MutAnyOrigin],
    partial: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
):
    """Pass 2: block `j`, `STATS_TPB` threads, `column_mean_kernel`'s fold
    and quotient over the chains pass 1 stored."""
    var j = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var acc = partial.unsafe_load(j * STATS_TPB + tid)
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        var m = ftz(s0 / Float32(Int(n_rows_in)))
        mu.unsafe_store(j, m)


def column_mean_launch(
    ctx: DeviceContext,
    mut mu: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
) raises:
    """`mu = column_mean_kernel(x)` bit for bit. The coalesced form
    allocates its workspace here and SYNCHRONIZES before releasing it;
    otherwise the one-block-per-column launch, asynchronous as before.
    IDN_XTY_TILED (IDENTICAL, default): the tile order of `xty_tiled`."""
    comptime if IDN_XTY_TILED:
        if n_rows >= 1 and n_cols >= 1:
            var xp = x.unsafe_ptr()
            xty_tiled(ctx, mu.unsafe_ptr(), xp, xp, n_rows, n_cols, True)
            return
    if xtdz_coalesced_applies(n_cols, 1):
        var ws = ctx.enqueue_create_buffer[DType.float32](n_cols * STATS_TPB)
        var s = max(1, XTDZ_CO_BLOCK_TARGET // n_cols)
        ctx.enqueue_function[mean_partial_kernel](
            ws.unsafe_ptr(), x.unsafe_ptr(), Int32(n_rows), Int32(n_cols), Int32(s),
            grid_dim=((STATS_TPB + s - 1) // s, 1, 1),
            block_dim=(s * n_cols, 1, 1),
        )
        ctx.enqueue_function[mean_fold_kernel](
            mu.unsafe_ptr(), ws.unsafe_ptr(), Int32(n_rows),
            grid_dim=(n_cols, 1, 1),
            block_dim=(STATS_TPB, 1, 1),
        )
        ctx.synchronize()
        _ = ws^
        return
    ctx.enqueue_function[column_mean_kernel](
        mu.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )
