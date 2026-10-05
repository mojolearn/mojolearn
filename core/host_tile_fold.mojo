# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The row-tile-then-fold order of `core/xtdz_coalesced.mojo::xty_launch` and
`column_mean_launch` (lane fam2-shared, 2026-10-04), and its host column.

THE ORDER. Rows are cut into tiles of `XTY_TILE_ROWS`. Cell `(k, j)` (tile
`k`, column `j`) is one chain over the tile's rows ascending from `0.0`:
`identical_mul_add(x[r, j], y[r], acc)` for `A^T b`, `ftz(acc + x[r, j])` for
the column mean. Column `j`'s `tiles` partials then fold as
`glm/impl/qn/glm_base.mojo::qn_tile_sum_fold_kernel` folds: lane `t` of
`STATS_TPB` takes tiles `t, t + STATS_TPB, ...` (`ftz(acc + p)`), then the
halving tree of `pinned_block_sum`, flushed. The mean is `ftz(s0 / n)`.

This file imports no GPU module: the device kernels and the host oracles both
read the switch and the tile size from here, so one define moves NVIDIA, AMD,
Apple and the host column together.

`-D MOJOLEARN_IDN_XTY_TILED_OFF` (or the master `-D MOJOLEARN_IDN_ALL_OFF`)
restores the `STATS_TPB`-chain order on the device and on the host.
Candidate arms for the tile size (default 256 rows):
`-D MOJOLEARN_IDN_XTY_TILE_64`, `-D MOJOLEARN_IDN_XTY_TILE_1024`. Pass the
same define to the host build.
"""

from std.sys.compile import is_defined

from checks.kernel_matrix import (
    K_LIB_COLUMN_STATS,
    TARGET_COLUMN,
    lib_block_size_for,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
)


comptime IDN_XTY_TILED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_XTY_TILED_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)

comptime XTY_TILE_ROWS = (
    64 if is_defined["MOJOLEARN_IDN_XTY_TILE_64"]()
    else (1024 if is_defined["MOJOLEARN_IDN_XTY_TILE_1024"]() else 256)
)

#: `core/column_stats.mojo::STATS_TPB`, restated because that file imports
#: the GPU (the same read `decomposition/host/pca_oracle.mojo` makes).
comptime _HT_STATS_TPB = lib_block_size_for[K_LIB_COLUMN_STATS, TARGET_COLUMN]()


def xty_tiles(n: Int) -> Int:
    return (n + XTY_TILE_ROWS - 1) // XTY_TILE_ROWS


def _host_tile_halving(mut red: List[Float32]) -> Float32:
    """`pinned_block_sum`'s halving tree under IDENTICAL: `red[t] += red[t +
    step]`, `step = width / 2 .. 1`, plain float32 adds, unflushed."""
    var step = len(red) // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    return red[0]


def host_tile_fold(part: List[Float32], off: Int, tiles: Int) -> Float32:
    """`xty_tile_fold_kernel`'s fold of `part[off, off + tiles)`."""
    var red = List[Float32](length=_HT_STATS_TPB, fill=Float32(0.0))
    for t in range(_HT_STATS_TPB):
        var acc = Float32(0.0)
        var k = t
        while k < tiles:
            acc = ftz(acc + part[off + k])
            k += _HT_STATS_TPB
        red[t] = acc
    return ftz(_host_tile_halving(red))


def host_xty_tiled(
    x: List[Float32], y: List[Float32], n_rows: Int, n_cols: Int,
) -> List[Float32]:
    """`xty_launch` in the tile order: `A^T b`, `x` row-major `n_rows x
    n_cols`."""
    var tiles = xty_tiles(n_rows)
    var part = List[Float32](length=tiles * n_cols, fill=Float32(0.0))
    for k in range(tiles):
        var r0 = k * XTY_TILE_ROWS
        var r1 = min(n_rows, r0 + XTY_TILE_ROWS)
        for j in range(n_cols):
            var acc = Float32(0.0)
            for r in range(r0, r1):
                acc = identical_mul_add(x[r * n_cols + j], y[r], acc)
            part[j * tiles + k] = acc
    var out = List[Float32](length=n_cols, fill=Float32(0.0))
    for j in range(n_cols):
        out[j] = host_tile_fold(part, j * tiles, tiles)
    return out^


def host_column_mean_tiled(
    x: List[Float32], n_rows: Int, n_cols: Int,
) -> List[Float32]:
    """`column_mean_launch` in the tile order."""
    var tiles = xty_tiles(n_rows)
    var part = List[Float32](length=tiles * n_cols, fill=Float32(0.0))
    for k in range(tiles):
        var r0 = k * XTY_TILE_ROWS
        var r1 = min(n_rows, r0 + XTY_TILE_ROWS)
        for j in range(n_cols):
            var acc = Float32(0.0)
            for r in range(r0, r1):
                acc = ftz(acc + x[r * n_cols + j])
            part[j * tiles + k] = acc
    var mu = List[Float32](length=n_cols, fill=Float32(0.0))
    for j in range(n_cols):
        var s0 = host_tile_fold(part, j * tiles, tiles)
        mu[j] = ftz(s0 / Float32(n_rows))
    return mu^
