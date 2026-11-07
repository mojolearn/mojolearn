# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The tile-order `X^T dZ` kernels of the QN objective (lane fam2-linear,
2026-10-04; lane debt-block-per-n, 2026-10-21), in a module of their own so
the single-device objective (`glm/impl/qn/glm_base.mojo` `linear_bwd`,
`enqueue_tiled`) and the GLM column shards (`glm/impl/qn/multi_gpu.mojo`
`gradient_columns`) launch the SAME kernels without an import cycle.

Lane ne2-i-j (2026-11-07, plan row N453): moved here unchanged from
glm_base.mojo. `QNTM_ROWS` is `QNT_ROWS` (256, `qn_tiled_rule.mojo`),
`qn_tiled_multi_tiles(n)` is `qnt_tiles(n)` (the same `(n + 255) // 256`),
`QNTM_TPB` is `QNT_TPB` (256; a launch width, it maps `gid` to the same
`(k, bl)` whatever its value). NO BIT MOVES.

A cell's chain reads `x[r * D + j]` for its own column `j` and `dz[r * C +
c]`; it never reads another column. So a column shard that packs columns
`[first, first + width)` into a `rows x width` buffer and runs
`qntm_tiled_into(.., width, C)` stores, in its cells `c + C * (j - first)`,
the words the one-device call stores in cells `c + C * j`.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz
from core.column_stats import STATS_TPB
from core.pinned_reduce import pinned_block_sum
from core.strided_walk import strided_ftz_sum, strided_mul_add
from glm.impl.qn.qn_tiled_rule import QNTM_ROWS, qn_tiled_multi_tiles, qntm_chunk_cells

#: threads per block of pass 1 (= glm_base QNT_TPB)
comptime QNTM_TPB = 256


def qntm_partial_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    dz: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    c_in: Int32,
    tiles_in: Int32,
    b0_in: Int32,
    cnt_in: Int32,
):
    """Pass 1: thread `(k, bl)`, `bl` fastest, the `X^T dZ` cell `b = b0 +
    bl` (`b = c + C*j`) over tile `k`'s rows ascending. Stores
    `part[bl * tiles + k]`. The cells `[b0, b0 + cnt)` are one chunk of
    `qntm_tiled_into`; a chunk never changes a cell's chain."""
    var n = Int(n_in)
    var D = Int(d_in)
    var C = Int(c_in)
    var tiles = Int(tiles_in)
    var cnt = Int(cnt_in)
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var k = gid // cnt
    var bl = gid - k * cnt
    if k >= tiles:
        return
    var b = Int(b0_in) + bl
    var c = b % C
    var j = b // C
    var r0 = k * QNTM_ROWS
    var r1 = min(n, r0 + QNTM_ROWS)
    var acc = strided_mul_add[1](x, D, j, dz, C, c, r1, r0)
    part.unsafe_store(bl * tiles + k, acc)


def qntm_fold_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    tiles_in: Int32,
    b0_in: Int32,
):
    """Pass 2: block `bl`, STATS_TPB lanes, the pinned fold of cell `b0 +
    bl`'s tile partials into `out_v[b0 + bl]` (the cuBLAS epilogue follows
    as before)."""
    var tiles = Int(tiles_in)
    var bl = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var acc = strided_ftz_sum[STATS_TPB](part, 1, bl * tiles, tiles, tid, Float32(0.0))
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        out_v.unsafe_store(Int(b0_in) + bl, s0)


def qntm_tiled_into(
    ctx: DeviceContext,
    mut xtdz: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut dz: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    n_rows: Int,
    d: Int,
    C: Int,
) raises:
    """`xtdz[c + C*j] = X^T dZ` in the tile order (`qntm_partial_kernel`,
    then `qntm_fold_kernel` per cell), for any `C >= 1`: the order of the
    host column's `host_qnt_xty` (C == 1) and its tiled multinomial
    gradient (C > 1). The cells run in chunks of `qntm_chunk_cells` so the
    partials fit `ws` (`qntm_ws_floats` floats); every chunk is the same
    chains and fold, so the bits do not depend on the chunking.
    Asynchronous. Lane debt-block-per-n (2026-10-21): this replaced
    `xtdz_multi_kernel` / `xty_kernel`, one block per cell walking all n
    rows, wherever the gradient took them."""
    comptime assert not (is_defined["MOJOLEARN_QN_TILED_MULTI_OFF"]() or is_defined["MOJOLEARN_QN_TILED_MULTI_ALL"]()), "removed (lane debt-block-per-n, 2026-10-21): QN_TILED_MULTI_OFF selected xtdz_multi_kernel (one block per gradient cell walking all n rows) and QN_TILED_MULTI_ALL is now the rule"
    var cd = C * d
    if cd < 1:
        return
    var tiles = qn_tiled_multi_tiles(n_rows)
    var chunk = qntm_chunk_cells(n_rows, cd)
    var n_chunks = (cd + chunk - 1) // chunk
    for ci in range(n_chunks):  # driver-loop: over workspace-bounded chunks of gradient cells
        var b0 = ci * chunk
        var cnt = min(chunk, cd - b0)
        var threads = tiles * cnt
        if threads >= 1:
            ctx.enqueue_function[qntm_partial_kernel](
                ws.unsafe_ptr(), x.unsafe_ptr(), dz.unsafe_ptr(),
                Int32(n_rows), Int32(d), Int32(C), Int32(tiles),
                Int32(b0), Int32(cnt),
                grid_dim=((threads + QNTM_TPB - 1) // QNTM_TPB, 1, 1),
                block_dim=(QNTM_TPB, 1, 1),
            )
        ctx.enqueue_function[qntm_fold_kernel](
            xtdz.unsafe_ptr(), ws.unsafe_ptr(), Int32(tiles), Int32(b0),
            grid_dim=(cnt, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )
