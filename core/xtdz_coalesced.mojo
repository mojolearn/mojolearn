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

from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz
from core.column_stats import STATS_TPB, xty_kernel
from core.pinned_reduce import pinned_block_sum
from core.strided_walk import APPLE_IDENTICAL_STEP_UNROLL, strided_mul_add


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
    one-block-per-column launch, asynchronous as before."""
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
