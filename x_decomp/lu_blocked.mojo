# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The 128-wide blocked LU's device half (lane gap-linalg, 2026-10-08;
docs/plans/gaps-2026-10-08.md section 3.2). CANDIDATE, default off:
`-D MOJOLEARN_IDN_LU_NB128` on an IDENTICAL build; the order is stated once,
at `x_decomp/cells.mojo::IDN_LU_NB128` beside its host twin.

`launch_lu` (x_decomp/device.mojo) keeps its panel machinery and widens the
panel to LUB_NB = 128: `lu_fast_panel` (one launch per column: the pivot
partials folded in every block, the swap, the multipliers and the panel
update; generic in the panel bounds) and `lu_swaps_trsm_kernel` (the
trailing columns' swaps and U rows, one thread per column, generic in the
panel bounds). What this module adds:

  * `lub_trailing_update`: `A22 -= L21 U12` through gemm/block_update.mojo
    (pack, the identical GEMM at k = 128 = one leaf per cell, one
    subtraction), with the packed L column of a skipped step (act = 0)
    zeroed so the step contributes nothing, as the per-step kernels skipped
    it. Replaces `lu_trail_rb_kernel`'s per-cell register chains, which ran
    the trailing update at an unstaged scalar rate.
  * `lub_feed`: the solve's block feed (`trs_feed_kernel`: every cell past
    the finished diagonal block takes that block's rows) as one GEMM and one
    subtraction per block when there are at least LUB_FEED_MIN_RHS right-hand
    sides.

Launches per factorization at n = 8192: 64 panels x (128 column steps + 2
panel kernels + 1 pack pair + 1 GEMM + 1 subtract) ~ 8.5k, against ~41k for
the 32-wide route with its five kernels a column.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gemm.block_update import blk_gemm_sub, blk_workspace_floats
from gemm.contract import OP_NN, OP_TN
from x_decomp.cells import F32Ptr, I32Ptr, IDN_LU_NB128, LUB_FEED_MIN_RHS, LUB_NB


def lub_workspace_floats(n: Int) -> Int:
    """Floats the trailing updates of an `n x n` factorization need: the
    largest `blk_workspace_floats` over the panels (the first panel's is the
    largest product, but the GEMM share is sized per call, so every panel
    is asked). Never below 1."""
    var need = 1
    var k0 = 0
    while k0 < n:
        var k1 = min(k0 + LUB_NB, n)
        var t = n - k1
        if t > 0:
            var c = blk_workspace_floats(t, t, k1 - k0)
            if c > need:
                need = c
        k0 = k1
    return need


def lub_trailing_update(
    ctx: DeviceContext, a: F32Ptr, act: F32Ptr, k0: Int, k1: Int, n: Int, mut ws: DeviceBuffer[DType.float32]
) raises:
    """`a[k1:, k1:] -= a[k1:, k0:k1] . a[k0:k1, k1:]` (L21 times the U12 rows
    `lu_swaps_trsm_kernel` finished), the packed L column p zeroed where
    `act[k0 + p] == 0`. Enqueued, no wait."""
    var t = n - k1
    if t <= 0:
        return
    var w = k1 - k0
    # the GUARD=False instantiation never reads `info`; any pointer typed
    # Int32 serves, and nothing is allocated for it
    var no_info = I32Ptr(unsafe_from_address=Int(a))
    blk_gemm_sub[False](
        ctx, a, n, k1, k1,
        a, n, k1, k0,
        a, n, k0, k1,
        t, t, w, OP_NN, False, ws, no_info, act + k0, True,
    )


def lub_feed_workspace_floats(n: Int, nrhs: Int) -> Int:
    """Floats `lub_feed` needs over every block of an `n`-row solve with
    `nrhs` right-hand sides. Never below 1."""
    var need = 1
    var nblk = (n + LUB_NB - 1) // LUB_NB
    for q in range(nblk):
        var lo = q * LUB_NB
        var hi = min(n, lo + LUB_NB)
        var rows = max(n - hi, lo)
        if rows > 0:
            var c = blk_workspace_floats(rows, nrhs, hi - lo)
            if c > need:
                need = c
    return need


def lub_feed(
    ctx: DeviceContext, lu: F32Ptr, b: F32Ptr, n: Int, nrhs: Int, tri: Int, lo: Int, hi: Int,
    mut ws: DeviceBuffer[DType.float32],
) raises:
    """The rows the finished block [lo, hi) feeds (rows [hi, n) going
    forward, [0, lo) going backward) take the block's rows as one GEMM and
    one subtraction: `b[i, :] -= sum_j coef(i, j) b[j, :]`, j ascending
    over the block (the packed leaf's order on every triangle). tri as
    `trs_coef`: 0 unit L forward (coef lu[i, j]), 1 U backward (lu[i, j]),
    2 U^T forward (lu[j, i]), 3 unit L^T backward (lu[j, i]); the transposed
    coefficients are the stored block rows [lo, hi) x the fed columns,
    read through OP_TN. Enqueued, no wait."""
    var fwd = tri == 0 or tri == 2
    var i0 = hi if fwd else 0
    var rows = (n - hi) if fwd else lo
    if rows <= 0:
        return
    var k = hi - lo
    var no_info = I32Ptr(unsafe_from_address=Int(b))
    if tri >= 2:
        # A stored as the block rows [lo, hi) x columns [i0, i0 + rows): k x m
        blk_gemm_sub[False](
            ctx, b, nrhs, i0, 0,
            lu, n, lo, i0,
            b, nrhs, lo, 0,
            rows, nrhs, k, OP_TN, False, ws, no_info, lu, False,
        )
    else:
        # A stored as rows [i0, i0 + rows) x the block columns [lo, hi): m x k
        blk_gemm_sub[False](
            ctx, b, nrhs, i0, 0,
            lu, n, i0, lo,
            b, nrhs, lo, 0,
            rows, nrhs, k, OP_NN, False, ws, no_info, lu, False,
        )


def lub_feed_applies(nrhs: Int) -> Bool:
    """Whether the GEMM feed runs for this many right-hand sides (the width
    rule at `LUB_FEED_MIN_RHS`); False on a build without the define."""
    comptime if not IDN_LU_NB128:
        return False
    return nrhs >= LUB_FEED_MIN_RHS
