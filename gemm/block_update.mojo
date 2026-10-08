# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The identical GEMM on STRIDED sub-blocks with an ACCUMULATE epilogue
(lane gap-linalg, 2026-10-08; docs/plans/gaps-2026-10-08.md section 2).

`identical_gemm_into` (gemm/checks/gemm_identical.mojo) takes contiguous
row-major operands and writes a contiguous product. The blocked dense
factorizations (Cholesky, LU, the Householder tridiagonalization and its
back-transform) update a trailing SUB-BLOCK of a bigger matrix:

    C[r0 + i, c0 + j] -= sum_p op(A)[i, p] op(B)[p, j]

with A and B themselves sub-blocks (row stride `ld`). This module is that
entry, built ADDITIVELY on the contiguous one so the 10k-line GEMM and its
contract are untouched:

  1. `blk_pack_kernel`: each operand's sub-block copied to a contiguous
     scratch (a copy and nothing else: no arithmetic, nothing to pin). An
     optional column mask zeroes whole packed columns (LU's skipped steps).
  2. `identical_gemm_into` on the packed operands into a contiguous product
     G (m x n) in the same scratch.
  3. `blk_sub_kernel`: C[r0 + i, c0 + j] = ftz(ftz(C) - ftz(G[i, j])), one
     thread per cell, optionally only the cells on or below the diagonal of
     the FULL matrix (Cholesky's lower triangle).

THE FOLD, STATED ONCE. The contract's batch-composition invariance says
every product cell is a pure function of `k` and the two operand vectors:
leaves of `contract_leaf_size(k)` ascending, each leaf the chain
`acc = ftz(fma(ftz(a), ftz(b), acc))` from `+0.0`, the leaves folded by
the contract's balanced tree, the result `ftz`'d. For `k <= 128`
(`CONTRACT_K_LEAF_MIN`) that is ONE leaf: the plain ascending chain. The
subtraction then rounds once more. So a cell's new value is

    ftz( ftz(C_old) - ftz( chain_p=0..k-1 fma(ftz(a_ip), ftz(b_pj), .) ) )

on NVIDIA, on AMD and on the host column (`gemm/host/gemm_oracle.mojo`
`gemm_oracle_cell`), whatever plan the dispatcher picks and whatever the
launch geometry: that is the identity argument of every caller below, and
it is the ONLY place the trailing update's arithmetic is stated.

Cost against a fused `beta = 1` epilogue inside the GEMM: one write and one
read of G per block update (m n floats). For a 128-wide panel over an n^2
trailing square that is n^3 / (3 * 128) floats each way over the whole
factorization: at n = 8192, 5.7 GB each way, tens of milliseconds against
the hundreds the factorizations spend in the GEMM itself. The fused form is
a later GEMM-lane change; it would move no bits (the subtraction would still
be one rounding after the fold).
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz
from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.contract import OP_NN, OP_NT, OP_TN
from x_decomp.cells import F32Ptr, I32Ptr

#: Threads per block of the pack and subtract kernels (scheduling only: one
#: thread per element, no cross-thread arithmetic in either).
comptime BLK_TPB = 256


def _blk_grid(count: Int) -> Int:
    return (count + BLK_TPB - 1) // BLK_TPB if count > 0 else 1


def blk_pack_kernel(
    dst: F32Ptr,
    src: F32Ptr,
    dst_ld: Int32,
    dst_c0: Int32,
    ld: Int32,
    r0: Int32,
    c0: Int32,
    rows: Int32,
    cols: Int32,
    mask: F32Ptr,
    mask_on: Int32,
):
    """dst[i * dst_ld + dst_c0 + c] = src[(r0 + i) * ld + c0 + c] for the
    `rows x cols` sub-block; a column c whose mask[c] == 0 (mask_on != 0)
    is packed as +0.0 (LU: a step with a zero pivot contributes nothing).
    A copy: no rounding."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nr = Int(rows)
    var nc = Int(cols)
    if idx >= nr * nc:
        return
    var i = idx // nc
    var c = idx - i * nc
    var v = src.unsafe_load((Int(r0) + i) * Int(ld) + Int(c0) + c)
    if mask_on != Int32(0):
        if mask.unsafe_load(c) == Float32(0):
            v = Float32(0)
    dst.unsafe_store(i * Int(dst_ld) + Int(dst_c0) + c, v)


def blk_sub_kernel[GUARD: Bool](
    c: F32Ptr,
    g: F32Ptr,
    info: I32Ptr,
    ldc: Int32,
    r0: Int32,
    c0: Int32,
    m: Int32,
    n: Int32,
    lower: Int32,
):
    """C[r0 + i, c0 + j] = ftz(ftz(C) - ftz(G[i, j])), one thread per cell of
    the `m x n` product. `lower != 0`: only cells with column index <= row
    index in the FULL matrix (Cholesky). GUARD: the whole launch returns at
    once when `info[0] != 0` (a failed earlier panel: nothing more is
    written, the factor stays LAPACK's partial one)."""
    comptime if GUARD:
        if info[0] != Int32(0):
            return
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var mm = Int(m)
    var nn = Int(n)
    if idx >= mm * nn:
        return
    var i = idx // nn
    var j = idx - i * nn
    var row = Int(r0) + i
    var col = Int(c0) + j
    if lower != Int32(0) and col > row:
        return
    var at = row * Int(ldc) + col
    var cur = ftz(c.unsafe_load(at))
    var upd = ftz(g.unsafe_load(idx))
    c.unsafe_store(at, ftz(cur - upd))


def blk_operand_dims(op: Int, m: Int, n: Int, k: Int) -> Tuple[Int, Int, Int, Int]:
    """(a_rows, a_cols, b_rows, b_cols) of the STORED operand sub-blocks for
    `op` (the contract's four lines, gemm_identical.gemm_operand_strides):
    OP_NN / OP_NT store A as m x k, OP_TN as k x m; OP_NN / OP_TN store B as
    k x n, OP_NT as n x k."""
    var ar = m
    var ac = k
    if op == OP_TN:
        ar = k
        ac = m
    var br = k
    var bc = n
    if op == OP_NT:
        br = n
        bc = k
    return (ar, ac, br, bc)


def blk_workspace_floats(m: Int, n: Int, k: Int) -> Int:
    """Floats `blk_gemm_sub` needs for an `m x n` product over `k`: the two
    packed operands, the product, and `identical_gemm_into`'s own workspace
    (sized by its helper, never a guess: its docstring's warning). Never
    below 1."""
    if m <= 0 or n <= 0 or k <= 0:
        return 1
    var need = m * k + n * k + m * n + identical_gemm_workspace_max_floats(m, n, k)
    return max(1, need)


def blk_gemm_sub[GUARD: Bool = False](
    ctx: DeviceContext,
    c: F32Ptr,
    ldc: Int,
    cr0: Int,
    cc0: Int,
    a: F32Ptr,
    lda: Int,
    ar0: Int,
    ac0: Int,
    b: F32Ptr,
    ldb: Int,
    br0: Int,
    bc0: Int,
    m: Int,
    n: Int,
    k: Int,
    op: Int,
    lower: Bool,
    mut ws: DeviceBuffer[DType.float32],
    info: I32Ptr,
    mask: F32Ptr,
    mask_on: Bool,
) raises:
    """`C[cr0:cr0+m, cc0:cc0+n] -= op(A_blk) . op(B_blk)` where `A_blk` is
    the stored sub-block of `a` at (ar0, ac0) with row stride `lda` (its
    shape from `blk_operand_dims`), `B_blk` likewise, and `C` has row
    stride `ldc`. `lower` keeps only the cells on or below the full
    matrix's diagonal. `mask` (when `mask_on`) zeroes packed A columns.
    `ws` holds at least `blk_workspace_floats(m, n, k)`. Enqueued, no wait;
    the caller keeps every buffer alive past its own synchronize. Nothing
    to do when a dimension is 0."""
    if m <= 0 or n <= 0 or k <= 0:
        return
    var need = blk_workspace_floats(m, n, k)
    if len(ws) < need:
        raise Error(
            "blk_gemm_sub: the workspace holds " + String(len(ws))
            + " floats, the " + String(m) + " x " + String(n) + " x " + String(k)
            + " block update needs " + String(need) + " (blk_workspace_floats)"
        )
    var dims = blk_operand_dims(op, m, n, k)
    var a_cells = dims[0] * dims[1]
    var b_cells = dims[2] * dims[3]
    var off_a = 0
    var off_b = a_cells
    var off_g = a_cells + b_cells
    var off_w = off_g + m * n
    var pa = ws.create_sub_buffer[DType.float32](off_a, a_cells)
    var pb = ws.create_sub_buffer[DType.float32](off_b, b_cells)
    var pg = ws.create_sub_buffer[DType.float32](off_g, m * n)
    var wlen = len(ws) - off_w
    if wlen < 1:
        wlen = 1
    var pw = ws.create_sub_buffer[DType.float32](off_w, wlen)
    var pap = F32Ptr(unsafe_from_address=Int(pa.unsafe_ptr()))
    var pbp = F32Ptr(unsafe_from_address=Int(pb.unsafe_ptr()))
    var pgp = F32Ptr(unsafe_from_address=Int(pg.unsafe_ptr()))
    # A: the mask applies to the packed COLUMNS, which for OP_NN / OP_NT are
    # the contraction index p (the panel's steps); OP_TN callers pass no mask.
    ctx.enqueue_function[blk_pack_kernel](
        pap, a, Int32(dims[1]), Int32(0), Int32(lda), Int32(ar0), Int32(ac0),
        Int32(dims[0]), Int32(dims[1]), mask, Int32(1) if mask_on else Int32(0),
        grid_dim=_blk_grid(a_cells), block_dim=BLK_TPB,
    )
    ctx.enqueue_function[blk_pack_kernel](
        pbp, b, Int32(dims[3]), Int32(0), Int32(ldb), Int32(br0), Int32(bc0),
        Int32(dims[2]), Int32(dims[3]), mask, Int32(0),
        grid_dim=_blk_grid(b_cells), block_dim=BLK_TPB,
    )
    identical_gemm_into(ctx, pg, pa, pb, pw, m, n, k, op)
    ctx.enqueue_function[blk_sub_kernel[GUARD]](
        c, pgp, info, Int32(ldc), Int32(cr0), Int32(cc0), Int32(m), Int32(n),
        Int32(1) if lower else Int32(0),
        grid_dim=_blk_grid(m * n), block_dim=BLK_TPB,
    )
    _ = pa^
    _ = pb^
    _ = pg^
    _ = pw^
