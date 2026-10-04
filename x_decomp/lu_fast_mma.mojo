# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-w2-linalg (2026-10-04): LU_FAST_MMA (FAST+Apple default, rollback `-D MOJOLEARN_LU_FAST_MMA_OFF`),
the blocked LU with delayed trailing updates on the Apple matrix unit (FAST
on Apple only; needs LU_FAST_STEP1, so `-D MOJOLEARN_LU_FAST_STEP1_OFF`
turns it off too).

Main's blocked route (`launch_lu`, panel width LU_PANEL_NB = 32) updates the
WHOLE trailing square after every 32-column panel with `lu_trail_rb_kernel`
(scalar multiply-adds, 4 x 4 cells a thread, k = 32 per launch): at n = 8192
that is 256 passes that each read and write the trailing square once (about
46 GB of traffic for 3.7e11 flops, every flop a scalar FMA).

Here the columns are taken in outer blocks of LFM_NB (256) columns, LAPACK
getrf's form (right-looking, delayed update):

  for each outer block [K0, K1):
    for each 32-column inner panel [k0, k1) of it:
      - `lu_fast_panel` (LU_FAST_STEP1's one-launch-per-column panel; it also
        applies the panel's swaps to every column left of k0);
      - one thread per column j >= k1: the panel's swaps on column j (every
        trailing column, both inside the block and right of it: right of
        K1 neither row of a swap has received this block's updates yet, so
        swapping now or after the block is the same), then for j < K1 only
        the 32-row unit-lower solve (U rows k0 .. k1 - 1);
      - rows k1 .. n-1 x columns k1 .. K1-1 -= L(:, k0:k1) U(k0:k1, :), on
        the matrix unit (k = 32).
    then for the columns right of the block, j >= K1:
      - the band rows K0 .. K1-1 solved against the block's unit-lower L11,
        32 rows at a time (one thread per column for the 32-row solve, then
        the later band rows -= L U for that chunk on the matrix unit);
      - rows K1 .. n-1 x columns K1 .. n-1 -= L21 U12 (k = K1 - K0 = 256) on
        the matrix unit: one pass over the trailing square per 256 columns
        instead of per 32.

Every GEMM is `lfm_gemm_sub_kernel`: 64 x 64 output tiles, 2 x 2 simdgroups
of 4 x 4 8x8 fragments, k in AFN_GEMM_KB-step windows staged through
threadgroup memory (`gemm/afn_apple_fast.mojo`'s loaders and
`air.simdgroup_matrix_8x8_multiply_accumulate`, fp32 in, fp32 accumulate),
reading L and U in place from `a` (row stride n) and subtracting into `a`
(the C, A and B regions of every call are disjoint).

Zero pivots: main skips a step whose pivot is exactly 0 (act[k] = 0). The
pivot is the largest |a[i, k]| over rows i >= k, so a zero pivot means every
entry of L's column k is exactly 0 and its products add nothing: the matrix
unit's sum needs no mask. The row solves keep main's act test.

Bits: the trailing sums run in the matrix unit's order over k = 32 or 256
instead of main's ascending scalar chain per cell, so FAST words change (a
different f32 rounding of the same sums); pivots can differ only on a
near-tie. IDENTICAL and every other vendor compile main's route unchanged.

LU_FAST_TSLU (candidate, `-D MOJOLEARN_LU_FAST_TSLU`, x_decomp/lu_fast_tslu.mojo)
replaces each inner panel's `lu_fast_panel` + `_lfm_cols` with tournament
pivoting (~6 launches a panel instead of ~37); the GEMMs are unchanged.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.defines import get_defined_int
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import COLUMN_APPLE, lib_smem_page_fits_for
from checks.numerics import ftz, identical_mul_add
from gemm.afn_apple_fast import AFN_GEMM_APPLE, AFN_GEMM_KB, _afn_gload, _afn_load_t, _afn_mma, _afn_stage
from x_decomp.cells import F32Ptr, I32Ptr, lu_swap_elem
from x_decomp.lu_fast import LFS_TPB, LU_FAST_STEP1, lfs_blocks, lu_fast_panel
from x_decomp.lu_fast_tslu import (
    LTS_MAP_FLOATS, LTS_ORIG_FLOATS, LTS_U11_FLOATS, LU_FAST_TSLU, lts_cand_floats, lu_tslu_panel,
)

#: DEFAULT in FAST + Apple (lane/apple-fast-w2-linalg). M3, one run per arm:
#: lu-factor synthetic 971.7 -> 849.3 ms, lu-solve synthetic 971.9 -> 852.1 ms;
#: w2-lumma-quality PASS (factor/solve residual gates, info equal, finite).
#: Still needs LU_FAST_STEP1 and AFN_GEMM_APPLE. `-D MOJOLEARN_LU_FAST_MMA_OFF`
#: restores main's per-32-column scalar trailing updates.
comptime LU_FAST_MMA = LU_FAST_STEP1 and AFN_GEMM_APPLE and not is_defined["MOJOLEARN_LU_FAST_MMA_OFF"]()
#: The outer block (the big GEMM's k). A multiple of the 32-column panel.
comptime LFM_NB = get_defined_int["MOJOLEARN_LU_FAST_MMA_NB", 256]()
comptime LFM_PANEL = 32

comptime _M64 = SIMD[DType.float32, 64]
comptime LFM_SGM = 2
comptime LFM_SGN = 2
comptime LFM_FM = 4
comptime LFM_FN = 4
comptime LFM_NT = LFM_SGM * LFM_SGN * 32
comptime LFM_BM = 8 * LFM_FM * LFM_SGM
comptime LFM_BN = 8 * LFM_FN * LFM_SGN


def lfm_gemm_sub_kernel(
    a: F32Ptr, n_in: Int32, r0_in: Int32, c0_in: Int32, p0_in: Int32, m_in: Int32, nc_in: Int32, k_in: Int32
):
    """a[r0 + i, c0 + j] -= sum_p a[r0 + i, p0 + p] * a[p0 + p, c0 + j] for
    i < m, j < nc, p < k (row stride n), one block per 64 x 64 tile of
    (i, j). The operands are addressed absolutely through the loaders'
    outer base and step offset (no pointer arithmetic on the host)."""
    comptime KB = AFN_GEMM_KB
    comptime SGN = LFM_SGN
    comptime FM = LFM_FM
    comptime FN = LFM_FN
    comptime NT = LFM_NT
    comptime BM = LFM_BM
    comptime BN = LFM_BN
    comptime AST = BM + 4
    comptime BST = KB + 4
    comptime NF = FM * FN
    comptime ASZ = KB * AST
    comptime BSZ = BN * BST
    comptime PAGE_BYTES = (ASZ + BSZ) * 4
    comptime assert lib_smem_page_fits_for[COLUMN_APPLE, PAGE_BYTES](), (
        "lfm_gemm_sub_kernel: one staged page must fit Apple's threadgroup memory"
    )
    comptime assert KB % 8 == 0, "lfm_gemm_sub_kernel: whole 8-step fragments"
    var n = Int(n_in)
    var r0 = Int(r0_in)
    var c0 = Int(c0_in)
    var p0 = Int(p0_in)
    var m = Int(m_in)
    var nc = Int(nc_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var sgm = sg // SGN
    var sgn = sg % SGN
    var nbn = (nc + BN - 1) // BN
    var bid = Int(block_idx.x)
    var m0 = (bid // nbn) * BM
    var n0 = (bid % nbn) * BN
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var at = stack_allocation[ASZ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bt = stack_allocation[BSZ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = InlineArray[_M64, NF](fill=_M64(0))
    # A_eff[i, p] = a[(r0 + i) * n + p0 + p]: outer = row (stride n), step stride 1.
    # B_eff[p, j] = a[(p0 + p) * n + c0 + j]: outer = column (stride 1), step stride n.
    var windows = (k + KB - 1) // KB
    var ra = _afn_gload[BM, KB, NT, DType.float32](a, n, 1, r0 + m0, r0 + m, p0, min(KB, k), tid, False)
    var rb = _afn_gload[BN, KB, NT, DType.float32](a, 1, n, c0 + n0, c0 + nc, p0, min(KB, k), tid, True)
    for w in range(windows):
        _afn_stage[BM, KB, NT, True, AST](at, ra, tid, False)
        _afn_stage[BN, KB, NT, False, BST](bt, rb, tid, True)
        barrier()
        if w + 1 < windows:
            var k1 = (w + 1) * KB
            ra = _afn_gload[BM, KB, NT, DType.float32](a, n, 1, r0 + m0, r0 + m, p0 + k1, min(KB, k - k1), tid, False)
            rb = _afn_gload[BN, KB, NT, DType.float32](a, 1, n, c0 + n0, c0 + nc, p0 + k1, min(KB, k - k1), tid, True)
        comptime for p8 in range(KB // 8):
            var af = InlineArray[_M64, FM](fill=_M64(0))
            var bf = InlineArray[_M64, FN](fill=_M64(0))
            comptime for fm in range(FM):
                af[fm] = _afn_load_t(at + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
            comptime for fq in range(FN):
                bf[fq] = _afn_load_t(bt + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
            comptime for fm in range(FM):
                comptime for fq in range(FN):
                    acc[fm * FN + fq] = _afn_mma(af[fm], bf[fq], acc[fm * FN + fq])
        barrier()
    comptime for fm in range(FM):
        comptime for fq in range(FN):
            comptime for e in range(2):
                var gi = m0 + (sgm * FM + fm) * 8 + frow
                var gj = n0 + (sgn * FN + fq) * 8 + fcol + e
                if gi < m and gj < nc:
                    var cell = (r0 + gi) * n + c0 + gj
                    a.unsafe_store(cell, a.unsafe_load(cell) - acc[fm * FN + fq][e])


def lfm_swaps_trsm_kernel(
    a: F32Ptr, piv: I32Ptr, act: F32Ptr, k0: Int32, k1: Int32, jlo: Int32, jtri: Int32, jhi: Int32, n: Int32,
    swaps: Int32,
):
    """One thread per column j in [jlo, jhi): with `swaps`, the swaps k0 ..
    k1-1 on column j in order (`lu_apply_swaps_kernel`'s statements); then,
    for j < jtri, rows k0+1 .. k1-1 of column j solved against the unit-lower
    multipliers a[k, k0:k] (`lu_trsm_kernel`'s statements, act test kept)."""
    var nn = Int(n)
    var j = Int(jlo) + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(jhi):
        if Int(swaps) != 0:
            for k in range(Int(k0), Int(k1)):
                lu_swap_elem(a, piv, k, j, nn)
        if j < Int(jtri):
            for k in range(Int(k0) + 1, Int(k1)):
                var acc = ftz(a.unsafe_load(k * nn + j))
                for kp in range(Int(k0), k):
                    if act.unsafe_load(kp) != Float32(0):
                        var l = a.unsafe_load(k * nn + kp)
                        acc = ftz(identical_mul_add(-l, ftz(a.unsafe_load(kp * nn + j)), ftz(acc)))
                a.unsafe_store(k * nn + j, acc)


def _lfm_gemm_sub(ctx: DeviceContext, a: F32Ptr, n: Int, r0: Int, c0: Int, p0: Int, m: Int, nc: Int, k: Int) raises:
    if m <= 0 or nc <= 0 or k <= 0:
        return
    var tiles = ((m + LFM_BM - 1) // LFM_BM) * ((nc + LFM_BN - 1) // LFM_BN)
    ctx.enqueue_function[lfm_gemm_sub_kernel](
        a, Int32(n), Int32(r0), Int32(c0), Int32(p0), Int32(m), Int32(nc), Int32(k),
        grid_dim=(tiles, 1, 1), block_dim=(LFM_NT, 1, 1),
    )


def _lfm_cols(
    ctx: DeviceContext, a: F32Ptr, piv: I32Ptr, act: F32Ptr, k0: Int, k1: Int, jlo: Int, jtri: Int, jhi: Int, n: Int,
    swaps: Bool,
) raises:
    if jhi <= jlo:
        return
    ctx.enqueue_function[lfm_swaps_trsm_kernel](
        a, piv, act, Int32(k0), Int32(k1), Int32(jlo), Int32(jtri), Int32(jhi), Int32(n), Int32(1 if swaps else 0),
        grid_dim=lfs_blocks(jhi - jlo), block_dim=LFS_TPB,
    )


def lu_fast_mma_factor(
    ctx: DeviceContext, a: F32Ptr, piv: I32Ptr, info: F32Ptr, act: F32Ptr,
    p0: F32Ptr, p1: F32Ptr, pa: F32Ptr, pb: F32Ptr, n: Int, maxb: Int,
) raises:
    """The whole factorization (see the module docstring), enqueued, no sync.
    `info` must be initialized; p0/p1 hold n * LFM_PANEL floats each, pa/pb
    2 * maxb (`launch_lu`'s LU_FAST_STEP1 scratch)."""
    comptime if LU_FAST_TSLU:
        # LU_FAST_TSLU (candidate, -D MOJOLEARN_LU_FAST_TSLU, x_decomp/lu_fast_tslu.mojo):
        # tournament-pivoting panels; their scratch lives here, so this arm
        # drains before returning (launch_lu drains right after anyway).
        var tca = ctx.enqueue_create_buffer[DType.float32](lts_cand_floats(n))
        var tcb = ctx.enqueue_create_buffer[DType.float32](lts_cand_floats(n))
        var tu = ctx.enqueue_create_buffer[DType.float32](LTS_U11_FLOATS)
        var tor = ctx.enqueue_create_buffer[DType.float32](LTS_ORIG_FLOATS)
        var tm = ctx.enqueue_create_buffer[DType.float32](LTS_MAP_FLOATS)
        _lfm_run[True](
            ctx, a, piv, info, act, p0, p1, pa, pb, n, maxb,
            _lfm_p(tca), _lfm_p(tcb), _lfm_p(tu), _lfm_p(tor), _lfm_p(tm),
        )
        ctx.synchronize()
        _ = tca^
        _ = tcb^
        _ = tu^
        _ = tor^
        _ = tm^
    else:
        _lfm_run[False](ctx, a, piv, info, act, p0, p1, pa, pb, n, maxb, p0, p0, p0, p0, p0)


def _lfm_p(buf: DeviceBuffer[DType.float32]) -> F32Ptr:
    """A device buffer's address as the cells' pointer type (device.mojo's `_p`)."""
    return F32Ptr(unsafe_from_address=Int(buf.unsafe_ptr()))


def _lfm_run[TSLU: Bool](
    ctx: DeviceContext, a: F32Ptr, piv: I32Ptr, info: F32Ptr, act: F32Ptr,
    p0: F32Ptr, p1: F32Ptr, pa: F32Ptr, pb: F32Ptr, n: Int, maxb: Int,
    tca: F32Ptr, tcb: F32Ptr, tu11: F32Ptr, torig: F32Ptr, tmap: F32Ptr,
) raises:
    """`lu_fast_mma_factor`'s launches; with TSLU the inner panels by
    `lu_tslu_panel` (tca .. tmap its scratch), else main's two calls."""
    comptime assert LFM_NB % LFM_PANEL == 0 and LFM_NB > 0, "LU_FAST_MMA: the outer block is whole 32-column panels"
    var K0 = 0
    while K0 < n:
        var K1 = min(K0 + LFM_NB, n)
        var k0 = K0
        while k0 < K1:
            var k1 = min(k0 + LFM_PANEL, K1)
            comptime if TSLU:
                # the panel, its swaps on every column and the U rows inside
                # the block: ~levels + 1 launches (tournament pivots)
                lu_tslu_panel(ctx, a, piv, info, act, tca, tcb, tu11, torig, tmap, k0, k1, K1, n)
            else:
                lu_fast_panel(ctx, a, piv, info, act, p0, p1, pa, pb, k0, k1, n, maxb)
                # every trailing column's swaps; the U rows inside the block
                _lfm_cols(ctx, a, piv, act, k0, k1, k1, K1, n, n, True)
            # rows k1.. x the block's columns right of the panel
            _lfm_gemm_sub(ctx, a, n, k1, k1, k0, n - k1, K1 - k1, k1 - k0)
            k0 = k1
        if K1 < n:
            # the band rows K0..K1-1 of the columns right of the block: solved
            # 32 rows at a time, each chunk feeding the band rows below it
            var c = K0
            while c < K1:
                var ce = min(c + LFM_PANEL, K1)
                _lfm_cols(ctx, a, piv, act, c, ce, K1, n, n, n, False)
                _lfm_gemm_sub(ctx, a, n, ce, K1, c, K1 - ce, n - K1, ce - c)
                c = ce
            # the trailing square, k = K1 - K0
            _lfm_gemm_sub(ctx, a, n, K1, K1, K0, n - K1, n - K1, K1 - K0)
        K0 = K1
