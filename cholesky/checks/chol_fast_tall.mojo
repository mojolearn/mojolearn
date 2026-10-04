# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-w3-linalg (2026-10-04): CHOL_FAST_TALL (opt-in,
`-D MOJOLEARN_CHOL_FAST_TALL`), the Apple FAST Cholesky column panel as one
tall blocked factor with matrix-unit solves (FAST on Apple only; only the
public `Cholesky.fit` route reaches it, `cholesky_factor_devio`'s
`defer_ok` sweep, which redoes a failed factor on main's route).

Main's FAST panel (`potrf_lower` at CHOL_FAST_NB = 256) per outer panel:
`fast_diag_factor` (four 64-column inner steps, each the one-block
left-looking `panel_factor_kernel` with global-memory dot chains, an
identity launch, `fts_lower_diag_kernel` for the 64 x 64 inverse at two
block reductions per row, a vendor GEMM, an unpack and a vendor GEMM
epilogue on the 256 x 256 block only), then for the rows below a second
inverse of the WHOLE 256 x 256 L11 (`fts_lower_diag_kernel`: 256 serial
rows of two block reductions each), a pack, a vendor GEMM and an unpack:
about 30 dependent launches and ~1,000 serial reduction rounds per panel,
32 panels at n = 8192.

Here each 64-column inner step works on the whole column panel (rows c ..
n - 1), so the 256 x 256 inverse, its pack and unpack are gone:

  1. `ctl_diag_inv_kernel` (one threadgroup, the 64 x 64 diagonal block in
     threadgroup memory): right-looking factor, one barrier per column;
     then each thread forms one column of L11^{-1} from threadgroup memory
     (stored in the free upper triangle), no barriers; writes L11 into `a`
     and L11^{-1} into `linv`.
  2. rows below: L21 = A21 L11^{-T} on the matrix unit (`ctl_mma_kernel`,
     in place: one 64-column output tile per row band, so a block reads
     only its own rows of A21 and writes them after its last window).
  3. the panel's later columns, rows c + 64 .. n - 1:
     A[r, q] -= sum_p L[r, p] L[q, p] (lower cells only) on the matrix
     unit, reading both operands in place from `a`.

Three launches per inner step, 11 per outer panel (plus main's pack and
trailing update, unchanged). The outer trailing update is main's.

Bits: FAST words change (the 64-column inverse solves replace the 256
inverse, the matrix unit's sum order replaces the vendor GEMM's); the
factor is the same blocked Cholesky. IDENTICAL and every other vendor
compile main's route unchanged (CHOL_FAST_TALL is False there).
"""

from std.gpu import block_idx, thread_idx
from std.math import sqrt
from std.memory import stack_allocation
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import COLUMN_APPLE, lib_smem_page_fits_for
from gemm.afn_apple_fast import AFN_GEMM_APPLE, AFN_GEMM_KB, _afn_gload, _afn_load_t, _afn_mma, _afn_stage

#: Opt-in candidate (M3 A/B owed: tags w2-cholt-*). `-D MOJOLEARN_CHOL_FAST_TALL`.
comptime CHOL_FAST_TALL = AFN_GEMM_APPLE and is_defined["MOJOLEARN_CHOL_FAST_TALL"]()
#: The inner step (columns per diagonal block). At most the tile width, so
#: the in-place solve has one output tile per row band.
comptime CTL_NB = 64
comptime CTL_SST = CTL_NB + 1
comptime CTL_DIAG_TPB = 256

comptime _M64 = SIMD[DType.float32, 64]
comptime CTL_SGM = 2
comptime CTL_SGN = 2
comptime CTL_FM = 4
comptime CTL_FN = 4
comptime CTL_NT = CTL_SGM * CTL_SGN * 32
comptime CTL_BM = 8 * CTL_FM * CTL_SGM
comptime CTL_BN = 8 * CTL_FN * CTL_SGN


def ctl_diag_inv_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    linv: MutPointer[Float32, MutAnyOrigin],
    info: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    c_in: Int32,
    sw_in: Int32,
):
    """The `sw x sw` diagonal block at `[c, c + sw)^2` (sw <= CTL_NB),
    ONE threadgroup (small-launch: a 64 x 64 block): `a` gets L11 (lower
    cells only), `linv` (sw x sw row-major) gets L11^{-1}. A non-positive
    (or NaN) pivot at column k writes `info = c + k + 1` and every thread
    returns at the same point (each reads the same threadgroup word)."""
    var n = Int(n_in)
    var c = Int(c_in)
    var sw = Int(sw_in)
    var tid = Int(thread_idx.x)
    comptime SST = CTL_SST
    comptime T = CTL_DIAG_TPB
    var s = stack_allocation[CTL_NB * SST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var e = tid
    while e < sw * sw:
        var i = e // sw
        var j = e - i * sw
        if j <= i:
            s[i * SST + j] = a[(c + i) * n + c + j]
        e += T
    barrier()
    for k in range(sw):
        var d2 = s[k * SST + k]
        if not (d2 > Float32(0.0)):
            if tid == 0:
                info[0] = Int32(c + k + 1)
            return
        var d = sqrt(d2)
        # the trailing lower cells (i, j), k < j <= i < sw, from the
        # UNSCALED column k (not written in this phase)
        var m = sw - k - 1
        var q = tid
        while q < m * m:
            var ii = k + 1 + q // m
            var jj = k + 1 + q % m
            if jj <= ii:
                var lik = s[ii * SST + k] / d
                var ljk = s[jj * SST + k] / d
                s[ii * SST + jj] = s[ii * SST + jj] - lik * ljk
            q += T
        barrier()
        # column k scaled, in threadgroup memory and in `a`; no later step
        # reads or writes column k, so no barrier before step k + 1
        if tid == 0:
            s[k * SST + k] = d
            a[(c + k) * n + c + k] = d
        var r = tid
        while r < m:
            var i = k + 1 + r
            var l = s[i * SST + k] / d
            s[i * SST + k] = l
            a[(c + i) * n + c + k] = l
            r += T
    barrier()
    # L11^{-1}: thread j forms column j; X[i, j] (i > j) lives in the free
    # upper cell s[j, i], read and written only by thread j.
    if tid < sw:
        var j = tid
        var xjj = Float32(1.0) / s[j * SST + j]
        for i in range(j + 1, sw):
            var acc = s[i * SST + j] * xjj
            for kk in range(j + 1, i):
                acc += s[i * SST + kk] * s[j * SST + kk]
            s[j * SST + i] = -acc / s[i * SST + i]
    barrier()
    e = tid
    while e < sw * sw:
        var i = e // sw
        var j = e - i * sw
        var v = Float32(0.0)
        if i == j:
            v = Float32(1.0) / s[i * SST + i]
        elif i > j:
            v = s[j * SST + i]
        linv[e] = v
        e += T


def ctl_mma_kernel[
    SUB: Bool, LOWER: Bool, B_IS_A: Bool
](
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    ar0_in: Int32,
    ac0_in: Int32,
    br0_in: Int32,
    bc0_in: Int32,
    ldb_in: Int32,
    cr0_in: Int32,
    cc0_in: Int32,
    m_in: Int32,
    nc_in: Int32,
    k_in: Int32,
):
    """C[i, j] (-)= sum_p A[i, p] B[j, p] (the NT form) on the matrix unit,
    i < m, j < nc, p < k, one block per 64 x 64 tile:
      A[i, p] = a[(ar0 + i) * n + ac0 + p]
      B[j, p] = src[(br0 + j) * ldb + bc0 + p], src = `a` if B_IS_A else `b`
      C[i, j] = a[(cr0 + i) * n + cc0 + j]   (SUB: subtract, else store)
    LOWER: only cells with cc0 + j <= cr0 + i (tiles wholly above that
    diagonal return at once). The C region must not overlap B, and may
    overlap A only within the block's own rows (one column tile)."""
    comptime KB = AFN_GEMM_KB
    comptime SGN = CTL_SGN
    comptime FM = CTL_FM
    comptime FN = CTL_FN
    comptime NT = CTL_NT
    comptime BM = CTL_BM
    comptime BN = CTL_BN
    comptime AST = BM + 4
    comptime BST = KB + 4
    comptime NF = FM * FN
    comptime ASZ = KB * AST
    comptime BSZ = BN * BST
    comptime PAGE_BYTES = (ASZ + BSZ) * 4
    comptime assert lib_smem_page_fits_for[COLUMN_APPLE, PAGE_BYTES](), (
        "ctl_mma_kernel: one staged page must fit Apple's threadgroup memory"
    )
    comptime assert KB % 8 == 0, "ctl_mma_kernel: whole 8-step fragments"
    var n = Int(n_in)
    var ar0 = Int(ar0_in)
    var ac0 = Int(ac0_in)
    var br0 = Int(br0_in)
    var bc0 = Int(bc0_in)
    var ldb = Int(ldb_in)
    var cr0 = Int(cr0_in)
    var cc0 = Int(cc0_in)
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
    comptime if LOWER:
        if cc0 + n0 > cr0 + m0 + BM - 1:
            return
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var at = stack_allocation[ASZ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bt = stack_allocation[BSZ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = InlineArray[_M64, NF](fill=_M64(0))
    var bsrc = b
    comptime if B_IS_A:
        bsrc = a
    var windows = (k + KB - 1) // KB
    var ra = _afn_gload[BM, KB, NT, DType.float32](a, n, 1, ar0 + m0, ar0 + m, ac0, min(KB, k), tid, False)
    var rb = _afn_gload[BN, KB, NT, DType.float32](bsrc, ldb, 1, br0 + n0, br0 + nc, bc0, min(KB, k), tid, False)
    for w in range(windows):
        _afn_stage[BM, KB, NT, True, AST](at, ra, tid, False)
        _afn_stage[BN, KB, NT, False, BST](bt, rb, tid, False)
        barrier()
        if w + 1 < windows:
            var k1 = (w + 1) * KB
            ra = _afn_gload[BM, KB, NT, DType.float32](a, n, 1, ar0 + m0, ar0 + m, ac0 + k1, min(KB, k - k1), tid, False)
            rb = _afn_gload[BN, KB, NT, DType.float32](bsrc, ldb, 1, br0 + n0, br0 + nc, bc0 + k1, min(KB, k - k1), tid, False)
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
                var ok = gi < m and gj < nc
                comptime if LOWER:
                    ok = ok and cc0 + gj <= cr0 + gi
                if ok:
                    var cell = (cr0 + gi) * n + cc0 + gj
                    comptime if SUB:
                        a.unsafe_store(cell, a.unsafe_load(cell) - acc[fm * FN + fq][e])
                    else:
                        a.unsafe_store(cell, acc[fm * FN + fq][e])


def chol_fast_tall_panel(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut dinfo: DeviceBuffer[DType.int32],
    mut linv: DeviceBuffer[DType.float32],
    n: Int,
    j0: Int,
    w: Int,
) raises:
    """CHOL_FAST_TALL: the column panel `[j0, j0 + w)`, rows j0 .. n - 1,
    factored in CTL_NB-column steps (module docstring): on return the
    panel's L11 and L21 are in `a` (lower cells), exactly what main's
    diagonal factor + panel solve leave there. `linv` holds at least
    CTL_NB * CTL_NB floats. A failed pivot sets `dinfo` and the later
    launches compute garbage that the caller discards (it redoes the
    factor on main's route)."""
    comptime assert CTL_NB <= CTL_BN, "CHOL_FAST_TALL: the in-place solve needs one output tile per row band"
    var s0 = 0
    while s0 < w:
        var sw = min(CTL_NB, w - s0)
        var c = j0 + s0
        ctx.enqueue_function[ctl_diag_inv_kernel](  # small-launch(n: leading dimension only): factors and inverts the sw x sw (<= 64) diagonal block in threadgroup memory
            a.unsafe_ptr(), linv.unsafe_ptr(), dinfo.unsafe_ptr(),
            Int32(n), Int32(c), Int32(sw),
            grid_dim=(1, 1, 1), block_dim=(CTL_DIAG_TPB, 1, 1),
        )
        var rows = n - c - sw
        if rows > 0:
            var mt = (rows + CTL_BM - 1) // CTL_BM
            # L21 = A21 L11^{-T}: B[j, p] = linv[j * sw + p] = L11^{-1}[j, p]
            ctx.enqueue_function[ctl_mma_kernel[False, False, False]](
                a.unsafe_ptr(), linv.unsafe_ptr(), Int32(n),
                Int32(c + sw), Int32(c), Int32(0), Int32(0), Int32(sw),
                Int32(c + sw), Int32(c), Int32(rows), Int32(sw), Int32(sw),
                grid_dim=(mt, 1, 1), block_dim=(CTL_NT, 1, 1),
            )
            var cols = j0 + w - c - sw
            if cols > 0:
                # the panel's later columns: A[r, q] -= L[r, c:c+sw] . L[q, c:c+sw]
                var ntl = (cols + CTL_BN - 1) // CTL_BN
                ctx.enqueue_function[ctl_mma_kernel[True, True, True]](
                    a.unsafe_ptr(), linv.unsafe_ptr(), Int32(n),
                    Int32(c + sw), Int32(c), Int32(c + sw), Int32(c), Int32(n),
                    Int32(c + sw), Int32(c + sw), Int32(rows), Int32(cols), Int32(sw),
                    grid_dim=(mt * ntl, 1, 1), block_dim=(CTL_NT, 1, 1),
                )
        s0 += sw
