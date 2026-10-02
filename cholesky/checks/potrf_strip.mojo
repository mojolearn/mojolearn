# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The strip schedule of profile `mojolearn.identical.cholesky.fp32.v1` on
NVIDIA and AMD (lane neural-pass136, 2026-10-02): the SAME per-cell
arithmetic as `potrf.mojo::potrf_lower`'s right-looking loop, in a schedule
that does not rewrite the whole trailing square once per 32-wide panel.

THE CONTRACT EACH CELL KEEPS (what `potrf_lower` and
`cholesky/host/chol_oracle.mojo` compute, unchanged):

  * trailing cell (i, j), j <= i, takes, for every panel p with
    32 (p + 1) <= j in ASCENDING p, the one step
        G = +0.0; G = ftz(fma(ftz(L[i][k]), ftz(L[j][k]), G)) for k in
        panel p ascending (the gemm profile's one leaf at k = 32);
        A = ftz(ftz(A) - ftz(G))
  * the diagonal block: `panel_factor_kernel`'s columns ascending, the
    pivot on its ftz chain, `not (s > 0)`, `identical_sqrt`, the column
    chains, `identical_div`;
  * the panel solve: `trsm_panel_kernel`'s per-row chains, k ascending.

THE SCHEDULE. Column strips of CHOL_STRIP_W (256) columns. For a strip
[J, J + sw): one launch applies panels 0 .. J/32 - 1 to every lower cell of
the strip, LEFT-LOOKING (each cell held in a register, the panels walked in
order); then the strip's own panels are factored right-looking inside it
(diagonal block, panel solve, the panel's step on the strip's later
columns). A cell therefore takes panels 0, 1, 2, ... in the same order with
the same step: the same bits as the right-looking loop. Only scheduling
moved (which cells a launch touches, and when).

`info` is not read back per panel: every kernel returns at once when it is
already set, so after a failing panel nothing more is written, and the
driver completes the right-looking partial factor (later strips take the
panels before the failing one) after one read at the end.

THE TRAILING STEP on an ADMITTED panel window runs the bare FMA: the gemm
profile's window admission (`gemm_identical.GEMM_ADMIT_EXP_SUM`): when the
block's staged, flushed operand words prove Ea + Eb >= 174 and the
accumulator starts at +0.0 (every panel does), no exact step result is
subnormal, so the flush is the identity. Every other window runs the exact
step `_tuned_step`.
"""

from std.gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from std.sys.defines import get_defined_int
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_NVIDIA,
    TARGET_COLUMN,
    lib_smem_page_fits_for,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_mul_add,
    identical_sqrt,
)
from gemm.checks.gemm_identical import (
    GEMM_ADMIT_EXP_SUM,
    _admit_step,
    _admit_warp_min,
    _tuned_step,
)

#: The pinned panel width (`potrf.CHOL_NB_PINNED`). The strip route runs only
#: at this width.
comptime CS_NB = 32
#: SCHEDULING: columns per strip, a multiple of CS_NB. Bits do not depend on
#: it. `-D MOJOLEARN_CHOL_STRIP_W=<n>` is a measurement arm.
comptime CHOL_STRIP_W = get_defined_int["MOJOLEARN_CHOL_STRIP_W", 256]()
#: SCHEDULING: the update tile (64 x 64 cells, 256 threads, 4 x 4 cells each).
comptime CS_BM = 64
comptime CS_TPB = 256
comptime CS_ST = CS_BM + 1
#: Threads of the diagonal-block and panel-solve kernels.
comptime CS_DIAG_TPB = 64
comptime CS_TRSM_TPB = 128
comptime CS_DST = CS_NB + 1

#: Shared pages, gated per column (`lib_smem_page_fits_for`).
comptime CS_UPDATE_PAGE_BYTES = 2 * CS_NB * CS_ST * 4 + 16 * 4
comptime CS_DIAG_PAGE_BYTES = CS_NB * CS_DST * 4 + 4
comptime CHOL_STRIP_FITS = (
    lib_smem_page_fits_for[TARGET_COLUMN, CS_UPDATE_PAGE_BYTES]()
    and lib_smem_page_fits_for[TARGET_COLUMN, CS_DIAG_PAGE_BYTES]()
)

#: The route: NVIDIA and AMD (CDNA) columns, pages that fit. Apple keeps its
#: left-looking matrix-unit route (`potrf.CHOL_APPLE_LEFT`). Off: `-D
#: MOJOLEARN_CHOL_STRIP_OFF` here, or `MOJOLEARN_CHOL_STRIP_OFF=1` in the
#: environment at run time (`potrf_lower`), the A/B arm.
comptime CHOL_STRIP_ROUTE = (
    (TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD)
    and CHOL_STRIP_FITS
    and not is_defined["MOJOLEARN_CHOL_STRIP_OFF"]()
)

#: The window admission is compiled in under IDENTICAL on the two columns.
#: `-D MOJOLEARN_CHOL_STRIP_NO_ADMIT` runs the exact step on every window.
comptime CS_ADMIT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and (TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD)
    and not is_defined["MOJOLEARN_CHOL_STRIP_NO_ADMIT"]()
)
comptime CS_NW = CS_TPB // WARP_SIZE


@always_inline
def _cs_exp_nz(v: Float32) -> UInt32:
    """Biased exponent field of a flushed word; a zero word constrains
    nothing and reads 255 (`gemm_identical._admit_exp_min`'s rule)."""
    var e = (bitcast[DType.uint32](v) >> UInt32(23)) & UInt32(0xFF)
    return UInt32(0xFF) if e == UInt32(0) else e


@always_inline
def _cs_bare_step(a: Float32, b: Float32, acc: Float32) -> Float32:
    """The admitted step: the round-to-nearest FMA whose flush the window
    proved is the identity."""
    comptime if TARGET_COLUMN == COLUMN_NVIDIA and GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return _admit_step(a, b, acc)
    return identical_mul_add(a, b, acc)


def chol_strip_diag_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    info: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    j0_in: Int32,
    w_in: Int32,
):
    """`panel_factor_kernel` on a shared copy of the `w x w` diagonal block
    (w <= 32), returning at once when `info` is already set. The same
    statements on the same values: every read is `ftz` of the cell's current
    word, every write is the same `ftz(...)`. On a failing pivot the column
    is left unwritten, as there; the lower cells are copied back either way
    (an unwritten cell goes back as the word it was loaded as)."""
    if info[0] != Int32(0):
        return
    var n = Int(n_in)
    var j0 = Int(j0_in)
    var w = Int(w_in)
    var tid = Int(thread_idx.x)
    var width = Int(block_dim.x)
    var s = stack_allocation[
        CS_NB * CS_DST, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var flag = stack_allocation[
        1, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    for idx in range(tid, w * w, width):
        var r = idx // w
        var c = idx % w
        if c <= r:
            s[r * CS_DST + c] = a[(j0 + r) * n + j0 + c]
    if tid == 0:
        flag[0] = Int32(0)
    barrier()
    for c in range(w):
        if tid == 0:
            var d = ftz(s[c * CS_DST + c])
            for k in range(c):
                var v = ftz(s[c * CS_DST + k])
                d = ftz(identical_mul_add(-v, v, d))
            if not (d > Float32(0.0)):
                info[0] = Int32(j0 + c + 1)
                flag[0] = Int32(1)
            else:
                s[c * CS_DST + c] = ftz(identical_sqrt(d))
        barrier()
        if flag[0] != Int32(0):
            break
        var ljj = ftz(s[c * CS_DST + c])
        var i = c + 1 + tid
        while i < w:
            var t = ftz(s[i * CS_DST + c])
            for k in range(c):
                var lrk = ftz(s[i * CS_DST + k])
                var lck = ftz(s[c * CS_DST + k])
                t = ftz(identical_mul_add(-lrk, lck, t))
            s[i * CS_DST + c] = ftz(identical_div(t, ljj))
            i += width
        barrier()
    for idx in range(tid, w * w, width):
        var r = idx // w
        var c = idx % w
        if c <= r:
            a[(j0 + r) * n + j0 + c] = s[r * CS_DST + c]


def chol_strip_trsm_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    info: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    j0_in: Int32,
    n_trail_in: Int32,
):
    """`trsm_panel_kernel` at the pinned width 32, one thread per trailing
    row, the factored diagonal block in shared memory and the row's solved
    values in registers (each is the word the reference stores and then
    reads back). Returns at once when `info` is already set."""
    if info[0] != Int32(0):
        return
    var n = Int(n_in)
    var j0 = Int(j0_in)
    var n_trail = Int(n_trail_in)
    var tid = Int(thread_idx.x)
    var s = stack_allocation[
        CS_NB * CS_DST, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    for idx in range(tid, CS_NB * CS_NB, Int(block_dim.x)):
        var r = idx // CS_NB
        var c = idx % CS_NB
        if c <= r:
            s[r * CS_DST + c] = a[(j0 + r) * n + j0 + c]
    barrier()
    var row = Int(block_idx.x) * Int(block_dim.x) + tid
    if row >= n_trail:
        return
    var base = (j0 + CS_NB + row) * n + j0
    var y = InlineArray[Float32, CS_NB](fill=Float32(0.0))
    comptime for c in range(CS_NB):
        var t = ftz(a[base + c])
        comptime for k in range(c):
            t = ftz(identical_mul_add(-y[k], ftz(s[c * CS_DST + k]), t))
        y[c] = ftz(identical_div(t, ftz(s[c * CS_DST + c])))
        a[base + c] = y[c]


def chol_strip_update_kernel[GUARD: Bool](
    a: MutPointer[Float32, MutAnyOrigin],
    info: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    col0_in: Int32,
    col_end_in: Int32,
    p_lo_in: Int32,
    p_hi_in: Int32,
):
    """Every lower cell (i, j), j <= i, i >= col0, col0 <= j < col_end,
    takes panels p_lo .. p_hi - 1 in order, each by the contract's step
    `A = ftz(ftz(A) - ftz(G_p))`, with A held in a register across panels.
    One 64 x 64 tile per block (block x = column tile, block y = row tile
    from col0); a tile wholly above the diagonal returns at once. GUARD:
    return at once when `info` is already set."""
    comptime if GUARD:
        if info[0] != Int32(0):
            return
    var n = Int(n_in)
    var col0 = Int(col0_in)
    var col_end = Int(col_end_in)
    var p_lo = Int(p_lo_in)
    var p_hi = Int(p_hi_in)
    var c0 = col0 + Int(block_idx.x) * CS_BM
    var r0 = col0 + Int(block_idx.y) * CS_BM
    if c0 > r0 + CS_BM - 1:
        return
    var tid = Int(thread_idx.x)
    var tx = tid % 16
    var ty = tid // 16
    var lk = tid % CS_NB
    var lm = tid // CS_NB
    var As = stack_allocation[
        CS_NB * CS_ST, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        CS_NB * CS_ST, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var wmin = stack_allocation[
        16, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    var c = InlineArray[Float32, 16](fill=Float32(0.0))
    comptime for ii in range(4):
        comptime for jj in range(4):
            var i = r0 + ty + 16 * ii
            var j = c0 + tx + 16 * jj
            if i < n and j < col_end and j <= i:
                c[ii * 4 + jj] = a[i * n + j]
    var ra = InlineArray[Float32, 8](fill=Float32(0.0))
    var rb = InlineArray[Float32, 8](fill=Float32(0.0))
    comptime for q in range(8):
        var m = lm + 8 * q
        ra[q] = a[(r0 + m) * n + p_lo * CS_NB + lk] if r0 + m < n else Float32(0.0)
        rb[q] = a[(c0 + m) * n + p_lo * CS_NB + lk] if c0 + m < n else Float32(0.0)
    for p in range(p_lo, p_hi):
        var ea = UInt32(0xFF)
        var eb = UInt32(0xFF)
        comptime for q in range(8):
            var m = lm + 8 * q
            var va = ftz(ra[q])
            var vb = ftz(rb[q])
            As[lk * CS_ST + m] = va
            Bs[lk * CS_ST + m] = vb
            comptime if CS_ADMIT:
                var xa = _cs_exp_nz(va)
                var xb = _cs_exp_nz(vb)
                ea = xa if xa < ea else ea
                eb = xb if xb < eb else eb
        comptime if CS_ADMIT:
            ea = _admit_warp_min(ea)
            eb = _admit_warp_min(eb)
            if tid % WARP_SIZE == 0:
                wmin[tid // WARP_SIZE] = ea
                wmin[8 + tid // WARP_SIZE] = eb
        barrier()
        if p + 1 < p_hi:
            comptime for q in range(8):
                var m = lm + 8 * q
                ra[q] = a[(r0 + m) * n + (p + 1) * CS_NB + lk] if r0 + m < n else Float32(0.0)
                rb[q] = a[(c0 + m) * n + (p + 1) * CS_NB + lk] if c0 + m < n else Float32(0.0)
        var admitted = False
        comptime if CS_ADMIT:
            var bea = UInt32(0xFF)
            var beb = UInt32(0xFF)
            comptime for s in range(CS_NW):
                bea = wmin[s] if wmin[s] < bea else bea
                beb = wmin[8 + s] if wmin[8 + s] < beb else beb
            admitted = (bea + beb) >= UInt32(GEMM_ADMIT_EXP_SUM)
        var acc = InlineArray[Float32, 16](fill=Float32(0.0))
        if admitted:
            comptime for k in range(CS_NB):
                var av = InlineArray[Float32, 4](fill=Float32(0.0))
                var bv = InlineArray[Float32, 4](fill=Float32(0.0))
                comptime for ii in range(4):
                    av[ii] = As[k * CS_ST + ty + 16 * ii]
                    bv[ii] = Bs[k * CS_ST + tx + 16 * ii]
                comptime for ii in range(4):
                    comptime for jj in range(4):
                        acc[ii * 4 + jj] = _cs_bare_step(av[ii], bv[jj], acc[ii * 4 + jj])
        else:
            comptime for k in range(CS_NB):
                var av = InlineArray[Float32, 4](fill=Float32(0.0))
                var bv = InlineArray[Float32, 4](fill=Float32(0.0))
                comptime for ii in range(4):
                    av[ii] = As[k * CS_ST + ty + 16 * ii]
                    bv[ii] = Bs[k * CS_ST + tx + 16 * ii]
                comptime for ii in range(4):
                    comptime for jj in range(4):
                        acc[ii * 4 + jj] = _tuned_step(av[ii], bv[jj], acc[ii * 4 + jj])
        comptime for q in range(16):
            var g = ftz(ftz(acc[q]))
            comptime if is_defined["MOJOLEARN_CHOL_STRIP_SABOTAGE"]():
                g = g * Float32(1.0000001)
            c[q] = ftz(ftz(c[q]) - g)
        barrier()
    comptime for ii in range(4):
        comptime for jj in range(4):
            var i = r0 + ty + 16 * ii
            var j = c0 + tx + 16 * jj
            if i < n and j < col_end and j <= i:
                a[i * n + j] = c[ii * 4 + jj]
