# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`core/householder_qr.mojo::qr_factor`, BOUNDED IN WORK PER LAUNCH
(lane/lle-timeout, 2026-09-29).

`qr_panel_kernel` factors a whole slice in ONE launch: a square n x n
matrix is one threadgroup doing O(n^3) work behind n^2 / 2 barriers (0.75 s
at 500 columns on the M2 Pro, tens of seconds at 2,000). macOS aborts a long
Metal launch SILENTLY (a ~4 s launch on the M2 Pro was cut, its output partly
stale, synchronize silent), and a stale R became a stale embedding with no
error. `qr_panel_cols_kernel` is the SAME column step (the same reflector
cells, `qr_reflector_r/u1/tau` imported from core, the same folds, the same
order) over columns [j0, j1) only; every value a column step reads is in `a`
and `r_out`, so a run of launches over consecutive column ranges stores the
bits one launch stores. The last range also writes R's strict triangles.
`qr_factor_bounded` cuts the passes `qr_factor` runs (the same slice count,
the same TSQR stack) into ranges of about QRB_CELLS cell updates, waits for
the device after every launch, and POISONS r_out (and the TSQR tiles) with
NaN first: every cell of them is written by a finished pass, so a cut launch
leaves a NaN the caller's read-back refuses (`DevExec.svd`).
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx

from checks.numerics import ftz, identical_div, identical_mul_add, identical_sqrt
from core.device_zero import enqueue_fill
from core.householder_qr import (
    QR_TPB,
    fold_and_broadcast,
    qr_reflector_r,
    qr_reflector_tau,
    qr_reflector_u1,
    qr_slice_count,
)

#: Cell updates per launch (a column step at j costs (rows - j) (n - j) cells
#: plus QRB_BARRIER_CELLS per barrier): about 0.3 s on the M2 Pro.
comptime QRB_CELLS = 1 << 24
comptime QRB_BARRIER_CELLS = 256


def qr_panel_cols_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    r_out: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    lda_in: Int32,
    n_slices_in: Int32,
    j0_in: Int32,
    j1_in: Int32,
    finish_in: Int32,
):
    """`qr_panel_kernel` over columns [j0, j1); `finish` != 0 also writes the
    strict triangles of every slice's R (the launch holding the last column).
    LAUNCH WITH EXACTLY `QR_TPB` THREADS, one block per slice."""
    var m = Int(m_in)
    var n = Int(n_in)
    var lda = Int(lda_in)
    var n_slices = Int(n_slices_in)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var rb = (b * m) // n_slices
    var re = ((b + 1) * m) // n_slices
    var ms = re - rb
    var rbase = b * n * n

    for j in range(Int(j0_in), Int(j1_in)):
        # --- the column norm of A[j:ms, j], strided partials then a fold ---
        var acc = Float32(0.0)
        var i = j + tid
        while i < ms:
            var v = ftz(a.unsafe_load((rb + i) * lda + j))
            acc = ftz(identical_mul_add(v, v, acc))
            i += QR_TPB
        var sigma = fold_and_broadcast[QR_TPB](acc)
        # THE TEST IS ON `normx`, NOT ON `sigma`. `identical_sqrt` of a
        # subnormal `sigma` can flush to zero through `ftz`, and then a
        # `sigma != 0` guard would let a zero `normx` reach the division in
        # `qr_reflector_tau`. Testing the value that is actually divided by
        # is the guard that cannot be skipped past.
        var normx = ftz(identical_sqrt(sigma))
        var ajj = ftz(a.unsafe_load((rb + j) * lda + j))

        # `normx` came out of a BROADCAST fold, so every thread of the block
        # holds the same bits and this branch is uniform. That is what makes
        # the barriers inside both arms legal.
        if normx == Float32(0.0):
            # DEVIATION 588: rank deficiency is a ZERO SINGULAR VALUE here,
            # not a refusal. `H_j = I`, `R_jj = 0`, carry on.
            if tid == 0:
                r_out.unsafe_store(rbase + j * n + j, Float32(0.0))
            barrier()
        else:
            var r_jj = qr_reflector_r(ajj, normx)
            var u1 = qr_reflector_u1(ajj, r_jj)
            var tau = qr_reflector_tau(ajj, normx, u1)
            if tid == 0:
                r_out.unsafe_store(rbase + j * n + j, r_jj)

            # Pack `w` into the subdiagonal with `w_j = 1` implicit, which
            # is LAPACK's layout and arima's.
            #
            # `u1` cannot be zero when `normx` is not: with `s = -sign(ajj)`
            # the two terms of `ajj - s*normx` have the same sign and
            # `|u1| >= normx > 0`. That is the sign's whole job, so the
            # division needs no second guard -- and if the sign is ever
            # sabotaged, this is one of the places it shows.
            var i2 = j + 1 + tid
            while i2 < ms:
                var cur = ftz(a.unsafe_load((rb + i2) * lda + j))
                a.unsafe_store(
                    (rb + i2) * lda + j, ftz(identical_div(cur, u1))
                )
                i2 += QR_TPB
            barrier()

            # Apply `H = I - tau w w'` to the trailing columns.
            for c in range(j + 1, n):
                var dacc = Float32(0.0)
                var i3 = j + 1 + tid
                while i3 < ms:
                    var w = ftz(a.unsafe_load((rb + i3) * lda + j))
                    var x = ftz(a.unsafe_load((rb + i3) * lda + c))
                    dacc = ftz(identical_mul_add(w, x, dacc))
                    i3 += QR_TPB
                var tail = fold_and_broadcast[QR_TPB](dacc)
                var ajc = ftz(a.unsafe_load((rb + j) * lda + c))
                # arima SEEDS the fold with `a[j][c]` and adds upward; a
                # block fold cannot, so the implicit `w_j = 1` term is added
                # to the folded tail instead. Same multiset, different
                # association: DEVIATION 587, stated at the one line where
                # the two routines visibly part.
                var total = ftz(ajc + tail)
                var td = ftz(tau * total)
                if tid == 0:
                    a.unsafe_store((rb + j) * lda + c, ftz(ajc - td))
                var i4 = j + 1 + tid
                while i4 < ms:
                    var w2 = ftz(a.unsafe_load((rb + i4) * lda + j))
                    var cur2 = ftz(a.unsafe_load((rb + i4) * lda + c))
                    a.unsafe_store(
                        (rb + i4) * lda + c,
                        ftz(identical_mul_add(-td, w2, cur2)),
                    )
                    i4 += QR_TPB
                barrier()
    if Int(finish_in) != 0:
        # The strict upper triangle of `R` is row `j` of the factored slice; the
        # diagonal was written above and is not touched here. A slice with fewer
        # rows than columns (only reachable at `n_slices == 1`, where the host
        # refuses it, or on the second TSQR pass, where `ms = n_slices * n >= n`)
        # would read past its own rows, so those cells are zeroed instead.
        barrier()
        var t = tid
        while t < n * n:
            var rr = t // n
            var cc = t - rr * n
            if cc > rr:
                if rr < ms:
                    r_out.unsafe_store(
                        rbase + t, ftz(a.unsafe_load((rb + rr) * lda + cc))
                    )
                else:
                    r_out.unsafe_store(rbase + t, Float32(0.0))
            elif cc < rr:
                r_out.unsafe_store(rbase + t, Float32(0.0))
            t += QR_TPB




def _qrb_pass(
    ctx: DeviceContext,
    a: MutPointer[Float32, MutAnyOrigin],
    r_out: MutPointer[Float32, MutAnyOrigin],
    m: Int,
    n: Int,
    ns: Int,
    cells: Int,
) raises:
    """One `qr_panel_kernel` launch's work (grid `ns`, one block per slice) as
    consecutive column ranges of about QRB_CELLS cells, each waited for."""
    var rows = (m + ns - 1) // ns
    var j0 = 0
    while j0 < n:
        var cost = 0
        var j1 = j0
        while j1 < n and (j1 == j0 or cost < cells):
            var h = rows - j1
            if h < 0:
                h = 0
            cost += h * (n - j1) + QRB_BARRIER_CELLS * (n - j1 + 2)
            j1 += 1
        ctx.enqueue_function[qr_panel_cols_kernel](
            a, r_out, Int32(m), Int32(n), Int32(n), Int32(ns), Int32(j0), Int32(j1), Int32(1 if j1 == n else 0),
            grid_dim=(ns, 1, 1),
            block_dim=(QR_TPB, 1, 1),
        )
        ctx.synchronize()
        j0 = j1


def qr_factor_bounded(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut r_scratch: DeviceBuffer[DType.float32],
    mut r_out: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    cells: Int = QRB_CELLS,
) raises -> Int:
    """`qr_factor(ctx, a, r_scratch, r_out, n_rows, n_cols)` (the same slices,
    the same passes, the same bits; DESTROYS `a`), each pass cut into bounded
    launches of about `cells` cells (the bits do not depend on it), r_out
    (and the TSQR tiles) poisoned with NaN first."""
    if n_rows < n_cols:
        raise Error("qr_factor_bounded needs at least as many rows as columns, got "
                    + String(n_rows) + " x " + String(n_cols))
    var ns = qr_slice_count(n_rows, n_cols)
    enqueue_fill(ctx, r_out, Float32(0.0) / Float32(0.0))
    if ns == 1:
        _qrb_pass(ctx, a.unsafe_ptr(), r_out.unsafe_ptr(), n_rows, n_cols, 1, cells)
        return 1
    enqueue_fill(ctx, r_scratch, Float32(0.0) / Float32(0.0))
    _qrb_pass(ctx, a.unsafe_ptr(), r_scratch.unsafe_ptr(), n_rows, n_cols, ns, cells)
    _qrb_pass(ctx, r_scratch.unsafe_ptr(), r_out.unsafe_ptr(), ns * n_cols, n_cols, 1, cells)
    return ns
