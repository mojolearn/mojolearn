# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""QuantileRegressor's ADMM on the grid (cgr-linear, 2026-10-03).

The fit kernel ran the whole ADMM on ONE block of 256 threads: every row
fold was one thread's loop over every row, every cell of A'A one thread's
loop over every row. Here every row pass is a grid launch in the blocked
order of x_linear/quantile.mojo (FOLD_BLOCK rows from zero, then the block
partials folded ascending, a thread per cell), the m x m factor and solves
run on one block (`t_cholesky`, `t_chol_solve_cols`: a column's entries
across the block), and the d-sized z/v update and the stop rule
(`q_tail`) on one thread. The host column (`_quantile_fit_host`) runs the
same folds in the same order, so the words agree.

Per iteration: the right-hand side's partials (only after a rescale; the
previous iteration's fold pass made them otherwise), the solve, the row
update, the fold pass (A' dr, next A'(y - r - u) and the four row norms),
the tail, and u's rescale. A converged fit sets a state word every later
launch reads first, so iterations are queued in batches and the host reads
the state word once a batch. Off Apple nothing waits; on Apple each
iteration's launches are one witness unit (in place, so a cut raises).
"""
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from x_linear.ops import FP, IP, fa, fs, fm, fd, fmad, ld, st, ldi, i2f, row_dot
from x_linear.tops import fold_parts, fold_blocks, fold_fa, fold_sq, FOLD_BLOCK, t_cholesky
from x_linear.team import device_team, team_work, LINEAR_TPB
from x_linear.finite_device import XLIN_IDN_DEV_FINITE, xlin_finite_device
from x_linear.witness import Witness, witness_end, WITNESS_TRIES
from x_linear.quantile import (
    q_lower_cell, q_a, q_gram_part, q_gram_cell, q_spread_part, q_resid, q_prox,
    t_chol_solve_cols, q_layout, q_start, q_tail,
)

comptime QG_TPB = 256
#: Iterations queued between reads of the stop word.
comptime QG_BATCH = 16

# state words
comptime QS_DEN = 0
comptime QS_RHO = 1
comptime QS_YNORM = 2
comptime QS_DONE = 3
comptime QS_ITERS = 4
comptime QS_READY = 5
comptime QS_RESCALE = 6
comptime QS_INV = 7
comptime QS_CONV = 8
comptime QS_WORDS = 16


@always_inline
def _qb(count: Int) -> Int:
    return (count + QG_TPB - 1) // QG_TPB


@always_inline
def _gid() -> Int:
    return Int(block_idx.x) * QG_TPB + Int(thread_idx.x)


@always_inline
def _done(state: FP) -> Bool:
    return ld(state, QS_DONE) != Float32(0)


# ------------------------------------------------ setup
@always_inline
def _qg_gram_parts_body(x: FP, n: Int32, d: Int32, cols: Int32, gp: FP):
    var nn = Int(n)
    var mi = Int(cols)
    var cells = mi * (mi + 1) // 2
    var nb = fold_blocks(nn)
    var t = _gid()
    if t < cells * nb:
        var c = t % cells
        var b = t // cells
        var jk = q_lower_cell(c)
        var lo = b * FOLD_BLOCK
        st(gp, c * nb + b, q_gram_part(x, Int(d), jk[0], jk[1], lo, min(FOLD_BLOCK, nn - lo)))


def qg_gram_parts_kernel(x: FP, n: Int32, d: Int32, cols: Int32, gp: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread (cell, block): the cell's row-block partial of A'A."""
    _qg_gram_parts_body(x, n, d, cols, gp)
    witness_end(wf, woff, nonce)


@always_inline
def _qg_gram_fin_body(gp: FP, nb: Int32, d: Int32, cols: Int32, fw: FP):
    var mi = Int(cols)
    var cells = mi * (mi + 1) // 2
    var c = _gid()
    if c < cells:
        var jk = q_lower_cell(c)
        var o = q_layout(0, Int(d), mi)
        q_gram_cell(fw, o[0], mi, Int(d), o[10], jk[0], jk[1], fold_parts(gp, c * Int(nb), Int(nb)))


def qg_gram_fin_kernel(gp: FP, nb: Int32, d: Int32, cols: Int32, fw: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread per cell: the partials folded blocks ascending, then M's cell."""
    _qg_gram_fin_body(gp, nb, d, cols, fw)
    witness_end(wf, woff, nonce)


@always_inline
def _qg_scal_parts_body(y: FP, n: Int32, sw: Int32, sp: FP):
    var nn = Int(n)
    var nb = fold_blocks(nn)
    var b = _gid()
    if b < nb:
        var lo = b * FOLD_BLOCK
        var cnt = min(FOLD_BLOCK, nn - lo)
        st(sp, b, fold_fa(y, nn + lo, 1, cnt) if sw != 0 else Float32(0))
        st(sp, nb + b, fold_fa(y, lo, 1, cnt))
        st(sp, 2 * nb + b, fold_sq(y, lo, cnt))


def qg_scal_parts_kernel(y: FP, n: Int32, sw: Int32, sp: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread per row block: sum(w), sum(y), |y|^2 from zero."""
    _qg_scal_parts_body(y, n, sw, sp)
    witness_end(wf, woff, nonce)


@always_inline
def _qg_spread_parts_body(y: FP, n: Int32, sp: FP):
    var nn = Int(n)
    var nb = fold_blocks(nn)
    var b = _gid()
    if b < nb:
        var ym = fd(fold_parts(sp, nb, nb), i2f(nn))
        var lo = b * FOLD_BLOCK
        st(sp, 3 * nb + b, q_spread_part(y, ym, lo, min(FOLD_BLOCK, nn - lo)))


def qg_spread_parts_kernel(y: FP, n: Int32, sp: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread per row block: sum |y - ym| from zero (ym the blocked mean)."""
    _qg_spread_parts_body(y, n, sp)
    witness_end(wf, woff, nonce)


@always_inline
def _qg_setup_body(nb: Int32, nf: Float32, cols: Int32, sw: Int32, fw: FP, sp: FP, state: FP, tw: FP):
    var t = device_team(tw, 0, 3, 0)
    _ = t_cholesky(t, fw, 0, Int(cols))
    if t.lead():
        var b = Int(nb)
        var s3 = q_start(nf, sw != 0, fold_parts(sp, 0, b), fold_parts(sp, 3 * b, b), fold_parts(sp, 2 * b, b))
        for k in range(QS_WORDS):
            st(state, k, Float32(0))
        st(state, QS_DEN, s3[0])
        st(state, QS_RHO, s3[1])
        st(state, QS_YNORM, s3[2])


def qg_setup_kernel(nb: Int32, nf: Float32, cols: Int32, sw: Int32, fw: FP, sp: FP, state: FP, tw: FP,
                    wf: IP, woff: Int32, nonce: Int32):
    """One block: the Cholesky factor of M (m x m); the lead folds the nb
    row-block partials into the starting scalars."""
    _qg_setup_body(nb, nf, cols, sw, fw, sp, state, tw)
    witness_end(wf, woff, nonce)


# ------------------------------------------------ one iteration
# rows: r n | u n | ab n | tmp n (the row vectors; fw holds only the m- and
# d-sized ones, x_linear/quantile.mojo `q_layout` with no rows)
@always_inline
def _qg_rhs_parts_body(x: FP, y: FP, n: Int32, d: Int32, cols: Int32, rows: FP, state: FP, pn: FP):
    if _done(state) or ld(state, QS_READY) != Float32(0):
        return
    var nn = Int(n)
    var dd = Int(d)
    var mi = Int(cols)
    var nb = fold_blocks(nn)
    var t = _gid()
    if t < mi * nb:
        var j = t % mi
        var b = t // mi
        var lo = b * FOLD_BLOCK
        var acc = Float32(0)
        for i in range(lo, lo + min(FOLD_BLOCK, nn - lo)):
            acc = fmad(q_a(x, i, dd, j), q_resid(y, rows, 0, nn, i), acc)
        st(pn, j * nb + b, acc)


def qg_rhs_parts_kernel(x: FP, y: FP, n: Int32, d: Int32, cols: Int32, rows: FP, state: FP, pn: FP,
                        wf: IP, woff: Int32, nonce: Int32):
    """Thread (column, row block): A'(y - r - u)'s partial, when the last
    fold pass did not make it (the first iteration, after a rescale)."""
    _qg_rhs_parts_body(x, y, n, d, cols, rows, state, pn)
    witness_end(wf, woff, nonce)


@always_inline
def _qg_solve_body(nb: Int32, d: Int32, cols: Int32, fw: FP, state: FP, pn: FP, tw: FP):
    if _done(state):
        return
    var dd = Int(d)
    var mi = Int(cols)
    var b = Int(nb)
    var o = q_layout(0, dd, mi)
    var t = device_team(tw, 0, 3, 0)
    for j in range(t.tid, mi, t.nt):
        var acc = fold_parts(pn, j * b, b)
        if j < dd:
            acc = fa(acc, fm(ld(fw, o[10] + j), fs(ld(fw, o[7] + j), ld(fw, o[8] + j))))
        st(fw, o[2] + j, acc)
    t.sync()
    t_chol_solve_cols(t, fw, o[0], mi, fw, o[2], fw, o[1])
    for j in range(t.tid, mi, t.nt):
        st(fw, o[1] + j, ld(fw, o[2] + j))


def qg_solve_kernel(nb: Int32, d: Int32, cols: Int32, fw: FP, state: FP, pn: FP, tw: FP,
                    wf: IP, woff: Int32, nonce: Int32):
    """One block: rhs = the folded partials + D^2 (z - v), then beta (m x m)."""
    _qg_solve_body(nb, d, cols, fw, state, pn, tw)
    witness_end(wf, woff, nonce)


@always_inline
def _qg_rmap_body(x: FP, y: FP, n: Int32, d: Int32, cols: Int32, fi: Int32, sw: Int32, fw: FP, rows: FP, fp: FP,
                  state: FP, prb: FP):
    if _done(state):
        return
    var nn = Int(n)
    var dd = Int(d)
    var i = _gid()
    if i < nn:
        var o = q_layout(0, dd, Int(cols))
        var q = ld(fp, 0)
        var b = ld(fw, o[1] + dd) if fi != 0 else Float32(0)
        var kq = fd(Float32(1), fm(ld(state, QS_DEN), ld(state, QS_RHO)))
        var up = fm(q, kq)
        var lo_ = fm(fs(Float32(1), q), kq)
        var abi = fa(row_dot(x, i, dd, fw, o[1]), b)
        st(rows, 2 * nn + i, abi)
        var vv = fs(fs(ld(y, i), abi), ld(rows, nn + i))
        if sw != 0:
            var ki = fm(ld(y, nn + i), kq)
            up = fm(q, ki)
            lo_ = fm(fs(Float32(1), q), ki)
        var nr = q_prox(vv, up, lo_)
        st(rows, 3 * nn + i, fs(nr, ld(rows, i)))
        st(rows, i, nr)
        var pr = fs(fa(abi, nr), ld(y, i))
        st(prb, i, pr)
        st(rows, nn + i, fa(ld(rows, nn + i), pr))


def qg_rmap_kernel(x: FP, y: FP, n: Int32, d: Int32, cols: Int32, fi: Int32, sw: Int32, fw: FP, rows: FP, fp: FP,
                   state: FP, prb: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread per row: A beta, the r-update (the old r's change in tmp), the
    primal residual and the dual update of u."""
    _qg_rmap_body(x, y, n, d, cols, fi, sw, fw, rows, fp, state, prb)
    witness_end(wf, woff, nonce)


@always_inline
def _qg_iter_parts_body(x: FP, y: FP, n: Int32, d: Int32, cols: Int32, rows: FP, state: FP, prb: FP,
                        pdr: FP, pn: FP, sc: FP):
    if _done(state):
        return
    var nn = Int(n)
    var dd = Int(d)
    var mi = Int(cols)
    var nb = fold_blocks(nn)
    var t = _gid()
    if t < (mi + 1) * nb:
        var j = t % (mi + 1)
        var b = t // (mi + 1)
        var lo = b * FOLD_BLOCK
        var cnt = min(FOLD_BLOCK, nn - lo)
        if j < mi:
            var a1 = Float32(0)
            var a2 = Float32(0)
            for i in range(lo, lo + cnt):
                var aij = q_a(x, i, dd, j)
                a1 = fmad(aij, ld(rows, 3 * nn + i), a1)
                a2 = fmad(aij, q_resid(y, rows, 0, nn, i), a2)
            st(pdr, j * nb + b, a1)
            st(pn, j * nb + b, a2)
        else:
            st(sc, b, fold_sq(rows, 2 * nn + lo, cnt))
            st(sc, nb + b, fold_sq(rows, lo, cnt))
            st(sc, 2 * nb + b, fold_sq(prb, lo, cnt))
            st(sc, 3 * nb + b, fold_sq(rows, nn + lo, cnt))


def qg_iter_parts_kernel(x: FP, y: FP, n: Int32, d: Int32, cols: Int32, rows: FP, state: FP, prb: FP,
                         pdr: FP, pn: FP, sc: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread (column, row block): A' dr's and the next A'(y - r - u)'s
    partials; column m: the block's |A beta|^2, |r|^2, primal |.|^2, |u|^2."""
    _qg_iter_parts_body(x, y, n, d, cols, rows, state, prb, pdr, pn, sc)
    witness_end(wf, woff, nonce)


@always_inline
def _qg_iter_fin_body(nb: Int32, nd: Int32, d: Int32, cols: Int32, fw: FP, fp: FP, state: FP, pdr: FP, sc: FP,
                      it: Int32, tw: FP):
    if _done(state):
        return
    var dd = Int(d)
    var mi = Int(cols)
    var b = Int(nb)
    var o = q_layout(0, dd, mi)
    var t = device_team(tw, 0, 3, 0)
    for j in range(t.tid, mi, t.nt):
        st(fw, o[2] + j, fold_parts(pdr, j * b, b))
    t.sync()
    if t.lead():
        var rho = ld(state, QS_RHO)
        var tl = q_tail(fw, Int(nd), dd, mi, o[1], o[2], o[7], o[8], o[10],
                        fold_parts(sc, 0, b), fold_parts(sc, b, b), fold_parts(sc, 2 * b, b),
                        fold_parts(sc, 3 * b, b), ld(state, QS_YNORM), rho,
                        ld(fp, 1), ld(fp, 2), ld(fp, 3), ld(fp, 4) != Float32(0), Int(it))
        st(state, QS_ITERS, i2f(Int(it) + 1))
        st(state, QS_RESCALE, Float32(0))
        st(state, QS_READY, Float32(1))
        if tl[0] == 1:
            st(state, QS_CONV, Float32(1))
            st(state, QS_DONE, Float32(1))
        elif tl[0] == 2:
            st(state, QS_RHO, fm(rho, tl[1]))
            st(state, QS_INV, fd(Float32(1), tl[1]))
            st(state, QS_RESCALE, Float32(1))
            st(state, QS_READY, Float32(0))


def qg_iter_fin_kernel(nb: Int32, nd: Int32, d: Int32, cols: Int32, fw: FP, fp: FP, state: FP, pdr: FP, sc: FP,
                       it: Int32, tw: FP, wf: IP, woff: Int32, nonce: Int32):
    """One block: A' dr folded per column (nb partials each), then the
    lead's d-sized `q_tail` (nd = n + d)."""
    _qg_iter_fin_body(nb, nd, d, cols, fw, fp, state, pdr, sc, it, tw)
    witness_end(wf, woff, nonce)


@always_inline
def _qg_rescale_body(n: Int32, rows: FP, state: FP):
    if _done(state) or ld(state, QS_RESCALE) == Float32(0):
        return
    var i = _gid()
    var nn = Int(n)
    if i < nn:
        st(rows, nn + i, fm(ld(rows, nn + i), ld(state, QS_INV)))


def qg_rescale_kernel(n: Int32, rows: FP, state: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread per row: u scaled by 1 / factor after a rescale."""
    _qg_rescale_body(n, rows, state)
    witness_end(wf, woff, nonce)


@always_inline
def _qg_finish_body(d: Int32, cols: Int32, fi: Int32, fw: FP, state: FP, res: FP):
    var dd = Int(d)
    var o = q_layout(0, dd, Int(cols))
    for j in range(Int(thread_idx.x), dd, Int(block_dim.x)):
        st(res, j, ld(fw, o[7] + j))
    if Int(thread_idx.x) == 0:
        st(res, dd, ld(fw, o[1] + dd) if fi != 0 else Float32(0))
        st(res, dd + 1, ld(state, QS_ITERS))
        st(res, dd + 2, ld(state, QS_CONV))


def qg_finish_kernel(d: Int32, cols: Int32, fi: Int32, fw: FP, state: FP, res: FP,
                     wf: IP, woff: Int32, nonce: Int32):
    """res: coef (z) d, intercept, n_iter, converged."""
    _qg_finish_body(d, cols, fi, fw, state, res)
    witness_end(wf, woff, nonce)


def quantile_fit_grid(
    var ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, res: FP,
) raises:
    """ip: [max_iter, fit_intercept, sample_weight]; fp: [quantile, alpha,
    eps_abs, eps_rel, relative balancing]. res: coef d, intercept, n_iter,
    converged (x_linear/quantile.mojo `quantile_fit`)."""
    var max_iter = Int(ip[0])
    var fi = Int(ip[1]) if len(ip) > 1 else 1
    var sw = Int(ip[2]) if len(ip) > 2 else 0
    var cols = d + 1 if fi != 0 else d
    var nb = fold_blocks(n)
    var cells = cols * (cols + 1) // 2
    var o = q_layout(0, d, cols)
    var hfp = fp.copy()
    while len(hfp) < 5:
        hfp.append(Float32(0))
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var dfp = ctx.enqueue_create_buffer[DType.float32](len(hfp))
    var dfw = ctx.enqueue_create_buffer[DType.float32](max(o[11], 1))
    var drows = ctx.enqueue_create_buffer[DType.float32](max(4 * n, 1))
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n_out, d + 3))
    var dgp = ctx.enqueue_create_buffer[DType.float32](max(cells * nb, 1))
    var dsp = ctx.enqueue_create_buffer[DType.float32](max(4 * nb, 1))
    var dpn = ctx.enqueue_create_buffer[DType.float32](max(cols * nb, 1))
    var dpdr = ctx.enqueue_create_buffer[DType.float32](max(cols * nb, 1))
    var dsc = ctx.enqueue_create_buffer[DType.float32](max(4 * nb, 1))
    var dprb = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dstate = ctx.enqueue_create_buffer[DType.float32](QS_WORDS)
    var dtw = ctx.enqueue_create_buffer[DType.float32](team_work(0, 3, 0))
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    comptime if XLIN_IDN_DEV_FINITE:
        xlin_finite_device(ctx, dx, n_x, y, n_y)
    ctx.enqueue_copy(dst_buf=dfp, src_ptr=hfp.unsafe_ptr())
    var xp = dx.unsafe_ptr()
    var yp = dy.unsafe_ptr()
    var fwp = dfw.unsafe_ptr()
    var rp = drows.unsafe_ptr()
    var stp = dstate.unsafe_ptr()
    var nf = i2f(n)
    var g_parts = max(_qb(cells * nb), 1)
    var g_cells = max(_qb(cells), 1)
    var g_nb = max(_qb(nb), 1)
    var g_rows = max(_qb(n), 1)
    var g_rhs = max(_qb(cols * nb), 1)
    var g_iter = max(_qb((cols + 1) * nb), 1)
    var w_setup = g_parts + g_cells + 2 * g_nb + 1
    var w_iter = g_rhs + 1 + g_rows + g_iter + 1 + g_rows
    var wit = Witness(ctx, max(w_setup, w_iter) + 1)
    # the setup rebuilds from zeroed scratch: one unit, rerun on a cut
    var tries = 0
    while True:
        var nonce = wit.begin()
        var wo = 0
        dfw.enqueue_fill(Float32(0))
        drows.enqueue_fill(Float32(0))
        ctx.enqueue_function[qg_gram_parts_kernel](
            xp, Int32(n), Int32(d), Int32(cols), dgp.unsafe_ptr(), wit.p(), Int32(wo), nonce,
            grid_dim=g_parts, block_dim=QG_TPB,
        )
        wo += g_parts
        ctx.enqueue_function[qg_gram_fin_kernel](
            dgp.unsafe_ptr(), Int32(nb), Int32(d), Int32(cols), fwp, wit.p(), Int32(wo), nonce,
            grid_dim=g_cells, block_dim=QG_TPB,
        )
        wo += g_cells
        ctx.enqueue_function[qg_scal_parts_kernel](
            yp, Int32(n), Int32(sw), dsp.unsafe_ptr(), wit.p(), Int32(wo), nonce,
            grid_dim=g_nb, block_dim=QG_TPB,
        )
        wo += g_nb
        ctx.enqueue_function[qg_spread_parts_kernel](
            yp, Int32(n), dsp.unsafe_ptr(), wit.p(), Int32(wo), nonce,
            grid_dim=g_nb, block_dim=QG_TPB,
        )
        wo += g_nb
        ctx.enqueue_function[qg_setup_kernel](
            Int32(nb), nf, Int32(cols), Int32(sw), fwp, dsp.unsafe_ptr(), stp, dtw.unsafe_ptr(),
            wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=LINEAR_TPB,
        )
        wo += 1
        if wit.ok(ctx, wo, "QuantileRegressor setup"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    var hst = List[Float32](length=QS_WORDS, fill=Float32(0))
    for it in range(max_iter):
        # in place: a cut raises
        var nonce = wit.begin()
        var wo = 0
        ctx.enqueue_function[qg_rhs_parts_kernel](
            xp, yp, Int32(n), Int32(d), Int32(cols), rp, stp, dpn.unsafe_ptr(), wit.p(), Int32(wo), nonce,
            grid_dim=g_rhs, block_dim=QG_TPB,
        )
        wo += g_rhs
        ctx.enqueue_function[qg_solve_kernel](
            Int32(nb), Int32(d), Int32(cols), fwp, stp, dpn.unsafe_ptr(), dtw.unsafe_ptr(), wit.p(), Int32(wo), nonce,
            grid_dim=1, block_dim=LINEAR_TPB,
        )
        wo += 1
        ctx.enqueue_function[qg_rmap_kernel](
            xp, yp, Int32(n), Int32(d), Int32(cols), Int32(fi), Int32(sw), fwp, rp, dfp.unsafe_ptr(), stp,
            dprb.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=g_rows, block_dim=QG_TPB,
        )
        wo += g_rows
        ctx.enqueue_function[qg_iter_parts_kernel](
            xp, yp, Int32(n), Int32(d), Int32(cols), rp, stp, dprb.unsafe_ptr(), dpdr.unsafe_ptr(),
            dpn.unsafe_ptr(), dsc.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=g_iter, block_dim=QG_TPB,
        )
        wo += g_iter
        ctx.enqueue_function[qg_iter_fin_kernel](
            Int32(nb), Int32(n + d), Int32(d), Int32(cols), fwp, dfp.unsafe_ptr(), stp, dpdr.unsafe_ptr(),
            dsc.unsafe_ptr(), Int32(it), dtw.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=LINEAR_TPB,
        )
        wo += 1
        ctx.enqueue_function[qg_rescale_kernel](
            Int32(n), rp, stp, wit.p(), Int32(wo), nonce, grid_dim=g_rows, block_dim=QG_TPB,
        )
        wo += g_rows
        if not wit.ok(ctx, wo, "QuantileRegressor iteration"):
            wit.fail()
        if (it + 1) % QG_BATCH == 0 and it + 1 < max_iter:
            ctx.enqueue_copy(dst_ptr=hst.unsafe_ptr(), src_buf=dstate)
            ctx.synchronize()
            if hst[QS_DONE] != Float32(0):
                break
    var nf2 = wit.begin()
    ctx.enqueue_function[qg_finish_kernel](
        Int32(d), Int32(cols), Int32(fi), fwp, stp, dout.unsafe_ptr(), wit.p(), Int32(0), nf2,
        grid_dim=1, block_dim=QG_TPB,
    )
    if not wit.ok(ctx, 1, "QuantileRegressor finish"):
        wit.fail()
    if n_out > 0:
        ctx.enqueue_copy(dst_ptr=res, src_buf=dout.create_sub_buffer[DType.float32](0, n_out))
    ctx.synchronize()
    _ = hfp^
    _ = hst^
    _ = dx^
    _ = dy^
    _ = dfp^
    _ = dfw^
    _ = drows^
    _ = dout^
    _ = dgp^
    _ = dsp^
    _ = dpn^
    _ = dpdr^
    _ = dsc^
    _ = dprb^
    _ = dstate^
    _ = dtw^
    _ = wit^
