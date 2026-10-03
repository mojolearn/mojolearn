# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Ridge / RidgeClassifier / RidgeCV (leave-one-out) on the grid (cgr-linear,
2026-10-03).

The fit kernel ran these on ONE block: the weighted statistics (one thread a
cell over every row), every alpha's leave-one-out rows (256 threads, each a
d x d solve per row) and the error folds on the lead. Here:
  * unweighted: the moments of [X | Y] from x_linear/moments_grid.mojo (the
    host column's chains); weighted: the blocked statistics of
    x_linear/ridge.mojo (`ridge_w_part`, `ridge_wgram_part`,
    `ridge_wxty_part`), a thread per (statistic, row block), then a thread
    per statistic;
  * per alpha: the factor on one block team (`t_cholesky`, d-sized) and the
    lead's trust test and right-hand solve; each row's leave-one-out residual
    a thread (`_loo_rows`, its own d words of scratch, rows in slices); the
    error's FOLD_BLOCK partials a thread each; one thread folds them and
    keeps the first minimum;
  * the fit at the chosen alpha on one block team (`t_ridge_solve_best`).
The float-float refit (status 1) stays x_linear/device.mojo's `_ridge_ff_grid`.
"""
from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from x_linear.ops import FP, IP, fa, fd, ld, st, i2f, copy, chol_solve
from x_linear.tops import fold_parts, fold_blocks, FOLD_BLOCK, t_cholesky, upper_cell
from x_linear.team import device_team, team_work, LINEAR_TPB
from x_linear.witness import Witness, witness_end, WITNESS_TRIES
from x_linear.moments_grid import mg_means_kernel, mg_cross_kernel, mg_tiles, MG_NT, MOMENTS_GRID
from x_linear.ridge import (
    ridge_w_part, ridge_wgram_part, ridge_wxty_part, ridge_err_part, t_ridge_solve_best, _loo_rows,
    _chol_trusted, BIG_ERR,
)

comptime RG_TPB = 256
comptime RS_TRUST = 0
comptime RS_BEST = 1
comptime RS_BERR = 2
comptime RS_WSUM = 3
comptime RS_WORDS = 8


@always_inline
def _rb(count: Int) -> Int:
    return max((count + RG_TPB - 1) // RG_TPB, 1)


@always_inline
def _gid() -> Int:
    return Int(block_idx.x) * RG_TPB + Int(thread_idx.x)


@always_inline
def _ridge_off(d: Int, t_n: Int) -> InlineArray[Int, 6]:
    """fw: xm d | G d*d | M d*d | rhs d | ym T | xty d*T."""
    var o = InlineArray[Int, 6](fill=0)
    o[0] = 0
    o[1] = d
    o[2] = d + d * d
    o[3] = o[2] + d * d
    o[4] = o[3] + d
    o[5] = o[4] + t_n
    return o^


def rw_parts_kernel(x: FP, y: FP, n: Int32, d: Int32, t_n: Int32, wp: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread (column of [X | Y | 1], block): the weighted block sums."""
    var nn = Int(n)
    var dd = Int(d)
    var tn = Int(t_n)
    var nb = fold_blocks(nn)
    var cols = dd + tn + 1
    var t = _gid()
    if t < cols * nb:
        var c = t % cols
        var b = t // cols
        var lo = b * FOLD_BLOCK
        st(wp, c * nb + b, ridge_w_part(x, y, dd, tn, nn * tn, c, lo, min(FOLD_BLOCK, nn - lo)))
    witness_end(wf, woff, nonce)


def rw_means_kernel(n: Int32, d: Int32, t_n: Int32, fi: Int32, wp: FP, fw: FP, state: FP,
                    wf: IP, woff: Int32, nonce: Int32):
    """Thread per column: the weighted mean (0 without an intercept); thread 0
    also keeps sum(w)."""
    var dd = Int(d)
    var tn = Int(t_n)
    var nb = fold_blocks(Int(n))
    var c = _gid()
    var wsum = fold_parts(wp, (dd + tn) * nb, nb)
    if c < dd + tn:
        var o = _ridge_off(dd, tn)
        var v = fd(fold_parts(wp, c * nb, nb), wsum) if fi != 0 else Float32(0)
        st(fw, (o[0] + c) if c < dd else (o[4] + c - dd), v)
    if c == 0:
        st(state, RS_WSUM, wsum)
    witness_end(wf, woff, nonce)


def rw_gram_parts_kernel(x: FP, y: FP, n: Int32, d: Int32, t_n: Int32, fw: FP, gp: FP,
                         wf: IP, woff: Int32, nonce: Int32):
    """Thread (statistic, block): an upper cell of the weighted Gram, or
    (after the cells) an X'Y entry."""
    var nn = Int(n)
    var dd = Int(d)
    var tn = Int(t_n)
    var nb = fold_blocks(nn)
    var cells = dd * (dd + 1) // 2
    var stats = cells + dd * tn
    var t = _gid()
    if t < stats * nb:
        var c = t % stats
        var b = t // stats
        var lo = b * FOLD_BLOCK
        var cnt = min(FOLD_BLOCK, nn - lo)
        var o = _ridge_off(dd, tn)
        if c < cells:
            var jk = upper_cell(c, dd)
            st(gp, c * nb + b, ridge_wgram_part(x, y, dd, nn * tn, fw, o[0], jk[0], jk[1], lo, cnt))
        else:
            var q = c - cells
            st(gp, c * nb + b, ridge_wxty_part(x, y, dd, tn, nn * tn, fw, o[0], o[4], q // dd, q % dd, lo, cnt))
    witness_end(wf, woff, nonce)


def rw_gram_fin_kernel(n: Int32, d: Int32, t_n: Int32, gp: FP, fw: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread per statistic: its partials folded blocks ascending into fw."""
    var dd = Int(d)
    var tn = Int(t_n)
    var nb = fold_blocks(Int(n))
    var cells = dd * (dd + 1) // 2
    var c = _gid()
    if c < cells + dd * tn:
        var o = _ridge_off(dd, tn)
        var v = fold_parts(gp, c * nb, nb)
        if c < cells:
            var jk = upper_cell(c, dd)
            st(fw, o[1] + jk[0] * dd + jk[1], v)
            st(fw, o[1] + jk[1] * dd + jk[0], v)
        else:
            st(fw, o[5] + (c - cells), v)
    witness_end(wf, woff, nonce)


def ra_factor_kernel(d: Int32, alpha: Float32, fw: FP, state: FP, tw: FP, wf: IP, woff: Int32, nonce: Int32):
    """One block team: M = G + alpha I and its factor (d x d); the lead's
    trust test and the right-hand solve (rhs = M^-1 X'y)."""
    var dd = Int(d)
    var o = _ridge_off(dd, 1)
    var t = device_team(tw, 0, 3, 0)
    for c in range(t.tid, dd * dd, t.nt):
        var v = ld(fw, o[1] + c)
        if c // dd == c % dd:
            v = fa(v, alpha)
        st(fw, o[2] + c, v)
    t.sync()
    var ok = t_cholesky(t, fw, o[2], dd)
    if t.lead():
        var tr = _chol_trusted(ok, fw, o[2], o[1], dd, alpha)
        st(state, RS_TRUST, Float32(1) if tr else Float32(0))
        if tr:
            copy(fw, o[3], fw, o[5], dd)
            chol_solve(fw, o[2], dd, fw, o[3])
    witness_end(wf, woff, nonce)


def ra_loo_kernel(x: FP, y: FP, n: Int32, d: Int32, sw: Int32, fi: Int32, lo: Int32, cnt: Int32, fw: FP,
                  state: FP, scr: FP, la: FP, lb: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread per row of [lo, lo + cnt): its leave-one-out terms (its own d
    words of scratch)."""
    var q = _gid()
    if q < Int(cnt) and ld(state, RS_TRUST) != Float32(0):
        var dd = Int(d)
        var nn = Int(n)
        var o = _ridge_off(dd, 1)
        var i = Int(lo) + q
        _loo_rows(x, y, nn, dd, fw, o[4], o[0], o[3], scr + q * dd, o[2], sw != 0, fi != 0, nn,
                  ld(state, RS_WSUM), la, lb, i, i + 1)
    witness_end(wf, woff, nonce)


def ra_err_parts_kernel(n: Int32, la: FP, lb: FP, state: FP, ep: FP, wf: IP, woff: Int32, nonce: Int32):
    var nn = Int(n)
    var b = _gid()
    if b < fold_blocks(nn) and ld(state, RS_TRUST) != Float32(0):
        var lo = b * FOLD_BLOCK
        st(ep, b, ridge_err_part(la, lb, lo, min(FOLD_BLOCK, nn - lo)))
    witness_end(wf, woff, nonce)


def ra_err_fin_kernel(nb: Int32, nf: Float32, a: Int32, slot: Int32, ep: FP, state: FP, res: FP,
                      wf: IP, woff: Int32, nonce: Int32):
    """One thread: alpha a's error (nb partials folded) and the first minimum."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        var ai = Int(a)
        if ld(state, RS_TRUST) == Float32(0):
            st(res, Int(slot), BIG_ERR)
            if ai == 0:
                st(state, RS_BERR, BIG_ERR)
        else:
            var err = fd(fold_parts(ep, 0, Int(nb)), nf)
            st(res, Int(slot), err)
            if ai == 0 or err < ld(state, RS_BERR):  # DEVIATION 5005: the first minimum
                st(state, RS_BEST, i2f(ai))
                st(state, RS_BERR, err)
    witness_end(wf, woff, nonce)


def ra_best_kernel(d: Int32, t_n: Int32, fi: Int32, a_n: Int32, fp: FP, res: FP, fw: FP, state: FP, scr: FP, tw: FP,
                   wf: IP, woff: Int32, nonce: Int32):
    """One block team: the fit at the chosen alpha (`t_ridge_solve_best`)."""
    var t = device_team(tw, 0, 3, 0)
    _ = t_ridge_solve_best(t, fp, res, fw, Int(d), Int(t_n), fi != 0, Int(ld(state, RS_BEST)),
                           ld(state, RS_BERR), Int(a_n), scr)
    witness_end(wf, woff, nonce)


def ridge_fit_grid(
    var ctx: DeviceContext, xp: FP, yp: FP, n: Int, d: Int, ip: List[Int32], fp: List[Float32],
    n_out: Int, res: FP,
) raises:
    """xp, yp: the device X and Y (y: targets n*T | weights n). ip: [T,
    fit_intercept, A, sample_weight]; fp: alphas. res as x_linear/ridge.mojo
    `ridge_fit` says (status at T*d + T + 2 + A for the float-float refit)."""
    comptime assert MOMENTS_GRID, "the moments grid's page must fit every GPU column"
    var t_n = Int(ip[0])
    var fi = Int(ip[1])
    var a_n = Int(ip[2])
    var sw = Int(ip[3]) if len(ip) > 3 else 0
    var o = _ridge_off(d, t_n)
    var nb = fold_blocks(n)
    var cells = d * (d + 1) // 2
    var hfp = fp.copy()
    var dfp = ctx.enqueue_create_buffer[DType.float32](max(len(hfp), 1))
    var dfw = ctx.enqueue_create_buffer[DType.float32](o[5] + d * t_n)
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n_out, 1))
    var dstate = ctx.enqueue_create_buffer[DType.float32](RS_WORDS)
    var dtw = ctx.enqueue_create_buffer[DType.float32](team_work(0, 3, 0))
    var dwp = ctx.enqueue_create_buffer[DType.float32](max((d + t_n + 1) * nb, 1) if sw != 0 else 1)
    var dgp = ctx.enqueue_create_buffer[DType.float32](max((cells + d * t_n) * nb, 1) if sw != 0 else 1)
    var dtscr = ctx.enqueue_create_buffer[DType.float32](max(t_n * d, 1))
    var slice = max(1, min(n, (1 << 24) // max(d, 1)))
    var loo = a_n > 1
    var dscr = ctx.enqueue_create_buffer[DType.float32](max(slice * d, 1) if loo else 1)
    var dla = ctx.enqueue_create_buffer[DType.float32](max(n, 1) if loo else 1)
    var dlb = ctx.enqueue_create_buffer[DType.float32](max(n, 1) if loo else 1)
    var dep = ctx.enqueue_create_buffer[DType.float32](max(nb, 1) if loo else 1)
    ctx.enqueue_copy(dst_buf=dfp, src_ptr=hfp.unsafe_ptr())
    var fwp = dfw.unsafe_ptr()
    var stp = dstate.unsafe_ptr()
    var nf = i2f(n)
    var g_w = _rb((d + t_n + 1) * nb)
    var g_m = _rb(d + t_n)
    var g_g = _rb((cells + d * t_n) * nb)
    var g_f = _rb(cells + d * t_n)
    var nsl = (n + slice - 1) // slice
    var g_a = 1 + nsl * _rb(slice) + _rb(nb) + 1
    var wit = Witness(ctx, max(g_w + g_m + g_g + g_f, g_a) + 2)
    var tl = mg_tiles(d, t_n)
    var tries = 0
    while True:
        var nonce = wit.begin()
        var wo = 0
        dfw.enqueue_fill(Float32(0))
        dstate.enqueue_fill(Float32(0))
        dout.enqueue_fill(Float32(0))
        if sw != 0:
            ctx.enqueue_function[rw_parts_kernel](xp, yp, Int32(n), Int32(d), Int32(t_n), dwp.unsafe_ptr(),
                                                  wit.p(), Int32(wo), nonce, grid_dim=g_w, block_dim=RG_TPB)
            wo += g_w
            ctx.enqueue_function[rw_means_kernel](Int32(n), Int32(d), Int32(t_n), Int32(fi), dwp.unsafe_ptr(), fwp, stp,
                                                  wit.p(), Int32(wo), nonce, grid_dim=g_m, block_dim=RG_TPB)
            wo += g_m
            ctx.enqueue_function[rw_gram_parts_kernel](xp, yp, Int32(n), Int32(d), Int32(t_n), fwp, dgp.unsafe_ptr(),
                                                       wit.p(), Int32(wo), nonce, grid_dim=g_g, block_dim=RG_TPB)
            wo += g_g
            ctx.enqueue_function[rw_gram_fin_kernel](Int32(n), Int32(d), Int32(t_n), dgp.unsafe_ptr(), fwp,
                                                     wit.p(), Int32(wo), nonce, grid_dim=g_f, block_dim=RG_TPB)
            wo += g_f
        else:
            # the moments of [X | Y] (lane/neural-pass120): xm, G, ym, X'Y
            ctx.enqueue_function[mg_means_kernel](
                xp, yp, Int32(n), Int32(d), Int32(t_n), Int32(fi), fwp, Int32(o[0]), Int32(o[4]),
                grid_dim=tl, block_dim=MG_NT,
            )
            ctx.enqueue_function[mg_cross_kernel](
                xp, yp, Int32(n), Int32(d), Int32(t_n), fwp, Int32(o[0]), Int32(o[4]), Int32(o[1]), Int32(o[5]),
                grid_dim=tl * (tl + 1) // 2, block_dim=MG_NT,
            )
        if wit.ok(ctx, wo, "Ridge statistics"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    for a in range(a_n if loo else 0):
        # in place (the best so far): a cut raises
        var nonce = wit.begin()
        var wo = 0
        ctx.enqueue_function[ra_factor_kernel](Int32(d), hfp[a], fwp, stp, dtw.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                               grid_dim=1, block_dim=LINEAR_TPB)
        wo += 1
        var lo = 0
        while lo < n:
            var cnt = min(slice, n - lo)
            ctx.enqueue_function[ra_loo_kernel](xp, yp, Int32(n), Int32(d), Int32(sw), Int32(fi), Int32(lo), Int32(cnt), fwp,
                                                stp, dscr.unsafe_ptr(), dla.unsafe_ptr(), dlb.unsafe_ptr(),
                                                wit.p(), Int32(wo), nonce, grid_dim=_rb(cnt), block_dim=RG_TPB)
            wo += _rb(cnt)
            lo += cnt
        ctx.enqueue_function[ra_err_parts_kernel](Int32(n), dla.unsafe_ptr(), dlb.unsafe_ptr(), stp, dep.unsafe_ptr(),
                                                  wit.p(), Int32(wo), nonce, grid_dim=_rb(nb), block_dim=RG_TPB)
        wo += _rb(nb)
        ctx.enqueue_function[ra_err_fin_kernel](Int32(nb), nf, Int32(a), Int32(t_n * d + t_n + 2 + a), dep.unsafe_ptr(),
                                                stp, dout.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=1)
        wo += 1
        if not wit.ok(ctx, wo, "RidgeCV alpha"):
            wit.fail()
    var nb2 = wit.begin()
    ctx.enqueue_function[ra_best_kernel](Int32(d), Int32(t_n), Int32(fi), Int32(a_n), dfp.unsafe_ptr(), dout.unsafe_ptr(),
                                         fwp, stp, dtscr.unsafe_ptr(), dtw.unsafe_ptr(), wit.p(), Int32(0), nb2,
                                         grid_dim=1, block_dim=LINEAR_TPB)
    if not wit.ok(ctx, 1, "Ridge fit"):
        wit.fail()
    if n_out > 0:
        ctx.enqueue_copy(dst_ptr=res, src_buf=dout)
    ctx.synchronize()
    _ = hfp^
    _ = dfp^
    _ = dfw^
    _ = dout^
    _ = dstate^
    _ = dtw^
    _ = dwp^
    _ = dgp^
    _ = dtscr^
    _ = dscr^
    _ = dla^
    _ = dlb^
    _ = dep^
    _ = wit^
