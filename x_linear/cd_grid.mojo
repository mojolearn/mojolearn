# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LassoCV / ElasticNetCV on the whole device (lane/neural-pass109, 2026-10-02).

The team form (x_linear/cd.mojo `enetcv_fit`) ran the whole fit on ONE
block: every fold's Gram on 256 threads, every path's coordinate descent on
the lead thread, every held-out sum on the lead. bench_board 0.8.34: taxi
3.2 s on the M3 Ultra against sklearn's 0.23, istella 76 s against 5.5.
Here the fit is launches:
  1. the means (and the y mean, the training row count) of every fold and of
     the full data, one thread per value;
  2. their Gram, X'y and |yc|^2, one thread per value;
  3. the alpha grids from the full data's X'y, one thread;
  4. one block per (fold, l1_ratio) path, the coordinate descent on the
     block (`t_enet_gram_cd`), warm started along the alphas;
  5. one thread per (fold, l1_ratio, alpha) held-out squared-error sum;
  6. one block: the choice, then the refit on the full data from zero.
Every value is the host schedule's statements over the same rows in the
same order (cd.mojo `ecv_*`), so every word is the host's.
Steps 1, 2 and 5 stage rows in threadgroup memory (lane/neural-pass110,
below) when the page fits; the per-value kernels otherwise. (The
`MOJOLEARN_X_LINEAR_ENETCV_GRID` / `_STAGED` A/B switches were deleted,
cpu-gpu-cleanup c-linear.)
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.atomic import Atomic
from max.gpu.sync import barrier
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for
from max.gpu.host import DeviceContext
from x_linear.finite_device import XLIN_IDN_DEV_FINITE, xlin_finite_device
from x_linear.witness import Witness, witness_end, WITNESS_TRIES
from x_linear.ops import FP, IP, fa, fm, fs, fd, fmad, ld, st, ldi, i2f, fill
from x_linear.team import Team, TEAM_SLOTS, LINEAR_TPB, team_at
from x_linear.tops import upper_cell, _acc_fa, _acc_fmad
from experiments.classical_identical_ideas.linear_controls import ENETCV_FOLD_BLOCKS
from x_linear.enetcv_blocks import (
    FB_C, FB_NT, FB_TPB, FB_TC, fb_words, fb_means_kernel, fb_fold_means_kernel, fb_gram_kernel,
    fb_fold_gram_kernel, fb_combine_kernel,
)
from x_linear.cd import (
    ecv_alphas, ecv_alpha_cell, ecv_choose, ecv_finish, ecv_rows, ecv_fold_fa, ecv_cfmad, ecv_held_sse, ECV_UH,
    t_enet_gram_cd,
)

comptime ECV_TPB = 64


struct _EcvLayout(ImplicitlyCopyable, Movable):
    """Words of the fit's work buffer: P = F + 1 preps (prep F the full
    data), each xm d | G d*d | q d | (y mean, |yc|^2, rows); then F*L + 1
    paths (the last the refit), each Qw d | w d | path A*(d + 1)."""
    var d: Int
    var a_n: Int
    var f_n: Int
    var l_n: Int
    var ps: Int
    var rs: Int

    def __init__(out self, d: Int, a_n: Int, f_n: Int, l_n: Int):
        self.d = d
        self.a_n = a_n
        self.f_n = f_n
        self.l_n = l_n
        self.ps = 2 * d + d * d + 3
        self.rs = 2 * d + a_n * (d + 1)

    @always_inline
    def prep(self, p: Int) -> Int:
        return p * self.ps

    @always_inline
    def path(self, r: Int) -> Int:
        return (self.f_n + 1) * self.ps + r * self.rs

    def words(self) -> Int:
        return self.path(self.f_n * self.l_n + 1)


@always_inline
def _fold_of(p: Int, f_n: Int) -> Int:
    return -1 if p == f_n else p


@always_inline
def _ecv_means_kernel_body(x: FP, y: FP, n: Int32, d: Int32, ip: IP, ew: FP):
    """Thread (p, j): column j's mean of prep p (j == d: the y mean and the
    training row count)."""
    var dd = Int(d)
    var nn = Int(n)
    var lay = _EcvLayout(dd, ldi(ip, 2), ldi(ip, 3), ldi(ip, 4))
    var g = Int(block_idx.x) * ECV_TPB + Int(thread_idx.x)
    if g >= (lay.f_n + 1) * (dd + 1):
        return
    var p = g // (dd + 1)
    var j = g % (dd + 1)
    var fold = _fold_of(p, lay.f_n)
    var fi = ldi(ip, 1) != 0
    var fid = y + nn
    var base = lay.prep(p)
    var rows = ecv_rows(fid, nn, fold)
    if j < dd:
        st(ew, base + j, fd(ecv_fold_fa(x, j, dd, fid, nn, fold), i2f(rows)) if fi else Float32(0))
    else:
        var sc = base + 2 * dd + dd * dd
        st(ew, sc, fd(ecv_fold_fa(y, 0, 1, fid, nn, fold), i2f(rows)) if fi else Float32(0))
        st(ew, sc + 2, i2f(Int(rows)))


def ecv_means_kernel(x: FP, y: FP, n: Int32, d: Int32, ip: IP, ew: FP, wf: IP, woff: Int32, nonce: Int32):
    _ecv_means_kernel_body(x, y, n, d, ip, ew)
    witness_end(wf, woff, nonce)

@always_inline
def _ecv_gram_kernel_body(x: FP, y: FP, n: Int32, d: Int32, ip: IP, ew: FP):
    """Thread (p, c): Gram cell c (upper triangle, mirrored), X'y entry or
    |yc|^2 of prep p, from the means of `ecv_means_kernel`."""
    var dd = Int(d)
    var nn = Int(n)
    var lay = _EcvLayout(dd, ldi(ip, 2), ldi(ip, 3), ldi(ip, 4))
    var cells = dd * (dd + 1) // 2
    var per = cells + dd + 1
    var g = Int(block_idx.x) * ECV_TPB + Int(thread_idx.x)
    if g >= (lay.f_n + 1) * per:
        return
    var p = g // per
    var c = g % per
    var fold = _fold_of(p, lay.f_n)
    var fid = y + nn
    var base = lay.prep(p)
    var gg = base + dd
    var q = gg + dd * dd
    var sc = q + dd
    var ym = ld(ew, sc)
    if c < cells:
        var jk = upper_cell(c, dd)
        var j = jk[0]
        var k = jk[1]
        var acc = ecv_cfmad(x, j, dd, ld(ew, base + j), x, k, dd, ld(ew, base + k), fid, nn, fold)
        st(ew, gg + j * dd + k, acc)
        st(ew, gg + k * dd + j, acc)
    elif c < cells + dd:
        var j = c - cells
        st(ew, q + j, ecv_cfmad(x, j, dd, ld(ew, base + j), y, 0, 1, ym, fid, nn, fold))
    else:
        st(ew, sc + 1, ecv_cfmad(y, 0, 1, ym, y, 0, 1, ym, fid, nn, fold))


def ecv_gram_kernel(x: FP, y: FP, n: Int32, d: Int32, ip: IP, ew: FP, wf: IP, woff: Int32, nonce: Int32):
    _ecv_gram_kernel_body(x, y, n, d, ip, ew)
    witness_end(wf, woff, nonce)

@always_inline
def _ecv_grid_kernel_body(n: Int32, d: Int32, ip: IP, fp: FP, res: FP, ew: FP):
    """Thread (l, k): cell (l, k) of the alpha grids from the full data's
    X'y (`ecv_alpha_cell`, the host's statements, one cell per thread)."""
    var dd = Int(d)
    var lay = _EcvLayout(dd, ldi(ip, 2), ldi(ip, 3), ldi(ip, 4))
    var g = Int(block_idx.x) * ECV_TPB + Int(thread_idx.x)
    if g >= lay.l_n * lay.a_n:
        return
    var q = lay.prep(lay.f_n) + dd + dd * dd
    ecv_alpha_cell(res, dd + 4, fp, lay.l_n, lay.a_n, ldi(ip, 5) != 0, ld(fp, 0), ew, q, dd, Int(n),
                   g // lay.a_n, g % lay.a_n)


def ecv_grid_kernel(n: Int32, d: Int32, ip: IP, fp: FP, res: FP, ew: FP, wf: IP, woff: Int32, nonce: Int32):
    _ecv_grid_kernel_body(n, d, ip, fp, res, ew)
    witness_end(wf, woff, nonce)

@always_inline
def _ecv_path_kernel_body(d: Int32, ip: IP, fp: FP, res: FP, ew: FP, tw: FP):
    """Block r = f * L + l: the warm-started path of fold f at l1_ratio l,
    each point's (coef, intercept) into the path's words."""
    var dd = Int(d)
    var lay = _EcvLayout(dd, ldi(ip, 2), ldi(ip, 3), ldi(ip, 4))
    var r = Int(block_idx.x)
    var f = r // lay.l_n
    var l = r % lay.l_n
    var t = team_at(Int(thread_idx.x), Int(block_dim.x), tw + r * TEAM_SLOTS, 0, 0, 0)
    var base = lay.prep(f)
    var gg = base + dd
    var q = gg + dd * dd
    var sc = q + dd
    var qw = lay.path(r)
    var w = qw + dd
    var pb = w + dd
    var ym = ld(ew, sc)
    var yn = ld(ew, sc + 1)
    var rows = ld(ew, sc + 2)
    var l1r = ld(fp, 2 + l)
    for j in range(t.tid, dd, t.nt):
        st(ew, w + j, Float32(0))
    t.sync()
    for k in range(lay.a_n):
        var alpha = ld(res, dd + 4 + l * lay.a_n + k)
        var l1 = fm(fm(alpha, l1r), rows)
        var l2 = fm(fm(alpha, fs(Float32(1), l1r)), rows)
        _ = t_enet_gram_cd(t, ew, gg, q, qw, w, dd, yn, l1, l2, ldi(ip, 0), ld(fp, 1), ldi(ip, 6) != 0)
        if t.lead():
            var b = ym
            for j in range(dd):
                b = fs(b, fm(ld(ew, base + j), ld(ew, w + j)))
            for j in range(dd):
                st(ew, pb + k * (dd + 1) + j, ld(ew, w + j))
            st(ew, pb + k * (dd + 1) + dd, b)
        t.sync()


def ecv_path_kernel(d: Int32, ip: IP, fp: FP, res: FP, ew: FP, tw: FP, wf: IP, woff: Int32, nonce: Int32):
    _ecv_path_kernel_body(d, ip, fp, res, ew, tw)
    witness_end(wf, woff, nonce)

@always_inline
def _ecv_score_kernel_body(x: FP, y: FP, n: Int32, d: Int32, ip: IP, res: FP, ew: FP):
    """Thread (f, l, k): the held-out mean squared error of that path point
    into res's mse words."""
    var dd = Int(d)
    var nn = Int(n)
    var lay = _EcvLayout(dd, ldi(ip, 2), ldi(ip, 3), ldi(ip, 4))
    var g = Int(block_idx.x) * ECV_TPB + Int(thread_idx.x)
    if g >= lay.f_n * lay.l_n * lay.a_n:
        return
    var k = g % lay.a_n
    var r = g // lay.a_n
    var f = r // lay.l_n
    var l = r % lay.l_n
    var n_te = nn - Int(ld(ew, lay.prep(f) + 2 * dd + dd * dd + 2))
    var o = lay.path(r) + 2 * dd + k * (dd + 1)
    var acc = ecv_held_sse(x, y, y + nn, nn, dd, f, ew, o)
    var mse = dd + 4 + lay.l_n * lay.a_n
    st(res, mse + (l * lay.a_n + k) * lay.f_n + f, fd(acc, i2f(n_te)) if n_te > 0 else Float32(0))


def ecv_score_kernel(x: FP, y: FP, n: Int32, d: Int32, ip: IP, res: FP, ew: FP, wf: IP, woff: Int32, nonce: Int32):
    _ecv_score_kernel_body(x, y, n, d, ip, res, ew)
    witness_end(wf, woff, nonce)

# Held-out rows a score block stages: SCORE_WORDS floats of rows (as many
# rows as fit, at most ECV_SR), their y and fold ids.
comptime ECV_SR = 256
comptime ECV_SW = 6144
comptime ECV_SNT = 128
comptime ECV_SCORE_BYTES = (ECV_SW + 2 * ECV_SR) * 4
comptime ECV_SCORE_STAGED = lib_smem_page_fits_for[TARGET_COLUMN, ECV_SCORE_BYTES]()


@always_inline
def _ecv_score_staged_kernel_body(x: FP, y: FP, n: Int32, d: Int32, ip: IP, res: FP, ew: FP, spans: IP):
    """Block r = f * L + l: the held-out mean squared errors of the path of
    (fold f, l1_ratio l). The fold's held-out rows lie in
    [spans[2f], spans[2f] + spans[2f + 1]); tiles of them are staged in
    threadgroup memory, and thread k (alpha k, then k + ECV_SNT, ...) runs
    `ecv_held_sse`'s statements on them: ECV_UH rows' predictions side by
    side, folded in row order. The same words."""
    var dd = Int(d)
    var nn = Int(n)
    var lay = _EcvLayout(dd, ldi(ip, 2), ldi(ip, 3), ldi(ip, 4))
    var r = Int(block_idx.x)
    var f = r // lay.l_n
    var l = r % lay.l_n
    var tid = Int(thread_idx.x)
    var lo = Int(spans.unsafe_load(2 * f))
    var span = Int(spans.unsafe_load(2 * f + 1))
    var fv = Float32(f)
    var tr = max(1, min(ECV_SR, ECV_SW // max(dd, 1)))
    var xs = stack_allocation[ECV_SW, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ys = stack_allocation[ECV_SR, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var fs_ = stack_allocation[ECV_SR, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var n_te = nn - Int(ld(ew, lay.prep(f) + 2 * dd + dd * dd + 2))
    var mse = dd + 4 + lay.l_n * lay.a_n
    var kb = 0
    while kb < lay.a_n:
        var k = kb + tid
        var live = k < lay.a_n
        var o = lay.path(r) + 2 * dd + min(k, lay.a_n - 1) * (dd + 1)
        var b0 = ld(ew, o + dd)
        var acc = Float32(0)
        var r0 = 0
        while r0 < span:
            var cnt = min(tr, span - r0)
            barrier()
            var base = (lo + r0) * dd
            for u in range(tid, cnt * dd, ECV_SNT):
                xs[u] = ld(x, base + u)
            for u in range(tid, cnt, ECV_SNT):
                ys[u] = ld(y, lo + r0 + u)
                fs_[u] = ld(y, nn + lo + r0 + u)
            barrier()
            if live:
                var q = 0
                while q + ECV_UH <= cnt:
                    var pv = SIMD[DType.float32, ECV_UH](b0)
                    for j in range(dd):
                        var wj = ld(ew, o + j)
                        comptime for u in range(ECV_UH):
                            pv[u] = fmad(xs[(q + u) * dd + j], wj, pv[u])
                    comptime for u in range(ECV_UH):
                        if fs_[q + u] == fv:
                            var e = fs(pv[u], ys[q + u])
                            acc = _acc_fmad(e, e, acc)
                    q += ECV_UH
                while q < cnt:
                    if fs_[q] == fv:
                        var pp = b0
                        for j in range(dd):
                            pp = fmad(xs[q * dd + j], ld(ew, o + j), pp)
                        var e = fs(pp, ys[q])
                        acc = _acc_fmad(e, e, acc)
                    q += 1
            r0 += cnt
        if live:
            st(res, mse + (l * lay.a_n + k) * lay.f_n + f, fd(acc, i2f(n_te)) if n_te > 0 else Float32(0))
        kb += ECV_SNT


def ecv_score_staged_kernel(x: FP, y: FP, n: Int32, d: Int32, ip: IP, res: FP, ew: FP, spans: IP, wf: IP, woff: Int32, nonce: Int32):
    _ecv_score_staged_kernel_body(x, y, n, d, ip, res, ew, spans)
    witness_end(wf, woff, nonce)

@always_inline
def _ecv_refit_kernel_body(d: Int32, ip: IP, fp: FP, res: FP, ew: FP, tw: FP):
    """One block team: the choice, then the refit on the full data from
    zero. Its work is the d x d Gram's coordinate descent (one problem);
    the rows were folded by the means and Gram kernels, and the row count
    is the full prep's (the word the means kernel wrote, as the path kernel
    reads its fold's)."""
    var dd = Int(d)
    var lay = _EcvLayout(dd, ldi(ip, 2), ldi(ip, 3), ldi(ip, 4))
    var r = lay.f_n * lay.l_n
    var t = team_at(Int(thread_idx.x), Int(block_dim.x), tw + r * TEAM_SLOTS, 0, 0, 0)
    if t.lead():
        ecv_choose(res, fp, dd, lay.l_n, lay.a_n, lay.f_n)
    var base = lay.prep(lay.f_n)
    var gg = base + dd
    var q = gg + dd * dd
    var sc = q + dd
    var qw = lay.path(r)
    var w = qw + dd
    for j in range(t.tid, dd, t.nt):
        st(ew, w + j, Float32(0))
    t.sync()
    var alpha = ld(res, dd + 1)
    var l1r = ld(res, dd + 2)
    var nf = ld(ew, sc + 2)
    var iters = t_enet_gram_cd(t, ew, gg, q, qw, w, dd, ld(ew, sc + 1), fm(fm(alpha, l1r), nf),
                               fm(fm(alpha, fs(Float32(1), l1r)), nf), ldi(ip, 0), ld(fp, 1), ldi(ip, 6) != 0)
    if t.lead():
        ecv_finish(res, ew, base, w, sc, dd, ldi(ip, 1) != 0, alpha, l1r, iters)


def ecv_refit_kernel(d: Int32, ip: IP, fp: FP, res: FP, ew: FP, tw: FP, wf: IP, woff: Int32, nonce: Int32):
    _ecv_refit_kernel_body(d, ip, fp, res, ew, tw)
    witness_end(wf, woff, nonce)

# ------------------------------------------------ staged preps (lane/neural-pass110)
# The means and the Gram as above were one thread per value folding a
# million rows from DEVICE memory, each load a stall (M4 taxi 1M: 0.42 s
# and 0.61 s of the 1.4 s fit). Here a block stages ECV_TR rows of a
# 16-column tile (two tiles for the Gram) in threadgroup memory, loading
# the next tile into registers while the threads fold this one: a means
# thread folds one (column, prep), a Gram block one prep's cells of a tile
# pair, the rows ascending: the same statements on the same rows in the
# same order, so the same words. y is column d of the Gram tiles (its
# X'y cells and |yc|^2). Up to 16 preps (cv <= 15); more folds, or a
# column whose page does not fit, run the per-value kernels above.
comptime ECV_TR = 128
comptime ECV_TC = 16
comptime ECV_NT = 256
# Rows a fold reads from threadgroup memory before folding them (scheduling only).
comptime ECV_RU = 16
comptime ECV_RUS = 16
comptime ECV_STAGE_BYTES = (ECV_TR * 2 * ECV_TC + ECV_TR) * 4
comptime ECV_STAGED = lib_smem_page_fits_for[TARGET_COLUMN, ECV_STAGE_BYTES]()
# Words of a tile each thread loads: the means stage ECV_TC columns, the
# Gram 2 * ECV_TC; ECV_NT threads, ECV_TR rows.
comptime ECV_LM = ECV_TR * ECV_TC // ECV_NT
comptime ECV_LG = ECV_TR * 2 * ECV_TC // ECV_NT


@always_inline
def _aug(x: FP, y: FP, n: Int, d: Int, row: Int, col: Int) -> Float32:
    """[X | y] at (row, col); 0 past the edges. Both loads are issued from
    clamped in-range addresses and the word selected, so a tile's loads
    carry no branch."""
    var rr = min(row, n - 1)
    var xv = ld(x, rr * d + min(col, d - 1))
    var yv = ld(y, rr)
    var v = yv if col == d else xv
    return v if (row < n and col <= d) else Float32(0)


@always_inline
def _aug_mean(ew: FP, lay: _EcvLayout, p: Int, col: Int) -> Float32:
    var d = lay.d
    if col > d:
        return Float32(0)
    if col == d:
        return ld(ew, lay.prep(p) + 2 * d + d * d)
    return ld(ew, lay.prep(p) + col)


@always_inline
def _fid_at(fid: FP, n: Int, row: Int) -> Float32:
    var v = ld(fid, min(row, n - 1))
    return v if row < n else Float32(-1)


@always_inline
def _ecv_means_staged_kernel_body(x: FP, y: FP, n: Int32, d: Int32, ip: IP, ew: FP):
    """Block b: columns [16b, 16b + 16) of [X | y]; thread (p, c) folds
    column 16b + c over prep p's training rows (`ecv_means_kernel`'s
    values). ECV_NT threads; the next tile's words are loaded into
    registers while this tile is folded from threadgroup memory."""
    var dd = Int(d)
    var nn = Int(n)
    var lay = _EcvLayout(dd, ldi(ip, 2), ldi(ip, 3), ldi(ip, 4))
    var pn = lay.f_n + 1
    var tid = Int(thread_idx.x)
    var c = tid % ECV_TC
    var p = tid // ECV_TC
    var c0 = Int(block_idx.x) * ECV_TC
    var col = c0 + c
    var fid = y + nn
    var live = p < pn and col <= dd
    var xs = stack_allocation[ECV_TR * ECV_TC, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var fs_ = stack_allocation[ECV_TR, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var reg = SIMD[DType.float32, ECV_LM]()
    var rf = Float32(-1)
    comptime for v in range(ECV_LM):
        var u = v * ECV_NT + tid
        reg[v] = _aug(x, y, nn, dd, u // ECV_TC, c0 + u % ECV_TC)
    if tid < ECV_TR:
        rf = _fid_at(fid, nn, tid)
    var acc = Float32(0)
    var rows = Int32(0)
    # fold ids are small integers stored as float32: comparing the words is
    # comparing the ids (the host's Int(fid) != fold)
    var fpv = Float32(p)
    var full = p == lay.f_n
    var r0 = 0
    while r0 < nn:
        barrier()
        comptime for v in range(ECV_LM):
            xs[v * ECV_NT + tid] = reg[v]
        if tid < ECV_TR:
            fs_[tid] = rf
        barrier()
        var r1 = r0 + ECV_TR
        if r1 < nn:
            comptime for v in range(ECV_LM):
                var u = v * ECV_NT + tid
                reg[v] = _aug(x, y, nn, dd, r1 + u // ECV_TC, c0 + u % ECV_TC)
            if tid < ECV_TR:
                rf = _fid_at(fid, nn, r1 + tid)
        if live:
            var cnt = Int32(min(ECV_TR, nn - r0))
            var px = xs + c
            var pf = fs_
            var r = Int32(0)
            while r + ECV_RU <= cnt:
                var bv = SIMD[DType.float32, ECV_RU]()
                var bf = SIMD[DType.float32, ECV_RU]()
                comptime for u in range(ECV_RU):
                    bv[u] = px[u * ECV_TC]
                    bf[u] = pf[u]
                comptime for u in range(ECV_RU):
                    if full or bf[u] != fpv:
                        acc = _acc_fa(acc, bv[u])
                        rows += 1
                px += ECV_RU * ECV_TC
                pf += ECV_RU
                r += ECV_RU
            while r < cnt:
                if full or pf[0] != fpv:
                    acc = _acc_fa(acc, px[0])
                    rows += 1
                px += ECV_TC
                pf += 1
                r += 1
        r0 = r1
    if live:
        var fi = ldi(ip, 1) != 0
        var base = lay.prep(p)
        if col < dd:
            st(ew, base + col, fd(acc, i2f(Int(rows))) if fi else Float32(0))
        else:
            var sc = base + 2 * dd + dd * dd
            st(ew, sc, fd(acc, i2f(Int(rows))) if fi else Float32(0))
            st(ew, sc + 2, i2f(Int(rows)))


def ecv_means_staged_kernel(x: FP, y: FP, n: Int32, d: Int32, ip: IP, ew: FP, wf: IP, woff: Int32, nonce: Int32):
    _ecv_means_staged_kernel_body(x, y, n, d, ip, ew)
    witness_end(wf, woff, nonce)

@always_inline
def _ecv_gram_staged_kernel_body(x: FP, y: FP, n: Int32, d: Int32, ip: IP, ew: FP):
    """Block b = pair * P + p: prep p's cells of the tile pair (J, K) =
    upper_cell(pair, tiles) of [X | y]; thread (a, b) the cell
    (16J + a, 16K + b) when it is on or above the diagonal
    (`ecv_gram_kernel`'s values). ECV_NT threads; the next tile is loaded
    while this one is folded."""
    var dd = Int(d)
    var nn = Int(n)
    var aug = dd + 1
    var lay = _EcvLayout(dd, ldi(ip, 2), ldi(ip, 3), ldi(ip, 4))
    var pn = lay.f_n + 1
    var tiles = (aug + ECV_TC - 1) // ECV_TC
    var p = Int(block_idx.x) % pn
    var jk = upper_cell(Int(block_idx.x) // pn, tiles)
    var tid = Int(thread_idx.x)
    var j = jk[0] * ECV_TC + tid // ECV_TC
    var k = jk[1] * ECV_TC + tid % ECV_TC
    var live = j <= k and k < aug
    var fid = y + nn
    var xs = stack_allocation[ECV_TR * 2 * ECV_TC, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var fs_ = stack_allocation[ECV_TR, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var mj = Float32(0)
    var mk = Float32(0)
    if live:
        mj = _aug_mean(ew, lay, p, j)
        mk = _aug_mean(ew, lay, p, k)
    var fpv = Float32(p)
    var full = p == lay.f_n
    var acc = Float32(0)
    var a = tid // ECV_TC
    var b = ECV_TC + tid % ECV_TC
    var cj = jk[0] * ECV_TC
    var ck = jk[1] * ECV_TC
    var reg = SIMD[DType.float32, ECV_LG]()
    var rf = Float32(-1)
    comptime for v in range(ECV_LG):
        var u = v * ECV_NT + tid
        var cc = u % (2 * ECV_TC)
        reg[v] = _aug(x, y, nn, dd, u // (2 * ECV_TC), cj + cc if cc < ECV_TC else ck + cc - ECV_TC)
    if tid < ECV_TR:
        rf = _fid_at(fid, nn, tid)
    var r0 = 0
    while r0 < nn:
        barrier()
        comptime for v in range(ECV_LG):
            xs[v * ECV_NT + tid] = reg[v]
        if tid < ECV_TR:
            fs_[tid] = rf
        barrier()
        var r1 = r0 + ECV_TR
        if r1 < nn:
            comptime for v in range(ECV_LG):
                var u = v * ECV_NT + tid
                var cc = u % (2 * ECV_TC)
                reg[v] = _aug(x, y, nn, dd, r1 + u // (2 * ECV_TC), cj + cc if cc < ECV_TC else ck + cc - ECV_TC)
            if tid < ECV_TR:
                rf = _fid_at(fid, nn, r1 + tid)
        if live:
            var cnt = Int32(min(ECV_TR, nn - r0))
            var pj = xs + a
            var pk = xs + b
            var pf = fs_
            var r = Int32(0)
            while r + ECV_RU <= cnt:
                var bj = SIMD[DType.float32, ECV_RU]()
                var bk = SIMD[DType.float32, ECV_RU]()
                var bf = SIMD[DType.float32, ECV_RU]()
                comptime for u in range(ECV_RU):
                    bj[u] = pj[u * 2 * ECV_TC]
                    bk[u] = pk[u * 2 * ECV_TC]
                    bf[u] = pf[u]
                comptime for u in range(ECV_RU):
                    if full or bf[u] != fpv:
                        acc = _acc_fmad(fs(bj[u], mj), fs(bk[u], mk), acc)
                pj += ECV_RU * 2 * ECV_TC
                pk += ECV_RU * 2 * ECV_TC
                pf += ECV_RU
                r += ECV_RU
            while r < cnt:
                if full or pf[0] != fpv:
                    acc = _acc_fmad(fs(pj[0], mj), fs(pk[0], mk), acc)
                pj += 2 * ECV_TC
                pk += 2 * ECV_TC
                pf += 1
                r += 1
        r0 = r1
    if live:
        var base = lay.prep(p)
        var gg = base + dd
        var q = gg + dd * dd
        if k < dd:
            st(ew, gg + j * dd + k, acc)
            st(ew, gg + k * dd + j, acc)
        elif j < dd:
            st(ew, q + j, acc)
        else:
            st(ew, q + dd + 1, acc)


def ecv_gram_staged_kernel(x: FP, y: FP, n: Int32, d: Int32, ip: IP, ew: FP, wf: IP, woff: Int32, nonce: Int32):
    _ecv_gram_staged_kernel_body(x, y, n, d, ip, ew)
    witness_end(wf, woff, nonce)

#: the witness words of one fit: every launch's blocks (x_linear/witness.mojo)
comptime ECV_WIT_CAP = 1 << 20


def _blocks(count: Int) -> Int:
    return max((count + ECV_TPB - 1) // ECV_TPB, 1)


def _fb_blocks(count: Int) -> Int:
    return max((count + FB_TPB - 1) // FB_TPB, 1)


def ecv_span_init_kernel(lohi: MutPointer[Int32, MutAnyOrigin], f_n_in: Int32, n_in: Int32):
    """One thread a fold: its span starts empty (lo = n, hi = 0)."""
    var f = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if f < Int(f_n_in):
        lohi.unsafe_store(2 * f, n_in)
        lohi.unsafe_store(2 * f + 1, Int32(0))


def ecv_span_rows_kernel(
    dy: MutPointer[Float32, MutAnyOrigin], n_in: Int32, f_n_in: Int32,
    lohi: MutPointer[Int32, MutAnyOrigin],
):
    """One thread a row: its fold id (y's second n words, truncated as the
    host loop did) widens that fold's [lo, hi) by integer atomics (exact,
    order-free)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    if i < n:
        var f = Int(dy.unsafe_load(n + i))
        if f >= 0 and f < Int(f_n_in):
            _ = Atomic.min(lohi.unsafe_offset(2 * f), Int32(i))
            _ = Atomic.max(lohi.unsafe_offset(2 * f + 1), Int32(i + 1))


def ecv_span_finish_kernel(
    lohi: MutPointer[Int32, MutAnyOrigin], f_n_in: Int32, span: MutPointer[Int32, MutAnyOrigin],
):
    """One thread a fold: (lo, hi - lo), or (0, 0) for a fold with no row."""
    var f = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if f < Int(f_n_in):
        var lo = lohi.unsafe_load(2 * f)
        var hi = lohi.unsafe_load(2 * f + 1)
        if hi > Int32(0):
            span.unsafe_store(2 * f, lo)
            span.unsafe_store(2 * f + 1, hi - lo)
        else:
            span.unsafe_store(2 * f, Int32(0))
            span.unsafe_store(2 * f + 1, Int32(0))


def enetcv_fit_grid(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, res: FP,
) raises:
    """`enetcv_fit` as the launches above (ip, fp, y, res as enetcv_fit's)."""
    var a_n = Int(ip[2])
    var f_n = Int(ip[3])
    var l_n = Int(ip[4])
    var lay = _EcvLayout(d, a_n, f_n, l_n)
    var paths = f_n * l_n
    var nt = min(LINEAR_TPB, max(32, (d + 31) // 32 * 32))
    var hip = ip.copy()
    var hfp = fp.copy()
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var dip = ctx.enqueue_create_buffer[DType.int32](max(len(hip), 1))
    var dfp = ctx.enqueue_create_buffer[DType.float32](max(len(hfp), 1))
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n_out, 1))
    var dew = ctx.enqueue_create_buffer[DType.float32](max(lay.words(), 1))
    var dtw = ctx.enqueue_create_buffer[DType.float32]((paths + 1) * TEAM_SLOTS)
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    comptime if XLIN_IDN_DEV_FINITE:
        xlin_finite_device(ctx, dx, n_x, y, n_y)
    ctx.enqueue_copy(dst_buf=dip, src_ptr=hip.unsafe_ptr())
    if len(hfp) > 0:
        ctx.enqueue_copy(dst_buf=dfp, src_ptr=hfp.unsafe_ptr())
    var staged = ECV_STAGED and f_n + 1 <= ECV_NT // ECV_TC
    var staged_score = ECV_SCORE_STAGED and n_y >= 2 * n
    # each fold's held-out rows lie in [lo, lo + span) (KFold: exactly
    # them; rows of other folds inside are skipped by their id)
    # (lane cpu3-core: the spans come from the resident fold ids on the
    # device, not from a host walk over the n rows)
    var sp_len = max(2 * f_n, 1)
    var dspan = ctx.enqueue_create_buffer[DType.int32](sp_len)
    ctx.enqueue_memset(dspan, Int32(0))
    # ENETCV_FOLD_BLOCKS (x_linear/enetcv_blocks.mojo): every fold's
    # statistics from fold-aligned compensated chunk partials, one read of
    # each row per pass, instead of F + 1 full scans
    var m = d + 1
    var fb_tiles = (m + FB_TC - 1) // FB_TC
    var fb_pairs = fb_tiles * (fb_tiles + 1) // 2
    var fb_ps = ctx.enqueue_create_buffer[DType.float32](max(f_n * FB_C * m, 1) if ENETCV_FOLD_BLOCKS else 1)
    var fb_pn = ctx.enqueue_create_buffer[DType.int32](max(f_n * FB_C, 1) if ENETCV_FOLD_BLOCKS else 1)
    var fb_pg = ctx.enqueue_create_buffer[DType.float32](max(f_n * FB_C * m * m, 1) if ENETCV_FOLD_BLOCKS else 1)
    var fb_st = ctx.enqueue_create_buffer[DType.float32](max(f_n * fb_words(m), 1) if ENETCV_FOLD_BLOCKS else 1)
    var fb_cn = ctx.enqueue_create_buffer[DType.int32](max(f_n, 1) if ENETCV_FOLD_BLOCKS else 1)
    if staged_score or ENETCV_FOLD_BLOCKS:
        var dlohi = ctx.enqueue_create_buffer[DType.int32](sp_len)
        ctx.enqueue_function[ecv_span_init_kernel](
            dlohi.unsafe_ptr(), Int32(f_n), Int32(n), grid_dim=_blocks(f_n), block_dim=ECV_TPB,
        )
        ctx.enqueue_function[ecv_span_rows_kernel](
            dy.unsafe_ptr(), Int32(n), Int32(f_n), dlohi.unsafe_ptr(), grid_dim=_blocks(n), block_dim=ECV_TPB,
        )
        ctx.enqueue_function[ecv_span_finish_kernel](
            dlohi.unsafe_ptr(), Int32(f_n), dspan.unsafe_ptr(), grid_dim=_blocks(f_n), block_dim=ECV_TPB,
        )
        _ = dlohi^
    var dsp = ctx.enqueue_create_buffer[DType.int32](sp_len)
    var ctx_w = ctx.copy()
    # the whole fit (no host step between its launches) as ONE guarded unit
    # from zeroed scratch: a cut launch reruns it (x_linear/witness.mojo)
    var wit = Witness(ctx_w, ECV_WIT_CAP)
    var tries = 0
    while True:
        var nonce = wit.begin()
        var wo = 0
        dout.enqueue_fill(Float32(0))
        dew.enqueue_fill(Float32(0))
        dtw.enqueue_fill(Float32(0))
        comptime if ENETCV_FOLD_BLOCKS:
            ctx.enqueue_function[fb_means_kernel](
                dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(f_n), dspan.unsafe_ptr(),
                fb_ps.unsafe_ptr(), fb_pn.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                grid_dim=_fb_blocks(f_n * FB_C * m), block_dim=FB_TPB,
            )
            wo += _fb_blocks(f_n * FB_C * m)
            ctx.enqueue_function[fb_fold_means_kernel](
                fb_ps.unsafe_ptr(), fb_pn.unsafe_ptr(), Int32(d), Int32(f_n), fb_st.unsafe_ptr(), fb_cn.unsafe_ptr(),
                wit.p(), Int32(wo), nonce, grid_dim=_fb_blocks(f_n * m), block_dim=FB_TPB,
            )
            wo += _fb_blocks(f_n * m)
            if f_n > 0:
                ctx.enqueue_function[fb_gram_kernel](
                    dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(f_n), dspan.unsafe_ptr(),
                    fb_st.unsafe_ptr(), fb_pg.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                    grid_dim=f_n * FB_C * fb_pairs, block_dim=FB_NT,
                )
                wo += f_n * FB_C * fb_pairs
            ctx.enqueue_function[fb_fold_gram_kernel](
                fb_pg.unsafe_ptr(), Int32(d), Int32(f_n), fb_st.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                grid_dim=_fb_blocks(f_n * m * m), block_dim=FB_TPB,
            )
            wo += _fb_blocks(f_n * m * m)
            ctx.enqueue_function[fb_combine_kernel](
                fb_st.unsafe_ptr(), fb_cn.unsafe_ptr(), Int32(d), Int32(f_n), ip[1], Int32(lay.ps), dew.unsafe_ptr(),
                wit.p(), Int32(wo), nonce, grid_dim=_fb_blocks((f_n + 1) * m * m), block_dim=FB_TPB,
            )
            wo += _fb_blocks((f_n + 1) * m * m)
        else:
            if staged:
                var tiles = (d + 1 + ECV_TC - 1) // ECV_TC
                ctx.enqueue_function[ecv_means_staged_kernel](
                    dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), dip.unsafe_ptr(), dew.unsafe_ptr(),
                    wit.p(), Int32(wo), nonce, grid_dim=tiles, block_dim=ECV_NT,
                )
                wo += tiles
                ctx.enqueue_function[ecv_gram_staged_kernel](
                    dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), dip.unsafe_ptr(), dew.unsafe_ptr(),
                    wit.p(), Int32(wo), nonce, grid_dim=tiles * (tiles + 1) // 2 * (f_n + 1), block_dim=ECV_NT,
                )
                wo += tiles * (tiles + 1) // 2 * (f_n + 1)
            else:
                ctx.enqueue_function[ecv_means_kernel](
                    dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), dip.unsafe_ptr(), dew.unsafe_ptr(),
                    wit.p(), Int32(wo), nonce, grid_dim=_blocks((f_n + 1) * (d + 1)), block_dim=ECV_TPB,
                )
                wo += _blocks((f_n + 1) * (d + 1))
                ctx.enqueue_function[ecv_gram_kernel](
                    dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), dip.unsafe_ptr(), dew.unsafe_ptr(),
                    wit.p(), Int32(wo), nonce, grid_dim=_blocks((f_n + 1) * (d * (d + 1) // 2 + d + 1)), block_dim=ECV_TPB,
                )
                wo += _blocks((f_n + 1) * (d * (d + 1) // 2 + d + 1))
        ctx.enqueue_function[ecv_grid_kernel](
            Int32(n), Int32(d), dip.unsafe_ptr(), dfp.unsafe_ptr(), dout.unsafe_ptr(), dew.unsafe_ptr(),
            wit.p(), Int32(wo), nonce, grid_dim=_blocks(l_n * a_n), block_dim=ECV_TPB,
        )
        wo += _blocks(l_n * a_n)
        if paths > 0:
            ctx.enqueue_function[ecv_path_kernel](
                Int32(d), dip.unsafe_ptr(), dfp.unsafe_ptr(), dout.unsafe_ptr(), dew.unsafe_ptr(), dtw.unsafe_ptr(),
                wit.p(), Int32(wo), nonce, grid_dim=paths, block_dim=nt,
            )
            wo += paths
            if staged_score:
                ctx.enqueue_copy(dst_buf=dsp, src_buf=dspan)
                ctx.enqueue_function[ecv_score_staged_kernel](
                    dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), dip.unsafe_ptr(), dout.unsafe_ptr(),
                    dew.unsafe_ptr(), dsp.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=paths, block_dim=ECV_SNT,
                )
                wo += paths
            else:
                ctx.enqueue_function[ecv_score_kernel](
                    dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), dip.unsafe_ptr(), dout.unsafe_ptr(),
                    dew.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=_blocks(paths * a_n), block_dim=ECV_TPB,
                )
                wo += _blocks(paths * a_n)
        ctx.enqueue_function[ecv_refit_kernel](
            Int32(d), dip.unsafe_ptr(), dfp.unsafe_ptr(), dout.unsafe_ptr(), dew.unsafe_ptr(),
            dtw.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=nt,
        )
        wo += 1
        if n_out > 0:
            ctx.enqueue_copy(dst_ptr=res, src_buf=dout)
        if wo > ECV_WIT_CAP:
            raise Error("ElasticNetCV: witness capacity")
        if wit.ok(ctx_w, wo, "ElasticNetCV fit"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    ctx.synchronize()
    _ = hip^
    _ = hfp^
    _ = dx^
    _ = dy^
    _ = dip^
    _ = dfp^
    _ = dout^
    _ = dew^
    _ = dtw^
    _ = dsp^
    _ = dspan^
    _ = fb_ps^
    _ = fb_pn^
    _ = fb_pg^
    _ = fb_st^
    _ = fb_cn^
    _ = wit^
