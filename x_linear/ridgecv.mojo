# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RidgeCV with cv = k (lane/neural-pass91, 2026-10-01): the mean held-out
R^2 of every alpha over scikit-learn's KFold(k) (contiguous folds, the first
n % k one row longer, no shuffle), the scores their GridSearchCV ranks
(scoring None: Ridge.score, r2_score). The Python side takes the first best
alpha and refits Ridge on every row (their refit=True).

Per fold f, every value is one chain in a fixed order (the means and the
Gram / X'y cells compensated, `_two_sum`, since lane/gap-board-refusals), the same on the host
and on every device: the training means (rows ascending, the fold's rows
skipped), the centered Gram and X'y cells (fmad of the centered words, rows
ascending), the solve (G + alpha I by `cholesky` / `chol_solve`), the
held-out predictions (`row_dot` plus the intercept ym - xm.w) and the R^2
folds over the held-out rows ascending; the fold scores are summed folds
ascending and divided by k. Without an intercept nothing is centered.
"""
from x_linear.ops import FP, IP, fa, fs, fm, fd, fmad, ld, st, ldi, i2f, fill, cholesky, chol_solve, row_dot, seq_rows
from x_linear.ridge import chol_trusted, ridge_ff_unit, ridge_ff_units, ridge_ff_solve
from x_linear.tops import FOLD_BLOCK


@always_inline
def kf_start(n: Int, k: Int, f: Int) -> Int:
    return f * (n // k) + min(f, n % k)


@always_inline
def kf_end(n: Int, k: Int, f: Int) -> Int:
    return kf_start(n, k, f) + n // k + (1 if f < n % k else 0)


@always_inline
def _two_sum(mut s: Float32, mut c: Float32, v: Float32):
    """s += v with the rounding error carried in c (Knuth's TwoSum, every
    step one rounded fa/fs, branch-free): the k-fold chains run over up to
    n rows, and a plain float32 chain loses ~n*eps of the cell (Istella-S,
    800,000 training rows: 5e-4 of a Gram diagonal, hundreds of units, so
    G + alpha I was indefinite in float32 for every alpha). The compensated
    chain keeps the cell to a few rounding errors."""
    var t = fa(s, v)
    var bp = fs(t, s)
    var ap = fs(t, bp)
    c = fa(c, fa(fs(s, ap), fs(v, bp)))
    s = t


def kf_mean(v: FP, step: Int, off: Int, n: Int, s: Int, e: Int) -> Float32:
    """The mean of v[off + i*step] over the rows outside [s, e), ascending
    (the compensated chain `_two_sum`)."""
    var acc = Float32(0)
    var cc = Float32(0)
    for i in range(s):
        _two_sum(acc, cc, ld(v, off + i * step))
    for i in range(e, n):
        _two_sum(acc, cc, ld(v, off + i * step))
    return fd(fa(acc, cc), i2f(n - (e - s)))


def kf_cross(a: FP, astep: Int, aoff: Int, ma: Float32, b: FP, bstep: Int, boff: Int, mb: Float32,
             n: Int, s: Int, e: Int) -> Float32:
    """The sum of (a_i - ma)(b_i - mb) over the rows outside [s, e),
    ascending: each product rounded once (fm), summed by the compensated
    chain `_two_sum`."""
    var acc = Float32(0)
    var cc = Float32(0)
    for i in range(s):
        _two_sum(acc, cc, fm(fs(ld(a, aoff + i * astep), ma), fs(ld(b, boff + i * bstep), mb)))
    for i in range(e, n):
        _two_sum(acc, cc, fm(fs(ld(a, aoff + i * astep), ma), fs(ld(b, boff + i * bstep), mb)))
    return fa(acc, cc)


def kf_solve(g: FP, xty: FP, xm: FP, ym: Float32, d: Int, alpha: Float32, fi: Bool, aw: FP, w: FP) -> Tuple[Float32, Bool]:
    """w = (G + alpha I)^-1 X'y (aw: d*d scratch); (the intercept, whether the
    float32 factor is trusted: x_linear/ridge.mojo `chol_trusted`). An
    untrusted alpha is solved again in float-float by the caller."""
    for j in range(d):
        for k in range(d):
            var v = ld(g, j * d + k)
            if j == k:
                v = fa(v, alpha)
            st(aw, j * d + k, v)
    for j in range(d):
        st(w, j, ld(xty, j))
    var ok = cholesky(aw, 0, d)
    if not chol_trusted(ok, aw, g, d, alpha):
        return (Float32(0), False)
    chol_solve(aw, 0, d, w, 0)
    if not fi:
        return (Float32(0), True)
    var acc = Float32(0)
    for j in range(d):
        acc = fmad(ld(xm, j), ld(w, j), acc)
    return (fs(ym, acc), True)


def kf_ff_solve(d: Int, fi: Bool, alpha: Float32, sh: FP, sl: FP, bh: FP, bl: FP, fh: FP, fl: FP,
                tmp: FP, w: FP) -> Float32:
    """The float-float solve of one alpha from the fold's float-float
    statistics (ridge_ff_unit over the training rows): w and the intercept,
    or NaN words when float-float cannot factor it either."""
    if ridge_ff_solve(d, 1, fi, alpha, sh, sl, bh, bl, tmp, fh, fl):
        for j in range(d):
            st(w, j, ld(tmp, j))
        return ld(tmp, d)
    var nan = Float32(0) / Float32(0)
    for j in range(d):
        st(w, j, nan)
    return nan


@always_inline
def kf_pred(x: FP, i: Int, d: Int, w: FP, b: Float32) -> Float32:
    return fa(row_dot(x, i, d, w, 0), b)


# the held-out r2 in the blocked order (cgr-linear): FOLD_BLOCK rows of the
# held-out fold from zero, the block partials folded ascending; the device
# runs a thread per (alpha, block), then a thread per alpha
@always_inline
def kf_blocks(s: Int, e: Int) -> Int:
    return (e - s + FOLD_BLOCK - 1) // FOLD_BLOCK


@always_inline
def kf_ysum_part(y: FP, s: Int, e: Int, b: Int) -> Float32:
    var lo = s + b * FOLD_BLOCK
    var acc = Float32(0)
    for i in range(lo, min(lo + FOLD_BLOCK, e)):
        acc = fa(acc, ld(y, i))
    return acc


@always_inline
def kf_sq_part(y: FP, p: FP, s: Int, e: Int, mt: Float32, b: Int) -> Tuple[Float32, Float32]:
    """(sum (y - p)^2, sum (y - mt)^2) over block b of [s, e) from zero."""
    var lo = s + b * FOLD_BLOCK
    var ssr = Float32(0)
    var sst = Float32(0)
    for i in range(lo, min(lo + FOLD_BLOCK, e)):
        var r = fs(ld(y, i), ld(p, i - s))
        ssr = fmad(r, r, ssr)
        var c = fs(ld(y, i), mt)
        sst = fmad(c, c, sst)
    return (ssr, sst)


@always_inline
def kf_score_final(ssr: Float32, sst: Float32) -> Float32:
    if sst == 0:
        return Float32(1) if ssr == 0 else Float32(0)
    return fs(Float32(1), fd(ssr, sst))


def kf_score(y: FP, p: FP, s: Int, e: Int) -> Float32:
    """r2_score of p[0, e - s) against y[s, e) (their force_finite: a zero
    total sum of squares scores 1 for a perfect fit, else 0), blocked."""
    var nb = kf_blocks(s, e)
    var acc = Float32(0)
    for b in range(nb):
        acc = fa(acc, kf_ysum_part(y, s, e, b))
    var mt = fd(acc, i2f(e - s))
    var ssr = Float32(0)
    var sst = Float32(0)
    for b in range(nb):
        var q = kf_sq_part(y, p, s, e, mt, b)
        ssr = fa(ssr, q[0])
        sst = fa(sst, q[1])
    return kf_score_final(ssr, sst)


def kf_fw_words(n: Int, d: Int, k: Int, na: Int) -> Int:
    """Host scratch: xm d | ym 1 | G d*d | xty d | aw d*d | w na*d | b na | p na*(n//k + 1) | sums na."""
    return d + 1 + d * d + d + d * d + na * d + na + na * (n // k + 1) + na


def ridge_kfold_fit(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP):
    """ip: [k, fit_intercept, n_alphas]; fp: alphas. res: the mean score of
    each alpha. The host form: cells and rows as independent units."""
    var k = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var na = ldi(ip, 2)
    var xm = fw
    var ymp = xm + d
    var g = ymp + 1
    var xty = g + d * d
    var aw = xty + d
    var w = aw + d * d
    var bb = w + na * d
    var p = bb + na
    var sums = p + na * (n // k + 1)
    fill(sums, 0, na, Float32(0))
    # float-float scratch (lane/neural-pass93): stats hi | lo, b hi | lo, factor hi | lo, tmp d + 1
    var ffw = d + 1 + d * d + d
    var ffl = List[Float32](length=2 * ffw + 2 * d + 2 * d * d + d + 1, fill=Float32(0))
    var ffb = FP(unsafe_from_address=Int(ffl.unsafe_ptr()))
    for f in range(k):
        var s = kf_start(n, k, f)
        var e = kf_end(n, k, f)

        def means(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm s, imm e, imm fi, imm xm, imm ymp}:
            for j in range(lo, hi):
                if j < d:
                    st(xm, j, kf_mean(x, d, j, n, s, e) if fi else Float32(0))
                else:
                    st(ymp, 0, kf_mean(y, 1, 0, n, s, e) if fi else Float32(0))

        seq_rows(means, d + 1, 1)

        def cells(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm s, imm e, imm xm, imm ymp, imm g, imm xty}:
            for j in range(lo, hi):
                var mj = ld(xm, j)
                for c in range(j, d):
                    var v = kf_cross(x, d, j, mj, x, d, c, ld(xm, c), n, s, e)
                    st(g, j * d + c, v)
                    st(g, c * d + j, v)
                st(xty, j, kf_cross(x, d, j, mj, y, 1, 0, ld(ymp, 0), n, s, e))

        seq_rows(cells, d, 1)
        var have_ff = False
        for a in range(na):
            var r = kf_solve(g, xty, xm, ld(ymp, 0), d, ld(fp, a), fi, aw, w + a * d)
            if r[1]:
                st(bb, a, r[0])
                continue
            # lane/neural-pass93: this alpha in float-float over the fold's training rows
            if not have_ff:
                _kf_ff_stats_host(x, y, n, d, fi, s, e, ffb)
                have_ff = True
            st(bb, a, kf_ff_solve(d, fi, ld(fp, a), ffb, ffb + ffw, ffb + 2 * ffw, ffb + 2 * ffw + d,
                                  ffb + 2 * ffw + 2 * d, ffb + 2 * ffw + 2 * d + d * d,
                                  ffb + 2 * ffw + 2 * d + 2 * d * d, w + a * d))
        for a in range(na):
            for i in range(s, e):
                st(p, i - s, kf_pred(x, i, d, w + a * d, ld(bb, a)))
            st(sums, a, fa(ld(sums, a), kf_score(y, p, s, e)))
    for a in range(na):
        st(res, a, fd(ld(sums, a), i2f(k)))
    _ = ffl^


def _kf_ff_stats_host(x: FP, y: FP, n: Int, d: Int, fi: Bool, s: Int, e: Int, ffb: FP):
    """ridge_ff_unit's float-float statistics over the rows outside [s, e)."""
    var ffw = d + 1 + d * d + d
    var sh = ffb
    var sl = ffb + ffw
    var units = ridge_ff_units(n, d, 1)

    def means(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm fi, imm sh, imm sl, imm s, imm e}:
        for u in range(lo, hi):
            ridge_ff_unit(u, x, y, n, d, 1, fi, False, n, sh, sl, s, e)

    def cells(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm fi, imm sh, imm sl, imm s, imm e}:
        for u in range(lo, hi):
            ridge_ff_unit(d + 1 + u, x, y, n, d, 1, fi, False, n, sh, sl, s, e)

    seq_rows(means, d + 1, 1)
    seq_rows(cells, units - (d + 1), 1)
