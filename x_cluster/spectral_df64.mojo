# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SpectralClustering assign_labels='discretize' in FLOAT-FLOAT (df64)
arithmetic, one algorithm on every column (lane n9b-device-svd-qr,
2026-11-09). The device driver is x_cluster/spectral_df64_device.mojo; the
host column is `discretize_df64_host` below. Both run the per-index bodies
of this file, so the words are the same everywhere. No GPU import.

Reference: scikit-learn `sklearn/cluster/_spectral.py::discretize` (Yu and
Shi 2003), the statements x_cluster/spectral_assign.mojo ran in binary64 on
the host (that host algorithm is gone: its n x k loops were the fit's last
host step). Every value is an `FF` (x_linear/ff.mojo: hi + lo float32
words, about 48 bits; each operation a fixed sequence of IDENTICAL float32
operations), so the device and the host add the same pairs:

- the embedding's columns scaled to norm sqrt(n) and signed so the first
  row is not positive; the column sums of squares fold SD_ROWS rows a
  partial (rows ascending), then the partials ascending;
- rows scaled to unit length (each row's k squares ascending);
- the greedy orthogonal start from row `rng.below(n)` (the lane's
  splitmix64 stream), each step's argmin the lowest index on a tie;
- the iteration: labels = the first argmax of each row against the rotation,
  the k x k class sums (SD_ROWS rows a partial, rows ascending; the partials
  ascending), their one-sided (Hestenes) Jacobi SVD with the pairs (p < q)
  in row order every sweep until a sweep rotates nothing (cap
  SD_SVD_SWEEPS), a rotation skipped when |gamma| <= SD_SVD_EPS
  sqrt(alpha beta) (SD_SVD_EPS sits above float-float's rounding, as the
  binary64 form's 2^-52 sat above binary64's), U completed by Gram-Schmidt
  against e_0, e_1, ... where a singular value is 0; then ncut = 2 (n -
  sum S) and the rotation V U^T, until ncut moves by less than SD_NCUT_EPS
  or n_iter exceeds n_iter_max. DEVIATION 5119.

BITS. The binary64 host algorithm's labels came from binary64 sums in one
row order; these are float-float sums in the blocked order: labels can move
where two classes tie to within float-float rounding (spectral-clustering
identity lane, every column together).
"""
from cluster.host.host_cells import host_cells
from x_cluster.bodies import SplitMix64
from x_linear.ff import FF, ff_add, ff_div, ff_mul, ff_neg, ff_sqrt, ff_sub, two_sum
from x_linear.ops import FP, IP

comptime SD_ROWS = 1024
"""Rows per partial of the column and class sums (the fold's leaves)."""
comptime SD_SVD_SWEEPS = 80
comptime SD_SVD_EPS = Float32(5.684341886080802e-14)
"""2^-44: the Jacobi skip test's relative bound."""
comptime SD_NCUT_EPS = Float32(2.220446049250313e-16)
"""The ncut stop test's absolute bound (the binary64 form's float64 eps)."""
comptime SD_COMPLETE_MIN = Float32(1e-8)
"""A Gram-Schmidt completion candidate's smallest kept norm."""

#: the state words (int32): done, n_iter, status, argmin
comptime SD_DONE = 0
comptime SD_ITER = 1
comptime SD_STATUS = 2
comptime SD_AMIN = 3
comptime SD_STATE_WORDS = 4
#: status codes
comptime SD_BAD_COL = 1
comptime SD_BAD_ROW = 2
comptime SD_BAD_BASIS = 3


@always_inline
def sd_chunks(n: Int) -> Int:
    return (n + SD_ROWS - 1) // SD_ROWS


@always_inline
def sd_ld(h: FP, l: FP, i: Int) -> FF:
    return FF(h.unsafe_load(i), l.unsafe_load(i))


@always_inline
def sd_st(h: FP, l: FP, i: Int, v: FF):
    h.unsafe_store(i, v.hi)
    l.unsafe_store(i, v.lo)


@always_inline
def sd_zero() -> FF:
    return FF(Float32(0), Float32(0))


@always_inline
def sd_abs(x: FF) -> FF:
    return ff_neg(x) if x.hi < Float32(0) else x


@always_inline
def sd_lt(a: FF, b: FF) -> Bool:
    """a < b for normalized float-float values (hi first, then lo)."""
    return a.hi < b.hi or (a.hi == b.hi and a.lo < b.lo)


@always_inline
def sd_sqrt0(a: FF) -> FF:
    """sqrt(a), 0 for a <= 0 (`ff_sqrt` needs a > 0)."""
    if not (a.hi > Float32(0)):
        return sd_zero()
    return ff_sqrt(a)


@always_inline
def sd_of_int(n: Int) -> FF:
    """n exactly (n < 2^48): the float32 rounding plus the exact rest."""
    var hi = Float32(n)
    var rest = n - Int(hi)
    return two_sum(hi, Float32(rest))


# ---------------------------------------------------------------- bodies


@always_inline
def sd_load_cell(x: FP, vh: FP, vl: FP, t: Int):
    vh.unsafe_store(t, x.unsafe_load(t))
    vl.unsafe_store(t, Float32(0))


@always_inline
def sd_colsq_part(vh: FP, vl: FP, n: Int, k: Int, ch: Int, c: Int, ph: FP, pl: FP):
    """Partial ch of column c's sum of squares: rows [ch SD_ROWS, ...) ascending."""
    var acc = sd_zero()
    var lo = ch * SD_ROWS
    var hi = min(lo + SD_ROWS, n)
    for i in range(lo, hi):
        var v = sd_ld(vh, vl, i * k + c)
        acc = ff_add(acc, ff_mul(v, v))
    sd_st(ph, pl, ch * k + c, acc)


@always_inline
def sd_colnorm(ph: FP, pl: FP, nch: Int, k: Int, c: Int, cnh: FP, cnl: FP, state: IP):
    """Column c's norm from its partials (ascending); a zero column sets the
    status and stores 1 (nothing divides by zero)."""
    var acc = sd_zero()
    for ch in range(nch):
        acc = ff_add(acc, sd_ld(ph, pl, ch * k + c))
    if not (acc.hi > Float32(0)):
        state.unsafe_store(SD_STATUS, Int32(SD_BAD_COL))
        sd_st(cnh, cnl, c, FF(Float32(1), Float32(0)))
        return
    sd_st(cnh, cnl, c, ff_sqrt(acc))


@always_inline
def sd_colscale_cell(vh: FP, vl: FP, k: Int, cnh: FP, cnl: FP, rootn: FF, sgn: FP, t: Int):
    """vec[i, c] = vec[i, c] / norm_c * sqrt(n); row 0 records the column's
    sign word for `sd_colsign_cell`."""
    var c = t % k
    var v = ff_mul(ff_div(sd_ld(vh, vl, t), sd_ld(cnh, cnl, c)), rootn)
    sd_st(vh, vl, t, v)
    if t < k:
        sgn.unsafe_store(c, v.hi)


@always_inline
def sd_colsign_cell(vh: FP, vl: FP, k: Int, sgn: FP, t: Int):
    """-vec * sign(f0) with f0 the column's first entry: a positive f0
    negates the column, a negative or zero one keeps it (exact)."""
    if sgn.unsafe_load(t % k) > Float32(0):
        sd_st(vh, vl, t, ff_neg(sd_ld(vh, vl, t)))


@always_inline
def sd_rownorm_row(vh: FP, vl: FP, k: Int, i: Int, state: IP):
    """Row i to unit length (its k squares ascending); a zero row sets the status."""
    var acc = sd_zero()
    for c in range(k):
        var v = sd_ld(vh, vl, i * k + c)
        acc = ff_add(acc, ff_mul(v, v))
    if not (acc.hi > Float32(0)):
        state.unsafe_store(SD_STATUS, Int32(SD_BAD_ROW))
        return
    var rn = ff_sqrt(acc)
    for c in range(k):
        sd_st(vh, vl, i * k + c, ff_div(sd_ld(vh, vl, i * k + c), rn))


@always_inline
def sd_row_dot_rot(vh: FP, vl: FP, k: Int, roth: FP, rotl: FP, i: Int, col: Int) -> FF:
    """sum_r vec[i, r] rot[r, col], r ascending."""
    var s = sd_zero()
    for r in range(k):
        s = ff_add(s, ff_mul(sd_ld(vh, vl, i * k + r), sd_ld(roth, rotl, r * k + col)))
    return s


@always_inline
def sd_cacc_row(vh: FP, vl: FP, k: Int, roth: FP, rotl: FP, col: Int, cah: FP, cal: FP, i: Int):
    """c[i] += |vec_i . rot[:, col]| (the greedy start's accumulator)."""
    var s = sd_row_dot_rot(vh, vl, k, roth, rotl, i, col)
    sd_st(cah, cal, i, ff_add(sd_ld(cah, cal, i), sd_abs(s)))


@always_inline
def sd_take_row_cell(vh: FP, vl: FP, k: Int, row: Int, roth: FP, rotl: FP, col: Int, r: Int):
    """rot[r, col] = vec[row, r]."""
    sd_st(roth, rotl, r * k + col, sd_ld(vh, vl, row * k + r))


@always_inline
def sd_better_min(v: FF, i: Int, bv: FF, bi: Int) -> Bool:
    """(v, i) before (bv, bi): the smaller value, the lower index on a tie
    (bi < 0: nothing yet). The order of the comparisons does not matter."""
    if bi < 0:
        return True
    if sd_lt(v, bv):
        return True
    return not sd_lt(bv, v) and i < bi


@always_inline
def sd_label_row(vh: FP, vl: FP, k: Int, roth: FP, rotl: FP, i: Int) -> Int32:
    """The first argmax over c of vec_i . rot[:, c]."""
    var best = 0
    var bv = sd_zero()
    for c in range(k):
        var s = sd_row_dot_rot(vh, vl, k, roth, rotl, i, c)
        if c == 0 or sd_lt(bv, s):
            best = c
            bv = s
    return Int32(best)


@always_inline
def sd_class_part(vh: FP, vl: FP, labels: IP, n: Int, k: Int, ch: Int, cell: Int, ph: FP, pl: FP):
    """Partial ch of class sum cell (cls, r): vec[i, r] over the chunk's
    rows labelled cls, rows ascending."""
    var cls = cell // k
    var r = cell - cls * k
    var acc = sd_zero()
    var lo = ch * SD_ROWS
    var hi = min(lo + SD_ROWS, n)
    for i in range(lo, hi):
        if Int(labels.unsafe_load(i)) == cls:
            acc = ff_add(acc, sd_ld(vh, vl, i * k + r))
    sd_st(ph, pl, ch * k * k + cell, acc)


@always_inline
def sd_class_fold(ph: FP, pl: FP, nch: Int, kk: Int, cell: Int, mh: FP, ml: FP):
    var acc = sd_zero()
    for ch in range(nch):
        acc = ff_add(acc, sd_ld(ph, pl, ch * kk + cell))
    sd_st(mh, ml, cell, acc)


# ---- the k x k one-sided Jacobi SVD's pieces (w = m's columns rotated)


@always_inline
def sd_dot_cols(wh: FP, wl: FP, k: Int, p: Int, q: Int) -> FF:
    """sum_r w[r, p] w[r, q], rows ascending."""
    var s = sd_zero()
    for r in range(k):
        s = ff_add(s, ff_mul(sd_ld(wh, wl, r * k + p), sd_ld(wh, wl, r * k + q)))
    return s


@always_inline
def sd_rotation(alpha: FF, beta: FF, gamma: FF) -> Tuple[Bool, FF, FF]:
    """(rotate, c, s) of a pair from its sums (skip: gamma == 0 or |gamma| <=
    SD_SVD_EPS sqrt(alpha beta))."""
    var g = sd_abs(gamma)
    var bound = ff_mul(FF(SD_SVD_EPS, Float32(0)), sd_sqrt0(ff_mul(alpha, beta)))
    if gamma.hi == Float32(0) or not sd_lt(bound, g):
        return (False, FF(Float32(1), Float32(0)), sd_zero())
    var two = FF(Float32(2), Float32(0))
    var one = FF(Float32(1), Float32(0))
    var zeta = ff_div(ff_sub(beta, alpha), ff_mul(two, gamma))
    var t = ff_div(one, ff_add(sd_abs(zeta), ff_sqrt(ff_add(one, ff_mul(zeta, zeta)))))
    if zeta.hi < Float32(0):
        t = ff_neg(t)
    var c = ff_div(one, ff_sqrt(ff_add(one, ff_mul(t, t))))
    var s = ff_mul(c, t)
    return (True, c, s)


@always_inline
def sd_apply_row(wh: FP, wl: FP, vh: FP, vl: FP, k: Int, p: Int, q: Int, c: FF, s: FF, r: Int):
    """Row r of the rotation of columns p, q of w and of v."""
    var wp = sd_ld(wh, wl, r * k + p)
    var wq = sd_ld(wh, wl, r * k + q)
    sd_st(wh, wl, r * k + p, ff_sub(ff_mul(c, wp), ff_mul(s, wq)))
    sd_st(wh, wl, r * k + q, ff_add(ff_mul(s, wp), ff_mul(c, wq)))
    var vp = sd_ld(vh, vl, r * k + p)
    var vq = sd_ld(vh, vl, r * k + q)
    sd_st(vh, vl, r * k + p, ff_sub(ff_mul(c, vp), ff_mul(s, vq)))
    sd_st(vh, vl, r * k + q, ff_add(ff_mul(s, vp), ff_mul(c, vq)))


@always_inline
def sd_norm_col(wh: FP, wl: FP, uh: FP, ul: FP, svh: FP, svl: FP, filled: IP, k: Int, j: Int):
    """sv[j] = ||w[:, j]||, u[:, j] = w[:, j] / sv[j] (filled[j] = 1), or
    filled[j] = 0 for a zero column (u's column left for the completion)."""
    var nj = sd_sqrt0(sd_dot_cols(wh, wl, k, j, j))
    sd_st(svh, svl, j, nj)
    if nj.hi == Float32(0):
        filled.unsafe_store(j, Int32(0))
        for r in range(k):
            sd_st(uh, ul, r * k + j, sd_zero())
        return
    filled.unsafe_store(j, Int32(1))
    for r in range(k):
        sd_st(uh, ul, r * k + j, ff_div(sd_ld(wh, wl, r * k + j), nj))


def sd_complete_u(uh: FP, ul: FP, filled: IP, ch: FP, cl: FP, k: Int) -> Bool:
    """U's zero columns, lowest first, by e_0, e_1, ... minus their
    projections on the filled columns, the first whose norm passes
    SD_COMPLETE_MIN; ch/cl are k words of scratch. False when no basis
    vector completes U. Serial by nature (each completed column joins the
    projections of the next); k x k work, one thread."""
    var e = 0
    for j in range(k):
        if Int(filled.unsafe_load(j)) != 0:
            continue
        while True:
            if e >= k:
                return False
            for r in range(k):
                sd_st(ch, cl, r, FF(Float32(1) if r == e else Float32(0), Float32(0)))
            e += 1
            for c in range(k):
                if Int(filled.unsafe_load(c)) == 0:
                    continue
                var proj = sd_zero()
                for r in range(k):
                    proj = ff_add(proj, ff_mul(sd_ld(uh, ul, r * k + c), sd_ld(ch, cl, r)))
                for r in range(k):
                    sd_st(ch, cl, r, ff_sub(sd_ld(ch, cl, r), ff_mul(proj, sd_ld(uh, ul, r * k + c))))
            var nn = sd_zero()
            for r in range(k):
                var x = sd_ld(ch, cl, r)
                nn = ff_add(nn, ff_mul(x, x))
            nn = sd_sqrt0(nn)
            if nn.hi > SD_COMPLETE_MIN:
                for r in range(k):
                    sd_st(uh, ul, r * k + j, ff_div(sd_ld(ch, cl, r), nn))
                filled.unsafe_store(j, Int32(1))
                break
    return True


@always_inline
def sd_polar_cell(vh: FP, vl: FP, uh: FP, ul: FP, k: Int, t: Int, roth: FP, rotl: FP):
    """rot[i, j] = sum_c v[i, c] u[j, c] (c ascending): V U^T."""
    var i = t // k
    var j = t - i * k
    var s = sd_zero()
    for c in range(k):
        s = ff_add(s, ff_mul(sd_ld(vh, vl, i * k + c), sd_ld(uh, ul, j * k + c)))
    sd_st(roth, rotl, t, s)


@always_inline
def sd_ncut_step(svh: FP, svl: FP, k: Int, n: Int, lasth: FP, lastl: FP, state: IP, n_iter_max: Int) -> Bool:
    """After the SVD: ncut = 2 (n - sum sv) (sv ascending by index), the
    stop test; True (and done) on a stop, else last = ncut."""
    var ssum = sd_zero()
    for c in range(k):
        ssum = ff_add(ssum, sd_ld(svh, svl, c))
    var ncut = ff_mul(FF(Float32(2), Float32(0)), ff_sub(sd_of_int(n), ssum))
    var diff = sd_abs(ff_sub(ncut, sd_ld(lasth, lastl, 0)))
    var it = Int(state.unsafe_load(SD_ITER))
    if sd_lt(diff, FF(SD_NCUT_EPS, Float32(0))) or it > n_iter_max:
        state.unsafe_store(SD_DONE, Int32(1))
        return True
    sd_st(lasth, lastl, 0, ncut)
    return False


# ---------------------------------------------------------------- the host column


@always_inline
def _lp(mut l: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(l.unsafe_ptr()))


@always_inline
def _li(mut l: List[Int32]) -> IP:
    return IP(unsafe_from_address=Int(l.unsafe_ptr()))


def sd_raise_status(code: Int) raises:
    if code == SD_BAD_COL:
        raise Error("SpectralClustering discretize: an embedding column is all zeros")
    if code == SD_BAD_ROW:
        raise Error("SpectralClustering discretize: an embedding row is all zeros")
    if code == SD_BAD_BASIS:
        raise Error("SpectralClustering discretize: SVD did not converge (no basis vector completes U)")
    if code != 0:
        raise Error("SpectralClustering discretize: status " + String(code))


def discretize_df64_host(
    x: List[Float32], n: Int, k: Int, seed: UInt64, max_svd_restarts: Int, n_iter_max: Int, mut n_iter_out: Int
) raises -> List[Int32]:
    """The host column of the device driver (x_cluster/spectral_df64_device.mojo):
    the same bodies in the same order, the row and partial steps in parallel
    over the host pool (each index writes only its own words), the k x k SVD
    serial (each rotation reads the previous one's words)."""
    if n < 1 or k < 1 or len(x) != n * k:
        raise Error("SpectralClustering discretize: the embedding is not n x k")
    if max_svd_restarts < 1:
        raise Error("SpectralClustering discretize: SVD did not converge")
    var nk = n * k
    var kk = k * k
    var nch = sd_chunks(n)
    var xl = x.copy()
    var vhl = List[Float32](length=nk, fill=Float32(0))
    var vll = List[Float32](length=nk, fill=Float32(0))
    var phl = List[Float32](length=max(nch * kk, nch * k), fill=Float32(0))
    var pll = List[Float32](length=max(nch * kk, nch * k), fill=Float32(0))
    var cnhl = List[Float32](length=k, fill=Float32(0))
    var cnll = List[Float32](length=k, fill=Float32(0))
    var sgnl = List[Float32](length=k, fill=Float32(0))
    var stl = List[Int32](length=SD_STATE_WORDS, fill=Int32(0))
    var px = _lp(xl)
    var vh = _lp(vhl)
    var vl = _lp(vll)
    var ph = _lp(phl)
    var pl = _lp(pll)
    var cnh = _lp(cnhl)
    var cnl = _lp(cnll)
    var sgn = _lp(sgnl)
    var state = _li(stl)

    def load(t: Int) {imm px, imm vh, imm vl}:
        sd_load_cell(px, vh, vl, t)

    host_cells(load, nk, 2)

    def colsq(t: Int) {imm vh, imm vl, imm n, imm k, imm ph, imm pl}:
        sd_colsq_part(vh, vl, n, k, t // k, t % k, ph, pl)

    host_cells(colsq, nch * k, 20 * SD_ROWS)
    for c in range(k):  # small-loop(k: embedding columns): each folds its nch partials
        sd_colnorm(ph, pl, nch, k, c, cnh, cnl, state)
    sd_raise_status(Int(stl[SD_STATUS]))
    var rootn = ff_sqrt(sd_of_int(n))

    def colscale(t: Int) {imm vh, imm vl, imm k, imm cnh, imm cnl, imm rootn, imm sgn}:
        sd_colscale_cell(vh, vl, k, cnh, cnl, rootn, sgn, t)

    host_cells(colscale, nk, 40)

    def colsign(t: Int) {imm vh, imm vl, imm k, imm sgn}:
        sd_colsign_cell(vh, vl, k, sgn, t)

    host_cells(colsign, nk, 2)

    def rownorm(i: Int) {imm vh, imm vl, imm k, imm state}:
        sd_rownorm_row(vh, vl, k, i, state)

    host_cells(rownorm, n, 40 * k)
    sd_raise_status(Int(stl[SD_STATUS]))
    # the greedy orthogonal start
    var rothl = List[Float32](length=kk, fill=Float32(0))
    var rotll = List[Float32](length=kk, fill=Float32(0))
    var cahl = List[Float32](length=n, fill=Float32(0))
    var call = List[Float32](length=n, fill=Float32(0))
    var roth = _lp(rothl)
    var rotl = _lp(rotll)
    var cah = _lp(cahl)
    var cal = _lp(call)
    var bmhl = List[Float32](length=nch, fill=Float32(0))
    var bmll = List[Float32](length=nch, fill=Float32(0))
    var bmil = List[Int32](length=nch, fill=Int32(0))
    var bmh = _lp(bmhl)
    var bml = _lp(bmll)
    var bmi = _li(bmil)
    var rng = SplitMix64(state=seed)
    var r0 = rng.below(n)
    for r in range(k):  # small-loop(k: embedding columns): the start row's words
        sd_take_row_cell(vh, vl, k, r0, roth, rotl, 0, r)
    for j in range(1, k):  # driver-loop: over the start's k - 1 greedy columns (model structure)
        var col = j - 1

        def cacc(i: Int) {imm vh, imm vl, imm k, imm roth, imm rotl, imm col, imm cah, imm cal}:
            sd_cacc_row(vh, vl, k, roth, rotl, col, cah, cal, i)

        host_cells(cacc, n, 20 * k)
        # the argmin: each SD_ROWS chunk's (value, index) minimum in
        # parallel, then the chunks' (exact compares: any order gives the
        # device's answer)

        def cmin(ch: Int) {imm cah, imm cal, imm n, imm bmh, imm bml, imm bmi}:
            var bi = -1
            var bv = sd_zero()
            for i in range(ch * SD_ROWS, min(ch * SD_ROWS + SD_ROWS, n)):
                var v = sd_ld(cah, cal, i)
                if sd_better_min(v, i, bv, bi):
                    bv = v
                    bi = i
            sd_st(bmh, bml, ch, bv)
            bmi.unsafe_store(ch, Int32(bi))

        host_cells(cmin, nch, 4 * SD_ROWS)
        var bi = -1
        var bv = sd_zero()
        for ch in range(nch):  # small-loop(nch: row chunks): the chunk minima, exact compares
            var v = sd_ld(bmh, bml, ch)
            var vi = Int(bmi.unsafe_load(ch))
            if sd_better_min(v, vi, bv, bi):
                bv = v
                bi = vi
        for r in range(k):  # small-loop(k: embedding columns): the argmin row's words
            sd_take_row_cell(vh, vl, k, bi, roth, rotl, j, r)
    # the iterations
    var labl = List[Int32](length=n, fill=Int32(0))
    var lab = _li(labl)
    var mhl = List[Float32](length=kk, fill=Float32(0))
    var mll = List[Float32](length=kk, fill=Float32(0))
    var mh = _lp(mhl)
    var ml = _lp(mll)
    var vvhl = List[Float32](length=kk, fill=Float32(0))
    var vvll = List[Float32](length=kk, fill=Float32(0))
    var uhl = List[Float32](length=kk, fill=Float32(0))
    var ull = List[Float32](length=kk, fill=Float32(0))
    var svhl = List[Float32](length=k, fill=Float32(0))
    var svll = List[Float32](length=k, fill=Float32(0))
    var chl = List[Float32](length=k, fill=Float32(0))
    var cll = List[Float32](length=k, fill=Float32(0))
    var fil = List[Int32](length=k, fill=Int32(0))
    var lastl = List[Float32](length=2, fill=Float32(0))
    var vvh = _lp(vvhl)
    var vvl = _lp(vvll)
    var uh = _lp(uhl)
    var ul = _lp(ull)
    var svh = _lp(svhl)
    var svl = _lp(svll)
    var cch = _lp(chl)
    var ccl = _lp(cll)
    var filled = _li(fil)
    var lh = _lp(lastl)
    var ll = lh + 1
    while stl[SD_DONE] == Int32(0):  # driver-loop: over the discretize iterations (at most n_iter_max + 1)
        stl[SD_ITER] = stl[SD_ITER] + 1

        def label(i: Int) {imm vh, imm vl, imm k, imm roth, imm rotl, imm lab}:
            lab.unsafe_store(i, sd_label_row(vh, vl, k, roth, rotl, i))

        host_cells(label, n, 20 * k * k)

        def cpart(t: Int) {imm vh, imm vl, imm lab, imm n, imm k, imm kk, imm ph, imm pl}:
            sd_class_part(vh, vl, lab, n, k, t // kk, t % kk, ph, pl)

        host_cells(cpart, nch * kk, 4 * SD_ROWS)

        def cfold(t: Int) {imm ph, imm pl, imm nch, imm kk, imm mh, imm ml}:
            sd_class_fold(ph, pl, nch, kk, t, mh, ml)

        host_cells(cfold, kk, 20 * nch)
        _sd_svd_host(mh, ml, vvh, vvl, uh, ul, svh, svl, filled, cch, ccl, k, state)
        sd_raise_status(Int(stl[SD_STATUS]))
        if not sd_ncut_step(svh, svl, k, n, lh, ll, state, n_iter_max):
            # rotation = V U^T

            def polar(t: Int) {imm vvh, imm vvl, imm uh, imm ul, imm k, imm roth, imm rotl}:
                sd_polar_cell(vvh, vvl, uh, ul, k, t, roth, rotl)

            host_cells(polar, kk, 20 * k)
    n_iter_out = Int(stl[SD_ITER])
    _ = xl^
    _ = vhl^
    _ = vll^
    _ = phl^
    _ = pll^
    _ = cnhl^
    _ = cnll^
    _ = sgnl^
    _ = rothl^
    _ = rotll^
    _ = cahl^
    _ = call^
    _ = bmhl^
    _ = bmll^
    _ = bmil^
    _ = mhl^
    _ = mll^
    _ = vvhl^
    _ = vvll^
    _ = uhl^
    _ = ull^
    _ = svhl^
    _ = svll^
    _ = chl^
    _ = cll^
    _ = fil^
    _ = lastl^
    return labl^


def _sd_svd_host(
    mh: FP, ml: FP, vh: FP, vl: FP, uh: FP, ul: FP, svh: FP, svl: FP, filled: IP, ch: FP, cl: FP, k: Int, state: IP
):
    """The device step kernel's SVD on the host: m (k x k) rotated in place
    (it is the class sums, rebuilt every iteration), v = I rotated, then sv,
    u and the completion. Serial: each rotation reads the previous one's
    words."""
    for t in range(k * k):
        var i = t // k
        sd_st(vh, vl, t, FF(Float32(1) if i * k + i == t else Float32(0), Float32(0)))
    for _sweep in range(SD_SVD_SWEEPS):
        var rotated = False
        for p in range(k - 1):
            for q in range(p + 1, k):
                var rot = sd_rotation(sd_dot_cols(mh, ml, k, p, p), sd_dot_cols(mh, ml, k, q, q), sd_dot_cols(mh, ml, k, p, q))
                if not rot[0]:
                    continue
                rotated = True
                for r in range(k):
                    sd_apply_row(mh, ml, vh, vl, k, p, q, rot[1], rot[2], r)
        if not rotated:
            break
    for j in range(k):
        sd_norm_col(mh, ml, uh, ul, svh, svl, filled, k, j)
    if not sd_complete_u(uh, ul, filled, ch, cl, k):
        state.unsafe_store(SD_STATUS, Int32(SD_BAD_BASIS))
