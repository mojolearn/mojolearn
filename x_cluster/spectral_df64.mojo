# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SpectralClustering assign_labels='discretize' (and, below, 'cluster_qr') in FLOAT-FLOAT (df64)
arithmetic, one algorithm on every column (lane n9b-device-svd-qr,
2026-11-09). The device driver is x_cluster/spectral_df64_device.mojo; the
host column is x_cluster/host/spectral_df64_host.mojo `discretize_df64_host`. Both run the per-index bodies
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


# ---------------------------------------------------------------- cluster_qr
# Lane m3-linalg-and-housekeeping (2026-11-09): assign_labels='cluster_qr'
# in float-float on every column (was binary64 on the host, the n x k
# pivoted QR and the k x k Jacobi SVD serial in x_cluster/spectral_assign.mojo).
# scikit-learn: `_, _, piv = qr(V.T, pivoting=True)`, `ut, _, v =
# svd(V[piv[:k]].T)`, labels = argmax |V (ut v)|. The QR here is a
# Householder QR of A = V^T (column c = row c of the embedding) that
# RECOMPUTES every remaining column's norm each step (LAPACK geqp3
# downdates it), the largest norm first and the lowest column on a tie
# (idamax's rule); only the pivots are used. The SVD is the discretize SVD
# above on M = V[piv[:k]] (row c = embedding row piv[c]), so with M = U S
# V^T the rotation V U^T (`sd_polar_cell`) is B's U_B Vh_B for B = M^T,
# scikit-learn's ut v. Each row's label is the first argmax of |row . Q|.


@always_inline
def sd_better_max(v: FF, i: Int, bv: FF, bi: Int) -> Bool:
    """(v, i) before (bv, bi): the larger value, the lower index on a tie
    (bi < 0: nothing yet). The order of the comparisons does not matter."""
    if bi < 0:
        return True
    if sd_lt(bv, v):
        return True
    return not sd_lt(v, bv) and i < bi


@always_inline
def cq_colnorm_cell(ah: FP, al: FP, used: IP, k: Int, j: Int, c: Int, nh: FP, nl: FP):
    """nrm[c] = sum_{r >= j} a[c, r]^2 (r ascending) for a column not yet a
    pivot; a pivot's word is left (the max skips it)."""
    if Int(used.unsafe_load(c)) != 0:
        return
    var s = sd_zero()
    for r in range(j, k):
        var v = sd_ld(ah, al, c * k + r)
        s = ff_add(s, ff_mul(v, v))
    sd_st(nh, nl, c, s)


@always_inline
def cq_chunk_max(nh: FP, nl: FP, used: IP, n: Int, ch: Int, bmh: FP, bml: FP, bmi: IP):
    """Chunk ch's (value, index) maximum over the columns not yet a pivot,
    columns ascending; index -1 when every column of the chunk is one."""
    var bi = -1
    var bv = sd_zero()
    for c in range(ch * SD_ROWS, min(ch * SD_ROWS + SD_ROWS, n)):
        if Int(used.unsafe_load(c)) != 0:
            continue
        var v = sd_ld(nh, nl, c)
        if sd_better_max(v, c, bv, bi):
            bv = v
            bi = c
    sd_st(bmh, bml, ch, bv)
    bmi.unsafe_store(ch, Int32(bi))


def cq_pivot_step(
    ah: FP, al: FP, used: IP, piv: IP, k: Int, j: Int, bmh: FP, bml: FP, bmi: IP, nch: Int, hvh: FP, hvl: FP,
    hbh: FP, hbl: FP,
) -> Int:
    """Step j's pivot: the chunk maxima folded (exact compares, the lowest
    column on a tie), marked used, piv[j] set; then the Householder vector
    hv (rows j..k-1) zeroing a[best, j+1:] and hb = hv . hv. Returns 1 when
    the reflector applies (j < k - 1, a nonzero norm and hb != 0), else 0.
    k work, one thread."""
    var bi = -1
    var bv = sd_zero()
    for ch in range(nch):
        var vi = Int(bmi.unsafe_load(ch))
        if vi < 0:
            continue
        var v = sd_ld(bmh, bml, ch)
        if sd_better_max(v, vi, bv, bi):
            bv = v
            bi = vi
    used.unsafe_store(bi, Int32(1))
    piv.unsafe_store(j, Int32(bi))
    if j == k - 1 or bv.hi == Float32(0):
        return 0
    var nrm = sd_sqrt0(bv)
    var x0 = sd_ld(ah, al, bi * k + j)
    var alpha = ff_neg(nrm) if x0.hi >= Float32(0) else nrm
    for r in range(k):
        var v = sd_zero()
        if r == j:
            v = ff_sub(x0, alpha)
        elif r > j:
            v = sd_ld(ah, al, bi * k + r)
        sd_st(hvh, hvl, r, v)
    var hb = sd_zero()
    for r in range(j, k):
        var v = sd_ld(hvh, hvl, r)
        hb = ff_add(hb, ff_mul(v, v))
    sd_st(hbh, hbl, 0, hb)
    return 0 if hb.hi == Float32(0) else 1


@always_inline
def cq_apply_col(ah: FP, al: FP, used: IP, k: Int, j: Int, hvh: FP, hvl: FP, hb: FF, c: Int):
    """Column c (not a pivot) -= (2 (hv . a_c) / hb) hv, rows j..k-1
    ascending. A pivot column is never read again, so it is left."""
    if Int(used.unsafe_load(c)) != 0:
        return
    var d = sd_zero()
    for r in range(j, k):
        d = ff_add(d, ff_mul(sd_ld(hvh, hvl, r), sd_ld(ah, al, c * k + r)))
    var f = ff_div(ff_mul(FF(Float32(2), Float32(0)), d), hb)
    for r in range(j, k):
        sd_st(ah, al, c * k + r, ff_sub(sd_ld(ah, al, c * k + r), ff_mul(f, sd_ld(hvh, hvl, r))))


@always_inline
def cq_gather_cell(vh: FP, vl: FP, piv: IP, k: Int, t: Int, mh: FP, ml: FP):
    """M[c, r] = embedding[piv[c], r]."""
    var c = t // k
    var r = t - c * k
    sd_st(mh, ml, t, sd_ld(vh, vl, Int(piv.unsafe_load(c)) * k + r))


@always_inline
def sd_label_abs_row(vh: FP, vl: FP, k: Int, roth: FP, rotl: FP, i: Int) -> Int32:
    """The first argmax over c of |vec_i . rot[:, c]| (cluster_qr)."""
    var best = 0
    var bv = sd_zero()
    for c in range(k):
        var s = sd_abs(sd_row_dot_rot(vh, vl, k, roth, rotl, i, c))
        if c == 0 or sd_lt(bv, s):
            best = c
            bv = s
    return Int32(best)


def cq_raise_status(code: Int) raises:
    if code == SD_BAD_BASIS:
        raise Error("SpectralClustering cluster_qr: SVD did not converge (no basis vector completes U)")
    if code != 0:
        raise Error("SpectralClustering cluster_qr: status " + String(code))


def sd_raise_status(code: Int) raises:
    if code == SD_BAD_COL:
        raise Error("SpectralClustering discretize: an embedding column is all zeros")
    if code == SD_BAD_ROW:
        raise Error("SpectralClustering discretize: an embedding row is all zeros")
    if code == SD_BAD_BASIS:
        raise Error("SpectralClustering discretize: SVD did not converge (no basis vector completes U)")
    if code != 0:
        raise Error("SpectralClustering discretize: status " + String(code))
