# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVGP in float-float (lane/neural-pass106, 2026-10-01), the fallback when the
float32 factors fail (taxi: Sigma = Kuu + jitter I + B / noise has
eigenvalues from 1e-6 to 9e5, and B's float32 accumulation error is larger
than the smallest). B = Kuf Kfu and b = Kuf y are accumulated in float-float
from the float32 Kfu values (`matmul_tn_acc_ff_item`: every product exact
(two-product), the sums double-double), so B is a Gram matrix to ~1e-14
and Sigma is positive definite as it should be; then the solve in
float-float (x_linear/ff.mojo): the two Cholesky factors, alpha, C = Kuu^-1 -
Sigma^-1, q_mu, q_sqrt and the bound, each rounded to float32 at the store.
Choice (Andrew's standing order, best accuracy): the jitter escalation needed
jitter 1 to 10 on taxi (held-out R2 0.13 / -0.04).

The solve is a fixed sequence of parallel item passes (cpu-gpu-cleanup
c-xneighbors, 2026-10-02; before, the whole solve ran on the host with host
threads per column). Each pass is one launch on the device and one loop over
the same items on the CPU column (x_neighbors/iter_host.mojo), so the words
match:
  1. `svgp_ff_init_item` (m*m items): A_u = Kuu + jitter I, A_s = A_u + B / noise.
  2. per column j: `svgp_ff_chol_item` (2 (m - j) items) writes column j of
     both Cholesky factors, out of place; every item recomputes the pivot in
     the same order.
  3. `svgp_ff_column_item` (m items, one per column): Kuu'^-1 e_j and
     Sigma^-1 e_j (C's column), Sigma^-1's column, S = Kuu' Sigma^-1 Kuu''s
     column and tr(Kuu^-1 B)'s term j.
  4. `svgp_ff_x_item` (m): x = Sigma^-1 b (row i of Sigma^-1 times b, k
     ascending), alpha = x / noise. `svgp_ff_qmu_item` (m): q_mu = Kuu' alpha.
  5. per column j: `svgp_ff_chol_s_item` (m - j) factors S; `svgp_ff_qsqrt_item` (m*m).
  6. `svgp_ff_part_item` (blocked partials: y'y over n in XN_FOLD_BLOCK
     blocks, b'x, the trace and the two log-determinants over m in
     SVGP_FF_FOLD blocks), then `svgp_ff_fin_item` folds the partials
     ascending and writes the bound.

Workspace layout (`svgp_ff_ws_size`): the flags and partials first (their
offsets depend only on the block counts), then the matrices."""
from x_neighbors.items import FP, XN_FOLD_BLOCK
from x_linear.ff import (
    FF, ff_of, two_prod, ff_add, ff_add_f, ff_sub, ff_mul, ff_div, ff_sqrt, ff_f32, ff_ld, ff_st, ff_chol_solve,
)
from checks.numerics import ftz, identical_log
from std.sys.compile import is_defined

#: elements of one partial of the m-long folds
comptime SVGP_FF_FOLD = 64


def matmul_tn_acc_ff_item(t: Int, a: FP, b: FP, rh: FP, rl: FP, rows: Int, n: Int, m: Int):
    """A^T B carried over row tiles (`matmul_item`'s fold, p ascending) in float-float: (rh, rl)[t] continued by
    sum_p a[p, i] b[p, j] over `rows` rows, p ascending, t = i*m + j."""
    var i = t // m
    var j = t - i * m
    var acc = FF(rh.unsafe_load(t), rl.unsafe_load(t))
    for p in range(rows):
        acc = ff_add(acc, two_prod(ftz(a.unsafe_load(p * n + i)), ftz(b.unsafe_load(p * m + j))))
    rh.unsafe_store(t, acc.hi)
    rl.unsafe_store(t, acc.lo)


#: lane apple-fast-gap-kapprox2-svgp (2026-10-03): MOJOLEARN_SVGP_FAST_SYMTILE
#: (FAST + Apple, off unless defined). B = Kuf Kfu is symmetric and
#: two_prod(x, y) == two_prod(y, x) bit for bit (hi = x*y and lo =
#: fma(x, y, -hi) commute), so only the upper 4 x 4 blocks are summed, each
#: thread carrying 16 entries over the rows (p ascending, the same fold as
#: `matmul_tn_acc_ff_item` per entry) and writing the mirror too: the same
#: words, half the products, a quarter of the loads.
comptime SVGP_SYM_TB = 4


@always_inline
def svgp_sym_nb(m: Int) -> Int:
    return (m + SVGP_SYM_TB - 1) // SVGP_SYM_TB


def matmul_tn_sym_ff_tile_item(t: Int, a: FP, rh: FP, rl: FP, rows: Int, m: Int):
    """t = bi * nb + bj (bj >= bi, else nothing): (rh, rl)[i, j] and [j, i]
    continued by sum_p a[p, i] a[p, j] for the 4 x 4 block (bi, bj)."""
    comptime TB = SVGP_SYM_TB
    var nb = svgp_sym_nb(m)
    var bi = t // nb
    var bj = t - bi * nb
    if bj < bi:
        return
    var i0 = bi * TB
    var j0 = bj * TB
    var hh = SIMD[DType.float32, TB * TB](0)
    var ll = SIMD[DType.float32, TB * TB](0)
    comptime for u in range(TB):
        comptime for v in range(TB):
            var i = i0 + u
            var j = j0 + v
            if i < m and j < m:
                hh[u * TB + v] = rh.unsafe_load(i * m + j)
                ll[u * TB + v] = rl.unsafe_load(i * m + j)
    for p in range(rows):
        var ai = SIMD[DType.float32, TB](0)
        var aj = SIMD[DType.float32, TB](0)
        comptime for u in range(TB):
            if i0 + u < m:
                ai[u] = ftz(a.unsafe_load(p * m + i0 + u))
            if j0 + u < m:
                aj[u] = ftz(a.unsafe_load(p * m + j0 + u))
        comptime for u in range(TB):
            comptime for v in range(TB):
                var c = ff_add(FF(hh[u * TB + v], ll[u * TB + v]), two_prod(ai[u], aj[v]))
                hh[u * TB + v] = c.hi
                ll[u * TB + v] = c.lo
    comptime for u in range(TB):
        comptime for v in range(TB):
            var i = i0 + u
            var j = j0 + v
            if i < m and j < m:
                rh.unsafe_store(i * m + j, hh[u * TB + v])
                rl.unsafe_store(i * m + j, ll[u * TB + v])
                if bi != bj:
                    rh.unsafe_store(j * m + i, hh[u * TB + v])
                    rl.unsafe_store(j * m + i, ll[u * TB + v])


@always_inline
def _kj(kuu: FP, m: Int, i: Int, k: Int, jitter: Float32) -> FF:
    """(Kuu + jitter I)[i, k] as float-float."""
    var v = ff_of(kuu.unsafe_load(i * m + k))
    if i == k:
        v = ff_add_f(v, jitter)
    return v


@always_inline
def svgp_ff_nbm(m: Int) -> Int:
    return (m + SVGP_FF_FOLD - 1) // SVGP_FF_FOLD if m > 0 else 0


@always_inline
def svgp_ff_nbn(n: Int) -> Int:
    return (n + XN_FOLD_BLOCK - 1) // XN_FOLD_BLOCK if n > 0 else 0


# ---- workspace offsets. Flags: 0 ok_u, 1 ok_s, 2 ok_S.
# Partials (from 4): yty hi/lo (nbn each), then bsb hi/lo, trq hi/lo, lds, ldu (nbm each).
@always_inline
def _o_yty(nbn: Int, nbm: Int) -> Int:
    return 4


@always_inline
def _o_bsb(nbn: Int, nbm: Int) -> Int:
    return 4 + 2 * nbn


@always_inline
def _o_trq(nbn: Int, nbm: Int) -> Int:
    return 4 + 2 * nbn + 2 * nbm


@always_inline
def _o_lds(nbn: Int, nbm: Int) -> Int:
    return 4 + 2 * nbn + 4 * nbm


@always_inline
def _o_ldu(nbn: Int, nbm: Int) -> Int:
    return 4 + 2 * nbn + 5 * nbm


@always_inline
def _o_mat(m: Int, n: Int) -> Int:
    return 4 + 2 * svgp_ff_nbn(n) + 6 * svgp_ff_nbm(m)


# matrices, each m*m (k = 0 .. 13): A_u h/l, A_s h/l, L_u h/l, L_s h/l,
# S h/l, Q (S's factor) h/l, Sigma^-1 h/l; then the vectors (m each): x h/l,
# alpha h/l, trace terms h/l; then the column scratch, 6 m per column.
@always_inline
def _mat(w: FP, m: Int, n: Int, k: Int) -> FP:
    return w + _o_mat(m, n) + k * m * m


@always_inline
def svgp_ff_mat(w: FP, m: Int, n: Int, k: Int) -> FP:
    """`_mat` for the device drivers (MOJOLEARN_SVGP_FAST_BLKCHOL)."""
    return _mat(w, m, n, k)


@always_inline
def _vec(w: FP, m: Int, n: Int, k: Int) -> FP:
    return w + _o_mat(m, n) + 14 * m * m + k * m


@always_inline
def _scratch(w: FP, m: Int, n: Int) -> FP:
    return w + _o_mat(m, n) + 14 * m * m + 6 * m


def svgp_ff_ws_size(m: Int, n: Int) -> Int:
    """Floats of the workspace."""
    return _o_mat(m, n) + 14 * m * m + 6 * m + 6 * m * m


def svgp_ff_init_item(t: Int, kuu: FP, bh: FP, bl: FP, w: FP, m: Int, n: Int, noise: Float32, jitter: Float32):
    """A_u[t] and A_s[t], t = i*m + k; item 0 also sets the three flags."""
    var i = t // m
    var k = t - i * m
    var v = _kj(kuu, m, i, k, jitter)
    ff_st(_mat(w, m, n, 0), _mat(w, m, n, 1), t, v)
    ff_st(_mat(w, m, n, 2), _mat(w, m, n, 3), t, ff_add(v, ff_div(ff_ld(bh, bl, t), ff_of(noise))))
    if t == 0:
        w.unsafe_store(0, Float32(1))
        w.unsafe_store(1, Float32(1))
        w.unsafe_store(2, Float32(1))


@always_inline
def _chol_col(i: Int, ah: FP, al: FP, lh: FP, ll: FP, ok: FP, m: Int, j: Int):
    """Row i >= j of column j of the Cholesky factor L of A (out of place):
    the pivot s = A[j, j] - sum_k L[j, k]^2 (k ascending), recomputed by every
    item in the same order; L[i, j] = (A[i, j] - sum_k L[i, k] L[j, k]) / sqrt(s).
    Item j clears the flag on a pivot that is not positive."""
    var s = ff_ld(ah, al, j * m + j)
    for k in range(j):
        var l = ff_ld(lh, ll, j * m + k)
        s = ff_sub(s, ff_mul(l, l))
    var r = ff_sqrt(s)
    if i == j:
        if not (s.hi > 0):
            ok.unsafe_store(0, Float32(0))
        ff_st(lh, ll, j * m + j, r)
        return
    var tv = ff_ld(ah, al, i * m + j)
    for k in range(j):
        tv = ff_sub(tv, ff_mul(ff_ld(lh, ll, i * m + k), ff_ld(lh, ll, j * m + k)))
    ff_st(lh, ll, i * m + j, ff_div(tv, r))


def svgp_ff_chol_item(t: Int, w: FP, m: Int, n: Int, j: Int):
    """Column j of L_u (items 0 .. m-j-1) and of L_s (items m-j .. 2(m-j)-1)."""
    var span = m - j
    if t < span:
        _chol_col(j + t, _mat(w, m, n, 0), _mat(w, m, n, 1), _mat(w, m, n, 4), _mat(w, m, n, 5), w, m, j)
    else:
        _chol_col(j + t - span, _mat(w, m, n, 2), _mat(w, m, n, 3), _mat(w, m, n, 6), _mat(w, m, n, 7), w + 1, m, j)


def svgp_ff_chol_s_item(t: Int, w: FP, m: Int, n: Int, j: Int):
    """Column j of Q, the Cholesky factor of S."""
    _chol_col(j + t, _mat(w, m, n, 8), _mat(w, m, n, 9), _mat(w, m, n, 10), _mat(w, m, n, 11), w + 2, m, j)


def svgp_ff_column_item(j: Int, kuu: FP, bh: FP, bl: FP, cmat: FP, w: FP, m: Int, n: Int, jitter: Float32):
    """Column j: C[:, j] = Kuu'^-1 e_j - Sigma^-1 e_j, Sigma^-1[:, j],
    S[:, j] = Kuu' Sigma^-1 Kuu'[:, j] (k ascending), and the trace term
    (Kuu'^-1 B[:, j])[j]."""
    var luh = _mat(w, m, n, 4)
    var lul = _mat(w, m, n, 5)
    var lsh = _mat(w, m, n, 6)
    var lsl = _mat(w, m, n, 7)
    var sh = _mat(w, m, n, 8)
    var sl = _mat(w, m, n, 9)
    var vih = _mat(w, m, n, 12)
    var vil = _mat(w, m, n, 13)
    var uh = _scratch(w, m, n) + j * 6 * m
    var ul = uh + m
    var vh = ul + m
    var vl = vh + m
    var kh = vl + m
    var kl = kh + m
    for i in range(m):
        var e = ff_of(Float32(1) if i == j else Float32(0))
        ff_st(uh, ul, i, e)
        ff_st(vh, vl, i, e)
        ff_st(kh, kl, i, _kj(kuu, m, i, j, jitter))
    ff_chol_solve(luh, lul, m, uh, ul)
    ff_chol_solve(lsh, lsl, m, vh, vl)
    for i in range(m):
        cmat.unsafe_store(i * m + j, ff_f32(ff_sub(ff_ld(uh, ul, i), ff_ld(vh, vl, i))))
        ff_st(vih, vil, i * m + j, ff_ld(vh, vl, i))
    ff_chol_solve(lsh, lsl, m, kh, kl)
    for i in range(m):
        var s = ff_of(Float32(0))
        for k in range(m):
            s = ff_add(s, ff_mul(_kj(kuu, m, i, k, jitter), ff_ld(kh, kl, k)))
        ff_st(sh, sl, i * m + j, s)
    # tr(Kuu^-1 B): the solve against B's column j, its entry j
    for i in range(m):
        ff_st(uh, ul, i, ff_ld(bh, bl, i * m + j))
    ff_chol_solve(luh, lul, m, uh, ul)
    var tp = _vec(w, m, n, 4)
    ff_st(tp, tp + m, j, ff_ld(uh, ul, j))


def svgp_ff_col_solve_item(t: Int, kuu: FP, bh: FP, bl: FP, w: FP, xb: FP, m: Int, n: Int, jitter: Float32):
    """MOJOLEARN_SVGP_FAST_COLSPLIT (lane apple-fast-gap-kapprox2-svgp): the
    four independent triangular solves of `svgp_ff_column_item` as four
    items per column, t = which * m + j: 0 Kuu'^-1 e_j (scratch u), 1
    Sigma^-1 e_j (scratch v), 2 Sigma^-1 Kuu'[:, j] (scratch k), 3 Kuu'^-1
    B[:, j] into xb (its entry j is the trace term). The same operations in
    the same order per solve: the same words, a quarter of the chain per
    thread and four times the threads."""
    var which = t // m
    var j = t - which * m
    var base = _scratch(w, m, n) + j * 6 * m
    if which == 0:
        for i in range(m):
            ff_st(base, base + m, i, ff_of(Float32(1) if i == j else Float32(0)))
        ff_chol_solve(_mat(w, m, n, 4), _mat(w, m, n, 5), m, base, base + m)
    elif which == 1:
        var vh = base + 2 * m
        for i in range(m):
            ff_st(vh, vh + m, i, ff_of(Float32(1) if i == j else Float32(0)))
        ff_chol_solve(_mat(w, m, n, 6), _mat(w, m, n, 7), m, vh, vh + m)
    elif which == 2:
        var kh = base + 4 * m
        for i in range(m):
            ff_st(kh, kh + m, i, _kj(kuu, m, i, j, jitter))
        ff_chol_solve(_mat(w, m, n, 6), _mat(w, m, n, 7), m, kh, kh + m)
    else:
        var xh = xb + j * 2 * m
        for i in range(m):
            ff_st(xh, xh + m, i, ff_ld(bh, bl, i * m + j))
        ff_chol_solve(_mat(w, m, n, 4), _mat(w, m, n, 5), m, xh, xh + m)
        var tp = _vec(w, m, n, 4)
        ff_st(tp, tp + m, j, ff_ld(xh, xh + m, j))


def svgp_ff_col_fin_item(t: Int, kuu: FP, cmat: FP, w: FP, m: Int, n: Int, jitter: Float32):
    """MOJOLEARN_SVGP_FAST_COLSPLIT: t = i * m + j, entry (i, j) of
    `svgp_ff_column_item`'s outputs from the solves: C, Sigma^-1 and S =
    Kuu' (Sigma^-1 Kuu'[:, j]) (k ascending)."""
    var i = t // m
    var j = t - i * m
    var base = _scratch(w, m, n) + j * 6 * m
    var u = ff_ld(base, base + m, i)
    var vh = base + 2 * m
    var v = ff_ld(vh, vh + m, i)
    cmat.unsafe_store(i * m + j, ff_f32(ff_sub(u, v)))
    ff_st(_mat(w, m, n, 12), _mat(w, m, n, 13), i * m + j, v)
    var kh = base + 4 * m
    var s = ff_of(Float32(0))
    for k in range(m):
        s = ff_add(s, ff_mul(_kj(kuu, m, i, k, jitter), ff_ld(kh, kh + m, k)))
    ff_st(_mat(w, m, n, 8), _mat(w, m, n, 9), i * m + j, s)


def svgp_ff_x_item(i: Int, bvh: FP, bvl: FP, alpha: FP, w: FP, m: Int, n: Int, noise: Float32):
    """x[i] = sum_k Sigma^-1[i, k] b[k] (k ascending); alpha[i] = x[i] / noise."""
    var vih = _mat(w, m, n, 12)
    var vil = _mat(w, m, n, 13)
    var s = ff_of(Float32(0))
    for k in range(m):
        s = ff_add(s, ff_mul(ff_ld(vih, vil, i * m + k), ff_ld(bvh, bvl, k)))
    var xh = _vec(w, m, n, 0)
    ff_st(xh, xh + m, i, s)
    var a = ff_div(s, ff_of(noise))
    var ah = _vec(w, m, n, 2)
    ff_st(ah, ah + m, i, a)
    alpha.unsafe_store(i, ff_f32(a))


def svgp_ff_qmu_item(i: Int, kuu: FP, qmu: FP, w: FP, m: Int, n: Int, jitter: Float32):
    """q_mu[i] = sum_k (Kuu + jitter I)[i, k] alpha[k], k ascending."""
    var ah = _vec(w, m, n, 2)
    var s = ff_of(Float32(0))
    for k in range(m):
        s = ff_add(s, ff_mul(_kj(kuu, m, i, k, jitter), ff_ld(ah, ah + m, k)))
    qmu.unsafe_store(i, ff_f32(s))


def svgp_ff_qsqrt_item(t: Int, qsqrt: FP, w: FP, m: Int, n: Int):
    """q_sqrt = Q's lower triangle, rounded to float32."""
    var i = t // m
    var k = t - i * m
    qsqrt.unsafe_store(t, ff_f32(ff_ld(_mat(w, m, n, 10), _mat(w, m, n, 11), t)) if k <= i else Float32(0))


def svgp_ff_part_item(t: Int, y: FP, bvh: FP, bvl: FP, w: FP, m: Int, n: Int):
    """Items 0 .. nbn-1: y'y over n-block t, ascending. Items nbn ..: over
    m-block t - nbn, ascending: b'x, the trace terms, and the float32
    log-diagonals of L_s and L_u."""
    var nbn = svgp_ff_nbn(n)
    var nbm = svgp_ff_nbm(m)
    if t < nbn:
        var lo = t * XN_FOLD_BLOCK
        var hi = min(lo + XN_FOLD_BLOCK, n)
        var acc = ff_of(Float32(0))
        for i in range(lo, hi):
            var yv = y.unsafe_load(i)
            acc = ff_add(acc, two_prod(yv, yv))
        var o = w + _o_yty(nbn, nbm)
        ff_st(o, o + nbn, t, acc)
        return
    var b = t - nbn
    var lo = b * SVGP_FF_FOLD
    var hi = min(lo + SVGP_FF_FOLD, m)
    var xh = _vec(w, m, n, 0)
    var tp = _vec(w, m, n, 4)
    var luh = _mat(w, m, n, 4)
    var lul = _mat(w, m, n, 5)
    var lsh = _mat(w, m, n, 6)
    var lsl = _mat(w, m, n, 7)
    var bsb = ff_of(Float32(0))
    var trq = ff_of(Float32(0))
    var lds = Float32(0)
    var ldu = Float32(0)
    for i in range(lo, hi):
        bsb = ff_add(bsb, ff_mul(ff_ld(bvh, bvl, i), ff_ld(xh, xh + m, i)))
        trq = ff_add(trq, ff_ld(tp, tp + m, i))
        lds = ftz(lds + ftz(identical_log(ff_f32(ff_ld(lsh, lsl, i * m + i)))))
        ldu = ftz(ldu + ftz(identical_log(ff_f32(ff_ld(luh, lul, i * m + i)))))
    var ob = w + _o_bsb(nbn, nbm)
    ff_st(ob, ob + nbm, b, bsb)
    var ot = w + _o_trq(nbn, nbm)
    ff_st(ot, ot + nbm, b, trq)
    w.unsafe_store(_o_lds(nbn, nbm) + b, lds)
    w.unsafe_store(_o_ldu(nbn, nbm) + b, ldu)


#: lane apple-fast-purity (2026-10-03): the partials fold in a fixed tree so
#: the device folds them with one block of SVGP_TREE threads instead of one
#: thread (`-D MOJOLEARN_PURITY_5_OFF`: the ascending chains). Slot s folds
#: blocks s, s + SVGP_TREE, ... from zero, ascending; then halving: slot s
#: combines slot s + h for h = SVGP_TREE / 2 .. 1. The CPU column
#: (`svgp_ff_solve`) runs the same steps, so every column has the same word.
comptime SVGP_TREE = 256
comptime SVGP_TREE_ON = not is_defined["MOJOLEARN_PURITY_5_OFF"]()
#: one slot: y'y (hi, lo), b'x (hi, lo), the trace (hi, lo), log det S, log det Kuu'
comptime SV8 = SIMD[DType.float32, 8]


@always_inline
def svgp_ff_tree_comb(a: SV8, b: SV8) -> SV8:
    """Slot a combined with slot b (each fold's own operation)."""
    var y = ff_add(FF(a[0], a[1]), FF(b[0], b[1]))
    var bs = ff_add(FF(a[2], a[3]), FF(b[2], b[3]))
    var tr = ff_add(FF(a[4], a[5]), FF(b[4], b[5]))
    return SV8(y.hi, y.lo, bs.hi, bs.lo, tr.hi, tr.lo, ftz(a[6] + b[6]), ftz(a[7] + b[7]))


@always_inline
def svgp_ff_tree_slot(w: FP, s: Int, nbn: Int, nbm: Int) -> SV8:
    """Slot s: blocks s, s + SVGP_TREE, ... of every fold, ascending from zero."""
    var yty = ff_of(Float32(0))
    var oy = w + _o_yty(nbn, nbm)
    var b = s
    while b < nbn:
        yty = ff_add(yty, ff_ld(oy, oy + nbn, b))
        b += SVGP_TREE
    var bsb = ff_of(Float32(0))
    var trq = ff_of(Float32(0))
    var lds = Float32(0)
    var ldu = Float32(0)
    var ob = w + _o_bsb(nbn, nbm)
    var ot = w + _o_trq(nbn, nbm)
    b = s
    while b < nbm:
        bsb = ff_add(bsb, ff_ld(ob, ob + nbm, b))
        trq = ff_add(trq, ff_ld(ot, ot + nbm, b))
        lds = ftz(lds + w.unsafe_load(_o_lds(nbn, nbm) + b))
        ldu = ftz(ldu + w.unsafe_load(_o_ldu(nbn, nbm) + b))
        b += SVGP_TREE
    return SV8(yty.hi, yty.lo, bsb.hi, bsb.lo, trq.hi, trq.lo, lds, ldu)


@always_inline
def svgp_ff_bound(w: FP, info: FP, yty: FF, bsb: FF, trq: FF, lds: Float32, ldu: Float32,
                  nf: Float32, noise: Float32, kdiag: Float32):
    """The collapsed bound from the folded totals; info = [elbo, ok]."""
    var nz = ff_of(noise)
    var quad = ff_sub(ff_div(yty, nz), ff_div(bsb, ff_mul(nz, nz)))
    var logdet = ff_add(ff_mul(ff_of(Float32(2)), ff_sub(ff_of(lds), ff_of(ldu))),
                        ff_mul(ff_of(nf), ff_of(ftz(identical_log(noise)))))
    var trace_term = ff_div(ff_sub(ff_mul(ff_of(nf), ff_of(kdiag)), trq), nz)
    var total = ff_add(ff_add(ff_mul(ff_of(nf), ff_of(Float32(1.8378770664093453))), logdet), ff_add(quad, trace_term))
    info.unsafe_store(0, ff_f32(ff_mul(ff_of(Float32(-0.5)), total)))
    info.unsafe_store(1, Float32(1) if w.unsafe_load(2) > 0 else Float32(0))


@always_inline
def svgp_ff_failed(w: FP, info: FP) -> Bool:
    """[0, 0] into info when a factor of Kuu' or Sigma failed."""
    if not (w.unsafe_load(0) > 0 and w.unsafe_load(1) > 0):
        info.unsafe_store(0, Float32(0))
        info.unsafe_store(1, Float32(0))
        return True
    return False


def svgp_ff_fin_item(w: FP, info: FP, nbn: Int, nbm: Int, nf: Float32, noise: Float32, kdiag: Float32):
    """The partials folded (the SVGP_TREE order; ascending under
    -D MOJOLEARN_PURITY_5_OFF), the collapsed bound; info = [elbo, ok]
    ([0, 0] when a factor of Kuu' or Sigma failed)."""
    if svgp_ff_failed(w, info):
        return
    comptime if SVGP_TREE_ON:
        var sl = InlineArray[SV8, SVGP_TREE](fill=SV8(0))
        for s in range(SVGP_TREE):
            sl[s] = svgp_ff_tree_slot(w, s, nbn, nbm)
        var h = SVGP_TREE // 2
        while h > 0:
            for s in range(h):
                sl[s] = svgp_ff_tree_comb(sl[s], sl[s + h])
            h //= 2
        var t = sl[0]
        svgp_ff_bound(w, info, FF(t[0], t[1]), FF(t[2], t[3]), FF(t[4], t[5]), t[6], t[7], nf, noise, kdiag)
    else:
        var yty = ff_of(Float32(0))
        var oy = w + _o_yty(nbn, nbm)
        for b in range(nbn):
            yty = ff_add(yty, ff_ld(oy, oy + nbn, b))
        var bsb = ff_of(Float32(0))
        var trq = ff_of(Float32(0))
        var lds = Float32(0)
        var ldu = Float32(0)
        var ob = w + _o_bsb(nbn, nbm)
        var ot = w + _o_trq(nbn, nbm)
        for b in range(nbm):
            bsb = ff_add(bsb, ff_ld(ob, ob + nbm, b))
            trq = ff_add(trq, ff_ld(ot, ot + nbm, b))
            lds = ftz(lds + w.unsafe_load(_o_lds(nbn, nbm) + b))
            ldu = ftz(ldu + w.unsafe_load(_o_ldu(nbn, nbm) + b))
        svgp_ff_bound(w, info, yty, bsb, trq, lds, ldu, nf, noise, kdiag)


def svgp_ff_solve(
    kuu: FP, bh: FP, bl: FP, bvh: FP, bvl: FP, y: FP, alpha: FP, cmat: FP, qmu: FP, qsqrt: FP, info: FP,
    m: Int, n: Int, noise: Float32, jitter: Float32, kdiag: Float32,
):
    """The CPU column (CPU-only installs, verification digests): the device
    passes in the same order over the same items, one item at a time."""
    var wl = List[Float32](length=max(svgp_ff_ws_size(m, n), 1), fill=Float32(0))
    var w = FP(unsafe_from_address=Int(wl.unsafe_ptr()))
    for t in range(m * m):
        svgp_ff_init_item(t, kuu, bh, bl, w, m, n, noise, jitter)
    for j in range(m):
        for t in range(2 * (m - j)):
            svgp_ff_chol_item(t, w, m, n, j)
    for j in range(m):
        svgp_ff_column_item(j, kuu, bh, bl, cmat, w, m, n, jitter)
    for i in range(m):
        svgp_ff_x_item(i, bvh, bvl, alpha, w, m, n, noise)
    for i in range(m):
        svgp_ff_qmu_item(i, kuu, qmu, w, m, n, jitter)
    for j in range(m):
        for t in range(m - j):
            svgp_ff_chol_s_item(t, w, m, n, j)
    for t in range(m * m):
        svgp_ff_qsqrt_item(t, qsqrt, w, m, n)
    var nbn = svgp_ff_nbn(n)
    var nbm = svgp_ff_nbm(m)
    for t in range(nbn + nbm):
        svgp_ff_part_item(t, y, bvh, bvl, w, m, n)
    svgp_ff_fin_item(w, info, nbn, nbm, Float32(n), noise, kdiag)
    _ = wl^
