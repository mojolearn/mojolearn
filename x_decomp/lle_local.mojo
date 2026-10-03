# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LocallyLinearEmbedding's LOCAL steps as cells (cgr-decomp, 2026-10-03):
LTSA, Hessian and modified LLE built their stacked factor B with a Python
loop over the samples, one kit eigh (and for Hessian one orth) per sample
and per-sample float64 Python arithmetic for modified LLE. Here every
per-sample step is a cell: the device runs one thread per output (or per
sample for the k-sized serial steps), the host the same cell per index, so
the CPU column and every GPU column compute the same words. The local
eigensolves are x_decomp/rr_batch.mojo's batched round-robin Jacobi.

Layout: x (n x d) the data, idx (n x nn) the neighbor indices as exact
floats (`graph_knn`'s), V the batched eigh's vectors (problem i's nn x nn
in rows i nn .. (i + 1) nn, vectors in COLUMNS, ascending), W its values
(n x nn ascending). Every sum is one sequential loop, ascending; every
product `identical_mul` / `identical_mul_add`; every quotient `div0`.

  centered Gram (LTSA, Hessian): mu_i = the neighborhood's column means
    (`lle_mean_cell`), G_i[a, b] = sum_f (x_a - mu_i)_f (x_b - mu_i)_f.
  modified: the same with x_i in place of mu_i.
  LTSA: B row (i nn + a), column idx_b = delta_ab - (1/nn + sum_c
    U_ac U_bc), U the nc top eigenvectors (`ltsa_cell`).
  Hessian: Yi = [1, U, U_a U_b (a <= b)], Q its two-pass modified
    Gram-Schmidt basis (a column left under HS_DEP_TOL of its norm is
    zeroed: dependent), w = Q's columns past the linear ones then the
    `extra` top eigenvectors of I - Q Q^T (a second batched eigh), each
    column divided by its sum (a sum under hessian_tol counts as 1).
  modified: the regularized weights and rho per sample (`mlle_weights_cell`),
    eta = the median of rho (an exact sort), then the s_i columns
    (`mlle_rows_cell`), stored as rows i nn .. i nn + s_i - 1 of B (rows
    past s_i stay zero: B^T B, and so the SVD's right vectors, unchanged).
"""
from std.memory import bitcast

from checks.numerics import ftz, identical_mul_add
from x_decomp.cells import F32Ptr, add, div0, mul, sqrt0, sub

#: A Gram-Schmidt column left with under this fraction of its norm is
#: dependent on the earlier ones and zeroed (the kit orth's rule, DEVIATION 5318).
comptime HS_DEP_TOL = Float32(1.0e-4)


@always_inline
def _fma(a: Float32, b: Float32, c: Float32) -> Float32:
    return ftz(identical_mul_add(ftz(a), ftz(b), ftz(c)))


@always_inline
def _ix(idx: F32Ptr, i: Int, a: Int, nn: Int) -> Int:
    return Int(idx.unsafe_load(i * nn + a))


def lle_mean_cell(x: F32Ptr, idx: F32Ptr, mu: F32Ptr, i: Int, f: Int, d: Int, nn: Int):
    """mu[i, f] = the mean of feature f over sample i's neighbors."""
    var acc = Float32(0.0)
    for a in range(nn):
        acc = add(acc, x.unsafe_load(_ix(idx, i, a, nn) * d + f))
    mu.unsafe_store(i * d + f, div0(acc, Float32(nn)))


def lle_gram_cell(x: F32Ptr, idx: F32Ptr, c: F32Ptr, g: F32Ptr, i: Int, a: Int, b: Int, d: Int, nn: Int):
    """G_i[a, b] and G_i[b, a] (a <= b) of the neighbors centered at row i of
    `c` (the means for LTSA / Hessian, the data itself for modified)."""
    if a > b:
        return
    var ra = _ix(idx, i, a, nn) * d
    var rb = _ix(idx, i, b, nn) * d
    var rc = i * d
    var acc = Float32(0.0)
    for f in range(d):
        var cf = c.unsafe_load(rc + f)
        acc = _fma(sub(x.unsafe_load(ra + f), cf), sub(x.unsafe_load(rb + f), cf), acc)
    var base = i * nn * nn
    g.unsafe_store(base + a * nn + b, acc)
    g.unsafe_store(base + b * nn + a, acc)


def ltsa_cell(v: F32Ptr, idx: F32Ptr, bmat: F32Ptr, i: Int, a: Int, b: Int, n: Int, nn: Int, nc: Int):
    """B[(i nn + a), idx_b] = (I - G_i G_i^T)[a, b], G_i = [1/sqrt(nn), the nc
    top eigenvectors]."""
    var inv = div0(Float32(1.0), sqrt0(Float32(nn)))
    var acc = mul(inv, inv)
    var base = i * nn * nn
    for c in range(nc):
        var col = nn - 1 - c
        acc = _fma(v.unsafe_load(base + a * nn + col), v.unsafe_load(base + b * nn + col), acc)
    var delta = Float32(1.0) if a == b else Float32(0.0)
    bmat.unsafe_store((i * nn + a) * n + _ix(idx, i, b, nn), sub(delta, acc))


@always_inline
def hessian_ncy(nc: Int) -> Int:
    return 1 + nc + nc * (nc + 1) // 2


def hessian_q_cell(v: F32Ptr, q: F32Ptr, i: Int, nn: Int, nc: Int):
    """Sample i's Q (nn x ncy, row major at i nn ncy): Yi = [1, U, U_a U_b
    (a ascending, b from a)] then two passes of modified Gram-Schmidt per
    column (`HS_DEP_TOL` zeroes a dependent one). One thread a sample."""
    var ncy = hessian_ncy(nc)
    var qb = i * nn * ncy
    var vb = i * nn * nn
    for a in range(nn):
        q.unsafe_store(qb + a * ncy, Float32(1.0))
        for c in range(nc):
            q.unsafe_store(qb + a * ncy + 1 + c, v.unsafe_load(vb + a * nn + (nn - 1 - c)))
        var j = 1 + nc
        for p in range(nc):
            var up = v.unsafe_load(vb + a * nn + (nn - 1 - p))
            for r in range(p, nc):
                q.unsafe_store(qb + a * ncy + j, mul(v.unsafe_load(vb + a * nn + (nn - 1 - r)), up))
                j += 1
    for j in range(ncy):
        var n0 = Float32(0.0)
        for a in range(nn):
            var y = q.unsafe_load(qb + a * ncy + j)
            n0 = _fma(y, y, n0)
        for _ in range(2):
            for p in range(j):
                var r = Float32(0.0)
                for a in range(nn):
                    r = _fma(q.unsafe_load(qb + a * ncy + p), q.unsafe_load(qb + a * ncy + j), r)
                for a in range(nn):
                    q.unsafe_store(qb + a * ncy + j, _fma(-r, q.unsafe_load(qb + a * ncy + p), q.unsafe_load(qb + a * ncy + j)))
        var n1 = Float32(0.0)
        for a in range(nn):
            var y = q.unsafe_load(qb + a * ncy + j)
            n1 = _fma(y, y, n1)
        var nr = sqrt0(n1)
        var keep = nr > mul(HS_DEP_TOL, sqrt0(n0))
        for a in range(nn):
            var y = q.unsafe_load(qb + a * ncy + j)
            q.unsafe_store(qb + a * ncy + j, div0(y, nr) if keep else Float32(0.0))


def hessian_comp_cell(q: F32Ptr, cmat: F32Ptr, i: Int, a: Int, b: Int, nn: Int, nc: Int):
    """C_i = I - Q Q^T (a <= b, stored to both halves)."""
    if a > b:
        return
    var ncy = hessian_ncy(nc)
    var qb = i * nn * ncy
    var acc = Float32(0.0)
    for c in range(ncy):
        acc = _fma(q.unsafe_load(qb + a * ncy + c), q.unsafe_load(qb + b * ncy + c), acc)
    var delta = Float32(1.0) if a == b else Float32(0.0)
    var r = sub(delta, acc)
    cmat.unsafe_store(i * nn * nn + a * nn + b, r)
    cmat.unsafe_store(i * nn * nn + b * nn + a, r)


@always_inline
def _hs_w(q: F32Ptr, vc: F32Ptr, i: Int, a: Int, c: Int, nn: Int, nc: Int) -> Float32:
    """Column c of sample i's w: Q's column nc + 1 + c (c < dp), then the
    eigenvectors of I - Q Q^T from the top."""
    var ncy = hessian_ncy(nc)
    var dp = nc * (nc + 1) // 2
    if c < dp:
        return q.unsafe_load(i * nn * ncy + a * ncy + nc + 1 + c)
    return vc.unsafe_load(i * nn * nn + a * nn + (nn - 1 - (c - dp)))


def hessian_cell(
    q: F32Ptr, vc: F32Ptr, idx: F32Ptr, bmat: F32Ptr, i: Int, c: Int, n: Int, nn: Int, nc: Int, tol: Float32
):
    """B[(i ncol + c), idx_a] = w[a, c] / its column sum (a sum under tol
    counts as 1). One thread a (sample, column)."""
    var ncol = nn - 1 - nc
    var s = Float32(0.0)
    for a in range(nn):
        s = add(s, _hs_w(q, vc, i, a, c, nn, nc))
    if abs(s) < tol:
        s = Float32(1.0)
    var row = (i * ncol + c) * n
    for a in range(nn):
        bmat.unsafe_store(row + _ix(idx, i, a, nn), div0(_hs_w(q, vc, i, a, c, nn, nc), s))


@always_inline
def _ev(w: F32Ptr, i: Int, c: Int, nn: Int) -> Float32:
    """Sample i's c-th largest local eigenvalue."""
    return w.unsafe_load(i * nn + nn - 1 - c)


@always_inline
def _vd(v: F32Ptr, i: Int, a: Int, c: Int, nn: Int) -> Float32:
    """Row a of sample i's eigenvector for its c-th largest eigenvalue."""
    return v.unsafe_load(i * nn * nn + a * nn + (nn - 1 - c))


def mlle_weights_cell(
    w: F32Ptr, v: F32Ptr, wreg: F32Ptr, rho: F32Ptr, scr: F32Ptr, i: Int, nn: Int, nev: Int, nc: Int
):
    """Sample i's regularized weights (wreg row i) and rho_i (sklearn
    `_locally_linear_embedding` modified: reg = 1e-3 sum(ev), tmp = V^T 1 /
    (ev + reg), w = V tmp normalized; rho = sum(ev[nc:]) / sum(ev[:nc])).
    scr: 2 nn floats of sample i's own scratch. One thread a sample."""
    var tmp = scr + i * 3 * nn
    var wr = tmp + nn
    var se = Float32(0.0)
    for c in range(nev):
        se = add(se, _ev(w, i, c, nn))
    var reg = mul(Float32(1.0e-3), se)
    for c in range(nn):
        var cs = Float32(0.0)
        for a in range(nn):
            cs = add(cs, _vd(v, i, a, c, nn))
        tmp.unsafe_store(c, div0(cs, add(_ev(w, i, c, nn), reg)) if c < nev else div0(cs, reg))
    var tot = Float32(0.0)
    for a in range(nn):
        var acc = Float32(0.0)
        for c in range(nn):
            acc = _fma(_vd(v, i, a, c, nn), tmp.unsafe_load(c), acc)
        wr.unsafe_store(a, acc)
        tot = add(tot, acc)
    for a in range(nn):
        wreg.unsafe_store(i * nn + a, div0(wr.unsafe_load(a), tot))
    var top = Float32(0.0)
    var rest = Float32(0.0)
    for c in range(nev):
        if c < nc:
            top = add(top, _ev(w, i, c, nn))
        else:
            rest = add(rest, _ev(w, i, c, nn))
    rho.unsafe_store(i, div0(rest, top))


@always_inline
def mlle_eta(sorted_rho: F32Ptr, n: Int) -> Float32:
    """The median of the ascending rho (numpy's: the mean of the two middle
    values for even n)."""
    if n % 2 == 1:
        return sorted_rho.unsafe_load(n // 2)
    return mul(add(sorted_rho.unsafe_load(n // 2 - 1), sorted_rho.unsafe_load(n // 2)), Float32(0.5))


def mlle_rows_cell(
    w: F32Ptr, v: F32Ptr, wreg: F32Ptr, idx: F32Ptr, bmat: F32Ptr, scr: F32Ptr, eta: Float32, i: Int,
    n: Int, nn: Int, nev: Int, tol: Float32,
):
    """Sample i's s_i columns of sklearn's modified-LLE factor (the
    Householder-reflected smallest local directions plus (1 - alpha_i) w_i,
    -1 on sample i itself), stored as rows i nn .. i nn + s_i - 1 of B.
    scr: sample i's 3 nn floats (vs, h). One thread a sample."""
    # s_i = #{c < nev - 1: sum(ev) / cumsum(ev)_c - 1 < eta} + nn - nev
    var total = Float32(0.0)
    for c in range(nev):
        total = add(total, _ev(w, i, c, nn))
    var cum = Float32(0.0)
    var si = nn - nev
    for c in range(nev - 1):
        cum = add(cum, _ev(w, i, c, nn))
        if sub(div0(total, cum), Float32(1.0)) < eta:
            si += 1
    if si <= 0:
        return
    var vs = scr + i * 3 * nn
    var h = vs + nn
    # Vi[a][e] = the eigenvector of the (nn - si + e)-th largest value
    var ss = Float32(0.0)
    for e in range(si):
        var acc = Float32(0.0)
        for a in range(nn):
            acc = add(acc, _vd(v, i, a, nn - si + e, nn))
        vs.unsafe_store(e, acc)
        ss = _fma(acc, acc, ss)
    var alpha = div0(sqrt0(ss), sqrt0(Float32(si)))
    var hh = Float32(0.0)
    for e in range(si):
        var x = sub(alpha, vs.unsafe_load(e))
        h.unsafe_store(e, x)
        hh = _fma(x, x, hh)
    var nh = sqrt0(hh)
    for e in range(si):
        h.unsafe_store(e, Float32(0.0) if nh < tol else div0(h.unsafe_load(e), nh))
    var oma = sub(Float32(1.0), alpha)
    for c in range(si):
        var row = (i * nn + c) * n
        var hc = h.unsafe_load(c)
        for a in range(nn):
            var vh = Float32(0.0)
            for e in range(si):
                vh = _fma(_vd(v, i, a, nn - si + e, nn), h.unsafe_load(e), vh)
            var val = add(sub(_vd(v, i, a, nn - si + c, nn), mul(mul(Float32(2.0), vh), hc)), mul(oma, wreg.unsafe_load(i * nn + a)))
            bmat.unsafe_store(row + _ix(idx, i, a, nn), val)
        bmat.unsafe_store(row + i, Float32(-1.0))


@always_inline
def mlle_key(v: Float32) -> UInt32:
    """v's order-preserving uint32 key (the radix sort's; -0.0 before +0.0)."""
    var b = bitcast[DType.uint32](v)
    if (b & UInt32(0x80000000)) != UInt32(0):
        return ~b
    return b | UInt32(0x80000000)


@always_inline
def mlle_unkey(k: UInt32) -> Float32:
    if (k & UInt32(0x80000000)) != UInt32(0):
        return bitcast[DType.float32](k & UInt32(0x7FFFFFFF))
    return bitcast[DType.float32](~k)


def lle_apply_cell(wb: F32Ptr, idx: F32Ptr, emb: F32Ptr, dst: F32Ptr, i: Int, c: Int, nn: Int, nc: Int):
    """LLE transform: out[i, c] = sum_a W[i, a] emb[idx[i, a], c] (a ascending)."""
    var acc = Float32(0.0)
    for a in range(nn):
        acc = _fma(wb.unsafe_load(i * nn + a), emb.unsafe_load(_ix(idx, i, a, nn) * nc + c), acc)
    dst.unsafe_store(i * nc + c, acc)
