# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""KBinsDiscretizer units, in the prep lane's program model (x_prep/common.mojo).

Reference: scikit-learn 1.9 `sklearn/preprocessing/_discretization.py`
(`fit`: uniform = numpy linspace, quantile = numpy percentile with
quantile_method 'averaged_inverted_cdf' (default) or 'linear', kmeans = 1-D
Lloyd from the uniform bin centres, the `> 1e-8` width filter; `transform`:
searchsorted(edges[1:-1], x, side='right')), numpy
`lib/_function_base_impl.py` (`_quantile` methods, `linspace`), and
`sklearn/cluster/_kmeans.py` (`_kmeans_single_lloyd`: strict convergence on
unchanged labels, else centre shift <= tol * mean variance). The percentile
positions are exact rationals i / n_bins, so the integer/fraction split of a
position is decided in integer arithmetic.
"""
from std.memory import bitcast
from x_prep.common import FP, IP, p, ld, st, canonical_nan, is_nan
from x_prep.prims import add, sub, mul, div

comptime KMEANS_MAX_ITER = 300


@always_inline
def _pos_inf() -> Float32:
    return bitcast[DType.float32](UInt32(0x7F800000))


@always_inline
def _neg_inf() -> Float32:
    return bitcast[DType.float32](UInt32(0xFF800000))


@always_inline
def _fdiv(a: Int, b: Int) -> Int:
    """floor(a / b) for b > 0 (a may be negative)."""
    if a >= 0:
        return a // b
    return -((-a + b - 1) // b)


def _method_quantile(f: FP, S: Int, n: Int, nb: Int, i: Int, strat: Int) -> Float32:
    """numpy's percentile of the sorted column S at level i / nb for the
    quantile_method STRAT (4 inverted_cdf, 5 closest_observation,
    6 interpolated_inverted_cdf, 7 hazen, 8 weibull, 9 median_unbiased,
    10 normal_unbiased; numpy `_QuantileMethods`). The virtual index is the
    exact rational N / M (Hyndman and Fan's n*q + alpha + q*(1 - alpha - beta)
    - 1 with q = i / nb), so its floor and fraction are integer arithmetic;
    the discrete methods pick one order statistic (an index below 0 is 0),
    the continuous ones lerp as numpy's `_lerp`, clamped to the ends."""
    if strat == 4 or strat == 5:
        var N = n * i
        var idx: Int
        if strat == 4:
            idx = N // nb - 1 if N % nb == 0 else N // nb
        else:
            var num = 2 * N - 3 * nb
            var prev = _fdiv(num, 2 * nb)
            var odd = (prev - 2 * _fdiv(prev, 2)) == 1
            idx = prev if (num - prev * 2 * nb == 0 and odd) else prev + 1
        if idx < 0:
            idx = 0
        if idx > n - 1:
            idx = n - 1
        return ld(f, S + idx)
    var N: Int
    var M: Int
    if strat == 6:
        N = i * n - nb
        M = nb
    elif strat == 7:
        N = 2 * i * n - nb
        M = 2 * nb
    elif strat == 8:
        N = i * (n + 1) - nb
        M = nb
    elif strat == 9:
        N = i * (3 * n + 1) - 2 * nb
        M = 3 * nb
    else:
        N = 2 * i * (4 * n + 1) - 5 * nb
        M = 8 * nb
    if N < 0:
        return ld(f, S)
    if N >= (n - 1) * M:
        return ld(f, S + n - 1)
    var k = N // M
    var g = div(Float32(N % M), Float32(M))
    var a = ld(f, S + k)
    var b = ld(f, S + k + 1)
    var diff = sub(b, a)
    if g >= Float32(0.5):
        return sub(b, mul(diff, sub(Float32(1), g)))
    return add(a, mul(diff, g))


def kbins_edges_unit(t: Int, f: FP, q: IP):
    kbins_edges[False](t, f, q)


def _kmeans_update_host(f: FP, S: Int, LAB: Int, CEN: Int, n: Int, nb: Int) -> Float32:
    """The kmeans centre update of `kbins_edges` in one ascending walk over the
    rows (the host binding, x_prep/host/program.mojo): each row folds into
    its label's sum, so each label still sees exactly its own rows in
    ascending order, the words of the device's per-label walks. Returns the
    shift."""
    var hs = List[Float32](length=nb, fill=Float32(0))
    var hc = List[Int](length=nb, fill=0)
    for i in range(n):
        var k = Int(ld(f, LAB + i))
        if k >= 0 and k < nb:
            hs[k] = add(hs[k], ld(f, S + i))
            hc[k] += 1
    var shift = Float32(0)
    for k in range(nb):
        if hc[k] > 0:
            var nc = div(hs[k], Float32(hc[k]))
            var dc = sub(nc, ld(f, CEN + k))
            shift = add(shift, mul(dc, dc))
            st(f, CEN + k, nc)
    return shift


def kbins_edges[HOST: Bool](t: Int, f: FP, q: IP):
    """q = [S, n, d, NB, NBMAX, STRAT, ST, EDGES, NEDGE, LAB, CEN]; t = column.
    S: columns sorted ascending (column-major, n each). NB[c]: requested bins.
    STRAT: 0 uniform, 1 quantile averaged_inverted_cdf, 2 quantile linear,
    3 kmeans, 4-10 the other numpy quantile methods, 11 edges already in
    EDGES (the sample_weight units). ST: col_stats rows (min at 3d, max at 4d,
    var at 2d).
    EDGES[c*(NBMAX+1) :] the kept edges, NEDGE[c] their count (n_bins_ + 1).
    LAB (n*d) and CEN (d*NBMAX) are kmeans scratch."""
    var n = p(q, 1)
    var d = p(q, 2)
    var c = t
    var nb = Int(ld(f, p(q, 3) + c))
    var E = p(q, 7) + c * (p(q, 4) + 1)
    var strat = p(q, 5)
    var S = p(q, 0) + c * n
    var lo = ld(f, p(q, 6) + 3 * d + c)
    var hi = ld(f, p(q, 6) + 4 * d + c)
    if lo == hi:
        st(f, E, _neg_inf())
        st(f, E + 1, _pos_inf())
        st(f, p(q, 8) + c, Float32(2))
        return
    if strat == 0:
        var step = div(sub(hi, lo), Float32(nb))
        for i in range(nb):
            st(f, E + i, add(mul(Float32(i), step), lo))
        st(f, E + nb, hi)
        st(f, p(q, 8) + c, Float32(nb + 1))
        return
    if strat == 11:
        pass    # edges written by kbins_wq / kbins_wkm (sample_weight)
    elif strat == 1 or strat == 2 or strat >= 4:
        for i in range(nb + 1):
            var v: Float32
            if strat == 1:
                var hnum = n * i            # h = n * i / nb
                var k = hnum // nb
                if hnum % nb == 0:
                    var a = ld(f, S + (k - 1 if k - 1 >= 0 else 0))
                    var b = ld(f, S + (k if k < n else n - 1))
                    v = sub(b, mul(sub(b, a), Float32(0.5)))
                else:
                    v = ld(f, S + (k if k < n else n - 1))
            elif strat >= 4:
                v = _method_quantile(f, S, n, nb, i, strat)
            else:
                var pnum = (n - 1) * i      # position (n - 1) * i / nb
                var k = pnum // nb
                var g = div(Float32(pnum % nb), Float32(nb))
                var a = ld(f, S + k)
                var b = ld(f, S + (k + 1 if k + 1 < n else n - 1))
                var diff = sub(b, a)
                if g >= Float32(0.5):
                    v = sub(b, mul(diff, sub(Float32(1), g)))
                else:
                    v = add(a, mul(diff, g))
            st(f, E + i, v)
    else:
        # kmeans: centres start at the uniform bin midpoints
        var CEN = p(q, 10) + c * p(q, 4)
        var LAB = p(q, 9) + c * n
        var step = div(sub(hi, lo), Float32(nb))
        for i in range(nb):
            var e0 = add(mul(Float32(i), step), lo)
            var e1 = hi if i + 1 == nb else add(mul(Float32(i + 1), step), lo)
            st(f, CEN + i, mul(add(e1, e0), Float32(0.5)))
        for i in range(n):
            st(f, LAB + i, Float32(-1))
        var tol = mul(ld(f, p(q, 6) + 2 * d + c), Float32(1.0e-4))
        for _ in range(KMEANS_MAX_ITER):
            var changed = False
            for i in range(n):
                var x = ld(f, S + i)
                var best = 0
                var bd = abs(sub(x, ld(f, CEN)))
                for k in range(1, nb):
                    var dk = abs(sub(x, ld(f, CEN + k)))
                    if dk < bd:
                        bd = dk
                        best = k
                if Int(ld(f, LAB + i)) != best:
                    changed = True
                    st(f, LAB + i, Float32(best))
            if not changed:
                break
            var shift = Float32(0)
            comptime if HOST:
                shift = _kmeans_update_host(f, S, LAB, CEN, n, nb)
            else:
                for k in range(nb):
                    var s = Float32(0)
                    var cnt = 0
                    for i in range(n):
                        if Int(ld(f, LAB + i)) == k:
                            s = add(s, ld(f, S + i))
                            cnt += 1
                    if cnt > 0:
                        var nc = div(s, Float32(cnt))
                        var dc = sub(nc, ld(f, CEN + k))
                        shift = add(shift, mul(dc, dc))
                        st(f, CEN + k, nc)
            if shift <= tol:
                break
        # sort the centres ascending (insertion, stable)
        for i in range(1, nb):
            var v = ld(f, CEN + i)
            var j = i - 1
            while j >= 0 and ld(f, CEN + j) > v:
                st(f, CEN + j + 1, ld(f, CEN + j))
                j -= 1
            st(f, CEN + j + 1, v)
        st(f, E, lo)
        for i in range(1, nb):
            st(f, E + i, mul(add(ld(f, CEN + i), ld(f, CEN + i - 1)), Float32(0.5)))
        st(f, E + nb, hi)
    # drop edges closer than 1e-8 to the previous kept one (the first is kept)
    var kept = 1
    for i in range(1, nb + 1):
        var v = ld(f, E + i)
        if sub(v, ld(f, E + kept - 1)) > Float32(1.0e-8):
            st(f, E + kept, v)
            kept += 1
    st(f, p(q, 8) + c, Float32(kept))


def kbins_gw_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, W, GW]; t = column c. GW[c*n + k] = the sum of the
    weights W[i] of the rows whose code (the index of their value among the
    column's distinct values) is k, folded in ascending row order (the arena
    arrives zeroed)."""
    var n = p(q, 1)
    var d = p(q, 2)
    var c = t
    for i in range(n):
        var k = Int(ld(f, p(q, 0) + i * d + c))
        var o = p(q, 4) + c * n + k
        st(f, o, add(ld(f, o), ld(f, p(q, 3) + i)))


@always_inline
def _gw(f: FP, U: Int, G: Int, g: Int) -> Float32:
    """Group g's weight; a NaN value weighs 0 (the reference's NaN rule)."""
    if is_nan(ld(f, U + g)):
        return Float32(0)
    return ld(f, G + g)


def kbins_wq_unit(t: Int, f: FP, q: IP):
    """q = [UG, n, d, UCNT, NB, NBMAX, LEV, AVG, EDGES]; t = column c. The
    reference's `_weighted_percentile` (inverted_cdf, or with AVG
    averaged_inverted_cdf) over the column's distinct values U = UG[c*n :]
    (UCNT[c] of them, ascending) with their summed weights G = UG[n*d + c*n :]:
    at each level LEV[c*(NBMAX+1) + i] (percent, float32), adj = level / 100 *
    total; the value of the first group whose cumulative weight reaches adj
    (adj == 0: the first of positive weight, the reference's nextafter(0, 1);
    none: the largest value, the reference's clipped index); with AVG, when
    that cumulative exceeds adj by no more than float32 eps, the mean of it
    and the next value of positive weight (itself when there is none). A NaN
    value weighs 0."""
    var n = p(q, 1)
    var d = p(q, 2)
    var c = t
    var U = p(q, 0) + c * n
    var G = p(q, 0) + n * d + c * n
    var m = Int(ld(f, p(q, 3) + c))
    var nb = Int(ld(f, p(q, 4) + c))
    var E = p(q, 8) + c * (p(q, 5) + 1)
    var L = p(q, 6) + c * (p(q, 5) + 1)
    var total = Float32(0)
    for g in range(m):
        total = add(total, _gw(f, U, G, g))
    for i in range(nb + 1):
        var adj = mul(div(ld(f, L + i), Float32(100)), total)
        var cum = Float32(0)
        var found = -1
        for g in range(m):
            cum = add(cum, _gw(f, U, G, g))
            if (adj > Float32(0) and cum >= adj) or (adj == Float32(0) and cum > Float32(0)):
                found = g
                break
        if found < 0:
            found = m - 1
        var v = ld(f, U + found)
        if p(q, 7) != 0 and not (sub(cum, adj) > Float32(1.1920929e-07)):
            var nxt = found
            for h in range(found + 1, m):
                if _gw(f, U, G, h) > Float32(0):
                    nxt = h
                    break
            v = div(add(v, ld(f, U + nxt)), Float32(2))
        st(f, E + i, v)


def kbins_wkm_unit(t: Int, f: FP, q: IP):
    """q = [UG, n, d, UCNT, NB, NBMAX, ST, STW, EDGES, CEN, LAB]; t = column c.
    kmeans with sample_weight (the reference's KMeans(init=uniform centres,
    n_init=1).fit(column, sample_weight)): a weighted 1-D Lloyd over the
    column's distinct values U with their summed weights G (UG as kbins_wq),
    centres from the uniform midpoints of the min / max over the rows of
    nonzero weight (STW), tol from the variance of every row (ST); labels
    over every value, strict convergence on unchanged labels, else centre
    shift <= tol; a centre whose values weigh 0 stays. Edges: min, the sorted
    centres' midpoints, max."""
    var n = p(q, 1)
    var d = p(q, 2)
    var c = t
    var U = p(q, 0) + c * n
    var G = p(q, 0) + n * d + c * n
    var m = Int(ld(f, p(q, 3) + c))
    var nb = Int(ld(f, p(q, 4) + c))
    var E = p(q, 8) + c * (p(q, 5) + 1)
    var CEN = p(q, 9) + c * p(q, 5)
    var LAB = p(q, 10) + c * n
    var lo = ld(f, p(q, 7) + 3 * d + c)
    var hi = ld(f, p(q, 7) + 4 * d + c)
    if lo == hi:
        return
    var step = div(sub(hi, lo), Float32(nb))
    for i in range(nb):
        var e0 = add(mul(Float32(i), step), lo)
        var e1 = hi if i + 1 == nb else add(mul(Float32(i + 1), step), lo)
        st(f, CEN + i, mul(add(e1, e0), Float32(0.5)))
    for g in range(m):
        st(f, LAB + g, Float32(-1))
    var tol = mul(ld(f, p(q, 6) + 2 * d + c), Float32(1.0e-4))
    for _ in range(KMEANS_MAX_ITER):
        var changed = False
        for g in range(m):
            var x = ld(f, U + g)
            var best = 0
            var bd = abs(sub(x, ld(f, CEN)))
            for k in range(1, nb):
                var dk = abs(sub(x, ld(f, CEN + k)))
                if dk < bd:
                    bd = dk
                    best = k
            if Int(ld(f, LAB + g)) != best:
                changed = True
                st(f, LAB + g, Float32(best))
        if not changed:
            break
        var shift = Float32(0)
        for k in range(nb):
            var s = Float32(0)
            var w = Float32(0)
            for g in range(m):
                if Int(ld(f, LAB + g)) == k:
                    var wg = ld(f, G + g)
                    s = add(s, mul(wg, ld(f, U + g)))
                    w = add(w, wg)
            if w > Float32(0):
                var nc = div(s, w)
                var dc = sub(nc, ld(f, CEN + k))
                shift = add(shift, mul(dc, dc))
                st(f, CEN + k, nc)
        if shift <= tol:
            break
    for i in range(1, nb):
        var v = ld(f, CEN + i)
        var j = i - 1
        while j >= 0 and ld(f, CEN + j) > v:
            st(f, CEN + j + 1, ld(f, CEN + j))
            j -= 1
        st(f, CEN + j + 1, v)
    st(f, E, lo)
    for i in range(1, nb):
        st(f, E + i, mul(add(ld(f, CEN + i), ld(f, CEN + i - 1)), Float32(0.5)))
    st(f, E + nb, hi)


def kbins_codes_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, EDGES, STRIDE, NEDGE, OUT]; t = element. The number of
    interior edges <= x (searchsorted(edges[1:-1], x, side='right'))."""
    var d = p(q, 2)
    var c = t % d
    var E = p(q, 3) + c * p(q, 4)
    var ne = Int(ld(f, p(q, 5) + c))
    var x = ld(f, p(q, 0) + t)
    var k = 0
    for i in range(1, ne - 1):
        if ld(f, E + i) <= x:
            k += 1
    st(f, p(q, 6) + t, Float32(k))


def kbins_inverse_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, EDGES, STRIDE, OUT]; t = element: the reference's
    inverse_transform, the bin centre (edges[k] + edges[k+1]) * 0.5 of bin
    k = CODES[t]. A constant column's (-inf, inf) centre, and a negative
    code, are the canonical NaN."""
    var d = p(q, 2)
    var c = t % d
    var k = Int(ld(f, p(q, 0) + t))
    if k < 0:
        f.unsafe_store(p(q, 5) + t, canonical_nan())
        return
    var E = p(q, 3) + c * p(q, 4)
    var v = mul(add(ld(f, E + k), ld(f, E + k + 1)), Float32(0.5))
    if v != v:
        f.unsafe_store(p(q, 5) + t, canonical_nan())
        return
    st(f, p(q, 5) + t, v)
