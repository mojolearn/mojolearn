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
from x_prep.common import FP, IP, p, ld, st
from x_prep.prims import add, sub, mul, div

comptime KMEANS_MAX_ITER = 300


@always_inline
def _pos_inf() -> Float32:
    return bitcast[DType.float32](UInt32(0x7F800000))


@always_inline
def _neg_inf() -> Float32:
    return bitcast[DType.float32](UInt32(0xFF800000))


def kbins_edges_unit(t: Int, f: FP, q: IP):
    """q = [S, n, d, NB, NBMAX, STRAT, ST, EDGES, NEDGE, LAB, CEN]; t = column.
    S: columns sorted ascending (column-major, n each). NB[c]: requested bins.
    STRAT: 0 uniform, 1 quantile averaged_inverted_cdf, 2 quantile linear,
    3 kmeans. ST: col_stats rows (min at 3d, max at 4d, var at 2d).
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
    if strat == 1 or strat == 2:
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
