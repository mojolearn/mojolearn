# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CLUSTER LANE'S HOST ORACLES (pass 2, lane/algos-cluster): every seam of
`x_cluster/bodies.mojo` restated as plain host loops written from the
reference, NOT by calling the bodies, plus the UNPINNED spelling of each seam
(the alternative a separating fixture must tell apart). Each function names
its DEVIATION (IDENTITY_PATHS.md rows 110-119)."""
from std.math import fma

from checks.numerics import ftz, identical_div, identical_exp, identical_log, identical_mul, identical_pow, identical_sqrt


# DEVIATION 5100 (fold order) and 5101 (contraction): squared distance
def oracle_sqdist(a: List[Float32], na: Int, b: List[Float32], nb: Int, d: Int, order: Int = 0) -> List[Float32]:
    """order 0: the pinned spelling (ascending, pinned product); 1: descending;
    2: fused multiply-add into the chain."""
    var out = List[Float32](capacity=na * nb)
    for i in range(na):
        for j in range(nb):
            var acc = Float32(0)
            for q in range(d):
                var f = d - 1 - q if order == 1 else q
                var t = ftz(ftz(a[i * d + f]) - ftz(b[j * d + f]))
                if order == 2:
                    acc = ftz(fma(t, t, acc))
                else:
                    acc = ftz(acc + ftz(identical_mul(t, t)))
            out.append(acc)
    return out^


# DEVIATION 5102: argmin tie, the lowest index
def oracle_nearest(dist: List[Float32], na: Int, nb: Int, highest: Bool = False) -> List[Int32]:
    var out = List[Int32](capacity=na)
    for i in range(na):
        var bi = 0
        for j in range(1, nb):
            var v = dist[i * nb + j]
            if v < dist[i * nb + bi] or (highest and v == dist[i * nb + bi]):
                bi = j
        out.append(Int32(bi))
    return out^


# DEVIATION 5103: the row order statistic, by sorting
def oracle_kth(m: List[Float32], n_rows: Int, n_cols: Int, k: Int) -> List[Float32]:
    var out = List[Float32](capacity=n_rows)
    for r in range(n_rows):
        var row = List[Float32](capacity=n_cols)
        for j in range(n_cols):
            var v = m[r * n_cols + j]
            row.append(Float32(0) if v == Float32(0) else v)
        for a in range(1, n_cols):
            var b = a
            while b > 0 and row[b - 1] > row[b]:
                var t = row[b - 1]
                row[b - 1] = row[b]
                row[b] = t
                b -= 1
        out.append(row[k - 1])
    return out^


# DEVIATION 5104: the mean-shift fold (rows ascending, one quotient)
def oracle_meanshift(
    x: List[Float32], n: Int, d: Int, bw: Float32, stop: Float32, max_iter: Int, seeds: List[Float32], ns: Int,
    reverse: Bool = False,
) -> List[Float32]:
    var out = List[Float32](capacity=ns * d)
    for s in range(ns):
        var c = List[Float32](capacity=d)
        for f in range(d):
            c.append(seeds[s * d + f])
        var completed = 0
        while True:
            var acc = List[Float32](length=d, fill=Float32(0))
            var within = 0
            for q in range(n):
                var p = n - 1 - q if reverse else q
                var dd = Float32(0)
                for f in range(d):
                    var t = ftz(ftz(c[f]) - ftz(x[p * d + f]))
                    dd = ftz(dd + ftz(identical_mul(t, t)))
                if identical_sqrt(dd) <= bw:
                    within += 1
                    for f in range(d):
                        acc[f] = ftz(acc[f] + ftz(x[p * d + f]))
            if within == 0:
                break
            var sh = Float32(0)
            for f in range(d):
                var m = ftz(identical_div(acc[f], Float32(within)))
                var t = ftz(m - c[f])
                sh = ftz(sh + ftz(identical_mul(t, t)))
                c[f] = m
            if identical_sqrt(sh) <= stop or completed == max_iter:
                break
            completed += 1
        for f in range(d):
            out.append(c[f])
    return out^


# DEVIATION 5105 (damping contraction) and 5106 (availability column fold)
def oracle_ap_step(
    s_m: List[Float32], mut a_m: List[Float32], mut r_m: List[Float32], n: Int, damping: Float32, alt: Int = 0
):
    """One sklearn iteration. alt 1: the damping as one fused multiply-add;
    alt 2: the availability column sums descending."""
    var om = ftz(Float32(1) - damping)
    for i in range(n):
        var first = Float32(0)
        var arg = 0
        for k in range(n):
            var v = ftz(a_m[i * n + k] + s_m[i * n + k])
            if k == 0 or v > first:
                first = v
                arg = k
        var second = Float32(0)
        var have = False
        for k in range(n):
            if k == arg:
                continue
            var v = ftz(a_m[i * n + k] + s_m[i * n + k])
            if not have or v > second:
                second = v
                have = True
        for k in range(n):
            var new = ftz(s_m[i * n + k] - (second if k == arg else first))
            if alt == 1:
                r_m[i * n + k] = ftz(fma(r_m[i * n + k], damping, ftz(identical_mul(new, om))))
            else:
                r_m[i * n + k] = ftz(ftz(identical_mul(r_m[i * n + k], damping)) + ftz(identical_mul(new, om)))
    for k in range(n):
        var col = Float32(0)
        for q in range(n):
            var i = n - 1 - q if alt == 2 else q
            var v = r_m[i * n + k]
            col = ftz(col + (v if (i == k or v > Float32(0)) else Float32(0)))
        for i in range(n):
            var v = r_m[i * n + k]
            var rp = v if (i == k or v > Float32(0)) else Float32(0)
            var new = ftz(col - rp)
            if i != k and new > Float32(0):
                new = Float32(0)
            a_m[i * n + k] = ftz(ftz(identical_mul(a_m[i * n + k], damping)) + ftz(identical_mul(new, om)))


# DEVIATION 5107: the bisecting tree descent, the left child on a tie
def oracle_descend(x: List[Float32], n: Int, d: Int, centers: List[Float32], nodes: List[Int32], right_on_tie: Bool = False) -> List[Int32]:
    var out = List[Int32](capacity=n)
    for i in range(n):
        var node = 0
        while nodes[node * 3] >= 0:
            var l = Int(nodes[node * 3])
            var r = Int(nodes[node * 3 + 1])
            var dl = Float32(0)
            var dr = Float32(0)
            for f in range(d):
                var tl = ftz(ftz(x[i * d + f]) - ftz(centers[l * d + f]))
                var tr = ftz(ftz(x[i * d + f]) - ftz(centers[r * d + f]))
                dl = ftz(dl + ftz(identical_mul(tl, tl)))
                dr = ftz(dr + ftz(identical_mul(tr, tr)))
            node = r if (dr < dl or (right_on_tie and dr == dl)) else l
        out.append(nodes[node * 3 + 2])
    return out^


# DEVIATION 5108: the Mahalanobis fold (j, then a ascending, pinned products)
def oracle_gauss_q(
    x: List[Float32], n: Int, d: Int, means: List[Float32], pchol: List[Float32], kc: Int, alt: Bool = False,
    descending: Bool = False,
) -> List[Float32]:
    """alt: sklearn's `X @ P - mu @ P` spelling instead of the difference
    first; descending: the inner fold over `a` walked from `j` down."""
    var out = List[Float32](capacity=n * kc)
    for i in range(n):
        for k in range(kc):
            var acc = Float32(0)
            for j in range(d):
                var y = Float32(0)
                if alt:
                    var yx = Float32(0)
                    var ym = Float32(0)
                    for a in range(j + 1):
                        var p = pchol[k * d * d + a * d + j]
                        yx = ftz(yx + ftz(identical_mul(ftz(x[i * d + a]), p)))
                        ym = ftz(ym + ftz(identical_mul(ftz(means[k * d + a]), p)))
                    y = ftz(yx - ym)
                else:
                    for aa in range(j + 1):
                        var a = j - aa if descending else aa
                        var diff = ftz(ftz(x[i * d + a]) - ftz(means[k * d + a]))
                        y = ftz(y + ftz(identical_mul(diff, pchol[k * d * d + a * d + j])))
                acc = ftz(acc + ftz(identical_mul(y, y)))
            out.append(acc)
    return out^


# DEVIATION 5109: the E-step log-sum-exp (row max, ascending exp sum, portable exp/log)
def oracle_resp(q: List[Float32], c: List[Float32], n: Int, kc: Int, descending: Bool = False) -> List[Float32]:
    var out = List[Float32](length=n * kc, fill=Float32(0))
    for i in range(n):
        var mx = Float32(0)
        for k in range(kc):
            var v = ftz(c[k] - ftz(identical_mul(Float32(0.5), q[i * kc + k])))
            out[i * kc + k] = v
            if k == 0 or v > mx:
                mx = v
        var s = Float32(0)
        for t in range(kc):
            var k = kc - 1 - t if descending else t
            s = ftz(s + ftz(identical_exp(ftz(out[i * kc + k] - mx))))
        var lse = ftz(mx + ftz(identical_log(s)))
        for k in range(kc):
            out[i * kc + k] = ftz(out[i * kc + k] - lse)
    return out^


# DEVIATION 5109 (second half): the M-step moment folds, rows ascending
def oracle_moments(
    resp: List[Float32], x: List[Float32], n: Int, d: Int, kc: Int, reg: Float32, descending: Bool = False
) -> List[Float32]:
    """nk (kc), then means (kc x d), then cov (kc x d x d), concatenated."""
    var nk = List[Float32](capacity=kc)
    for k in range(kc):
        var acc = Float32(0)
        for t in range(n):
            var i = n - 1 - t if descending else t
            acc = ftz(acc + resp[i * kc + k])
        nk.append(ftz(acc + Float32(1.1920929e-06)))
    var means = List[Float32](capacity=kc * d)
    for k in range(kc):
        for a in range(d):
            var acc = Float32(0)
            for t in range(n):
                var i = n - 1 - t if descending else t
                acc = ftz(acc + ftz(identical_mul(resp[i * kc + k], ftz(x[i * d + a]))))
            means.append(ftz(identical_div(acc, nk[k])))
    var out = nk.copy()
    for v in means:
        out.append(v)
    for k in range(kc):
        for a in range(d):
            for b in range(d):
                var acc = Float32(0)
                for t in range(n):
                    var i = n - 1 - t if descending else t
                    var da = ftz(ftz(x[i * d + a]) - means[k * d + a])
                    var db = ftz(ftz(x[i * d + b]) - means[k * d + b])
                    acc = ftz(acc + ftz(identical_mul(resp[i * kc + k], ftz(identical_mul(da, db)))))
                var v = ftz(identical_div(acc, nk[k]))
                if a == b:
                    v = ftz(v + reg)
                out.append(v)
    return out^


# DEVIATION 5111: the non-euclidean metrics (OPTICS's metric option)
def oracle_pdist(
    a: List[Float32], na: Int, b: List[Float32], nb: Int, d: Int, metric: Int, p: Float32, descending: Bool = False
) -> List[Float32]:
    """metric 1 manhattan, 2 chebyshev, 3 minkowski p, 4 cosine; each fold
    over the features ascending (descending: the unpinned order)."""
    var out = List[Float32](capacity=na * nb)
    for i in range(na):
        for j in range(nb):
            var v = Float32(0)
            var dot = Float32(0)
            var n1 = Float32(0)
            var n2 = Float32(0)
            for q in range(d):
                var f = d - 1 - q if descending else q
                var x = ftz(a[i * d + f])
                var y = ftz(b[j * d + f])
                var t = abs(ftz(x - y))
                if metric == 1:
                    v = ftz(v + t)
                elif metric == 2:
                    if t > v:
                        v = t
                elif metric == 3:
                    v = ftz(v + ftz(identical_pow(t, p)))
                else:
                    dot = ftz(dot + ftz(identical_mul(x, y)))
                    n1 = ftz(n1 + ftz(identical_mul(x, x)))
                    n2 = ftz(n2 + ftz(identical_mul(y, y)))
            if metric == 3:
                v = ftz(identical_pow(v, ftz(identical_div(Float32(1), p))))
            elif metric == 4:
                if n1 == Float32(0) or n2 == Float32(0):
                    v = Float32(1)
                else:
                    var den = ftz(identical_mul(identical_sqrt(n1), identical_sqrt(n2)))
                    v = ftz(Float32(1) - ftz(identical_div(dot, den)))
            out.append(v if v > Float32(0) else Float32(0))
    return out^
