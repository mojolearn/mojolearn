# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE NEIGHBORS LANE'S HOST ORACLES (pass 2): every seam of
x_neighbors/items.mojo restated as plain host loops written from the
reference, NOT by calling the items, plus the UNPINNED spelling of each seam
(`variant` != 0: the alternative a separating fixture must tell apart). Each
function names its DEVIATION (IDENTITY_PATHS.md rows 120-129)."""
from std.math import fma
from std.memory import bitcast
from checks.numerics import (
    ftz, identical_mul_add, identical_mul, identical_div, identical_exp, identical_log, identical_sqrt,
    identical_tanh, identical_cos, identical_sin,
)


def _a(x: Float32, y: Float32) -> Float32:
    return ftz(ftz(x) + ftz(y))


def _s(x: Float32, y: Float32) -> Float32:
    return ftz(ftz(x) - ftz(y))


def _inf() -> Float32:
    return bitcast[DType.float32](UInt32(0x7F800000))


# DEVIATION 5206: every squared distance folds features ascending through the pinned fma.
# variant 1: descending; variant 2: unfused (a product rounded, then the add).
def o_sqdist(x: List[Float32], y: List[Float32], n: Int, m: Int, d: Int, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=n * m)
    for i in range(n):
        for j in range(m):
            var acc = Float32(0)
            for q in range(d):
                var f = d - 1 - q if variant == 1 else q
                var t = _s(x[i * d + f], y[j * d + f])
                if variant == 2:
                    acc = _a(acc, ftz(identical_mul(t, t)))
                else:
                    acc = ftz(identical_mul_add(t, t, acc))
            out.append(acc)
    return out^


def o_l1dist(x: List[Float32], y: List[Float32], n: Int, m: Int, d: Int, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=n * m)
    for i in range(n):
        for j in range(m):
            var acc = Float32(0)
            for q in range(d):
                var f = d - 1 - q if variant == 1 else q
                acc = _a(acc, abs(_s(x[i * d + f], y[j * d + f])))
            out.append(acc)
    return out^


# DEVIATION 5206 (nan_euclidean): the present-coordinate fold, then /present, then *d; -1 for none.
def o_nan_sqdist(x: List[Float32], y: List[Float32], n: Int, m: Int, d: Int, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=n * m)
    for i in range(n):
        for j in range(m):
            var acc = Float32(0)
            var p = 0
            for f in range(d):
                var a = x[i * d + f]
                var b = y[j * d + f]
                if a != a or b != b:
                    continue
                p += 1
                var t = _s(a, b)
                acc = ftz(identical_mul_add(t, t, acc))
            if p == 0:
                out.append(Float32(-1))
            elif variant == 1:
                # their other legal order: scale by d / present first
                out.append(ftz(identical_mul(acc, ftz(identical_div(Float32(d), Float32(p))))))
            else:
                out.append(ftz(identical_mul(ftz(identical_div(acc, Float32(p))), Float32(d))))
    return out^


# DEVIATION 5207: selection ascending by (value, column); an equal value keeps the lower column.
# variant 1: an equal value takes the HIGHER column.
def o_knn_select(dm: List[Float32], n: Int, m: Int, k: Int, excl: Bool, variant: Int = 0) -> Tuple[List[Float32], List[Int32]]:
    var dist = List[Float32](capacity=n * k)
    var idx = List[Int32](capacity=n * k)
    for i in range(n):
        var used = List[Bool](length=m, fill=False)
        for _ in range(k):
            var bj = -1
            var bv = _inf()
            for j in range(m):
                if used[j] or (excl and j == i):
                    continue
                var v = dm[i * m + j]
                if bj < 0 and v < bv:
                    bj = j
                    bv = v
                elif bj >= 0 and (v < bv or (variant == 1 and v == bv)):
                    bj = j
                    bv = v
            if bj >= 0:
                used[bj] = True
                dist.append(bv)
                idx.append(Int32(bj))
            else:
                dist.append(_inf())
                idx.append(Int32(-1))
    return (dist^, idx^)


# DEVIATION 5208: the kernel epilogues (sklearn metrics/pairwise.py), each on the pinned fold.
# variant 1: the dot product folded descending.
def o_kernel(x: List[Float32], y: List[Float32], n: Int, m: Int, d: Int, kind: Int, gamma: Float32,
             coef0: Float32, degree: Int, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=n * m)
    for i in range(n):
        for j in range(m):
            var dot = Float32(0)
            var sq = Float32(0)
            var l1 = Float32(0)
            var nx = Float32(0)
            var ny = Float32(0)
            var chi = Float32(0)
            for q in range(d):
                var f = d - 1 - q if variant == 1 else q
                var a = ftz(x[i * d + f])
                var b = ftz(y[j * d + f])
                dot = ftz(identical_mul_add(a, b, dot))
                var t = _s(a, b)
                sq = ftz(identical_mul_add(t, t, sq))
                l1 = _a(l1, abs(t))
                nx = ftz(identical_mul_add(a, a, nx))
                ny = ftz(identical_mul_add(b, b, ny))
                var den = _a(a, b)
                if den != Float32(0):
                    chi = _a(chi, ftz(identical_div(ftz(identical_mul(t, t)), den)))
            var r: Float32
            if kind == 0:
                r = dot
            elif kind == 1:
                var z = ftz(identical_mul_add(gamma, dot, coef0))
                r = Float32(1)
                for _ in range(degree):
                    r = ftz(identical_mul(r, z))
            elif kind == 2:
                r = ftz(identical_exp(ftz(identical_mul(-gamma, sq))))
            elif kind == 3:
                r = ftz(identical_tanh(ftz(identical_mul_add(gamma, dot, coef0))))
            elif kind == 4:
                r = ftz(identical_exp(ftz(identical_mul(-gamma, l1))))
            elif kind == 5:
                if nx == Float32(0) or ny == Float32(0):
                    r = Float32(0)
                else:
                    r = ftz(identical_div(dot, ftz(identical_mul(ftz(identical_sqrt(nx)), ftz(identical_sqrt(ny))))))
            elif kind == 7:
                r = -chi
            else:
                r = ftz(identical_exp(ftz(identical_mul(-gamma, chi))))
            out.append(r)
    return out^


# DEVIATION 5209: the dense folds, ascending (matmul over p, row / column sums, group means).
# variant 1: descending.
def o_matmul(a: List[Float32], b: List[Float32], n: Int, k: Int, m: Int, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=n * m)
    for i in range(n):
        for j in range(m):
            var acc = Float32(0)
            for q in range(k):
                var p = k - 1 - q if variant == 1 else q
                acc = ftz(identical_mul_add(ftz(a[i * k + p]), ftz(b[p * m + j]), acc))
            out.append(acc)
    return out^


def o_rowsum(a: List[Float32], n: Int, m: Int, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for i in range(n):
        var acc = Float32(0)
        for q in range(m):
            acc = _a(acc, a[i * m + (m - 1 - q if variant == 1 else q)])
        out.append(acc)
    return out^


def o_colsum(a: List[Float32], n: Int, m: Int, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=m)
    for j in range(m):
        var acc = Float32(0)
        for q in range(n):
            acc = _a(acc, a[(n - 1 - q if variant == 1 else q) * m + j])
        out.append(acc)
    return out^


def o_group_mean(x: List[Float32], lab: List[Int32], n: Int, d: Int, g: Int, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=g * d)
    for c in range(g):
        for f in range(d):
            var acc = Float32(0)
            var cnt = 0
            for q in range(n):
                var i = n - 1 - q if variant == 1 else q
                if Int(lab[i]) == c:
                    acc = _a(acc, x[i * d + f])
                    cnt += 1
            out.append(Float32(0) if cnt == 0 else ftz(identical_div(acc, Float32(cnt))))
    return out^


def o_variance(x: List[Float32], variant: Int = 0) -> Float32:
    var n = len(x)
    var acc = Float32(0)
    for q in range(n):
        acc = _a(acc, x[n - 1 - q if variant == 1 else q])
    var mean = ftz(identical_div(acc, Float32(n)))
    var ss = Float32(0)
    for q in range(n):
        var t = _s(x[n - 1 - q if variant == 1 else q], mean)
        ss = ftz(identical_mul_add(t, t, ss))
    return ftz(identical_div(ss, Float32(n)))


def o_row_normalize(a: List[Float32], n: Int, m: Int, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=n * m)
    var s = o_rowsum(a, n, m, variant)
    for i in range(n):
        var den = s[i] if s[i] != Float32(0) else Float32(1)
        for j in range(m):
            out.append(ftz(identical_div(ftz(a[i * m + j]), den)))
    return out^


def o_softmax(x: List[Float32], n: Int, c: Int, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=n * c)
    for i in range(n):
        var mx = x[i * c]
        for j in range(1, c):
            if x[i * c + j] > mx:
                mx = x[i * c + j]
        var e = List[Float32](capacity=c)
        var acc = Float32(0)
        for j in range(c):
            e.append(ftz(identical_exp(_s(x[i * c + j], mx))))
        for q in range(c):
            acc = _a(acc, e[c - 1 - q if variant == 1 else q])
        for j in range(c):
            out.append(ftz(identical_div(e[j], acc)))
    return out^


# DEVIATION 5209: predict_log_proba, (x - max) - log(sum exp(x - max)), the sum ascending.
# variant 1: the sum descending.
def o_log_softmax(x: List[Float32], n: Int, c: Int, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=n * c)
    for i in range(n):
        var mx = x[i * c]
        for j in range(1, c):
            if x[i * c + j] > mx:
                mx = x[i * c + j]
        var acc = Float32(0)
        for q in range(c):
            var j = c - 1 - q if variant == 1 else q
            acc = _a(acc, ftz(identical_exp(_s(x[i * c + j], mx))))
        var ls = ftz(identical_log(acc))
        for j in range(c):
            out.append(_s(_s(x[i * c + j], mx), ls))
    return out^


# DEVIATION 5210: LOF's reach distance max(dist, k-distance of the neighbor), the mean
# over ranks ascending, then 1 / (mean + 1e-10); the score folds the lrd ratios ascending.
# variant 1: ranks descending.
def o_lof_lrd(dist: List[Float32], idx: List[Int32], fit_dist: List[Float32], n: Int, k: Int, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for i in range(n):
        var acc = Float32(0)
        for q in range(k):
            var j = k - 1 - q if variant == 1 else q
            var kd = fit_dist[Int(idx[i * k + j]) * k + k - 1]
            var dv = dist[i * k + j]
            acc = _a(acc, kd if kd >= dv else dv)
        var mean = ftz(identical_div(acc, Float32(k)))
        out.append(ftz(identical_div(Float32(1), _a(mean, Float32(1e-10)))))
    return out^


def o_lof_score(idx: List[Int32], fit_lrd: List[Float32], lrd: List[Float32], n: Int, k: Int, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for i in range(n):
        var acc = Float32(0)
        for q in range(k):
            var j = k - 1 - q if variant == 1 else q
            acc = _a(acc, ftz(identical_div(fit_lrd[Int(idx[i * k + j])], lrd[i])))
        out.append(-ftz(identical_div(acc, Float32(k))))
    return out^


# DEVIATION 5200: libsvm's one-class Solver in float32, WSS3; equal scores resolve as
# libsvm's `>=` / `<=` scans do (the LAST index). variant 1: the FIRST index on a tie.
def o_ocsvm(q: List[Float32], cv: List[Float32], alpha0: List[Float32], n: Int, eps: Float32, max_iter: Int,
            variant: Int = 0) -> Tuple[List[Float32], Float32, Int]:
    """variant 1: the first index of equal gradient wins; variant 2: every upper
    bound 1 (the per-sample C of sample_weight ignored)."""
    var alpha = alpha0.copy()
    var c = cv.copy()
    if variant == 2:
        for i in range(n):
            c[i] = Float32(1)
    var g = List[Float32](length=n, fill=Float32(0))
    var ninf = -_inf()
    for i in range(n):
        var acc = Float32(0)
        for j in range(n):
            if alpha[j] != Float32(0):
                acc = ftz(identical_mul_add(ftz(q[i * n + j]), alpha[j], acc))
        g[i] = acc
    var it = 0
    while it < max_iter:
        var gmax = ninf
        var gi = -1
        for t in range(n):
            if alpha[t] < c[t]:
                var ng = -g[t]
                if ng > gmax or (variant != 1 and ng == gmax):
                    gmax = ng
                    gi = t
        var gmax2 = ninf
        var gj = -1
        var omin = _inf()
        if gi >= 0:
            for j in range(n):
                if alpha[j] > Float32(0):
                    var gd = _a(gmax, g[j])
                    if g[j] >= gmax2:
                        gmax2 = g[j]
                    if gd > Float32(0):
                        var quad = _s(_a(q[gi * n + gi], q[j * n + j]), ftz(identical_mul(Float32(2), q[gi * n + j])))
                        var num = ftz(identical_mul(gd, gd))
                        var ob = -ftz(identical_div(num, quad if quad > Float32(0) else Float32(1e-12)))
                        if ob < omin or (variant != 1 and ob == omin):
                            gj = j
                            omin = ob
        if gi < 0 or gj < 0 or _a(gmax, gmax2) < eps:
            break
        it += 1
        var ai0 = alpha[gi]
        var aj0 = alpha[gj]
        var quad = _s(_a(q[gi * n + gi], q[gj * n + gj]), ftz(identical_mul(Float32(2), q[gi * n + gj])))
        if quad <= Float32(0):
            quad = Float32(1e-12)
        var delta = ftz(identical_div(_s(g[gi], g[gj]), quad))
        var tot = _a(ai0, aj0)
        var ai = _s(ai0, delta)
        var aj = _a(aj0, delta)
        if tot > c[gi]:
            if ai > c[gi]:
                ai = c[gi]
                aj = _s(tot, c[gi])
        elif aj < Float32(0):
            aj = Float32(0)
            ai = tot
        if tot > c[gj]:
            if aj > c[gj]:
                aj = c[gj]
                ai = _s(tot, c[gj])
        elif ai < Float32(0):
            ai = Float32(0)
            aj = tot
        alpha[gi] = ai
        alpha[gj] = aj
        var dai = _s(ai, ai0)
        var daj = _s(aj, aj0)
        for k in range(n):
            var gk = ftz(identical_mul_add(ftz(q[gi * n + k]), dai, g[k]))
            g[k] = ftz(identical_mul_add(ftz(q[gj * n + k]), daj, gk))
    var ub = _inf()
    var lb = ninf
    var nf = 0
    var sf = Float32(0)
    for i in range(n):
        if alpha[i] >= c[i]:
            lb = g[i] if g[i] > lb else lb
        elif alpha[i] <= Float32(0):
            ub = g[i] if g[i] < ub else ub
        else:
            nf += 1
            sf = _a(sf, g[i])
    var rho = ftz(identical_div(sf, Float32(nf))) if nf > 0 else ftz(identical_mul(_a(ub, lb), Float32(0.5)))
    return (alpha^, rho, it)


# DEVIATION 5212: NearestCentroid's within-class std (rows ascending), the shrink
# (DEVIATION 5201: m*s == 0 gives deviation 0), the discriminant (sqrt then square).
# variant 1: rows descending.
def o_nc_std(x: List[Float32], lab: List[Int32], cent: List[Float32], n: Int, d: Int, c: Int, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=d)
    for f in range(d):
        var ss = Float32(0)
        for q in range(n):
            var i = n - 1 - q if variant == 1 else q
            var t = _s(x[i * d + f], cent[Int(lab[i]) * d + f])
            ss = ftz(identical_mul_add(t, t, ss))
        out.append(Float32(0) if n - c <= 0 else ftz(identical_sqrt(ftz(identical_div(ss, Float32(n - c))))))
    return out^


def o_nc_shrink(x: List[Float32], cent: List[Float32], nk: List[Float32], std: List[Float32], n: Int, d: Int,
                c: Int, med: Float32, shrink: Float32, variant: Int = 0) -> List[Float32]:
    return o_nc_shrink_dev(x, cent, nk, std, n, d, c, med, shrink, variant)[0].copy()


def o_nc_shrink_dev(x: List[Float32], cent: List[Float32], nk: List[Float32], std: List[Float32], n: Int, d: Int,
                    c: Int, med: Float32, shrink: Float32, variant: Int = 0,
                    do_shrink: Bool = True) -> Tuple[List[Float32], List[Float32]]:
    """(centroids, deviations_). variant 2: deviations_ reported before the
    soft threshold."""
    var out = List[Float32](capacity=c * d)
    var devs = List[Float32](capacity=c * d)
    for k in range(c):
        for f in range(d):
            var acc = Float32(0)
            for q in range(n):
                acc = _a(acc, x[(n - 1 - q if variant == 1 else q) * d + f])
            var dsc = ftz(identical_div(acc, Float32(n)))
            var mm = ftz(identical_sqrt(_s(ftz(identical_div(Float32(1), nk[k])), ftz(identical_div(Float32(1), Float32(n))))))
            var ms = ftz(identical_mul(mm, _a(std[f], med)))
            var dev = Float32(0) if ms == Float32(0) else ftz(identical_div(_s(cent[k * d + f], dsc), ms))
            if not do_shrink:
                devs.append(dev)
                out.append(cent[k * d + f])
                continue
            var mag = _s(abs(dev), shrink)
            if mag < Float32(0):
                mag = Float32(0)
            var sd = Float32(0)
            if dev < Float32(0):
                sd = -mag
            elif dev > Float32(0):
                sd = mag
            devs.append(dev if variant == 2 else sd)
            out.append(_a(dsc, ftz(identical_mul(ms, sd))))
    return (out^, devs^)


def o_nc_decision(q: List[Float32], cent: List[Float32], std: List[Float32], prior: List[Float32], n: Int, d: Int,
                  c: Int, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=n * c)
    for i in range(n):
        for k in range(c):
            var acc = Float32(0)
            for qq in range(d):
                var f = d - 1 - qq if variant == 1 else qq
                var a = ftz(q[i * d + f])
                var b = ftz(cent[k * d + f])
                if std[f] != Float32(0):
                    a = ftz(identical_div(a, std[f]))
                    b = ftz(identical_div(b, std[f]))
                var t = _s(a, b)
                acc = ftz(identical_mul_add(t, t, acc))
            var dd = ftz(identical_sqrt(acc))
            out.append(_a(-ftz(identical_mul(dd, dd)), ftz(identical_mul(Float32(2), ftz(identical_log(prior[k]))))))
    return out^


# DEVIATION 5202: KernelPCA's centering in KernelCenterer's order and sklearn's svd_flip
# sign rule (the FIRST row of largest |value|). variant 1: the LAST such row.
def o_kpca_center(k: List[Float32], cols: List[Float32], rows: List[Float32], all_: Float32, n: Int, m: Int,
                  variant: Int = 0) -> List[Float32]:
    """variant 1: the constant added first, (K + all) - cols - rows."""
    var out = List[Float32](capacity=n * m)
    for i in range(n):
        for j in range(m):
            if variant == 1:
                out.append(_s(_s(_a(k[i * m + j], all_), cols[j]), rows[i]))
            else:
                out.append(_a(_s(_s(k[i * m + j], cols[j]), rows[i]), all_))
    return out^


def o_svd_flip(v: List[Float32], n: Int, c: Int, variant: Int = 0) -> List[Float32]:
    var out = v.copy()
    for j in range(c):
        var best = Float32(-1)
        var at = 0
        for i in range(n):
            var a = abs(v[i * c + j])
            if a > best or (variant == 1 and a == best):
                best = a
                at = i
        if v[at * c + j] < Float32(0):
            for i in range(n):
                out[i * c + j] = -v[i * c + j]
    return out^


# DEVIATION 5203: PolynomialCountSketch's count sketch (features ascending) and the
# circular convolution summed directly, shift ascending. variant 1: shift descending.
def o_pcs(x: List[Float32], hidx: List[Int32], hbit: List[Int32], n: Int, d_in: Int, nf: Int, nc: Int, degree: Int,
          gamma: Float32, coef0: Float32, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=n * nc)
    var sg = ftz(identical_sqrt(gamma))
    var sc = ftz(identical_sqrt(coef0))
    for i in range(n):
        var accr = List[Float32](length=nc, fill=Float32(0))
        for p in range(degree):
            var sk = List[Float32](length=nc, fill=Float32(0))
            for j in range(nf):
                var v = ftz(identical_mul(sg, ftz(x[i * d_in + j]))) if j < d_in else sc
                if Int(hbit[p * nf + j]) < 0:
                    v = -v
                var h = Int(hidx[p * nf + j])
                sk[h] = _a(sk[h], v)
            if p == 0:
                accr = sk.copy()
            else:
                var nxt = List[Float32](length=nc, fill=Float32(0))
                for h in range(nc):
                    var s = Float32(0)
                    for qa in range(nc):
                        var a = nc - 1 - qa if variant == 1 else qa
                        var b = (h - a + nc) % nc
                        s = ftz(identical_mul_add(accr[a], sk[b], s))
                    nxt[h] = s
                accr = nxt^
        for h in range(nc):
            out.append(accr[h])
    return out^


# DEVIATION 5213: the chi-squared samplers' transcendental compositions: cosh as the
# mean of two portable exps, tan as sin / cos, and the skewed transform's fold ascending.
# variant 1: the factor as sqrt(step) / sqrt(cosh) instead of sqrt(step / cosh).
def o_achi2(x: List[Float32], n: Int, d: Int, steps: Int, interval: Float32, variant: Int = 0) -> List[Float32]:
    var w = d * (2 * steps - 1)
    var out = List[Float32](length=n * w, fill=Float32(0))
    var pi = Float32(3.14159265358979323846)
    for i in range(n):
        for f in range(d):
            var xv = ftz(x[i * d + f])
            if xv == Float32(0):
                continue
            out[i * w + f] = ftz(identical_sqrt(ftz(identical_mul(xv, interval))))
            var ls = ftz(identical_mul(interval, ftz(identical_log(xv))))
            var st = ftz(identical_mul(ftz(identical_mul(Float32(2), xv)), interval))
            for j in range(1, steps):
                var z = ftz(identical_mul(ftz(identical_mul(pi, Float32(j))), interval))
                var ch = ftz(identical_mul(Float32(0.5), _a(ftz(identical_exp(z)), ftz(identical_exp(-z)))))
                var fac: Float32
                if variant == 1:
                    fac = ftz(identical_div(ftz(identical_sqrt(st)), ftz(identical_sqrt(ch))))
                else:
                    fac = ftz(identical_sqrt(ftz(identical_div(st, ch))))
                var arg = ftz(identical_mul(Float32(j), ls))
                out[i * w + (2 * j - 1) * d + f] = ftz(identical_mul(fac, ftz(identical_cos(arg))))
                out[i * w + 2 * j * d + f] = ftz(identical_mul(fac, ftz(identical_sin(arg))))
    return out^


def o_skew_weights(z: List[Float32], variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=len(z))
    var pi = Float32(3.14159265358979323846)
    for i in range(len(z)):
        var zv = ftz(z[i])
        var tn = ftz(identical_div(ftz(identical_sin(zv)), ftz(identical_cos(zv))))
        if variant == 1:
            out.append(ftz(identical_div(ftz(identical_log(tn)), pi)))
        else:
            out.append(ftz(identical_mul(ftz(identical_div(Float32(1), pi)), ftz(identical_log(tn)))))
    return out^


def o_skew_transform(lx: List[Float32], w: List[Float32], off: List[Float32], n: Int, d: Int, nc: Int,
                     variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=n * nc)
    var scale = ftz(identical_div(ftz(identical_sqrt(Float32(2))), ftz(identical_sqrt(Float32(nc)))))
    for i in range(n):
        for c in range(nc):
            var acc = Float32(0)
            for q in range(d):
                var f = d - 1 - q if variant == 1 else q
                acc = ftz(identical_mul_add(ftz(lx[i * d + f]), ftz(w[f * nc + c]), acc))
            out.append(ftz(identical_mul(ftz(identical_cos(_a(acc, off[c]))), scale)))
    return out^


# DEVIATION 5214: label propagation / spreading: the hard clamp after the row
# normalization, the soft clamp alpha*L + Y (product, then add), and the normalized
# Laplacian with IN-degrees, entry / w_j / w_i. variant 1: / w_i first.
def o_ls_laplacian(a: List[Float32], n: Int, variant: Int = 0) -> List[Float32]:
    var deg = List[Float32](capacity=n)
    for j in range(n):
        var s = Float32(0)
        for k in range(n):
            if k != j:
                s = _a(s, a[k * n + j])
        deg.append(s)
    var out = List[Float32](capacity=n * n)
    for i in range(n):
        for j in range(n):
            if i == j:
                out.append(Float32(0))
                continue
            var wi = ftz(identical_sqrt(deg[i])) if deg[i] != Float32(0) else Float32(1)
            var wj = ftz(identical_sqrt(deg[j])) if deg[j] != Float32(0) else Float32(1)
            if variant == 1:
                out.append(ftz(identical_div(ftz(identical_div(ftz(a[i * n + j]), wi)), wj)))
            else:
                out.append(ftz(identical_div(ftz(identical_div(ftz(a[i * n + j]), wj)), wi)))
    return out^


def o_ls_clamp(ld: List[Float32], ys: List[Float32], alpha: Float32, variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=len(ld))
    for i in range(len(ld)):
        if variant == 1:
            out.append(ftz(identical_mul_add(alpha, ftz(ld[i]), ys[i])))
        else:
            out.append(_a(ftz(identical_mul(alpha, ftz(ld[i]))), ys[i]))
    return out^


def o_lp_clamp(ld: List[Float32], ys: List[Float32], unl: List[Int32], n: Int, c: Int, variant: Int = 0) -> List[Float32]:
    var nrm = o_row_normalize(ld, n, c, variant)
    var out = List[Float32](capacity=n * c)
    for i in range(n):
        for j in range(c):
            out.append(nrm[i * c + j] if Int(unl[i]) != 0 else ys[i * c + j])
    return out^


# DEVIATION 5215: KNNImputer's donors: nearest first by sqrt(nan_euclidean), the lower row
# on a tie, weights uniform or 1/d (zero distances only when any), the weighted mean
# folded by rank. variant 1: an equal distance takes the HIGHER row.
def o_knn_impute(x: List[Float32], fx: List[Float32], n: Int, m: Int, d: Int, k: Int, weights: Int,
                 variant: Int = 0) -> List[Float32]:
    var out = List[Float32](capacity=n * d)
    for r in range(n):
        for c in range(d):
            var v = x[r * d + c]
            if v == v:
                out.append(v)
                continue
            var cand_d = List[Float32]()
            var cand_i = List[Int]()
            var nd = 0
            for j in range(m):
                var dv = fx[j * d + c]
                if dv != dv:
                    continue
                nd += 1
                var acc = Float32(0)
                var p = 0
                for f in range(d):
                    var a = x[r * d + f]
                    var b = fx[j * d + f]
                    if a != a or b != b:
                        continue
                    p += 1
                    var t = _s(a, b)
                    acc = ftz(identical_mul_add(t, t, acc))
                if p == 0:
                    continue
                cand_d.append(ftz(identical_sqrt(ftz(identical_mul(ftz(identical_div(acc, Float32(p))), Float32(d))))))
                cand_i.append(j)
            var kk = min(k, nd)
            var sel_d = List[Float32]()
            var sel_i = List[Int]()
            var used = List[Bool](length=len(cand_d), fill=False)
            for _ in range(min(kk, len(cand_d))):
                var b = -1
                for t in range(len(cand_d)):
                    if used[t]:
                        continue
                    if b < 0 or cand_d[t] < cand_d[b] or (variant == 1 and cand_d[t] == cand_d[b]):
                        b = t
                used[b] = True
                sel_d.append(cand_d[b])
                sel_i.append(cand_i[b])
            if len(sel_d) == 0:
                var acc = Float32(0)
                var cnt = 0
                for j in range(m):
                    var dv = fx[j * d + c]
                    if dv == dv:
                        acc = _a(acc, dv)
                        cnt += 1
                out.append(ftz(identical_div(acc, Float32(cnt))) if cnt > 0 else Float32(0))
                continue
            var anyz = False
            for s in range(len(sel_d)):
                if sel_d[s] == Float32(0):
                    anyz = True
            var num = Float32(0)
            var den = Float32(0)
            for s in range(len(sel_d)):
                var w = Float32(1)
                if weights == 1:
                    if anyz:
                        w = Float32(1) if sel_d[s] == Float32(0) else Float32(0)
                    else:
                        w = ftz(identical_div(Float32(1), sel_d[s]))
                num = ftz(identical_mul_add(ftz(fx[sel_i[s] * d + c]), w, num))
                den = _a(den, w)
            out.append(ftz(identical_div(num, den)))
    return out^


# DEVIATION 5216: the PageRank step alpha * (x @ Q + dangling_sum * p) + (1 - alpha) * p,
# both folds ascending, the inner through the pinned fma. variant 1: x @ Q descending.
def o_pagerank_step(q: List[Float32], x: List[Float32], p: List[Float32], dw: List[Float32], dang: List[Int32], n: Int,
                    alpha: Float32, variant: Int = 0) -> List[Float32]:
    """variant 2: the dangling mass follows p instead of the dangling weights."""
    var out = List[Float32](capacity=n)
    for t in range(n):
        var acc = Float32(0)
        for qq in range(n):
            var i = n - 1 - qq if variant == 1 else qq
            acc = ftz(identical_mul_add(ftz(x[i]), ftz(q[i * n + t]), acc))
        var ds = Float32(0)
        for i in range(n):
            if Int(dang[i]) != 0:
                ds = _a(ds, x[i])
        var inner = ftz(identical_mul_add(ds, ftz(p[t] if variant == 2 else dw[t]), acc))
        var tel = ftz(identical_mul(_s(Float32(1), alpha), ftz(p[t])))
        out.append(ftz(identical_mul_add(alpha, inner, tel)))
    return out^


# DEVIATION 5217: weak connectivity as the min-label product (integers).
# variant 1: the max label (a different, equally legal representative).
def o_cc_step(a: List[Float32], lab: List[Int32], n: Int, variant: Int = 0) -> List[Int32]:
    var out = List[Int32](capacity=n)
    for t in range(n):
        var best = lab[t]
        for j in range(n):
            if a[t * n + j] != Float32(0) or a[j * n + t] != Float32(0):
                if (variant == 0 and lab[j] < best) or (variant == 1 and lab[j] > best):
                    best = lab[j]
        out.append(best)
    return out^


# DEVIATION 5205: the SVGP system's Cholesky (columns left to right, each fold
# ascending) and its substitutions. variant 1: the Cholesky's inner fold descending.
def o_cholesky(a: List[Float32], m: Int, variant: Int = 0) -> List[Float32]:
    var l = a.copy()
    for j in range(m):
        var s = l[j * m + j]
        for qk in range(j):
            var k = j - 1 - qk if variant == 1 else qk
            s = ftz(identical_mul_add(-l[j * m + k], l[j * m + k], s))
        var dd = ftz(identical_sqrt(s))
        l[j * m + j] = dd
        for i in range(j + 1, m):
            var t = l[i * m + j]
            for qk in range(j):
                var k = j - 1 - qk if variant == 1 else qk
                t = ftz(identical_mul_add(-l[i * m + k], l[j * m + k], t))
            l[i * m + j] = ftz(identical_div(t, dd))
        for i in range(j):
            l[i * m + j] = Float32(0)
    return l^


def _chol_solve(l: List[Float32], m: Int, b: List[Float32]) -> List[Float32]:
    var x = List[Float32](length=m, fill=Float32(0))
    for i in range(m):
        var s = b[i]
        for k in range(i):
            s = ftz(identical_mul_add(-l[i * m + k], x[k], s))
        x[i] = ftz(identical_div(s, l[i * m + i]))
    for ii in range(m):
        var i = m - 1 - ii
        var s = x[i]
        for k in range(i + 1, m):
            s = ftz(identical_mul_add(-l[k * m + i], x[k], s))
        x[i] = ftz(identical_div(s, l[i * m + i]))
    return x^


def o_svgp(kuu: List[Float32], bmat: List[Float32], b: List[Float32], y: List[Float32], m: Int, n: Int,
           noise: Float32, jitter: Float32, kdiag: Float32, variant: Int = 0) -> List[Float32]:
    """svgp_item restated: returns alpha (m) ++ C (m*m) ++ q_mu (m) ++ q_sqrt (m*m) ++ [elbo]."""
    var kj = kuu.copy()
    for i in range(m):
        kj[i * m + i] = _a(kj[i * m + i], jitter)
    var sg = List[Float32](capacity=m * m)
    for i in range(m * m):
        sg.append(_a(kj[i], ftz(identical_div(bmat[i], noise))))
    var luu = o_cholesky(kj, m, variant)
    var ls = o_cholesky(sg, m, variant)
    var alpha = _chol_solve(ls, m, b)
    for i in range(m):
        alpha[i] = ftz(identical_div(alpha[i], noise))
    var c = List[Float32](length=m * m, fill=Float32(0))
    for j in range(m):
        var e = List[Float32](length=m, fill=Float32(0))
        e[j] = Float32(1)
        var c1 = _chol_solve(luu, m, e)
        var c2 = _chol_solve(ls, m, e)
        for i in range(m):
            c[i * m + j] = _s(c1[i], c2[i])
    var qmu = List[Float32](capacity=m)
    for i in range(m):
        var s = Float32(0)
        for k in range(m):
            s = ftz(identical_mul_add(kj[i * m + k], alpha[k], s))
        qmu.append(s)
    var sm = List[Float32](length=m * m, fill=Float32(0))
    for j in range(m):
        var e = List[Float32](capacity=m)
        for i in range(m):
            e.append(kj[i * m + j])
        var col = _chol_solve(ls, m, e)
        for i in range(m):
            var s = Float32(0)
            for k in range(m):
                s = ftz(identical_mul_add(kj[i * m + k], col[k], s))
            sm[i * m + j] = s
    var qs = o_cholesky(sm, m, variant)
    var yty = Float32(0)
    for i in range(n):
        yty = ftz(identical_mul_add(y[i], y[i], yty))
    var sb = _chol_solve(ls, m, b)
    var bsb = Float32(0)
    for i in range(m):
        bsb = ftz(identical_mul_add(b[i], sb[i], bsb))
    var quad = _s(ftz(identical_div(yty, noise)), ftz(identical_div(bsb, ftz(identical_mul(noise, noise)))))
    var lds = Float32(0)
    var ldu = Float32(0)
    for i in range(m):
        lds = _a(lds, ftz(identical_log(ls[i * m + i])))
        ldu = _a(ldu, ftz(identical_log(luu[i * m + i])))
    var logdet = _a(ftz(identical_mul(Float32(2), _s(lds, ldu))), ftz(identical_mul(Float32(n), ftz(identical_log(noise)))))
    var trq = Float32(0)
    for j in range(m):
        var e = List[Float32](capacity=m)
        for i in range(m):
            e.append(bmat[i * m + j])
        trq = _a(trq, _chol_solve(luu, m, e)[j])
    var tt = ftz(identical_div(_s(ftz(identical_mul(Float32(n), kdiag)), trq), noise))
    var elbo = -ftz(identical_mul(Float32(0.5), _a(_a(ftz(identical_mul(Float32(n), Float32(1.8378770664093453))), logdet), _a(quad, tt))))
    var out = alpha.copy()
    out.extend(c.copy())
    out.extend(qmu.copy())
    out.extend(qs.copy())
    out.append(elbo)
    return out^


def _o_modularity(w: List[Float32], comm: List[Int], nn: Int, m: Float32, res: Float32) -> Float32:
    var tot = List[Float32](length=nn, fill=Float32(0))
    var inner = List[Float32](length=nn, fill=Float32(0))
    for u in range(nn):
        var cu = comm[u]
        var dg = Float32(0)
        for v in range(nn):
            var wv = w[u * nn + v]
            dg = _a(dg, wv)
            if v == u:
                dg = _a(dg, wv)
                inner[cu] = _a(inner[cu], wv)
            elif v > u and comm[v] == cu:
                inner[cu] = _a(inner[cu], wv)
        tot[cu] = _a(tot[cu], dg)
    var q = Float32(0)
    var two_m = ftz(identical_mul(Float32(2), m))
    for c in range(nn):
        var fr = ftz(identical_div(tot[c], two_m))
        q = _a(q, _s(ftz(identical_div(inner[c], m)), ftz(identical_mul(res, ftz(identical_mul(fr, fr))))))
    return q


# DEVIATION 5204: Louvain with the visit order pinned ascending and ties to the lowest
# community. variant 1: nodes visited in DESCENDING order.
def o_louvain(a: List[Float32], n: Int, max_level: Int, res: Float32, thr: Float32,
              variant: Int = 0) -> Tuple[List[Int32], Float32, Int]:
    var m = Float32(0)
    for u in range(n):
        for v in range(u, n):
            m = _a(m, a[u * n + v])
    var two_m2 = ftz(identical_mul(Float32(2), ftz(identical_mul(m, m))))
    var w = a.copy()
    var nn = n
    var labels = List[Int](capacity=n)
    var comm = List[Int](capacity=n)
    for u in range(n):
        labels.append(u)
        comm.append(u)
    var mod = _o_modularity(w, comm, nn, m, res)
    var levels = 0
    while max_level <= 0 or levels < max_level:
        comm = List[Int](capacity=nn)
        var deg = List[Float32](capacity=nn)
        for u in range(nn):
            comm.append(u)
            var dg = Float32(0)
            for v in range(nn):
                dg = _a(dg, w[u * nn + v])
            deg.append(_a(dg, w[u * nn + u]))
        var stot = deg.copy()
        var improvement = False
        var moves = 1
        while moves > 0:
            moves = 0
            for qu in range(nn):
                var u = nn - 1 - qu if variant == 1 else qu
                var cu = comm[u]
                var k2c = List[Float32](length=nn, fill=Float32(0))
                for v in range(nn):
                    if v != u and w[u * nn + v] != Float32(0):
                        k2c[comm[v]] = _a(k2c[comm[v]], w[u * nn + v])
                var du = deg[u]
                stot[cu] = _s(stot[cu], du)
                var rc = _a(-ftz(identical_div(k2c[cu], m)),
                            ftz(identical_div(ftz(identical_mul(res, ftz(identical_mul(stot[cu], du)))), two_m2)))
                var best = cu
                var bg = Float32(0)
                for c in range(nn):
                    if k2c[c] == Float32(0):
                        continue
                    var gain = _s(_a(rc, ftz(identical_div(k2c[c], m))),
                                  ftz(identical_div(ftz(identical_mul(res, ftz(identical_mul(stot[c], du)))), two_m2)))
                    if gain > bg:
                        bg = gain
                        best = c
                stot[best] = _a(stot[best], du)
                if best != cu:
                    comm[u] = best
                    moves += 1
                    improvement = True
        if levels > 0 and not improvement:
            break
        var newid = List[Int](length=nn, fill=-1)
        var nc = 0
        for c in range(nn):
            for u in range(nn):
                if comm[u] == c:
                    newid[c] = nc
                    nc += 1
                    break
        for u in range(nn):
            comm[u] = newid[comm[u]]
        for u in range(n):
            labels[u] = comm[labels[u]]
        levels += 1
        var new_mod = _o_modularity(w, comm, nn, m, res)
        if not (_s(new_mod, mod) > thr):
            break
        mod = new_mod
        var w2 = List[Float32](length=nc * nc, fill=Float32(0))
        for u in range(nn):
            for v in range(u, nn):
                var wv = w[u * nn + v]
                if wv == Float32(0):
                    continue
                var c1 = comm[u]
                var c2 = comm[v]
                w2[c1 * nc + c2] = _a(w2[c1 * nc + c2], wv)
                if c1 != c2:
                    w2[c2 * nc + c1] = _a(w2[c2 * nc + c1], wv)
        w = w2^
        nn = nc
    var lab32 = List[Int32](capacity=n)
    for u in range(n):
        lab32.append(Int32(labels[u]))
    return (lab32^, _o_modularity(a, labels, n, m, res), levels)
