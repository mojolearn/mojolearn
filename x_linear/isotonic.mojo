# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IsotonicRegression (lane/algos-linear, 2026-09-27).

Reference: scikit-learn `sklearn/isotonic.py` (`IsotonicRegression._build_y`:
rows ordered by (X, y) -- the order comes from Python --, `_make_unique`,
`isotonic_regression` with the order reversed for a decreasing fit and the
clip to [y_min, y_max], then the trim that keeps the first and last of each
run of equal fitted values) and `sklearn/_isotonic.pyx`
(`_make_unique`: equal X within float32 resolution 1e-6 pooled by weighted
mean; `_inplace_contiguous_isotonic_regression`: the single-pass
pool-adjacent-violators with backtracking), and scipy's `interp1d(kind=
'linear')` for predict (searchsorted left, clamped to [1, m-1],
slope * (x - x_lo) + y_lo). THIS PASS IS SEQUENTIAL; the brief's parallel
PAVA (a prefix-scan formulation) is pass 2's speed work. Out-of-bounds
'nan' writes the canonical quiet NaN word 0x7FC00000 (a constant, never a
computed NaN, so its bits are the same on every target).
"""
from std.memory import bitcast
from x_linear.ops import FP, IP, fa, fs, fm, fd, fmad, fmax, fmin, ld, st, ldi, sti, i2f, copy
from x_linear.team import Team

comptime OOB_NAN = 0
comptime OOB_CLIP = 1


def isotonic_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """x: sorted by (x, y), n values (d == 1); y: targets n | weights n.
    ip: [increasing, has_y_min, has_y_max]; fp: [y_min, y_max].
    res: m | X_min | X_max | xs n | ys n.
    fw: ux n | uy n | uw n.  iw: target n.
    Sequential (pool-adjacent-violators): the lead thread alone."""
    if not t.lead():
        return
    var inc = ldi(ip, 0) != 0
    var ux = 0
    var uy = n
    var uw = 2 * n
    # _make_unique
    var m = 0
    var cx = ld(x, 0)
    var cy = Float32(0)
    var cw = Float32(0)
    for j in range(n):
        var xj = ld(x, j)
        var wj = ld(y, n + j)
        if fs(xj, cx) >= Float32(1e-6):
            st(fw, ux + m, cx)
            st(fw, uw + m, cw)
            st(fw, uy + m, fd(cy, cw))
            m += 1
            cx = xj
            cw = wj
            cy = fm(ld(y, j), wj)
        else:
            cw = fa(cw, wj)
            cy = fmad(ld(y, j), wj, cy)
    st(fw, ux + m, cx)
    st(fw, uw + m, cw)
    st(fw, uy + m, fd(cy, cw))
    m += 1
    # a decreasing fit runs PAVA on the reversed sequence
    if not inc:
        for a in range(m // 2):
            var b = m - 1 - a
            var sv = ld(fw, uy + a)
            st(fw, uy + a, ld(fw, uy + b))
            st(fw, uy + b, sv)
            sv = ld(fw, uw + a)
            st(fw, uw + a, ld(fw, uw + b))
            st(fw, uw + b, sv)
    # _inplace_contiguous_isotonic_regression
    for i in range(m):
        sti(iw, i, i)
    var i = 0
    while i < m:
        var k = ldi(iw, i) + 1
        if k == m:
            break
        if ld(fw, uy + i) < ld(fw, uy + k):
            i = k
            continue
        var swy = fm(ld(fw, uw + i), ld(fw, uy + i))
        var sw = ld(fw, uw + i)
        while True:
            var prev_y = ld(fw, uy + k)
            swy = fmad(ld(fw, uw + k), ld(fw, uy + k), swy)
            sw = fa(sw, ld(fw, uw + k))
            k = ldi(iw, k) + 1
            if k == m or prev_y < ld(fw, uy + k):
                st(fw, uy + i, fd(swy, sw))
                st(fw, uw + i, sw)
                sti(iw, i, k - 1)
                sti(iw, k - 1, i)
                if i > 0:
                    i = ldi(iw, i - 1)
                break
    i = 0
    while i < m:
        var k = ldi(iw, i) + 1
        for j in range(i + 1, k):
            st(fw, uy + j, ld(fw, uy + i))
        i = k
    if not inc:
        for a in range(m // 2):
            var b = m - 1 - a
            var sv = ld(fw, uy + a)
            st(fw, uy + a, ld(fw, uy + b))
            st(fw, uy + b, sv)
    if ldi(ip, 1) != 0 or ldi(ip, 2) != 0:
        for j in range(m):
            var v = ld(fw, uy + j)
            if ldi(ip, 1) != 0:
                v = fmax(v, ld(fp, 0))
            if ldi(ip, 2) != 0:
                v = fmin(v, ld(fp, 1))
            st(fw, uy + j, v)
    # the trim: keep the ends of every run of equal values
    var kept = 0
    for j in range(m):
        var keep = True
        if j > 0 and j < m - 1:
            var v = ld(fw, uy + j)
            keep = v != ld(fw, uy + j - 1) or v != ld(fw, uy + j + 1)
        if keep:
            st(res, 3 + kept, ld(fw, ux + j))
            st(res, 3 + n + kept, ld(fw, uy + j))
            kept += 1
    st(res, 0, i2f(kept))
    st(res, 1, ld(fw, ux))
    st(res, 2, ld(fw, ux + m - 1))


def isotonic_predict(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """x: the n query points; y: xs m | ys m. ip: [m, out_of_bounds (0 nan, 1 clip)];
    fp: [X_min, X_max]. res: n predictions. Queries dealt across the team."""
    var m = ldi(ip, 0)
    var oob = ldi(ip, 1)
    # DEVIATION 5006 (IDENTITY_PATHS row 106): the constant word, never 0/0
    var nan = bitcast[DType.float32](UInt32(0x7FC00000))
    for q in range(t.tid, n, t.nt):
        var tq = ld(x, q)
        if oob == OOB_CLIP:
            tq = fmin(fmax(tq, ld(fp, 0)), ld(fp, 1))
        if m == 1:
            st(res, q, ld(y, m))
            continue
        if tq < ld(y, 0) or tq > ld(y, m - 1):
            st(res, q, nan)
            continue
        # searchsorted(xs, t, 'left'), clamped to [1, m - 1]
        var lo = 0
        var hi = m
        while lo < hi:
            var mid = (lo + hi) // 2
            if ld(y, mid) < tq:
                lo = mid + 1
            else:
                hi = mid
        var idx = lo
        if idx < 1:
            idx = 1
        if idx > m - 1:
            idx = m - 1
        var x_lo = ld(y, idx - 1)
        var x_hi = ld(y, idx)
        var y_lo = ld(y, m + idx - 1)
        var y_hi = ld(y, m + idx)
        var slope = fd(fs(y_hi, y_lo), fs(x_hi, x_lo))
        st(res, q, fmad(slope, fs(tq, x_lo), y_lo))
