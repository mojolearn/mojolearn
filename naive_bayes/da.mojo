# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Discriminant analysis units (the 13th family), in the prep lane's program
model (x_prep/common.mojo): one source for the CPU and every GPU.

Reference: scikit-learn `sklearn/discriminant_analysis.py`:
LinearDiscriminantAnalysis `_solve_svd` (the default solver) and
QuadraticDiscriminantAnalysis `fit` / `_decision_function`. The reference's
two SVDs are replaced by the symmetric eigendecomposition of the Gram
matrices (x_prep/eigh.mojo): Z = U S V^T gives Z^T Z = V S^2 V^T, so S and V
are the same numbers up to each vector's sign, which the decision function
does not see. Arrays are laid out with row stride d and zero padding past
the data-dependent ranks, so every stage has a size known when the program is
built.
"""
from checks.numerics import ftz
from x_prep.common import FP, IP, p, ld, st
from x_prep.prims import add, sub, mul, div, logf, sqrtf


def lda_prep_unit(t: Int, f: FP, q: IP):
    """q = [CNT, MEAN, K, d, n, PRIORS, XBAR]; t = 0. PRIORS = CNT / n;
    XBAR[c] = sum_k PRIORS[k] * MEAN[k, c] (sklearn `priors_ @ means_`)."""
    var K = p(q, 2)
    var d = p(q, 3)
    for k in range(K):
        st(f, p(q, 5) + k, div(ld(f, p(q, 0) + k), Float32(p(q, 4))))
    for c in range(d):
        var s = Float32(0)
        for k in range(K):
            s = add(s, mul(ld(f, p(q, 5) + k), ld(f, p(q, 1) + k * d + c)))
        st(f, p(q, 6) + c, s)


def lda_w_unit(t: Int, f: FP, q: IP):
    """q = [VARROW, d, n, K, STD, W]; t = column. STD = sqrt(var), a zero
    std is one (sklearn `std[std == 0] = 1.0`); W = sqrt(1 / n) / STD
    (scikit-learn 1.9's `fac = 1 / n_samples`)."""
    var s = sqrtf(ld(f, p(q, 0) + t))
    if s == Float32(0):
        s = Float32(1)
    st(f, p(q, 4) + t, s)
    var fac = div(Float32(1), Float32(p(q, 2)))
    st(f, p(q, 5) + t, div(sqrtf(fac), s))


def lda_stage2_unit(t: Int, f: FP, q: IP):
    """q = [E1, V1, STD, MEAN, XBAR, PRIORS, K, d, n, META, SCAL1, G2, MS]; t = 0.
    META = [tol, rank1, rank2]. S = sqrt(max(E1, 0)); rank1 = #(S > tol);
    SCAL1[c, r] = V1[c, r] / STD[c] / S[r] (r < rank1, else 0);
    MS[k, r] = sum_c sqrt(n prior_k fac) (MEAN[k, c] - XBAR[c]) SCAL1[c, r],
    fac = 1 / (K - 1) (1 when K == 1);
    G2 = MS^T MS (d x d, zero past rank1)."""
    var K = p(q, 6)
    var d = p(q, 7)
    var n = p(q, 8)
    var META = p(q, 9)
    var tol = ld(f, META)
    var rank = 0
    for r in range(d):
        var e = ld(f, p(q, 0) + r)
        if e < Float32(0):
            e = Float32(0)
        if sqrtf(e) > tol:
            rank += 1
    st(f, META + 1, Float32(rank))
    for c in range(d):
        for r in range(d):
            var v = Float32(0)
            if r < rank:
                var e = ld(f, p(q, 0) + r)
                v = div(div(ld(f, p(q, 1) + c * d + r), ld(f, p(q, 2) + c)), sqrtf(e))
            st(f, p(q, 10) + c * d + r, v)
    var fac = Float32(1) if K == 1 else div(Float32(1), Float32(K - 1))
    for k in range(K):
        var wk = sqrtf(mul(mul(Float32(n), ld(f, p(q, 5) + k)), fac))
        for r in range(d):
            var s = Float32(0)
            for c in range(d):
                var cen = mul(wk, sub(ld(f, p(q, 3) + k * d + c), ld(f, p(q, 4) + c)))
                s = add(s, mul(cen, ld(f, p(q, 10) + c * d + r)))
            st(f, p(q, 12) + k * d + r, s)
    for a in range(d):
        for b in range(d):
            var s = Float32(0)
            for k in range(K):
                s = add(s, mul(ld(f, p(q, 12) + k * d + a), ld(f, p(q, 12) + k * d + b)))
            st(f, p(q, 11) + a * d + b, s)


def lda_stage3_unit(t: Int, f: FP, q: IP):
    """q = [E2, V2, SCAL1, MEAN, XBAR, PRIORS, K, d, META, SCAL, COEF, INTER, EVR, TMP]; t = 0.
    S2 = sqrt(max(E2, 0)); rank2 = #(S2 > tol * S2[0]); SCAL = SCAL1 V2[:, :rank2];
    c_k = (MEAN_k - XBAR) SCAL; INTER_k = -0.5 |c_k|^2 + log prior_k;
    COEF_k = c_k SCAL^T; INTER_k -= XBAR . COEF_k; EVR = S2^2 / sum S2^2."""
    var K = p(q, 6)
    var d = p(q, 7)
    var META = p(q, 8)
    var tol = ld(f, META)
    var s0 = Float32(0)
    var e0 = ld(f, p(q, 0))
    if e0 > Float32(0):
        s0 = sqrtf(e0)
    var rank2 = 0
    var tot = Float32(0)
    for r in range(d):
        var e = ld(f, p(q, 0) + r)
        if e < Float32(0):
            e = Float32(0)
        if sqrtf(e) > mul(tol, s0):
            rank2 += 1
        tot = add(tot, e)
    st(f, META + 2, Float32(rank2))
    for r in range(d):
        var e = ld(f, p(q, 0) + r)
        if e < Float32(0):
            e = Float32(0)
        st(f, p(q, 12) + r, div(e, tot) if tot > Float32(0) else Float32(0))
    for c in range(d):
        for r in range(d):
            var s = Float32(0)
            if r < rank2:
                for r1 in range(d):
                    s = add(s, mul(ld(f, p(q, 2) + c * d + r1), ld(f, p(q, 1) + r1 * d + r)))
            st(f, p(q, 9) + c * d + r, s)
    for k in range(K):
        var ss = Float32(0)
        for r in range(d):
            var s = Float32(0)
            for c in range(d):
                s = add(s, mul(sub(ld(f, p(q, 3) + k * d + c), ld(f, p(q, 4) + c)), ld(f, p(q, 9) + c * d + r)))
            st(f, p(q, 13) + k * d + r, s)
            ss = add(ss, mul(s, s))
        var inter = add(mul(Float32(-0.5), ss), logf(ld(f, p(q, 5) + k)))
        var dot = Float32(0)
        for c in range(d):
            var s = Float32(0)
            for r in range(d):
                s = add(s, mul(ld(f, p(q, 13) + k * d + r), ld(f, p(q, 9) + c * d + r)))
            st(f, p(q, 10) + k * d + c, s)
            dot = add(dot, mul(ld(f, p(q, 4) + c), s))
        st(f, p(q, 11) + k, sub(inter, dot))


def qda_cov_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, MEAN, CNT, COV]; t = (k*d + a)*d + b. The class-k
    covariance entry, divisor CNT[k] (scikit-learn 1.9 `_solve_svd`:
    scaling = S**2 / n_samples of the class), rows ascending."""
    var n = p(q, 1)
    var d = p(q, 2)
    var b = t % d
    var a = (t // d) % d
    var k = t // (d * d)
    var s = Float32(0)
    for i in range(n):
        if Int(ld(f, p(q, 3) + i)) != k:
            continue
        var ea = sub(ld(f, p(q, 0) + i * d + a), ld(f, p(q, 4) + k * d + a))
        var eb = sub(ld(f, p(q, 0) + i * d + b), ld(f, p(q, 4) + k * d + b))
        s = add(s, mul(ea, eb))
    st(f, p(q, 6) + t, div(s, ld(f, p(q, 5) + k)))


def qda_prep_unit(t: Int, f: FP, q: IP):
    """q = [EVAL, EVEC, K, d, REG, CNT, n, R, LOGC, S2OUT]; t = class k.
    S2 = (1 - reg) max(E, 0) + reg, floored at max(S2) * 2^-23 (a singular
    class covariance, where the reference divides by zero and returns NaN,
    stays finite); R[k][c, r] = V[c, r] / sqrt(S2[r]);
    LOGC[k] = log(CNT[k] / n) - 0.5 sum_r log S2[r]."""
    var d = p(q, 3)
    var k = t
    var reg = ld(f, p(q, 4))
    var E = p(q, 0) + k * d
    var V = p(q, 1) + k * d * d
    var smax = Float32(0)
    for r in range(d):
        var e = ld(f, E + r)
        if e < Float32(0):
            e = Float32(0)
        var s2 = add(mul(sub(Float32(1), reg), e), reg)
        st(f, p(q, 9) + k * d + r, s2)
        if s2 > smax:
            smax = s2
    var floor = mul(smax, Float32(1.1920929e-07))
    if smax <= Float32(0):
        floor = Float32(1)
    var sl = Float32(0)
    for r in range(d):
        var s2 = ld(f, p(q, 9) + k * d + r)
        if s2 < floor:
            s2 = floor
            st(f, p(q, 9) + k * d + r, s2)
        sl = add(sl, logf(s2))
        var inv = div(Float32(1), sqrtf(s2))
        for c in range(d):
            st(f, p(q, 7) + k * d * d + c * d + r, mul(ld(f, V + c * d + r), inv))
    var prior = div(ld(f, p(q, 5) + k), Float32(p(q, 6)))
    st(f, p(q, 8) + k, sub(logf(prior), mul(Float32(0.5), sl)))


def qda_dec_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, MEAN, R, LOGC, K, OUT]; t = i*K + k.
    OUT = LOGC[k] - 0.5 sum_r (sum_c (x_c - MEAN[k, c]) R[k][c, r])^2."""
    var d = p(q, 2)
    var K = p(q, 6)
    var i = t // K
    var k = t % K
    var norm2 = Float32(0)
    for r in range(d):
        var s = Float32(0)
        for c in range(d):
            s = add(s, mul(sub(ld(f, p(q, 0) + i * d + c), ld(f, p(q, 3) + k * d + c)),
                           ld(f, p(q, 4) + k * d * d + c * d + r)))
        norm2 = add(norm2, mul(s, s))
    st(f, p(q, 7) + t, sub(ld(f, p(q, 5) + k), mul(Float32(0.5), norm2)))
