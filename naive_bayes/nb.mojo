# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Naive Bayes units (the 13th family), in the prep lane's program model
(x_prep/common.mojo): one source for the CPU and every GPU.

Reference: scikit-learn `sklearn/naive_bayes.py` -- GaussianNB `_partial_fit`
(var_smoothing: epsilon_ = var_smoothing * max(var(X, axis=0))) and
`_joint_log_likelihood`; MultinomialNB `_update_feature_log_prob`;
BernoulliNB `_update_feature_log_prob` / `_joint_log_likelihood`;
ComplementNB `_update_feature_log_prob` / `_joint_log_likelihood`;
CategoricalNB `_update_feature_log_prob` / `_joint_log_likelihood`;
`_BaseDiscreteNB._update_class_log_prior`.
Float32 throughout (the reference is float64); every sum runs in ascending
index order inside one unit.
"""
from checks.numerics import ftz
from x_prep.common import FP, IP, p, ld, st
from x_prep.prims import add, sub, mul, div, logf, expf

#: 2 * pi, float32 (0x40C90FDB)
comptime TWO_PI = Float32(6.2831855)


def gnb_eps_unit(t: Int, f: FP, q: IP):
    """q = [VARROW, d, EPS, VS]; t = 0: EPS = VS * max_c VARROW[c]."""
    var m = Float32(0)
    for c in range(p(q, 1)):
        var v = ld(f, p(q, 0) + c)
        if v > m:
            m = v
    st(f, p(q, 2), mul(ld(f, p(q, 3)), m))


def gnb_params_unit(t: Int, f: FP, q: IP):
    """q = [CNT, VAR, K, d, n, EPS, PRIOR, CONST]; t = class k.
    VAR[k, :] += EPS (a variance still <= 0 is one: never 0/0 downstream);
    PRIOR[k] = CNT[k] / n; CONST[k] = log(PRIOR[k]) - 0.5 * sum_c log(2 pi VAR[k, c])."""
    var d = p(q, 3)
    var k = t
    var sl = Float32(0)
    for c in range(d):
        var v = add(ld(f, p(q, 1) + k * d + c), ld(f, p(q, 5)))
        if v <= Float32(0):
            v = Float32(1)
        st(f, p(q, 1) + k * d + c, v)
        sl = add(sl, logf(mul(TWO_PI, v)))
    var prior = div(ld(f, p(q, 0) + k), Float32(p(q, 4)))
    st(f, p(q, 6) + k, prior)
    st(f, p(q, 7) + k, sub(logf(prior), mul(Float32(0.5), sl)))


def gnb_jll_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, THETA, VAR, CONST, K, OUT]; t = i*K + k.
    OUT = CONST[k] - 0.5 * sum_c (x_c - theta_kc)^2 / var_kc."""
    var d = p(q, 2)
    var K = p(q, 6)
    var i = t // K
    var k = t % K
    var s = Float32(0)
    for c in range(d):
        var e = sub(ld(f, p(q, 0) + i * d + c), ld(f, p(q, 3) + k * d + c))
        s = add(s, div(mul(e, e), ld(f, p(q, 4) + k * d + c)))
    st(f, p(q, 7) + t, sub(ld(f, p(q, 5) + k), mul(Float32(0.5), s)))


def class_log_prior_unit(t: Int, f: FP, q: IP):
    """q = [CNT, K, OUT]; t = k: log(CNT[k]) - log(sum CNT)."""
    var tot = Float32(0)
    for k in range(p(q, 1)):
        tot = add(tot, ld(f, p(q, 0) + k))
    st(f, p(q, 2) + t, sub(logf(ld(f, p(q, 0) + t)), logf(tot)))


def mnb_params_unit(t: Int, f: FP, q: IP):
    """q = [FC, K, d, ALPHA, FLP]; t = k (MultinomialNB):
    FLP[k, c] = log(FC[k, c] + a) - log(sum_c (FC[k, c] + a))."""
    var d = p(q, 2)
    var k = t
    var a = ld(f, p(q, 3))
    var tot = Float32(0)
    for c in range(d):
        tot = add(tot, add(ld(f, p(q, 0) + k * d + c), a))
    var lt = logf(tot)
    for c in range(d):
        st(f, p(q, 4) + k * d + c, sub(logf(add(ld(f, p(q, 0) + k * d + c), a)), lt))


def bnb_params_unit(t: Int, f: FP, q: IP):
    """q = [FC, CNT, K, d, ALPHA, CLP, FLP, W, BIAS]; t = k (BernoulliNB):
    FLP = log(FC + a) - log(CNT + 2a); neg = log(1 - exp(FLP));
    W = FLP - neg; BIAS[k] = CLP[k] + sum_c neg."""
    var d = p(q, 3)
    var k = t
    var a = ld(f, p(q, 4))
    var ld_cnt = logf(add(ld(f, p(q, 1) + k), mul(Float32(2), a)))
    var sn = Float32(0)
    for c in range(d):
        var flp = sub(logf(add(ld(f, p(q, 0) + k * d + c), a)), ld_cnt)
        var neg = logf(sub(Float32(1), expf(flp)))
        st(f, p(q, 6) + k * d + c, flp)
        st(f, p(q, 7) + k * d + c, sub(flp, neg))
        sn = add(sn, neg)
    st(f, p(q, 8) + k, add(ld(f, p(q, 5) + k), sn))


def cnb_params_unit(t: Int, f: FP, q: IP):
    """q = [FC, K, d, ALPHA, NORM, FLP]; t = k (ComplementNB):
    comp = sum_j FC[j, c] + a - FC[k, c]; logged = log(comp / sum_c comp);
    FLP = logged / sum_c logged (NORM) or -logged."""
    var K = p(q, 1)
    var d = p(q, 2)
    var k = t
    var a = ld(f, p(q, 3))
    var tot = Float32(0)
    for c in range(d):
        var all_c = Float32(0)
        for j in range(K):
            all_c = add(all_c, ld(f, p(q, 0) + j * d + c))
        var comp = sub(add(all_c, a), ld(f, p(q, 0) + k * d + c))
        st(f, p(q, 5) + k * d + c, comp)
        tot = add(tot, comp)
    var sl = Float32(0)
    for c in range(d):
        var lg = logf(div(ld(f, p(q, 5) + k * d + c), tot))
        st(f, p(q, 5) + k * d + c, lg)
        sl = add(sl, lg)
    for c in range(d):
        var lg = ld(f, p(q, 5) + k * d + c)
        if p(q, 4) != 0:
            st(f, p(q, 5) + k * d + c, div(lg, sl))
        else:
            st(f, p(q, 5) + k * d + c, sub(Float32(0), lg))


def cat_params_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, K, NCAT, CMAX, CNT, ALPHA, FLP]; t = (j*K + k)*CMAX + v
    (CategoricalNB): the rows of class k whose feature j equals v, counted in
    ascending row order; FLP = log(count + a) - log(CNT[k] + a * NCAT[j]).
    Slots v >= NCAT[j] are left as they are."""
    var n = p(q, 1)
    var d = p(q, 2)
    var K = p(q, 4)
    var cmax = p(q, 6)
    var v = t % cmax
    var jk = t // cmax
    var k = jk % K
    var j = jk // K
    var ncat = Int(ld(f, p(q, 5) + j))
    if v >= ncat:
        return
    var cnt = 0
    for i in range(n):
        if Int(ld(f, p(q, 3) + i)) == k and Int(ld(f, p(q, 0) + i * d + j)) == v:
            cnt += 1
    var a = ld(f, p(q, 8))
    var den = add(ld(f, p(q, 7) + k), mul(a, Float32(ncat)))
    st(f, p(q, 9) + t, sub(logf(add(Float32(cnt), a)), logf(den)))


def cat_jll_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, FLP, K, CMAX, CLP, OUT]; t = i*K + k:
    CLP[k] + sum_j FLP[(j*K + k)*CMAX + x_ij], j ascending."""
    var d = p(q, 2)
    var K = p(q, 4)
    var cmax = p(q, 5)
    var i = t // K
    var k = t % K
    var s = Float32(0)
    for j in range(d):
        var v = Int(ld(f, p(q, 0) + i * d + j))
        s = add(s, ld(f, p(q, 3) + (j * K + k) * cmax + v))
    st(f, p(q, 7) + t, add(s, ld(f, p(q, 6) + k)))
