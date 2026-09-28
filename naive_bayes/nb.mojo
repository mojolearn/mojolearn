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
    """q = [CNT, VAR, K, d, n, EPS, PRIOR, CONST, GIVEN, PIN]; t = class k.
    VAR[k, :] += EPS (a variance still <= 0 is one: never 0/0 downstream);
    PRIOR[k] = CNT[k] / n, or PIN[k] when GIVEN (the `priors` option);
    CONST[k] = log(PRIOR[k]) - 0.5 * sum_c log(2 pi VAR[k, c])."""
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
    if p(q, 8) != 0:
        prior = ld(f, p(q, 9) + k)
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
    """q = [X, n, d, Y, K, NCAT, CMAX, CNT, ALPHA, FLP, W]; t = (j*K + k)*CMAX + v
    (CategoricalNB): the rows of class k whose feature j equals v, counted in
    ascending row order (W >= 0: their weights W[i] summed instead, the
    sample_weight option; the caller passes -1 for none);
    FLP = log(count + a) - log(CNT[k] + a * NCAT[j]).
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
    var W = p(q, 10)
    var m = 0
    var cw = Float32(0)
    for i in range(n):
        if Int(ld(f, p(q, 3) + i)) == k and Int(ld(f, p(q, 0) + i * d + j)) == v:
            if W >= 0:
                cw = add(cw, ld(f, W + i))
            else:
                m += 1
    var cnt = cw if W >= 0 else Float32(m)
    var a = ld(f, p(q, 8))
    var den = add(ld(f, p(q, 7) + k), mul(a, Float32(ncat)))
    st(f, p(q, 9) + t, sub(logf(add(cnt, a)), logf(den)))


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


def log_unit(t: Int, f: FP, q: IP):
    """q = [X, OUT]; t = element: OUT = log(X) (a given class prior's log)."""
    st(f, p(q, 1) + t, logf(ld(f, p(q, 0) + t)))


def gnb_merge_unit(t: Int, f: FP, q: IP):
    """q = [OCNT, OMEAN, OVAR, NCNT, NMEAN, NVAR, K, d, CNT, MEAN, VAR, KEEP];
    t = k*d + c (GaussianNB.partial_fit, sklearn `_update_mean_variance`):
    the running class count, mean and variance (no epsilon) merged with a
    batch's. n = n_past + n_new; mean = (n_new*mu_new + n_past*mu) / n;
    ssd = n_past*var + n_new*var_new + (n_new*n_past / n) * (mu - mu_new)^2;
    var = ssd / n. A class the batch lacks keeps its values; a class not seen
    before takes the batch's. KEEP != 0 (StandardScaler.partial_fit; the
    naive Bayes callers pass 0): two exactly constant parts with the same
    value (both variances zero, equal means) keep that value and variance
    zero, the exact answer the float32 weighted mean can miss by an ulp,
    which a later merge would read as a nonzero variance (STD-1's exact-
    constant rule carried across batches)."""
    var d = p(q, 7)
    var k = t // d
    var c = t % d
    var np_ = ld(f, p(q, 0) + k)
    var nn = ld(f, p(q, 3) + k)
    var mu = ld(f, p(q, 1) + t)
    var va = ld(f, p(q, 2) + t)
    var nmu = ld(f, p(q, 4) + t)
    var nva = ld(f, p(q, 5) + t)
    var m = mu
    var v = va
    if nn != Float32(0):
        if np_ == Float32(0):
            m = nmu
            v = nva
        elif p(q, 11) != 0 and va == Float32(0) and nva == Float32(0) and mu == nmu:
            v = Float32(0)
        else:
            var tot = add(np_, nn)
            m = div(add(mul(nn, nmu), mul(np_, mu)), tot)
            var e = sub(mu, nmu)
            var ssd = add(add(mul(np_, va), mul(nn, nva)), mul(div(mul(nn, np_), tot), mul(e, e)))
            v = div(ssd, tot)
    st(f, p(q, 9) + t, m)
    st(f, p(q, 10) + t, v)
    if c == 0:
        st(f, p(q, 8) + k, add(np_, nn))


def cat_counts_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, K, NCAT, CMAX, W, OUT]; t = (j*K + k)*CMAX + v
    (CategoricalNB `_count`): the rows of class k whose feature j equals v,
    counted in ascending row order (W >= 0: their weights W[i] summed), as
    OUT[t]. Slots v >= NCAT[j] are left as they are."""
    var n = p(q, 1)
    var d = p(q, 2)
    var K = p(q, 4)
    var cmax = p(q, 6)
    var v = t % cmax
    var jk = t // cmax
    var k = jk % K
    var j = jk // K
    if v >= Int(ld(f, p(q, 5) + j)):
        return
    var W = p(q, 7)
    var m = 0
    var cw = Float32(0)
    for i in range(n):
        if Int(ld(f, p(q, 3) + i)) == k and Int(ld(f, p(q, 0) + i * d + j)) == v:
            if W >= 0:
                cw = add(cw, ld(f, W + i))
            else:
                m += 1
    st(f, p(q, 8) + t, cw if W >= 0 else Float32(m))


def cat_flp_unit(t: Int, f: FP, q: IP):
    """q = [CC, K, NCAT, CMAX, CNT, ALPHA, FLP]; t = (j*K + k)*CMAX + v
    (CategoricalNB `_update_feature_log_prob` from the category counts CC):
    FLP = log(CC + a) - log(CNT[k] + a * NCAT[j]), cat_params' arithmetic.
    Slots v >= NCAT[j] are left as they are."""
    var K = p(q, 1)
    var cmax = p(q, 3)
    var v = t % cmax
    var j = (t // cmax) // K
    var k = (t // cmax) % K
    var ncat = Int(ld(f, p(q, 2) + j))
    if v >= ncat:
        return
    var a = ld(f, p(q, 5))
    var den = add(ld(f, p(q, 4) + k), mul(a, Float32(ncat)))
    st(f, p(q, 6) + t, sub(logf(add(ld(f, p(q, 0) + t), a)), logf(den)))
