# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""TargetEncoder units, in the prep lane's program model (x_prep/common.mojo).

Reference: scikit-learn 1.9 `sklearn/preprocessing/_target_encoder.py`
(`fit_transform` cross fitting, `_fit_encodings_all`, `_transform_X_ordinal`)
and `_target_encoder_fast.pyx` (`_fit_encoding_fast`,
`_fit_encoding_fast_auto_smooth`). A fold index `fi` in [0, F) means "every
row whose fold is not fi" (the cross-fit training rows of fold fi); fi == F
means every row (plain `fit`). T target columns: 1 for a binary or
continuous target, one per class (one-vs-rest) for a multiclass one.
Where the reference's lambda is NaN (an empty category, or zero variance and
zero spread) it returns the target mean; so does this, by test, not by NaN.
"""
from x_prep.common import FP, IP, p, ld, st
from x_prep.prims import add, sub, mul, div


def te_global_unit(t: Int, f: FP, q: IP):
    """q = [Y, n, T, FOLD, META]; t = fi*T + tt. META[2t] = mean,
    META[2t+1] = population variance of target column tt over fold fi's rows."""
    var n = p(q, 1)
    var T = p(q, 2)
    var fi = t // T
    var tt = t % T
    var s = Float32(0)
    var cnt = 0
    for i in range(n):
        if Int(ld(f, p(q, 3) + i)) == fi:
            continue
        s = add(s, ld(f, p(q, 0) + i * T + tt))
        cnt += 1
    var mean = Float32(0)
    var ss = Float32(0)
    if cnt > 0:
        mean = div(s, Float32(cnt))
        for i in range(n):
            if Int(ld(f, p(q, 3) + i)) == fi:
                continue
            var e = sub(ld(f, p(q, 0) + i * T + tt), mean)
            ss = add(ss, mul(e, e))
        ss = div(ss, Float32(cnt))
    st(f, p(q, 4) + 2 * t, mean)
    st(f, p(q, 4) + 2 * t + 1, ss)


def te_enc_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, Y, T, FOLD, CMAX, NCAT, META, SMOOTH, ENC];
    t = ((fi*d + j)*CMAX + cat)*T + tt. SMOOTH < 0: the empirical Bayes
    ("auto") encoding; else (sum + s*mean) / (count + s)."""
    var n = p(q, 1)
    var d = p(q, 2)
    var T = p(q, 4)
    var cmax = p(q, 6)
    var tt = t % T
    var r = t // T
    var cat = r % cmax
    var r2 = r // cmax
    var j = r2 % d
    var fi = r2 // d
    if cat >= Int(ld(f, p(q, 7) + j)):
        return
    var ymean = ld(f, p(q, 8) + 2 * (fi * T + tt))
    var yvar = ld(f, p(q, 8) + 2 * (fi * T + tt) + 1)
    var s = Float32(0)
    var cnt = 0
    for i in range(n):
        if Int(ld(f, p(q, 5) + i)) == fi:
            continue
        if Int(ld(f, p(q, 0) + i * d + j)) != cat:
            continue
        s = add(s, ld(f, p(q, 3) + i * T + tt))
        cnt += 1
    var smooth = ld(f, p(q, 9))
    var enc = ymean
    if smooth >= Float32(0):
        var den = add(Float32(cnt), smooth)
        if den > Float32(0):
            enc = div(add(s, mul(smooth, ymean)), den)
    elif cnt > 0:
        var mean = div(s, Float32(cnt))
        var ssd = Float32(0)
        for i in range(n):
            if Int(ld(f, p(q, 5) + i)) == fi:
                continue
            if Int(ld(f, p(q, 0) + i * d + j)) != cat:
                continue
            var e = sub(ld(f, p(q, 3) + i * T + tt), mean)
            ssd = add(ssd, mul(e, e))
        var vc = mul(yvar, Float32(cnt))
        var den = add(vc, div(ssd, Float32(cnt)))
        if den > Float32(0):
            var lam = div(vc, den)
            enc = add(mul(lam, mean), mul(sub(Float32(1), lam), ymean))
    st(f, p(q, 10) + t, enc)


def te_apply_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, T, FOLD, ENC, CMAX, META, F, OUT]; t = (i*d + j)*T + tt.
    Row i reads the encodings of its own fold (FOLD < 0: fi = F, the full
    fit); an unknown code (-1) reads that fold's target mean."""
    var d = p(q, 2)
    var T = p(q, 3)
    var tt = t % T
    var ij = t // T
    var j = ij % d
    var i = ij // d
    var fi = p(q, 8)
    if p(q, 4) >= 0:
        fi = Int(ld(f, p(q, 4) + i))
    var code = Int(ld(f, p(q, 0) + i * d + j))
    var v: Float32
    if code < 0:
        v = ld(f, p(q, 7) + 2 * (fi * T + tt))
    else:
        v = ld(f, p(q, 5) + ((fi * d + j) * p(q, 6) + code) * T + tt)
    st(f, p(q, 9) + t, v)
