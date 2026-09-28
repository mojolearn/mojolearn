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
from std.memory import bitcast
from x_prep.common import FP, IP, p, ld, raw, st, ldi, sti, RUN, run_block
from checks.numerics import ftz
from x_prep.prims import add, acc_add, sub, mul, div


def te_global_unit(t: Int, f: FP, q: IP):
    """q = [Y, n, T, FOLD, META]; t = fi*T + tt. META[2t] = mean,
    META[2t+1] = population variance of target column tt over fold fi's rows
    (rows loaded RUN at a time, folded ascending)."""
    var n = p(q, 1)
    var T = p(q, 2)
    var Y = p(q, 0)
    var FO = p(q, 3)
    var fi = t // T
    var tt = t % T
    var s = Float32(0)
    var cnt = 0
    var full = n - n % RUN
    for i0 in range(0, full, RUN):
        var bf = run_block[RUN](f, FO + i0, 1)
        var by = run_block[RUN](f, Y + i0 * T + tt, T)
        comptime for u in range(RUN):
            if Int(ftz(bf[u])) != fi:
                s = acc_add(s, ftz(by[u]))
                cnt += 1
    for i in range(full, n):
        if Int(ld(f, FO + i)) == fi:
            continue
        s = acc_add(s, ld(f, Y + i * T + tt))
        cnt += 1
    var mean = Float32(0)
    var ss = Float32(0)
    if cnt > 0:
        mean = div(s, Float32(cnt))
        for i0 in range(0, full, RUN):
            var bf = run_block[RUN](f, FO + i0, 1)
            var by = run_block[RUN](f, Y + i0 * T + tt, T)
            comptime for u in range(RUN):
                if Int(ftz(bf[u])) != fi:
                    var e = sub(ftz(by[u]), mean)
                    ss = acc_add(ss, mul(e, e))
        for i in range(full, n):
            if Int(ld(f, FO + i)) == fi:
                continue
            var e = sub(ld(f, Y + i * T + tt), mean)
            ss = acc_add(ss, mul(e, e))
        ss = div(ss, Float32(cnt))
    st(f, p(q, 4) + 2 * t, mean)
    st(f, p(q, 4) + 2 * t + 1, ss)


def te_bucket_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, CMAX, START, ROWS]; t = column j (lane prep-apple).
    The rows of each category of column j, ascending: START[j*(CMAX+1) + c]
    .. START[j*(CMAX+1) + c + 1] index ROWS[j*n + ...], which holds the row
    numbers whose code is c, in row order (int words). A code outside
    [0, CMAX) is in no bucket. `te_enc` then walks one category's rows
    instead of every row: the same rows in the same order."""
    var n = p(q, 1)
    var d = p(q, 2)
    var cmax = p(q, 3)
    var C = p(q, 0)
    var S = p(q, 4) + t * (cmax + 1)
    var R = p(q, 5) + t * n
    for c in range(cmax + 1):
        sti(f, S + c, 0)
    var full = n - n % RUN
    for i0 in range(0, full, RUN):
        var bc = run_block[RUN](f, C + i0 * d + t, d)
        comptime for u in range(RUN):
            var code = Int(ftz(bc[u]))
            if code >= 0 and code < cmax:
                sti(f, S + code + 1, ldi(f, S + code + 1) + 1)
    for i in range(full, n):
        var code = Int(ld(f, C + i * d + t))
        if code >= 0 and code < cmax:
            sti(f, S + code + 1, ldi(f, S + code + 1) + 1)
    for c in range(cmax):
        sti(f, S + c + 1, ldi(f, S + c + 1) + ldi(f, S + c))
    # scatter: START[c] is advanced as its bucket fills, then restored
    for i0 in range(0, full, RUN):
        var bc = run_block[RUN](f, C + i0 * d + t, d)
        comptime for u in range(RUN):
            var code = Int(ftz(bc[u]))
            if code >= 0 and code < cmax:
                var at = ldi(f, S + code)
                sti(f, R + at, i0 + u)
                sti(f, S + code, at + 1)
    for i in range(full, n):
        var code = Int(ld(f, C + i * d + t))
        if code >= 0 and code < cmax:
            var at = ldi(f, S + code)
            sti(f, R + at, i)
            sti(f, S + code, at + 1)
    for c in range(cmax, 0, -1):
        sti(f, S + c, ldi(f, S + c - 1))
    sti(f, S, 0)


def te_enc_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, Y, T, FOLD, CMAX, NCAT, META, SMOOTH, ENC, BK, ROWS, GB];
    t = ((fi*d + j)*CMAX + cat)*T + tt. SMOOTH < 0: the empirical Bayes
    ("auto") encoding; else (sum + s*mean) / (count + s). With BK > 0 the
    rows walked are category cat's bucket (`te_bucket`, ascending), else
    every row; either way the rows of cat outside fold fi, in row order."""
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
    # BK = START + 1 of `te_bucket` (0: no buckets, every row is scanned)
    var bk = p(q, 11)
    var lo = 0
    var hi = n
    var R = 0
    if bk > 0:
        var S = bk - 1 + j * (cmax + 1)
        lo = ldi(f, S + cat)
        hi = ldi(f, S + cat + 1)
        R = p(q, 12) + j * n
    var smooth = ld(f, p(q, 9))
    var mean = Float32(0)
    var ssd = Float32(0)
    # GB = `te_gather`'s offset + 1 (lane prep-apple2): the bucket's folds and
    # targets in bucket order, streamed RUN at a time; the same rows in the
    # same order, so the same words
    var gb = p(q, 13) if bk > 0 else 0
    if gb > 0:
        var BF = gb - 1 + j * n
        var BY = gb - 1 + d * n + (j * T + tt) * n
        var cut = lo + (hi - lo) - (hi - lo) % RUN
        for k0 in range(lo, cut, RUN):
            var bf = run_block[RUN](f, BF + k0, 1)
            var by = run_block[RUN](f, BY + k0, 1)
            comptime for u in range(RUN):
                if Int(ftz(bf[u])) != fi:
                    s = acc_add(s, ftz(by[u]))
                    cnt += 1
        for k in range(cut, hi):
            if Int(ld(f, BF + k)) == fi:
                continue
            s = acc_add(s, ld(f, BY + k))
            cnt += 1
        if smooth < Float32(0) and cnt > 0:
            mean = div(s, Float32(cnt))
            for k0 in range(lo, cut, RUN):
                var bf = run_block[RUN](f, BF + k0, 1)
                var by = run_block[RUN](f, BY + k0, 1)
                comptime for u in range(RUN):
                    if Int(ftz(bf[u])) != fi:
                        var e = sub(ftz(by[u]), mean)
                        ssd = acc_add(ssd, mul(e, e))
            for k in range(cut, hi):
                if Int(ld(f, BF + k)) == fi:
                    continue
                var e = sub(ld(f, BY + k), mean)
                ssd = acc_add(ssd, mul(e, e))
        st(f, p(q, 10) + t, te_value(ymean, yvar, smooth, s, cnt, mean, ssd))
        return
    for k in range(lo, hi):
        var i = k
        if bk > 0:
            i = ldi(f, R + k)
        if Int(ld(f, p(q, 5) + i)) == fi:
            continue
        if bk == 0 and Int(ld(f, p(q, 0) + i * d + j)) != cat:
            continue
        s = add(s, ld(f, p(q, 3) + i * T + tt))
        cnt += 1
    if smooth < Float32(0) and cnt > 0:
        mean = div(s, Float32(cnt))
        for k in range(lo, hi):
            var i = k
            if bk > 0:
                i = ldi(f, R + k)
            if Int(ld(f, p(q, 5) + i)) == fi:
                continue
            if bk == 0 and Int(ld(f, p(q, 0) + i * d + j)) != cat:
                continue
            var e = sub(ld(f, p(q, 3) + i * T + tt), mean)
            ssd = add(ssd, mul(e, e))
    st(f, p(q, 10) + t, te_value(ymean, yvar, smooth, s, cnt, mean, ssd))


def te_gather_unit(t: Int, f: FP, q: IP):
    """q = [ROWS, n, d, Y, T, FOLD, G]; t = j*n + k (lane prep-apple2). The
    k-th row of column j's buckets (`te_bucket`'s ROWS), i: G[j*n + k] =
    FOLD[i], G[d*n + (j*T + tt)*n + k] = Y[i*T + tt], raw words (`te_enc`
    flushes them as it reads them). A slot past the bucketed rows (codes
    outside [0, CMAX)) is never read."""
    var n = p(q, 1)
    var d = p(q, 2)
    var T = p(q, 4)
    var j = t // n
    var k = t % n
    var G = p(q, 6)
    var i = ldi(f, p(q, 0) + t)
    if i < 0 or i >= n:
        return
    f.unsafe_store(G + t, raw(f, p(q, 5) + i))
    for tt in range(T):
        f.unsafe_store(G + d * n + (j * T + tt) * n + k, raw(f, p(q, 3) + i * T + tt))


@always_inline
def te_value(ymean: Float32, yvar: Float32, smooth: Float32, s: Float32, cnt: Int, mean: Float32,
             ssd: Float32) -> Float32:
    """One category's encoding from its target sum s over cnt rows and, for
    the "auto" encoding (smooth < 0) with cnt > 0, its mean s / cnt and
    squared deviations ssd (both folded in ascending row order). The host's
    one-pass-per-feature spelling (x_prep/host/target.mojo) calls this too."""
    var enc = ymean
    if smooth >= Float32(0):
        var den = add(Float32(cnt), smooth)
        if den > Float32(0):
            enc = div(add(s, mul(smooth, ymean)), den)
    elif cnt > 0:
        var vc = mul(yvar, Float32(cnt))
        var den = add(vc, div(ssd, Float32(cnt)))
        if den > Float32(0):
            var lam = div(vc, den)
            enc = add(mul(lam, mean), mul(sub(Float32(1), lam), ymean))
    return enc


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
