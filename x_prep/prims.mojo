# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The prep lane's shared units (x_prep/common.mojo has the program model).

Each unit documents its parameter layout `q = [...]` (arena offsets unless
named as a count) and its work item `t`. Every loop inside a unit runs in
ascending index order; that order IS the reduction order on every column.
References: sklearn 1.x `preprocessing/_data.py` (scalers, Binarizer,
Normalizer, `_handle_zeros_in_scale`), `preprocessing/_encoders.py`
(`_unique`, `_encode`), numpy `lib/_function_base_impl.py` (`_lerp`, the
linear percentile) for the quantile unit.
"""
from std.memory import bitcast
from checks.numerics import ftz, identical_mul, identical_div, identical_sqrt, identical_exp, identical_log
from x_prep.common import FP, IP, p, ld, raw, st, ldi, sti, is_nan, canon, canonical_nan, key, heap_sort, X_PREP_HOST_SABOTAGE, RUN, run_block

#: float32 machine epsilon; `_handle_zeros_in_scale` maps scale < 10 * eps to 1.
comptime F32_EPS = Float32(1.1920929e-07)


@always_inline
def add(a: Float32, b: Float32) -> Float32:
    comptime if X_PREP_HOST_SABOTAGE:
        var r = ftz(ftz(a) + ftz(b))
        if r != Float32(0) and r == r:
            return bitcast[DType.float32](bitcast[DType.uint32](r) + UInt32(1))
        return r
    return ftz(ftz(a) + ftz(b))


@always_inline
def sub(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) - ftz(b))


@always_inline
def mul(a: Float32, b: Float32) -> Float32:
    return ftz(identical_mul(ftz(a), ftz(b)))


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(ftz(a), ftz(b)))


@always_inline
def logf(a: Float32) -> Float32:
    return ftz(identical_log(ftz(a)))


@always_inline
def expf(a: Float32) -> Float32:
    return ftz(identical_exp(ftz(a)))


@always_inline
def sqrtf(a: Float32) -> Float32:
    return ftz(identical_sqrt(ftz(a)))


@always_inline
def zero_to_one(s: Float32) -> Float32:
    """sklearn `_handle_zeros_in_scale` for a float32 scale."""
    if s < mul(Float32(10), F32_EPS):
        return Float32(1)
    return s


# ---------------------------------------------------------------- columns
def sort_cols_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, S, canon]; t = column. S[c*n : c*n+n] = column c sorted
    by `key` (NaN last); with canon, -0.0 -> 0.0 and one NaN word."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var S = p(q, 3)
    var c = t
    for i in range(n):
        var v = ftz(raw(f, X + i * d + c))
        if p(q, 4) != 0:
            v = canon(v)
        f.unsafe_store(S + c * n + i, v)
    heap_sort(f, S + c * n, n)


@always_inline
def _cs_take(v: Float32, mut cnt: Int, mut s: Float32, mut lo: Float32, mut hi: Float32, mut ma: Float32):
    """One row of col_stats' first pass (v already flushed)."""
    if is_nan(v):
        return
    if cnt == 0:
        lo = v
        hi = v
    else:
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    if abs(v) > ma:
        ma = abs(v)
    s = add(s, v)
    cnt += 1


@always_inline
def _ss_take(v: Float32, mean: Float32, mut ss: Float32):
    """One row of col_stats' second pass (v already flushed)."""
    if is_nan(v):
        return
    var e = sub(v, mean)
    ss = add(ss, mul(e, e))


def col_stats_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, OUT]; t = column. OUT rows of d: count, mean, var
    (population), min, max, maxabs, over the non-NaN entries; an empty column
    writes zeros (no 0/0). DEVIATION 5400 (rows fold ascending), 5403 (the
    empty-column guard), 5408 (operands flushed by `ld`). Rows are loaded
    RUN at a time (`run_block`) and folded one by one in the same order."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var O = p(q, 3)
    var c = t
    var cnt = 0
    var s = Float32(0)
    var lo = Float32(0)
    var hi = Float32(0)
    var ma = Float32(0)
    var full = n - n % RUN
    for i0 in range(0, full, RUN):
        var blk = run_block[RUN](f, X + i0 * d + c, d)
        comptime for u in range(RUN):
            _cs_take(ftz(blk[u]), cnt, s, lo, hi, ma)
    for i in range(full, n):
        _cs_take(ld(f, X + i * d + c), cnt, s, lo, hi, ma)
    var mean = Float32(0)
    var var_ = Float32(0)
    if cnt > 0:
        mean = div(s, Float32(cnt))
        var ss = Float32(0)
        for i0 in range(0, full, RUN):
            var blk = run_block[RUN](f, X + i0 * d + c, d)
            comptime for u in range(RUN):
                _ss_take(ftz(blk[u]), mean, ss)
        for i in range(full, n):
            _ss_take(ld(f, X + i * d + c), mean, ss)
        var_ = div(ss, Float32(cnt))
    st(f, O + c, Float32(cnt))
    st(f, O + d + c, mean)
    st(f, O + 2 * d + c, var_)
    st(f, O + 3 * d + c, lo)
    st(f, O + 4 * d + c, hi)
    st(f, O + 5 * d + c, ma)


def quantile_unit(t: Int, f: FP, q: IP):
    """q = [S, n, d, QF, nq, OUT, CNT]; t = c*nq + j. numpy's linear
    percentile of the first CNT[c] (or n when CNT < 0) sorted entries of
    column c at fraction QF[j], numpy's `_lerp` spelling (DEVIATION 5409: the
    upper form b - d (1 - g) from g >= 0.5); empty -> 0."""
    var S = p(q, 0)
    var n = p(q, 1)
    var nq = p(q, 4)
    var c = t // nq
    var j = t % nq
    var cnt = n
    if p(q, 6) >= 0:
        cnt = Int(ld(f, p(q, 6) + c))
    var out = Float32(0)
    if cnt > 0:
        var idx = mul(ld(f, p(q, 3) + j), Float32(cnt - 1))
        var lo = Int(idx)
        if lo > cnt - 1:
            lo = cnt - 1
        if lo < 0:
            lo = 0
        var g = sub(idx, Float32(lo))
        var hi_i = lo + 1 if lo + 1 < cnt else cnt - 1
        var a = ld(f, S + c * n + lo)
        var b = ld(f, S + c * n + hi_i)
        var diff = sub(b, a)
        if g >= Float32(0.5):
            out = sub(b, mul(diff, sub(Float32(1), g)))
        else:
            out = add(a, mul(diff, g))
    st(f, p(q, 5) + t, out)


def affine_unit(t: Int, f: FP, q: IP):
    """q = [X, count, d, C, S, OUT]; t = element. OUT = (X - C[c]) / S[c]
    (C or S < 0: skipped). A NaN input is copied bit for bit."""
    var x = raw(f, p(q, 0) + t)
    var d = p(q, 2)
    var c = t % d
    if is_nan(x):
        f.unsafe_store(p(q, 5) + t, x)
        return
    var v = ftz(x)
    if p(q, 3) >= 0:
        v = sub(v, ld(f, p(q, 3) + c))
    if p(q, 4) >= 0:
        v = div(v, ld(f, p(q, 4) + c))
    st(f, p(q, 5) + t, v)


def scale_params_unit(t: Int, f: FP, q: IP):
    """q = [Q, nq, d, CENTER, SCALE, kind, ilo, ihi, imid]; t = column.
    kind 0 (robust): CENTER = Q[c*nq+imid], SCALE = Q[ihi] - Q[ilo];
    kind 1 (maxabs): SCALE = Q[c] (the maxabs row); both zero -> one."""
    var Q = p(q, 0)
    var nq = p(q, 1)
    var c = t
    if p(q, 5) == 0:
        if p(q, 3) >= 0:
            st(f, p(q, 3) + c, ld(f, Q + c * nq + p(q, 8)))
        if p(q, 4) >= 0:
            st(f, p(q, 4) + c, zero_to_one(sub(ld(f, Q + c * nq + p(q, 7)), ld(f, Q + c * nq + p(q, 6)))))
    else:
        st(f, p(q, 4) + c, zero_to_one(ld(f, Q + c)))


# ---------------------------------------------------------------- encoders
def unique_cols_unit(t: Int, f: FP, q: IP):
    """q = [S, n, d, U, CNT]; t = column. The distinct words of sorted column
    c (by `key` equality) into U[c*n : ...], their count into CNT[c]."""
    var S = p(q, 0)
    var n = p(q, 1)
    var U = p(q, 3)
    var c = t
    var k = 0
    for i in range(n):
        var v = raw(f, S + c * n + i)
        if k == 0 or key(v) != key(raw(f, U + c * n + k - 1)):
            f.unsafe_store(U + c * n + k, v)
            k += 1
    st(f, p(q, 4) + c, Float32(k))


def mode_cols_unit(t: Int, f: FP, q: IP):
    """q = [S, n, d, OUT, CNT]; t = column. The most frequent non-NaN word of
    sorted column c, the smallest on a tie (sklearn `_most_frequent`); no
    valid entry -> 0. CNT >= 0 receives the count of valid entries."""
    var S = p(q, 0)
    var n = p(q, 1)
    var c = t
    var best = Float32(0)
    var best_n = 0
    var run = 0
    var valid = 0
    for i in range(n):
        var v = raw(f, S + c * n + i)
        if is_nan(v):
            break
        valid += 1
        if i > 0 and ftz(v) == ftz(raw(f, S + c * n + i - 1)):   # -0.0 joins 0.0's run
            run += 1
        else:
            run = 1
        if run > best_n:
            best_n = run
            best = v
    st(f, p(q, 3) + c, best)
    if p(q, 4) >= 0:
        st(f, p(q, 4) + c, Float32(valid))


def lookup_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, U, ustride, CNT, OUT]; t = element. The index of
    canon(X) among column c's categories U[c*ustride : +CNT[c]] (sorted by
    `key`), or -1; written as a float code."""
    var d = p(q, 2)
    var c = t % d
    var v = canon(ftz(raw(f, p(q, 0) + t)))
    var base = p(q, 3) + c * p(q, 4)
    var cnt = Int(ld(f, p(q, 5) + c))
    var kv = key(v)
    var lo = 0
    var hi = cnt
    while lo < hi:
        var mid = (lo + hi) // 2
        if key(raw(f, base + mid)) < kv:
            lo = mid + 1
        else:
            hi = mid
    var code = -1
    if lo < cnt and key(raw(f, base + lo)) == kv:
        code = lo
    st(f, p(q, 6) + t, Float32(code))


def count_neg_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, OUT]; t = column: how many codes are negative."""
    var n = p(q, 1)
    var d = p(q, 2)
    var k = 0
    for i in range(n):
        if ld(f, p(q, 0) + i * d + t) < Float32(0):
            k += 1
    st(f, p(q, 3) + t, Float32(k))


def onehot_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, START, DROP, W, OUT]; t = element. Writes 1 at
    OUT[i*W + START[c] + code] (a dropped category is skipped and the ones
    after it shift down); a negative code writes nothing. OUT arrives zeroed."""
    var d = p(q, 2)
    var i = t // d
    var c = t % d
    var code = Int(ld(f, p(q, 0) + t))
    if code < 0:
        return
    var pos = code
    if p(q, 4) >= 0:
        var drop = Int(ld(f, p(q, 4) + c))
        if drop >= 0:
            if code == drop:
                return
            if code > drop:
                pos = code - 1
    st(f, p(q, 6) + i * p(q, 5) + Int(ld(f, p(q, 3) + c)) + pos, Float32(1))


def i2f_unit(t: Int, f: FP, q: IP):
    """q = [SRC, OUT]; t = element: int32 bits -> float value."""
    st(f, p(q, 1) + t, Float32(ldi(f, p(q, 0) + t)))


def f2i_unit(t: Int, f: FP, q: IP):
    """q = [SRC, OUT]; t = element: an integral float value -> int32 bits."""
    sti(f, p(q, 1) + t, Int(ld(f, p(q, 0) + t)))


def binarize_unit(t: Int, f: FP, q: IP):
    """q = [X, count, THR, OUT]; t = element: 1 when X > THR, else 0; a NaN
    input is copied (sklearn's Binarizer leaves it)."""
    var x = raw(f, p(q, 0) + t)
    if is_nan(x):
        f.unsafe_store(p(q, 3) + t, x)
        return
    st(f, p(q, 3) + t, Float32(1) if ftz(x) > ld(f, p(q, 2)) else Float32(0))


# ---------------------------------------------------------------- dense
def matmul_unit(t: Int, f: FP, q: IP):
    """q = [A, sa0, sa1, B, sb0, sb1, C, ncols, K, BIAS, ALPHA]; t = i*ncols + j.
    C[t] = ALPHA * sum_l A[i*sa0 + l*sa1] * B[l*sb0 + j*sb1] (+ BIAS[j]),
    l ascending (ALPHA < 0: no scale). DEVIATION 5401: each product rounded
    (`identical_mul`) before its add, never contracted into an FMA."""
    var nc = p(q, 7)
    var i = t // nc
    var j = t % nc
    var acc = Float32(0)
    for l in range(p(q, 8)):
        acc = add(acc, mul(ld(f, p(q, 0) + i * p(q, 1) + l * p(q, 2)), ld(f, p(q, 3) + l * p(q, 4) + j * p(q, 5))))
    if p(q, 10) >= 0:
        acc = mul(acc, ld(f, p(q, 10)))
    if p(q, 9) >= 0:
        acc = add(acc, ld(f, p(q, 9) + j))
    st(f, p(q, 6) + t, acc)


def row_softmax_unit(t: Int, f: FP, q: IP):
    """q = [S, n, K, LOGP, PROBA]; t = row. sklearn's `logsumexp`
    normalisation: LOGP = S - (m + log(sum exp(S - m))), PROBA = exp(LOGP).
    A row whose maximum is -inf (every class impossible) is uniform, never NaN."""
    var K = p(q, 2)
    var S = p(q, 0) + t * K
    var m = ld(f, S)
    for k in range(1, K):
        var v = ld(f, S + k)
        if v > m:
            m = v
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    if m == neg_inf:
        for k in range(K):
            if p(q, 3) >= 0:
                st(f, p(q, 3) + t * K + k, sub(Float32(0), logf(Float32(K))))
            if p(q, 4) >= 0:
                st(f, p(q, 4) + t * K + k, div(Float32(1), Float32(K)))
        return
    var s = Float32(0)
    for k in range(K):
        s = add(s, expf(sub(ld(f, S + k), m)))
    var lse = add(m, logf(s))
    for k in range(K):
        var lp = sub(ld(f, S + k), lse)
        if p(q, 3) >= 0:
            st(f, p(q, 3) + t * K + k, lp)
        if p(q, 4) >= 0:
            st(f, p(q, 4) + t * K + k, expf(lp))


def row_argmax_unit(t: Int, f: FP, q: IP):
    """q = [S, n, K, OUT]; t = row: first-max-wins argmax as int32 bits
    (DEVIATION 5404: the lower index wins a tie)."""
    var K = p(q, 2)
    var S = p(q, 0) + t * K
    var best = 0
    var bv = ld(f, S)
    for k in range(1, K):
        var v = ld(f, S + k)
        if v > bv:
            bv = v
            best = k
    sti(f, p(q, 3) + t, best)


def class_stats_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, K, CNT, MEAN, VAR, SUM]; t = k*d + c. Over the rows
    whose class code Y[i] == k, ascending: the sum, mean and population
    variance of column c (and, for c == 0, the row count). Offsets < 0 are not
    written; an empty class writes zeros."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var Y = p(q, 3)
    var k = t // d
    var c = t % d
    var cnt = 0
    var s = Float32(0)
    for i in range(n):
        if Int(ld(f, Y + i)) != k:
            continue
        s = add(s, ld(f, X + i * d + c))
        cnt += 1
    var mean = Float32(0)
    var ss = Float32(0)
    if cnt > 0:
        mean = div(s, Float32(cnt))
        if p(q, 7) >= 0:
            for i in range(n):
                if Int(ld(f, Y + i)) != k:
                    continue
                var e = sub(ld(f, X + i * d + c), mean)
                ss = add(ss, mul(e, e))
            ss = div(ss, Float32(cnt))
    if c == 0 and p(q, 5) >= 0:
        st(f, p(q, 5) + k, Float32(cnt))
    if p(q, 6) >= 0:
        st(f, p(q, 6) + t, mean)
    if p(q, 7) >= 0:
        st(f, p(q, 7) + t, ss)
    if p(q, 8) >= 0:
        st(f, p(q, 8) + t, s)


def class_stats_w_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, K, CNT, MEAN, VAR, SUM, W]; t = k*d + c: class_stats
    with the per-row weight W[i] (the naive Bayes sample_weight): over the rows
    of class k, ascending, SUM = sum w x, CNT = sum w (for c == 0),
    MEAN = SUM / CNT and VAR = sum w (x - MEAN)^2 / CNT (numpy `average`
    with weights). Offsets < 0 are not written; a class of zero weight
    writes zeros."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var Y = p(q, 3)
    var W = p(q, 9)
    var k = t // d
    var c = t % d
    var sw = Float32(0)
    var s = Float32(0)
    for i in range(n):
        if Int(ld(f, Y + i)) != k:
            continue
        var w = ld(f, W + i)
        s = add(s, mul(w, ld(f, X + i * d + c)))
        sw = add(sw, w)
    var mean = Float32(0)
    var ss = Float32(0)
    if sw != Float32(0):
        mean = div(s, sw)
        if p(q, 7) >= 0:
            for i in range(n):
                if Int(ld(f, Y + i)) != k:
                    continue
                var e = sub(ld(f, X + i * d + c), mean)
                ss = add(ss, mul(ld(f, W + i), mul(e, e)))
            ss = div(ss, sw)
    if c == 0 and p(q, 5) >= 0:
        st(f, p(q, 5) + k, sw)
    if p(q, 6) >= 0:
        st(f, p(q, 6) + t, mean)
    if p(q, 7) >= 0:
        st(f, p(q, 7) + t, ss)
    if p(q, 8) >= 0:
        st(f, p(q, 8) + t, s)


def center_rows_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, M, Y, W, OUT]; t = element. OUT = (X - M[y_i, c]) * W[c]
    (Y < 0: row 0 of M for every row; W < 0: no scale)."""
    var d = p(q, 2)
    var i = t // d
    var c = t % d
    var row = 0
    if p(q, 4) >= 0:
        row = Int(ld(f, p(q, 4) + i))
    var v = sub(ld(f, p(q, 0) + t), ld(f, p(q, 3) + row * d + c))
    if p(q, 5) >= 0:
        v = mul(v, ld(f, p(q, 5) + c))
    st(f, p(q, 6) + t, v)


def where_neg_unit(t: Int, f: FP, q: IP):
    """q = [CODES, count, VAL, OUT]; t = element: VAL where the code is
    negative (an unknown category), else the code."""
    var v = ld(f, p(q, 0) + t)
    st(f, p(q, 3) + t, ld(f, p(q, 2)) if v < Float32(0) else v)


def mark_missing_unit(t: Int, f: FP, q: IP):
    """q = [X, count, VAL, OUT]; t = element: the one quiet NaN word where X
    equals VAL (the imputer's numeric `missing_values`), else X's bits."""
    var x = raw(f, p(q, 0) + t)
    if not is_nan(x) and ftz(x) == ld(f, p(q, 2)):
        f.unsafe_store(p(q, 3) + t, canonical_nan())
    else:
        f.unsafe_store(p(q, 3) + t, x)


def fill_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, STAT, OUT, KEEP, dout]; t = i*dout + jj. Output column jj
    is input column KEEP[jj]; a NaN entry becomes STAT[column], anything else
    is copied bit for bit."""
    var dout = p(q, 6)
    var i = t // dout
    var jj = t % dout
    var c = Int(ld(f, p(q, 5) + jj))
    var x = raw(f, p(q, 0) + i * p(q, 2) + c)
    if is_nan(x):
        f.unsafe_store(p(q, 4) + t, raw(f, p(q, 3) + c))
    else:
        f.unsafe_store(p(q, 4) + t, x)


def label_binarize_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, K, BINARY, NEG, POS, W, OUT]; t = i*W + j. int32 bits:
    POS where row i's class code is j (BINARY: where it is 1, one column),
    else NEG; an unknown code (-1) is a NEG row (sklearn `label_binarize`)."""
    var W = p(q, 6)
    var i = t // W
    var j = t % W
    var code = Int(ld(f, p(q, 0) + i))
    var hit = code == 1 if p(q, 3) != 0 else code == j
    sti(f, p(q, 7) + t, p(q, 5) if hit else p(q, 4))


def scatter_ones_unit(t: Int, f: FP, q: IP):
    """q = [CODES, ROWS, W, OUT]; t = entry: int32 1 at OUT[ROWS[t]*W + code]
    for a known code. Two entries of one row with one code write the same
    word, so their order cannot matter."""
    var code = Int(ld(f, p(q, 0) + t))
    if code < 0:
        return
    sti(f, p(q, 3) + Int(ld(f, p(q, 1) + t)) * p(q, 2) + code, 1)


def gather_cols_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, KEEP, dout, OUT]; t = i*dout + jj: OUT = X[i, KEEP[jj]],
    bit for bit."""
    var dout = p(q, 4)
    var i = t // dout
    var c = Int(ld(f, p(q, 3) + t % dout))
    f.unsafe_store(p(q, 5) + t, raw(f, p(q, 0) + i * p(q, 2) + c))


def var_ptp_unit(t: Int, f: FP, q: IP):
    """q = [ST, d, OUT, PTP]; t = column: the variance row of col_stats, or
    min(variance, max - min) when PTP (VarianceThreshold's threshold == 0
    rule, so an exactly constant column reads exactly 0)."""
    var d = p(q, 1)
    var v = ld(f, p(q, 0) + 2 * d + t)
    if p(q, 3) != 0:
        var ptp = sub(ld(f, p(q, 0) + 4 * d + t), ld(f, p(q, 0) + 3 * d + t))
        if ptp < v:
            v = ptp
    st(f, p(q, 2) + t, v)


def sqsum_cols_unit(t: Int, f: FP, q: IP):
    """q = [C, rows, d, OUT]; t = column: sum over rows (ascending) of C[r, c]^2
    (RFE's squared importance, summed over a multi-row coef_)."""
    var d = p(q, 2)
    var s = Float32(0)
    for r in range(p(q, 1)):
        var v = ld(f, p(q, 0) + r * d + t)
        s = add(s, mul(v, v))
    st(f, p(q, 3) + t, s)


# ---------------------------------------------------------------- inverses
def block_argmax_unit(t: Int, f: FP, q: IP):
    """q = [X, n, W, d, START, WIDTH, DROP, CHECK, OUT]; t = i*d + c: the
    reference's one-hot inverse (OneHotEncoder, KBinsDiscretizer's onehot,
    LabelBinarizer's multiclass). Over row i's block X[i*W + START[c] :
    + WIDTH[c]]: numpy's argmax (first max wins, the first NaN wins over
    everything), shifted past the dropped category DROP[c] (DROP < 0: none).
    With CHECK, a block summing (ascending) to exactly 0 is the dropped
    category, or -1 (unknown) when none was dropped; a zero-width block is
    the dropped category. Written as a float code."""
    var d = p(q, 3)
    var i = t // d
    var c = t % d
    var w = Int(ld(f, p(q, 5) + c))
    var drop = -1
    if p(q, 6) >= 0:
        drop = Int(ld(f, p(q, 6) + c))
    if w == 0:
        st(f, p(q, 8) + t, Float32(drop))
        return
    var R = p(q, 0) + i * p(q, 2) + Int(ld(f, p(q, 4) + c))
    var bv = ld(f, R)
    var best = 0
    var nan_hit = bv != bv
    var s = bv
    for k in range(1, w):
        var v = ld(f, R + k)
        s = add(s, v)
        if not nan_hit:
            if v != v:
                best = k
                nan_hit = True
            elif v > bv:
                bv = v
                best = k
    var code = best
    if drop >= 0 and code >= drop:
        code += 1
    if p(q, 7) != 0 and s == Float32(0):
        code = drop
    st(f, p(q, 8) + t, Float32(code))


@always_inline
def _matches(x: Float32, v: Float32) -> Bool:
    """sklearn `_get_mask`: NaN matches NaN, anything else by value."""
    if is_nan(v):
        return is_nan(x)
    return not is_nan(x) and ftz(x) == ftz(v)


def ord_inverse_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, MISS, EMV, UNK_ON, UNK, NCAT, OUT]; t = element: an
    ordinal code back to a category index (OrdinalEncoder, KBinsDiscretizer
    ordinal). A value matching EMV (NaN matches NaN) in a column with a
    missing category MISS[c] >= 0 is MISS[c]; else (UNK_ON) a value matching
    UNK is -1 (unknown); else numpy's astype(int64), truncation, which must
    land in [0, NCAT[c]) or the code is -2 (invalid)."""
    var d = p(q, 2)
    var c = t % d
    var x = raw(f, p(q, 0) + t)
    var miss = Int(ld(f, p(q, 3) + c))
    var code: Int
    if miss >= 0 and _matches(x, raw(f, p(q, 4))):
        code = miss
    elif p(q, 5) != 0 and _matches(x, raw(f, p(q, 6))):
        code = -1
    elif is_nan(x):
        code = -2
    else:
        var v = ftz(x)
        if v <= Float32(-1) or v >= ld(f, p(q, 7) + c):
            code = -2
        else:
            code = Int(v)
    st(f, p(q, 8) + t, Float32(code))


def cat_gather_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, CATS, kmax, OUT]; t = element: column c's category
    CATS[c*kmax + code] bit for bit (the NaN category is the canonical NaN),
    or the canonical NaN for a negative code."""
    var d = p(q, 2)
    var c = t % d
    var code = Int(ld(f, p(q, 0) + t))
    if code < 0:
        f.unsafe_store(p(q, 5) + t, canonical_nan())
        return
    f.unsafe_store(p(q, 5) + t, raw(f, p(q, 3) + c * p(q, 4) + code))


def where_code_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, MISS, VAL, SRC, OUT]; t = element: VAL (bit for bit)
    where CODES[t] is column c's missing category MISS[c] >= 0
    (OrdinalEncoder's encoded_missing_value), else SRC[t] bit for bit."""
    var d = p(q, 2)
    var c = t % d
    var miss = Int(ld(f, p(q, 3) + c))
    if miss >= 0 and Int(ld(f, p(q, 0) + t)) == miss:
        f.unsafe_store(p(q, 6) + t, raw(f, p(q, 4)))
        return
    f.unsafe_store(p(q, 6) + t, raw(f, p(q, 5) + t))


# ---------------------------------------------------------------- option parity
def indicator_unit(t: Int, f: FP, q: IP):
    """q = [Y, count, MODE, THR, NEG, POS, OUT]; t = element. int32 bits: POS
    on a hit, else NEG. MODE 0: a hit is Y != 0 (LabelBinarizer's multilabel
    transform, sklearn `label_binarize`); MODE 1: a hit is Y > THR[0] (its
    inverse, `_inverse_binarize_multilabel`; NaN is no hit)."""
    var y = ld(f, p(q, 0) + t)
    var hit: Bool
    if p(q, 2) == 0:
        hit = y != Float32(0)
    else:
        hit = y > ld(f, p(q, 3))
    sti(f, p(q, 6) + t, p(q, 5) if hit else p(q, 4))


def code_counts_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, KSTRIDE, OUT]; t = column c. For every row, ascending,
    a code k >= 0 of column c adds one to the int32 count OUT[c*KSTRIDE + k]
    (OUT arrives zeroed): the encoders' per-category counts (sklearn
    `_unique(..., return_counts=True)`, `_get_counts`)."""
    var n = p(q, 1)
    var d = p(q, 2)
    var c = t
    var base = p(q, 4) + c * p(q, 3)
    for i in range(n):
        var code = Int(ld(f, p(q, 0) + i * d + c))
        if code >= 0:
            sti(f, base + code, ldi(f, base + code) + 1)


def remap_codes_unit(t: Int, f: FP, q: IP):
    """q = [CODES, count, d, MAP, MSTRIDE, NMAP, NEG, OUT]; t = element of
    column c = t % d. A code k in [0, NMAP[c]) becomes MAP[c*MSTRIDE + k] (the
    encoders' infrequent grouping, sklearn `_map_infrequent_categories`); a
    negative code becomes NEG[c] (NEG < 0: stays); any other code stays."""
    var d = p(q, 2)
    var c = t % d
    var code = ld(f, p(q, 0) + t)
    var v = code
    if code < Float32(0):
        if p(q, 6) >= 0:
            v = ld(f, p(q, 6) + c)
    elif Int(code) < Int(ld(f, p(q, 5) + c)):
        v = ld(f, p(q, 3) + c * p(q, 4) + Int(code))
    st(f, p(q, 7) + t, v)


def add_arrays_unit(t: Int, f: FP, q: IP):
    """q = [A, B, OUT]; t = element: OUT = A + B (a running count plus a
    batch's, the naive Bayes partial_fit)."""
    st(f, p(q, 2) + t, add(ld(f, p(q, 0) + t), ld(f, p(q, 1) + t)))


# ---------------------------------------------------------------- scalers (item 5)
def scaler_stats_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, W, OUT]; t = column (StandardScaler fit with NaN and / or
    sample_weight, sklearn `_incremental_mean_and_var` on a fresh scaler).
    Over the rows ascending whose entry is not NaN and whose weight is not
    zero (W < 0: every weight is one): OUT rows of d are the count (the
    weight sum; unweighted, the row count), the mean sum(w x) / sum(w) and the
    population variance sum(w (x - mean)^2) / sum(w). A column whose counted
    entries are all equal keeps that value as its mean and variance zero
    (STD-1's exact-constant rule, the binding's standard_fit). No counted
    entry: the count is zero and mean and variance are NaN (the reference's
    0 / 0). DEVIATION 5400 (rows fold ascending), 5408 (operands flushed by
    `ld`)."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var W = p(q, 3)
    var O = p(q, 4)
    var c = t
    var cnt = 0
    var sw = Float32(0)
    var s = Float32(0)
    var first = Float32(0)
    var changed = False
    for i in range(n):
        var v = ld(f, X + i * d + c)
        if is_nan(v):
            continue
        var w = Float32(1)
        if W >= 0:
            w = ld(f, W + i)
            if w == Float32(0):
                continue
        if cnt == 0:
            first = v
        elif v != first:
            changed = True
        if W >= 0:
            s = add(s, mul(w, v))
            sw = add(sw, w)
        else:
            s = add(s, v)
        cnt += 1
    if W < 0:
        sw = Float32(cnt)
    var mean = canonical_nan()
    var var_ = canonical_nan()
    if cnt > 0:
        if not changed:
            mean = first
            var_ = Float32(0)
        else:
            mean = div(s, sw)
            var ss = Float32(0)
            for i in range(n):
                var v = ld(f, X + i * d + c)
                if is_nan(v):
                    continue
                var w = Float32(1)
                if W >= 0:
                    w = ld(f, W + i)
                    if w == Float32(0):
                        continue
                var e = sub(v, mean)
                if W >= 0:
                    ss = add(ss, mul(w, mul(e, e)))
                else:
                    ss = add(ss, mul(e, e))
            var_ = div(ss, sw)
    st(f, O + c, sw)
    f.unsafe_store(O + d + c, mean)
    f.unsafe_store(O + 2 * d + c, var_)


def std_scale_unit(t: Int, f: FP, q: IP):
    """q = [VAR, SCALE]; t = column: StandardScaler's scale from a variance,
    the binding's rule (STD-2): exactly zero -> one, else sqrt(var); a NaN
    variance (a column never seen) stays NaN."""
    var v = raw(f, p(q, 0) + t)
    if is_nan(v):
        f.unsafe_store(p(q, 1) + t, v)
    elif ftz(v) == Float32(0):
        st(f, p(q, 1) + t, Float32(1))
    else:
        st(f, p(q, 1) + t, sqrtf(v))


def nan_keep_unit(t: Int, f: FP, q: IP):
    """q = [X, d, T, COLNAN, OUT]; t = element (a scaler transform with NaN):
    a NaN entry of X is copied bit for bit, an element of a column flagged in
    COLNAN (its statistics are NaN: never seen in fit) is the quiet NaN word,
    anything else is T's bits (the binding's transform of the NaN-filled X)."""
    var x = raw(f, p(q, 0) + t)
    if is_nan(x):
        f.unsafe_store(p(q, 4) + t, x)
    elif raw(f, p(q, 3) + t % p(q, 1)) != Float32(0):
        f.unsafe_store(p(q, 4) + t, canonical_nan())
    else:
        f.unsafe_store(p(q, 4) + t, raw(f, p(q, 2) + t))
