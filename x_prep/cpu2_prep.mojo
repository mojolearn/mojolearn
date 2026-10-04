# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE cpu2-l3-prep (2026-10-04): the host numeric and data work that still
sat inside the prep family's GPU routes (re-audit L3,
~/mojolearn-evidence/cpu-reaudit-2026-10-04.md), as x_prep units in the
x_prep/fam2.mojo op range (x_prep/common.mojo's program model: one GPU
thread per unit, the host column running the same units in a loop, so
NVIDIA, AMD, Apple and the host column share one arithmetic). These are
fixes, not experiments: every tier, no `_OFF` arm.

Ops (F2_BASE + 12 ..; python/mojolearn/_expansion_prep.py `_OPS`):
  242 c2_bin_code     LabelBinarizer binary inverse: the last column > THR as a code
  243 c2_kfold        TargetEncoder KFold fold per row (keyed permutation)
  244 c2_strat_meta   stratified folds: each class's start in first-seen order
  245 c2_strat_flag   stratified folds: every class smaller than n_folds?
  246 c2_strat_fold   stratified folds: fold per row
  247 c2_integral     count of non-integral float words per chunk
  248 c2_isum         int32 sum of a small partial vector
  249 c2_inf_m0       infrequent categories: count below min_frequency
  250 c2_inf_m1       infrequent categories: max_categories by stable count rank
  251 c2_inf_map      infrequent categories: category -> grouped code
  252 c2_imp_stats    SimpleImputer statistics_ and fill (empty columns)
  253 c2_key64        an ordered 64-bit key per value (f32 / f64 / int words)
  254 c2_topk         SelectKBest's top-k mask (stable ascending rank)
  255 c2_gt           VarianceThreshold's value > threshold mask (binary64)
  256 c2_rfe_step     RFE: drop the `step` weakest active features
  257 c2_rfe_rank     RFE: ranking + 1 for every dropped feature
  258 c2_ii_miss      IterativeImputer: missing count per kept column
  259 c2_ii_pos       IterativeImputer: position of a feature in a fixed order
  260 c2_ii_ord       IterativeImputer: the fixed order, missing features only
  261 c2_ii_rand      IterativeImputer: one Fisher-Yates order per round
  262 c2_colmax       the largest of a vector (NaN skipped; 0 when empty)
  263 c2_grid         percent / ratio level grids in binary64, rounded to float32
  264 c2_nan_sub      scaler transform statistics: NaN columns substituted, checked
  265 c2_nan_rows     NaN written into the `colnan` columns of a row block
  266 c2_cnt0         colnan = (count == 0)
  267 c2_mm_keep      MinMaxScaler fitted rows: checked, colnan columns NaN
  268 c2_mm_merge     MinMaxScaler partial_fit: the running extrema rows
  269 c2_add_i64      64-bit integer add (n_samples_seen_)
"""
from std.math import isfinite
from std.memory import bitcast
from x_prep.common import FP, IP, p, raw, ld, st, ldi, sti, canonical_nan
from x_prep.py2mojo import splitmix_at
from checks.soft_f64 import sf64_from_f32, sf64_from_int, sf64_to_f32, sf64_div, sf64_mul

comptime C2_FIRST = 12  # F2_BASE + C2_FIRST is c2_bin_code
comptime C2_N = 28

comptime _SIGN = UInt64(0x8000000000000000)
comptime _NEG_INF = UInt64(0xFFF0000000000000)
comptime _KEY_ROUNDS = 6
comptime _KEY_MUL = UInt64(0xD1B54A32D192ED03)
comptime _CLASS_KEY = UInt64(0x2545F4914F6CDD1D)


@always_inline
def _u32(f: FP, i: Int) -> UInt64:
    return UInt64(f.bitcast[UInt32]().unsafe_load(i))


@always_inline
def _put_u64(f: FP, i: Int, v: UInt64):
    f.bitcast[UInt32]().unsafe_store(i, UInt32(v & UInt64(0xFFFFFFFF)))
    f.bitcast[UInt32]().unsafe_store(i + 1, UInt32(v >> UInt64(32)))


@always_inline
def _get_u64(f: FP, i: Int) -> UInt64:
    return (_u32(f, i + 1) << UInt64(32)) | _u32(f, i)


@always_inline
def _seed(f: FP, at: Int) -> UInt64:
    """Two int32 words (low, high) as the uint64 seed (`_seed_words`)."""
    return _get_u64(f, at)


@always_inline
def _perm_image(s: UInt64, n: Int, t: Int) -> Int:
    """pi(t) for a key-`s` permutation pi of [0, n): x_prep/fam2.mojo
    f2_perm_rows' balanced Feistel network, cycle walked into range."""
    var h = 1
    while (1 << (2 * h)) < n:
        h += 1
    var mask = (UInt64(1) << UInt64(h)) - UInt64(1)
    var x = UInt64(t)
    var again = True
    while again:
        var L = x >> UInt64(h)
        var R = x & mask
        for r in range(_KEY_ROUNDS):
            var F = splitmix_at(s + UInt64(r + 1) * _KEY_MUL, Int(R)) & mask
            var nl = R
            R = L ^ F
            L = nl
        x = (L << UInt64(h)) | R
        again = x >= UInt64(n)
    return Int(x)


@always_inline
def _okey(b: UInt64) -> UInt64:
    """The ordered key of binary64 bits: unsigned order = numeric order, a
    NaN as -inf, -0 as +0."""
    var w = b
    if ((w >> UInt64(52)) & UInt64(0x7FF)) == UInt64(0x7FF) and (w & UInt64(0x000FFFFFFFFFFFFF)) != UInt64(0):
        w = _NEG_INF
    if (w & ~_SIGN) == UInt64(0):
        w = UInt64(0)
    if (w & _SIGN) != UInt64(0):
        return ~w
    return w | _SIGN


@always_inline
def _word64(f: FP, base: Int, kind: Int, t: Int) -> UInt64:
    """Value t of a raw label-style buffer (x_prep/labels.mojo lab_load's
    KIND: 0 float32, 1 int32, 3 int64, 4 float64) as binary64 bits."""
    if kind == 0:
        return sf64_from_f32(raw(f, base + t))
    if kind == 1:
        return sf64_from_int(Int(bitcast[DType.int32](f.bitcast[UInt32]().unsafe_load(base + t))))
    if kind == 3:
        return sf64_from_int(Int(bitcast[DType.int64](_get_u64(f, base + 2 * t))))
    return _get_u64(f, base + 2 * t)


@always_inline
def _fold_of(pos: Int, n: Int, F: Int) -> Int:
    """The fold of sorted position `pos` when n positions are cut into F
    folds, the first n % F one longer (numpy KFold's split)."""
    var q0 = n // F
    var r = n % F
    var B = r * (q0 + 1)
    if pos < B:
        return pos // (q0 + 1)
    return r + (pos - B) // q0


@always_inline
def _cnt_mod(a: Int, f: Int, F: Int) -> Int:
    """How many p in [0, a) have p % F == f."""
    if a <= f:
        return 0
    return (a - f + F - 1) // F


# ---------------------------------------------------------------- 242 LabelBinarizer
def c2_bin_code_unit(t: Int, f: FP, q: IP):
    """q = [X, n, W, THR, OUT]; t = row: OUT[t] = 1.0 when X[t, W - 1] > THR
    (the reference's `Y > threshold`, a NaN is 0), else 0.0."""
    var W = p(q, 2)
    var v = ld(f, p(q, 0) + t * W + W - 1)
    st(f, p(q, 4) + t, Float32(1) if v > ld(f, p(q, 3)) else Float32(0))


# ---------------------------------------------------------------- 243-246 TargetEncoder folds
def c2_kfold_unit(t: Int, f: FP, q: IP):
    """q = [SEED, n, F, SHUF, FOLD]; t = row: FOLD[t] (a float code) = the
    fold of the row's position, its image under the seed-keyed permutation
    (SHUF 1) or the row itself (SHUF 0: numpy KFold's unshuffled folds)."""
    var n = p(q, 1)
    var pos = t
    if p(q, 3) != 0:
        pos = _perm_image(_seed(f, p(q, 0)), n, t)
    st(f, p(q, 4) + t, Float32(_fold_of(pos, n, p(q, 2))))


def c2_strat_meta_unit(t: Int, f: FP, q: IP):
    """q = [START, TOT, ROWS, K, META]; t = class c (int32 words in; ROWS
    the rows grouped by class, ascending inside a class). META[2c] = the
    rows of the classes first seen before c (StratifiedKFold numbers the
    classes by first appearance and walks them in that order), META[2c + 1]
    = c's first-seen rank. An absent class: 0, -1."""
    var K = p(q, 3)
    var tot = ldi(f, p(q, 1) + t)
    if tot == 0:
        sti(f, p(q, 4) + 2 * t, 0)
        sti(f, p(q, 4) + 2 * t + 1, -1)
        return
    var first = ldi(f, p(q, 2) + ldi(f, p(q, 0) + t))
    var S = 0
    var rank = 0
    for c in range(K):
        var tc = ldi(f, p(q, 1) + c)
        if tc == 0 or c == t:
            continue
        if ldi(f, p(q, 2) + ldi(f, p(q, 0) + c)) < first:
            S += tc
            rank += 1
    sti(f, p(q, 4) + 2 * t, S)
    sti(f, p(q, 4) + 2 * t + 1, rank)


def c2_strat_flag_unit(t: Int, f: FP, q: IP):
    """q = [TOT, K, F, FLAG]; one unit over the K class counts: FLAG = 1.0
    when every present class has fewer rows than F (the reference refuses
    the split), else 0.0."""
    var all_small = True
    for c in range(p(q, 1)):
        var tc = ldi(f, p(q, 0) + c)
        if tc > 0 and not (p(q, 2) > tc):
            all_small = False
    st(f, p(q, 3), Float32(1) if all_small else Float32(0))


def c2_strat_fold_unit(t: Int, f: FP, q: IP):
    """q = [ROWS, START, TOT, CODES, META, F, SEED, SHUF, FOLD]; t = grouped
    position g: row = ROWS[g] of class c = CODES[row], its rank r inside the
    class (ascending rows). The class's fold block holds fold f
    #{p in [S, S + tot) : p % F == f} times, ascending (StratifiedKFold's
    `_make_test_folds` allocation over the first-seen class order); slot r
    (SHUF 0) or its image under the class's keyed permutation (SHUF 1)
    names the row's fold (a float code at FOLD[row])."""
    var row = ldi(f, p(q, 0) + t)
    var c = Int(ld(f, p(q, 3) + row))
    var tot = ldi(f, p(q, 2) + c)
    var r = t - ldi(f, p(q, 1) + c)
    var S = ldi(f, p(q, 4) + 2 * c)
    var F = p(q, 5)
    var slot = r
    if p(q, 7) != 0:
        var e = ldi(f, p(q, 4) + 2 * c + 1)
        slot = _perm_image(splitmix_at(_seed(f, p(q, 6)) ^ _CLASS_KEY, e), tot, r)
    var fold = F - 1
    for k in range(F):
        var a = _cnt_mod(S + tot, k, F) - _cnt_mod(S, k, F)
        if slot < a:
            fold = k
            break
        slot -= a
    st(f, p(q, 8) + row, Float32(fold))


# ---------------------------------------------------------------- 247-248 integral test, small sums
@always_inline
def _integral_bits(b: UInt64, wide: Bool) -> Bool:
    """Whether float bits b (binary64 when wide, else binary32 in the low
    word) are finite and integer valued."""
    var mbits = 52 if wide else 23
    var bias = 1023 if wide else 127
    var emax = UInt64(0x7FF) if wide else UInt64(0xFF)
    var e = (b >> UInt64(mbits)) & emax
    if e == emax:
        return False
    var mag = b & ((UInt64(1) << UInt64(mbits + (11 if wide else 8))) - UInt64(1))
    var E = Int(e) - bias
    if E < 0:
        return mag == UInt64(0)
    if E >= mbits:
        return True
    var frac = (UInt64(1) << UInt64(mbits - E)) - UInt64(1)
    return (b & frac) == UInt64(0)


def c2_integral_unit(t: Int, f: FP, q: IP):
    """q = [RAW, KIND, n, CH, CNT]; t = chunk of CH values (KIND 0 float32,
    one word a value; 4 float64, two words): CNT[t] (int32 bits) = how many
    are not finite or not integer valued (`reduce_stat`'s integral test)."""
    var n = p(q, 2)
    var ch = p(q, 3)
    var wide = p(q, 1) == 4
    var k = 0
    for i in range(t * ch, min(n, t * ch + ch)):
        var b = _get_u64(f, p(q, 0) + 2 * i) if wide else _u32(f, p(q, 0) + i)
        if not _integral_bits(b, wide):
            k += 1
    sti(f, p(q, 4) + t, k)


def c2_isum_unit(t: Int, f: FP, q: IP):
    """q = [SRC, m, OUT]; one unit: OUT (int32 bits) = the sum of the m
    int32 words at SRC (chunk partials, a handful of words)."""
    var s = 0
    for i in range(p(q, 1)):
        s += ldi(f, p(q, 0) + i)
    sti(f, p(q, 2), s)


# ---------------------------------------------------------------- 249-251 infrequent categories
def c2_inf_m0_unit(t: Int, f: FP, q: IP):
    """q = [CNT, KC, kmax, THR, M0]; t = j*kmax + i: M0 (int32 bits) = 1
    when category i < KC[j] of column j has fewer than THR rows (the
    reference's min_frequency mask; THR = 0 without one), else 0."""
    var kmax = p(q, 2)
    var j = t // kmax
    var i = t % kmax
    var k = Int(ld(f, p(q, 1) + j))
    var m = 1 if (i < k and ldi(f, p(q, 0) + t) < p(q, 3)) else 0
    sti(f, p(q, 4) + t, m)


def c2_inf_m1_unit(t: Int, f: FP, q: IP):
    """q = [CNT, KC, kmax, M0, MAXC, M1]; t = j*kmax + i: the reference's
    `_identify_infrequent`: with MAXC >= 1 below the frequent count + 1,
    the categories before the last MAXC - 1 of a stable ascending sort by
    count are infrequent too (all of them when MAXC is 1)."""
    var kmax = p(q, 2)
    var j = t // kmax
    var i = t % kmax
    var k = Int(ld(f, p(q, 1) + j))
    var base = j * kmax
    if i >= k:
        sti(f, p(q, 5) + t, 0)
        return
    var m = ldi(f, p(q, 3) + t)
    var maxc = p(q, 4)
    var nm = 0
    for a in range(k):
        nm += ldi(f, p(q, 3) + base + a)
    if maxc >= 1 and maxc < k - nm + 1:
        var keep = maxc - 1
        if keep == 0:
            m = 1
        else:
            var ci = ldi(f, p(q, 0) + base + i)
            var rank = 0
            for a in range(k):
                var ca = ldi(f, p(q, 0) + base + a)
                if ca < ci or (ca == ci and a < i):
                    rank += 1
            if rank < k - keep:
                m = 1
    sti(f, p(q, 5) + t, m)


def c2_inf_map_unit(t: Int, f: FP, q: IP):
    """q = [KC, kmax, M1, MAP]; t = j*kmax + i: MAP (int32 bits) = the
    grouped code of category i (the frequent ones in order, every
    infrequent one the last code); -1 past the column's categories."""
    var kmax = p(q, 1)
    var j = t // kmax
    var i = t % kmax
    var k = Int(ld(f, p(q, 0) + j))
    if i >= k:
        sti(f, p(q, 3) + t, -1)
        return
    var base = j * kmax
    var ninf = 0
    var before = 0
    for a in range(k):
        var ma = ldi(f, p(q, 2) + base + a)
        ninf += ma
        if a < i and ma == 0:
            before += 1
    sti(f, p(q, 3) + t, (k - ninf) if ldi(f, p(q, 2) + t) != 0 else before)


# ---------------------------------------------------------------- 252 SimpleImputer
def c2_imp_stats_unit(t: Int, f: FP, q: IP):
    """q = [CNT, SRC, d, KEEP, CONST, FV, STATS, FILL]; t = column. CNT is
    col_stats' count row; SRC the strategy's statistic row (unused when
    CONST). An empty column's statistic is NaN, or with KEEP 0 (or the
    constant); its fill 0 (or the constant). Else both the statistic."""
    var empty = ld(f, p(q, 0) + t) == Float32(0)
    var konst = p(q, 4) != 0
    var s = raw(f, p(q, 5)) if konst else raw(f, p(q, 1) + t)
    var stat = s
    var fill = s
    if empty and not konst:
        fill = Float32(0)
        stat = Float32(0)
    if empty and p(q, 3) == 0:
        stat = canonical_nan()
    f.unsafe_store(p(q, 6) + t, stat)
    f.unsafe_store(p(q, 7) + t, fill)


# ---------------------------------------------------------------- 253-257 keys, top-k, thresholds, RFE
def c2_key64_unit(t: Int, f: FP, q: IP):
    """q = [RAW, KIND, KEY]; t = value: KEY[2t], KEY[2t + 1] = the ordered
    64-bit key of value t (`_okey`: a NaN orders as -inf)."""
    _put_u64(f, p(q, 2) + 2 * t, _okey(_word64(f, p(q, 0), p(q, 1), t)))


@always_inline
def _rank(f: FP, K: Int, m: Int, t: Int) -> Int:
    """Position of key t in a stable ascending sort of the m keys at K."""
    var kt = _get_u64(f, K + 2 * t)
    var r = 0
    for a in range(m):
        var ka = _get_u64(f, K + 2 * a)
        if ka < kt or (ka == kt and a < t):
            r += 1
    return r


def c2_topk_unit(t: Int, f: FP, q: IP):
    """q = [KEY, d, k, MASK]; t = column: MASK = 1.0 when the column is
    among the k largest (stable ascending sort: the later of tied columns
    ranks higher), else 0.0."""
    var d = p(q, 1)
    st(f, p(q, 3) + t, Float32(1) if _rank(f, p(q, 0), d, t) >= d - p(q, 2) else Float32(0))


def c2_gt_unit(t: Int, f: FP, q: IP):
    """q = [KEY, THR, MASK]; t = column: MASK = 1.0 when value t exceeds
    the binary64 threshold at THR (two raw words; a NaN value never does)."""
    var thr = _okey(_get_u64(f, p(q, 1)))
    st(f, p(q, 2) + t, Float32(1) if _get_u64(f, p(q, 0) + 2 * t) > thr else Float32(0))


def c2_rfe_step_unit(t: Int, f: FP, q: IP):
    """q = [KEY, FEAT, m, DROP, SUP]; t = active feature r: when r is among
    the DROP weakest (stable ascending sort, the lower column first)
    SUP[FEAT[r]] = 0.0."""
    if _rank(f, p(q, 0), p(q, 2), t) < p(q, 3):
        st(f, p(q, 4) + ldi(f, p(q, 1) + t), Float32(0))


def c2_rfe_rank_unit(t: Int, f: FP, q: IP):
    """q = [SUP, RANK]; t = column: RANK[t] (a float count, exact below
    2^24) + 1 when the column is no longer supported."""
    if ld(f, p(q, 0) + t) == Float32(0):
        st(f, p(q, 1) + t, ld(f, p(q, 1) + t) + Float32(1))


# ---------------------------------------------------------------- 258-262 IterativeImputer
def c2_ii_miss_unit(t: Int, f: FP, q: IP):
    """q = [CNT, KEEP, n, MISS]; t = kept column c: MISS[c] (int32 bits) =
    n - the non-missing count of column KEEP[c] (col_stats' count row)."""
    var j = ldi(f, p(q, 1) + t)
    sti(f, p(q, 3) + t, p(q, 2) - Int(ld(f, p(q, 0) + j)))


def c2_ii_pos_unit(t: Int, f: FP, q: IP):
    """q = [MISS, dk, MODE, POS]; t = feature: its position in the
    reference's fixed order (MODE 0 ascending: a stable argsort of the
    missing counts; 1 descending: that order reversed; 2 roman: as is; 3
    arabic: reversed)."""
    var dk = p(q, 1)
    var mode = p(q, 2)
    var pos = t
    if mode == 2:
        pos = t
    elif mode == 3:
        pos = dk - 1 - t
    else:
        var mt = ldi(f, p(q, 0) + t)
        var r = 0
        for a in range(dk):
            var ma = ldi(f, p(q, 0) + a)
            if ma < mt or (ma == mt and a < t):
                r += 1
        pos = r if mode == 0 else dk - 1 - r
    sti(f, p(q, 3) + t, pos)


def c2_ii_ord_unit(t: Int, f: FP, q: IP):
    """q = [MISS, dk, POS, rounds, ORD, LEN]; t = feature: a feature with
    missing entries goes to slot #{missing features before it in POS order}
    of every round's row ORD[r*dk ..] (int32 bits); feature 0 also writes
    every LEN[r] = the count of missing features."""
    var dk = p(q, 1)
    var rounds = p(q, 3)
    if t == 0:
        var nm = 0
        for a in range(dk):
            if ldi(f, p(q, 0) + a) > 0:
                nm += 1
        for r in range(rounds):
            sti(f, p(q, 5) + r, nm)
    if ldi(f, p(q, 0) + t) <= 0:
        return
    var pt = ldi(f, p(q, 2) + t)
    var slot = 0
    for a in range(dk):
        if ldi(f, p(q, 0) + a) > 0 and ldi(f, p(q, 2) + a) < pt:
            slot += 1
    for r in range(rounds):
        sti(f, p(q, 4) + r * dk + slot, t)


def c2_ii_rand_unit(t: Int, f: FP, q: IP):
    """q = [MISS, dk, SKIP, SEED, ORD, LEN]; t = round: the reference's
    'random' order. The candidates (every feature, or with SKIP those with
    missing entries) in a Fisher-Yates permutation drawn from the
    instance's splitmix64 stream (round t takes words t*(m-1) .. of it, m
    candidates: the same draws `_draw` made in turn), then the features
    with missing entries kept in that order; LEN[t] their count."""
    var dk = p(q, 1)
    var row = p(q, 4) + t * dk
    var m = 0
    for a in range(dk):
        if p(q, 2) == 0 or ldi(f, p(q, 0) + a) > 0:
            sti(f, row + m, a)
            m += 1
    var seed = _seed(f, p(q, 3))
    var at = t * max(m - 1, 0)
    for i in range(m - 1, 0, -1):
        var z = splitmix_at(seed, at)
        at += 1
        var k = Int(z % UInt64(i + 1))
        var a = ldi(f, row + i)
        sti(f, row + i, ldi(f, row + k))
        sti(f, row + k, a)
    var out = 0
    for i in range(m):
        var j = ldi(f, row + i)
        if ldi(f, p(q, 0) + j) > 0:
            sti(f, row + out, j)
            out += 1
    sti(f, p(q, 5) + t, out)


def c2_colmax_unit(t: Int, f: FP, q: IP):
    """q = [SRC, d, OUT]; one unit: OUT = the largest of the d words (NaN
    skipped; 0.0 when none)."""
    var best = Float32(0)
    var seen = False
    for i in range(p(q, 1)):
        var v = ld(f, p(q, 0) + i)
        if v == v and (not seen or v > best):
            best = v
            seen = True
    st(f, p(q, 2), best)


# ---------------------------------------------------------------- 263 level grids
def c2_grid_unit(t: Int, f: FP, q: IP):
    """q = [NB, w, KIND, OUT]; t = j*w + i, w the row stride: b = NB[j];
    in binary64, rounded to float32 once (the Python list's arithmetic,
    then the float32 upload):
      KIND 0 (KBinsDiscretizer): i * (100 / b) for i < b, 100 at i = b;
      KIND 1 (quantile references): i / b (0 when b = 0);
      KIND 2 (SplineTransformer weighted): 100 * (i * (1 / b)) for i < b,
        100 at i = b.
    Past b: 0."""
    var w = p(q, 1)
    var j = t // w
    var i = t % w
    var b = Int(ld(f, p(q, 0) + j))
    var kind = p(q, 2)
    var v = Float32(0)
    if i < b:
        var fi = sf64_from_int(i)
        var fb = sf64_from_int(b)
        if kind == 0:
            v = sf64_to_f32(sf64_mul(fi, sf64_div(sf64_from_int(100), fb)))
        elif kind == 1:
            v = sf64_to_f32(sf64_div(fi, fb))
        else:
            v = sf64_to_f32(sf64_mul(sf64_from_int(100), sf64_mul(fi, sf64_div(sf64_from_int(1), fb))))
    elif i == b:
        v = Float32(1) if kind == 1 else Float32(100)
        if kind == 1 and b == 0:
            v = Float32(0)
    f.unsafe_store(p(q, 3) + t, v)


# ---------------------------------------------------------------- 264-269 scalers
def c2_nan_sub_unit(t: Int, f: FP, q: IP):
    """q = [A, B, UA, UB, FA, FB, OA, OB, CN, NF, NP]; t = column. colnan =
    (UA and A[t] is NaN) or (UB and B[t] is NaN); OA, OB = FA, FB there,
    else A, B. CN = colnan, NF = OA or OB not finite, NP = OB <= 0 (int32
    bits; the caller sums them)."""
    var a = raw(f, p(q, 0) + t)
    var b = raw(f, p(q, 1) + t)
    var cn = (p(q, 2) != 0 and a != a) or (p(q, 3) != 0 and b != b)
    if cn:
        a = raw(f, p(q, 4))
        b = raw(f, p(q, 5))
    f.unsafe_store(p(q, 6) + t, a)
    f.unsafe_store(p(q, 7) + t, b)
    var nf = not (isfinite(a) and isfinite(b))
    sti(f, p(q, 8) + t, 1 if cn else 0)
    sti(f, p(q, 9) + t, 1 if nf else 0)
    sti(f, p(q, 10) + t, 1 if (not nf and not (b > Float32(0))) else 0)


def c2_nan_rows_unit(t: Int, f: FP, q: IP):
    """q = [SRC, d, CN, OUT]; t = r*d + c: OUT = NaN where CN[c] (int32
    bits) is set, else SRC bit for bit."""
    var c = t % p(q, 1)
    if ldi(f, p(q, 2) + c) != 0:
        f.unsafe_store(p(q, 3) + t, canonical_nan())
    else:
        f.unsafe_store(p(q, 3) + t, raw(f, p(q, 0) + t))


def c2_cnt0_unit(t: Int, f: FP, q: IP):
    """q = [CNT, CN]; t = column: CN (int32 bits) = 1 when the count is 0."""
    sti(f, p(q, 1) + t, 1 if ld(f, p(q, 0) + t) == Float32(0) else 0)


def c2_mm_keep_unit(t: Int, f: FP, q: IP):
    """q = [SRC, d, CN, OUT, BAD]; t = column of MinMaxScaler's five fitted
    rows (data_min, data_max, data_range, scale, min) at SRC: BAD[t] (int32
    bits) = 1 when one of them is not finite or the scale is not positive;
    OUT = the rows with NaN in the CN columns."""
    var d = p(q, 1)
    var bad = False
    for r in range(5):
        if not isfinite(raw(f, p(q, 0) + r * d + t)):
            bad = True
    if not (raw(f, p(q, 0) + 3 * d + t) > Float32(0)):
        bad = True
    sti(f, p(q, 4) + t, 1 if bad else 0)
    var cn = ldi(f, p(q, 2) + t) != 0
    for r in range(5):
        f.unsafe_store(p(q, 3) + r * d + t, canonical_nan() if cn else raw(f, p(q, 0) + r * d + t))


def c2_mm_merge_unit(t: Int, f: FP, q: IP):
    """q = [LO_OLD, HI_OLD, LO_NEW, HI_NEW, d, OUT, CN]; t = column: the
    reference's np.minimum / np.maximum merge rows. colnan = either minimum
    is NaN (CN, int32 bits); OUT's four rows (old min, old max, new min,
    new max) hold 0 there, else the values bit for bit."""
    var d = p(q, 4)
    var lo_o = raw(f, p(q, 0) + t)
    var lo_n = raw(f, p(q, 2) + t)
    var cn = lo_o != lo_o or lo_n != lo_n
    sti(f, p(q, 6) + t, 1 if cn else 0)
    for r in range(4):
        f.unsafe_store(p(q, 5) + r * d + t, Float32(0) if cn else raw(f, p(q, r) + t))


def c2_add_i64_unit(t: Int, f: FP, q: IP):
    """q = [A, B, OUT]; t = element: OUT = A + B over int64 words (two
    int32 words each, little-endian)."""
    _put_u64(f, p(q, 2) + 2 * t, _get_u64(f, p(q, 0) + 2 * t) + _get_u64(f, p(q, 1) + 2 * t))


@always_inline
def run_c2_unit[K: Int](t: Int, f: FP, q: IP):
    """K = op - F2_BASE - C2_FIRST."""
    comptime if K == 0:
        c2_bin_code_unit(t, f, q)
    comptime if K == 1:
        c2_kfold_unit(t, f, q)
    comptime if K == 2:
        c2_strat_meta_unit(t, f, q)
    comptime if K == 3:
        c2_strat_flag_unit(t, f, q)
    comptime if K == 4:
        c2_strat_fold_unit(t, f, q)
    comptime if K == 5:
        c2_integral_unit(t, f, q)
    comptime if K == 6:
        c2_isum_unit(t, f, q)
    comptime if K == 7:
        c2_inf_m0_unit(t, f, q)
    comptime if K == 8:
        c2_inf_m1_unit(t, f, q)
    comptime if K == 9:
        c2_inf_map_unit(t, f, q)
    comptime if K == 10:
        c2_imp_stats_unit(t, f, q)
    comptime if K == 11:
        c2_key64_unit(t, f, q)
    comptime if K == 12:
        c2_topk_unit(t, f, q)
    comptime if K == 13:
        c2_gt_unit(t, f, q)
    comptime if K == 14:
        c2_rfe_step_unit(t, f, q)
    comptime if K == 15:
        c2_rfe_rank_unit(t, f, q)
    comptime if K == 16:
        c2_ii_miss_unit(t, f, q)
    comptime if K == 17:
        c2_ii_pos_unit(t, f, q)
    comptime if K == 18:
        c2_ii_ord_unit(t, f, q)
    comptime if K == 19:
        c2_ii_rand_unit(t, f, q)
    comptime if K == 20:
        c2_colmax_unit(t, f, q)
    comptime if K == 21:
        c2_grid_unit(t, f, q)
    comptime if K == 22:
        c2_nan_sub_unit(t, f, q)
    comptime if K == 23:
        c2_nan_rows_unit(t, f, q)
    comptime if K == 24:
        c2_cnt0_unit(t, f, q)
    comptime if K == 25:
        c2_mm_keep_unit(t, f, q)
    comptime if K == 26:
        c2_mm_merge_unit(t, f, q)
    comptime if K == 27:
        c2_add_i64_unit(t, f, q)
