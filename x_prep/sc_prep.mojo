# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Count-data preprocessing as x_prep units (lane sc-prep-1, 2026-11-28): the row-total normalizer with a column
subset, the log transform, the clipped scaler, the variable-column recipes (seurat, cell_ranger, seurat_v3) and the
covariate regression.

References: Scanpy 1.10 `scanpy/preprocessing/_normalization.py` (`normalize_total`, `_normalize_data`),
`_simple.py` (`log1p`, `scale`, `regress_out`), `_highly_variable_genes.py` (`_highly_variable_genes_single_batch`,
`_highly_variable_genes_seurat_v3`, `_stats_seurat`, `_stats_cell_ranger`, `_subset_genes`); Cleveland and Devlin
1988 (loess, degree 2, tricube weights); numpy `percentile` (linear, `_lerp`) and `pinv`.

The units run in the prep lane's program model (x_prep/common.mojo): one source, the device launches one thread per
unit, the host column runs the same function over its pool, so CPU == GPU bit for bit by construction. Each unit's
loops run in ascending index order and that order is the fold order on every column. Every float a unit stores goes
through `st` (x_prep/common.mojo), every int word through `sti`.

Layouts. A dense X is n x d float32 row-major. A CSR X is INDPTR (n + 1 int words), INDICES and DATA (nnz words
each, int and float); a unit with a MODE word reads dense (0) or CSR (1). Per-batch gene arrays are b*d + g. The
column statistics read a gene-major, batch-ordered block G (dc genes x n rows, `sp_gather_*`): the door builds it in
blocks of genes, a memory bound only (every gene's units read only its column, so the blocks change no bit).

Ranks are counted (`sp_rank`: the position of a key among the keys of its batch and group, ties by index): one
thread per gene, O(d) each. They are the order every median, percentile and top-n choice reads.
"""
from checks.numerics import ftz, portable_log1pf
from x_prep.common import FP, IP, p, ld, raw, st, ldi, sti, is_nan, canonical_nan, canon, key
from x_prep.blocked import csr_dense_unit
from x_prep.prims import add, sub, mul, div, logf, expf, sqrtf

#: The op range of these units (after lane sc-stats-1's 280 .. 290, x_prep/markers.mojo).
comptime SP_BASE = 300
comptime SP_N = 47

comptime SP_ZERO = SP_BASE + 0
comptime SP_INCLUDE = SP_BASE + 1
comptime SP_ROW_SUM = SP_BASE + 2
comptime SP_EXCL_RULE = SP_BASE + 3
comptime SP_MEDIAN_POS = SP_BASE + 4
comptime SP_SCALE_ROWS = SP_BASE + 5
comptime SP_LOG1P = SP_BASE + 6
comptime SP_ORDER = SP_BASE + 7
comptime SP_GATHER_DENSE = SP_BASE + 8
comptime SP_GATHER_CSR = SP_BASE + 9
comptime SP_SEG_STATS = SP_BASE + 10
comptime SP_SEG_CLIP = SP_BASE + 11
comptime SP_DISP = SP_BASE + 12
comptime SP_MINMAX = SP_BASE + 13
comptime SP_BIN_SEURAT = SP_BASE + 14
comptime SP_PERCENTILES = SP_BASE + 15
comptime SP_BIN_CR = SP_BASE + 16
comptime SP_BIN_STATS = SP_BASE + 17
comptime SP_RANK = SP_BASE + 18
comptime SP_GROUP_START = SP_BASE + 19
comptime SP_SCATTER = SP_BASE + 20
comptime SP_SEG_MEDIAN = SP_BASE + 21
comptime SP_ABSDEV = SP_BASE + 22
comptime SP_NORM_DISP = SP_BASE + 23
comptime SP_HV_CUT = SP_BASE + 24
comptime SP_COMBINE = SP_BASE + 25
comptime SP_HV_BATCH = SP_BASE + 26
comptime SP_V3_KEY = SP_BASE + 27
comptime SP_LOESS = SP_BASE + 28
comptime SP_COUNT = SP_BASE + 29
comptime SP_V3_CLIPVAL = SP_BASE + 30
comptime SP_V3_NORMVAR = SP_BASE + 31
comptime SP_V3_SELECT = SP_BASE + 32
comptime SP_V3_HV = SP_BASE + 33
comptime SP_SCAN_PART = SP_BASE + 34
comptime SP_SCAN_TOP = SP_BASE + 35
comptime SP_SCAN_FILL = SP_BASE + 36
comptime SP_COMPACT = SP_BASE + 37
comptime SP_COL_SELECT = SP_BASE + 38
comptime SP_CSR_SELECT = SP_BASE + 39
comptime SP_SCALE_APPLY = SP_BASE + 40
comptime SP_SCALE_CSR = SP_BASE + 41
comptime SP_PINV = SP_BASE + 42
comptime SP_DENSIFY = SP_BASE + 43
comptime SP_RESID = SP_BASE + 44
#: The rank's pairs and chunk sorts (lane sc-leftovers-1; THE RANK, below).
comptime SP_RANK_KEYS = SP_BASE + 45
comptime SP_RANK_SORT = SP_BASE + 46

#: Input layout words.
comptime SP_DENSE = 0
comptime SP_CSR = 1
#: Gather transforms (PRE).
comptime SP_PRE_NONE = 0
comptime SP_PRE_EXPM1 = 1
#: -0.0 -> +0.0 and one NaN word (`canon`): equal values get equal sort keys (rank_genes_groups' block)
comptime SP_PRE_CANON = 2
#: Variable-column flavors.
comptime SP_SEURAT = 0
comptime SP_CELL_RANGER = 1
comptime SP_SEURAT_V3 = 2
#: Rows a scan block counts (`sp_scan_*`).
comptime SP_SCAN_BLOCK = 1024
#: Cell Ranger's percentile edges: 10, 15, .., 100 (19 edges, 20 bins).
comptime SP_CR_EDGES = 19

comptime _LN10 = Float32(2.3025851)
comptime _MAD_SCALE = Float32(0.6744898)  # scipy.stats.norm.ppf(0.75), statsmodels' `mad` normalization
comptime _MEAN_FLOOR = Float32(1.0e-12)    # Scanpy: mean[mean == 0] = 1e-12
comptime _F32_EPS = Float32(1.1920929e-07)


@always_inline
def is_sp_op(op: Int) -> Bool:
    return op >= SP_BASE and op < SP_BASE + SP_N


@always_inline
def _nan() -> Float32:
    return canonical_nan()


@always_inline
def _nan_to_num(v: Float32) -> Float32:
    """numpy `nan_to_num` on a float32: NaN -> 0 (infinities keep their order against every finite cut)."""
    if is_nan(v):
        return Float32(0)
    return v


@always_inline
def _log1p(x: Float32) -> Float32:
    return ftz(portable_log1pf(ftz(x)))


@always_inline
def expm1f(x: Float32) -> Float32:
    """expm1 from exp and log (Kahan's form): u = exp(x); x when u == 1, -1 when u - 1 == -1, else
    (u - 1) * x / log(u)."""
    var u = expf(x)
    if u == Float32(1):
        return x
    var um = sub(u, Float32(1))
    if um == Float32(-1):
        return Float32(-1)
    return div(mul(um, x), logf(u))


@always_inline
def _row_lo(f: FP, indptr: Int, r: Int) -> Int:
    return ldi(f, indptr + r)


# ---------------------------------------------------------------- small helpers


def sp_zero_unit(t: Int, f: FP, q: IP):
    """q = [OUT, ONE]; t = word: OUT[t] = ONE (0 or 1) as a float."""
    f.unsafe_store(p(q, 0) + t, Float32(p(q, 1)))


def sp_include_unit(t: Int, f: FP, q: IP):
    """q = [IDX, EXCL]; t = i: EXCL[IDX[i]] = 0 (the column subset of the row totals; EXCL starts all ones)."""
    f.unsafe_store(p(q, 1) + ldi(f, p(q, 0) + t), Float32(0))


# ---------------------------------------------------------------- the row totals


def sp_row_sum_unit(t: Int, f: FP, q: IP):
    """q = [MODE, X, INDPTR, INDICES, n, d, EXCL, OUT]; t = row. OUT[t] = the row's sum over the columns c with
    EXCL[c] == 0 (EXCL < 0: every column), columns ascending (dense) or in stored order (CSR)."""
    var mode = p(q, 0)
    var X = p(q, 1)
    var d = p(q, 5)
    var excl = p(q, 6)
    var s = Float32(0)
    if mode == SP_DENSE:
        for c in range(d):
            if excl < 0 or raw(f, excl + c) == Float32(0):
                s = add(s, ld(f, X + t * d + c))
    else:
        var ix = p(q, 3)
        var lo = _row_lo(f, p(q, 2), t)
        var hi = _row_lo(f, p(q, 2), t + 1)
        for e in range(lo, hi):
            if excl < 0 or raw(f, excl + ldi(f, ix + e)) == Float32(0):
                s = add(s, ld(f, X + e))
    st(f, p(q, 7) + t, s)


def sp_excl_rule_unit(t: Int, f: FP, q: IP):
    """q = [MODE, X, INDPTR, INDICES, n, d, SUMS, FRAC, EXCL]; t = row. Scanpy's `exclude_highly_expressed`: a
    column holding more than FRAC (a float word) of this row's total SUMS[t] is marked EXCL[c] = 1. A column is
    marked by every row that finds it and every mark writes the same word 1.0, so the marks are the same whatever
    order the rows run in (EXCL starts with the caller's subset: 0 counted, 1 left out)."""
    var mode = p(q, 0)
    var X = p(q, 1)
    var d = p(q, 5)
    var cut = mul(ld(f, p(q, 6) + t), ld(f, p(q, 7)))
    var excl = p(q, 8)
    if mode == SP_DENSE:
        for c in range(d):
            if ld(f, X + t * d + c) > cut:
                f.unsafe_store(excl + c, Float32(1))
    else:
        var ix = p(q, 3)
        var lo = _row_lo(f, p(q, 2), t)
        var hi = _row_lo(f, p(q, 2), t + 1)
        for e in range(lo, hi):
            if ld(f, X + e) > cut:
                f.unsafe_store(excl + ldi(f, ix + e), Float32(1))


def sp_median_pos_unit(t: Int, f: FP, q: IP):
    """q = [S, n, OUT]; t = 0. S holds n totals sorted ascending (`sort_cols`, NaN last). OUT = the median of the
    totals above zero (numpy median: the mean of the two middle values), NaN when none is."""
    var S = p(q, 0)
    var n = p(q, 1)
    var lo = 0
    var hi = n
    while lo < hi:  # the first total above zero
        var mid = (lo + hi) // 2
        var v = ld(f, S + mid)
        if v > Float32(0) and not is_nan(v):
            hi = mid
        else:
            lo = mid + 1
    var a = lo
    var b = n
    while a < b:  # the first NaN after them
        var mid = (a + b) // 2
        if is_nan(raw(f, S + mid)):
            b = mid
        else:
            a = mid + 1
    var m = a - lo
    if m <= 0:
        f.unsafe_store(p(q, 2), _nan())
        return
    var k = lo + m // 2
    if m % 2 == 1:
        st(f, p(q, 2), ld(f, S + k))
    else:
        st(f, p(q, 2), mul(add(ld(f, S + k - 1), ld(f, S + k)), Float32(0.5)))


def sp_scale_rows_unit(t: Int, f: FP, q: IP):
    """q = [MODE, X, INDPTR, n, d, SUMS, TARGET, OUT]; t = row. Scanpy's `_normalize_data`: a zero total counts as
    one, the row's scale is total / TARGET, every value of the row (the left-out columns too) is divided by it.
    Dense: OUT is n x d; CSR: OUT is the row's stored values (the structure is the input's)."""
    var mode = p(q, 0)
    var X = p(q, 1)
    var d = p(q, 4)
    var c = ld(f, p(q, 5) + t)
    if c == Float32(0):
        c = Float32(1)
    var s = div(c, ld(f, p(q, 6)))
    var o = p(q, 7)
    if mode == SP_DENSE:
        for j in range(d):
            st(f, o + t * d + j, div(ld(f, X + t * d + j), s))
    else:
        var lo = _row_lo(f, p(q, 2), t)
        var hi = _row_lo(f, p(q, 2), t + 1)
        for e in range(lo, hi):
            st(f, o + e, div(ld(f, X + e), s))


# ---------------------------------------------------------------- the log transform


def sp_log1p_unit(t: Int, f: FP, q: IP):
    """q = [X, BASE, OUT]; t = value (dense: every value; CSR: the stored values, so log1p(0) = 0 keeps the
    structure). OUT = log1p(x), divided by log(BASE) when BASE (a float word offset) is not -1."""
    var y = _log1p(ld(f, p(q, 0) + t))
    if p(q, 1) >= 0:
        y = div(y, logf(ld(f, p(q, 1))))
    st(f, p(q, 2) + t, y)


# ---------------------------------------------------------------- the gene-major block


def sp_order_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, B, START, PERM]; t = batch b. START[b] = the rows whose code is below b (thread B - 1 also
    writes START[B] = n), PERM[START[b] ..] = the rows of batch b ascending. CODES < 0: one batch, PERM the
    identity (the rows in order)."""
    var codes = p(q, 0)
    var n = p(q, 1)
    var k = p(q, 2)
    var start = p(q, 3)
    var perm = p(q, 4)
    if codes < 0:
        sti(f, start, 0)
        sti(f, start + 1, n)
        for i in range(n):
            sti(f, perm + i, i)
        return
    var lo = 0
    for i in range(n):
        if ldi(f, codes + i) < t:
            lo += 1
    sti(f, start + t, lo)
    if t == k - 1:
        sti(f, start + k, n)
    var w = lo
    for i in range(n):
        if ldi(f, codes + i) == t:
            sti(f, perm + w, i)
            w += 1


@always_inline
def _pre(v: Float32, pre: Int) -> Float32:
    if pre == SP_PRE_EXPM1:
        return expm1f(v)
    if pre == SP_PRE_CANON:
        return canon(v)
    return v


def sp_gather_dense_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, c0, PERM, G, PRE]; t = c*n + pos. G[t] = pre(X[PERM[pos], c0 + c])."""
    var n = p(q, 1)
    var d = p(q, 2)
    var c = t // n
    var pos = t - c * n
    var row = ldi(f, p(q, 4) + pos)
    st(f, p(q, 5) + t, _pre(ld(f, p(q, 0) + row * d + p(q, 3) + c), p(q, 6)))


def sp_gather_csr_unit(t: Int, f: FP, q: IP):
    """q = [INDPTR, INDICES, DATA, n, c0, dc, PERM, G, PRE]; t = pos. Row PERM[pos]'s values of the columns
    [c0, c0 + dc) into G[c*n + pos]: zeros, each stored entry added in stored order (a repeated column sums, as
    SciPy's), then pre (expm1(0) = 0, so the zeros stay zeros)."""
    var n = p(q, 3)
    var c0 = p(q, 4)
    var dc = p(q, 5)
    var g = p(q, 7)
    var pre = p(q, 8)
    var row = ldi(f, p(q, 6) + t)
    for c in range(dc):
        f.unsafe_store(g + c * n + t, Float32(0))
    var lo = ldi(f, p(q, 0) + row)
    var hi = ldi(f, p(q, 0) + row + 1)
    for e in range(lo, hi):
        var c = ldi(f, p(q, 1) + e) - c0
        if c >= 0 and c < dc:
            var at = g + c * n + t
            f.unsafe_store(at, add(raw(f, at), ld(f, p(q, 2) + e)))
    if pre != SP_PRE_NONE:
        for c in range(dc):
            var at = g + c * n + t
            st(f, at, _pre(raw(f, at), pre))


@always_inline
def seg_sum(f: FP, base: Int, lo: Int, hi: Int, s_in: Float32) -> Float32:
    """s_in + the words base[lo .. hi), ascending (the first pass of every segment mean in this file and in
    x_prep/markers.mojo; a second range continues the fold through s_in)."""
    var s = s_in
    for i in range(lo, hi):
        s = add(s, ld(f, base + i))
    return s


@always_inline
def seg_ss(f: FP, base: Int, lo: Int, hi: Int, mean: Float32, s_in: Float32) -> Float32:
    """s_in + the squared deviations of base[lo .. hi) about `mean`, ascending (the second pass)."""
    var s = s_in
    for i in range(lo, hi):
        var e = sub(ld(f, base + i), mean)
        s = add(s, mul(e, e))
    return s


def sp_seg_stats_unit(t: Int, f: FP, q: IP):
    """q = [G, n, dc, START, NSEG, ALL, OUT]; t = s*dc + c, s < NSEG + ALL. Segment s < NSEG is batch s's rows
    [START[s], START[s + 1]); s == NSEG (ALL = 1) is every row. Two passes (the mean, then the squared deviations,
    rows in batch order): OUT[s*dc + c] the mean, OUT[NS*dc + s*dc + c] the variance with ddof 1 (Scanpy's
    `_get_mean_var`); no row: both NaN, one row: the variance NaN."""
    var G = p(q, 0)
    var n = p(q, 1)
    var dc = p(q, 2)
    var start = p(q, 3)
    var nseg = p(q, 4)
    var ns = nseg + p(q, 5)
    var s = t // dc
    var c = t - s * dc
    var lo = 0
    var hi = n
    if s < nseg:
        lo = ldi(f, start + s)
        hi = ldi(f, start + s + 1)
    var base = G + c * n
    var o = p(q, 6)
    var cnt = hi - lo
    if cnt <= 0:
        f.unsafe_store(o + s * dc + c, _nan())
        f.unsafe_store(o + ns * dc + s * dc + c, _nan())
        return
    var mean = div(seg_sum(f, base, lo, hi, Float32(0)), Float32(cnt))
    st(f, o + s * dc + c, mean)
    if cnt < 2:
        f.unsafe_store(o + ns * dc + s * dc + c, _nan())
        return
    var ss = seg_ss(f, base, lo, hi, mean, Float32(0))
    st(f, o + ns * dc + s * dc + c, div(ss, Float32(cnt - 1)))


def sp_seg_clip_unit(t: Int, f: FP, q: IP):
    """q = [G, n, dc, START, NSEG, CLIP, c0, d, OUT]; t = s*dc + c. Seurat v3's clipped sums over batch s's rows:
    x clipped at CLIP[s*d + c0 + c]; OUT[s*dc + c] the sum, OUT[NSEG*dc + s*dc + c] the sum of squares."""
    var G = p(q, 0)
    var n = p(q, 1)
    var dc = p(q, 2)
    var start = p(q, 3)
    var nseg = p(q, 4)
    var s = t // dc
    var c = t - s * dc
    var clip = ld(f, p(q, 5) + s * p(q, 7) + p(q, 6) + c)
    var lo = ldi(f, start + s)
    var hi = ldi(f, start + s + 1)
    var base = G + c * n
    var sm = Float32(0)
    var sq = Float32(0)
    for i in range(lo, hi):
        var v = ld(f, base + i)
        if v > clip:
            v = clip
        sm = add(sm, v)
        sq = add(sq, mul(v, v))
    st(f, p(q, 8) + s * dc + c, sm)
    st(f, p(q, 8) + nseg * dc + s * dc + c, sq)


# ---------------------------------------------------------------- seurat and cell_ranger


def sp_disp_unit(t: Int, f: FP, q: IP):
    """q = [MEAN, VAR, FLAVOR, OUTM, OUTD]; t = b*d + g. Scanpy: a zero mean becomes 1e-12; dispersion = var /
    mean; seurat: a zero dispersion is NaN, the dispersion its log and the mean log1p(mean)."""
    var m = ld(f, p(q, 0) + t)
    var v = ld(f, p(q, 1) + t)
    if m == Float32(0):
        m = _MEAN_FLOOR
    var disp = div(v, m)
    if p(q, 2) == SP_SEURAT:
        if disp == Float32(0):
            disp = _nan()
        elif not is_nan(disp):
            disp = logf(disp)
        m = _log1p(m)
    if is_nan(disp):
        f.unsafe_store(p(q, 4) + t, _nan())
    else:
        st(f, p(q, 4) + t, disp)
    st(f, p(q, 3) + t, m)


def sp_minmax_unit(t: Int, f: FP, q: IP):
    """q = [M, d, OUT]; t = batch b. OUT[2b], OUT[2b + 1] = the minimum and maximum of batch b's non-NaN means."""
    var M = p(q, 0) + t * p(q, 1)
    var lo = _nan()
    var hi = _nan()
    for g in range(p(q, 1)):
        var v = ld(f, M + g)
        if is_nan(v):
            continue
        if is_nan(lo) or v < lo:
            lo = v
        if is_nan(hi) or v > hi:
            hi = v
    f.unsafe_store(p(q, 2) + 2 * t, lo)
    f.unsafe_store(p(q, 2) + 2 * t + 1, hi)


def sp_bin_seurat_unit(t: Int, f: FP, q: IP):
    """q = [M, d, MINMAX, NBINS, BIN]; t = b*d + g. pandas `cut(mean, bins=NBINS)`: NBINS equal-width bins over
    [min, max], the lowest edge moved down by 0.1% of the range (a constant range: both ends moved by 0.1% of
    |min|, or 0.001), right-inclusive; BIN the bin index (an int word), -1 for a NaN mean."""
    var d = p(q, 1)
    var b = t // d
    var v = ld(f, p(q, 0) + t)
    if is_nan(v):
        sti(f, p(q, 4) + t, -1)
        return
    var mn = ld(f, p(q, 2) + 2 * b)
    var mx = ld(f, p(q, 2) + 2 * b + 1)
    var nb = p(q, 3)
    if mn == mx:
        var adj = mul(abs(mn), Float32(0.001)) if mn != Float32(0) else Float32(0.001)
        mn = sub(mn, adj)
        mx = add(mx, adj)
    else:
        var adj = mul(sub(mx, mn), Float32(0.001))
        var first = sub(mn, adj)
        var step = div(sub(mx, mn), Float32(nb))
        var below = 0
        if first < v:
            below = 1
        for i in range(1, nb):
            if add(mn, mul(Float32(i), step)) < v:
                below += 1
        sti(f, p(q, 4) + t, min(max(below - 1, 0), nb - 1))
        return
    var step = div(sub(mx, mn), Float32(nb))
    var below = 0
    for i in range(nb):
        if add(mn, mul(Float32(i), step)) < v:
            below += 1
    sti(f, p(q, 4) + t, min(max(below - 1, 0), nb - 1))


@always_inline
def _valid_count(f: FP, S: Int, d: Int) -> Int:
    """The non-NaN words of a block sorted NaN last: the first NaN's index."""
    var a = 0
    var b = d
    while a < b:
        var mid = (a + b) // 2
        if is_nan(raw(f, S + mid)):
            b = mid
        else:
            a = mid + 1
    return a


def sp_percentiles_unit(t: Int, f: FP, q: IP):
    """q = [S, d, OUT]; t = b*19 + k. numpy `percentile(mean, 10 + 5k)` (linear, `_lerp`) of batch b's sorted
    non-NaN means S[b*d ..]; NaN when there is none."""
    var d = p(q, 1)
    var b = t // SP_CR_EDGES
    var k = t - b * SP_CR_EDGES
    var S = p(q, 0) + b * d
    var m = _valid_count(f, S, d)
    if m == 0:
        f.unsafe_store(p(q, 2) + t, _nan())
        return
    var pos = div(mul(Float32(m - 1), Float32(10 + 5 * k)), Float32(100))
    var lo = Int(pos)
    if lo > m - 1:
        lo = m - 1
    var frac = sub(pos, Float32(lo))
    var hi = min(lo + 1, m - 1)
    var a = ld(f, S + lo)
    var c = ld(f, S + hi)
    var diff = sub(c, a)
    var r: Float32
    if frac >= Float32(0.5):
        r = sub(c, mul(diff, sub(Float32(1), frac)))
    else:
        r = add(a, mul(diff, frac))
    st(f, p(q, 2) + t, r)


def sp_bin_cr_unit(t: Int, f: FP, q: IP):
    """q = [M, d, PCT, BIN]; t = b*d + g. Cell Ranger's bins: edges -inf, the 19 percentiles, +inf, right-inclusive:
    BIN = the percentiles below the mean (0 .. 19), -1 for a NaN mean."""
    var d = p(q, 1)
    var b = t // d
    var v = ld(f, p(q, 0) + t)
    if is_nan(v):
        sti(f, p(q, 3) + t, -1)
        return
    var below = 0
    for k in range(SP_CR_EDGES):
        if ld(f, p(q, 2) + b * SP_CR_EDGES + k) < v:
            below += 1
    sti(f, p(q, 3) + t, below)


def sp_bin_stats_unit(t: Int, f: FP, q: IP):
    """q = [D, BIN, d, NBINS, AVG, DEV]; t = b*NBINS + k. Seurat's bin statistics over batch b's genes in bin k
    with a non-NaN dispersion, genes ascending: the mean and the standard deviation (ddof 1). Scanpy's
    `_postprocess_dispersions_seurat`: a bin whose deviation is NaN (one gene) takes DEV = its mean, AVG = 0."""
    var d = p(q, 2)
    var nb = p(q, 3)
    var b = t // nb
    var k = t - b * nb
    var D = p(q, 0) + b * d
    var B = p(q, 1) + b * d
    var cnt = 0
    var sm = Float32(0)
    for g in range(d):
        var v = ld(f, D + g)
        if ldi(f, B + g) == k and not is_nan(v):
            cnt += 1
            sm = add(sm, v)
    var avg = _nan()
    var dev = _nan()
    if cnt > 0:
        avg = div(sm, Float32(cnt))
    if cnt > 1:
        var ss = Float32(0)
        for g in range(d):
            var v = ld(f, D + g)
            if ldi(f, B + g) == k and not is_nan(v):
                var e = sub(v, avg)
                ss = add(ss, mul(e, e))
        dev = sqrtf(div(ss, Float32(cnt - 1)))
    if is_nan(dev):
        dev = avg
        avg = Float32(0)
    if is_nan(avg):
        f.unsafe_store(p(q, 4) + t, _nan())
    else:
        st(f, p(q, 4) + t, avg)
    if is_nan(dev):
        f.unsafe_store(p(q, 5) + t, _nan())
    else:
        st(f, p(q, 5) + t, dev)


# THE RANK (lane sc-leftovers-1, 2026-11-28). RANK[b*d + g] = the genes of batch b in g's group (GROUP < 0: one
# group) with a non-NaN key that sort before g's (ascending, or descending when DESC; equal keys, -0.0 == +0.0, by
# index), -1 for a NaN key or a group below zero. Before, one unit per gene counted the whole batch: d^2 work per
# batch (sp_rank_unit, recoverable at this lane's parent sha). Now three stages (`sp_rank_stages`,
# bindings/prep_scprep_run.mojo), split across every core and the device's threads:
#   SP_RANK_KEYS  one unit per gene: its (order word, gene) pair;
#   SP_RANK_SORT  one unit per chunk of a batch (at most SP_RANK_CHUNKS_MAX chunks of at least SP_RANK_CHUNK_MIN
#                 genes): the chunk heap-sorted by (excluded, group, order word, gene);
#   SP_RANK       one unit per gene: per chunk, the genes before it minus the genes of a lower group (two binary
#                 searches), summed over the batch's chunks.
# The rank is an exact count under a total order, so the words are the counted form's on every column and every
# chunk count: the bits do not change. Work per batch: d log d for the sorts, d x chunks x log(chunk) for the
# searches (a size rule: the chunk length grows with d so the chunk count stays bounded).
comptime SP_RANK_CHUNK_MIN = 1024
comptime SP_RANK_CHUNKS_MAX = 64


@always_inline
def sp_rank_chunk_len(d: Int) -> Int:
    """Genes per sorted chunk of a batch of `d` (module note, THE RANK)."""
    var wide = (d + SP_RANK_CHUNKS_MAX - 1) // SP_RANK_CHUNKS_MAX
    return wide if wide > SP_RANK_CHUNK_MIN else SP_RANK_CHUNK_MIN


@always_inline
def sp_rank_chunks(d: Int) -> Int:
    """Sorted chunks of a batch of `d` (at least one)."""
    var w = sp_rank_chunk_len(d)
    var c = (d + w - 1) // w
    return c if c > 1 else 1


@always_inline
def _rank_class(f: FP, kb: Int, grp: Int, gb: Int, g: Int) -> UInt64:
    """Gene g's (excluded, group) as one word: 1 << 32 for a NaN key or a group below zero (after every ranked
    gene), else its group (0 for one group)."""
    var mine = 0
    if grp >= 0:
        mine = ldi(f, gb + g)
    if is_nan(ld(f, kb + g)) or mine < 0:
        return UInt64(1) << 32
    return UInt64(UInt32(mine))


@always_inline
def _rank_order(value: Float32, desc: Bool) -> UInt32:
    """A value's order word: `key` with -0.0 folded to +0.0 (equal values tie), complemented when `desc`."""
    var x = value
    if x == Float32(0):
        x = Float32(0)
    var k = key(x)
    return ~k if desc else k


@always_inline
def _rank_word(f: FP, o: Int) -> UInt64:
    """A pair's (order word, gene) as one word."""
    var w = f.bitcast[UInt32]()
    return (UInt64(w.unsafe_load(o)) << 32) | UInt64(w.unsafe_load(o + 1))


@always_inline
def _rank_less(f: FP, kb: Int, grp: Int, gb: Int, o: Int, cls: UInt64, word: UInt64) -> Bool:
    """The pair at word `o` sorts before the key (cls, word)."""
    var w = f.bitcast[UInt32]()
    var c = _rank_class(f, kb, grp, gb, Int(w.unsafe_load(o + 1)))
    if c != cls:
        return c < cls
    return _rank_word(f, o) < word


def sp_rank_keys_unit(t: Int, f: FP, q: IP):
    """q = [KEY, d, DESC, PAIRS]; t = b*d + g. PAIRS[2t], PAIRS[2t + 1] = (order word, g): `key` of the value
    (-0.0 folded to +0.0, so equal values tie), complemented when DESC."""
    var d = p(q, 1)
    var g = t % d
    var w = f.bitcast[UInt32]()
    w.unsafe_store(p(q, 3) + 2 * t, _rank_order(ld(f, p(q, 0) + t), p(q, 2) != 0))
    w.unsafe_store(p(q, 3) + 2 * t + 1, UInt32(g))


@always_inline
def _rank_sift(f: FP, kb: Int, grp: Int, gb: Int, base: Int, start: Int, n: Int):
    var w = f.bitcast[UInt32]()
    var root = start
    while True:
        var child = 2 * root + 1
        if child >= n:
            return
        if child + 1 < n:
            var oc = base + 2 * (child + 1)
            if _rank_less(f, kb, grp, gb, base + 2 * child, _rank_class(f, kb, grp, gb, Int(w.unsafe_load(oc + 1))),
                          _rank_word(f, oc)):
                child += 1
        var oc2 = base + 2 * child
        if not _rank_less(f, kb, grp, gb, base + 2 * root,
                          _rank_class(f, kb, grp, gb, Int(w.unsafe_load(oc2 + 1))), _rank_word(f, oc2)):
            return
        var a0 = w.unsafe_load(base + 2 * root)
        var a1 = w.unsafe_load(base + 2 * root + 1)
        w.unsafe_store(base + 2 * root, w.unsafe_load(oc2))
        w.unsafe_store(base + 2 * root + 1, w.unsafe_load(oc2 + 1))
        w.unsafe_store(oc2, a0)
        w.unsafe_store(oc2 + 1, a1)
        root = child


def sp_rank_sort_unit(t: Int, f: FP, q: IP):
    """q = [KEY, GROUP, d, PAIRS]; t = b*C + c (C = `sp_rank_chunks(d)`). Chunk c of batch b's pairs heap-sorted
    ascending by (class, order word, gene) (module note, THE RANK). The order is total (the gene breaks every
    tie), so the sorted chunk has one answer."""
    var d = p(q, 2)
    var nc = sp_rank_chunks(d)
    var b = t // nc
    var c = t - b * nc
    var cl = sp_rank_chunk_len(d)
    var lo = c * cl
    var n = min(d - lo, cl)
    if n < 2:
        return
    var kb = p(q, 0) + b * d
    var grp = p(q, 1)
    var gb = grp + b * d
    var w = f.bitcast[UInt32]()
    var base = p(q, 3) + 2 * (b * d + lo)
    var start = n // 2 - 1
    while start >= 0:
        _rank_sift(f, kb, grp, gb, base, start, n)
        start -= 1
    var end = n - 1
    while end > 0:
        var a0 = w.unsafe_load(base)
        var a1 = w.unsafe_load(base + 1)
        w.unsafe_store(base, w.unsafe_load(base + 2 * end))
        w.unsafe_store(base + 1, w.unsafe_load(base + 2 * end + 1))
        w.unsafe_store(base + 2 * end, a0)
        w.unsafe_store(base + 2 * end + 1, a1)
        _rank_sift(f, kb, grp, gb, base, 0, end)
        end -= 1


@always_inline
def _rank_lower(f: FP, kb: Int, grp: Int, gb: Int, base: Int, n: Int, cls: UInt64, word: UInt64) -> Int:
    """The pairs of a sorted chunk (n at word `base`) before the key (cls, word): a binary search."""
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) // 2
        if _rank_less(f, kb, grp, gb, base + 2 * mid, cls, word):
            lo = mid + 1
        else:
            hi = mid
    return lo


def sp_rank_unit(t: Int, f: FP, q: IP):
    """q = [KEY, GROUP, d, DESC, RANK, PAIRS]; t = b*d + g. RANK (an int word) as the module note, THE RANK,
    states, from the chunks SP_RANK_SORT left in PAIRS: per chunk, its pairs before g's minus its pairs of a lower
    class (a lower group)."""
    var d = p(q, 2)
    var b = t // d
    var g = t - b * d
    var kb = p(q, 0) + b * d
    var grp = p(q, 1)
    var gb = grp + b * d
    var cls = _rank_class(f, kb, grp, gb, g)
    if cls == (UInt64(1) << 32):
        sti(f, p(q, 4) + t, -1)
        return
    var word = (UInt64(_rank_order(ld(f, kb + g), p(q, 3) != 0)) << 32) | UInt64(UInt32(g))
    var nc = sp_rank_chunks(d)
    var cl = sp_rank_chunk_len(d)
    var r = 0
    for c in range(nc):  # small-loop(the batch's sorted chunks, at most SP_RANK_CHUNKS_MAX)
        var lo = c * cl
        var n = min(d - lo, cl)
        var base = p(q, 5) + 2 * (b * d + lo)
        r += _rank_lower(f, kb, grp, gb, base, n, cls, word) - _rank_lower(f, kb, grp, gb, base, n, cls, UInt64(0))
    sti(f, p(q, 4) + t, r)


def sp_group_start_unit(t: Int, f: FP, q: IP):
    """q = [KEY, GROUP, d, NG, START]; t = b*(NG + 1) + k. START[t] = batch b's genes with a non-NaN key whose
    group is below k (k = NG: all of them), the start of group k's run in the batch's sorted block."""
    var d = p(q, 2)
    var ng = p(q, 3)
    var b = t // (ng + 1)
    var k = t - b * (ng + 1)
    var cnt = 0
    for g in range(d):
        var grp = ldi(f, p(q, 1) + b * d + g)
        if grp >= 0 and grp < k and not is_nan(ld(f, p(q, 0) + b * d + g)):
            cnt += 1
    sti(f, p(q, 4) + t, cnt)


def sp_scatter_unit(t: Int, f: FP, q: IP):
    """q = [KEY, GROUP, RANK, d, NG, START, SEG]; t = b*d + g. SEG[b*d + START[b*(NG + 1) + group] + rank] = key:
    each group's keys in sorted order (unranked genes write nothing)."""
    var d = p(q, 3)
    var ng = p(q, 4)
    var b = t // d
    var r = ldi(f, p(q, 2) + t)
    if r < 0:
        return
    var grp = ldi(f, p(q, 1) + t)
    var at = ldi(f, p(q, 5) + b * (ng + 1) + grp)
    f.unsafe_store(p(q, 6) + b * d + at + r, raw(f, p(q, 0) + t))


def sp_seg_median_unit(t: Int, f: FP, q: IP):
    """q = [SEG, d, NG, START, OUT]; t = b*NG + k. The median of group k's sorted run in batch b (pandas median:
    the mean of the two middle values), NaN for an empty group."""
    var d = p(q, 1)
    var ng = p(q, 2)
    var b = t // ng
    var k = t - b * ng
    var lo = ldi(f, p(q, 3) + b * (ng + 1) + k)
    var hi = ldi(f, p(q, 3) + b * (ng + 1) + k + 1)
    var m = hi - lo
    if m <= 0:
        f.unsafe_store(p(q, 4) + t, _nan())
        return
    var S = p(q, 0) + b * d + lo
    var mid = m // 2
    if m % 2 == 1:
        st(f, p(q, 4) + t, ld(f, S + mid))
    else:
        st(f, p(q, 4) + t, mul(add(ld(f, S + mid - 1), ld(f, S + mid)), Float32(0.5)))


def sp_absdev_unit(t: Int, f: FP, q: IP):
    """q = [D, BIN, MED, d, NG, OUT]; t = b*d + g. |D - its bin's median| (NaN for a NaN dispersion or no bin)."""
    var d = p(q, 3)
    var b = t // d
    var v = ld(f, p(q, 0) + t)
    var k = ldi(f, p(q, 1) + t)
    if is_nan(v) or k < 0:
        f.unsafe_store(p(q, 5) + t, _nan())
        return
    var med = ld(f, p(q, 2) + b * p(q, 4) + k)
    if is_nan(med):
        f.unsafe_store(p(q, 5) + t, _nan())
        return
    st(f, p(q, 5) + t, abs(sub(v, med)))


def sp_norm_disp_unit(t: Int, f: FP, q: IP):
    """q = [D, BIN, AVG, DEV, d, NG, FLAVOR, OUT]; t = b*d + g. The normalized dispersion (D - AVG) / DEV of the
    gene's bin; cell_ranger's DEV is the median absolute deviation over 0.6744898 (statsmodels `mad`)."""
    var d = p(q, 4)
    var b = t // d
    var v = ld(f, p(q, 0) + t)
    var k = ldi(f, p(q, 1) + t)
    if is_nan(v) or k < 0:
        f.unsafe_store(p(q, 7) + t, _nan())
        return
    var at = b * p(q, 5) + k
    var avg = ld(f, p(q, 2) + at)
    var dev = ld(f, p(q, 3) + at)
    if p(q, 6) == SP_CELL_RANGER:
        dev = div(dev, _MAD_SCALE)
    var r = div(sub(v, avg), dev)
    if is_nan(r):
        f.unsafe_store(p(q, 7) + t, _nan())
    else:
        st(f, p(q, 7) + t, r)


def sp_hv_cut_unit(t: Int, f: FP, q: IP):
    """q = [M, DN, d, NTOP, CUT, HV]; t = b*d + g. Scanpy's `_subset_genes` in batch b, HV an int word 1 / 0.
    NTOP >= 0: the gene's nan_to_num(DN) is at least the NTOP-th largest non-NaN DN, i.e. fewer than NTOP non-NaN
    DN of the batch are above it (ties at the cut all kept, as Scanpy's `>= disp_cut_off`). NTOP < 0: CUT[0..3]
    (float words min_mean, max_mean, min_disp, max_disp) as open bounds on the mean and nan_to_num(DN)."""
    var d = p(q, 2)
    var b = t // d
    var ntop = p(q, 3)
    var v = _nan_to_num(ld(f, p(q, 1) + t))
    var hv = False
    if ntop >= 0:
        var above = 0
        for h in range(d):
            var w = ld(f, p(q, 1) + b * d + h)
            if not is_nan(w) and w > v:
                above += 1
        hv = above < ntop
    else:
        var cut = p(q, 4)
        var m = ld(f, p(q, 0) + t)
        hv = (m > ld(f, cut) and m < ld(f, cut + 1) and v > ld(f, cut + 2) and v < ld(f, cut + 3))
    sti(f, p(q, 5) + t, 1 if hv else 0)


def sp_combine_unit(t: Int, f: FP, q: IP):
    """q = [M, D, DN, HV, d, NB, OUTM, OUTD, OUTDN, NBAT]; t = g. Scanpy's batch merge: the mean over batches of
    the means, dispersions and normalized dispersions (NaN skipped, pandas `mean`), batches ascending; NBAT = the
    batches that selected the gene (an int word)."""
    var d = p(q, 4)
    var nb = p(q, 5)
    var g = t
    var nbat = 0
    comptime for which in range(3):
        var src = p(q, which)
        var cnt = 0
        var sm = Float32(0)
        for b in range(nb):
            var v = ld(f, src + b * d + g)
            if not is_nan(v):
                cnt += 1
                sm = add(sm, v)
        var dst = p(q, 6 + which) + g
        if cnt == 0:
            f.unsafe_store(dst, _nan())
        else:
            st(f, dst, div(sm, Float32(cnt)))
    for b in range(nb):
        nbat += ldi(f, p(q, 3) + b * d + g)
    sti(f, p(q, 9) + g, nbat)


@always_inline
def _dn_before(a: Float32, b: Float32) -> Bool:
    """Descending, NaN last."""
    if is_nan(a):
        return False
    if is_nan(b):
        return True
    return a > b


@always_inline
def _dn_equal(a: Float32, b: Float32) -> Bool:
    return (is_nan(a) and is_nan(b)) or a == b


def sp_hv_batch_unit(t: Int, f: FP, q: IP):
    """q = [NBAT, DN, M, d, NTOP, CUT, HV]; t = g. Scanpy's batched selection. NTOP >= 0: the genes sorted by
    (NBAT descending, DN descending, NaN last; a stable sort: ties by index), the first NTOP selected. NTOP < 0:
    the cutoffs CUT on the merged mean and the merged DN with NaN as 0."""
    var d = p(q, 3)
    var ntop = p(q, 4)
    var g = t
    var hv = False
    if ntop >= 0:
        var nbg = ldi(f, p(q, 0) + g)
        var dg = ld(f, p(q, 1) + g)
        var r = 0
        for h in range(d):
            if h == g:
                continue
            var nbh = ldi(f, p(q, 0) + h)
            var dh = ld(f, p(q, 1) + h)
            if nbh > nbg or (nbh == nbg and (_dn_before(dh, dg) or (_dn_equal(dh, dg) and h < g))):
                r += 1
        hv = r < ntop
    else:
        var cut = p(q, 5)
        var m = ld(f, p(q, 2) + g)
        var v = _nan_to_num(ld(f, p(q, 1) + g))
        hv = (m > ld(f, cut) and m < ld(f, cut + 1) and v > ld(f, cut + 2) and v < ld(f, cut + 3))
    sti(f, p(q, 6) + g, 1 if hv else 0)


# ---------------------------------------------------------------- seurat_v3


def sp_v3_key_unit(t: Int, f: FP, q: IP):
    """q = [MEAN, VAR, LX, LY]; t = b*d + g. A gene with variance above zero: LX = log10(mean), LY = log10(var);
    a constant gene: both NaN (Scanpy's `not_const`)."""
    var m = ld(f, p(q, 0) + t)
    var v = ld(f, p(q, 1) + t)
    if not (v > Float32(0)) or not (m > Float32(0)):
        f.unsafe_store(p(q, 2) + t, _nan())
        f.unsafe_store(p(q, 3) + t, _nan())
        return
    st(f, p(q, 2) + t, div(logf(m), _LN10))
    st(f, p(q, 3) + t, div(logf(v), _LN10))


@always_inline
def _det3(a: Float32, b: Float32, c: Float32, d: Float32, e: Float32, g: Float32, h: Float32, i: Float32,
          j: Float32) -> Float32:
    """The determinant of [[a, b, c], [d, e, g], [h, i, j]], the cofactors of the first row, left to right."""
    var c0 = sub(mul(e, j), mul(g, i))
    var c1 = sub(mul(d, j), mul(g, h))
    var c2 = sub(mul(d, i), mul(e, h))
    return add(sub(mul(a, c0), mul(b, c1)), mul(c, c2))


def sp_loess_unit(t: Int, f: FP, q: IP):
    """q = [LX, LY, SX, d, SPAN, EST]; t = b*d + g. The loess fit (degree 2, tricube weights, no robustness
    steps; Cleveland's direct surface) of LY on LX among batch b's non-NaN genes, evaluated at the gene's own LX:
    h = the distance to its q-th nearest neighbour (q = floor(SPAN * m), at least 3, at most m, read off the
    batch's sorted LX block SX), weights (1 - (|x - x0| / h)^3)^3 for |x - x0| < h, u = (x - x0) / h, the 3 x 3
    weighted normal equations solved by Cramer's rule, EST = the intercept (a singular system: the weighted mean).
    A NaN LX (a constant gene): EST = 0 (Scanpy's `estimat_var` zeros)."""
    var d = p(q, 3)
    var b = t // d
    var x0 = ld(f, p(q, 0) + t)
    if is_nan(x0):
        st(f, p(q, 5) + t, Float32(0))
        return
    var SX = p(q, 2) + b * d
    var m = _valid_count(f, SX, d)
    var qq = Int(mul(ld(f, p(q, 4)), Float32(m)))
    if qq < 3:
        qq = 3
    if qq > m:
        qq = m
    # the first sorted x at or above x0
    var lo = 0
    var hi = m
    while lo < hi:
        var mid = (lo + hi) // 2
        if ld(f, SX + mid) < x0:
            lo = mid + 1
        else:
            hi = mid
    var left = lo - 1
    var right = lo
    var h = Float32(0)
    for _ in range(qq):
        var dl = Float32(0)
        var dr = Float32(0)
        var take_left: Bool
        if left < 0:
            take_left = False
        elif right >= m:
            take_left = True
        else:
            dl = sub(x0, ld(f, SX + left))
            dr = sub(ld(f, SX + right), x0)
            take_left = dl <= dr
        if take_left:
            h = sub(x0, ld(f, SX + left))
            left -= 1
        else:
            h = sub(ld(f, SX + right), x0)
            right += 1
    var X = p(q, 0) + b * d
    var Y = p(q, 1) + b * d
    var s0 = Float32(0)
    var s1 = Float32(0)
    var s2 = Float32(0)
    var s3 = Float32(0)
    var s4 = Float32(0)
    var t0 = Float32(0)
    var t1 = Float32(0)
    var t2 = Float32(0)
    for k in range(d):
        var x = ld(f, X + k)
        if is_nan(x):
            continue
        var dist = abs(sub(x, x0))
        if h > Float32(0):
            if not (dist < h):
                continue
        elif dist != Float32(0):
            continue
        var w = Float32(1)
        var u = Float32(0)
        if h > Float32(0):
            u = div(sub(x, x0), h)
            var r = div(dist, h)
            var c = sub(Float32(1), mul(mul(r, r), r))
            w = mul(mul(c, c), c)
        var y = ld(f, Y + k)
        var wu = mul(w, u)
        var wu2 = mul(wu, u)
        s0 = add(s0, w)
        s1 = add(s1, wu)
        s2 = add(s2, wu2)
        s3 = add(s3, mul(wu2, u))
        s4 = add(s4, mul(mul(wu2, u), u))
        t0 = add(t0, mul(w, y))
        t1 = add(t1, mul(wu, y))
        t2 = add(t2, mul(wu2, y))
    var det = _det3(s0, s1, s2, s1, s2, s3, s2, s3, s4)
    var est: Float32
    if det == Float32(0) or is_nan(det):
        est = div(t0, s0)
    else:
        est = div(_det3(t0, s1, s2, t1, s2, s3, t2, s3, s4), det)
    st(f, p(q, 5) + t, est)


def sp_count_unit(t: Int, f: FP, q: IP):
    """q = [START, OUT]; t = batch b. OUT[b] = START[b + 1] - START[b] (the batch's rows, an int word)."""
    sti(f, p(q, 1) + t, ldi(f, p(q, 0) + t + 1) - ldi(f, p(q, 0) + t))


def sp_v3_clipval_unit(t: Int, f: FP, q: IP):
    """q = [MEAN, EST, NCNT, d, REGSTD, CLIP]; t = b*d + g. reg_std = sqrt(10^EST), clip = reg_std * sqrt(N_b) +
    mean (Scanpy's `clip_val`)."""
    var d = p(q, 3)
    var b = t // d
    var reg = sqrtf(expf(mul(ld(f, p(q, 1) + t), _LN10)))
    st(f, p(q, 4) + t, reg)
    st(f, p(q, 5) + t, add(mul(reg, sqrtf(Float32(ldi(f, p(q, 2) + b)))), ld(f, p(q, 0) + t)))


def sp_v3_normvar_unit(t: Int, f: FP, q: IP):
    """q = [MEAN, REGSTD, SUM, SQ, NCNT, d, OUT]; t = b*d + g. Scanpy's `norm_gene_var`:
    (N mean^2 + sq - 2 sum mean) / ((N - 1) reg_std^2)."""
    var d = p(q, 5)
    var b = t // d
    var nn = Float32(ldi(f, p(q, 4) + b))
    var m = ld(f, p(q, 0) + t)
    var reg = ld(f, p(q, 1) + t)
    var num = sub(add(mul(nn, mul(m, m)), ld(f, p(q, 3) + t)), mul(mul(Float32(2), ld(f, p(q, 2) + t)), m))
    st(f, p(q, 6) + t, div(num, mul(sub(nn, Float32(1)), mul(reg, reg))))


def sp_v3_select_unit(t: Int, f: FP, q: IP):
    """q = [RANK, NV, d, NB, NTOP, OUTRANK, OUTNB, OUTVN]; t = g. Over the batches: OUTNB the batches where the
    gene's rank (descending normalized variance) is below NTOP, OUTRANK the median of those ranks (np.ma.median:
    the mean of the two middle values; NaN when none), OUTVN the mean normalized variance."""
    var d = p(q, 2)
    var nb = p(q, 3)
    var ntop = p(q, 4)
    var g = t
    var cnt = 0
    var sm = Float32(0)
    for b in range(nb):
        sm = add(sm, ld(f, p(q, 1) + b * d + g))
        if ldi(f, p(q, 0) + b * d + g) < ntop:
            cnt += 1
    st(f, p(q, 7) + g, div(sm, Float32(nb)))
    sti(f, p(q, 6) + g, cnt)
    if cnt == 0:
        f.unsafe_store(p(q, 5) + g, _nan())
        return
    var k1 = (cnt - 1) // 2
    var k2 = cnt // 2
    var v1 = 0
    var v2 = 0
    for b in range(nb):
        var r = ldi(f, p(q, 0) + b * d + g)
        if r >= ntop:
            continue
        var pos = 0
        for b2 in range(nb):
            var r2 = ldi(f, p(q, 0) + b2 * d + g)
            if r2 < ntop and (r2 < r or (r2 == r and b2 < b)):
                pos += 1
        if pos == k1:
            v1 = r
        if pos == k2:
            v2 = r
    st(f, p(q, 5) + g, mul(add(Float32(v1), Float32(v2)), Float32(0.5)))


def sp_v3_hv_unit(t: Int, f: FP, q: IP):
    """q = [RANK, NBAT, d, NTOP, HV]; t = g. Scanpy's seurat_v3 choice: the genes sorted by (median rank
    ascending, NaN last; batches descending; a stable sort: ties by index), the first NTOP selected."""
    var d = p(q, 2)
    var g = t
    var rg = ld(f, p(q, 0) + g)
    var ng = ldi(f, p(q, 1) + g)
    var r = 0
    for h in range(d):
        if h == g:
            continue
        var rh = ld(f, p(q, 0) + h)
        var nh = ldi(f, p(q, 1) + h)
        var first: Bool
        if is_nan(rh) and is_nan(rg):
            first = nh > ng or (nh == ng and h < g)
        elif is_nan(rh):
            first = False
        elif is_nan(rg):
            first = True
        elif rh != rg:
            first = rh < rg
        else:
            first = nh > ng or (nh == ng and h < g)
        if first:
            r += 1
    sti(f, p(q, 4) + g, 1 if r < p(q, 3) else 0)


# ---------------------------------------------------------------- prefix sums and column selection


def sp_scan_part_unit(t: Int, f: FP, q: IP):
    """q = [CNT, n, PART]; t = block. PART[t] = the sum of the int words CNT[t*1024 .. min(n, (t+1)*1024))."""
    var lo = t * SP_SCAN_BLOCK
    var hi = min(p(q, 1), lo + SP_SCAN_BLOCK)
    var s = 0
    for i in range(lo, hi):
        s += ldi(f, p(q, 0) + i)
    sti(f, p(q, 2) + t, s)


def sp_scan_top_unit(t: Int, f: FP, q: IP):
    """q = [PART, nb]; t = 0. PART becomes its exclusive prefix sums (nb = ceil(n / 1024) blocks: a short serial
    pass over the block sums, the scan's middle step)."""
    var acc = 0
    for i in range(p(q, 1)):
        var v = ldi(f, p(q, 0) + i)
        sti(f, p(q, 0) + i, acc)
        acc += v
    sti(f, p(q, 0) + p(q, 1), acc)


def sp_scan_fill_unit(t: Int, f: FP, q: IP):
    """q = [CNT, n, PART, OUT]; t = block. OUT[i] = the exclusive prefix sum of CNT at i; the last block also
    writes OUT[n] = the total (OUT is n + 1 int words, a CSR INDPTR)."""
    var n = p(q, 1)
    var lo = t * SP_SCAN_BLOCK
    var hi = min(n, lo + SP_SCAN_BLOCK)
    var acc = ldi(f, p(q, 2) + t)
    for i in range(lo, hi):
        sti(f, p(q, 3) + i, acc)
        acc += ldi(f, p(q, 0) + i)
    if hi == n:
        sti(f, p(q, 3) + n, acc)


def sp_compact_unit(t: Int, f: FP, q: IP):
    """q = [HV, POS, IDX, INV]; t = g. POS the exclusive prefix of HV (int words): a kept gene g writes IDX[POS[g]] =
    g and INV[g] = POS[g]; a dropped gene INV[g] = -1."""
    if ldi(f, p(q, 0) + t) != 0:
        var at = ldi(f, p(q, 1) + t)
        sti(f, p(q, 2) + at, t)
        sti(f, p(q, 3) + t, at)
    else:
        sti(f, p(q, 3) + t, -1)


def sp_col_select_unit(t: Int, f: FP, q: IP):
    """q = [X, d, IDX, k, OUT]; t = i*k + j. OUT[t] = X[i*d + IDX[j]] (the word as it is)."""
    var k = p(q, 3)
    var i = t // k
    var j = t - i * k
    f.unsafe_store(p(q, 4) + t, raw(f, p(q, 0) + i * p(q, 1) + ldi(f, p(q, 2) + j)))


def sp_csr_select_unit(t: Int, f: FP, q: IP):
    """q = [INDPTR, INDICES, DATA, INV, OPTR, OIDX, ODATA, CNT]; t = row. Without CNT (< 0) the row's kept entries
    are written at OPTR[t] in stored order, each column renumbered by INV; with CNT the unit is
    `sp_csr_count_unit` (one op for both passes: q[7] >= 0 counts)."""
    var lo = ldi(f, p(q, 0) + t)
    var hi = ldi(f, p(q, 0) + t + 1)
    if p(q, 7) >= 0:
        var c = 0
        for e in range(lo, hi):
            if ldi(f, p(q, 3) + ldi(f, p(q, 1) + e)) >= 0:
                c += 1
        sti(f, p(q, 7) + t, c)
        return
    var w = ldi(f, p(q, 4) + t)
    for e in range(lo, hi):
        var nc = ldi(f, p(q, 3) + ldi(f, p(q, 1) + e))
        if nc >= 0:
            sti(f, p(q, 5) + w, nc)
            f.unsafe_store(p(q, 6) + w, raw(f, p(q, 2) + e))
            w += 1


# ---------------------------------------------------------------- the clipped scaler


@always_inline
def _clip(var v: Float32, mv: Float32, both: Bool) -> Float32:
    if v > mv:
        return mv
    if both and v < -mv:
        return -mv
    return v


def sp_scale_apply_unit(t: Int, f: FP, q: IP):
    """q = [X, d, MEAN, STD, ZC, MAXV, OUT]; t = i*d + c (dense). (x - mean) / std (without the centring when ZC
    is 0); MAXV (a float word, -1: none) clips at +max_value, and at -max_value too when centred (Scanpy 1.10)."""
    var c = t % p(q, 1)
    var v = ld(f, p(q, 0) + t)
    var zc = p(q, 4) != 0
    if zc:
        v = sub(v, ld(f, p(q, 2) + c))
    v = div(v, ld(f, p(q, 3) + c))
    if p(q, 5) >= 0:
        v = _clip(v, ld(f, p(q, 5)), zc)
    st(f, p(q, 6) + t, v)


def sp_scale_csr_unit(t: Int, f: FP, q: IP):
    """q = [DATA, INDICES, STD, MAXV, OUT]; t = stored entry. The uncentred scale of a CSR value, x / std of its
    column, clipped at +max_value (MAXV, -1: none); the structure is the input's."""
    var v = div(ld(f, p(q, 0) + t), ld(f, p(q, 2) + ldi(f, p(q, 1) + t)))
    if p(q, 3) >= 0:
        v = _clip(v, ld(f, p(q, 3)), False)
    st(f, p(q, 4) + t, v)


# ---------------------------------------------------------------- covariate regression


def sp_pinv_unit(t: Int, f: FP, q: IP):
    """q = [EVAL, EVEC, m, P]; t = i*m + j. The pseudo-inverse of the symmetric Gram from its eigenpairs (x_prep
    `eigh`: EVAL descending, EVEC[c*m + r] component c of eigenvector r): P[i, j] = sum over r with
    EVAL[r] > EVAL[0] * m * eps of V[i, r] V[j, r] / EVAL[r], r ascending (numpy pinv's cut, float32 eps)."""
    var m = p(q, 2)
    var i = t // m
    var j = t - i * m
    var W = p(q, 0)
    var V = p(q, 1)
    var top = abs(ld(f, W))
    var tol = mul(mul(top, Float32(m)), _F32_EPS)
    var s = Float32(0)
    for r in range(m):
        var lam = ld(f, W + r)
        if lam > tol:
            s = add(s, div(mul(ld(f, V + i * m + r), ld(f, V + j * m + r)), lam))
    st(f, p(q, 3) + t, s)


def sp_resid_unit(t: Int, f: FP, q: IP):
    """q = [R, d, Z, k, COEF]; t = i*d + j. R[t] -= sum_l Z[i*k + l] * COEF[l*d + j], l ascending, in place (the
    residual of the gene's least-squares fit on the standardized covariates; each product rounded before its add)."""
    var d = p(q, 1)
    var k = p(q, 3)
    var i = t // d
    var j = t - i * d
    var acc = Float32(0)
    for l in range(k):
        acc = add(acc, mul(ld(f, p(q, 2) + i * k + l), ld(f, p(q, 4) + l * d + j)))
    st(f, p(q, 0) + t, sub(ld(f, p(q, 0) + t), acc))


def run_sp_unit[OP: Int](t: Int, f: FP, q: IP):
    comptime if OP == SP_ZERO:
        sp_zero_unit(t, f, q)
    comptime if OP == SP_INCLUDE:
        sp_include_unit(t, f, q)
    comptime if OP == SP_ROW_SUM:
        sp_row_sum_unit(t, f, q)
    comptime if OP == SP_EXCL_RULE:
        sp_excl_rule_unit(t, f, q)
    comptime if OP == SP_MEDIAN_POS:
        sp_median_pos_unit(t, f, q)
    comptime if OP == SP_SCALE_ROWS:
        sp_scale_rows_unit(t, f, q)
    comptime if OP == SP_LOG1P:
        sp_log1p_unit(t, f, q)
    comptime if OP == SP_ORDER:
        sp_order_unit(t, f, q)
    comptime if OP == SP_GATHER_DENSE:
        sp_gather_dense_unit(t, f, q)
    comptime if OP == SP_GATHER_CSR:
        sp_gather_csr_unit(t, f, q)
    comptime if OP == SP_SEG_STATS:
        sp_seg_stats_unit(t, f, q)
    comptime if OP == SP_SEG_CLIP:
        sp_seg_clip_unit(t, f, q)
    comptime if OP == SP_DISP:
        sp_disp_unit(t, f, q)
    comptime if OP == SP_MINMAX:
        sp_minmax_unit(t, f, q)
    comptime if OP == SP_BIN_SEURAT:
        sp_bin_seurat_unit(t, f, q)
    comptime if OP == SP_PERCENTILES:
        sp_percentiles_unit(t, f, q)
    comptime if OP == SP_BIN_CR:
        sp_bin_cr_unit(t, f, q)
    comptime if OP == SP_BIN_STATS:
        sp_bin_stats_unit(t, f, q)
    comptime if OP == SP_RANK:
        sp_rank_unit(t, f, q)
    comptime if OP == SP_RANK_KEYS:
        sp_rank_keys_unit(t, f, q)
    comptime if OP == SP_RANK_SORT:
        sp_rank_sort_unit(t, f, q)
    comptime if OP == SP_GROUP_START:
        sp_group_start_unit(t, f, q)
    comptime if OP == SP_SCATTER:
        sp_scatter_unit(t, f, q)
    comptime if OP == SP_SEG_MEDIAN:
        sp_seg_median_unit(t, f, q)
    comptime if OP == SP_ABSDEV:
        sp_absdev_unit(t, f, q)
    comptime if OP == SP_NORM_DISP:
        sp_norm_disp_unit(t, f, q)
    comptime if OP == SP_HV_CUT:
        sp_hv_cut_unit(t, f, q)
    comptime if OP == SP_COMBINE:
        sp_combine_unit(t, f, q)
    comptime if OP == SP_HV_BATCH:
        sp_hv_batch_unit(t, f, q)
    comptime if OP == SP_V3_KEY:
        sp_v3_key_unit(t, f, q)
    comptime if OP == SP_LOESS:
        sp_loess_unit(t, f, q)
    comptime if OP == SP_COUNT:
        sp_count_unit(t, f, q)
    comptime if OP == SP_V3_CLIPVAL:
        sp_v3_clipval_unit(t, f, q)
    comptime if OP == SP_V3_NORMVAR:
        sp_v3_normvar_unit(t, f, q)
    comptime if OP == SP_V3_SELECT:
        sp_v3_select_unit(t, f, q)
    comptime if OP == SP_V3_HV:
        sp_v3_hv_unit(t, f, q)
    comptime if OP == SP_SCAN_PART:
        sp_scan_part_unit(t, f, q)
    comptime if OP == SP_SCAN_TOP:
        sp_scan_top_unit(t, f, q)
    comptime if OP == SP_SCAN_FILL:
        sp_scan_fill_unit(t, f, q)
    comptime if OP == SP_COMPACT:
        sp_compact_unit(t, f, q)
    comptime if OP == SP_COL_SELECT:
        sp_col_select_unit(t, f, q)
    comptime if OP == SP_CSR_SELECT:
        sp_csr_select_unit(t, f, q)
    comptime if OP == SP_SCALE_APPLY:
        sp_scale_apply_unit(t, f, q)
    comptime if OP == SP_SCALE_CSR:
        sp_scale_csr_unit(t, f, q)
    comptime if OP == SP_PINV:
        sp_pinv_unit(t, f, q)
    comptime if OP == SP_DENSIFY:
        # q = [INDPTR, INDICES, DATA, d, OUT]; t = row: MultinomialNB's CSR -> dense body (x_prep/blocked.mojo)
        csr_dense_unit(t, f, q)
    comptime if OP == SP_RESID:
        sp_resid_unit(t, f, q)
