# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Marker-gene tests (`rank_genes_groups`) as x_prep units (lane sc-stats-1, 2026-11-28).

References: Scanpy 1.10 `scanpy/tools/_rank_genes_groups.py` (`_RankGenes.wilcoxon`, `t_test`,
`compute_statistics`, the log fold change `log2((expm1(mean_g) + 1e-9) / (expm1(mean_rest) + 1e-9))`),
Seurat `FindMarkers` (the same two tests), scipy `ttest_ind_from_stats(equal_var=False)`, statsmodels
`multipletests(method="fdr_bh")`.

The units run in the prep lane's program model (x_prep/common.mojo): one source, the device launches one thread
per unit, the host column runs the same function over its pool, so CPU == GPU bit for bit by construction. Each
unit's loops run in ascending order and that order is the fold order on every column.

A chunk program (bindings/prep_markers_run.mojo) covers the genes [c0, c0 + dc):

  rg_order      t = g        the group starts START (K + 1 int words) and the row order PERM (n int words): rows
                             by group code, ascending row inside a group (a stable counting order).
  rg_gather_*   the chunk's values in group order, gene-major: G[c * n + pos] = X[PERM[pos], c0 + c]
                (dense: one thread per value; CSR: one thread per row writes its row's dc values). Every
                value goes through `canon` (-0.0 -> +0.0), so equal values have equal sort keys.
  rg_seg_sort   t = g*dc + c each group's segment of gene c sorted (`heap_sort`, the lane's one sort order).
  rg_stats      t = j*dc + c two-pass mean and variance (ddof 1) of the group and of its rest (the other
                groups, or the reference group).
  rg_ranksum    the Wilcoxon rank sums by a merge of the sorted segments: the values are walked once in
                ascending order; a run of t equal values over the pooled cells takes the average rank, so a
                group holding r of them adds r * (2 * below + t + 1) to TWICE its rank sum, an exact integer.
                The tie term sum(t^3 - t) comes out of the same walk. With reference "rest" one thread per
                gene merges all K segments (every group's sum in one pass); with a reference group one
                thread per (group, gene) merges the two segments.
  rg_score      t = j*dc + c the score, the two-sided p-value and the log fold change.

then one program over the whole n_out x n_genes result: rg_transpose (gene-major chunks -> row-major), and the
Benjamini-Hochberg adjustment per group (rg_bh_sort, rg_bh_scan, rg_adjust).

The rank sums and the tie sums are 64-bit integers (two int words each in the arena), so they carry no rounding;
the statistic is formed from them in float32 through the portable operations (x_prep/prims.mojo), the p-value
tails are the univariate scores' `gammaincc` and `betainc` (x_prep/stats.mojo): Q(1/2, z^2 / 2) is the normal
two-sided p-value erfc(|z| / sqrt 2), and I_{dof / (dof + t^2)}(dof / 2, 1/2) the Student two-sided one.
"""
from std.memory import bitcast
from x_prep.common import FP, IP, p, ld, raw, st, ldi, sti, canon, key, heap_sort
from x_prep.prims import add, sub, mul, div, logf, expf, sqrtf
from x_prep.stats import betainc, gammaincc

#: The op range of these units (after F2_BASE + F2_N = 272, x_prep/fam2.mojo).
comptime SC_BASE = 280
comptime SC_N = 11

comptime SC_ORDER = SC_BASE + 0
comptime SC_GATHER_DENSE = SC_BASE + 1
comptime SC_GATHER_CSR = SC_BASE + 2
comptime SC_SEG_SORT = SC_BASE + 3
comptime SC_STATS = SC_BASE + 4
comptime SC_RANKSUM = SC_BASE + 5
comptime SC_SCORE = SC_BASE + 6
comptime SC_TRANSPOSE = SC_BASE + 7
comptime SC_BH_SORT = SC_BASE + 8
comptime SC_BH_SCAN = SC_BASE + 9
comptime SC_ADJUST = SC_BASE + 10

#: rg_score FLAGS bits.
comptime RG_WILCOXON = 1
comptime RG_TIE_CORRECT = 2
#: rg_adjust METHOD words.
comptime RG_ADJ_BH = 0
comptime RG_ADJ_BONFERRONI = 1

comptime _LN2 = Float32(0.6931472)
comptime _FC_EPS = Float32(1.0e-9)


@always_inline
def is_sc_op(op: Int) -> Bool:
    return op >= SC_BASE and op < SC_BASE + SC_N


@always_inline
def _st64(f: FP, i: Int, v: Int):
    """A 64-bit integer as two int words (low, high) at f[i], f[i + 1]."""
    var w = f.bitcast[UInt32]()
    w.unsafe_store(i, UInt32(v & 0xFFFFFFFF))
    w.unsafe_store(i + 1, UInt32((v >> 32) & 0xFFFFFFFF))


@always_inline
def _ld64(f: FP, i: Int) -> Int:
    var w = f.bitcast[UInt32]()
    var lo = Int(w.unsafe_load(i))
    var hi = Int(w.unsafe_load(i + 1))
    return (hi << 32) | lo


@always_inline
def _group_of(j: Int, ref_group: Int) -> Int:
    """Output row j's group: every group with reference "rest" (ref < 0), else every group but the reference."""
    if ref_group < 0 or j < ref_group:
        return j
    return j + 1


@always_inline
def _rows_out(k: Int, ref_group: Int) -> Int:
    return k if ref_group < 0 else k - 1


# ---------------------------------------------------------------- the group order


def rg_order_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, K, START, PERM]; t = group g. START[g] = the rows whose code is below g; PERM[START[g] ..] =
    the rows of group g, ascending. Thread K - 1 also writes START[K] = n (every code is in [0, K), the door's
    label encoding)."""
    var codes = p(q, 0)
    var n = p(q, 1)
    var k = p(q, 2)
    var start = p(q, 3)
    var perm = p(q, 4)
    var g = t
    var lo = 0
    for i in range(n):
        if ldi(f, codes + i) < g:
            lo += 1
    sti(f, start + g, lo)
    if g == k - 1:
        sti(f, start + k, n)
    var w = lo
    for i in range(n):
        if ldi(f, codes + i) == g:
            sti(f, perm + w, i)
            w += 1


def rg_gather_dense_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, c0, PERM, G]; t = c*n + pos. G[t] = canon(X[PERM[pos], c0 + c])."""
    var n = p(q, 1)
    var d = p(q, 2)
    var c = t // n
    var pos = t - c * n
    var row = ldi(f, p(q, 4) + pos)
    f.unsafe_store(p(q, 5) + t, canon(ld(f, p(q, 0) + row * d + p(q, 3) + c)))


def rg_gather_csr_unit(t: Int, f: FP, q: IP):
    """q = [INDPTR, INDICES, DATA, n, c0, dc, PERM, G]; t = pos. Row PERM[pos]'s values of the columns
    [c0, c0 + dc) into G[c*n + pos]: zeros, then each stored entry added in stored order (a repeated column sums,
    as SciPy's), then `canon`. INDPTR and INDICES are int32 words, DATA float32."""
    var n = p(q, 3)
    var c0 = p(q, 4)
    var dc = p(q, 5)
    var g = p(q, 7)
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
    for c in range(dc):
        var at = g + c * n + t
        f.unsafe_store(at, canon(raw(f, at)))


def rg_seg_sort_unit(t: Int, f: FP, q: IP):
    """q = [G, n, dc, START]; t = g*dc + c. Group g's segment of gene c sorted ascending (`heap_sort`)."""
    var n = p(q, 1)
    var dc = p(q, 2)
    var start = p(q, 3)
    var g = t // dc
    var c = t - g * dc
    var lo = ldi(f, start + g)
    var hi = ldi(f, start + g + 1)
    heap_sort(f, p(q, 0) + c * n + lo, hi - lo)


# ---------------------------------------------------------------- group and rest statistics


@always_inline
def _range_sum(f: FP, base: Int, lo: Int, hi: Int, s_in: Float32) -> Float32:
    var s = s_in
    for i in range(lo, hi):
        s = add(s, raw(f, base + i))
    return s


@always_inline
def _range_ss(f: FP, base: Int, lo: Int, hi: Int, mean: Float32, s_in: Float32) -> Float32:
    var s = s_in
    for i in range(lo, hi):
        var e = sub(raw(f, base + i), mean)
        s = add(s, mul(e, e))
    return s


def rg_stats_unit(t: Int, f: FP, q: IP):
    """q = [G, n, K, dc, START, REF, STATS]; t = j*dc + c. STATS holds four blocks of KO*dc words (KO the output
    rows): MEAN, VAR of group g = row j's group, RMEAN, RVAR of its rest. Two passes (the mean, then the squares
    about it), positions ascending; the rest of "rest" is the gene's column without g's segment (positions
    [0, START[g]) then [START[g+1], n)), of a reference group its segment. Variances are ddof 1 (Scanpy's
    `_get_mean_var`); a part of one value has variance 0 (the door refuses groups below two cells)."""
    var base = p(q, 0)
    var n = p(q, 1)
    var k = p(q, 2)
    var dc = p(q, 3)
    var start = p(q, 4)
    var ref_group = p(q, 5)
    var stats = p(q, 6)
    var ko = _rows_out(k, ref_group)
    var j = t // dc
    var c = t - j * dc
    var g = _group_of(j, ref_group)
    var col = base + c * n
    var lo = ldi(f, start + g)
    var hi = ldi(f, start + g + 1)
    var m = hi - lo
    var mean = Float32(0)
    var var_ = Float32(0)
    if m > 0:
        mean = div(_range_sum(f, col, lo, hi, Float32(0)), Float32(m))
    if m > 1:
        var_ = div(_range_ss(f, col, lo, hi, mean, Float32(0)), Float32(m - 1))
    var rmean = Float32(0)
    var rvar = Float32(0)
    var rm: Int
    if ref_group >= 0:
        var rlo = ldi(f, start + ref_group)
        var rhi = ldi(f, start + ref_group + 1)
        rm = rhi - rlo
        if rm > 0:
            rmean = div(_range_sum(f, col, rlo, rhi, Float32(0)), Float32(rm))
        if rm > 1:
            rvar = div(_range_ss(f, col, rlo, rhi, rmean, Float32(0)), Float32(rm - 1))
    else:
        rm = n - m
        if rm > 0:
            var s = _range_sum(f, col, 0, lo, Float32(0))
            s = _range_sum(f, col, hi, n, s)
            rmean = div(s, Float32(rm))
        if rm > 1:
            var ss = _range_ss(f, col, 0, lo, rmean, Float32(0))
            ss = _range_ss(f, col, hi, n, rmean, ss)
            rvar = div(ss, Float32(rm - 1))
    var block = ko * dc
    st(f, stats + t, mean)
    st(f, stats + block + t, var_)
    st(f, stats + 2 * block + t, rmean)
    st(f, stats + 3 * block + t, rvar)


# ---------------------------------------------------------------- Wilcoxon rank sums


@always_inline
def _member(a: Int, rest: Bool, g: Int, ref_group: Int) -> Int:
    """The a-th segment of a merge: every group (rest), else (g, the reference)."""
    if rest:
        return a
    return g if a == 0 else ref_group


def rg_ranksum_unit(t: Int, f: FP, q: IP):
    """q = [G, n, K, dc, START, REF, CUR, R2T]. Reference "rest" (REF < 0): t = c, one merge of all K segments
    of gene c; else t = j*dc + c, the merge of group g's and the reference's segments. CUR: 2 int words per
    merged segment per unit (its position and its count of the current value), at CUR + 2*K*c (rest) or
    CUR + 4*t. R2T: four int words per (j, c) at R2T + 4*(j*dc + c): twice the rank sum of row j's group, then
    the tie sum T = sum(t^3 - t) over the pooled values, each a 64-bit integer (`_st64`).

    The walk: the smallest head value v over the segments (sort keys compared; after `canon` a key is the
    value), the r_a cells of v in each segment counted and passed, t = sum r_a; a segment's group adds
    r_a * (2 * below + t + 1), `below` the pooled cells passed before v. Every segment is sorted by the one
    order, so the walk visits each pooled value once in ascending order on every column."""
    var base = p(q, 0)
    var n = p(q, 1)
    var k = p(q, 2)
    var dc = p(q, 3)
    var start = p(q, 4)
    var ref_group = p(q, 5)
    var cur = p(q, 6)
    var r2t = p(q, 7)
    var rest = ref_group < 0
    var c: Int
    var j = 0
    var g = 0
    var members: Int
    var cb: Int
    if rest:
        c = t
        members = k
        cb = cur + 2 * k * c
    else:
        j = t // dc
        c = t - j * dc
        g = _group_of(j, ref_group)
        members = 2
        cb = cur + 4 * t
    var col = base + c * n
    for a in range(members):
        sti(f, cb + 2 * a, ldi(f, start + _member(a, rest, g, ref_group)))
        sti(f, cb + 2 * a + 1, 0)
    # the rank-sum accumulators of the rows this unit owns
    if rest:
        for a in range(members):
            _st64(f, r2t + 4 * (a * dc + c), 0)
    var own = 0
    var below = 0
    var ties = 0
    while True:
        var found = False
        var best = UInt32(0)
        for a in range(members):
            var pos = ldi(f, cb + 2 * a)
            var end = ldi(f, start + _member(a, rest, g, ref_group) + 1)
            if pos < end:
                var kv = key(raw(f, col + pos))
                if not found or kv < best:
                    best = kv
                    found = True
        if not found:
            break
        var run = 0
        for a in range(members):
            var pos = ldi(f, cb + 2 * a)
            var end = ldi(f, start + _member(a, rest, g, ref_group) + 1)
            var r = 0
            while pos < end and key(raw(f, col + pos)) == best:
                pos += 1
                r += 1
            sti(f, cb + 2 * a, pos)
            sti(f, cb + 2 * a + 1, r)
            run += r
        var step = 2 * below + run + 1
        if rest:
            for a in range(members):
                var r = ldi(f, cb + 2 * a + 1)
                if r > 0:
                    var at = r2t + 4 * (a * dc + c)
                    _st64(f, at, _ld64(f, at) + r * step)
        else:
            own += ldi(f, cb + 1) * step
        ties += run * run * run - run
        below += run
    if rest:
        for a in range(members):
            _st64(f, r2t + 4 * (a * dc + c) + 2, ties)
    else:
        _st64(f, r2t + 4 * t, own)
        _st64(f, r2t + 4 * t + 2, ties)


# ---------------------------------------------------------------- scores, p-values, fold changes


@always_inline
def _log2_fold(mg: Float32, mr: Float32) -> Float32:
    """Scanpy's log2((expm1(mean_g) + 1e-9) / (expm1(mean_rest) + 1e-9)); expm1 as exp(x) - 1."""
    var a = add(sub(expf(mg), Float32(1)), _FC_EPS)
    var b = add(sub(expf(mr), Float32(1)), _FC_EPS)
    return div(logf(div(a, b)), _LN2)


def rg_score_unit(t: Int, f: FP, q: IP):
    """q = [STATS, START, K, dc, REF, FLAGS, R2T, n, OUT]; t = j*dc + c. OUT gene-major for the chunk: the score
    at OUT[c*KO + j], the p-value KO*dc words on, the log fold change 2*KO*dc on.

    Wilcoxon (FLAGS & RG_WILCOXON): z = (R - n_g (N + 1) / 2) / sqrt(n_g n_r (N + 1) / 12 [* (1 - T / (N^3 - N))
    with RG_TIE_CORRECT]), from the exact 2R; p = Q(1/2, z^2 / 2). Welch's t: (m_g - m_r) / sqrt(v_g / n_g +
    v_r / n_r) with the Welch-Satterthwaite degrees of freedom, p = I_{dof / (dof + t^2)}(dof / 2, 1/2). A zero
    standard error gives score 0 and p-value 1 (Scanpy sets the NaN of 0/0 so)."""
    var stats = p(q, 0)
    var start = p(q, 1)
    var k = p(q, 2)
    var dc = p(q, 3)
    var ref_group = p(q, 4)
    var flags = p(q, 5)
    var r2t = p(q, 6)
    var n = p(q, 7)
    var out = p(q, 8)
    var ko = _rows_out(k, ref_group)
    var block = ko * dc
    var j = t // dc
    var c = t - j * dc
    var g = _group_of(j, ref_group)
    var ng = ldi(f, start + g + 1) - ldi(f, start + g)
    var nr: Int
    if ref_group >= 0:
        nr = ldi(f, start + ref_group + 1) - ldi(f, start + ref_group)
    else:
        nr = n - ng
    var mg = ld(f, stats + t)
    var vg = ld(f, stats + block + t)
    var mr = ld(f, stats + 2 * block + t)
    var vr = ld(f, stats + 3 * block + t)
    var score = Float32(0)
    var pv = Float32(1)
    if (flags & RG_WILCOXON) != 0:
        var big_n = ng + nr
        var num2 = _ld64(f, r2t + 4 * t) - ng * (big_n + 1)
        var var_ = div(mul(mul(Float32(ng), Float32(nr)), Float32(big_n + 1)), Float32(12))
        if (flags & RG_TIE_CORRECT) != 0:
            var span = big_n * big_n * big_n - big_n
            if span > 0:
                var_ = mul(var_, sub(Float32(1), div(Float32(_ld64(f, r2t + 4 * t + 2)), Float32(span))))
        if var_ > Float32(0):
            var sd2 = mul(Float32(2), sqrtf(var_))
            score = div(Float32(num2), sd2)
            pv = gammaincc(Float32(0.5), mul(Float32(0.5), mul(score, score)))
    else:
        var eg = div(vg, Float32(ng))
        var er = div(vr, Float32(nr))
        var s = add(eg, er)
        if s > Float32(0):
            score = div(sub(mg, mr), sqrtf(s))
            var den = add(div(mul(eg, eg), Float32(ng - 1)), div(mul(er, er), Float32(nr - 1)))
            if den > Float32(0):
                var dof = div(mul(s, s), den)
                pv = betainc(mul(Float32(0.5), dof), Float32(0.5), div(dof, add(dof, mul(score, score))))
    if pv > Float32(1):
        pv = Float32(1)
    var at = out + c * ko + j
    st(f, at, score)
    st(f, at + block, pv)
    st(f, at + 2 * block, _log2_fold(mg, mr))


# ---------------------------------------------------------------- the whole result


def rg_transpose_unit(t: Int, f: FP, q: IP):
    """q = [IN, KO, d, OUT]; t = j*d + c. OUT[j, c] = IN[c, j], bit for bit (gene-major chunks to rows)."""
    var ko = p(q, 1)
    var d = p(q, 2)
    var j = t // d
    var c = t - j * d
    f.unsafe_store(p(q, 3) + t, raw(f, p(q, 0) + c * ko + j))


def rg_bh_sort_unit(t: Int, f: FP, q: IP):
    """q = [PV, d, S]; t = row j. S[j] = row j of PV sorted ascending (`heap_sort`). One thread per output row:
    the sort is d log d steps of a row whose length is the gene count, against the n x d work before it."""
    var d = p(q, 1)
    var src = p(q, 0) + t * d
    var dst = p(q, 2) + t * d
    for c in range(d):
        f.unsafe_store(dst + c, raw(f, src + c))
    heap_sort(f, dst, d)


def rg_bh_scan_unit(t: Int, f: FP, q: IP):
    """q = [S, d, Q]; t = row j. Q[j, l] = min(1, min over l' >= l of S[j, l'] * d / (l' + 1)): statsmodels'
    fdr_bh cumulative minimum from the largest p-value down."""
    var d = p(q, 1)
    var s = p(q, 0) + t * d
    var o = p(q, 2) + t * d
    var running = Float32(1)
    var l = d - 1
    while l >= 0:
        var v = div(mul(ld(f, s + l), Float32(d)), Float32(l + 1))
        if v < running:
            running = v
        st(f, o + l, running)
        l -= 1


def rg_adjust_unit(t: Int, f: FP, q: IP):
    """q = [PV, S, Q, d, METHOD, ADJ]; t = j*d + c. Benjamini-Hochberg: Q at the LAST sorted position holding
    PV[j, c] (tied p-values share the adjusted value there, as statsmodels' cumulative minimum gives them);
    Bonferroni: min(1, PV * d)."""
    var d = p(q, 3)
    var j = t // d
    var v = ld(f, p(q, 0) + t)
    var adj: Float32
    if p(q, 4) == RG_ADJ_BONFERRONI:
        adj = mul(v, Float32(d))
        if adj > Float32(1):
            adj = Float32(1)
    else:
        var s = p(q, 1) + j * d
        var kv = key(v)
        var lo = 0
        var hi = d
        while lo < hi:
            var mid = (lo + hi) // 2
            if key(raw(f, s + mid)) <= kv:
                lo = mid + 1
            else:
                hi = mid
        adj = ld(f, p(q, 2) + j * d + lo - 1)
    st(f, p(q, 5) + t, adj)


@always_inline
def run_sc_unit[OP: Int](t: Int, f: FP, q: IP):
    comptime if OP == SC_ORDER:
        rg_order_unit(t, f, q)
    comptime if OP == SC_GATHER_DENSE:
        rg_gather_dense_unit(t, f, q)
    comptime if OP == SC_GATHER_CSR:
        rg_gather_csr_unit(t, f, q)
    comptime if OP == SC_SEG_SORT:
        rg_seg_sort_unit(t, f, q)
    comptime if OP == SC_STATS:
        rg_stats_unit(t, f, q)
    comptime if OP == SC_RANKSUM:
        rg_ranksum_unit(t, f, q)
    comptime if OP == SC_SCORE:
        rg_score_unit(t, f, q)
    comptime if OP == SC_TRANSPOSE:
        rg_transpose_unit(t, f, q)
    comptime if OP == SC_BH_SORT:
        rg_bh_sort_unit(t, f, q)
    comptime if OP == SC_BH_SCAN:
        rg_bh_scan_unit(t, f, q)
    comptime if OP == SC_ADJUST:
        rg_adjust_unit(t, f, q)
