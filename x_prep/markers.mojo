# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Marker-gene tests (`rank_genes_groups`) as x_prep units (lane sc-stats-1, 2026-11-28).

References: Scanpy 1.10 `scanpy/tools/_rank_genes_groups.py` (`_RankGenes.wilcoxon`, `t_test`,
`compute_statistics`, the log fold change `log2((expm1(mean_g) + 1e-9) / (expm1(mean_rest) + 1e-9))`),
Seurat `FindMarkers` (the same two tests), Welch's unequal-variance t-test with the Welch-Satterthwaite degrees
of freedom, Benjamini and Hochberg 1995 (statsmodels' `fdr_bh` arithmetic).

The units run in the prep lane's program model (x_prep/common.mojo): one source, the device launches one thread
per unit, the host column runs the same function over its pool, so CPU == GPU bit for bit by construction. Each
unit's loops run in ascending order and that order is the fold order on every column.

A chunk program (bindings/prep_markers_run.mojo) covers the genes [c0, c0 + dc):

  the block     the group starts START (K + 1 int words), the row order PERM and the chunk's values in group
                order, gene-major, G[c * n + pos] = canon(X[PERM[pos], c0 + c]): the count-data lane's block
                (bindings/prep_scprep_run.mojo `gene_block_stage`, x_prep/sc_prep.mojo SP_ORDER, SP_GATHER_*
                with SP_PRE_CANON, so equal values have equal sort keys).
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

then one program over the whole n_out x n_genes result: the transpose (x_prep/py2mojo.mojo, gene-major chunks -> row-major), the
Benjamini-Hochberg adjustment per group (rg_bh_sort, rg_bh_scan, rg_adjust), and Scanpy's sorted layout: each
group's genes by score descending, a tie by the lower gene index (rg_rank_keys, rg_rank_sort, rg_rank_gather),
with the gene indices beside the four sorted value rows.

The rank sums and the tie sums are 64-bit integers (two int words each in the arena), so they carry no rounding;
the statistic is formed from them in float32 through the portable operations (x_prep/prims.mojo), the p-value
tails are the univariate scores' `gammaincc` and `betainc` (x_prep/stats.mojo): Q(1/2, z^2 / 2) is the normal
two-sided p-value erfc(|z| / sqrt 2), and I_{dof / (dof + t^2)}(dof / 2, 1/2) the Student two-sided one.
"""
from std.memory import bitcast
from x_prep.common import FP, IP, p, ld, raw, st, ldi, sti, key, heap_sort
from x_prep.sc_prep import seg_sum, seg_ss, expm1f
from x_prep.prims import add, sub, mul, div, logf, expf, sqrtf
from x_prep.stats import betainc, gammaincc

#: The op range of these units (after F2_BASE + F2_N = 272, x_prep/fam2.mojo).
comptime SC_BASE = 280
comptime SC_N = 10

comptime SC_SEG_SORT = SC_BASE + 0
comptime SC_STATS = SC_BASE + 1
comptime SC_RANKSUM = SC_BASE + 2
comptime SC_SCORE = SC_BASE + 3
comptime SC_BH_SORT = SC_BASE + 4
comptime SC_BH_SCAN = SC_BASE + 5
comptime SC_ADJUST = SC_BASE + 6
comptime SC_RANK_KEYS = SC_BASE + 7
comptime SC_RANK_SORT = SC_BASE + 8
comptime SC_RANK_GATHER = SC_BASE + 9

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
        mean = div(seg_sum(f, col, lo, hi, Float32(0)), Float32(m))
    if m > 1:
        var_ = div(seg_ss(f, col, lo, hi, mean, Float32(0)), Float32(m - 1))
    var rmean = Float32(0)
    var rvar = Float32(0)
    var rm: Int
    if ref_group >= 0:
        var rlo = ldi(f, start + ref_group)
        var rhi = ldi(f, start + ref_group + 1)
        rm = rhi - rlo
        if rm > 0:
            rmean = div(seg_sum(f, col, rlo, rhi, Float32(0)), Float32(rm))
        if rm > 1:
            rvar = div(seg_ss(f, col, rlo, rhi, rmean, Float32(0)), Float32(rm - 1))
    else:
        rm = n - m
        if rm > 0:
            var s = seg_sum(f, col, 0, lo, Float32(0))
            s = seg_sum(f, col, hi, n, s)
            rmean = div(s, Float32(rm))
        if rm > 1:
            var ss = seg_ss(f, col, 0, lo, rmean, Float32(0))
            ss = seg_ss(f, col, hi, n, rmean, ss)
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
    """Scanpy's log2((expm1(mean_g) + 1e-9) / (expm1(mean_rest) + 1e-9)); expm1 is the count-data lane's `expm1f`."""
    var a = add(expm1f(mg), _FC_EPS)
    var b = add(expm1f(mr), _FC_EPS)
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


# ---------------------------------------------------------------- the per-group order (Scanpy's sorted result)


@always_inline
def _k64(f: FP, i: Int) -> UInt64:
    var w = f.bitcast[UInt32]()
    return (UInt64(w.unsafe_load(i)) << 32) | UInt64(w.unsafe_load(i + 1))


@always_inline
def _swap64(f: FP, i: Int, j: Int):
    var w = f.bitcast[UInt32]()
    var a0 = w.unsafe_load(i)
    var a1 = w.unsafe_load(i + 1)
    w.unsafe_store(i, w.unsafe_load(j))
    w.unsafe_store(i + 1, w.unsafe_load(j + 1))
    w.unsafe_store(j, a0)
    w.unsafe_store(j + 1, a1)


@always_inline
def _sift64(f: FP, base: Int, start: Int, n: Int):
    var root = start
    while True:
        var child = 2 * root + 1
        if child >= n:
            return
        if child + 1 < n and _k64(f, base + 2 * child) < _k64(f, base + 2 * child + 2):
            child += 1
        if _k64(f, base + 2 * root) < _k64(f, base + 2 * child):
            _swap64(f, base + 2 * root, base + 2 * child)
            root = child
        else:
            return


def rg_rank_keys_unit(t: Int, f: FP, q: IP):
    """q = [SC, KEYS, d]; t = j*d + c. KEYS[2t], KEYS[2t + 1] = (~key(score), c): ascending keys are the scores
    descending, a tie by the lower gene index (a stable descending sort, as Scanpy's)."""
    var w = f.bitcast[UInt32]()
    var d = p(q, 2)
    var c = t % d
    w.unsafe_store(p(q, 1) + 2 * t, ~key(ld(f, p(q, 0) + t)))
    w.unsafe_store(p(q, 1) + 2 * t + 1, UInt32(c))


def rg_rank_sort_unit(t: Int, f: FP, q: IP):
    """q = [KEYS, d]; t = row j. Row j's d (key, gene) pairs heap-sorted ascending as 64-bit words (the lane's
    heap sort on pairs; one thread per output row, as rg_bh_sort)."""
    var d = p(q, 1)
    var base = p(q, 0) + 2 * t * d
    if d < 2:
        return
    var start = d // 2 - 1
    while start >= 0:
        _sift64(f, base, start, d)
        start -= 1
    var end = d - 1
    while end > 0:
        _swap64(f, base, base + 2 * end)
        _sift64(f, base, 0, end)
        end -= 1


def rg_rank_gather_unit(t: Int, f: FP, q: IP):
    """q = [KEYS, d, SC, PV, ADJ, LFC, NAMES, OUT, TOTAL]; t = j*d + r (TOTAL = rows * d). Rank r of row j: NAMES[t] = its gene (an int
    word) and OUT's four row-major blocks (scores, p-values, adjusted p-values, log fold changes) its values, bit
    for bit."""
    var d = p(q, 1)
    var j = t // d
    var c = Int(f.bitcast[UInt32]().unsafe_load(p(q, 0) + 2 * t + 1))
    var src = j * d + c
    var total = p(q, 8)
    sti(f, p(q, 6) + t, c)
    var o = p(q, 7)
    f.unsafe_store(o + t, raw(f, p(q, 2) + src))
    f.unsafe_store(o + total + t, raw(f, p(q, 3) + src))
    f.unsafe_store(o + 2 * total + t, raw(f, p(q, 4) + src))
    f.unsafe_store(o + 3 * total + t, raw(f, p(q, 5) + src))


@always_inline
def run_sc_unit[OP: Int](t: Int, f: FP, q: IP):
    comptime if OP == SC_SEG_SORT:
        rg_seg_sort_unit(t, f, q)
    comptime if OP == SC_STATS:
        rg_stats_unit(t, f, q)
    comptime if OP == SC_RANKSUM:
        rg_ranksum_unit(t, f, q)
    comptime if OP == SC_SCORE:
        rg_score_unit(t, f, q)
    comptime if OP == SC_BH_SORT:
        rg_bh_sort_unit(t, f, q)
    comptime if OP == SC_BH_SCAN:
        rg_bh_scan_unit(t, f, q)
    comptime if OP == SC_ADJUST:
        rg_adjust_unit(t, f, q)
    comptime if OP == SC_RANK_KEYS:
        rg_rank_keys_unit(t, f, q)
    comptime if OP == SC_RANK_SORT:
        rg_rank_sort_unit(t, f, q)
    comptime if OP == SC_RANK_GATHER:
        rg_rank_gather_unit(t, f, q)
