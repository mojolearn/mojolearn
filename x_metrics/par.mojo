# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PARALLEL SCHEDULES OF THE SEQUENTIAL UNITS (lane/metrics phase C,
2026-09-27). Each unit here is still a plain function of `(t, f, q)`; the
planner (x_metrics/plan.mojo) replaces one single-thread stage of the
program with a few wide stages of these units, and BOTH runners run the
planned program, so CPU == GPU stays a property of construction. Every
schedule returns THE SAME BITS as the unit it replaces, by construction:

- counting sort (`cs_*`, replaces `group_sort_unit`): a stable sort's output
  is unique. Chunk c counts its rows per group, the counts are scanned group
  major (exact integers), and chunk c places its rows in ascending order at
  its own cursor, so a group's rows stay in ascending row order (6102).
- the fold (`fold_*`, replaces `group_sum_unit`'s PairSum): PairSum's tree is
  a function of the count alone (DEVIATION 6100): leaves of LEAF added in
  order, full leaves merged pairwise level by level (the older operand on the
  left), and the result the partial leaf with the leftover node of each
  level added smallest level first. One thread per leaf, one stage per level,
  one thread per (group, column) for the end.
- the sort (`sort_*`, replaces the heapsorts of col_sort, bin_curve and
  permute): every sort of the lane orders a STRICT total order (a key, ties
  by the row index, DEVIATION 6101), so every correct sort gives the same
  permutation. Runs of RUN rows are insertion sorted, then merged by rank:
  an element's place in the merged run is its own rank plus the number of
  elements of the other run before it (a binary search). The host runs the
  same merge as two-pointer merges of MERGE_SPAN-output spans of each run
  pair, each span started by a co-rank search (the same output).
- the weighted prefixes (the curve's TP/FP sums, wpercentile's CDF) are
  BLOCKED Float32 scans (lane cgr2-metrics-shap, 2026-10-03; they were one
  sequential walk on the host, DEVIATION 6107): chunk c of CURVE_CHUNK rows
  sums its weights in ascending order, the chunk sums become exclusive
  offsets in ascending chunk order, and chunk c rewalks its rows from its
  offset. One fixed order on every runner and every vendor; the unweighted
  prefixes are integer counts, exact in any order.
- the curve folds (`cf_*`: the ROC AUC and average precision sums) are
  float-float (x_linear/ff.mojo) chunk sums met in ascending chunk order.

Slots holding keys or row indices are words (`ldu`/`ldi`); the fold's
partial sums are stored raw, as PairSum keeps them.
"""
from std.memory import bitcast
from x_metrics.common import FP, IP, p, ld, st, ldi, sti, ldu, stu, key, fadd, leaf_add, LEAF
from checks.numerics import identical_mul
from checks.fixture_rng import splitmix_pair
from x_metrics.split import fold_of_position, st_row64
from x_linear.ff import FF, two_sum, ff_add, ff_mul, ff_div, ff_of

#: The sort's initial run length (insertion sorted by one thread).
comptime RUN = 16

comptime KEY_COL = 0
comptime KEY_CURVE = 1
comptime KEY_PERM = 2
comptime KEY_GROUP = 3


# ---------------------------------------------------------------------------
# Counting sort (group_sort)
# ---------------------------------------------------------------------------

def cs_hist_unit(t: Int, f: FP, q: IP):
    """q = [K, n, m, H, C, CH]; t = chunk. H[g*C + t] = rows of chunk t in group g."""
    var K = p(q, 0)
    var n = p(q, 1)
    var m = p(q, 2)
    var H = p(q, 3)
    var C = p(q, 4)
    var CH = p(q, 5)
    for g in range(m):
        sti(f, H + g * C + t, 0)
    var lo = t * CH
    var hi = min(lo + CH, n)
    for r in range(lo, hi):
        var k = ldi(f, K + r)
        if k >= 0 and k < m:
            var a = H + k * C + t
            sti(f, a, ldi(f, a) + 1)


def cs_scan_rows_unit(t: Int, f: FP, q: IP):
    """q = [H, C, GT]; t = group: row t of H becomes its exclusive prefix,
    GT[t] its total."""
    var C = p(q, 1)
    var H = p(q, 0) + t * C
    var acc = 0
    for c in range(C):
        var v = ldi(f, H + c)
        sti(f, H + c, acc)
        acc += v
    sti(f, p(q, 2) + t, acc)


def cs_scan_groups_unit(t: Int, f: FP, q: IP):
    """q = [GT, m, OFF]; t = 0: OFF[g] = the start of group g, OFF[m] = rows kept."""
    if t != 0:
        return
    var GT = p(q, 0)
    var m = p(q, 1)
    var OFF = p(q, 2)
    var acc = 0
    sti(f, OFF, 0)
    for g in range(m):
        acc += ldi(f, GT + g)
        sti(f, OFF + g + 1, acc)


def cs_place_unit(t: Int, f: FP, q: IP):
    """q = [K, n, m, H, C, CH, OFF, ORD]; t = chunk: its rows, ascending, at
    OFF[g] + (rows of group g in the chunks before t)."""
    var K = p(q, 0)
    var n = p(q, 1)
    var m = p(q, 2)
    var H = p(q, 3)
    var C = p(q, 4)
    var CH = p(q, 5)
    var OFF = p(q, 6)
    var ORD = p(q, 7)
    var lo = t * CH
    var hi = min(lo + CH, n)
    for r in range(lo, hi):
        var k = ldi(f, K + r)
        if k >= 0 and k < m:
            var a = H + k * C + t
            var c = ldi(f, a)
            sti(f, ORD + ldi(f, OFF + k) + c, r)
            sti(f, a, c + 1)


# ---------------------------------------------------------------------------
# The fold (group_sum)
# ---------------------------------------------------------------------------

@always_inline
def _slot_group(f: FP, OFF: Int, m: Int, u: Int) -> Int:
    """The group owning slot row u: the largest g < m whose first slot row,
    OFF[g] // LEAF + g, is <= u (the first rows strictly increase with g)."""
    var lo = 0
    var hi = m - 1
    while lo < hi:
        var mid = (lo + hi + 1) // 2
        if ldi(f, OFF + mid) // LEAF + mid <= u:
            lo = mid
        else:
            hi = mid - 1
    return lo


def fold_leaf_unit(t: Int, f: FP, q: IP):
    """q = [OFF, ORD, V, VS, W, S, D, m]; t = slot (row u = t / D, column
    c = t % D). Slot row OFF[g] // LEAF + g + i holds leaf i of group g: its
    rows (ORD order) i*LEAF .. added in order, as PairSum's leaf."""
    var OFF = p(q, 0)
    var ORD = p(q, 1)
    var V = p(q, 2)
    var VS = p(q, 3)
    var W = p(q, 4)
    var S = p(q, 5)
    var D = p(q, 6)
    var m = p(q, 7)
    var u = t // D
    var c = t - u * D
    var g = _slot_group(f, OFF, m, u)
    var lo = ldi(f, OFF + g)
    var hi = ldi(f, OFF + g + 1)
    var i = u - (lo // LEAF + g)
    var start = lo + i * LEAF
    if i < 0 or start >= hi:
        return
    var end = min(start + LEAF, hi)
    var leaf = Float32(0)
    for pos in range(start, end):
        var r = ldi(f, ORD + pos)
        var v = Float32(1) if V < 0 else ld(f, V + r * VS + c)
        if W >= 0:
            v = identical_mul(v, ld(f, W + r))
        leaf = leaf_add(leaf, v)
    f.unsafe_store(S + t, leaf)


def fold_level_unit(t: Int, f: FP, q: IP):
    """q = [OFF, S, D, m, j]; t = slot. At level j the full leaves of a group
    form floor(F / 2^j) nodes (node a at leaf a * 2^j); nodes 2b and 2b+1
    merge into node b of level j+1, the older on the left. An odd last node
    is PairSum's stack entry of that level and is left in place."""
    var OFF = p(q, 0)
    var S = p(q, 1)
    var D = p(q, 2)
    var m = p(q, 3)
    var j = p(q, 4)
    var u = t // D
    var g = _slot_group(f, OFF, m, u)
    var lo = ldi(f, OFF + g)
    var hi = ldi(f, OFF + g + 1)
    var i = u - (lo // LEAF + g)
    var full = (hi - lo) // LEAF
    if i < 0 or i >= full:
        return
    var step = 1 << j
    if (i & (2 * step - 1)) != 0:
        return
    if (i >> j) + 1 >= (full >> j):
        return
    f.unsafe_store(S + t, fadd(f.unsafe_load(S + t), f.unsafe_load(S + t + step * D)))


def fold_final_unit(t: Int, f: FP, q: IP):
    """q = [OFF, S, D, OUT, V, W]; t = group * D + column: PairSum.result():
    the partial leaf (+0.0 when none), then each level's leftover node,
    smallest level first, added on the left. Counts (V < 0, W < 0) are exact."""
    var OFF = p(q, 0)
    var S = p(q, 1)
    var D = p(q, 2)
    var OUT = p(q, 3)
    var g = t // D
    var c = t - g * D
    var lo = ldi(f, OFF + g)
    var hi = ldi(f, OFF + g + 1)
    if p(q, 4) < 0 and p(q, 5) < 0:
        st(f, OUT + t, Float32(hi - lo))
        return
    var size = hi - lo
    var full = size // LEAF
    var row0 = lo // LEAF + g
    var acc = Float32(0)
    if size - full * LEAF > 0:
        acc = f.unsafe_load(S + (row0 + full) * D + c)
    var j = 0
    while (full >> j) > 0:
        var nodes = full >> j
        if (nodes & 1) == 1:
            acc = fadd(f.unsafe_load(S + (row0 + ((nodes - 1) << j)) * D + c), acc)
        j += 1
    st(f, OUT + t, acc)


# ---------------------------------------------------------------------------
# The sort (col_sort, bin_curve, permute)
# ---------------------------------------------------------------------------

@always_inline
def key_lt(h1: UInt32, l1: UInt32, x1: Int, h2: UInt32, l2: UInt32, x2: Int) -> Bool:
    """THE sort order of the lane (DEVIATION 6101): the key word pair, ties
    by the row index."""
    if h1 != h2:
        return h1 < h2
    if l1 != l2:
        return l1 < l2
    return x1 < x2


def sort_key_unit(t: Int, f: FP, q: IP):
    """q = [mode, n, B, N, a0, a1, a2]; t = element e of N = P*n (problem
    e / n, row i = e % n). The buffer at B holds KH [B, B+N), KL [B+N, B+2N),
    the row IX [B+2N, B+3N).
    KEY_COL   (a0 = V, a1 = D): column e / n of V ascending by `key`.
    KEY_CURVE (a0 = S, a1 = SS, a2 = W): problem e / n's scores DESCENDING,
              rows of zero weight after every kept row.
    KEY_PERM  (a0 = salt_lo, a1 = salt_hi): the 64-bit splitmix_pair key."""
    var mode = p(q, 0)
    var n = p(q, 1)
    var B = p(q, 2)
    var N = p(q, 3)
    var pp = t // n
    var i = t - pp * n
    var kh = UInt32(0)
    var kl = UInt32(0)
    if mode == KEY_COL:
        kl = key(ld(f, p(q, 4) + i * p(q, 5) + pp))
    elif mode == KEY_GROUP:
        var group = ldi(f, p(q, 4) + i)
        var groups = p(q, 5)
        kh = UInt32(0) if group >= 0 and group < groups else UInt32(1)
        kl = UInt32(group) if kh == UInt32(0) else UInt32(0)
    elif mode == KEY_CURVE:
        var W = p(q, 6)
        if W >= 0 and ld(f, W + i) == Float32(0):
            kh = UInt32(1)
        kl = ~key(ld(f, p(q, 4) + pp + i * p(q, 5)))
    else:
        var lo = UInt64(UInt32(p(q, 4)))
        var hi = UInt64(UInt32(p(q, 5)))
        var z = splitmix_pair(i, Int((hi << 32) | lo))
        kh = UInt32(z >> 32)
        kl = UInt32(z & 0xFFFFFFFF)
    stu(f, B + t, kh)
    stu(f, B + N + t, kl)
    sti(f, B + 2 * N + t, i)


def sort_runs_unit(t: Int, f: FP, q: IP):
    """q = [n, B, N]; t = run: rows RUN*b .. of problem pp, insertion sorted in place."""
    var n = p(q, 0)
    var B = p(q, 1)
    var N = p(q, 2)
    var nb = (n + RUN - 1) // RUN
    var pp = t // nb
    var b = t - pp * nb
    var lo = pp * n + b * RUN
    var hi = min(lo + RUN, pp * n + n)
    for i in range(lo + 1, hi):
        var h = ldu(f, B + i)
        var l = ldu(f, B + N + i)
        var x = ldi(f, B + 2 * N + i)
        var j = i - 1
        while j >= lo and key_lt(h, l, x, ldu(f, B + j), ldu(f, B + N + j), ldi(f, B + 2 * N + j)):
            stu(f, B + j + 1, ldu(f, B + j))
            stu(f, B + N + j + 1, ldu(f, B + N + j))
            sti(f, B + 2 * N + j + 1, ldi(f, B + 2 * N + j))
            j -= 1
        stu(f, B + j + 1, h)
        stu(f, B + N + j + 1, l)
        sti(f, B + 2 * N + j + 1, x)


@always_inline
def _before_at(f: FP, B: Int, N: Int, e: Int, h: UInt32, l: UInt32, x: Int) -> Bool:
    return key_lt(ldu(f, B + e), ldu(f, B + N + e), ldi(f, B + 2 * N + e), h, l, x)


def sort_merge_unit(t: Int, f: FP, q: IP):
    """q = [n, w, SRC, DST, N]; t = element: the merge of sorted runs of w
    into runs of 2w by rank (the device schedule; the host merges spans of
    each pair with `sort_merge_span_unit`, the same output)."""
    var n = p(q, 0)
    var w = p(q, 1)
    var SRC = p(q, 2)
    var DST = p(q, 3)
    var N = p(q, 4)
    var pp = t // n
    var i = t - pp * n
    var seg = pp * n
    var base = (i // (2 * w)) * (2 * w)
    var mid = min(base + w, n)
    var end = min(base + 2 * w, n)
    var h = ldu(f, SRC + t)
    var l = ldu(f, SRC + N + t)
    var x = ldi(f, SRC + 2 * N + t)
    var lo: Int
    var hi: Int
    var own: Int
    var first: Int
    if i < mid:
        lo = mid
        hi = end
        own = i - base
        first = mid
    else:
        lo = base
        hi = mid
        own = i - mid
        first = base
    while lo < hi:
        var md = (lo + hi) // 2
        if _before_at(f, SRC, N, seg + md, h, l, x):
            lo = md + 1
        else:
            hi = md
    var d = seg + base + own + (lo - first)
    stu(f, DST + d, h)
    stu(f, DST + N + d, l)
    sti(f, DST + 2 * N + d, x)


#: outputs per thread of the device merge-path schedule (a power of two
#: that divides 2 * RUN, so a chunk never crosses a run pair)
comptime MERGE_CHUNK = 8


def merge_path_chunks(total: Int, n: Int) -> Int:
    """Threads of `sort_merge_path_unit` for a merge stage of `total` = P*n."""
    return (total // n) * ((n + MERGE_CHUNK - 1) // MERGE_CHUNK)


def sort_merge_path_unit(t: Int, f: FP, q: IP):
    """q = [n, w, SRC, DST, N]; t = chunk of MERGE_CHUNK merged outputs (the
    Apple-speed device schedule of a merge pass, lane metrics-apple): one
    binary search on the merge diagonal finds how many of the chunk's
    predecessors come from each run, then a two-pointer merge writes the
    chunk. The order is strict (6101), so this writes the unique merged
    order, the same words as `sort_merge_unit` and `sort_merge_pair_unit`."""
    var n = p(q, 0)
    var w = p(q, 1)
    var SRC = p(q, 2)
    var DST = p(q, 3)
    var N = p(q, 4)
    var cpp = (n + MERGE_CHUNK - 1) // MERGE_CHUNK
    var pp = t // cpp
    var o0 = (t - pp * cpp) * MERGE_CHUNK
    var seg = pp * n
    var base = (o0 // (2 * w)) * (2 * w)
    var mid = min(base + w, n)
    var end = min(base + 2 * w, n)
    var o = o0 - base
    var lo = max(0, o - (end - mid))
    var hi = min(o, mid - base)
    while lo < hi:
        var md = (lo + hi) // 2
        var bi = seg + mid + (o - md - 1)
        if _before_at(f, SRC, N, seg + base + md, ldu(f, SRC + bi), ldu(f, SRC + N + bi), ldi(f, SRC + 2 * N + bi)):
            lo = md + 1
        else:
            hi = md
    var a = base + lo
    var b = mid + (o - lo)
    var out = o0
    var stop = min(o0 + MERGE_CHUNK, end)
    while out < stop:
        var take_a = b >= end
        if a < mid and b < end:
            take_a = _before_at(f, SRC, N, seg + a, ldu(f, SRC + seg + b), ldu(f, SRC + N + seg + b),
                                ldi(f, SRC + 2 * N + seg + b))
        if take_a:
            _move(f, SRC, DST, N, seg + a, seg + out)
            a += 1
        else:
            _move(f, SRC, DST, N, seg + b, seg + out)
            b += 1
        out += 1


@always_inline
def _move(f: FP, SRC: Int, DST: Int, N: Int, a: Int, b: Int):
    stu(f, DST + b, ldu(f, SRC + a))
    stu(f, DST + N + b, ldu(f, SRC + N + a))
    sti(f, DST + 2 * N + b, ldi(f, SRC + 2 * N + a))


#: The host merge's span: a merge pass on the host is one unit per
#: MERGE_SPAN outputs of a run pair (`sort_merge_span_unit`), so the last
#: passes (one or two run pairs of the whole column) still split across the
#: host's threads. A function of nothing but this constant.
comptime MERGE_SPAN = 4096


@always_inline
def merge_span_units(n: Int, w: Int, P: Int) -> Int:
    """The host units of one merge pass of P problems of n rows, runs of w:
    every run pair gets ceil(min(2w, n) / MERGE_SPAN) spans (the shorter last
    pair's surplus spans return at once)."""
    var npairs = (n + 2 * w - 1) // (2 * w)
    var spans = (min(2 * w, n) + MERGE_SPAN - 1) // MERGE_SPAN
    return P * npairs * spans


def sort_merge_span_unit(t: Int, f: FP, q: IP):
    """The host schedule of `sort_merge_unit` (lane/metrics phase 5): t =
    (problem, run pair, span of MERGE_SPAN outputs). The span's first output
    k0 is placed by a co-rank search (how many of the pair's first k0 outputs
    come from the left run: A[i] is among them iff it orders before
    B[k0 - i - 1]), then a two-pointer merge writes its outputs. Every sort
    of the lane orders a STRICT total order (DEVIATION 6101), so the merged
    run is unique and every span, on any thread, writes the words the
    device's per-element merge writes."""
    var n = p(q, 0)
    var w = p(q, 1)
    var SRC = p(q, 2)
    var DST = p(q, 3)
    var N = p(q, 4)
    var npairs = (n + 2 * w - 1) // (2 * w)
    var spans = (min(2 * w, n) + MERGE_SPAN - 1) // MERGE_SPAN
    var per = npairs * spans
    var pp = t // per
    var r = t - pp * per
    var pr = r // spans
    var sp = r - pr * spans
    var seg = pp * n
    var base = seg + pr * (2 * w)
    var mid = min(base + w, seg + n)
    var end = min(base + 2 * w, seg + n)
    var k0 = sp * MERGE_SPAN
    if k0 >= end - base:
        return
    var k1 = min(k0 + MERGE_SPAN, end - base)
    var lo = max(0, k0 - (end - mid))
    var hi = min(k0, mid - base)
    while lo < hi:
        var m = (lo + hi) // 2
        var ai = base + m
        var bi = mid + k0 - m - 1
        if key_lt(ldu(f, SRC + ai), ldu(f, SRC + N + ai), ldi(f, SRC + 2 * N + ai),
                  ldu(f, SRC + bi), ldu(f, SRC + N + bi), ldi(f, SRC + 2 * N + bi)):
            lo = m + 1
        else:
            hi = m
    var a = base + lo
    var b = mid + (k0 - lo)
    var o = base + k0
    var oend = base + k1
    while o < oend:
        if b < end and (a >= mid or key_lt(ldu(f, SRC + b), ldu(f, SRC + N + b), ldi(f, SRC + 2 * N + b),
                                           ldu(f, SRC + a), ldu(f, SRC + N + a), ldi(f, SRC + 2 * N + a))):
            _move(f, SRC, DST, N, b, o)
            b += 1
        else:
            _move(f, SRC, DST, N, a, o)
            a += 1
        o += 1


def sort_emit_unit(t: Int, f: FP, q: IP):
    """q = [B, N, OUT]; t = element: OUT[t] = the row at sorted position t."""
    sti(f, p(q, 2) + t, ldi(f, p(q, 0) + 2 * p(q, 1) + t))


# ---------------------------------------------------------------------------
# bin_curve after the sort
# ---------------------------------------------------------------------------

def curve_gather_unit(t: Int, f: FP, q: IP):
    """q = [n, B, N, S, SS, POS, W, G, ORD]; t = sorted element (problem
    pp = t / n, position i): in sorted order, the row's positive flag at G
    (-1 for a dropped row of zero weight, which sort after every kept row),
    its weight at G+N and its score at G+2N; ORD[pp*n + i] = the row for
    kept positions."""
    var n = p(q, 0)
    var B = p(q, 1)
    var N = p(q, 2)
    var S = p(q, 3)
    var SS = p(q, 4)
    var POS = p(q, 5)
    var W = p(q, 6)
    var G = p(q, 7)
    var pp = t // n
    var r = ldi(f, B + 2 * N + t)
    if ldu(f, B + t) == UInt32(0):
        sti(f, p(q, 8) + t, r)
        sti(f, G + t, ldi(f, POS + pp * n + r))
    else:
        sti(f, G + t, -1)
    f.unsafe_store(G + N + t, Float32(1) if W < 0 else ld(f, W + r))
    f.unsafe_store(G + 2 * N + t, ld(f, S + pp + r * SS))


def curve_emit_unit(t: Int, f: FP, q: IP):
    """q = [n, G, N, FPS, TPS, THR, CNT]; t = sorted element: the last row
    of each distinct score writes its counts and score at its slot; the
    first element of each problem writes the problem's slot count."""
    var n = p(q, 0)
    var G = p(q, 1)
    var N = p(q, 2)
    var pp = t // n
    if t == pp * n:
        sti(f, p(q, 6) + pp, ldi(f, G + 6 * N + pp))
    var k = ldi(f, G + 5 * N + t)
    if k < 0:
        return
    var o = pp * n + k
    st(f, p(q, 4) + o, ld(f, G + 3 * N + t))
    st(f, p(q, 3) + o, ld(f, G + 4 * N + t))
    st(f, p(q, 5) + o, ld(f, G + 2 * N + t))


# ---------------------------------------------------------------------------
# wpercentile's CDF
# ---------------------------------------------------------------------------

def wpct_gather_unit(t: Int, f: FP, q: IP):
    """q = [n, ORD, W, G]; t = element c*n + i: G[t] = the weight of the row
    at sorted position i of column c (1 when there are no weights)."""
    var W = p(q, 2)
    var r = ldi(f, p(q, 1) + t)
    f.unsafe_store(p(q, 3) + t, Float32(1) if W < 0 else ld(f, W + r))


def copy_unit(t: Int, f: FP, q: IP):
    """q = [SRC, DST]; t = element: a word copy."""
    f.unsafe_store(p(q, 1) + t, f.unsafe_load(p(q, 0) + t))


# ---------------------------------------------------------------------------
# The column maximum (col_max, lane metrics-apple)
# ---------------------------------------------------------------------------
# `col_max_unit` keeps m = the column's first value and replaces it by each
# later value v with v > m: a first NaN is the answer; otherwise NaNs never
# win and the answer is the EARLIEST of the values equal to the largest
# (-0.0 and +0.0 compare equal, so the earlier one's bits stay). A chunk
# keeps its earliest largest non-NaN value; the chunks meet left to right
# by the same `>`, so the earliest one wins again: the same word.

def cm_chunk_unit(t: Int, f: FP, q: IP):
    """q = [V, n, D, S, C, CH]; t = column * C + chunk: S[2t] = the chunk's
    earliest largest non-NaN value, S[2t+1] = 1 when it has one."""
    var V = p(q, 0)
    var n = p(q, 1)
    var D = p(q, 2)
    var S = p(q, 3)
    var C = p(q, 4)
    var CH = p(q, 5)
    var col = t // C
    var c = t - col * C
    var m = Float32(0)
    var found = 0
    for r in range(c * CH, min(n, c * CH + CH)):
        var v = ld(f, V + r * D + col)
        if v == v:  # not NaN
            if found == 0:
                m = v
                found = 1
            elif v > m:
                m = v
    st(f, S + 2 * t, m)
    sti(f, S + 2 * t + 1, found)


def cm_final_unit(t: Int, f: FP, q: IP):
    """q = [V, D, S, C, OUT]; t = column: col_max_unit's word."""
    var V = p(q, 0)
    var D = p(q, 1)
    var S = p(q, 2)
    var C = p(q, 3)
    var v0 = ld(f, V + t)
    var m = v0
    if v0 == v0:  # not NaN
        var found = 0
        for c in range(C):
            var k = S + 2 * (t * C + c)
            if ldi(f, k + 1) != 0:
                var s = ld(f, k)
                if found == 0:
                    m = s
                    found = 1
                elif s > m:
                    m = s
    st(f, p(q, 4) + t, m)


# ---------------------------------------------------------------------------
# The prefixes in parallel (lane metrics-apple; weighted: cgr2-metrics-shap)
# ---------------------------------------------------------------------------
# Without weights the two prefixes are INTEGER counts: the percentile CDF
# adds 1.0 per row (exact while it stays below 2^24, so its i-th word is
# Float32(i + 1)), and the curve walk counts positives, negatives and
# distinct-score ends in Int, exact in any order. With weights they are
# BLOCKED Float32 scans of one fixed order: chunk sums in ascending row
# order, exclusive chunk offsets in ascending chunk order, each chunk's rows
# rewalked from its offset (`fadd` throughout). Host and device run these
# same units, so the words agree on every runner.

def wpct_iota_unit(t: Int, f: FP, q: IP):
    """q = [n, P]; t = element c*n + i: P[t] = Float32(i + 1), the
    unweighted CDF word (n <= 2^24, the planner's guard)."""
    var n = p(q, 0)
    st(f, p(q, 1) + t, Float32(t - (t // n) * n + 1))


def wpct_csum_unit(t: Int, f: FP, q: IP):
    """q = [n, G, S, C, CH]; t = column * C + chunk: S[t] = the chunk's
    gathered weights G summed in ascending order."""
    var n = p(q, 0)
    var C = p(q, 3)
    var CH = p(q, 4)
    var col = t // C
    var c = t - col * C
    var G = p(q, 1) + col * n
    var acc = Float32(0)
    for i in range(c * CH, min(n, c * CH + CH)):
        acc = fadd(acc, ld(f, G + i))
    st(f, p(q, 2) + t, acc)


def wpct_coff_unit(t: Int, f: FP, q: IP):
    """q = [S, C]; t = column: the column's chunk sums become exclusive
    offsets in place, in ascending chunk order."""
    var S = p(q, 0) + t * p(q, 1)
    var acc = Float32(0)
    for c in range(p(q, 1)):
        var v = ld(f, S + c)
        st(f, S + c, acc)
        acc = fadd(acc, v)


def wpct_cfill_unit(t: Int, f: FP, q: IP):
    """q = [n, G, P, S, C, CH]; t = column * C + chunk: P = the blocked
    CDF of the gathered weights G, the chunk's rows from its offset."""
    var n = p(q, 0)
    var C = p(q, 4)
    var CH = p(q, 5)
    var col = t // C
    var c = t - col * C
    var G = p(q, 1) + col * n
    var P = p(q, 2) + col * n
    var acc = ld(f, p(q, 3) + t)
    for i in range(c * CH, min(n, c * CH + CH)):
        acc = fadd(acc, ld(f, G + i))
        st(f, P + i, acc)


@always_inline
def _curve_last(f: FP, G: Int, N: Int, n: Int, i: Int, e: Int) -> Bool:
    """The end-of-distinct-score test for kept row e = pp*n + i."""
    if i == n - 1:
        return True
    if ldi(f, G + e + 1) < 0:
        return True
    return ld(f, G + 2 * N + e + 1) != ld(f, G + 2 * N + e)


def curve_cnt_unit(t: Int, f: FP, q: IP):
    """q = [n, G, N, S, C, CH, W]; t = problem * C + chunk: the chunk's kept
    positives, kept negatives (W < 0: Int counts; W >= 0: Float32 weight
    sums in ascending order, stored as floats) and distinct-score ends at
    S[3t .. 3t+2]."""
    var n = p(q, 0)
    var G = p(q, 1)
    var N = p(q, 2)
    var S = p(q, 3)
    var C = p(q, 4)
    var CH = p(q, 5)
    var weighted = p(q, 6) >= 0
    var pp = t // C
    var c = t - pp * C
    var tp = 0
    var fp = 0
    var tw = Float32(0)
    var fw = Float32(0)
    var ends = 0
    for i in range(c * CH, min(n, c * CH + CH)):
        var e = pp * n + i
        var pos = ldi(f, G + e)
        if pos < 0:
            continue
        if weighted:
            if pos == 1:
                tw = fadd(tw, ld(f, G + N + e))
            else:
                fw = fadd(fw, ld(f, G + N + e))
        elif pos == 1:
            tp += 1
        else:
            fp += 1
        if _curve_last(f, G, N, n, i, e):
            ends += 1
    if weighted:
        st(f, S + 3 * t, tw)
        st(f, S + 3 * t + 1, fw)
    else:
        sti(f, S + 3 * t, tp)
        sti(f, S + 3 * t + 1, fp)
    sti(f, S + 3 * t + 2, ends)


def curve_off_unit(t: Int, f: FP, q: IP):
    """q = [G, N, S, C, W]; t = problem: the chunks' counts (or weight
    sums) become exclusive offsets in place, in ascending chunk order, and
    G+6N+t = the problem's slot count."""
    var G = p(q, 0)
    var N = p(q, 1)
    var S = p(q, 2)
    var C = p(q, 3)
    var weighted = p(q, 4) >= 0
    var tp = 0
    var fp = 0
    var tw = Float32(0)
    var fw = Float32(0)
    var ends = 0
    for c in range(C):
        var k = S + 3 * (t * C + c)
        if weighted:
            var a = ld(f, k)
            var b = ld(f, k + 1)
            st(f, k, tw)
            st(f, k + 1, fw)
            tw = fadd(tw, a)
            fw = fadd(fw, b)
        else:
            var a = ldi(f, k)
            var b = ldi(f, k + 1)
            sti(f, k, tp)
            sti(f, k + 1, fp)
            tp += a
            fp += b
        var d = ldi(f, k + 2)
        sti(f, k + 2, ends)
        ends += d
    sti(f, G + 6 * N + t, ends)


def curve_fill_unit(t: Int, f: FP, q: IP):
    """q = [n, G, N, S, C, CH, W]; t = problem * C + chunk: the TP, FP
    (G+3N, G+4N) and IDX (G+5N) words of the chunk's rows, started from the
    chunk's offsets: Int counts unweighted, the blocked Float32 weight scan
    weighted. A dropped row's TP and FP words are 0 and its IDX -1."""
    var n = p(q, 0)
    var G = p(q, 1)
    var N = p(q, 2)
    var S = p(q, 3)
    var C = p(q, 4)
    var CH = p(q, 5)
    var weighted = p(q, 6) >= 0
    var pp = t // C
    var c = t - pp * C
    var tp = 0
    var fp = 0
    var tw = Float32(0)
    var fw = Float32(0)
    if weighted:
        tw = ld(f, S + 3 * t)
        fw = ld(f, S + 3 * t + 1)
    else:
        tp = ldi(f, S + 3 * t)
        fp = ldi(f, S + 3 * t + 1)
    var cnt = ldi(f, S + 3 * t + 2)
    for i in range(c * CH, min(n, c * CH + CH)):
        var e = pp * n + i
        var pos = ldi(f, G + e)
        if pos < 0:
            st(f, G + 3 * N + e, Float32(0))
            st(f, G + 4 * N + e, Float32(0))
            sti(f, G + 5 * N + e, -1)
            continue
        if weighted:
            if pos == 1:
                tw = fadd(tw, ld(f, G + N + e))
            else:
                fw = fadd(fw, ld(f, G + N + e))
            st(f, G + 3 * N + e, tw)
            st(f, G + 4 * N + e, fw)
        else:
            if pos == 1:
                tp += 1
            else:
                fp += 1
            st(f, G + 3 * N + e, Float32(tp))
            st(f, G + 4 * N + e, Float32(fp))
        if _curve_last(f, G, N, n, i, e):
            sti(f, G + 5 * N + e, cnt)
            cnt += 1
        else:
            sti(f, G + 5 * N + e, -1)


@always_inline
def _step(a: Float32, b: Float32) -> FF:
    """a - b exactly: Knuth's two-sum of two Float32 words (the error term
    is exact; its flush only reaches terms below 2^-126)."""
    return two_sum(a, -b)


@always_inline
def _same_step(x: FF, y: FF) -> Bool:
    """Two exact two-sum differences are the same value iff both words
    agree (the rounded sum and its exact remainder are unique)."""
    return x.hi == y.hi and x.lo == y.lo


def curve_keep_unit(t: Int, f: FP, q: IP):
    """q = [n, FPS, TPS, CNT, KEEP]; t = element pp*n + i (lane
    metrics-apple; weighted curves too, cgr2-metrics-shap): KEEP[t] = 1
    when `_drop_collinear` keeps slot i of the problem's CNT[pp] slots (the
    first, the last, and every slot where the fps or tps step changes),
    else 0; slots past the count are not written. The steps are the EXACT
    differences of the Float32 words (two-sum), compared exactly: integer
    counts decide as Int differences do, weight sums by their exact steps."""
    var n = p(q, 0)
    var pp = t // n
    var i = t - pp * n
    var c = ldi(f, p(q, 3) + pp)
    if i >= c:
        return
    var keep = 1
    if i > 0 and i < c - 1:
        var F = p(q, 1) + pp * n + i
        var T = p(q, 2) + pp * n + i
        var f0 = ld(f, F - 1)
        var f1 = ld(f, F)
        var f2 = ld(f, F + 1)
        var t0 = ld(f, T - 1)
        var t1 = ld(f, T)
        var t2 = ld(f, T + 1)
        var same = _same_step(_step(f2, f1), _step(f1, f0)) and _same_step(_step(t2, t1), _step(t1, t0))
        keep = 0 if same else 1
    sti(f, p(q, 4) + t, keep)


# ---------------------------------------------------------------------------
# The curve folds (curve_fold; lane cgr2-metrics-shap, 2026-10-03)
# ---------------------------------------------------------------------------
# One float-float sum per problem over its curve points, chunk sums of
# CF_CHUNK points in ascending order met in ascending chunk order (the same
# words on every runner). The host turns the few words into the score.
#   mode 0, the ROC AUC: sum of (fps[i] - fps[i-1]) * (tps[i] + tps[i-1])
#     over the (collinear-dropped) points, point -1 = (0, 0); the score is
#     the sum / (2 F T), F = fps[c-1], T = tps[c-1].
#   mode 1, the average precision: sum of (tps[i] - tps[i-1]) * tps[i] /
#     (tps[i] + fps[i]); the score is max(0, sum / T); the chunk's count
#     word holds its zero denominators (any sends the caller to Python).
#   mode 2, the partial AUC (max_fpr = MH + ML, float-float): mode 0's sum
#     over the points with fps[i] / F <= max_fpr (float-float quotient,
#     compared hi then lo), a prefix since fps ascends; the count word holds
#     how many.
comptime CF_OUT = 10


@always_inline
def _cf_term(f: FP, FPS: Int, TPS: Int, i: Int, mode: Int) -> FF:
    var f1 = ld(f, FPS + i)
    var t1 = ld(f, TPS + i)
    var f0 = Float32(0)
    var t0 = Float32(0)
    if i > 0:
        f0 = ld(f, FPS + i - 1)
        t0 = ld(f, TPS + i - 1)
    if mode == 1:
        var den = two_sum(t1, f1)
        if den.hi == Float32(0):
            return ff_of(Float32(0))
        return ff_mul(_step(t1, t0), ff_div(ff_of(t1), den))
    return ff_mul(_step(f1, f0), two_sum(t1, t0))


@always_inline
def _cf_in(f: FP, FPS: Int, i: Int, Fl: Float32, mh: Float32, ml: Float32) -> Bool:
    """mode 2: fps[i] / F <= max_fpr, the float-float quotient against the
    float-float bound."""
    var r = ff_div(ff_of(ld(f, FPS + i)), ff_of(Fl))
    return r.hi < mh or (r.hi == mh and r.lo <= ml)


@always_inline
def _cf_chunk(f: FP, FPS: Int, TPS: Int, c0: Int, c1: Int, mode: Int, Fl: Float32,
              mh: Float32, ml: Float32, mut k: Int) -> FF:
    """The chunk [c0, c1)'s sum, its count word added to k."""
    var acc = ff_of(Float32(0))
    for i in range(c0, c1):
        if mode == 2:
            if not _cf_in(f, FPS, i, Fl, mh, ml):
                break
            k += 1
        elif mode == 1:
            if two_sum(ld(f, TPS + i), ld(f, FPS + i)).hi == Float32(0):
                k += 1
        acc = ff_add(acc, _cf_term(f, FPS, TPS, i, mode))
    return acc


@always_inline
def _cf_bound(f: FP, q: IP, pp: Int, n: Int) -> Float32:
    var c = ldi(f, p(q, 3) + pp)
    if c <= 0:
        return Float32(0)
    return ld(f, p(q, 1) + pp * n + c - 1)


def _cf_finish(f: FP, n: Int, FPS: Int, TPS: Int, c: Int, OUT: Int, acc: FF, k: Int, mode: Int):
    """OUT[0..9] = sum hi, lo, F, T, the count word, then (mode 2) fps and
    tps at points k-1 and k (0 outside the curve)."""
    st(f, OUT, acc.hi)
    st(f, OUT + 1, acc.lo)
    st(f, OUT + 2, ld(f, FPS + c - 1) if c > 0 else Float32(0))
    st(f, OUT + 3, ld(f, TPS + c - 1) if c > 0 else Float32(0))
    sti(f, OUT + 4, k)
    var a = Float32(0)
    var b = Float32(0)
    var u = Float32(0)
    var v = Float32(0)
    if mode == 2:
        if k > 0 and k - 1 < c:
            a = ld(f, FPS + k - 1)
            u = ld(f, TPS + k - 1)
        if k < c:
            b = ld(f, FPS + k)
            v = ld(f, TPS + k)
    st(f, OUT + 5, a)
    st(f, OUT + 6, b)
    st(f, OUT + 7, u)
    st(f, OUT + 8, v)
    sti(f, OUT + 9, c)


def curve_fold_unit(t: Int, f: FP, q: IP):
    """The caller's op (unplanned: one problem per unit, the same chunks
    and the same meeting order as cf_chunk_unit + cf_final_unit, so the
    same words). q = [n, FPS, TPS, CNT, OUT, mode, MH, ML, CH]: FPS, TPS
    strided by n per problem, CNT[pp] points, OUT + CF_OUT*pp the words."""
    var n = p(q, 0)
    var mode = p(q, 5)
    var CH = p(q, 8)
    var FPS = p(q, 1) + t * n
    var TPS = p(q, 2) + t * n
    var c = ldi(f, p(q, 3) + t)
    var Fl = _cf_bound(f, q, t, n)
    var mh = bitcast[DType.float32](Int32(p(q, 6)))
    var ml = bitcast[DType.float32](Int32(p(q, 7)))
    var acc = ff_of(Float32(0))
    var k = 0
    var c0 = 0
    while c0 < c:
        var c1 = min(c, c0 + CH)
        var kc = 0
        var part = _cf_chunk(f, FPS, TPS, c0, c1, mode, Fl, mh, ml, kc)
        acc = ff_add(acc, part)
        k += kc
        c0 = c1
    _cf_finish(f, n, FPS, TPS, c, p(q, 4) + CF_OUT * t, acc, k, mode)


def cf_chunk_unit(t: Int, f: FP, q: IP):
    """q = [n, FPS, TPS, CNT, S, C, CH, mode, MH, ML]; t = problem * C +
    chunk: S[3t .. 3t+2] = the chunk's float-float sum and count word."""
    var n = p(q, 0)
    var C = p(q, 5)
    var CH = p(q, 6)
    var mode = p(q, 7)
    var pp = t // C
    var ch = t - pp * C
    var c = ldi(f, p(q, 3) + pp)
    var S = p(q, 4) + 3 * t
    var Fl = _cf_bound(f, q, pp, n)
    var mh = bitcast[DType.float32](Int32(p(q, 8)))
    var ml = bitcast[DType.float32](Int32(p(q, 9)))
    var k = 0
    var acc = ff_of(Float32(0))
    var c0 = ch * CH
    if c0 < c:
        acc = _cf_chunk(f, p(q, 1) + pp * n, p(q, 2) + pp * n, c0, min(c, c0 + CH), mode, Fl, mh, ml, k)
    f.unsafe_store(S, acc.hi)
    f.unsafe_store(S + 1, acc.lo)
    sti(f, S + 2, k)


def cf_final_unit(t: Int, f: FP, q: IP):
    """q = [n, FPS, TPS, CNT, S, C, OUT, mode]; t = problem: the chunk sums
    met in ascending chunk order (empty chunks add a zero, as the unplanned
    unit never sees them: ff_add of +0 keeps every word), then the words."""
    var n = p(q, 0)
    var C = p(q, 5)
    var mode = p(q, 7)
    var c = ldi(f, p(q, 3) + t)
    var acc = ff_of(Float32(0))
    var k = 0
    for ch in range(C):
        var S = p(q, 4) + 3 * (t * C + ch)
        acc = ff_add(acc, FF(f.unsafe_load(S), f.unsafe_load(S + 1)))
        k += ldi(f, S + 2)
    _cf_finish(f, n, p(q, 1) + t * n, p(q, 2) + t * n, c, p(q, 6) + CF_OUT * t, acc, k, mode)


# ---------------------------------------------------------------------------
# K-fold rows (fold_rows, lane metrics-apple2)
# ---------------------------------------------------------------------------

def fr_scatter_unit(t: Int, f: FP, q: IP):
    """q = [n, K, CODE, ORD]; t = position: CODE[ORD[t]] = t's contiguous
    fold. ORD is a permutation, so no two units write one word."""
    var n = p(q, 0)
    sti(f, p(q, 2) + ldi(f, p(q, 3) + t), fold_of_position(t, n, p(q, 1)))


def fr_cnt_unit(t: Int, f: FP, q: IP):
    """q = [n, K, CODE, S, C, CH]; t = fold * C + chunk: the chunk's rows
    of that fold at S[t] (exact integers)."""
    var n = p(q, 0)
    var CODE = p(q, 2)
    var C = p(q, 4)
    var CH = p(q, 5)
    var fo = t // C
    var c = t - fo * C
    var k = 0
    for i in range(c * CH, min(n, c * CH + CH)):
        if ldi(f, CODE + i) == fo:
            k += 1
    sti(f, p(q, 3) + t, k)


def fr_off_unit(t: Int, f: FP, q: IP):
    """q = [S, C, SZ]; t = fold: the chunk counts become exclusive offsets
    in place, and SZ[t] = the fold's test count."""
    var S = p(q, 0)
    var C = p(q, 1)
    var acc = 0
    for c in range(C):
        var k = S + t * C + c
        var v = ldi(f, k)
        sti(f, k, acc)
        acc += v
    sti(f, p(q, 2) + t, acc)


def fr_fill_unit(t: Int, f: FP, q: IP):
    """q = [n, K, CODE, S, C, CH, OUT, SZ]; t = fold * C + chunk: the
    chunk's rows go, in ascending order, after the fold's test rows of the
    earlier chunks (a test row) or after its test rows and the earlier
    chunks' train rows (a train row): `fold_rows_unit`'s words."""
    var n = p(q, 0)
    var CODE = p(q, 2)
    var C = p(q, 4)
    var CH = p(q, 5)
    var fo = t // C
    var c = t - fo * C
    var base = p(q, 6) + 2 * n * fo
    var a = ldi(f, p(q, 3) + t)
    var b = ldi(f, p(q, 7) + fo) + c * CH - a
    for i in range(c * CH, min(n, c * CH + CH)):
        if ldi(f, CODE + i) == fo:
            st_row64(f, base + 2 * a, i)
            a += 1
        else:
            st_row64(f, base + 2 * b, i)
            b += 1


# ---------------------------------------------------------------------------
# Curve compaction (bin_curve params 12, 13; lane metrics-apple2)
# ---------------------------------------------------------------------------

def ck_cnt_unit(t: Int, f: FP, q: IP):
    """q = [n, KEEP, CNT, S, C, CH]; t = problem * C + chunk: the chunk's
    kept slots (KEEP == 1 among the problem's CNT slots) at S[t]."""
    var n = p(q, 0)
    var C = p(q, 4)
    var CH = p(q, 5)
    var pp = t // C
    var c = t - pp * C
    var cnt = ldi(f, p(q, 2) + pp)
    var k = 0
    for i in range(c * CH, min(cnt, c * CH + CH)):
        if ldi(f, p(q, 1) + pp * n + i) != 0:
            k += 1
    sti(f, p(q, 3) + t, k)


def ck_off_unit(t: Int, f: FP, q: IP):
    """q = [S, C, CM]; t = problem: exclusive offsets in place, CM[t] = the
    problem's kept count."""
    var S = p(q, 0)
    var C = p(q, 1)
    var acc = 0
    for c in range(C):
        var k = S + t * C + c
        var v = ldi(f, k)
        sti(f, k, acc)
        acc += v
    sti(f, p(q, 2) + t, acc)


def ck_fill_unit(t: Int, f: FP, q: IP):
    """q = [n, KEEP, CNT, S, C, CH, FPS, TPS, THR, CF, N]; t = problem * C
    + chunk: each kept slot's fps, tps and threshold words, copied bit for
    bit, to its rank among the problem's kept slots."""
    var n = p(q, 0)
    var C = p(q, 4)
    var CH = p(q, 5)
    var N = p(q, 10)
    var pp = t // C
    var c = t - pp * C
    var cnt = ldi(f, p(q, 2) + pp)
    var j = ldi(f, p(q, 3) + t)
    var CF = p(q, 9) + pp * n
    for i in range(c * CH, min(cnt, c * CH + CH)):
        if ldi(f, p(q, 1) + pp * n + i) != 0:
            var e = pp * n + i
            f.unsafe_store(CF + j, f.unsafe_load(p(q, 6) + e))
            f.unsafe_store(CF + N + j, f.unsafe_load(p(q, 7) + e))
            f.unsafe_store(CF + 2 * N + j, f.unsafe_load(p(q, 8) + e))
            j += 1
