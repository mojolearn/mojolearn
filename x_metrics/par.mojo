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
  same merge as a two-pointer merge of each run pair (the same output).
- the prefixes (`curve_scan`, `wpct_prefix`) stay SEQUENTIAL Float32 prefixes
  (DEVIATION 6107); the planner first gathers their operands into sorted
  order in parallel, so the one thread reads memory in order.

Slots holding keys or row indices are words (`ldu`/`ldi`); the fold's
partial sums are stored raw, as PairSum keeps them.
"""
from std.memory import bitcast
from x_metrics.common import FP, IP, p, ld, st, ldi, sti, ldu, stu, key, fadd, leaf_add, LEAF
from checks.numerics import identical_mul
from checks.fixture_rng import splitmix_pair

#: The sort's initial run length (insertion sorted by one thread).
comptime RUN = 16

comptime KEY_COL = 0
comptime KEY_CURVE = 1
comptime KEY_PERM = 2


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
    into runs of 2w by rank (the device schedule; the host merges each pair
    with `sort_merge_pair_unit`, the same output)."""
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


@always_inline
def _move(f: FP, SRC: Int, DST: Int, N: Int, a: Int, b: Int):
    stu(f, DST + b, ldu(f, SRC + a))
    stu(f, DST + N + b, ldu(f, SRC + N + a))
    sti(f, DST + 2 * N + b, ldi(f, SRC + 2 * N + a))


def sort_merge_pair_unit(t: Int, f: FP, q: IP):
    """The host schedule of `sort_merge_unit`: t = run pair, a two-pointer merge."""
    var n = p(q, 0)
    var w = p(q, 1)
    var SRC = p(q, 2)
    var DST = p(q, 3)
    var N = p(q, 4)
    var npairs = (n + 2 * w - 1) // (2 * w)
    var pp = t // npairs
    var base = pp * n + (t - pp * npairs) * (2 * w)
    var mid = min(base + w, pp * n + n)
    var end = min(base + 2 * w, pp * n + n)
    var a = base
    var b = mid
    var o = base
    while a < mid and b < end:
        if key_lt(ldu(f, SRC + b), ldu(f, SRC + N + b), ldi(f, SRC + 2 * N + b),
                  ldu(f, SRC + a), ldu(f, SRC + N + a), ldi(f, SRC + 2 * N + a)):
            _move(f, SRC, DST, N, b, o)
            b += 1
        else:
            _move(f, SRC, DST, N, a, o)
            a += 1
        o += 1
    while a < mid:
        _move(f, SRC, DST, N, a, o)
        a += 1
        o += 1
    while b < end:
        _move(f, SRC, DST, N, b, o)
        b += 1
        o += 1


def sort_emit_unit(t: Int, f: FP, q: IP):
    """q = [B, N, OUT]; t = element: OUT[t] = the row at sorted position t."""
    sti(f, p(q, 2) + t, ldi(f, p(q, 0) + 2 * p(q, 1) + t))


# ---------------------------------------------------------------------------
# bin_curve after the sort
# ---------------------------------------------------------------------------

def curve_gather_unit(t: Int, f: FP, q: IP):
    """q = [n, B, N, S, SS, POS, W, G, ORD]; t = sorted element (problem
    pp = t / n, position i): the row's flag, weight and score in sorted
    order (G, G+N, G+2N), and ORD[pp*n + i] = the row for kept positions."""
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
    f.unsafe_store(G + N + t, Float32(1) if W < 0 else ld(f, W + r))
    f.unsafe_store(G + 2 * N + t, ld(f, S + pp + r * SS))


def curve_scan_unit(t: Int, f: FP, q: IP):
    """q = [n, B, N, G, W, FPS, TPS, THR, CNT]; t = problem: bin_curve_unit's
    cumulative counts over the gathered sorted rows (exact integers
    unweighted; a SEQUENTIAL ascending Float32 prefix weighted, DEVIATION
    6107), one output per distinct score."""
    var n = p(q, 0)
    var B = p(q, 1)
    var N = p(q, 2)
    var G = p(q, 3)
    var W = p(q, 4)
    var e0 = t * n
    var FPS = p(q, 5) + e0
    var TPS = p(q, 6) + e0
    var THR = p(q, 7) + e0
    var tp_i = 0
    var fp_i = 0
    var tp_w = Float32(0)
    var fp_w = Float32(0)
    var cnt = 0
    for i in range(n):
        var e = e0 + i
        if ldu(f, B + e) != UInt32(0):
            break
        var pos = ldi(f, G + e)
        if W >= 0:
            var w = ld(f, G + N + e)
            if pos == 1:
                tp_w = fadd(tp_w, w)
            else:
                fp_w = fadd(fp_w, w)
        else:
            if pos == 1:
                tp_i += 1
            else:
                fp_i += 1
        var s = ld(f, G + 2 * N + e)
        var last = i == n - 1
        if not last:
            if ldu(f, B + e + 1) != UInt32(0):
                last = True
            else:
                last = ld(f, G + 2 * N + e + 1) != s
        if last:
            if W >= 0:
                st(f, TPS + cnt, tp_w)
                st(f, FPS + cnt, fp_w)
            else:
                st(f, TPS + cnt, Float32(tp_i))
                st(f, FPS + cnt, Float32(fp_i))
            st(f, THR + cnt, s)
            cnt += 1
    sti(f, p(q, 8) + t, cnt)


# ---------------------------------------------------------------------------
# wpercentile's CDF
# ---------------------------------------------------------------------------

def wpct_gather_unit(t: Int, f: FP, q: IP):
    """q = [n, ORD, W, G]; t = element c*n + i: G[t] = the weight of the row
    at sorted position i of column c (1 when there are no weights)."""
    var W = p(q, 2)
    var r = ldi(f, p(q, 1) + t)
    f.unsafe_store(p(q, 3) + t, Float32(1) if W < 0 else ld(f, W + r))


def wpct_prefix_unit(t: Int, f: FP, q: IP):
    """q = [n, G, CDF]; t = column: CDF = the SEQUENTIAL ascending Float32
    prefix of the gathered weights (DEVIATION 6107), wpercentile_unit's loop."""
    var n = p(q, 0)
    var G = p(q, 1) + t * n
    var C = p(q, 2) + t * n
    var acc = Float32(0)
    for i in range(n):
        acc = fadd(acc, ld(f, G + i))
        st(f, C + i, acc)
