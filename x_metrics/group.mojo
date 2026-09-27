# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Grouped reductions: the one primitive every counting metric of the lane
stands on (confusion matrices and their margins, per-class supports, the
per-cluster sums of the dispersion scores, per-column regression sums).

`group_sort_unit` is a STABLE counting sort of the row indices by an integer
key (DEVIATION 6102: within a group the rows stay in ascending index order,
so every later fold over a group reads its rows in one order on every box).
`group_sum_unit` then folds each group's values through `PairSum`, or counts
them exactly as integers when there is neither a value nor a weight.
"""
from x_metrics.common import FP, IP, PairSum, p, ld, st, ldi, sti
from checks.numerics import identical_mul


def group_sort_unit(t: Int, f: FP, q: IP):
    """q = [K, n, m, OFF, ORD]. Keys ldi(K + r) outside [0, m) are dropped.
    Writes OFF[0..m] (group starts, OFF[m] = rows kept) and ORD[0..OFF[m])."""
    if t != 0:
        return
    var K = p(q, 0)
    var n = p(q, 1)
    var m = p(q, 2)
    var OFF = p(q, 3)
    var ORD = p(q, 4)
    for g in range(m + 1):
        sti(f, OFF + g, 0)
    for r in range(n):
        var k = ldi(f, K + r)
        if k >= 0 and k < m:
            sti(f, OFF + k + 1, ldi(f, OFF + k + 1) + 1)
    for g in range(m):
        sti(f, OFF + g + 1, ldi(f, OFF + g + 1) + ldi(f, OFF + g))
    # place: OFF[g] is the cursor of group g while placing
    for r in range(n):
        var k = ldi(f, K + r)
        if k >= 0 and k < m:
            var pos = ldi(f, OFF + k)
            sti(f, ORD + pos, r)
            sti(f, OFF + k, pos + 1)
    # OFF[g] now holds the END of group g; shift right to recover the starts
    var g2 = m
    while g2 >= 1:
        sti(f, OFF + g2, ldi(f, OFF + g2 - 1))
        g2 -= 1
    sti(f, OFF, 0)


def group_sum_unit(t: Int, f: FP, q: IP):
    """q = [OFF, ORD, V, VSTRIDE, W, OUT, D]; t in [0, m*D): group g = t / D,
    column c = t % D. OUT[t] = sum over the rows r of group g, ascending, of
    V[r*VSTRIDE + c] * W[r] (V < 0: 1; W < 0: no weight). With neither, the
    count is exact in Int and rounded once."""
    var OFF = p(q, 0)
    var ORD = p(q, 1)
    var V = p(q, 2)
    var VS = p(q, 3)
    var W = p(q, 4)
    var OUT = p(q, 5)
    var D = p(q, 6)
    var g = t // D
    var c = t - g * D
    var lo = ldi(f, OFF + g)
    var hi = ldi(f, OFF + g + 1)
    if V < 0 and W < 0:
        st(f, OUT + t, Float32(hi - lo))
        return
    var acc = PairSum()
    for i in range(lo, hi):
        var r = ldi(f, ORD + i)
        var v = Float32(1) if V < 0 else ld(f, V + r * VS + c)
        if W >= 0:
            v = identical_mul(v, ld(f, W + r))
        acc.add(v)
    st(f, OUT + t, acc.result())


def pair_key_unit(t: Int, f: FP, q: IP):
    """q = [A, B, OUT, k, mode]; one row per t. mode 0: A*k + B when both
    codes are in [0, k), else -1 (a confusion cell); mode 1: A when A == B
    and A is in [0, k), else -1 (a true positive's label); mode 2: 0 (every
    row in one group)."""
    var A = p(q, 0)
    var B = p(q, 1)
    var OUT = p(q, 2)
    var k = p(q, 3)
    var mode = p(q, 4)
    if mode == 2:
        sti(f, OUT + t, 0)
        return
    var a = ldi(f, A + t)
    var b = ldi(f, B + t)
    var ok = a >= 0 and a < k and b >= 0 and b < k
    if mode == 0:
        sti(f, OUT + t, a * k + b if ok else -1)
    else:
        sti(f, OUT + t, a if ok and a == b else -1)
