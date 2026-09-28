# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host's dense units, GROUPED (lane prep-cpu, 2026-09-28): `matmul`
(op 13), `class_stats` (16), `qda_cov` (40) and `qda_dec` (42) of
x_prep/units.mojo, the units' words with the loops turned inside out.

Each device unit folds ONE output over its inputs in ascending order. The
host walks a GROUP of outputs together and, at each step of that ascending
walk, folds the step into every output of the group:
- `matmul`: one output row i; at each l, every column j's accumulator
  (blocks of MM_BLOCK) takes A[i, l] * B[l, j]. B's row l is read once and
  in order, where the unit reads it once per column, strided.
- `class_stats` and `qda_cov`: one column c (one row a of the covariance);
  at each row i, the accumulator of row i's class takes the term. The unit
  of class k walks every row and skips the others; K classes read X K times.
- `qda_dec`: one unit (i, k); at each c, every projection r's accumulator
  takes (x_c - MEAN[k, c]) * R[k][c, r], R's row c read in order.
Every output still folds exactly its own terms, through the same `add`,
`sub`, `mul` and `div`, in the same ascending order, so every word is the
unit's. A stage whose shape does not factor as the grouping expects (or a
qda_dec wider than QDA_MAX_D) runs the device's units. Checked word for
word by `check_host_dense` in x_prep/seams/prep_check.mojo.
"""
from x_prep.common import FP, IP, p, ld, st
from x_prep.prims import add, sub, mul, div, matmul_unit, class_stats_unit
from naive_bayes.da import qda_cov_unit, qda_dec_unit

comptime MM_BLOCK = 16
comptime QDA_MAX_D = 64


# ---------------------------------------------------------------- matmul
@always_inline
def matmul_host_groups(total: Int, q: IP) -> Int:
    """Output rows of a matmul stage, or 0 when total is not rows * ncols."""
    var nc = p(q, 7)
    if nc <= 0 or total % nc != 0:
        return 0
    return total // nc


def matmul_host_row(i: Int, f: FP, q: IP):
    """q as `matmul_unit`; every unit i*ncols + j of output row i."""
    var nc = p(q, 7)
    var K = p(q, 8)
    var A = p(q, 0) + i * p(q, 1)
    var sa1 = p(q, 2)
    var B = p(q, 3)
    var sb0 = p(q, 4)
    var sb1 = p(q, 5)
    var j0 = 0
    while j0 < nc:
        var w = nc - j0
        if w > MM_BLOCK:
            w = MM_BLOCK
        var acc = InlineArray[Float32, MM_BLOCK](fill=Float32(0))
        for l in range(K):
            var a = ld(f, A + l * sa1)
            var brow = B + l * sb0 + j0 * sb1
            for jj in range(w):
                acc[jj] = add(acc[jj], mul(a, ld(f, brow + jj * sb1)))
        for jj in range(w):
            var v = acc[jj]
            if p(q, 10) >= 0:
                v = mul(v, ld(f, p(q, 10)))
            if p(q, 9) >= 0:
                v = add(v, ld(f, p(q, 9) + j0 + jj))
            st(f, p(q, 6) + i * nc + j0 + jj, v)
        j0 += w


# ---------------------------------------------------------------- class_stats
@always_inline
def class_stats_host_groups(total: Int, q: IP) -> Int:
    """Columns of a class_stats stage, or 0 when total is not K * d."""
    var d = p(q, 2)
    if d <= 0 or total != p(q, 4) * d:
        return 0
    return d


def class_stats_host_col(c: Int, f: FP, q: IP):
    """q as `class_stats_unit`; every unit k*d + c of column c."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var Y = p(q, 3)
    var K = p(q, 4)
    var s = List[Float32](length=K, fill=Float32(0))
    var cnt = List[Int](length=K, fill=0)
    for i in range(n):
        var k = Int(ld(f, Y + i))
        if k < 0 or k >= K:
            continue
        s[k] = add(s[k], ld(f, X + i * d + c))
        cnt[k] += 1
    var mean = List[Float32](length=K, fill=Float32(0))
    var ss = List[Float32](length=K, fill=Float32(0))
    for k in range(K):
        if cnt[k] > 0:
            mean[k] = div(s[k], Float32(cnt[k]))
    if p(q, 7) >= 0:
        for i in range(n):
            var k = Int(ld(f, Y + i))
            if k < 0 or k >= K or cnt[k] == 0:
                continue
            var e = sub(ld(f, X + i * d + c), mean[k])
            ss[k] = add(ss[k], mul(e, e))
        for k in range(K):
            if cnt[k] > 0:
                ss[k] = div(ss[k], Float32(cnt[k]))
    for k in range(K):
        var t = k * d + c
        if c == 0 and p(q, 5) >= 0:
            st(f, p(q, 5) + k, Float32(cnt[k]))
        if p(q, 6) >= 0:
            st(f, p(q, 6) + t, mean[k])
        if p(q, 7) >= 0:
            st(f, p(q, 7) + t, ss[k])
        if p(q, 8) >= 0:
            st(f, p(q, 8) + t, s[k])


# ---------------------------------------------------------------- qda_cov
@always_inline
def qda_cov_host_groups(total: Int, q: IP) -> Int:
    """Covariance rows a of a qda_cov stage (all classes at once), or 0 when
    total is not K * d * d."""
    var d = p(q, 2)
    if d <= 0 or total % (d * d) != 0:
        return 0
    return d


def qda_cov_host_row(a: Int, total: Int, f: FP, q: IP):
    """q as `qda_cov_unit`; every unit (k*d + a)*d + b of row a, every class."""
    var n = p(q, 1)
    var d = p(q, 2)
    var X = p(q, 0)
    var Y = p(q, 3)
    var M = p(q, 4)
    var K = total // (d * d)
    var acc = List[Float32](length=K * d, fill=Float32(0))
    for i in range(n):
        var k = Int(ld(f, Y + i))
        if k < 0 or k >= K:
            continue
        var ea = sub(ld(f, X + i * d + a), ld(f, M + k * d + a))
        var row = X + i * d
        var mk = M + k * d
        var base = k * d
        for b in range(d):
            var eb = sub(ld(f, row + b), ld(f, mk + b))
            acc[base + b] = add(acc[base + b], mul(ea, eb))
    for k in range(K):
        var cnt = ld(f, p(q, 5) + k)
        for b in range(d):
            st(f, p(q, 6) + (k * d + a) * d + b, div(acc[k * d + b], cnt))


# ---------------------------------------------------------------- qda_dec
def qda_dec_host_unit(t: Int, f: FP, q: IP):
    """q as `qda_dec_unit`; unit t = i*K + k, its projections accumulated
    together (d <= QDA_MAX_D; wider runs the unit)."""
    var d = p(q, 2)
    if d > QDA_MAX_D:
        qda_dec_unit(t, f, q)
        return
    var K = p(q, 6)
    var i = t // K
    var k = t % K
    var s = InlineArray[Float32, QDA_MAX_D](fill=Float32(0))
    var xrow = p(q, 0) + i * d
    var mk = p(q, 3) + k * d
    var R = p(q, 4) + k * d * d
    for c in range(d):
        var e = sub(ld(f, xrow + c), ld(f, mk + c))
        var rrow = R + c * d
        for r in range(d):
            s[r] = add(s[r], mul(e, ld(f, rrow + r)))
    var norm2 = Float32(0)
    for r in range(d):
        norm2 = add(norm2, mul(s[r], s[r]))
    st(f, p(q, 7) + t, sub(ld(f, p(q, 5) + k), mul(Float32(0.5), norm2)))
