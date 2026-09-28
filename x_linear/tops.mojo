# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""TEAM FORMS of the shared row passes of x_linear/ops.mojo (lane linear,
speed phase). Each is the one-thread function it names with the independent
outputs dealt across the team (x_linear/team.mojo): an output's fold is still
ONE thread's loop over ascending rows, so every value carries the one-thread
bits. Each returns with its outputs visible to the whole team; a scalar
result is the lead's fold, broadcast.
"""
from x_linear.ops import FP, IP, fa, fs, fm, fd, fmad, ld, st, i2f
from x_linear.team import Team


@always_inline
def upper_cell(c: Int, d: Int) -> Tuple[Int, Int]:
    """The c-th cell (j, k), k >= j, of a d x d upper triangle, row by row."""
    var j = 0
    var q = c
    while q >= d - j:
        q -= d - j
        j += 1
    return (j, j + q)


def t_col_means(t: Team, x: FP, n: Int, d: Int, res: FP, ooff: Int):
    """res[ooff + j] = (sum_i x_ij) / n, rows ascending (bayes `_center`)."""
    for j in range(t.tid, d, t.nt):
        var acc = Float32(0)
        for i in range(n):
            acc = fa(acc, ld(x, i * d + j))
        st(res, ooff + j, fd(acc, i2f(n)))
    t.sync()


def t_centered_gram(t: Team, x: FP, n: Int, d: Int, xm: FP, xmoff: Int, g: FP, goff: Int):
    """ops.centered_gram, one thread per upper-triangle cell."""
    var cells = d * (d + 1) // 2
    for c in range(t.tid, cells, t.nt):
        var jk = upper_cell(c, d)
        var j = jk[0]
        var k = jk[1]
        var acc = Float32(0)
        var mj = ld(xm, xmoff + j)
        var mk = ld(xm, xmoff + k)
        for i in range(n):
            acc = fmad(fs(ld(x, i * d + j), mj), fs(ld(x, i * d + k), mk), acc)
        st(g, goff + j * d + k, acc)
        st(g, goff + k * d + j, acc)
    t.sync()


def t_centered_xty(t: Team, x: FP, y: FP, n: Int, d: Int, xm: FP, xmoff: Int, ym: Float32, res: FP, ooff: Int):
    """ops.centered_xty, one thread per column."""
    for j in range(t.tid, d, t.nt):
        var acc = Float32(0)
        var mj = ld(xm, xmoff + j)
        for i in range(n):
            acc = fmad(fs(ld(x, i * d + j), mj), fs(ld(y, i), ym), acc)
        st(res, ooff + j, acc)
    t.sync()


def t_sum(t: Team, v: FP, n: Int, k: Int = 0) -> Float32:
    """The lead's ascending fa fold of v[0:n], broadcast (slot k)."""
    var acc = Float32(0)
    if t.lead():
        for i in range(n):
            acc = fa(acc, ld(v, i))
    return t.bcast(acc, k)


def t_mean(t: Team, v: FP, n: Int, k: Int = 0) -> Float32:
    """ops.mean_of, on the lead, broadcast."""
    var m = Float32(0)
    if t.lead():
        var acc = Float32(0)
        for i in range(n):
            acc = fa(acc, ld(v, i))
        m = fd(acc, i2f(n))
    return t.bcast(m, k)


def t_sumsq(t: Team, v: FP, n: Int, k: Int = 0) -> Float32:
    """The lead's ascending fold acc = v_i * v_i + acc (one rounding), broadcast."""
    var acc = Float32(0)
    if t.lead():
        for i in range(n):
            var r = ld(v, i)
            acc = fmad(r, r, acc)
    return t.bcast(acc, k)
