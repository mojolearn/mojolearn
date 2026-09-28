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
from x_linear.ops import fz as _fz
from checks.numerics import identical_mul_add, identical_mul


@always_inline
def _acc_fa(acc: Float32, b: Float32) -> Float32:
    """fa(acc, b) for an acc that is already flushed (every acc below starts
    flushed and is only ever an fa/fmad result): ftz(acc) is acc itself, so
    the chain's critical path skips it. The same word as fa."""
    return _fz(acc + _fz(b))


@always_inline
def _acc_fmad(a: Float32, b: Float32, acc: Float32) -> Float32:
    """fmad(a, b, acc) for an already flushed acc (see _acc_fa)."""
    return _fz(identical_mul_add(_fz(a), _fz(b), acc))


@always_inline
def _fm(a: Float32, b: Float32) -> Float32:
    """ops.fm with the branchless flush."""
    return _fz(identical_mul(_fz(a), _fz(b)))


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


# ----------------------------------------------------------------------------
# UNROLLED CHAINS: one thread's ascending fold over rows, its loads issued a
# block of CHAIN_U ahead of the arithmetic, so a device thread waits on memory
# once per block instead of once per row. The arithmetic is the plain loop's,
# operation for operation, in the same order: the bits do not move.
# ----------------------------------------------------------------------------

comptime CHAIN_U = 32


@always_inline
def fold_fa(v: FP, off: Int, step: Int, n: Int, init: Float32 = Float32(0)) -> Float32:
    """acc = fa(acc, v[off + i*step]), i ascending."""
    var acc = _fz(init)
    var i = 0
    while i + CHAIN_U <= n:
        var buf = SIMD[DType.float32, CHAIN_U]()
        comptime for u in range(CHAIN_U):
            buf[u] = ld(v, off + (i + u) * step)
        comptime for u in range(CHAIN_U):
            acc = _acc_fa(acc, buf[u])
        i += CHAIN_U
    while i < n:
        acc = _acc_fa(acc, ld(v, off + i * step))
        i += 1
    return acc


@always_inline
def fold_sq(v: FP, off: Int, n: Int, init: Float32 = Float32(0)) -> Float32:
    """acc = _acc_fmad(r, r, acc), r = v[off + i], i ascending."""
    var acc = _fz(init)
    var i = 0
    while i + CHAIN_U <= n:
        var buf = SIMD[DType.float32, CHAIN_U]()
        comptime for u in range(CHAIN_U):
            buf[u] = ld(v, off + i + u)
        comptime for u in range(CHAIN_U):
            acc = _acc_fmad(buf[u], buf[u], acc)
        i += CHAIN_U
    while i < n:
        var r = ld(v, off + i)
        acc = _acc_fmad(r, r, acc)
        i += 1
    return acc


@always_inline
def chain_fmad(a: FP, aoff: Int, astep: Int, b: FP, boff: Int, bstep: Int, n: Int,
               init: Float32 = Float32(0)) -> Float32:
    """acc = fmad(a[aoff + i*astep], b[boff + i*bstep], acc), i ascending."""
    var acc = _fz(init)
    var i = 0
    while i + CHAIN_U <= n:
        var pa = SIMD[DType.float32, CHAIN_U]()
        var pb = SIMD[DType.float32, CHAIN_U]()
        comptime for u in range(CHAIN_U):
            pa[u] = ld(a, aoff + (i + u) * astep)
            pb[u] = ld(b, boff + (i + u) * bstep)
        comptime for u in range(CHAIN_U):
            acc = _acc_fmad(pa[u], pb[u], acc)
        i += CHAIN_U
    while i < n:
        acc = _acc_fmad(ld(a, aoff + i * astep), ld(b, boff + i * bstep), acc)
        i += 1
    return acc


@always_inline
def chain_fmad_scaled(h: FP, x: FP, j: Int, k: Int, d: Int, n: Int) -> Float32:
    """acc = fmad(fm(h[i], x[i*d + j]), x[i*d + k], acc), i ascending (a
    weighted Gram cell)."""
    var acc = Float32(0)
    var i = 0
    while i + CHAIN_U <= n:
        var ph = SIMD[DType.float32, CHAIN_U]()
        var pj = SIMD[DType.float32, CHAIN_U]()
        var pk = SIMD[DType.float32, CHAIN_U]()
        comptime for u in range(CHAIN_U):
            ph[u] = ld(h, i + u)
            pj[u] = ld(x, (i + u) * d + j)
            pk[u] = ld(x, (i + u) * d + k)
        comptime for u in range(CHAIN_U):
            acc = _acc_fmad(_fm(ph[u], pj[u]), pk[u], acc)
        i += CHAIN_U
    while i < n:
        acc = _acc_fmad(_fm(ld(h, i), ld(x, i * d + j)), ld(x, i * d + k), acc)
        i += 1
    return acc
