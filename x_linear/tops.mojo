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
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from std.sys.compile import is_defined
from std.sys.info import is_amd_gpu, is_apple_gpu, is_nvidia_gpu
from x_linear.ops import fz as _fz
from x_linear.ops import xmad
from checks.numerics import identical_mul


@always_inline
def _acc_fa(acc: Float32, b: Float32) -> Float32:
    """fa(acc, b) for an acc that is already flushed (every acc below starts
    flushed and is only ever an fa/fmad result): ftz(acc) is acc itself, so
    the chain's critical path skips it. The same word as fa."""
    return _fz(acc + _fz(b))


@always_inline
def _acc_fmad(a: Float32, b: Float32, acc: Float32) -> Float32:
    """fmad(a, b, acc) for an already flushed acc (see _acc_fa)."""
    return _fz(xmad(_fz(a), _fz(b), acc))


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
        st(res, ooff + j, fd(fold_fa(x, j, d, n), i2f(n)))
    t.sync()


def t_centered_gram(t: Team, x: FP, n: Int, d: Int, xm: FP, xmoff: Int, g: FP, goff: Int):
    """ops.centered_gram, one thread per upper-triangle cell."""
    var cells = d * (d + 1) // 2
    for c in range(t.tid, cells, t.nt):
        var jk = upper_cell(c, d)
        var j = jk[0]
        var k = jk[1]
        var acc = chain_cfmad(x, j, d, ld(xm, xmoff + j), x, k, d, ld(xm, xmoff + k), n)
        st(g, goff + j * d + k, acc)
        st(g, goff + k * d + j, acc)
    t.sync()


def t_centered_xty(t: Team, x: FP, y: FP, n: Int, d: Int, xm: FP, xmoff: Int, ym: Float32, res: FP, ooff: Int):
    """ops.centered_xty, one thread per column."""
    for j in range(t.tid, d, t.nt):
        st(res, ooff + j, chain_cfmad(x, j, d, ld(xm, xmoff + j), y, 0, 1, ym, n))
    t.sync()


def t_sum(t: Team, v: FP, n: Int, k: Int = 0) -> Float32:
    """The lead's ascending fa fold of v[0:n], broadcast (slot k)."""
    var acc = Float32(0)
    if t.lead():
        acc = fold_fa(v, 0, 1, n)
    return t.bcast(acc, k)


def t_mean(t: Team, v: FP, n: Int, k: Int = 0) -> Float32:
    """ops.mean_of, on the lead, broadcast."""
    var m = Float32(0)
    if t.lead():
        m = fd(fold_fa(v, 0, 1, n), i2f(n))
    return t.bcast(m, k)


def t_sumsq(t: Team, v: FP, n: Int, k: Int = 0) -> Float32:
    """The lead's ascending fold acc = v_i * v_i + acc (one rounding), broadcast."""
    var acc = Float32(0)
    if t.lead():
        acc = fold_sq(v, 0, n)
    return t.bcast(acc, k)


# ----------------------------------------------------------------------------
# UNROLLED CHAINS: one thread's ascending fold over rows, its loads issued a
# block of CHAIN_U_DEVICE ahead of the arithmetic, so a device thread waits on
# memory once per block instead of once per row (the host keeps the plain
# loop: the blocked form was slower on an x86 core). The arithmetic is the
# plain loop's, operation for operation, in the same order: the bits do not
# move.
# ----------------------------------------------------------------------------

comptime CHAIN_U_DEVICE = CHAIN_U_APPLE if is_apple_gpu() else 32
#: Apple (lane/linear-apple2): the block of loads a chain issues before its
#: arithmetic. SCHEDULING only; set from the A/B on the M4 Pro.
comptime CHAIN_U_APPLE = 32


@always_inline
def fold_fa(v: FP, off: Int, step: Int, n: Int, init: Float32 = Float32(0)) -> Float32:
    """acc = fa(acc, v[off + i*step]), i ascending."""
    comptime U = CHAIN_U_DEVICE if (is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu()) else 1
    var acc = _fz(init)
    var i = 0
    while i + U <= n:
        var buf = SIMD[DType.float32, U]()
        comptime for u in range(U):
            buf[u] = ld(v, off + (i + u) * step)
        comptime for u in range(U):
            acc = _acc_fa(acc, buf[u])
        i += U
    while i < n:
        acc = _acc_fa(acc, ld(v, off + i * step))
        i += 1
    return acc


@always_inline
def fold_sq(v: FP, off: Int, n: Int, init: Float32 = Float32(0)) -> Float32:
    """acc = _acc_fmad(r, r, acc), r = v[off + i], i ascending."""
    comptime U = CHAIN_U_DEVICE if (is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu()) else 1
    var acc = _fz(init)
    var i = 0
    while i + U <= n:
        var buf = SIMD[DType.float32, U]()
        comptime for u in range(U):
            buf[u] = ld(v, off + i + u)
        comptime for u in range(U):
            acc = _acc_fmad(buf[u], buf[u], acc)
        i += U
    while i < n:
        var r = ld(v, off + i)
        acc = _acc_fmad(r, r, acc)
        i += 1
    return acc


@always_inline
def chain_fmad(a: FP, aoff: Int, astep: Int, b: FP, boff: Int, bstep: Int, n: Int,
               init: Float32 = Float32(0)) -> Float32:
    """acc = fmad(a[aoff + i*astep], b[boff + i*bstep], acc), i ascending."""
    comptime U = CHAIN_U_DEVICE if (is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu()) else 1
    var acc = _fz(init)
    var i = 0
    while i + U <= n:
        var pa = SIMD[DType.float32, U]()
        var pb = SIMD[DType.float32, U]()
        comptime for u in range(U):
            pa[u] = ld(a, aoff + (i + u) * astep)
            pb[u] = ld(b, boff + (i + u) * bstep)
        comptime for u in range(U):
            acc = _acc_fmad(pa[u], pb[u], acc)
        i += U
    while i < n:
        acc = _acc_fmad(ld(a, aoff + i * astep), ld(b, boff + i * bstep), acc)
        i += 1
    return acc


@always_inline
def chain_fmad_scaled(h: FP, x: FP, j: Int, k: Int, d: Int, n: Int) -> Float32:
    """acc = fmad(fm(h[i], x[i*d + j]), x[i*d + k], acc), i ascending (a
    weighted Gram cell)."""
    comptime U = CHAIN_U_DEVICE if (is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu()) else 1
    var acc = Float32(0)
    var i = 0
    while i + U <= n:
        var ph = SIMD[DType.float32, U]()
        var pj = SIMD[DType.float32, U]()
        var pk = SIMD[DType.float32, U]()
        comptime for u in range(U):
            ph[u] = ld(h, i + u)
            pj[u] = ld(x, (i + u) * d + j)
            pk[u] = ld(x, (i + u) * d + k)
        comptime for u in range(U):
            acc = _acc_fmad(_fm(ph[u], pj[u]), pk[u], acc)
        i += U
    while i < n:
        acc = _acc_fmad(_fm(ld(h, i), ld(x, i * d + j)), ld(x, i * d + k), acc)
        i += 1
    return acc


# Indexed forms: the rows are ix[0..cnt) (ascending row ids, a fold's
# training rows), element i of a row-indexed buffer is v[ix[q]].

@always_inline
def fold_fa_ix(v: FP, ix: IP, cnt: Int, off: Int = 0, step: Int = 1) -> Float32:
    """acc = fa(acc, v[off + ix[q]*step]), q ascending."""
    comptime U = CHAIN_U_DEVICE if (is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu()) else 1
    var acc = Float32(0)
    var q = 0
    while q + U <= cnt:
        var buf = SIMD[DType.float32, U]()
        comptime for u in range(U):
            buf[u] = ld(v, off + Int(ix.unsafe_load(q + u)) * step)
        comptime for u in range(U):
            acc = _acc_fa(acc, buf[u])
        q += U
    while q < cnt:
        acc = _acc_fa(acc, ld(v, off + Int(ix.unsafe_load(q)) * step))
        q += 1
    return acc


@always_inline
def fold_sq_ix(v: FP, ix: IP, cnt: Int) -> Float32:
    """acc = fmad(r, r, acc), r = v[ix[q]], q ascending."""
    comptime U = CHAIN_U_DEVICE if (is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu()) else 1
    var acc = Float32(0)
    var q = 0
    while q + U <= cnt:
        var buf = SIMD[DType.float32, U]()
        comptime for u in range(U):
            buf[u] = ld(v, Int(ix.unsafe_load(q + u)))
        comptime for u in range(U):
            acc = _acc_fmad(buf[u], buf[u], acc)
        q += U
    while q < cnt:
        var r = ld(v, Int(ix.unsafe_load(q)))
        acc = _acc_fmad(r, r, acc)
        q += 1
    return acc


@always_inline
def chain_fmad_ix(a: FP, b: FP, boff: Int, bstep: Int, ix: IP, cnt: Int) -> Float32:
    """acc = fmad(a[i], b[boff + i*bstep], acc), i = ix[q], q ascending."""
    comptime U = CHAIN_U_DEVICE if (is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu()) else 1
    var acc = Float32(0)
    var q = 0
    while q + U <= cnt:
        var pa = SIMD[DType.float32, U]()
        var pb = SIMD[DType.float32, U]()
        comptime for u in range(U):
            var i = Int(ix.unsafe_load(q + u))
            pa[u] = ld(a, i)
            pb[u] = ld(b, boff + i * bstep)
        comptime for u in range(U):
            acc = _acc_fmad(pa[u], pb[u], acc)
        q += U
    while q < cnt:
        var i = Int(ix.unsafe_load(q))
        acc = _acc_fmad(ld(a, i), ld(b, boff + i * bstep), acc)
        q += 1
    return acc


@always_inline
def chain_cfmad(a: FP, aoff: Int, astep: Int, ma: Float32, b: FP, boff: Int, bstep: Int, mb: Float32,
                n: Int) -> Float32:
    """acc = fmad(fs(a_i, ma), fs(b_i, mb), acc), a_i = a[aoff + i*astep],
    b_i = b[boff + i*bstep], i ascending (a centered cross product)."""
    comptime U = CHAIN_U_DEVICE if (is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu()) else 1
    var acc = Float32(0)
    var i = 0
    while i + U <= n:
        var pa = SIMD[DType.float32, U]()
        var pb = SIMD[DType.float32, U]()
        comptime for u in range(U):
            pa[u] = ld(a, aoff + (i + u) * astep)
            pb[u] = ld(b, boff + (i + u) * bstep)
        comptime for u in range(U):
            acc = _acc_fmad(fs(pa[u], ma), fs(pb[u], mb), acc)
        i += U
    while i < n:
        acc = _acc_fmad(fs(ld(a, aoff + i * astep), ma), fs(ld(b, boff + i * bstep), mb), acc)
        i += 1
    return acc


@always_inline
def chain_cfmad_ix(a: FP, aoff: Int, astep: Int, ma: Float32, b: FP, boff: Int, bstep: Int, mb: Float32,
                   ix: IP, cnt: Int) -> Float32:
    """chain_cfmad over the rows ix[0..cnt)."""
    comptime U = CHAIN_U_DEVICE if (is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu()) else 1
    var acc = Float32(0)
    var q = 0
    while q + U <= cnt:
        var pa = SIMD[DType.float32, U]()
        var pb = SIMD[DType.float32, U]()
        comptime for u in range(U):
            var i = Int(ix.unsafe_load(q + u))
            pa[u] = ld(a, aoff + i * astep)
            pb[u] = ld(b, boff + i * bstep)
        comptime for u in range(U):
            acc = _acc_fmad(fs(pa[u], ma), fs(pb[u], mb), acc)
        q += U
    while q < cnt:
        var i = Int(ix.unsafe_load(q))
        acc = _acc_fmad(fs(ld(a, aoff + i * astep), ma), fs(ld(b, boff + i * bstep), mb), acc)
        q += 1
    return acc


@always_inline
def fold_one_fmad(v: FP, off: Int, n: Int) -> Float32:
    """acc = fmad(1, v[off + i], acc), i ascending (an intercept column of
    ones in a cross product, spelled as the plain loop spells it)."""
    comptime U = CHAIN_U_DEVICE if (is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu()) else 1
    var acc = Float32(0)
    var i = 0
    while i + U <= n:
        var buf = SIMD[DType.float32, U]()
        comptime for u in range(U):
            buf[u] = ld(v, off + i + u)
        comptime for u in range(U):
            acc = _acc_fmad(Float32(1), buf[u], acc)
        i += U
    while i < n:
        acc = _acc_fmad(Float32(1), ld(v, off + i), acc)
        i += 1
    return acc


# ------------------------------------------------ staged folds (lane/neural-pass88)
# A fold whose order is the rows ascending runs on ONE thread; on the device
# that thread waited on global memory once per CHAIN_U_DEVICE rows (GLM's
# objective: a million-row fold per line-search trial). Here the whole team
# stages the next chunk of the vector into threadgroup memory (coalesced,
# double-buffered) while the lead folds the current chunk from it: the same
# `_acc_fa` steps in the same order, so the same word; the lead's chain no
# longer waits on DRAM. STAGE_CH floats a buffer, two buffers: 8 KB, inside
# every column's threadgroup limit (Apple 32 KB).
comptime STAGE_CH = 1024


def t_fold_fa_staged(t: Team, v: FP, off: Int, n: Int, init: Float32 = Float32(0)) -> Float32:
    """`fold_fa(v, off, 1, n, init)` on the team; the value is the lead's
    (other threads return their partial garbage: callers broadcast or use the
    lead's). A team of one runs `fold_fa`. `-D MOJOLEARN_X_LINEAR_NO_STAGED_FOLD=1`
    restores `fold_fa` on the lead."""
    comptime if not (is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu()) or is_defined["MOJOLEARN_X_LINEAR_NO_STAGED_FOLD"]():
        var r0 = Float32(0)
        if t.lead():
            r0 = fold_fa(v, off, 1, n, init)
        return r0
    else:
        if t.nt <= 1:
            return fold_fa(v, off, 1, n, init)
        var buf = stack_allocation[2 * STAGE_CH, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
        var chunks = (n + STAGE_CH - 1) // STAGE_CH
        var acc = _fz(init)
        if chunks > 0:
            for u in range(t.tid, min(STAGE_CH, n), t.nt):
                buf[u] = ld(v, off + u)
        t.sync()
        for c in range(chunks):
            var cur = (c & 1) * STAGE_CH
            if c + 1 < chunks:
                # the helpers load the next chunk while the lead folds this one
                var nxt = ((c + 1) & 1) * STAGE_CH
                var base = (c + 1) * STAGE_CH
                var cnt = min(STAGE_CH, n - base)
                if not t.lead():
                    for u in range(t.tid - 1, cnt, t.nt - 1):
                        buf[nxt + u] = ld(v, off + base + u)
            if t.lead():
                var cnt_c = min(STAGE_CH, n - c * STAGE_CH)
                for u in range(cnt_c):
                    acc = _acc_fa(acc, buf[cur + u])
            t.sync()
        return acc
