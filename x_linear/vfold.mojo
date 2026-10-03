# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PARAMETER-VECTOR FOLD (lane cgr4-device-optim, 2026-10-03): every
d- or P-sized sum of the L-BFGS fits (x_linear/lbfgs.mojo, the device
driver x_linear/lbfgs_device.mojo, the objectives' penalty terms) in one
order, the same words on the host and on every device:

  * the vector is cut into chunks of VCH = 32 entries; a chunk's partial is
    its entries folded ascending from zero (`fmad` for a product, `fmax` of
    `|v|` for a max norm);
  * the chunk partials are folded by the ALIGNED BINARY TREE: the node over
    chunks [i 2^L, (i+1) 2^L) is its left child `fa` its right child, or
    its left child alone when the right one is empty. A device team folds
    level by level (pairs (2i, 2i+1) to i, an odd last one carried, the
    levels ping-ponged through scratch); one thread folds the same tree
    with a binary counter (complete blocks merged as they close, the open
    right spine merged right to left at the end). Both evaluate the same
    nodes from the same children: the same words.

Before this lane the host L-BFGS (lead thread) folded every P-vector dot
ascending over all P entries; the bits move (old bits do not matter).
"""
from x_linear.ops import FP, fa, fm, fmad, fabs, fmax, ld, st
from x_linear.team import Team

comptime VCH = 32
comptime VOP_SUM = 0
comptime VOP_MAX = 1
#: chunks a one-thread fold can hold open: 2^48 chunks
comptime VSTACK = 48


@always_inline
def vchunks(p: Int) -> Int:
    return (p + VCH - 1) // VCH


@always_inline
def vscratch(p: Int) -> Int:
    """Device scratch words of one team fold over p entries."""
    return 2 * max(vchunks(p), 1)


@always_inline
def _vop[op: Int](a: Float32, b: Float32) -> Float32:
    comptime if op == VOP_SUM:
        return fa(a, b)
    else:
        return fmax(a, b)


@always_inline
def vpart_dot(a: FP, ia: Int, b: FP, ib: Int, p: Int, c: Int) -> Float32:
    var acc = Float32(0)
    var hi = min(c * VCH + VCH, p)
    for j in range(c * VCH, hi):
        acc = fmad(ld(a, ia + j), ld(b, ib + j), acc)
    return acc


@always_inline
def vpart_absmax(a: FP, ia: Int, p: Int, c: Int) -> Float32:
    var acc = Float32(0)
    var hi = min(c * VCH + VCH, p)
    for j in range(c * VCH, hi):
        acc = fmax(acc, fabs(ld(a, ia + j)))
    return acc


@always_inline
def vpart_wsq(th: FP, toff: Int, d: Int, stride: Int, q: Int, c: Int) -> Float32:
    """Entries e of [0, q): w = th[toff + (e // d) * stride + e % d], w*w
    (the penalty over every class's weights, intercepts skipped)."""
    var acc = Float32(0)
    var hi = min(c * VCH + VCH, q)
    for e in range(c * VCH, hi):
        var k = e // d
        var w = ld(th, toff + k * stride + (e - k * d))
        acc = fmad(w, w, acc)
    return acc


@always_inline
def vpart_abssum(a: FP, ia: Int, p: Int, c: Int) -> Float32:
    var acc = Float32(0)
    var hi = min(c * VCH + VCH, p)
    for j in range(c * VCH, hi):
        acc = fa(acc, fabs(ld(a, ia + j)))
    return acc


@always_inline
def _vpart[kind: Int](a: FP, ia: Int, b: FP, ib: Int, p: Int, d: Int, stride: Int, c: Int) -> Float32:
    comptime if kind == 0:
        return vpart_dot(a, ia, b, ib, p, c)
    elif kind == 1:
        return vpart_absmax(a, ia, p, c)
    elif kind == 3:
        return vpart_abssum(a, ia, p, c)
    else:
        return vpart_wsq(a, ia, d, stride, p, c)


def _vfold_serial[kind: Int, op: Int](a: FP, ia: Int, b: FP, ib: Int, p: Int, d: Int, stride: Int) -> Float32:
    """One thread: the aligned tree over the chunk partials (binary counter)."""
    var vals = InlineArray[Float32, VSTACK](fill=Float32(0))
    var lvls = InlineArray[Int, VSTACK](fill=0)
    var top = 0
    var m = vchunks(p)
    for c in range(m):
        var v = _vpart[kind](a, ia, b, ib, p, d, stride, c)
        var lv = 0
        while top > 0 and lvls[top - 1] == lv:
            v = _vop[op](vals[top - 1], v)
            top -= 1
            lv += 1
        vals[top] = v
        lvls[top] = lv
        top += 1
    if top == 0:
        return Float32(0)
    var r = vals[top - 1]
    var k = top - 2
    while k >= 0:
        r = _vop[op](vals[k], r)
        k -= 1
    return r


def _vfold_team[kind: Int, op: Int](t: Team, a: FP, ia: Int, b: FP, ib: Int, p: Int, d: Int, stride: Int,
                                    parts: FP) -> Float32:
    """The team: one thread a chunk, then the tree level by level through
    parts[0, m) and parts[m, 2m). Every thread returns the root."""
    var m = vchunks(p)
    if m == 0:
        return Float32(0)
    for c in range(t.tid, m, t.nt):
        st(parts, c, _vpart[kind](a, ia, b, ib, p, d, stride, c))
    t.sync()
    var src = 0
    var dst = m
    var cur = m
    while cur > 1:
        var h = cur // 2
        var nxt = (cur + 1) // 2
        for i in range(t.tid, nxt, t.nt):
            if i < h:
                st(parts, dst + i, _vop[op](ld(parts, src + 2 * i), ld(parts, src + 2 * i + 1)))
            else:
                st(parts, dst + i, ld(parts, src + cur - 1))
        t.sync()
        var s2 = src
        src = dst
        dst = s2
        cur = nxt
    var r = ld(parts, src)
    t.sync()
    return r


@always_inline
def _vfold[kind: Int, op: Int](t: Team, a: FP, ia: Int, b: FP, ib: Int, p: Int, d: Int, stride: Int,
                               parts: FP) -> Float32:
    if t.nt == 1:
        return _vfold_serial[kind, op](a, ia, b, ib, p, d, stride)
    return _vfold_team[kind, op](t, a, ia, b, ib, p, d, stride, parts)


def vdot(t: Team, a: FP, ia: Int, b: FP, ib: Int, p: Int, parts: FP) -> Float32:
    """sum_j a[ia + j] b[ib + j], j < p, in the fold order above. A team of
    one ignores parts; a device team needs vscratch(p) words there."""
    return _vfold[0, VOP_SUM](t, a, ia, b, ib, p, 0, 0, parts)


def vabsmax(t: Team, a: FP, ia: Int, p: Int, parts: FP) -> Float32:
    """max_j |a[ia + j]| (0 for p == 0), chunk maxima then the tree."""
    return _vfold[1, VOP_MAX](t, a, ia, a, ia, p, 0, 0, parts)


def vabssum(t: Team, a: FP, ia: Int, p: Int, parts: FP) -> Float32:
    """sum_j |a[ia + j]|, chunk sums then the tree."""
    return _vfold[3, VOP_SUM](t, a, ia, a, ia, p, 0, 0, parts)


def vwsq(t: Team, th: FP, toff: Int, d: Int, stride: Int, kp: Int, parts: FP) -> Float32:
    """sum over kp classes and j < d of th[toff + k stride + j]^2."""
    return _vfold[2, VOP_SUM](t, th, toff, th, toff, kp * d, d, stride, parts)


def vdot1(a: FP, ia: Int, b: FP, ib: Int, p: Int) -> Float32:
    """`vdot` on one thread (no team)."""
    return _vfold_serial[0, VOP_SUM](a, ia, b, ib, p, 0, 0)


def vwsq1(th: FP, toff: Int, d: Int, stride: Int, kp: Int) -> Float32:
    """`vwsq` on one thread."""
    return _vfold_serial[2, VOP_SUM](th, toff, th, toff, kp * d, d, stride)
