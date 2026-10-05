# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DART's boosting round as per-element units (lane fam2-forests,
2026-10-04, `IDN_DART_DEVICE`).

The bodies of xtrees/dart_device.mojo's kernels, lifted out so that the
device kernels (NVIDIA, AMD, Apple) and the host column
(xtrees/dart_host.mojo) run the SAME statements: one thread or one loop
turn per unit, every float operation written once.

FAST (`DART_PIN` False) compiles each helper to the plain expression the
kernels held before this file existed, so the FAST + Apple default of
lane/apple-fast-dart is unchanged. IDENTICAL (`DART_PIN` True) pins every
float operation:
  add / subtract   one IEEE operation, result through `ftz` (the denormal
                   policy of every IDENTICAL kernel);
  multiply         `identical_mul` (the contraction pin: no fused
                   multiply-add may form with a following add), result
                   through `ftz`;
  divide           `identical_div`;
  exp              `identical_exp` (the portable Cephes form);
  sigmoid          `identical_sigmoid`;
  compares, abs, negation, max: exact on every vendor, left as written.
Folds are sequential inside one unit in a fixed index order (trees
ascending, classes ascending, rows ascending within a chunk, chunks
ascending), never an atomic, so the order is the same on every column."""
from std.math import exp
from std.sys.compile import is_defined
from checks.numerics import (
    GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul, identical_div, identical_exp, identical_sigmoid,
)
from checks.soft_f64 import sf64_add, sf64_mul, sf64_from_f32
from xtrees.ops import draw

comptime DART_PIN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

#: The IDENTICAL route's gate, default ON on every column (device kernels in
#: xtrees/dart_device.mojo, host twin in xtrees/dart_host.mojo).
#: `-D MOJOLEARN_IDN_DART_DEVICE_OFF` or `-D MOJOLEARN_IDN_ALL_OFF` restores
#: main's `_boost_loop` everywhere (the entries are then not registered).
comptime IDN_DART_DEVICE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_DART_DEVICE_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

#: rows per leaf-sum chunk: one unit per (node, chunk) scans the chunk's
#: leaf indices in row order.
comptime DART_CHUNK = 8192

comptime F32P = MutPointer[Float32, MutAnyOrigin]
comptime I32P = MutPointer[Int32, MutAnyOrigin]
comptime I64P = MutPointer[Int64, MutAnyOrigin]
comptime U16P = MutPointer[UInt16, MutAnyOrigin]
comptime U64P = MutPointer[UInt64, MutAnyOrigin]


# ------------------------------------------------------------ pinned helpers
@always_inline
def _add(a: Float32, b: Float32) -> Float32:
    comptime if DART_PIN:
        return ftz(a + b)
    else:
        return a + b


@always_inline
def _sub(a: Float32, b: Float32) -> Float32:
    comptime if DART_PIN:
        return ftz(a - b)
    else:
        return a - b


@always_inline
def _mul(a: Float32, b: Float32) -> Float32:
    comptime if DART_PIN:
        return ftz(identical_mul(a, b))
    else:
        return a * b


@always_inline
def _mul_add(a: Float32, b: Float32, c: Float32) -> Float32:
    """c + a * b. IDENTICAL: a pinned product, then one add (never fused)."""
    comptime if DART_PIN:
        return ftz(c + ftz(identical_mul(a, b)))
    else:
        return c + a * b


@always_inline
def _div(a: Float32, b: Float32) -> Float32:
    comptime if DART_PIN:
        return identical_div(a, b)
    else:
        return a / b


@always_inline
def _exp(x: Float32) -> Float32:
    comptime if DART_PIN:
        return identical_exp(x)
    else:
        return exp(x)


@always_inline
def _in(x: Float32) -> Float32:
    """A caller-supplied value as the units read it: IDENTICAL flushes a
    subnormal input so every column starts from the same word."""
    comptime if DART_PIN:
        return ftz(x)
    else:
        return x


@always_inline
def _sigmoid(s: Float32) -> Float32:
    comptime if DART_PIN:
        return identical_sigmoid(s)
    else:
        return 1.0 / (1.0 + exp(-s))


# --------------------------------------------------------------------- units
@always_inline
def dart_init_unit(e: Int, n: Int, inits: F32P, score: F32P):
    """score[c * n + i] = inits[c]."""
    score[e] = _in(inits[e // n])


@always_inline
def dart_drop_unit(e: Int, base: UInt64, skip: Bool, thr: I64P, flags: I32P):
    """flags[e] = 1 when iteration e is dropped this round (draw 1 + e of
    the round's stream against thr[e] = ceil(rate_e * 2^53)); integers
    only."""
    if skip:
        flags[e] = Int32(0)
    else:
        flags[e] = Int32(1) if (draw(base, e + 1) >> 11) < UInt64(thr[e]) else Int32(0)


@always_inline
def dart_row_unit(
    i: Int, nn: Int, kk: Int, t: Int, kind: Int32, node_cap: Int, flags: I32P, coef: F32P, nodes: U16P,
    values: F32P, y: F32P, score: F32P, dsum: F32P, target: F32P, h: F32P,
):
    """Row i: dsum[c, i] = the dropped trees' coef x leaf value (trees in
    ascending order), taken off the score; then main's `gradients` (kind 0
    L2, 1 logloss, 2 softmax, class-major) into target = -g and h."""
    for c in range(kk):
        var s: Float32 = 0.0
        for it in range(t):
            if flags[it] != Int32(0):
                var j = it * kk + c
                s = _mul_add(coef[j], values[j * node_cap + Int(nodes[j * nn + i])], s)
        dsum[c * nn + i] = s
        score[c * nn + i] = _sub(score[c * nn + i], s)
    if kind == Int32(2):
        var m = score[i]
        for c in range(1, kk):
            if score[c * nn + i] > m:
                m = score[c * nn + i]
        var tot: Float32 = 0.0
        for c in range(kk):
            tot = _add(tot, _exp(_sub(score[c * nn + i], m)))
        var yi = Int(y[i])
        var factor = _div(Float32(kk), Float32(kk) - 1.0)
        for c in range(kk):
            var p = _div(_exp(_sub(score[c * nn + i], m)), tot)
            var g = _sub(p, 1.0) if yi == c else p
            target[c * nn + i] = -g
            h[c * nn + i] = _mul(_mul(factor, p), _sub(1.0, p))
    else:
        var s = score[i]
        var yv = _in(y[i])
        var g: Float32
        var hv: Float32
        if kind == Int32(0):
            g = _sub(s, yv)
            hv = 1.0
        else:
            var p = _sigmoid(s)
            g = _sub(p, yv)
            hv = _mul(p, _sub(1.0, p))
        target[i] = -g
        h[i] = hv


@always_inline
def dart_apply_unit(
    i: Int, dd: Int, cnt: Int, colid: I32P, quesval: F32P, left: I32P, x: F32P, row_off: Int, nodes: U16P,
    bad: I32P,
):
    """nodes[row_off + i] = the leaf node row i reaches (main's `apply_trees`
    walk: left child l, right l + 1, leaf where left is -1; the compare is
    `apply_trees`'s `x <= quesval`). A walk that leaves the tree sets bad[0]
    and lands on node 0; the caller raises."""
    var node = 0
    var steps = 0
    var ok = True
    while True:
        var l = Int(left[node])
        if l == -1:
            break
        var c = Int(colid[node])
        if c < 0 or c >= dd or l < 1 or l + 1 >= cnt or steps > cnt:
            ok = False
            break
        if x[i * dd + c] <= quesval[node]:
            node = l
        else:
            node = l + 1
        steps += 1
    if not ok:
        bad[0] = Int32(1)
        node = 0
    nodes[row_off + i] = UInt16(node)


@always_inline
def dart_leaf_sum_unit(
    e: Int, nn: Int, nk: Int, row_off: Int, class_off: Int, nodes: U16P, target: F32P, h: F32P, part: F32P,
):
    """part[(q * nk + k) * 2 + {0, 1}] = sum of g (= -target) and h over the
    rows of chunk q that reached node k, in row order (e = q * nk + k)."""
    var q = e // nk
    var kk = e - q * nk
    var r0 = q * DART_CHUNK
    var r1 = min(r0 + DART_CHUNK, nn)
    var sg: Float32 = 0.0
    var sh: Float32 = 0.0
    for i in range(r0, r1):
        if Int(nodes[row_off + i]) == kk:
            sg = _sub(sg, target[class_off + i])
            sh = _add(sh, h[class_off + i])
    part[e * 2] = sg
    part[e * 2 + 1] = sh


@always_inline
def dart_leaf_sum_rows_unit(
    e: Int, nn: Int, m: Int, nk: Int, row_off: Int, class_off: Int, rows: I32P, nodes: U16P, target: F32P, h: F32P,
    part: F32P, bad: I32P,
):
    """The bagged twin of `dart_leaf_sum_unit` (lane cpu2-l5-trees): chunk q
    is the bag LIST positions q * DART_CHUNK .. (q + 1) * DART_CHUNK of
    rows[0 .. m), scanned in list order, so only the bagged rows feed the
    leaf values (LightGBM: the leaf output comes from the bagged rows; main's
    `leaf_newton_rows`). The walk and the score update still cover every row.
    A bag row outside [0, nn) sets bad[0] and is skipped; the caller raises."""
    var q = e // nk
    var kk = e - q * nk
    var p0 = q * DART_CHUNK
    var p1 = min(p0 + DART_CHUNK, m)
    var sg: Float32 = 0.0
    var sh: Float32 = 0.0
    for p in range(p0, p1):
        var i = Int(rows[p])
        if i < 0 or i >= nn:
            bad[0] = Int32(1)
            continue
        if Int(nodes[row_off + i]) == kk:
            sg = _sub(sg, target[class_off + i])
            sh = _add(sh, h[class_off + i])
    part[e * 2] = sg
    part[e * 2 + 1] = sh


@always_inline
def dart_newton_unit(
    e: Int, nk: Int, n_chunks: Int, part: F32P, lam: Float32, l1: Float32, mds: Float32, voff: Int, values: F32P,
):
    """values[voff + e] = main's `_newton_values` of node e's chunk sums
    folded in chunk order: -ThresholdL1(sum g, l1) / (sum h + lambda),
    clipped to +-max_delta_step when that is > 0, 0 where the denominator is
    not positive."""
    var sg: Float32 = 0.0
    var sh: Float32 = 0.0
    for q in range(n_chunks):
        sg = _add(sg, part[(q * nk + e) * 2])
        sh = _add(sh, part[(q * nk + e) * 2 + 1])
    var den = _add(sh, lam)
    var ret: Float32 = 0.0
    if den > 0.0:
        var s = sg
        if l1 > 0.0:
            var a = _sub(s if s >= 0.0 else -s, l1)
            var reg = a if a > 0.0 else 0.0
            s = reg if s > 0.0 else (-reg if s < 0.0 else 0.0)
        ret = _div(-s, den)
        if mds > 0.0 and (ret if ret >= 0.0 else -ret) > mds:
            ret = mds if ret > 0.0 else -mds
    values[voff + e] = ret


@always_inline
def dart_add_unit(
    e: Int, class_off: Int, row_off: Int, voff: Int, factor: Float32, shrink: Float32, nodes: U16P, values: F32P,
    dsum: F32P, score: F32P,
):
    """score[c, i] = (score[c, i] + factor * dsum[c, i]) + shrink *
    values[leaf of row i]: the dropped trees back at their rescaled weight
    and the new tree at its shrinkage, in that order."""
    var ci = class_off + e
    score[ci] = _mul_add(shrink, values[voff + Int(nodes[row_off + e])], _mul_add(factor, dsum[ci], score[ci]))


@always_inline
def dart_predict_unit(
    e: Int, nn: Int, dd: Int, kk: Int, nt: Int, toff: I32P, colid: I32P, quesval: F32P, left: I32P, values: F32P,
    coef: U64P, inits: U64P, x: F32P, dst: U64P, bad: I32P,
):
    """DART's raw score of row i for class c (lane cpu2-l5-trees,
    `x_trees_dart_predict`; e = c * nn + i, the class-major layout of
    `_DARTBase._raw`): dst[e] = inits[c] + sum over the trees j = c, c + kk,
    c + 2 kk, ... (ascending) of coef[j] x values[leaf of row i in tree j].

    The forest is concatenated: tree j's nodes are toff[j] .. toff[j + 1] of
    colid / quesval / left / values (left and colid tree-relative, the walk
    of `apply_trees`: `x <= quesval` goes to the left child l, else l + 1,
    a leaf where left is -1). A walk that leaves its tree sets bad[0] and
    the tree adds nothing; the caller raises.

    BITS. float64 words (UInt64), every operation `checks/soft_f64.mojo`'s
    correctly rounded binary64: the product sf64_mul(coef, widen(value)),
    then one sf64_add onto the running score, per tree in ascending order.
    That is main's per-tree `tree_score_add` (`acc + identical_mul64(w,
    Float64(v))`, an unfused product and an IEEE add) word for word under
    IDENTICAL, on every vendor (Apple has no float64) and on the host twin.
    FAST spells the same operations (main's FAST host add could fuse the
    product into the add: at most the last bit of a raw score differs)."""
    var c = e // nn
    var i = e - c * nn
    var acc = inits[c]
    var j = c
    while j < nt:
        var lo = Int(toff[j])
        var cnt = Int(toff[j + 1]) - lo
        var node = 0
        var steps = 0
        var ok = cnt >= 1
        while ok:
            var l = Int(left[lo + node])
            if l == -1:
                break
            var cc = Int(colid[lo + node])
            if cc < 0 or cc >= dd or l < 1 or l + 1 >= cnt or steps > cnt:
                ok = False
                break
            if x[i * dd + cc] <= quesval[lo + node]:
                node = l
            else:
                node = l + 1
            steps += 1
        if ok:
            acc = sf64_add(acc, sf64_mul(coef[j], sf64_from_f32(values[lo + node])))
        else:
            bad[0] = Int32(1)
        j += kk
    dst[e] = acc
