# SPDX-License-Identifier: Apache-2.0
# Lane ag-pass6-fold (2026-10-19, PASS6_IDEAS idea D(2); identical): the loss
# terms folded inside the blocked fold's leaf, and the loss's one-launch VJP.
# Element bodies shared by HostExec and DeviceExec through
# sequence/dispatch.mojo (codes OP_AGS_FIRST..OP_AGS_LAST, 2300..).
#
# BEFORE (autograd/losses.mojo mse / mae / huber / bce and loss_value_into):
#   forward  OP_LS_ELEM_FWD writes a rows x cols terms buffer (st), then the
#            blocked fold reads it back (OP_AG_REDUCE_PART, ld) and folds:
#            3 launches above 64 terms (2 at most), 2 x the matrix of extra
#            traffic (the terms written and read);
#   backward OP_AG_REDUCE_VJP stores g = mul(ld(upstream, 0), factor) at
#            every element, then OP_LS_ELEM_VJP reads it back: 2 launches.
# AFTER (AUTOGRAD_LOSS_REDUCED, default on):
#   forward  OP_AGS_LOSS_PART computes each term in the leaf chain,
#            ftz(loss_term(ld p, ld y)) (the word st then ld gives back),
#            and adds the terms in the leaf's ascending order from +0.0
#            (eight terms computed ahead of their adds, _fold_chain's
#            order); the leaf's partial or finished output is
#            OP_AG_REDUCE_PART's (ag_part_finish), the tree levels and the
#            last level are the fold's own (fold.mojo blocked_fold_finish):
#            2 launches above 64 terms, 1 at most;
#   backward OP_AGS_LOSS_VJP forms g = mul(ld(upstream, 0), factor) in a
#            register (REDUCE_VJP's stored word: mul flushes, ld of a
#            flushed word is that word) and applies loss_term_dp / dy:
#            1 launch.
# Same words, same order: the loss value and both gradients are bit-equal to
# loss_terms followed by mean / sum, on every executor.
from sequence.ops import Args, FP, ld, st, add, mul
from checks.numerics import ftz
from autograd.ops import AXIS_ALL, AG_FOLD_LOADS, ag_part_finish
from autograd.loss_ops import loss_term, loss_term_dp, loss_term_dy

comptime OP_AGS_LOSS_PART = 2300
comptime OP_AGS_LOSS_VJP = 2301
comptime OP_AGS_FIRST = OP_AGS_LOSS_PART
comptime OP_AGS_LAST = OP_AGS_LOSS_VJP


@always_inline
def ags_loss_term(a: Args, i: Int) -> Float32:
    """Term i of OP_AGS_LOSS_PART: ftz(loss_term(kind i6, p0[i], p4[i],
    delta f2)), the word OP_LS_ELEM_FWD stores and the fold loads back."""
    return ftz(loss_term(a.i6, ld(a.p0, i), ld(a.p4, i), a.f2))


@always_inline
def _loss_chain(a: Args, base: Int, count: Int) -> Float32:
    """_fold_chain (autograd/ops.mojo) over the terms base .. base + count -
    1, each computed in place: eight terms computed before their adds, the
    adds ascending from +0.0."""
    var s = Float32(0)
    var k = 0
    while k + AG_FOLD_LOADS <= count:
        var i = base + k
        var v0 = ags_loss_term(a, i)
        var v1 = ags_loss_term(a, i + 1)
        var v2 = ags_loss_term(a, i + 2)
        var v3 = ags_loss_term(a, i + 3)
        var v4 = ags_loss_term(a, i + 4)
        var v5 = ags_loss_term(a, i + 5)
        var v6 = ags_loss_term(a, i + 6)
        var v7 = ags_loss_term(a, i + 7)
        s = add(s, v0)
        s = add(s, v1)
        s = add(s, v2)
        s = add(s, v3)
        s = add(s, v4)
        s = add(s, v5)
        s = add(s, v6)
        s = add(s, v7)
        k += AG_FOLD_LOADS
    while k < count:
        s = add(s, ags_loss_term(a, base + k))
        k += 1
    return s


@always_inline
def op_ags_loss_part(t: Int, a: Args):
    """The loss fold's leaf launch, AXIS_ALL over the i1 x i2 terms. The
    slots of OP_AG_REDUCE_PART (v2) with the source replaced by the terms:
    p0 prediction, p4 target, i6 loss kind, f2 delta (Huber); p1 partials,
    p2 the finished output when i4 = 1 (/ f0 when i5 = 1, else * f0), i3
    the leaf length (AG_FOLD_LEAF), i7 / p3 / f1 the curve word
    (ag_part_finish). Thread t = leaf t (terms t * i3 .. ascending)."""
    var lo = t * a.i3
    ag_part_finish(t, a, _loss_chain(a, lo, min(a.i3, a.i1 * a.i2 - lo)))


@always_inline
def op_ags_loss_vjp(t: Int, a: Args):
    """p0 prediction, p1 target, p2 the loss's upstream gradient (one word),
    p3 destination. i0 kind, i8 operand (0 prediction, 1 target), f0 delta,
    f1 the fold factor. Thread t = element t."""
    var g = mul(ld(a.p2, 0), a.f1)
    var p = ld(a.p0, t)
    var y = ld(a.p1, t)
    if a.i8 == 0:
        st(a.p3, t, loss_term_dp(a.i0, p, y, a.f0, g))
    else:
        st(a.p3, t, loss_term_dy(a.i0, p, y, a.f0, g))


@always_inline
def apply_fold_ops[OP: Int](t: Int, a: Args):
    """The OP_AGS_FIRST..OP_AGS_LAST family (sequence/dispatch.mojo)."""
    comptime if OP == OP_AGS_LOSS_PART:
        op_ags_loss_part(t, a)
    elif OP == OP_AGS_LOSS_VJP:
        op_ags_loss_vjp(t, a)
