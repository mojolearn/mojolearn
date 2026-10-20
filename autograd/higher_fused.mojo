# SPDX-License-Identifier: Apache-2.0
# RECORDED VJPS OF THE FUSED NODE KINDS (lane ag-ho-fused, H2 of
# docs/plans/AUTOGRAD_HIGHER_ORDER.md section 1.2). backward_recorded
# (autograd/higher.mojo, lane H1) calls lower_fused_vjp for every active
# LINEAR, LOSS_ELEM, LOSS_REDUCED, LOSS_CE, LOSS_SCE, DROPOUT and EMBEDDING
# node and each edge whose parent requires a gradient. The result is a
# RECORDED tensor at the parent's shape: the VJP expressed through the
# tape's own operations (matmul, transpose, multiply, sum, softmax,
# dropout, ...) so the gradient has a graph and a later backward
# differentiates it. The first-order fused kernels (_linear_vjp's GEMM
# epilogues, OP_LS_ELEM_VJP, OP_AGS_LOSS_VJP, OP_LS_CE_VJP, OP_P1_SCE_VJP,
# OP_AG_DROPOUT_VJP) are untouched: create_graph=False never reaches here.
#
# Identity: every value below is a tape operation whose kernel is already
# in the cross-vendor digest suites, the one-hot constant
# (autograd/higher_loss_ops.mojo) and H1's constants and node kinds
# (indicator, sign, clamp, expand: elementwise, no arithmetic beyond a
# compare or one multiply); every sum is Tape.sum, the blocked fold. No
# vendor branch, no size rule, no host loop over rows or elements: the
# function is a host driver over one node's edge, and every launch is
# parallel over elements (or rows, for the softmax rows the forward also
# runs). So the recorded words are the same on Apple, NVIDIA, AMD and the
# host for one recorded graph.
#
# The operand order of each lowering follows the fused kernel's
# arithmetic (loss_term_dp: mul(2, mul(g, d)); mul(g, sign(d)); mul(g,
# sigmoid(p) - y); the CE rows: mul(softmax - onehot, g)), so the recorded
# first-order values equal the fused path's wherever the same float32
# operations run (section 1.4 lists where they may not: the softmax rows,
# the GEMM epilogue's fused + l2 W).
#
# The four H1 helpers this file calls, from autograd.higher (section 1.2):
#   indicator[E](mut tape, mut ex, x) -> Tensor        1.0 where x > 0, a constant
#   sign[E](mut tape, mut ex, x) -> Tensor             -1, 0, 1, a constant
#   clamp[E](mut tape, mut ex, x, bound: Float32) -> Tensor        node kind CLAMP
#   expand[E](mut tape, mut ex, g, rows, cols, axis, factor: Float32) -> Tensor
#                                                       node kind EXPAND (REDUCE's VJP)
#
# Every tape call is its own statement: one mutable borrow of the tape and
# the executor at a time.
from sequence.exec_trait import Exec
from sequence.mlp import ACT_IDENTITY, ACT_LOGISTIC, ACT_TANH, ACT_RELU, ACT_SOFTMAX
from autograd.tape import Tape, Tensor
from autograd.ops import (
    LINEAR, LOSS_ELEM, LOSS_REDUCED, LOSS_CE, LOSS_SCE, AXIS_ALL, AXIS_ROWS, AXIS_COLS,
)
from autograd.loss_ops import LS_MSE, LS_MAE, LS_HUBER, LS_BCE_LOGITS
from autograd.data_ops import DROPOUT
from autograd.embedding_ops import EMBEDDING
from autograd.higher_loss_ops import onehot_into
from autograd.higher import indicator, sign, clamp, expand


def _handle(tape: Tape, i: Int) -> Tensor:
    """The handle of node i of the tape's current graph."""
    return Tensor(Int(tape._identity.value()), tape._generation, i)


def _parent(tape: Tape, i: Int, edge: Int) -> Tensor:
    """The handle of node i's input `edge`."""
    var n = tape._nodes[i]
    return _handle(tape, tape._parents[n.start + edge])


def _fused_refusal(family: String, kind: Int) -> String:
    # section 2's one error per family
    return (
        "autograd: create_graph cannot differentiate through " + family + " (kind " + String(kind)
        + "): it is a fused kernel with a first-order rule only; record the block with tape operations,"
        + " or take the first-order gradient"
    )


def onehot[E: Exec](mut tape: Tape, mut ex: E, codes: Tensor, cols: Int) raises -> Tensor:
    """The (rows, cols) one-hot matrix of class codes (rows, 1), recorded as
    a constant (no graph): one launch into tape storage, then a leaf that
    requires no gradient, as detach records a constant."""
    var ci = tape._use(ex, codes)
    var c = tape._nodes[ci]
    if c.cols != 1:
        raise Error("onehot: codes must have shape (rows, 1)")
    var dst = tape._alloc(ex, c.rows * cols)
    onehot_into(ex, c.value, dst, c.rows, cols)
    return tape.variable(ex, dst, c.rows, cols, False)


# ---------------------------------------------------------------- LINEAR
def _linear_dz[E: Exec](mut tape: Tape, mut ex: E, i: Int, g: Tensor) raises -> Tensor:
    """dL/dZ (Z the pre-activation) of linear node i from its output
    gradient g, read from the node's own output z = act(Z): z carries the
    graph, so a later backward differentiates the activation derivative
    through the linear node itself."""
    var n = tape._nodes[i]
    var act = n.saved.i0
    if act == ACT_IDENTITY:
        return g
    var z = _handle(tape, i)
    if act == ACT_RELU:
        # z > 0 exactly where Z > 0; the indicator is a constant (relu'' = 0)
        var on = indicator(tape, ex, z)
        return tape.multiply(ex, g, on)
    if act == ACT_LOGISTIC:
        # the SIGMOID row: (g z)(1 - z)
        var gz = tape.multiply(ex, g, z)
        var one = tape.scalar(ex, Float32(1))
        var rest = tape.subtract(ex, one, z)
        return tape.multiply(ex, gz, rest)
    if act == ACT_TANH:
        # the TANH row: g (1 - z z)
        var zz = tape.multiply(ex, z, z)
        var one = tape.scalar(ex, Float32(1))
        var rest = tape.subtract(ex, one, zz)
        return tape.multiply(ex, g, rest)
    if act == ACT_SOFTMAX:
        # the SOFTMAX row: z (g - rowsum(g z)); the row sum is the blocked fold
        var gz = tape.multiply(ex, g, z)
        var dot = tape.sum(ex, gz, AXIS_COLS)
        var centered = tape.subtract(ex, g, dot)
        return tape.multiply(ex, z, centered)
    raise Error("autograd: linear node with an unknown activation code " + String(act))


def _linear_recorded[E: Exec](mut tape: Tape, mut ex: E, i: Int, g: Tensor, edge: Int) raises -> Tensor:
    """Edge `edge` of linear node i (0 the input x, 1 the weights W, 2 the
    bias b) as recorded operations, with the saved slots _linear_vjp reads:
    i0 the activation code, i1 bias present, f0 l2, f1 divisor.
        dX = dZ W^T
        dW = x^T dZ, then (dW + l2 W) / divisor when divisor > 0
        db = column sums of dZ, then / divisor when divisor > 0
    (the first-order EPI_L2GRAD epilogue and OP_COLSUM_DIV divide, so these
    divide by a constant scalar too). dZ is recomputed for each edge (one
    to four launches): the recorded copies hold the same words, and their
    contributions to a second derivative add up to the one dZ's."""
    var n = tape._nodes[i]
    if edge < 0 or edge >= n.count:
        raise Error("autograd: linear node has no input " + String(edge))
    var l2 = n.saved.f0
    var divisor = n.saved.f1
    var dz = _linear_dz(tape, ex, i, g)
    if edge == 0:
        var w = _parent(tape, i, 1)
        var wt = tape.transpose(ex, w)
        return tape.matmul(ex, dz, wt)
    if edge == 1:
        var x = _parent(tape, i, 0)
        var xt = tape.transpose(ex, x)
        var dw = tape.matmul(ex, xt, dz)
        if divisor > 0:
            if l2 != 0:
                var w = _parent(tape, i, 1)
                var penalty = tape.scale(ex, w, l2)
                dw = tape.add(ex, dw, penalty)
            var d = tape.scalar(ex, divisor)
            dw = tape.divide(ex, dw, d)
        return dw
    var db = tape.sum(ex, dz, AXIS_ROWS)
    if divisor > 0:
        var d = tape.scalar(ex, divisor)
        db = tape.divide(ex, db, d)
    return db


# ---------------------------------------------------------------- losses
def _loss_terms_recorded[E: Exec](
    mut tape: Tape, mut ex: E, kind: Int, delta: Float32, p: Tensor, y: Tensor, g: Tensor, edge: Int,
) raises -> Tensor:
    """g times the derivative of loss_term(kind, p, y, delta) in p (edge 0)
    or y (edge 1), elementwise, as recorded operations (loss_term_dp and
    loss_term_dy of autograd/loss_ops.mojo, operand for operand):
        MSE    2 (g d)              d = p - y; edge 1 negated
        MAE    g sign(d)            sign a constant (zero curvature); edge 1 negated
        HUBER  g clamp(d, delta)    d inside the threshold, else delta sign(d); edge 1 negated
        BCE    g (sigmoid(p) - y)   edge 1: -(g p)"""
    if kind == LS_BCE_LOGITS:
        if edge == 0:
            var s = tape.sigmoid(ex, p)
            var r = tape.subtract(ex, s, y)
            return tape.multiply(ex, g, r)
        var gp = tape.multiply(ex, g, p)
        return tape.scale(ex, gp, Float32(-1))
    var d = tape.subtract(ex, p, y)
    var c: Tensor
    if kind == LS_MSE:
        var gd = tape.multiply(ex, g, d)
        c = tape.scale(ex, gd, Float32(2))
    elif kind == LS_MAE:
        var s = sign(tape, ex, d)
        c = tape.multiply(ex, g, s)
    elif kind == LS_HUBER:
        var q = clamp(tape, ex, d, delta)
        c = tape.multiply(ex, g, q)
    else:
        raise Error("autograd: loss node with an unknown loss kind " + String(kind))
    if edge == 0:
        return c
    return tape.scale(ex, c, Float32(-1))


def _cross_entropy_recorded[E: Exec](mut tape: Tape, mut ex: E, i: Int, g: Tensor) raises -> Tensor:
    """(softmax(logits) - onehot(codes)) * g for a LOSS_CE (g the (rows, 1)
    row gradients, a column broadcast) or LOSS_SCE (g the scalar gradient
    times the saved mean factor f0) node: op_ls_ce_vjp's and
    op_p1_sce_vjp's (p - [c == y]) * g. softmax is the recorded node, so a
    later backward gives the per-row diag(p) - p p^T curvature."""
    var n = tape._nodes[i]
    var logits = _parent(tape, i, 0)
    var codes = _parent(tape, i, 1)
    var cols = tape._nodes[tape._parents[n.start]].cols
    var p = tape.softmax(ex, logits)
    var target = onehot(tape, ex, codes, cols)
    var diff = tape.subtract(ex, p, target)
    if n.op == LOSS_SCE:
        var gs = tape.scale(ex, g, n.saved.f0)
        return tape.multiply(ex, diff, gs)
    return tape.multiply(ex, diff, g)


# ---------------------------------------------------------------- dropout
def _dropout_recorded[E: Exec](mut tape: Tape, mut ex: E, i: Int, g: Tensor) raises -> Tensor:
    """The VJP of dropout is dropout's own forward at the same (seed,
    counter, p): mask * s * g, the mask re-derived from the saved slots
    (dropout_args: i0..i3 the seed in 16-bit words, i4, i5 the counter,
    f0 p), never stored. Tape.dropout recomputes s = 1 / (1 - p) from the
    same p, so s is the forward's word and every element is
    OP_AG_DROPOUT_VJP's mul(g, s) or +0.0."""
    var s = tape._nodes[i].saved
    var seed = (UInt64(s.i0) | (UInt64(s.i1) << UInt64(16)) | (UInt64(s.i2) << UInt64(32))
                | (UInt64(s.i3) << UInt64(48)))
    var counter = s.i4 | (s.i5 << 16)
    return tape.dropout(ex, g, s.f0, seed, counter)


# ---------------------------------------------------------------- entry
def lower_fused_vjp[E: Exec](mut tape: Tape, mut ex: E, i: Int, upstream: Tensor, edge: Int) raises -> Tensor:
    """The recorded VJP of node i's input `edge` from the recorded upstream
    gradient (node i's shape), for the fused node kinds: LINEAR, LOSS_ELEM,
    LOSS_REDUCED, LOSS_CE, LOSS_SCE and DROPOUT. EMBEDDING refuses (its VJP
    is the run-sorted scatter-add, which no recorded operation expresses
    yet). The result has the input's shape and requires a gradient iff its
    operations touched a tensor that does."""
    var n = tape._nodes[i]
    if n.op == LINEAR:
        return _linear_recorded(tape, ex, i, upstream, edge)
    if n.op == LOSS_ELEM or n.op == LOSS_REDUCED:
        if edge != 0 and edge != 1:
            raise Error("autograd: loss node has no input " + String(edge))
        var p = _parent(tape, i, 0)
        var y = _parent(tape, i, 1)
        var g = upstream
        if n.op == LOSS_REDUCED:
            # the (1, 1) gradient times the saved mean factor at every
            # element: REDUCE's VJP (OP_AGS_LOSS_VJP's register word)
            var pn = tape._nodes[tape._parents[n.start]]
            g = expand(tape, ex, upstream, pn.rows, pn.cols, AXIS_ALL, n.saved.f1)
        return _loss_terms_recorded(tape, ex, n.saved.i0, n.saved.f0, p, y, g, edge)
    if n.op == LOSS_CE or n.op == LOSS_SCE:
        if edge != 0:
            raise Error("autograd: cross-entropy codes take no gradient")
        return _cross_entropy_recorded(tape, ex, i, upstream)
    if n.op == DROPOUT:
        if edge != 0:
            raise Error("autograd: dropout node has no input " + String(edge))
        return _dropout_recorded(tape, ex, i, upstream)
    if n.op == EMBEDDING:
        raise Error(_fused_refusal("embedding", EMBEDDING))
    raise Error("autograd: no recorded backward rule for operation kind " + String(n.op))
