# SPDX-License-Identifier: Apache-2.0
# HIGHER-ORDER GRADIENTS: THE NEW NODE KINDS AND ELEMENT BODIES (lane
# ag-ho-core, docs/plans/AUTOGRAD_HIGHER_ORDER.md sections 1.2-1.3).
#
# backward_recorded (autograd/higher.mojo) writes every VJP as recorded tape
# operations. Two of those operations are new node kinds whose own VJPs
# carry gradient, so the first-order Tape._backward knows them too:
#   EXPAND (32): out[r, c] = g[reduced index] * f0, the forward of a sum or
#     mean's VJP (the existing OP_AG_REDUCE_VJP body); its VJP is the
#     blocked fold Tape._fold over the same axis with factor f0.
#   CLAMP (33): out = min(max(x, -f0), f0) (OP_HO_CLAMP_FWD); its VJP is g
#     where |x| < f0, else +0.0 (OP_HO_CLAMP_VJP).
# The constants (indicator, sign, inside) are element bodies recorded as
# non-differentiable leaves: relu'', |x|'' and the clip's boundary are zero.
#
# Every body is one thread per element with no reduction: a compare, a
# select or min/max, ld/st with ftz. So the words are the same on the host,
# Apple, NVIDIA and AMD by construction. Metal: typed FP slots, slots p0..p3
# only, no pointer select, @always_inline.
#
# Codes OP_HO_FIRST..OP_HO_LAST (3600..3609) are this family;
# sequence/dispatch.mojo routes and fences them. This module imports no
# tape code (tape.mojo imports it for EXPAND, CLAMP and OP_HO_CLAMP_VJP).
from sequence.ops import Args, ld, st
from autograd.ops import (
    CUSTOM, LINEAR, LOSS_ELEM, LOSS_CE, LOSS_SCE, LOSS_REDUCED,
)
from autograd.data_ops import DROPOUT
from autograd.embedding_ops import EMBEDDING

# Node kinds (free below CUSTOM in autograd/ops.mojo).
comptime EXPAND = 32
comptime CLAMP = 33

comptime OP_HO_INDICATOR = 3600
comptime OP_HO_SIGN = 3601
comptime OP_HO_CLAMP_FWD = 3602
comptime OP_HO_CLAMP_VJP = 3603
comptime OP_HO_INSIDE = 3604
comptime OP_HO_FIRST = 3600
comptime OP_HO_LAST = 3609


@always_inline
def _abs(x: Float32) -> Float32:
    return x if x >= 0 else -x


# ---------------------------------------------------------------- bodies
@always_inline
def op_ho_indicator(t: Int, a: Args):
    """p0 x, p1 dst: 1 where x > 0, else 0 (relu's derivative)."""
    var x = ld(a.p0, t)
    st(a.p1, t, Float32(1) if x > 0 else Float32(0))


@always_inline
def op_ho_sign(t: Int, a: Args):
    """p0 x, p1 dst: 1, -1 or 0 (sign(0) = 0, the MAE VJP's convention,
    autograd/loss_ops.mojo _sign)."""
    var x = ld(a.p0, t)
    var s = Float32(0)
    if x > 0:
        s = Float32(1)
    elif x < 0:
        s = Float32(-1)
    st(a.p1, t, s)


@always_inline
def op_ho_clamp_fwd(t: Int, a: Args):
    """p0 x, p1 dst, f0 bound (> 0): min(max(x, -f0), f0)."""
    var x = ld(a.p0, t)
    st(a.p1, t, min(max(x, -a.f0), a.f0))


@always_inline
def op_ho_clamp_vjp(t: Int, a: Args):
    """p0 x (the clamp's input), p2 upstream, p3 dst, f0 bound: the upstream
    word where |x| < f0, else +0.0. p1 is unbound."""
    var x = ld(a.p0, t)
    if _abs(x) < a.f0:
        st(a.p3, t, ld(a.p2, t))
    else:
        st(a.p3, t, Float32(0))


@always_inline
def op_ho_inside(t: Int, a: Args):
    """p0 x, p1 dst, f0 bound: 1 where |x| < f0, else 0."""
    var x = ld(a.p0, t)
    st(a.p1, t, Float32(1) if _abs(x) < a.f0 else Float32(0))


@always_inline
def apply_higher[OP: Int](t: Int, a: Args):
    comptime if OP == OP_HO_INDICATOR:
        op_ho_indicator(t, a)
    elif OP == OP_HO_SIGN:
        op_ho_sign(t, a)
    elif OP == OP_HO_CLAMP_FWD:
        op_ho_clamp_fwd(t, a)
    elif OP == OP_HO_CLAMP_VJP:
        op_ho_clamp_vjp(t, a)
    elif OP == OP_HO_INSIDE:
        op_ho_inside(t, a)


# ---------------------------------------------------------------- refusals
def higher_family(op: Int) -> String:
    """The user-facing name of a node kind that has a first-order rule only."""
    if op == LINEAR:
        return "linear"
    if op == LOSS_ELEM:
        return "loss_terms"
    if op == LOSS_REDUCED:
        return "loss_reduced"
    if op == LOSS_CE:
        return "cross_entropy_rows"
    if op == LOSS_SCE:
        return "softmax_cross_entropy"
    if op == DROPOUT:
        return "dropout"
    if op == EMBEDDING:
        return "embedding"
    if op < CUSTOM:
        return "operation kind " + String(op)
    var tag = op - CUSTOM
    if tag == 1:
        return "layer_norm"
    if tag >= 16 and tag <= 18:
        return "recurrent"
    if tag >= 32 and tag <= 38:
        return "transformer"
    if tag >= 48 and tag <= 63:
        return "mamba"
    if tag >= 64 and tag <= 67:
        return "language_model"
    if tag >= 72 and tag <= 87:
        return "mamba3"
    if tag >= 88 and tag <= 95:
        return "samba"
    if tag >= 96 and tag <= 98:
        return "cnn"
    if tag == 120 or tag == 121:
        return "small_mlp"
    return "custom"


def higher_refusal(op: Int) -> String:
    """The create_graph refusal for node kind `op` (section 2 of the plan):
    one error naming the family and its tag (the custom tag, or the node
    kind for a fused tape operation)."""
    var tag = op - CUSTOM if op >= CUSTOM else op
    return ("autograd: create_graph cannot differentiate through " + higher_family(op) + " (tag "
            + String(tag) + "): it is a fused kernel with a first-order rule only; record the block with"
            + " tape operations, or take the first-order gradient")
