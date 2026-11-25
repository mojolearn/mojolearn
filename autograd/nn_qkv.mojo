# SPDX-License-Identifier: Apache-2.0
"""Attention over explicit q, k and v: `Session.attention` (lane nn-blocks-qkv, 2026-11-25).

PyTorch's `F.scaled_dot_product_attention(q, k, v, attn_mask, is_causal=..., scale=..., enable_gqa=True)`:
q (B, H, Lq, d), k (B, Hkv, Lk, d), v (B, Hkv, Lk, dv) (or the three without the heads axis, (B, L, ...)),
out (B, H, Lq, dv) = softmax(scale * q k^T + mask) v, query head h reading key/value head h // (H / Hkv).

One body per operation, every piece an existing one:
  - the two products are the rank-N batched matmul (autograd/rankn_bmm.mojo rankn_matmul, OP_RN_BMM), k^T a
    permute view read in place through its strides (no copy); grouped heads are the same product over a
    (B, Hkv, G, ...) view of q against a (B, Hkv, 1, ...) view of k and v, the size-1 axis broadcast by the
    product itself (no repeated k or v is ever written; their gradients sum the group in the product's fixed
    chain);
  - the masked softmax is one row launch (autograd/nn_qkv_ops.mojo OP_QKV_SOFTMAX), whose row softmax is the
    tape softmax's body; its node is recorded with QKV_TAG_SOFTMAX, whose rule is the classic attention
    layer's softmax VJP times the scale (autograd/nn_qkv_rules.mojo QkvRules, reached from ImportRules).
Launches: forward 3 (scores product, masked softmax, context product; grouped heads with a q that is itself a
pending view: one copy more). Backward: the softmax VJP, two product VJPs per product (the operands that want
a gradient), and k's permute view handing its gradient back.

The mask is read in place through broadcast strides (a stride 0 along an axis of length 1), never expanded,
unless it is a float mask that takes a gradient: then it is expanded once (imp_expand) and its gradient is
dL/d(scaled scores + mask), summed back over the broadcast axes by the expand's rule (QKV_TAG_SOFTMAX_MASK,
autograd/nn_qkv_rules.mojo). A boolean (ids) mask never takes one. Bits: the row softmax's fold is fixed (ascending, from 0); the products' chains are
OP_RN_BMM's; the host and every vendor run the same functions.
"""
from sequence.exec_trait import Exec
from sequence.ops import Args
from checks.numerics import ftz, identical_div, identical_sqrt
from autograd.tape import Tape, Tensor
from std.collections import Array
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.rankn_api import rn_shape
from autograd.rankn_bmm import rankn_matmul
from autograd.nn_qkv_rules import QKV_TAG_SOFTMAX, QKV_TAG_SOFTMAX_MASK
from autograd.import_api import imp_expand
from autograd.nn_qkv_ops import OP_QKV_SOFTMAX, QKV_CAUSAL, QKV_MASK_BOOL, QKV_MASK_ADD

#: the Session operation code (autograd/ops_api.mojo; python/mojolearn/ops.py `attention`), of lane
#: nn-blocks-qkv's range 122..125
comptime OPS_ATTENTION_QKV = 122


def _dims_text(s: ShapeN) -> String:
    var out = String("(")
    for k in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): message text, no data
        if k > 0:
            out += ", "
        out += String(s.d[k])
    return out + ")"


def _sh(dims: List[Int]) raises -> ShapeN:
    return rn_shape(dims, "attention")


def _mask_strides(ms: ShapeN, mst: Array[Int, RN_MAX_RANK], target: List[Int], expected: String) raises -> List[Int]:
    """The mask's element strides along the scores' axes (right-aligned; 0 along a broadcast axis), or the
    refusal of a mask that does not broadcast."""
    var nd = len(target)
    if ms.ndim > nd:
        raise Error("ValueError: attention: the mask of shape " + _dims_text(ms) + " has more axes than the "
                    + "scores " + expected)
    var st = List[Int](length=nd, fill=0)
    for u in range(nd):  # small-loop(nd: 3 or 4 axes): mask metadata, no data
        var mu = u - (nd - ms.ndim)
        if mu < 0:
            continue
        var e = ms.d[mu]
        if e == target[u]:
            st[u] = mst[mu] if e > 1 else 0
        elif e != 1:
            raise Error("ValueError: attention: the mask of shape " + _dims_text(ms) + " does not broadcast to "
                        + "the scores " + expected)
    return st^


def qkv_attention[E: Exec](
    mut tape: Tape, mut ex: E, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor], mask_bool: Bool,
    causal: Bool, scale_given: Bool, scale_in: Float32,
) raises -> Tensor:
    """softmax(scale * q k^T + mask) v on the tape; `mask_bool` reads the mask as ids words (nonzero attends),
    otherwise as float32 values added to the scaled scores. Refuses, in words, shapes that do not fit."""
    var qs = tape.shape_of(q)
    var ks = tape.shape_of(k)
    var vs = tape.shape_of(v)
    var nd = qs.ndim
    if (nd != 3 and nd != 4) or ks.ndim != nd or vs.ndim != nd:
        raise Error("ValueError: attention: q, k and v must all be (batch, heads, length, features) or all "
                    + "(batch, length, features); got q " + _dims_text(qs) + ", k " + _dims_text(ks) + ", v "
                    + _dims_text(vs))
    var B = qs.d[0]
    var H = qs.d[1] if nd == 4 else 1
    var Hkv = ks.d[1] if nd == 4 else 1
    var Lq = qs.d[nd - 2]
    var d = qs.d[nd - 1]
    var Lk = ks.d[nd - 2]
    var dv = vs.d[nd - 1]
    if ks.d[0] != B or vs.d[0] != B:
        raise Error("ValueError: attention: q, k and v must have the same batch size; got q " + _dims_text(qs)
                    + ", k " + _dims_text(ks) + ", v " + _dims_text(vs))
    if ks.d[nd - 1] != d:
        raise Error("ValueError: attention: q and k must have the same feature size; q has " + String(d)
                    + " and k has " + String(ks.d[nd - 1]))
    if vs.d[nd - 2] != Lk or (nd == 4 and vs.d[1] != Hkv):
        raise Error("ValueError: attention: k and v must have the same heads and length; got k " + _dims_text(ks)
                    + " and v " + _dims_text(vs))
    if H % Hkv != 0:
        raise Error("ValueError: attention: the query heads (" + String(H) + ") must be a multiple of the "
                    + "key/value heads (" + String(Hkv) + ")")
    var has_mask = Bool(mask)
    if causal and has_mask:
        raise Error("ValueError: attention: causal=True cannot be combined with an explicit mask (PyTorch refuses "
                    + "it too); put the causal pattern into the mask")
    var scale = scale_in
    if scale_given:
        if not (scale_in - scale_in == Float32(0)):
            raise Error("ValueError: attention: scale must be a finite number")
    else:
        scale = ftz(identical_div(Float32(1), identical_sqrt(Float32(d))))
    var G = H // Hkv
    var a = Args()
    var flags = QKV_CAUSAL if causal else 0
    var mask_full = Optional[Tensor](None)
    if has_mask:
        # The mask broadcasts to the scores (batch, heads, Lq, Lk) (without the heads axis for 3-D inputs),
        # aligned on the right as NumPy and PyTorch align it; it is read in place through its strides.
        var mo = tape.operand_nd(ex, mask.value())
        var ms = mo[2]
        var mst = mo[3].copy()
        var target = List[Int](capacity=nd)
        target.append(B)
        if nd == 4:
            target.append(H)
        target.append(Lq)
        target.append(Lk)
        var expected = String("(") + String(B) + (", " + String(H) if nd == 4 else String("")) + ", " + String(Lq)
        expected += ", " + String(Lk) + ")"
        var st = _mask_strides(ms, mst, target, expected)
        if not mask_bool and tape.requires_grad(mask.value()):
            # A float mask that takes a gradient (PyTorch differentiates it): it is expanded once to the scores'
            # shape (imp_expand, whose rule sums the gradient back over the broadcast axes) and becomes the
            # node's second input; the row body reads the same values, so the words are those of the broadcast
            # read. Only this case pays the expand launch and the second VJP launch.
            var full = imp_expand(tape, ex, mask.value(), target)
            mo = tape.operand_nd(ex, full)
            ms = mo[2]
            mst = mo[3].copy()
            st = _mask_strides(ms, mst, target, expected)
            mask_full = Optional[Tensor](full)
        a.p2 = mo[0]
        a.i4 = mo[1]
        a.i5 = st[0]
        a.i6 = st[1] if nd == 4 else 0
        a.i7 = st[nd - 2]
        a.i8 = st[nd - 1]
        flags |= QKV_MASK_BOOL if mask_bool else QKV_MASK_ADD
    var scores: Tensor
    if G == 1:
        var axes = List[Int](capacity=nd)
        for u in range(nd - 2):  # small-loop(nd: 3 or 4 axes): view metadata, no data
            axes.append(u)
        axes.append(nd - 1)
        axes.append(nd - 2)
        var kt1 = tape.permute(ex, k, axes)
        scores = rankn_matmul(tape, ex, q, kt1)
    else:
        # grouped heads: q (B, Hkv, G, Lq, d) against k (B, Hkv, 1, d, Lk), the product broadcasting the group
        var q5: List[Int] = [B, Hkv, G, Lq, d]
        var k5: List[Int] = [B, Hkv, 1, Lk, d]
        var kt: List[Int] = [0, 1, 2, 4, 3]
        var kflat = tape.view_nd(ex, k, _sh(k5))
        var kview = tape.permute(ex, kflat, kt)
        var qview = tape.view_nd(ex, q, _sh(q5))
        scores = rankn_matmul(tape, ex, qview, kview)
    var rows = B * H * Lq
    a.p0 = tape.value_of(ex, scores)
    var output = tape.reserve(ex, rows * Lk)
    a.p1 = output
    a.i0 = Lk
    a.i1 = Lq
    a.i2 = H
    a.i3 = flags
    a.f0 = scale
    ex.launch[OP_QKV_SOFTMAX](a, rows)
    # the rules (autograd/nn_qkv_rules.mojo QkvRules, in ImportRules) read saved.i0 (Lk) and saved.f0 (the scale)
    var probs: Tensor
    if mask_full:
        probs = tape.custom_nd(ex, [scores, mask_full.value()], output, tape.shape_of(scores), QKV_TAG_SOFTMAX_MASK, a)
    else:
        probs = tape.custom_nd(ex, [scores], output, tape.shape_of(scores), QKV_TAG_SOFTMAX, a)
    if G == 1:
        return rankn_matmul(tape, ex, probs, v)
    var v5: List[Int] = [B, Hkv, 1, Lk, dv]
    var vview = tape.view_nd(ex, v, _sh(v5))
    var ctx = rankn_matmul(tape, ex, probs, vview)   # (B, Hkv, G, Lq, dv)
    var o4: List[Int] = [B, H, Lq, dv]
    return tape.view_nd(ex, ctx, _sh(o4))
