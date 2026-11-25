# SPDX-License-Identifier: Apache-2.0
"""Element body of attention over explicit q, k and v (lane nn-blocks-qkv, 2026-11-25), read by both
executors through `sequence/dispatch.mojo::apply` (HostExec: a host loop over the rows; DeviceExec: one GPU
thread per row), so the host and the device column run the same function on the same words.

  op               cells            body
  OP_QKV_SOFTMAX   B H Lq rows      scale, the causal mask from the indices, the caller's boolean or additive
                                    mask read in place through broadcast strides, then the row softmax
                                    (core/row_cells.mojo softmax_row_from_max_f32, the tape softmax's body);
                                    a row with no position left is all zeros

The backward is the classic attention layer's (autograd/nn_attention.mojo NN_TAG_SOFTMAX, its rule launches
autograd/nn_attention_ops.mojo OP_NN_ATTN_SOFTMAX_VJP: the tape softmax's VJP times the scale): a masked word
and every word of an all-masked row hold y = 0, so their gradient is 0. No second VJP body exists.

Why a body of its own: nn_attn_softmax_row (6700) masks from indices and per-row lengths only and starts its
maximum at -1e30, so a row of additive masks at float32's lowest finite value would divide 0 by 0; this one reads
a mask tensor, starts at -inf, writes a masked word as -inf (exp gives +0.0) and zeros a row with nothing left.
The transformer block's attention kernels (attention_block / attention_step) fuse rotary positions and the
key/value cache into the product and take no explicit k, v or mask: they cannot serve explicit q, k, v.

Fold order: per row, the scaled scores left to right, the maximum left to right, the exponentials summed
ascending from 0 (softmax_row_from_max_f32): one order on the host and every vendor.
"""
from std.memory import bitcast
from sequence.ops import Args
from checks.numerics import ftz, identical_mul
from core.row_cells import softmax_row_from_max_f32

comptime OP_QKV_FIRST = 6840
comptime OP_QKV_SOFTMAX = 6840
comptime OP_QKV_LAST = 6859

#: the row's flags (a.i3)
comptime QKV_CAUSAL = 1
#: the mask words are int32 ids words: a word attends when it is nonzero (PyTorch's boolean mask, True attends)
comptime QKV_MASK_BOOL = 2
#: the mask words are float32 values added to the scaled scores (PyTorch's float mask); -inf masks the word
comptime QKV_MASK_ADD = 4


@always_inline
def _neg_inf() -> Float32:
    return bitcast[DType.float32](UInt32(0xFF800000))


@always_inline
def qkv_softmax_row(r: Int, a: Args):
    """Row r = (b, h, i): query i of head h of batch row b.
    p0 the scores (rows, S), p1 the probabilities, p2 the mask's source buffer (flags & 6);
    i0 S, i1 Lq, i2 H, i3 flags, i4 the mask's base offset, i5..i8 the mask's element strides along
    (b, h, i, j) (0 along a broadcast axis); f0 the scale.
    A word is left out by the causal mask (j > i, PyTorch's top-left aligned lower triangle), a zero boolean
    word, or an additive word that makes the score -inf. A left-out word is -inf before the softmax (its
    exponential is +0.0); a row with every word left out is written as zeros."""
    var S = a.i0
    var Lq = a.i1
    var i = r % Lq
    var bh = r // Lq
    var h = bh % a.i2
    var b = bh // a.i2
    var causal = (a.i3 & QKV_CAUSAL) != 0
    var boolean = (a.i3 & QKV_MASK_BOOL) != 0
    var additive = (a.i3 & QKV_MASK_ADD) != 0
    var mrow = a.i4 + b * a.i5 + h * a.i6 + i * a.i7
    var base = r * S
    var low = _neg_inf()
    var m = low
    var live = 0
    for j in range(S):
        var v = low
        var attend = not (causal and j > i)
        if attend and boolean:
            attend = bitcast[DType.int32](a.p2.unsafe_load(mrow + j * a.i8)) != Int32(0)
        if attend:
            v = ftz(identical_mul(ftz(a.p0.unsafe_load(base + j)), a.f0))
            if additive:
                v = ftz(v + ftz(a.p2.unsafe_load(mrow + j * a.i8)))
        if v != low:
            live += 1
            m = max(m, v)
        a.p1.unsafe_store(base + j, v)
    if live == 0:
        for j in range(S):
            a.p1.unsafe_store(base + j, Float32(0.0))
        return
    softmax_row_from_max_f32(a.p1, a.p1, base, S, m)


@always_inline
def apply_nn_qkv[OP: Int](t: Int, a: Args):
    comptime if OP == OP_QKV_SOFTMAX:
        qkv_softmax_row(t, a)
