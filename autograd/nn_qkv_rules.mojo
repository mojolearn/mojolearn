# SPDX-License-Identifier: Apache-2.0
"""The backward of `Session.attention`'s masked softmax node (lane nn-blocks-qkv, 2026-11-25), registered in
ImportRules (autograd/import_api.mojo, its nn-blocks-qkv group), so every rule set built on it differentiates
attention: the session's scalar and seeded backward, `vjp`, `Program.fit` (StepOpRules), the custom-backward
wrapper (CbwRules) and a Sequential's ModelRules. A recorded (create_graph) backward refuses it by name, as it
refuses every custom-rule node (the batched products included): autograd/higher_ops.mojo higher_family.

  tag                    inputs                      rule
  NN_TAG_SOFTMAX (6710)  [scores]                    the classic attention layer's tag and rule: d scores =
                                                     scale * softmax VJP (one OP_NN_ATTN_SOFTMAX_VJP launch)
  QKV_TAG_SOFTMAX_MASK   [scores, the float mask      the same launch for d scores; a second launch of the same
  (6850)                 expanded to the scores'      body with scale 1 for d mask (the mask is added after the
                         shape]                       scale, so dL/dmask = dL/d(scaled scores + mask)); the
                                                     expand's own rule sums it back over the broadcast axes

One body: both launches are autograd/nn_attention_ops.mojo nn_attn_softmax_vjp_row (the tape softmax's VJP
times f0); a masked word and every word of a fully masked row hold y = 0, so their gradients are 0.
"""
from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from autograd.tape import BackwardRules
from autograd.nn_attention import NN_TAG_SOFTMAX
from autograd.nn_attention_ops import OP_NN_ATTN_SOFTMAX_VJP

#: the masked softmax node of an attention whose float mask takes a gradient (a tag of lane nn-blocks-qkv's
#: range 6840..6859; no opcode of the range uses it)
comptime QKV_TAG_SOFTMAX_MASK = 6850


def _vjp_into[E: Exec](mut ex: E, output: FP, upstream: FP, dst: FP, cols: Int, rows: Int, factor: Float32) raises:
    var a = Args()
    a.p0 = output
    a.p1 = upstream
    a.p2 = dst
    a.i0 = 0
    a.i1 = cols
    a.f0 = factor
    ex.launch[OP_NN_ATTN_SOFTMAX_VJP](a, rows)


@fieldwise_init
struct QkvRules(BackwardRules, ImplicitlyCopyable, Movable):
    """The two tags above. saved: i0 the keys Lk (the row length), f0 the scale."""

    @staticmethod
    def owns(tag: Int) -> Bool:
        return tag == NN_TAG_SOFTMAX or tag == QKV_TAG_SOFTMAX_MASK

    def supports(self, tag: Int) -> Bool:
        return Self.owns(tag)

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # every launch writes every word of its destination
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        var n_rows = rows[0] * cols[0] // saved.i0
        if Int(gradients[0]) != 0:
            _vjp_into(ex, output, upstream, gradients[0], saved.i0, n_rows, saved.f0)
        if tag == QKV_TAG_SOFTMAX_MASK and len(gradients) > 1 and Int(gradients[1]) != 0:
            _vjp_into(ex, output, upstream, gradients[1], saved.i0, n_rows, Float32(1.0))
