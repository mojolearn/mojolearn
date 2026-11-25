# SPDX-License-Identifier: Apache-2.0
# THE BACKWARD RULES OF SEQUENTIAL'S LAYERS IN A SESSION (lane session-ops, 2026-11-25).
#
# The Session ops of lane session-ops (codes OPS_LAYER_FIRST..OPS_LAYER_LAST, autograd/ops_api.mojo) record the
# SAME tape nodes Sequential's layers record: the same forward functions (autograd/nn_conv.mojo, nn_conv3d.mojo,
# nn_norms.mojo, nn_attention.mojo, nn_ctc.mojo, ...), so the same custom tags and the same bits. Sequential
# differentiates those tags with ModelRules (autograd/model_blocks.mojo); a session differentiates with
# StepOpRules / ImportRules, which do not know them.
#
# NO EXISTING NODE PAYS FOR THEM. A session records `layer_nodes = True` the first time one of these ops records a
# node (OpsSession._layer_op), and each backward entry tests that Bool ONCE PER CALL (as nn-blocks-cbw tests
# `cbw.holds_rules()`): a session that never used a layer op differentiates with exactly the rules it used
# before. A flagged session wraps its usual rules in LayerRules below, which answers the layer families' tags
# and hands every other tag to the inner rules unchanged.
#
# Families (each a `# session-ops-<family>` group; a sub-lane whose op records a tag of another family adds it
# here in its own group):
#   NnConvRules       tags NNC_TAG_FIRST..NNC_TAG_LAST (560..): transposed convolution, reflection / replication
#                     padding, adaptive pools, upsampling, pixel shuffle (autograd/nn_conv.mojo)
#   NNAttentionRules  NN_TAG_SOFTMAX, NN_TAG_PE, NN_TAG_VIEW (autograd/nn_attention.mojo): multi-head attention
#                     and the positional encoding
#   NnCtcRules        NN_CTC_TAG (autograd/nn_ctc.mojo): the CTC loss
#   SessionOpsExtraRules  SOX_GATHER_TAG, SOX_LSE_TAG (autograd/session_ops_extra_rules.mojo): gather along an axis
#                     (also topk's values) and logsumexp (session-ops-extra)
# The 3-D convolution and pools, batch / group / instance norm and the activations record rank-N or import tags
# the inner rules already own.

# SESSION CODES (autograd/ops_api.mojo `_op_run_tensors`: one range test on the dispatch chain's no-match path,
# which no older operation reaches; then `_layer_op` picks the family). Each family's frontend codes (_SOP_*,
# autograd/autograd_ops_frontend_run.mojo) are 1000 + its session codes; its kernel opcodes, when it needs any
# (prefer the existing bodies), are its own range in sequence/dispatch.mojo.
comptime OPS_LAYER_FIRST = 130
comptime OPS_LAYER_CONV_FIRST = 130    # session-ops-conv: 130..139, kernels 6900..6919
comptime OPS_LAYER_POOL_FIRST = 140    # session-ops-pool: 140..149, kernels 6920..6939
comptime OPS_LAYER_RESIZE_FIRST = 150  # session-ops-resize: 150..159, kernels 6940..6959
comptime OPS_LAYER_REST_FIRST = 160    # session-ops-rest: 160..179, kernels 6960..6979
comptime OPS_LAYER_EXTRA_FIRST = 180   # session-ops-extra: 180..189, kernels 6980..6999 (sequence_mask, topk,
                                       # gather along an axis, logsumexp, soft-target cross-entropy)
comptime OPS_LAYER_LAST = 189

from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from autograd.tape import BackwardRules
from autograd.nn_conv import NnConvRules
from autograd.nn_attention import NNAttentionRules, nn_attention_rule
from autograd.nn_ctc import NnCtcRules
# session-ops-extra begin: gather along an axis and logsumexp (tags 6990..6991, autograd/session_ops_extra_rules.mojo)
from autograd.session_ops_extra_rules import SessionOpsExtraRules
# session-ops-extra end


def layer_rule(tag: Int) -> Bool:
    """True when `tag` is a Sequential layer tag LayerRules answers itself."""
    return NnConvRules.owns(tag) or nn_attention_rule(tag) or NnCtcRules.owns(tag) or SessionOpsExtraRules.owns(tag)


@fieldwise_init
struct LayerRules[R: BackwardRules & ImplicitlyCopyable & Movable](BackwardRules, ImplicitlyCopyable, Movable):
    """`inner` plus the tags of Sequential's layers recorded by Session ops. Only a session whose graph holds a
    layer op uses it (OpsSession.layer_nodes, tested once per backward call)."""

    var inner: Self.R

    def supports(self, tag: Int) -> Bool:
        if NnConvRules.owns(tag):
            return NnConvRules().supports(tag)
        if nn_attention_rule(tag):
            return NNAttentionRules().supports(tag)
        if NnCtcRules.owns(tag):
            return NnCtcRules().supports(tag)
        # session-ops-extra begin
        if SessionOpsExtraRules.owns(tag):
            return SessionOpsExtraRules().supports(tag)
        # session-ops-extra end
        return self.inner.supports(tag)

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if NnConvRules.owns(tag):
            return NnConvRules().zeroed_destination(tag, saved, edge)
        if nn_attention_rule(tag):
            return NNAttentionRules().zeroed_destination(tag, saved, edge)
        if NnCtcRules.owns(tag):
            return NnCtcRules().zeroed_destination(tag, saved, edge)
        # session-ops-extra begin
        if SessionOpsExtraRules.owns(tag):
            return SessionOpsExtraRules().zeroed_destination(tag, saved, edge)
        # session-ops-extra end
        return self.inner.zeroed_destination(tag, saved, edge)

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if NnConvRules.owns(tag):
            return NnConvRules().writes_total(tag, saved, edge)
        if nn_attention_rule(tag):
            return NNAttentionRules().writes_total(tag, saved, edge)
        if NnCtcRules.owns(tag):
            return NnCtcRules().writes_total(tag, saved, edge)
        # session-ops-extra begin
        if SessionOpsExtraRules.owns(tag):
            return SessionOpsExtraRules().writes_total(tag, saved, edge)
        # session-ops-extra end
        return self.inner.writes_total(tag, saved, edge)

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if NnConvRules.owns(tag):
            NnConvRules().apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
            return
        if nn_attention_rule(tag):
            NNAttentionRules().apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
            return
        if NnCtcRules.owns(tag):
            NnCtcRules().apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
            return
        # session-ops-extra begin
        if SessionOpsExtraRules.owns(tag):
            SessionOpsExtraRules().apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
            return
        # session-ops-extra end
        self.inner.apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
