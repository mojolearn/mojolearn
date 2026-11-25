# SPDX-License-Identifier: Apache-2.0
# THE ROUND'S ADDITIONS AS SESSION OPS (sub-lane session-ops-extra, Andrew 2026-11-25).
#
# Session codes 180..189 (autograd/session_layer_rules.mojo OPS_LAYER_EXTRA_FIRST; dispatched by
# OpsSession._layer_extra in autograd/ops_api.mojo), kernel codes 6980..6999 (autograd/session_ops_extra_ops.mojo),
# custom tags 6990..6991 (autograd/session_ops_extra_rules.mojo, answered by LayerRules). Every op reuses existing
# bodies; the only new bodies are the top-k order key and read-out and the logsumexp row and its VJP.
#
#   180 sequence_mask(lengths, max_length)  (B, T) float32, mask[b, t] = 1.0 when t < lengths[b] else 0.0: the
#       length mask Session.scan and the recurrent ops build (autograd/per_step_ops.mojo rnn_zero_padded: OP_SEQ_IOTA,
#       then the import family's compare), batch first. No gradient.
#   181 topk indices (x, k, axis, largest)  the positions of the k largest (or smallest) values along axis, sorted,
#       as an integer (ids) tensor: the order key pack (OP_SOX_TOPK_PACK), the embedding's bitonic stages
#       (OP_EMB_SORT_STAGE) over every line at once, the read-out (OP_SOX_TOPK_OUT). Ties: the lowest index first.
#       Session.topk returns (gather(x, axis, indices), indices): the values are a gather (182), so the gradient
#       reaches the selected positions through the gather's rule.
#   182 gather(x, axis, index)  torch.gather: out[..., j, ...] = x[..., index[..., j, ...], ...] along axis; the flat
#       source key of every output word by scatter_add's own key body (OP_SCT_KEYS), then the embedding's width-1
#       gather (OP_EMB_GATHER). Backward: the run-sorted scatter-add (SOX_GATHER_TAG): repeated indices accumulate.
#   183 logsumexp(x, axis, keepdims)  OP_SOX_LSE over the rows of x with axis moved last (a free view, materialised
#       once by the dense reader, as Session.log_softmax along a non-last axis does).
#   184 soft-target cross_entropy(logits, target, reduction)  target a distribution of the logits' shape:
#       loss_r = -sum_c target[r, c] * log_softmax(logits)[r, c], mean over rows or sum. Recorded with the tape's
#       own log_softmax, multiply, row sum and scale, so the gradient is theirs (softmax * sum(target) - target for
#       the logits, -log_softmax for the target) and a second-order backward works.
#   185 the same with per-position weights: weighted_reduce of autograd/per_step_ops.mojo (the weighted-loss path's
#       reduction): sum(w * loss) / sum(w) or sum(w * loss). A new code: the hard-label cross-entropy (28) and the
#       weighted-loss code (95) keep their branches.
#
# LAUNCHES. sequence_mask 2. topk: 1 pack + log2(Lp)(log2(Lp) + 1) / 2 stages + 1 read-out (Lp the padded axis
# length), then the gather. gather: the id check (one level per 256x ids and one two-word read-back), 1 key + 1
# gather; recording adds the runs (scan 2, sort 1 + stages + 1); backward 1. logsumexp: 1 (+ the permuted view's
# one materialisation), backward 1. Nothing is launched per row, per line or per element.
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_SEQ_IOTA, dummy_ptr
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN, rn_normalize_axis
from autograd.rankn_api import rn_shape, rn_dims, rn_as_shape
from autograd.ops import AXIS_COLS
from autograd.losses import REDUCTION_MEAN, REDUCTION_SUM
from autograd.import_api import imp_binary
from autograd.import_ops import IMP_LT
from autograd.per_step_ops import weighted_reduce
from autograd.embedding_ops import (
    OP_EMB_GATHER, OP_EMB_SORT_STAGE, AG_EMB_PLAN_AUTO, AG_EMB_PLAN_SORT, embedding_plan, embedding_perm_words,
    embedding_check_words, embedding_refuse_ids, embedding_runs_into, embedding_sort_length,
)
from autograd.nn_scatter_ops import OP_SCT_KEYS
from autograd.session_ops_extra_ops import OP_SOX_TOPK_PACK, OP_SOX_TOPK_OUT, OP_SOX_LSE
from autograd.session_ops_extra_rules import SOX_GATHER_TAG, SOX_LSE_TAG
from embedding.checks.embedding_oracle import EMB_MAX_POSITIONS
from autograd.session_ops_extra_sabotage import (
    SOX_SABOTAGE_SEQ_MASK, SOX_SABOTAGE_SEQ_MASK_HOST, SOX_SABOTAGE_TOPK_ZERO, SOX_SABOTAGE_TOPK_ZERO_HOST,
    SOX_SABOTAGE_LSE_SETTLE, SOX_SABOTAGE_LSE_SETTLE_HOST, SOX_SABOTAGE_SOFT_CE, SOX_SABOTAGE_SOFT_CE_HOST,
)
from autograd.import_ops import IMP_LE

#: Session codes (python/mojolearn/ops.py, the ops door); 186..189 are reserved for this family.
comptime OPS_SEQUENCE_MASK = 180
comptime OPS_TOPK_INDICES = 181
comptime OPS_GATHER_AXIS = 182
comptime OPS_LOGSUMEXP = 183
comptime OPS_SOFT_CROSS_ENTROPY = 184
comptime OPS_SOFT_CROSS_ENTROPY_WEIGHTED = 185

comptime _SOX_I32_MAX = 2147483647


def session_ops_extra_int_result(code: Int) -> Bool:
    """True when op `code` returns int32 words (an ids tensor): topk's indices."""
    return code == OPS_TOPK_INDICES


def sox_sequence_mask[E: Exec](mut tape: Tape, mut ex: E, lengths: Tensor, max_length: Int) raises -> Tensor:
    """(B, max_length) float32: 1.0 where t < lengths[b], else 0.0; B = lengths' element count."""
    if max_length < 1 or max_length >= (1 << 24):
        raise Error("ValueError: sequence_mask: max_length must be an integer in [1, 2^24), got " + String(max_length))
    var b = tape.shape_of(lengths).count()
    var pos = tape.reserve(ex, max_length)
    var ia = Args()
    ia.p0 = pos
    ex.launch[OP_SEQ_IOTA](ia, max_length)
    var times = tape.variable(ex, pos, 1, max_length, False)
    var lens = rn_as_shape(tape, ex, tape.detach(ex, lengths), ShapeN.matrix(b, 1))
    comptime if (SOX_SABOTAGE_SEQ_MASK and E.IS_DEVICE) or (SOX_SABOTAGE_SEQ_MASK_HOST and not E.IS_DEVICE):
        return imp_binary(tape, ex, times, lens, IMP_LE)  # SABOTAGE: t <= lengths[b]
    return imp_binary(tape, ex, times, lens, IMP_LT)


def _sox_lines(xs: ShapeN, d: Int) -> Tuple[Int, Int, Int]:
    """x seen as (outer, extent, inner) around axis d."""
    var outer = 1
    var inner = 1
    for k in range(xs.ndim):  # small-loop(nd: tensor rank, at most 8): shape metadata, no data
        if k < d:
            outer *= xs.d[k]
        elif k > d:
            inner *= xs.d[k]
    return (outer, xs.d[d], inner)


def sox_topk_indices[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, k: Int, axis: Int, largest: Bool,
) raises -> Tensor:
    """The positions of the k largest (smallest) values of x along axis, rank order, ties to the lowest index: a
    tensor of int32 words of x's shape with the axis length k. No gradient (the values are a gather)."""
    var xs = tape.shape_of(x)
    if xs.ndim < 1:
        raise Error("ValueError: topk: x must have at least one dimension")
    var d = rn_normalize_axis(axis, xs.ndim)
    var lines = _sox_lines(xs, d)
    var V = lines[1]
    if k < 1 or k > V:
        raise Error("ValueError: topk: k must be in [1, " + String(V) + "] (the length of axis " + String(axis)
                    + "), got " + String(k))
    var lp = embedding_sort_length(V)
    var n_lines = lines[0] * lines[2]
    if n_lines > _SOX_I32_MAX // lp:
        raise Error("ValueError: topk: x must hold fewer than 2^31 values once each line is padded to a power of two")
    var total = n_lines * lp
    var keys = tape.reserve(ex, total)
    var vals = tape.reserve(ex, total)
    var pa = Args()
    pa.p0 = tape.value_of(ex, x)
    pa.p1 = keys
    pa.p2 = vals
    pa.i0 = V
    pa.i1 = lines[2]
    pa.i2 = lp
    pa.i3 = 1 if largest else 0
    pa.i4 = 1
    comptime if (SOX_SABOTAGE_TOPK_ZERO and E.IS_DEVICE) or (SOX_SABOTAGE_TOPK_ZERO_HOST and not E.IS_DEVICE):
        pa.i4 = 0  # SABOTAGE: -0.0 kept below +0.0
    ex.launch[OP_SOX_TOPK_PACK](pa, total)
    # the embedding's network over every line at once: a stage's pairs stay inside one aligned block of lp words
    var m = 2
    while m <= lp:  # driver-loop: over sort network merge sizes
        var j = m // 2
        while j > 0:  # driver-loop: over sort network stages
            var s = Args()
            s.p0 = keys
            s.p1 = vals
            s.i0 = m
            s.i1 = j
            ex.launch[OP_EMB_SORT_STAGE](s, total // 2)
            j //= 2
        m *= 2
    var dims = rn_dims(xs)
    dims[d] = k
    var shape = rn_shape(dims, "topk")
    var count = shape.count()
    var out = tape.reserve(ex, count)
    var oa = Args()
    oa.p0 = vals
    oa.p1 = out
    oa.i0 = k
    oa.i1 = lines[2]
    oa.i2 = lp
    ex.launch[OP_SOX_TOPK_OUT](oa, count)
    return tape.variable_nd(ex, out, shape, False)


def sox_gather[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, axis: Int, index: FP, index_shape: ShapeN,
) raises -> Tensor:
    """torch.gather(x, axis, index): a tensor of index's shape, out[e] = x at e's position with the axis coordinate
    replaced by index[e]. `index` holds int32 words (an ids tensor) of index_shape, which matches x on every axis
    but `axis`; every index in [0, x.shape[axis]) (refused by position and value)."""
    var xs = tape.shape_of(x)
    var nd = xs.ndim
    if nd < 1 or index_shape.ndim != nd:
        raise Error("ValueError: gather: x and index must have the same number of dimensions (x has " + String(nd)
                    + ", index " + String(index_shape.ndim) + ")")
    var d = rn_normalize_axis(axis, nd)
    for q in range(nd):  # small-loop(nd: tensor rank, at most 8): shape metadata, no data
        if q != d and index_shape.d[q] != xs.d[q]:
            raise Error("ValueError: gather: index must have x's size on every dimension except axis (dimension "
                        + String(q) + ": x " + String(xs.d[q]) + ", index " + String(index_shape.d[q]) + ")")
    var lines = _sox_lines(xs, d)
    var V = lines[1]
    var K = index_shape.d[d]
    var P = index_shape.count()
    var C = xs.count()
    if P > EMB_MAX_POSITIONS or C > EMB_MAX_POSITIONS:
        raise Error("ValueError: gather: x and index must each hold fewer than 2^30 values")
    var cw = embedding_check_words(P)
    var scratch_a = tape.reserve(ex, cw)
    var scratch_b = tape.reserve(ex, cw)
    try:
        embedding_refuse_ids(ex, index, P, V, scratch_a, scratch_b)
    except e:
        var message = String(e)
        if "embedding: id " in message:
            raise Error("ValueError: gather: " + message.replace("embedding: id", "index"))
        raise Error(message)
    var keys = tape.reserve(ex, P)
    var ka = Args()
    ka.p0 = index
    ka.p1 = keys
    ka.i0 = K
    ka.i1 = lines[2]
    ka.i2 = V
    ex.launch[OP_SCT_KEYS](ka, P)
    var out = tape.reserve(ex, P)
    var ga = Args()
    ga.p0 = tape.value_of(ex, x)
    ga.p1 = keys
    ga.p2 = out
    ga.i0 = 1
    ex.launch[OP_EMB_GATHER](ga, P)
    if not (tape.is_recording() and tape.requires_grad(x)):
        return tape.variable_nd(ex, out, index_shape, False)
    # the runs of the backward's scatter-add, built once here (the keys are this node's)
    var plan = embedding_plan(AG_EMB_PLAN_AUTO, C, P)
    var pw = embedding_perm_words(plan, P)
    var run_begin = tape.reserve(ex, C + 1)
    var perm = tape.reserve(ex, pw)
    var sort_keys = tape.reserve(ex, pw) if plan == AG_EMB_PLAN_SORT else dummy_ptr()
    embedding_runs_into(ex, keys, P, C, -1, plan, run_begin, perm, sort_keys)
    var saved = Args()
    saved.p0 = keys
    saved.p1 = perm
    saved.p2 = run_begin
    saved.i0 = P
    saved.i1 = C
    var inputs = List[Tensor]()
    inputs.append(x)
    return tape.custom_nd(ex, inputs, out, index_shape, SOX_GATHER_TAG, saved)


def sox_logsumexp[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, axis: Int, keepdims: Bool) raises -> Tensor:
    """log(sum(exp(x))) along axis with log-softmax's max-shift fold: x's shape without the axis (keepdims: with
    it as length 1; a 1-D x gives shape (1,))."""
    var xs = tape.shape_of(x)
    if xs.ndim < 1:
        raise Error("ValueError: logsumexp: x must have at least one dimension")
    var d = rn_normalize_axis(axis, xs.ndim)
    var moved = x
    if d != xs.ndim - 1:
        var p = List[Int](capacity=xs.ndim)
        for q in range(xs.ndim):  # small-loop(nd: tensor rank, at most 8): shape metadata, no data
            if q != d:
                p.append(q)
        p.append(d)
        moved = tape.permute(ex, x, p)
    var cols = xs.d[d]
    var rows = xs.count() // cols
    var src = tape.value_of(ex, moved)
    var out = tape.reserve(ex, rows)
    var a = Args()
    a.p0 = src
    a.p1 = out
    a.i0 = cols
    a.i1 = 1
    comptime if (SOX_SABOTAGE_LSE_SETTLE and E.IS_DEVICE) or (SOX_SABOTAGE_LSE_SETTLE_HOST and not E.IS_DEVICE):
        a.i1 = 0  # SABOTAGE: an infinite maximum not settled
    ex.launch[OP_SOX_LSE](a, rows)
    var dims = List[Int]()
    for q in range(xs.ndim):  # small-loop(nd: tensor rank, at most 8): shape metadata, no data
        if q != d:
            dims.append(xs.d[q])
        elif keepdims:
            dims.append(1)
    if len(dims) == 0:
        dims.append(1)
    var shape = rn_shape(dims, "logsumexp")
    if not (tape.is_recording() and tape.requires_grad(x)):
        return tape.variable_nd(ex, out, shape, False)
    var saved = Args()
    saved.p0 = src
    saved.i0 = rows
    saved.i1 = cols
    var inputs = List[Tensor]()
    inputs.append(moved)
    return tape.custom_nd(ex, inputs, out, shape, SOX_LSE_TAG, saved)


def sox_soft_cross_entropy[E: Exec](
    mut tape: Tape, mut ex: E, logits: Tensor, target: Tensor, reduction: Int, weight: Optional[Tensor],
) raises -> Tensor:
    """The cross-entropy of (..., classes) logits against a target distribution of the same shape: per position
    -sum_c target * log_softmax(logits), then the mean over positions or the sum (weighted: weighted_reduce)."""
    if reduction != REDUCTION_MEAN and reduction != REDUCTION_SUM:
        raise Error("ValueError: reduction must be 'mean' or 'sum'")
    var zs = tape.shape_of(logits)
    if not zs.same(tape.shape_of(target)):
        raise Error("ValueError: cross_entropy: a target distribution must have the logits' shape")
    var rows = zs.rows()
    var cols = zs.cols()
    var z = rn_as_shape(tape, ex, logits, ShapeN.matrix(rows, cols))
    var t = rn_as_shape(tape, ex, target, ShapeN.matrix(rows, cols))
    var ls = tape.log_softmax(ex, z)
    var sign = Float32(-1.0)
    comptime if (SOX_SABOTAGE_SOFT_CE and E.IS_DEVICE) or (SOX_SABOTAGE_SOFT_CE_HOST and not E.IS_DEVICE):
        sign = Float32(1.0)  # SABOTAGE: the minus sign lost
    var terms = tape.scale(ex, tape.sum(ex, tape.multiply(ex, t, ls), AXIS_COLS), sign)
    if weight:
        var w = weight.value()
        var wn = tape.shape_of(w).count()
        if wn != rows:
            raise Error("ValueError: ops: weight must hold one value per position of the logits (" + String(rows)
                        + " values), got " + String(wn))
        return weighted_reduce(tape, ex, terms, rn_as_shape(tape, ex, w, ShapeN.matrix(rows, 1)), reduction)
    if reduction == REDUCTION_SUM:
        return tape.sum(ex, terms)
    return tape.mean(ex, terms)
