# SPDX-License-Identifier: Apache-2.0
"""Sequential's norms, parameterised activations, multi-head attention, positional encoding, losses and raw
state-space scans as `ops.Session` operations (lane session-ops-rest, 2026-11-25; session codes 160..179).

Each operation calls the forward function Sequential's layer (or `compile(loss=...)`) calls, on the session's
tape, so it records the same nodes and tags and gives Sequential's bits by construction:

  code  Session method                      forward (the Sequential layer's)                  backward tags
  160   batch_norm                          rankn_batch_norm_core (autograd/rankn_conv.mojo)   RN_BATCH_NORM
  161   group_norm                          graph_group_norm (autograd/nn_norms.mojo)          rank-N / import
  162   instance_norm                       graph_instance_norm (autograd/nn_norms.mojo)       rank-N / import
  163   prelu                               imp_binary IMP_PRELU (autograd/nn_activations.mojo) import
  164   mish                                imp_unary IMP_MISH (autograd/nn_activations.mojo)  import
  165   multi_head_attention                nn_mha_core (autograd/nn_attention.mojo)           NNAttentionRules
  166   positional_encoding                 nn_pos_enc_forward (autograd/nn_attention.mojo)    NNAttentionRules
  167   smooth_l1_loss                      losses.smooth_l1 (LS_SMOOTH_L1)                    loss_reduced
  168   nll_loss                            losses.nll (LS_NLL) on the one-hot of class ids    loss_reduced
  169   kl_div                              losses.kl_div (LS_KL_DIV)                          loss_reduced
  170   binary_cross_entropy_with_logits    losses.binary_cross_entropy_with_logits(pos_weight) loss_reduced
  171   ctc_loss                            nn_ctc.ctc_loss                                    NnCtcRules
  172   selective_scan                      graph_selective_scan (autograd/ssm_primitives.mojo) SsmRules
  173   ssd_scan                            graph_ssd_scan (autograd/ssm_m2_primitives.mojo)   SsmRules
  174   trapezoid_scan                      graph_trapezoid_scan (autograd/ssm_m3_primitives.mojo) SsmRules
  175..179 unused.

No new kernel: the glue below launches only existing bodies. RUNNING STATISTICS: Sequential keeps a norm's
running statistics as one [mean C | var C] block; a Session has PyTorch's two tensors (running_mean, running_var,
state tensors written in place). The norm reads and updates a [mean | var] block of tape storage: two OP_COPY
launches (C words each) fill it before the norm, two copy the updated words back after a training call (eval:
the two copies in only). CLASS IDS (nll_loss's target, ctc_loss's targets and lengths given with Session.ids):
OP_ONE_HOT, the fit's own target expansion (one launch: one-hot rows for nll_loss, the ids' values as float words
for ctc_loss). CTC TARGETS: OP_CTC_PACK, the fit's packing (one launch), then the admission the fit does on the
host before its upload, done here on the executor because the targets are session tensors (nn_ctc.ctc_refuse_rows:
OP_CTC_ADMIT flags each word, the embedding's first-bad search, a two-word read-back), because the CTC bodies read
the labels and lengths unchecked.

Launches beyond the Sequential layer's own, per call: batch_norm / instance_norm with running statistics 2
(eval) or 4 (training); nll_loss 2 (id check, one-hot) + the id check's two-word read-back; ctc_loss 1 (pack)
+ 1 per ids operand + the permute's copy of (T, N, C) into (N, T, C) + the admission (1 flag launch, the search's
levels, 1 value launch, a two-word read-back); every other op 0.
"""
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_COPY, OP_ONE_HOT, dummy_ptr
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN
from autograd.rankn_api import rn_shape
from autograd.rankn_conv import rankn_batch_norm_core
from autograd.nn_norms import graph_group_norm, graph_instance_norm
from autograd.import_api import imp_unary, imp_binary
from autograd.import_ops import IMP_MISH, IMP_PRELU
from autograd.nn_attention import nn_mha_core, nn_pos_enc_forward, NN_MHA_BIAS, NN_MHA_CAUSAL, NN_MHA_KEY_PADDING, NN_PE_MAX_LEN
from autograd.losses import smooth_l1, nll, kl_div, binary_cross_entropy_with_logits
from autograd.nn_ctc import ctc_loss, ctc_pack_targets, ctc_refuse_rows
from autograd.nn_ctc_ops import CTC_LABELS
from autograd.embedding_ops import embedding_check_words, embedding_refuse_ids
from autograd.ssm_primitives import graph_selective_scan
from autograd.ssm_m2_primitives import graph_ssd_scan
from autograd.ssm_m3_primitives import graph_trapezoid_scan
from autograd.session_ops_layer_sabotage import (
    SOL_SABOTAGE_NORM_EPS, SOL_SABOTAGE_NORM_EPS_HOST, SOL_SABOTAGE_MISH_SCALE, SOL_SABOTAGE_MISH_SCALE_HOST,
    SOL_SABOTAGE_MHA_CAUSAL, SOL_SABOTAGE_MHA_CAUSAL_HOST, SOL_SABOTAGE_SMOOTH_L1_BETA,
    SOL_SABOTAGE_SMOOTH_L1_BETA_HOST, SOL_SABOTAGE_CTC_REDUCTION, SOL_SABOTAGE_CTC_REDUCTION_HOST,
    SOL_SABOTAGE_SSM_SCALE, SOL_SABOTAGE_SSM_SCALE_HOST,
)

comptime SOR_BATCH_NORM = 160
comptime SOR_GROUP_NORM = 161
comptime SOR_INSTANCE_NORM = 162
comptime SOR_PRELU = 163
comptime SOR_MISH = 164
comptime SOR_MHA = 165
comptime SOR_POS_ENC = 166
comptime SOR_SMOOTH_L1 = 167
comptime SOR_NLL = 168
comptime SOR_KL_DIV = 169
comptime SOR_BCE_POS_WEIGHT = 170
comptime SOR_CTC = 171
comptime SOR_SELECTIVE_SCAN = 172
comptime SOR_SSD_SCAN = 173
comptime SOR_TRAPEZOID_SCAN = 174
comptime SOR_FIRST = 160
comptime SOR_LAST = 174


def sor_name(code: Int) -> String:
    """The public Session name of code `code` (program listings and refusals)."""
    if code == SOR_BATCH_NORM:
        return "batch_norm"
    if code == SOR_GROUP_NORM:
        return "group_norm"
    if code == SOR_INSTANCE_NORM:
        return "instance_norm"
    if code == SOR_PRELU:
        return "prelu"
    if code == SOR_MISH:
        return "mish"
    if code == SOR_MHA:
        return "multi_head_attention"
    if code == SOR_POS_ENC:
        return "positional_encoding"
    if code == SOR_SMOOTH_L1:
        return "smooth_l1_loss"
    if code == SOR_NLL:
        return "nll_loss"
    if code == SOR_KL_DIV:
        return "kl_div"
    if code == SOR_BCE_POS_WEIGHT:
        return "binary_cross_entropy_with_logits"
    if code == SOR_CTC:
        return "ctc_loss"
    if code == SOR_SELECTIVE_SCAN:
        return "selective_scan"
    if code == SOR_SSD_SCAN:
        return "ssd_scan"
    if code == SOR_TRAPEZOID_SCAN:
        return "trapezoid_scan"
    return "op_" + String(code)


def _refuse(name: String, words: String) raises:
    raise Error("ValueError: ops: " + name + ": " + words)


def _count(mut tape: Tape, t: Tensor) raises -> Int:
    return tape.rows(t) * tape.cols(t)


def _row_view[E: Exec](mut tape: Tape, mut ex: E, t: Tensor, n: Int) raises -> Tensor:
    """t (n values in any shape) as the (1, n) row Sequential's parameters are; t itself when it is one."""
    var s = tape.shape_of(t)
    if s.ndim == 2 and s.d[0] == 1:
        return t
    var d: List[Int] = [1, n]
    return tape.view_nd(ex, t, rn_shape(d, "ops"))


def _ncr(mut tape: Tape, x: Tensor, name: String) raises -> Tuple[Int, Int, Int]:
    """(N, C, rest) of x (N, C, *): the batch, the channel axis 1 and the product of the axes after it."""
    var s = tape.shape_of(x)
    if s.ndim < 2:
        _refuse(name, "input must be (N, C, ...)")
    var rest = 1
    for k in range(2, s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        rest *= s.d[k]
    return (s.d[0], s.d[1], rest)


# ---------------------------------------------------------------- running statistics
def _stats_in[E: Exec](mut tape: Tape, mut ex: E, mean: FP, var_: FP, C: Int) raises -> FP:
    """A [mean C | var C] block of tape storage holding the caller's two running-statistics tensors."""
    var block = tape.reserve(ex, 2 * C)
    var a = Args()
    a.p0 = mean
    a.p1 = block
    ex.launch[OP_COPY](a, C)
    var b = Args()
    b.p0 = var_
    b.p1 = block + C
    ex.launch[OP_COPY](b, C)
    return block


def _stats_out[E: Exec](mut ex: E, block: FP, mean: FP, var_: FP, C: Int) raises:
    """The updated [mean | var] block back into the caller's running_mean and running_var (in place)."""
    var a = Args()
    a.p0 = block
    a.p1 = mean
    ex.launch[OP_COPY](a, C)
    var b = Args()
    b.p0 = block + C
    b.p1 = var_
    ex.launch[OP_COPY](b, C)


def _check_channel_vector(mut tape: Tape, t: Optional[Tensor], C: Int, name: String, what: String) raises:
    if t and _count(tape, t.value()) != C:
        _refuse(name, what + " must hold one value per channel (" + String(C) + ")")


# ---------------------------------------------------------------- norms
def sor_batch_norm[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, running_mean: Optional[Tensor], running_var: Optional[Tensor],
    weight: Optional[Tensor], bias: Optional[Tensor], training: Bool, momentum: Float32, eps: Float32,
) raises -> Tensor:
    """torch.nn.functional.batch_norm over x (N, C, *): Sequential's batch_norm1d / 2d layer (x viewed (N, C,
    rest), the core's bodies), the running statistics updated in place when training."""
    var name = String("batch_norm")
    var ncr = _ncr(tape, x, name)
    var C = ncr[1]
    _check_channel_vector(tape, running_mean, C, name, "running_mean")
    _check_channel_vector(tape, running_var, C, name, "running_var")
    _check_channel_vector(tape, weight, C, name, "weight")
    _check_channel_vector(tape, bias, C, name, "bias")
    var s = tape.shape_of(x)
    var d3: List[Int] = [ncr[0], C, ncr[2]]
    var x3 = tape.view_nd(ex, x, rn_shape(d3, "batch_norm"))
    var running: Optional[FP] = None
    var mean_at = dummy_ptr()
    var var_at = dummy_ptr()
    if running_mean:
        mean_at = tape.value_of(ex, running_mean.value())
        var_at = tape.value_of(ex, running_var.value())
        running = Optional[FP](_stats_in(tape, ex, mean_at, var_at, C))
    var core_eps = eps
    comptime if (SOL_SABOTAGE_NORM_EPS and E.IS_DEVICE) or (SOL_SABOTAGE_NORM_EPS_HOST and not E.IS_DEVICE):
        core_eps = eps + Float32(1.0)  # SABOTAGE: the wrong epsilon
    var y = rankn_batch_norm_core(tape, ex, x3, weight, bias, running, core_eps, momentum, training)
    if running and training:
        _stats_out(ex, running.value(), mean_at, var_at, C)
    return tape.view_nd(ex, y, s)


def sor_group_norm[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, num_groups: Int, weight: Optional[Tensor], bias: Optional[Tensor],
    eps: Float32,
) raises -> Tensor:
    """torch.nn.functional.group_norm over x (N, C, *): Sequential's group_norm layer."""
    var name = String("group_norm")
    var ncr = _ncr(tape, x, name)
    var C = ncr[1]
    if num_groups < 1 or C % num_groups != 0:
        _refuse(name, "num_groups (" + String(num_groups) + ") must divide the channels (" + String(C) + ")")
    _check_channel_vector(tape, weight, C, name, "weight")
    _check_channel_vector(tape, bias, C, name, "bias")
    var w: Optional[Tensor] = None
    var b: Optional[Tensor] = None
    if weight:
        w = Optional[Tensor](_row_view(tape, ex, weight.value(), C))
        b = Optional[Tensor](_row_view(tape, ex, bias.value(), C))
    return graph_group_norm(tape, ex, x, ncr[0], C, ncr[2], num_groups, w, b, eps)


def sor_instance_norm[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, running_mean: Optional[Tensor], running_var: Optional[Tensor],
    weight: Optional[Tensor], bias: Optional[Tensor], use_input_stats: Bool, momentum: Float32, eps: Float32,
) raises -> Tensor:
    """torch.nn.functional.instance_norm over x (N, C, *): Sequential's instance_norm layer; with running
    statistics and use_input_stats they are updated in place, without use_input_stats they normalize."""
    var name = String("instance_norm")
    var ncr = _ncr(tape, x, name)
    var C = ncr[1]
    _check_channel_vector(tape, running_mean, C, name, "running_mean")
    _check_channel_vector(tape, running_var, C, name, "running_var")
    _check_channel_vector(tape, weight, C, name, "weight")
    _check_channel_vector(tape, bias, C, name, "bias")
    if not use_input_stats and not running_mean:
        _refuse(name, "use_input_stats=False normalizes with running_mean and running_var; pass both")
    if use_input_stats and ncr[2] < 2:
        _refuse(name, "expected more than 1 spatial element per channel")
    var w: Optional[Tensor] = None
    var b: Optional[Tensor] = None
    if weight:
        w = Optional[Tensor](_row_view(tape, ex, weight.value(), C))
        b = Optional[Tensor](_row_view(tape, ex, bias.value(), C))
    var running: Optional[FP] = None
    var mean_at = dummy_ptr()
    var var_at = dummy_ptr()
    if running_mean:
        mean_at = tape.value_of(ex, running_mean.value())
        var_at = tape.value_of(ex, running_var.value())
        running = Optional[FP](_stats_in(tape, ex, mean_at, var_at, C))
    var y = graph_instance_norm(tape, ex, x, ncr[0], C, ncr[2], w, b, running, momentum, eps, use_input_stats)
    if running and use_input_stats:
        _stats_out(ex, running.value(), mean_at, var_at, C)
    return y


# ---------------------------------------------------------------- activations
def sor_prelu[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, weight: Tensor) raises -> Tensor:
    """torch.nn.functional.prelu: Sequential's prelu layer, the slope on axis 1 ((C, 1, .., 1) right-aligned to
    the sample's shape) or one shared slope."""
    var name = String("prelu")
    var s = tape.shape_of(x)
    var n = _count(tape, weight)
    var C = s.d[1] if s.ndim >= 2 else 1
    if n != 1 and n != C:
        _refuse(name, "weight must hold 1 value or one per channel (" + String(C) + "), not " + String(n))
    var wd = List[Int]()
    wd.append(n)
    if n > 1:
        for _ in range(2, s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            wd.append(1)
    var wv = tape.view_nd(ex, weight, rn_shape(wd, "prelu"))
    return imp_binary(tape, ex, x, wv, IMP_PRELU)


def sor_mish[E: Exec](mut tape: Tape, mut ex: E, x: Tensor) raises -> Tensor:
    """torch.nn.functional.mish: Sequential's mish layer."""
    comptime if (SOL_SABOTAGE_MISH_SCALE and E.IS_DEVICE) or (SOL_SABOTAGE_MISH_SCALE_HOST and not E.IS_DEVICE):
        return tape.scale(ex, imp_unary(tape, ex, x, IMP_MISH), Float32(2.0))  # SABOTAGE: twice the value
    else:
        return imp_unary(tape, ex, x, IMP_MISH)


# ---------------------------------------------------------------- attention
def sor_multi_head_attention[E: Exec](
    mut tape: Tape, mut ex: E, query: Tensor, memory: Tensor, self_attention: Bool, num_heads: Int,
    w_in: Tensor, b_in: Optional[Tensor], w_kv: Optional[Tensor], b_kv: Optional[Tensor], wo: Tensor,
    bo: Optional[Tensor], key_lengths: Optional[Tensor], is_causal: Bool, dropout_p: Float32, training: Bool,
    seed: UInt64, counter: Int,
) raises -> Tensor:
    """Sequential's multihead_attention layer (nn_mha_core) on query (B, T, E) and keys/values `memory`
    (B, S, kdim; the query itself for self-attention), the weights as session tensors in the (in, out)
    layout of Session.linear."""
    var name = String("multi_head_attention")
    var qs = tape.shape_of(query)
    if qs.ndim != 3:
        _refuse(name, "query must be (batch, length, embed_dim)")
    var B = qs.d[0]
    var T = qs.d[1]
    var Ed = qs.d[2]
    if num_heads < 1 or Ed % num_heads != 0:
        _refuse(name, "num_heads (" + String(num_heads) + ") must divide embed_dim (" + String(Ed) + ")")
    var S = T
    var kd = Ed
    if not self_attention:
        var ms = tape.shape_of(memory)
        if ms.ndim != 3 or ms.d[0] != B:
            _refuse(name, "key and value must be (batch, length, kdim) with the query's batch")
        S = ms.d[1]
        kd = ms.d[2]
    if self_attention:
        if _count(tape, w_in) != Ed * 3 * Ed or tape.cols(w_in) != 3 * Ed:
            _refuse(name, "in_proj_weight must be (embed_dim, 3 * embed_dim): the (in, out) layout, the "
                    + "transpose of PyTorch's (3 * embed_dim, embed_dim)")
    else:
        if tape.rows(w_in) != Ed or tape.cols(w_in) != Ed:
            _refuse(name, "the query projection must be (embed_dim, embed_dim)")
        if tape.rows(w_kv.value()) != kd or tape.cols(w_kv.value()) != 2 * Ed:
            _refuse(name, "the key/value projection must be (kdim, 2 * embed_dim), the columns [key | value]")
    if tape.rows(wo) != Ed or tape.cols(wo) != Ed:
        _refuse(name, "out_proj_weight must be (embed_dim, embed_dim)")
    var flags = 0
    var b_in_row: Optional[Tensor] = None
    var b_kv_row: Optional[Tensor] = None
    var bo_row: Optional[Tensor] = None
    if b_in:
        flags |= NN_MHA_BIAS
        var nb = 3 * Ed if self_attention else Ed
        if _count(tape, b_in.value()) != nb:
            _refuse(name, "in_proj_bias must hold " + String(nb) + " values")
        if _count(tape, bo.value()) != Ed:
            _refuse(name, "out_proj_bias must hold embed_dim values")
        b_in_row = Optional[Tensor](_row_view(tape, ex, b_in.value(), nb))
        bo_row = Optional[Tensor](_row_view(tape, ex, bo.value(), Ed))
        if not self_attention:
            if _count(tape, b_kv.value()) != 2 * Ed:
                _refuse(name, "the key/value bias must hold 2 * embed_dim values")
            b_kv_row = Optional[Tensor](_row_view(tape, ex, b_kv.value(), 2 * Ed))
    if is_causal:
        if S != T:
            _refuse(name, "is_causal needs as many keys as queries")
        flags |= NN_MHA_CAUSAL
        comptime if (SOL_SABOTAGE_MHA_CAUSAL and E.IS_DEVICE) or (SOL_SABOTAGE_MHA_CAUSAL_HOST and not E.IS_DEVICE):
            flags = flags & ~NN_MHA_CAUSAL  # SABOTAGE: the causal mask dropped
    var lengths: Optional[FP] = None
    if key_lengths:
        if _count(tape, key_lengths.value()) != B:
            _refuse(name, "key_lengths must hold one length per batch row (" + String(B) + ")")
        flags |= NN_MHA_KEY_PADDING
        lengths = Optional[FP](tape.value_of(ex, key_lengths.value()))
    if not (dropout_p >= Float32(0) and dropout_p < Float32(1)):
        _refuse(name, "dropout_p must be in [0, 1)")
    var cur = ShapeN.matrix(T, Ed)
    return nn_mha_core(tape, ex, cur, num_heads, flags, dropout_p, query, memory, w_in, b_in_row, w_kv, b_kv_row, wo,
                       bo_row, training, seed, counter, lengths)


def sor_positional_encoding[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, dropout_p: Float32, training: Bool, seed: UInt64, counter: Int,
) raises -> Tensor:
    """Sequential's positional_encoding layer: x (..., length, d_model) + the sinusoidal table, then dropout."""
    var name = String("positional_encoding")
    var s = tape.shape_of(x)
    if s.ndim < 2:
        _refuse(name, "x must be (..., length, d_model)")
    var T = s.d[s.ndim - 2]
    var d = s.d[s.ndim - 1]
    if d % 2 != 0:
        _refuse(name, "d_model must be even")
    if T > NN_PE_MAX_LEN:
        _refuse(name, "the length " + String(T) + " is longer than " + String(NN_PE_MAX_LEN))
    if not (dropout_p >= Float32(0) and dropout_p < Float32(1)):
        _refuse(name, "dropout_p must be in [0, 1)")
    var y = nn_pos_enc_forward(tape, ex, dropout_p, ShapeN.matrix(T, d), x, training, seed, counter)
    return tape.view_nd(ex, y, s)


# ---------------------------------------------------------------- losses
def _same_shape(mut tape: Tape, a: Tensor, b: Tensor, name: String) raises:
    if not tape.shape_of(a).same(tape.shape_of(b)):
        _refuse(name, "target must have the input's shape")


def sor_smooth_l1_loss[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, target: Tensor, reduction: Int, beta: Float32,
) raises -> Tensor:
    _same_shape(tape, x, target, "smooth_l1_loss")
    if not (beta >= Float32(0)):
        _refuse("smooth_l1_loss", "beta must be a non-negative number")
    comptime if (SOL_SABOTAGE_SMOOTH_L1_BETA and E.IS_DEVICE) or (SOL_SABOTAGE_SMOOTH_L1_BETA_HOST and not E.IS_DEVICE):
        return smooth_l1(tape, ex, x, target, beta + Float32(1.0), reduction)  # SABOTAGE: the wrong beta
    else:
        return smooth_l1(tape, ex, x, target, beta, reduction)


def sor_kl_div[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, target: Tensor, reduction: Int) raises -> Tensor:
    _same_shape(tape, x, target, "kl_div")
    return kl_div(tape, ex, x, target, reduction)


def sor_bce_with_logits[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, target: Tensor, reduction: Int, pos_weight: Float32,
) raises -> Tensor:
    _same_shape(tape, x, target, "binary_cross_entropy_with_logits")
    if not (pos_weight > Float32(0)):
        _refuse("binary_cross_entropy_with_logits", "pos_weight must be positive")
    return binary_cross_entropy_with_logits(tape, ex, x, target, reduction, pos_weight)


def _refuse_ids[E: Exec](mut tape: Tape, mut ex: E, ids: FP, n: Int, bound: Int, name: String, word: String) raises:
    """An id outside [0, bound), by position and value, refused before any kernel reads it (the embedding's
    first-bad search: one two-word read-back)."""
    var cw = embedding_check_words(n)
    var scratch_a = tape.reserve(ex, cw)
    var scratch_b = tape.reserve(ex, cw)
    try:
        embedding_refuse_ids(ex, ids, n, bound, scratch_a, scratch_b)
    except e:
        var message = String(e)
        if "embedding: id " in message:
            raise Error("ValueError: ops: " + name + ": " + message.replace("embedding: id", word))
        raise Error(message)


def sor_nll_loss[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, target: Tensor, reduction: Int) raises -> Tensor:
    """torch.nn.functional.nll_loss of log-probabilities x (N, C) against N class ids: the fit's one-hot of the
    ids (OP_ONE_HOT) and the LS_NLL loss; 'mean' divides by N, as NLLLoss without class weights."""
    var name = String("nll_loss")
    var s = tape.shape_of(x)
    if s.ndim != 2 or s.d[1] < 2:
        _refuse(name, "input must be (N, C) log-probabilities with at least 2 classes")
    var N = s.d[0]
    var C = s.d[1]
    if _count(tape, target) != N:
        _refuse(name, "target must hold one class id per row of input (" + String(N) + ")")
    var ids = tape.value_of(ex, target)
    _refuse_ids(tape, ex, ids, N, C, name, "target class")
    var dense = tape.reserve(ex, N * C)
    var a = Args()
    a.p0 = ids
    a.p1 = dense
    a.i0 = C
    a.i1 = 0
    ex.launch[OP_ONE_HOT](a, N * C)
    var y = tape.variable(ex, dense, N, C, False)
    return nll(tape, ex, x, y, reduction)


def _as_values[E: Exec](mut tape: Tape, mut ex: E, t: Tensor, is_ids: Bool) raises -> FP:
    """The words of t as float values: an ids tensor's int32 words converted (OP_ONE_HOT's value form, one
    launch), a float tensor's words as they are."""
    var words = tape.value_of(ex, t)
    if not is_ids:
        return words
    var n = _count(tape, t)
    var dst = tape.reserve(ex, n)
    var a = Args()
    a.p0 = words
    a.p1 = dst
    a.i0 = 1
    a.i1 = 1
    ex.launch[OP_ONE_HOT](a, n)
    return dst


def sor_ctc_loss[E: Exec](
    mut tape: Tape, mut ex: E, log_probs: Tensor, targets: Tensor, targets_ids: Bool, input_lengths: Tensor,
    input_ids: Bool, target_lengths: Tensor, target_ids: Bool, blank: Int, reduction: Int, zero_infinity: Bool,
) raises -> Tensor:
    """torch.nn.functional.ctc_loss: log_probs (T, N, C), padded targets (N, S), input_lengths and
    target_lengths (N each). The log-probabilities are permuted to Sequential's (N, T, C) (one copy, its
    gradient permuted back), the targets packed as the fit packs them, then nn_ctc.ctc_loss."""
    var name = String("ctc_loss")
    var s = tape.shape_of(log_probs)
    if s.ndim != 3:
        _refuse(name, "log_probs must be (T, N, C)")
    var T = s.d[0]
    var N = s.d[1]
    var C = s.d[2]
    if C < 2:
        _refuse(name, "log_probs need at least two classes")
    if blank < 0 or blank >= C:
        _refuse(name, "blank must be a class index in [0, " + String(C) + ")")
    var ts = tape.shape_of(targets)
    if ts.ndim != 2 or ts.d[0] != N:
        _refuse(name, "targets must be (N, S) padded rows (the 1-D concatenated form is not supported)")
    var S = ts.d[1]
    if _count(tape, input_lengths) != N or _count(tape, target_lengths) != N:
        _refuse(name, "input_lengths and target_lengths must hold one length per sample (" + String(N) + ")")
    var labels = _as_values(tape, ex, targets, targets_ids)
    var il = _as_values(tape, ex, input_lengths, input_ids)
    var tl = _as_values(tape, ex, target_lengths, target_ids)
    var W = S + CTC_LABELS
    var packed = tape.reserve(ex, N * W)
    ctc_pack_targets(ex, packed, labels, tl, Optional[FP](il), N, S, T)
    # fable-review-2: the admission runs where the targets are (one flag launch, the embedding's first-bad
    # search, two words back); the packed rows are not downloaded and no host loop walks them
    var cw = embedding_check_words(N * W)
    var flags = tape.reserve(ex, N * W)
    var scratch_a = tape.reserve(ex, cw)
    var scratch_b = tape.reserve(ex, cw)
    ctc_refuse_rows(ex, packed, N, W, T, C, blank, flags, scratch_a, scratch_b, name)
    var rows = tape.variable(ex, packed, N, W, False)
    var perm: List[Int] = [1, 0, 2]
    var lp = tape.permute(ex, log_probs, perm)
    comptime if (SOL_SABOTAGE_CTC_REDUCTION and E.IS_DEVICE) or (SOL_SABOTAGE_CTC_REDUCTION_HOST and not E.IS_DEVICE):
        return ctc_loss(tape, ex, lp, rows, T, C, blank, 1 - reduction, zero_infinity)  # SABOTAGE: mean <-> sum
    else:
        return ctc_loss(tape, ex, lp, rows, T, C, blank, reduction, zero_infinity)


# ---------------------------------------------------------------- state-space scans
def sor_selective_scan[E: Exec](
    mut tape: Tape, mut ex: E, u: Tensor, delta: Tensor, a_log: Tensor, bm: Tensor, cm: Tensor, d_skip: Tensor,
) raises -> Tensor:
    comptime if (SOL_SABOTAGE_SSM_SCALE and E.IS_DEVICE) or (SOL_SABOTAGE_SSM_SCALE_HOST and not E.IS_DEVICE):
        return tape.scale(ex, graph_selective_scan(tape, ex, u, delta, bm, cm, a_log, d_skip),
                          Float32(2.0))  # SABOTAGE: twice the output
    else:
        return graph_selective_scan(tape, ex, u, delta, bm, cm, a_log, d_skip)


def sor_ssd_scan[E: Exec](
    mut tape: Tape, mut ex: E, xbc: Tensor, dt: Tensor, a_log: Tensor, d_skip: Tensor,
) raises -> Tensor:
    return graph_ssd_scan(tape, ex, xbc, dt, a_log, d_skip)


def sor_trapezoid_scan[E: Exec](
    mut tape: Tape, mut ex: E, v: Tensor, k: Tensor, q: Tensor, dt: Tensor, a: Tensor, trap: Tensor, d_skip: Tensor,
    heads: Int,
) raises -> Tensor:
    return graph_trapezoid_scan(tape, ex, v, k, q, dt, a, trap, d_skip, heads)
