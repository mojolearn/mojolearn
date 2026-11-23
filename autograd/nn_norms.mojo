# SPDX-License-Identifier: Apache-2.0
"""Group and instance normalization on the tape (lane nn-norms, 2026-11-23): Sequential layers 54 and 55.

One body per operation (AGENTS.md part 2): no new kernel and no second copy of a mean / variance fold. Both
layers take their statistics from x_cnn's BatchNorm bodies (the blocked folds of autograd/rankn_conv.mojo
rankn_batch_norm_core, the tape's RN_BATCH_NORM rule) over a regrouped view of the input, because the
statistics of a group or an instance ARE batch-norm statistics of a "channel" made of that group's values:

  group_norm     x (B, C, *) viewed (1, B G, (C / G) prod(*)): each of the B G rows is one (sample, group),
                 its values contiguous in memory; batch norm of that view, no affine, no running statistics,
                 gives (x - mean) / sqrt(var + eps) of the group (biased variance, as PyTorch). The per-channel
                 affine is then weight (1, C, 1) * y + bias (1, C, 1), two broadcasting tape ops
                 (autograd/rankn_fold.mojo rankn_binary; their backward folds over the batch and the spatial
                 axes). With G == C the group norm is an instance norm and takes the path below.
  instance_norm  x (B, C, *) viewed (1, B C, prod(*)): each row one (sample, channel). The affine folds into
                 the batch norm itself: weight and bias expanded to (B, C) (autograd/import_api.mojo imp_expand,
                 one strided copy of C words to B C; its backward sums over B), so the batch norm's own apply
                 and dx launches carry it. track_running_stats (PyTorch's semantics, F.instance_norm):
                 training normalizes with the instance statistics and updates running = [mean C | var C] by
                 PyTorch's rule, the running statistics repeated over the batch (one strided copy), each
                 updated by the batch-norm variance launch (bn_running_at: r = (1 - m) r + m stat, the
                 variance unbiased), then averaged back over the batch (two OP_COLSUM_DIV launches, one
                 ascending fold over the B rows); predict and validation normalize with the running
                 statistics (the batch norm's eval form over the (B, C, *) view, weight and bias as they are).
                 Without track_running_stats every call uses the instance statistics.

Launches (forward, backward), B samples:
  group_norm G < C   forward 6 (batch-norm prep, block sums, mean, block squares, variance, apply) + 2 affine
                     (multiply, add); backward 3 (block reductions, totals, dx) + the two broadcast VJPs.
                     Without affine: 6 and 3.
  instance_norm      forward 6 + 2 expands with affine; + 3 (repeat, two folds) when training with tracked
  (and G == C)       running statistics; eval with tracked statistics 2 (prep, apply). Backward 3 + the two
                     expand folds with affine.
Nothing is resolved, allocated or launched per element or per sample: the descriptors and buffers are the
tape's, once per forward; every option is a description argument read once.
"""
from std.collections import Array
from std.sys.compile import is_defined
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN, RN_MAX_RANK, rn_desc_view
from autograd.rankn_api import rn_shape
from autograd.rankn_fold import rankn_binary
from autograd.rankn_ops import rn_copy_strided
from autograd.rankn_conv import rankn_batch_norm_core
from autograd.import_api import imp_expand
from autograd.ops import ADD, MULTIPLY
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_COLSUM_DIV

# Sabotage arms (test defines, never in a product build): checks/native_tests/nn_norms.mojo must fail with either.
#: each channel normalized alone instead of with its group (group_norm with 1 < groups < channels)
comptime NN_NORMS_SABOTAGE_GROUP = is_defined["MOJOLEARN_NN_NORMS_SABOTAGE_GROUP"]()
#: the tracked running statistics take the first sample's update instead of the batch's average
comptime NN_NORMS_SABOTAGE_RUNNING = is_defined["MOJOLEARN_NN_NORMS_SABOTAGE_RUNNING"]()


def _sh3(a: Int, b: Int, c: Int) raises -> ShapeN:
    var d: List[Int] = [a, b, c]
    return rn_shape(d, "norm")


def _instance_core[E: Exec](
    mut tape: Tape, mut ex: E, h: Tensor, B: Int, C: Int, rest: Int, weight: Optional[Tensor],
    bias: Optional[Tensor], running: Optional[FP], momentum: Float32, eps: Float32, train: Bool,
) raises -> Tensor:
    """Per (sample, channel) normalization of h (B, C, rest elements each) with the per-channel affine folded in;
    the running statistics as instance_norm (module docstring)."""
    var s = tape.shape_of(h)
    if running and not train:
        # tracked statistics, eval: the batch norm's eval form, one statistic per channel
        var x3 = tape.view_nd(ex, h, _sh3(B, C, rest))
        var y = rankn_batch_norm_core(tape, ex, x3, weight, bias, running, eps, momentum, False)
        return tape.view_nd(ex, y, s)
    var xi = tape.view_nd(ex, h, _sh3(1, B * C, rest))
    var g: Optional[Tensor] = None
    var b: Optional[Tensor] = None
    if weight:
        var bc: List[Int] = [B, C]
        g = imp_expand(tape, ex, weight.value(), bc)
        b = imp_expand(tape, ex, bias.value(), bc)
    if not running:
        var y = rankn_batch_norm_core(tape, ex, xi, g, b, None, eps, momentum, True)
        return tape.view_nd(ex, y, s)
    # tracked, training: running [mean C | var C] repeated over the batch as [mean (B, C) | var (B, C)]
    var run = running.value()
    var rep = tape.reserve(ex, 2 * B * C)
    var st = Array[Int, RN_MAX_RANK](fill=0)
    st[0] = C
    st[1] = 0
    st[2] = 1
    var words = List[Int]()
    rn_desc_view(words, _sh3(2, B, C), st)
    var desc = tape.descriptor(ex, words)
    rn_copy_strided(ex, rep, run, 0, desc, 0, 2 * B * C)
    var y = rankn_batch_norm_core(tape, ex, xi, g, b, Optional(rep), eps, momentum, True)
    # PyTorch's running_mean.copy_(running_mean_.view(B, C).mean(0)), and the variance: the mean over the B rows,
    # one ascending fold per channel (op_colsum, then the division)
    for k in range(2):  # small-loop(2: the mean and the variance halves)
        var a = Args()
        a.p0 = rep + k * B * C
        a.p1 = run + k * C
        a.i0 = B
        a.i2 = C
        a.i3 = 0
        a.f0 = Float32(B)
        comptime if NN_NORMS_SABOTAGE_RUNNING:
            a.i0 = 1
            a.f0 = Float32(1)
        ex.launch[OP_COLSUM_DIV](a, C)
    return tape.view_nd(ex, y, s)


def graph_group_norm[E: Exec](
    mut tape: Tape, mut ex: E, h: Tensor, B: Int, C: Int, rest: Int, groups: Int, weight: Optional[Tensor],
    bias: Optional[Tensor], eps: Float32,
) raises -> Tensor:
    """PyTorch's group_norm of h ((B,) + (C, *), rest = prod(*)): every (sample, group) of C / groups channels is
    normalized with its own mean and biased variance, then y * weight[c] + bias[c] (weight and bias (1, C), or
    None for affine=False). The same words in training and inference (no running statistics)."""
    if groups < 1 or C % groups != 0:
        raise Error("group_norm: num_groups must divide num_channels")
    var per_channel = groups == C
    comptime if NN_NORMS_SABOTAGE_GROUP:
        per_channel = True
    if per_channel:
        return _instance_core(tape, ex, h, B, C, rest, weight, bias, None, Float32(0), eps, True)
    var s = tape.shape_of(h)
    var xg = tape.view_nd(ex, h, _sh3(1, B * groups, (C // groups) * rest))
    var y = rankn_batch_norm_core(tape, ex, xg, None, None, None, eps, Float32(0), True)
    if not weight:
        return tape.view_nd(ex, y, s)
    var y3 = tape.view_nd(ex, y, _sh3(B, C, rest))
    var w3 = tape.view_nd(ex, weight.value(), _sh3(1, C, 1))
    var b3 = tape.view_nd(ex, bias.value(), _sh3(1, C, 1))
    y3 = rankn_binary(tape, ex, y3, w3, MULTIPLY)
    y3 = rankn_binary(tape, ex, y3, b3, ADD)
    return tape.view_nd(ex, y3, s)


def graph_instance_norm[E: Exec](
    mut tape: Tape, mut ex: E, h: Tensor, B: Int, C: Int, rest: Int, weight: Optional[Tensor],
    bias: Optional[Tensor], running: Optional[FP], momentum: Float32, eps: Float32, train: Bool,
) raises -> Tensor:
    """PyTorch's instance_norm of h ((B,) + (C, *)): each (sample, channel) normalized over its rest = prod(*)
    values (biased variance), then the per-channel affine (weight and bias (1, C), or None). `running` the
    tracked statistics [mean C | var C] (executor words, updated in place when `train`), or None for
    track_running_stats=False."""
    if rest < 2 and (train or not running):
        raise Error("instance_norm: expected more than 1 spatial element")
    return _instance_core(tape, ex, h, B, C, rest, weight, bias, running, momentum, eps, train)
