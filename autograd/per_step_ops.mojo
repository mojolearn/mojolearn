# SPDX-License-Identifier: Apache-2.0
# The per-step training blocks of `mojolearn.ops.Session` (lane rnn-parity-B, 2026-11-25): a masked and
# weighted loss over every step, and the padded-step zeroing of a recurrent layer given row lengths.
#
# Both are compositions of existing tape operations (loss_terms / cross_entropy_rows, multiply, sum,
# add, divide; iota, compare, where), which already run one order on the host and on every vendor, so
# no kernel is added and the bits match across vendors and the host column by construction.
#
#   OPS_WEIGHTED_LOSS (95): [prediction, target, weight], ints [loss code 24..28 (the OPS_MSE ..
#   OPS_CROSS_ENTROPY code of the unweighted loss), reduction 0 mean / 1 sum], floats [delta (huber;
#   1 for the others)]. A session reaches it only when `weight=` is given: weight=None keeps the
#   unweighted codes 24..28, launch for launch.
from std.sys import is_defined
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN
from autograd.rankn_api import rn_as_shape
from autograd.import_api import imp_binary, imp_where
from autograd.import_ops import IMP_GE
from autograd.loss_ops import LS_MSE, LS_MAE, LS_HUBER, LS_BCE_LOGITS
from autograd.losses import REDUCTION_MEAN, REDUCTION_SUM
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_SEQ_IOTA
from sequence.recurrent import fill

comptime OPS_WEIGHTED_LOSS = 95

# The unweighted loss codes the weighted op takes as its first integer (autograd/ops_api.mojo OPS_MSE ..
# OPS_CROSS_ENTROPY; repeated here so this module does not import the session).
comptime WEIGHTED_LOSS_MSE = 24
comptime WEIGHTED_LOSS_MAE = 25
comptime WEIGHTED_LOSS_HUBER = 26
comptime WEIGHTED_LOSS_BCE_LOGITS = 27
comptime WEIGHTED_LOSS_CROSS_ENTROPY = 28

# The mean's denominator is sum(weight) + 2^-126 (the smallest normal float32): a zero total weight gives a
# loss of 0 and zero gradients, never NaN. For any total above 2^-100 the added term is below half an ulp of
# the total, so it moves no result.
comptime WEIGHTED_LOSS_TINY = Float32(1.1754943508222875e-38)

# Sabotage arms (tools/identity_lanes/coverage_sabotage.tsv coverage-autograd-rnn_parity_loss_weight_gpu /
# _cpu, lane ops-per-step-weighted-embedding): the weight is dropped from the numerator only, on the device executor
# or on the host one. Off in every build; the arm's patch switches one on.
comptime RNN_PARITY_SABOTAGE_LOSS_WEIGHT = is_defined["MOJOLEARN_RNN_PARITY_SABOTAGE_LOSS_WEIGHT"]()
comptime RNN_PARITY_SABOTAGE_LOSS_WEIGHT_HOST = is_defined["MOJOLEARN_RNN_PARITY_SABOTAGE_LOSS_WEIGHT_HOST"]()


def weighted_loss_kind(code: Int) raises -> Int:
    """The loss_terms kind of an elementwise loss code (24..27)."""
    if code == WEIGHTED_LOSS_MSE:
        return LS_MSE
    if code == WEIGHTED_LOSS_MAE:
        return LS_MAE
    if code == WEIGHTED_LOSS_HUBER:
        return LS_HUBER
    if code == WEIGHTED_LOSS_BCE_LOGITS:
        return LS_BCE_LOGITS
    raise Error("ValueError: ops: a weighted loss is mse, mae, huber, bce_with_logits or cross_entropy")


def weighted_reduce[E: Exec](
    mut tape: Tape, mut ex: E, terms: Tensor, weight: Tensor, reduction: Int,
) raises -> Tensor:
    """sum(weight * terms) (reduction sum) or sum(weight * terms) / (sum(weight) + 2^-126) (mean), a (1, 1)
    tensor. `weight` has the terms' (rows, cols) or one value per row (rows, 1); it takes no gradient (a
    constant view), so the terms' gradient is weight * g / (sum(weight) + 2^-126), the exact derivative."""
    if reduction != REDUCTION_MEAN and reduction != REDUCTION_SUM:
        raise Error("ValueError: reduction must be 'mean' or 'sum'")
    var w = tape.detach(ex, weight)
    var num: Tensor
    comptime if (RNN_PARITY_SABOTAGE_LOSS_WEIGHT and E.IS_DEVICE) or (
            RNN_PARITY_SABOTAGE_LOSS_WEIGHT_HOST and not E.IS_DEVICE):
        num = tape.sum(ex, terms)  # SABOTAGE: the weight dropped from the numerator
    else:
        num = tape.sum(ex, tape.multiply(ex, terms, w))
    if reduction == REDUCTION_SUM:
        return num
    var total = tape.sum(ex, w)
    var tiny_p = tape.reserve(ex, 1)
    fill(ex, tiny_p, 1, WEIGHTED_LOSS_TINY)
    var tiny = tape.variable(ex, tiny_p, 1, 1, False)
    return tape.divide(ex, num, tape.add(ex, total, tiny))


def weighted_loss[E: Exec](
    mut tape: Tape, mut ex: E, prediction: Tensor, target: Tensor, weight: Tensor, code: Int, reduction: Int,
    delta: Float32,
) raises -> Tensor:
    """The weighted loss of `Session.mse / mae / huber / bce_with_logits / cross_entropy(..., weight=)`.

    Elementwise losses (code 24..27): the per-element terms of prediction and target (same shape, any rank),
    weighted by `weight` holding one value per element (the prediction's element count, read in its order) or
    one per row of the prediction's (rows, last axis) matrix view. cross_entropy (code 28): logits
    (..., classes), codes one per position of the leading axes, `weight` one value per position. The
    reduction as weighted_reduce."""
    var rows = tape.rows(prediction)
    var cols = tape.cols(prediction)
    var terms: Tensor
    var term_cols = cols
    if code == WEIGHTED_LOSS_CROSS_ENTROPY:
        if tape.shape_of(target).count() != rows:
            raise Error("ops: cross_entropy takes one class code per row of logits")
        var z = rn_as_shape(tape, ex, prediction, ShapeN.matrix(rows, cols))
        var c = rn_as_shape(tape, ex, target, ShapeN.matrix(rows, 1))
        terms = tape.cross_entropy_rows(ex, z, c)
        term_cols = 1
    else:
        var kind = weighted_loss_kind(code)
        terms = tape.loss_terms(ex, prediction, target, kind, delta)
    var wn = tape.shape_of(weight).count()
    var w: Tensor
    if wn == rows * term_cols:
        w = rn_as_shape(tape, ex, weight, ShapeN.matrix(rows, term_cols))
    elif wn == rows:
        w = rn_as_shape(tape, ex, weight, ShapeN.matrix(rows, 1))
    else:
        raise Error("ValueError: ops: weight must hold one value per element of the prediction or one per row ("
                    + String(rows * term_cols) + " or " + String(rows) + " values), got " + String(wn))
    return weighted_reduce(tape, ex, terms, w, reduction)


def rnn_zero_padded[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, lengths: FP, T: Int, B: Int) raises -> Tensor:
    """x (T, B, D), time first, with every padded step's inputs replaced by +0.0: step t of row b is padded when
    t >= lengths[b] (B float32 words on the executor). A select on the executor (iota, compare, where; the
    model's block_recurrent does the same time-second), so no padded value, not even inf or NaN, reaches a
    product; the gradient of x is zero at those steps. rnn_scan_op takes the zeroed x."""
    var pos = tape.reserve(ex, T)
    var ia = Args()
    ia.p0 = pos
    ex.launch[OP_SEQ_IOTA](ia, T)
    var col = tape.variable(ex, pos, T, 1, False)
    var lens = tape.variable(ex, lengths, 1, B, False)
    var cond = imp_binary(tape, ex, col, lens, IMP_GE)
    var c3: List[Int] = [T, B, 1]
    cond = tape.view_nd(ex, cond, ShapeN.of(c3))
    var zp = tape.reserve(ex, 1)
    fill(ex, zp, 1, Float32(0))
    var zero = tape.variable(ex, zp, 1, 1, False)
    var xs = tape.shape_of(x)
    var src = x
    if xs.ndim != 3:
        var x3: List[Int] = [T, B, tape.cols(x)]
        src = rn_as_shape(tape, ex, x, ShapeN.of(x3))
    var z = imp_where(tape, ex, cond, zero, src)
    # the (T B, D) matrix view rnn_scan_op reads (free)
    return rn_as_shape(tape, ex, z, ShapeN.matrix(T * B, tape.cols(x)))
