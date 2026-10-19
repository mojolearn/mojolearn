# SPDX-License-Identifier: Apache-2.0
# One loss set for the tape and the trainers. Every loss is per-element (or
# per-row) terms from autograd/loss_ops.mojo followed by the tape's blocked
# fold (autograd/fold.mojo): the tape path reduces with Tape.mean / Tape.sum,
# the trainer path (loss_value_into) calls the same fold with the same
# extent and factor, so both give the same bits on Apple, NVIDIA, AMD and
# the host.
from autograd.tape import Tape, Tensor
from autograd.fold import blocked_fold, fold_partials_bound
from autograd.ops import AXIS_ALL
from autograd.loss_ops import (
    OP_LS_ELEM_FWD, OP_LS_CE_FWD, LS_MSE, LS_MAE, LS_HUBER, LS_BCE_LOGITS, LS_LOG_PROB,
    LS_BINARY_LOG_PROB, LS_CROSS_ENTROPY,
)
from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from checks.numerics import identical_div

comptime REDUCTION_MEAN = 0
comptime REDUCTION_SUM = 1


def reduction_factor(count: Int, reduction: Int) raises -> Float32:
    """The fold factor of a reduction over `count` terms: 1 / count for the
    mean (the factor Tape.mean uses), 1 for the sum."""
    if reduction == REDUCTION_MEAN:
        return identical_div(Float32(1), Float32(count))
    if reduction == REDUCTION_SUM:
        return Float32(1)
    raise Error("reduction must be REDUCTION_MEAN or REDUCTION_SUM")


def _reduce[E: Exec](mut tape: Tape, mut ex: E, terms: Tensor, reduction: Int) raises -> Tensor:
    if reduction == REDUCTION_MEAN:
        return tape.mean(ex, terms)
    if reduction == REDUCTION_SUM:
        return tape.sum(ex, terms)
    raise Error("reduction must be REDUCTION_MEAN or REDUCTION_SUM")


def mse[E: Exec](
    mut tape: Tape, mut ex: E, prediction: Tensor, target: Tensor, reduction: Int = REDUCTION_MEAN,
) raises -> Tensor:
    """Squared error (prediction - target)^2, averaged or summed.

    Both inputs have the same shape; both receive gradients. For same-shape
    inputs the value and the gradients equal Tape.mse_loss bit for bit.
    """
    return _reduce(tape, ex, tape.loss_terms(ex, prediction, target, LS_MSE), reduction)


def mae[E: Exec](
    mut tape: Tape, mut ex: E, prediction: Tensor, target: Tensor, reduction: Int = REDUCTION_MEAN,
) raises -> Tensor:
    """Absolute error |prediction - target|, averaged or summed. The
    gradient at a zero difference is zero."""
    return _reduce(tape, ex, tape.loss_terms(ex, prediction, target, LS_MAE), reduction)


def huber[E: Exec](
    mut tape: Tape, mut ex: E, prediction: Tensor, target: Tensor, delta: Float32 = 1,
    reduction: Int = REDUCTION_MEAN,
) raises -> Tensor:
    """Huber loss: 0.5 d^2 where |d| < delta, else delta (|d| - delta / 2),
    d = prediction - target; averaged or summed."""
    return _reduce(tape, ex, tape.loss_terms(ex, prediction, target, LS_HUBER, delta), reduction)


def binary_cross_entropy_with_logits[E: Exec](
    mut tape: Tape, mut ex: E, logits: Tensor, target: Tensor, reduction: Int = REDUCTION_MEAN,
) raises -> Tensor:
    """Binary cross-entropy of sigmoid(logits) against targets in [0, 1],
    computed stably from the logits: max(x, 0) - x y + log(1 + exp(-|x|))."""
    return _reduce(tape, ex, tape.loss_terms(ex, logits, target, LS_BCE_LOGITS), reduction)


def cross_entropy[E: Exec](
    mut tape: Tape, mut ex: E, logits: Tensor, codes: Tensor, reduction: Int = REDUCTION_MEAN,
) raises -> Tensor:
    """Softmax cross-entropy of logits (rows, classes) against class codes
    (rows, 1, the class index as a float value), averaged or summed over
    rows. Only the logits receive a gradient."""
    return _reduce(tape, ex, tape.cross_entropy_rows(ex, logits, codes), reduction)


def cross_entropy_dense[E: Exec](mut tape: Tape, mut ex: E, logits: Tensor, targets: Tensor) raises -> Tensor:
    """Mean cross-entropy against dense class weights of the logits' shape
    (one row per example); both inputs receive gradients."""
    return tape.cross_entropy(ex, logits, targets)


@fieldwise_init
struct LossScratch(ImplicitlyCopyable, Movable):
    """Executor storage for loss_value_into: the terms of up to `capacity`
    elements (or rows) and the fold partials that cover them."""
    var terms: FP
    var partials: FP
    var capacity: Int


def loss_scratch[E: Exec](mut ex: E, max_terms: Int) raises -> LossScratch:
    """Allocate scratch for losses of at most `max_terms` terms."""
    if max_terms <= 0:
        raise Error("loss scratch needs a positive capacity")
    return LossScratch(ex.alloc(max_terms), ex.alloc(fold_partials_bound(max_terms)), max_terms)


def release_loss_scratch[E: Exec](mut ex: E, scratch: LossScratch) raises:
    ex.free(scratch.terms)
    ex.free(scratch.partials)


def loss_value_into[E: Exec](
    mut ex: E, kind: Int, prediction: FP, target: FP, rows: Int, cols: Int, reduction: Int,
    dst: FP, scratch: LossScratch, delta: Float32 = 1,
) raises:
    """dst[0] = the loss of `kind` over executor buffers, with no tape.

    kind is LS_MSE, LS_MAE, LS_HUBER, LS_BCE_LOGITS, LS_LOG_PROB or
    LS_BINARY_LOG_PROB (rows x cols prediction and target), or
    LS_CROSS_ENTROPY (rows x cols logits, `target` the rows class codes,
    reduced over rows). The value equals the tape loss of the same kind and
    reduction bit for bit: the same term body and the same fold.
    """
    if rows <= 0 or cols <= 0:
        raise Error("loss dimensions must be positive")
    if kind == LS_HUBER and not (delta > 0):
        raise Error("Huber delta must be positive")
    var count = rows if kind == LS_CROSS_ENTROPY else rows * cols
    if count > scratch.capacity:
        raise Error("loss scratch is smaller than the loss")
    var a = Args()
    a.p0 = prediction
    a.p1 = target
    a.p2 = scratch.terms
    if kind == LS_CROSS_ENTROPY:
        a.i1 = cols
        ex.launch[OP_LS_CE_FWD](a, rows)
        blocked_fold(ex, dst, scratch.terms, rows, 1, AXIS_ALL, reduction_factor(count, reduction), scratch.partials)
        return
    if kind < LS_MSE or kind > LS_BINARY_LOG_PROB:
        raise Error("unknown loss kind")
    a.i0 = kind
    a.f0 = delta
    ex.launch[OP_LS_ELEM_FWD](a, count)
    blocked_fold(ex, dst, scratch.terms, rows, cols, AXIS_ALL, reduction_factor(count, reduction), scratch.partials)
