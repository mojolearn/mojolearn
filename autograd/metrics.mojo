# SPDX-License-Identifier: Apache-2.0
# Forward-only metrics on an executor with the tape's fixed folds. Each call
# takes executor buffers, runs parallel launches (terms from
# autograd/loss_ops.mojo, sums through autograd/fold.mojo) and downloads ONE
# value: no host loop over rows, the same bits on every executor.
#
# metrics/impl/log_loss.mojo and metrics/impl/binary_ranking.mojo are
# DeviceContext kernels with shared-memory block trees and the gbdt radix
# sort, so they cannot run on HostExec through the Exec mechanism. The terms
# here are theirs (the same clipping and -log for the log loss; the same
# tie-aware pair count for the AUC), with the sums in the tape's fold order.
#
# Class codes and binary labels are float values in executor buffers.
from autograd.fold import blocked_fold, fold_partials, fold_partials_bound
from autograd.ops import AXIS_ALL, ag_fold_block
from autograd.loss_ops import (
    OP_LS_METRIC_TERM, OP_LS_PACK, OP_LS_BITONIC, OP_LS_SCAN_PART, OP_LS_SCAN_OFFS,
    OP_LS_SCAN_FILL, OP_LS_AUC_TERM, OP_LS_FINISH, MT_EQUAL, MT_ARGMAX_EQUAL, MT_LOG_LOSS,
    MT_SQUARED, MT_ABSOLUTE, MT_CENTERED, FIN_SQRT, FIN_R2, FIN_AUC,
)
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, dummy_ptr
from autograd.row_ops import AUC_BLOCK_SORT, OP_ROW_BITONIC_BLOCK, ROW_SORT_BLOCK
from checks.numerics import identical_div

# Counts and indices travel as float values: exact below 2^24.
comptime METRIC_MAX_ROWS = 1 << 24


def _check_rows(n: Int) raises:
    if n <= 0 or n >= METRIC_MAX_ROWS:
        raise Error("metric needs between 1 and 2^24 - 1 rows")


def _read_word[E: Exec](mut ex: E, word: FP) raises -> Float32:
    var out = List[Float32](length=1, fill=Float32(0))
    ex.download(FP(unsafe_from_address=Int(out.unsafe_ptr())), word, 1)
    return out[0]


def _terms_fold[E: Exec](
    mut ex: E, kind: Int, p0: FP, p1: FP, n: Int, cols: Int, factor: Float32, dst: FP,
) raises:
    """dst[0] = factor * (the n metric terms of `kind` folded)."""
    var terms = ex.alloc(n)
    var partials = ex.alloc(fold_partials(n, 1, AXIS_ALL))
    var a = Args()
    a.p0 = p0
    a.p1 = p1
    a.p2 = terms
    a.i0 = kind
    a.i1 = cols
    ex.launch[OP_LS_METRIC_TERM](a, n)
    blocked_fold(ex, dst, terms, n, 1, AXIS_ALL, factor, partials)
    ex.free(terms)
    ex.free(partials)


struct MetricScratch(Movable):
    """Reusable executor storage for the metrics' terms and fold partials,
    so a caller that scores every step allocates once. Grows on demand;
    `release` returns the storage to the executor that `reserve`d it."""

    var terms: FP
    var terms_n: Int
    var partials: FP
    var partials_n: Int

    def __init__(out self):
        self.terms = dummy_ptr()
        self.terms_n = 0
        self.partials = dummy_ptr()
        self.partials_n = 0

    def reserve[E: Exec](mut self, mut ex: E, n: Int) raises:
        """Storage for a metric over n rows."""
        if n > self.terms_n:
            if self.terms_n > 0:
                ex.free(self.terms)
            self.terms = ex.alloc(n)
            self.terms_n = n
        var need = fold_partials_bound(n)
        if need > self.partials_n:
            if self.partials_n > 0:
                ex.free(self.partials)
            self.partials = ex.alloc(need)
            self.partials_n = need

    def release[E: Exec](mut self, mut ex: E) raises:
        if self.terms_n > 0:
            ex.free(self.terms)
        if self.partials_n > 0:
            ex.free(self.partials)
        self.terms = dummy_ptr()
        self.partials = dummy_ptr()
        self.terms_n = 0
        self.partials_n = 0


def _mean_of_terms_into[E: Exec](
    mut ex: E, kind: Int, p0: FP, p1: FP, n: Int, cols: Int, dst: FP, mut scratch: MetricScratch,
) raises:
    """dst[0] = the mean of the n metric terms of `kind`: the launches and
    the fold of `_mean_of_terms`, so the same bits, with no download and no
    allocation once `scratch` holds n rows."""
    _check_rows(n)
    scratch.reserve(ex, n)
    var a = Args()
    a.p0 = p0
    a.p1 = p1
    a.p2 = scratch.terms
    a.i0 = kind
    a.i1 = cols
    ex.launch[OP_LS_METRIC_TERM](a, n)
    blocked_fold(ex, dst, scratch.terms, n, 1, AXIS_ALL, identical_div(Float32(1), Float32(n)), scratch.partials)


def accuracy_into[E: Exec](
    mut ex: E, y_true: FP, y_pred: FP, n: Int, dst: FP, mut scratch: MetricScratch,
) raises:
    """`accuracy` written into the executor word dst[0], left on the device."""
    _mean_of_terms_into(ex, MT_EQUAL, y_true, y_pred, n, 1, dst, scratch)


def accuracy_from_scores_into[E: Exec](
    mut ex: E, y_true: FP, scores: FP, n: Int, classes: Int, dst: FP, mut scratch: MetricScratch,
) raises:
    """`accuracy_from_scores` written into the executor word dst[0]."""
    if classes < 1:
        raise Error("accuracy needs at least one class column")
    _mean_of_terms_into(ex, MT_ARGMAX_EQUAL, y_true, scores, n, classes, dst, scratch)


def log_loss_into[E: Exec](
    mut ex: E, y_true: FP, probability: FP, n: Int, classes: Int, dst: FP, mut scratch: MetricScratch,
) raises:
    """`log_loss` written into the executor word dst[0]."""
    if classes < 1:
        raise Error("log_loss needs at least one probability column")
    _mean_of_terms_into(ex, MT_LOG_LOSS, y_true, probability, n, classes, dst, scratch)


def _sort_blocks[E: Exec](mut ex: E, keys: FP, labels: FP, k_first: Int, k_last: Int, block: Int, blocks: Int) raises:
    var s = Args()
    s.p0 = keys
    s.p1 = labels
    s.i0 = k_first
    s.i1 = k_last
    s.i2 = block
    ex.launch[OP_ROW_BITONIC_BLOCK](s, blocks)


def _bitonic_stage[E: Exec](mut ex: E, keys: FP, labels: FP, padded: Int, k: Int, j: Int) raises:
    var s = Args()
    s.p0 = keys
    s.p1 = labels
    s.i0 = k
    s.i1 = j
    ex.launch[OP_LS_BITONIC](s, padded // 2)


def _mean_of_terms[E: Exec](mut ex: E, kind: Int, p0: FP, p1: FP, n: Int, cols: Int) raises -> Float32:
    _check_rows(n)
    var word = ex.alloc(1)
    _terms_fold(ex, kind, p0, p1, n, cols, identical_div(Float32(1), Float32(n)), word)
    var value = _read_word(ex, word)
    ex.free(word)
    return value


def accuracy[E: Exec](mut ex: E, y_true: FP, y_pred: FP, n: Int) raises -> Float32:
    """Fraction of the n rows whose predicted class code equals the true one."""
    return _mean_of_terms(ex, MT_EQUAL, y_true, y_pred, n, 1)


def accuracy_from_scores[E: Exec](mut ex: E, y_true: FP, scores: FP, n: Int, classes: Int) raises -> Float32:
    """Accuracy of the first highest score of each row of the n x classes
    scores (probabilities or logits) against the true class codes."""
    if classes < 1:
        raise Error("accuracy needs at least one class column")
    return _mean_of_terms(ex, MT_ARGMAX_EQUAL, y_true, scores, n, classes)


def log_loss[E: Exec](mut ex: E, y_true: FP, probability: FP, n: Int, classes: Int) raises -> Float32:
    """Mean negative log-likelihood of the true class codes under the n x
    classes probabilities, each clipped to [2^-23, 1 - 2^-23]. With
    classes = 1, `probability` holds the positive class only."""
    if classes < 1:
        raise Error("log_loss needs at least one probability column")
    return _mean_of_terms(ex, MT_LOG_LOSS, y_true, probability, n, classes)


def mean_squared_error[E: Exec](mut ex: E, y_true: FP, y_pred: FP, n: Int) raises -> Float32:
    """Mean of (y_true - y_pred)^2 over n values."""
    return _mean_of_terms(ex, MT_SQUARED, y_true, y_pred, n, 1)


def mean_absolute_error[E: Exec](mut ex: E, y_true: FP, y_pred: FP, n: Int) raises -> Float32:
    """Mean of |y_true - y_pred| over n values."""
    return _mean_of_terms(ex, MT_ABSOLUTE, y_true, y_pred, n, 1)


def root_mean_squared_error[E: Exec](mut ex: E, y_true: FP, y_pred: FP, n: Int) raises -> Float32:
    """Square root of the mean squared error over n values."""
    _check_rows(n)
    var mse_word = ex.alloc(1)
    var word = ex.alloc(1)
    _terms_fold(ex, MT_SQUARED, y_true, y_pred, n, 1, identical_div(Float32(1), Float32(n)), mse_word)
    var f = Args()
    f.p0 = mse_word
    f.p2 = word
    f.i0 = FIN_SQRT
    ex.launch[OP_LS_FINISH](f, 1)
    var value = _read_word(ex, word)
    ex.free(mse_word)
    ex.free(word)
    return value


def r2_score[E: Exec](mut ex: E, y_true: FP, y_pred: FP, n: Int) raises -> Float32:
    """Coefficient of determination 1 - SS_res / SS_tot of one output over n
    values. A constant y_true gives 1 for a perfect prediction, else 0."""
    _check_rows(n)
    var mean = ex.alloc(1)
    var ssr = ex.alloc(1)
    var sst = ex.alloc(1)
    var word = ex.alloc(1)
    var partials = ex.alloc(fold_partials(n, 1, AXIS_ALL))
    blocked_fold(ex, mean, y_true, n, 1, AXIS_ALL, identical_div(Float32(1), Float32(n)), partials)
    _terms_fold(ex, MT_CENTERED, y_true, mean, n, 1, Float32(1), sst)
    _terms_fold(ex, MT_SQUARED, y_true, y_pred, n, 1, Float32(1), ssr)
    var f = Args()
    f.p0 = ssr
    f.p1 = sst
    f.p2 = word
    f.i0 = FIN_R2
    ex.launch[OP_LS_FINISH](f, 1)
    var value = _read_word(ex, word)
    ex.free(mean)
    ex.free(ssr)
    ex.free(sst)
    ex.free(word)
    ex.free(partials)
    return value


def roc_auc[E: Exec](mut ex: E, y_true: FP, scores: FP, n: Int) raises -> Float32:
    """Area under the ROC curve of binary labels (1 positive, any other
    value negative) against finite scores, ties counted as one half.

    The rows are sorted on the executor by a bitonic network (padding to a
    power of two sorts last), positives before each row are counted by a
    blocked scan, each positive row's tie group comes from binary searches,
    and its pair count over twice the negatives is summed by the blocked
    fold, then divided by the positives. Raises when a class is missing.
    """
    _check_rows(n)
    var padded = 2
    while padded < n:
        padded *= 2
    var keys = ex.alloc(padded)
    var labels = ex.alloc(padded)
    var a = Args()
    a.p0 = scores
    a.p1 = y_true
    a.p2 = keys
    a.p3 = labels
    a.i0 = n
    ex.launch[OP_LS_PACK](a, padded)
    comptime if AUC_BLOCK_SORT:
        # the stages with a partner distance below the block run inside
        # blocks of `block` elements (autograd/row_ops.mojo idea I): one
        # launch for every merge size up to the block, then per larger merge
        # size its cross-block stages and one block launch; the same network
        var block = min(padded, ROW_SORT_BLOCK)
        var nblocks = padded // block
        _sort_blocks(ex, keys, labels, 2, block, block, nblocks)
        var k = 2 * block
        while k <= padded:  # driver-loop: over sort network merge sizes
            var j = k // 2
            while j >= block:  # driver-loop: over the network's cross-block stages
                _bitonic_stage(ex, keys, labels, padded, k, j)
                j //= 2
            _sort_blocks(ex, keys, labels, k, k, block, nblocks)
            k *= 2
    else:
        var k = 2
        while k <= padded:  # driver-loop: over sort network merge sizes
            var j = k // 2
            while j > 0:  # driver-loop: over sort network stages
                _bitonic_stage(ex, keys, labels, padded, k, j)
                j //= 2
            k *= 2
    # exclusive positive counts in blocks of the fold's length rule
    var block = ag_fold_block(n)
    var blocks = (n + block - 1) // block
    var parts = ex.alloc(blocks)
    var offs = ex.alloc(blocks)
    var prefix = ex.alloc(n + 1)
    var p = Args()
    p.p0 = labels
    p.p1 = parts
    p.i0 = n
    p.i1 = block
    ex.launch[OP_LS_SCAN_PART](p, blocks)
    var o = Args()
    o.p0 = parts
    o.p1 = offs
    o.i0 = blocks
    ex.launch[OP_LS_SCAN_OFFS](o, blocks)
    var fl = Args()
    fl.p0 = labels
    fl.p1 = offs
    fl.p2 = prefix
    fl.i0 = n
    fl.i1 = block
    fl.i2 = blocks
    ex.launch[OP_LS_SCAN_FILL](fl, blocks)
    var terms = ex.alloc(n)
    var at = Args()
    at.p0 = keys
    at.p1 = labels
    at.p2 = prefix
    at.p3 = terms
    at.i0 = n
    ex.launch[OP_LS_AUC_TERM](at, n)
    var total = ex.alloc(1)
    var partials = ex.alloc(fold_partials(n, 1, AXIS_ALL))
    blocked_fold(ex, total, terms, n, 1, AXIS_ALL, Float32(1), partials)
    var word = ex.alloc(1)
    var f = Args()
    f.p0 = total
    f.p1 = prefix
    f.p2 = word
    f.i0 = FIN_AUC
    f.i1 = n
    ex.launch[OP_LS_FINISH](f, 1)
    var value = _read_word(ex, word)
    ex.free(keys)
    ex.free(labels)
    ex.free(parts)
    ex.free(offs)
    ex.free(prefix)
    ex.free(terms)
    ex.free(total)
    ex.free(partials)
    ex.free(word)
    if value < Float32(0):
        raise Error("roc_auc needs both a positive and a negative row")
    return value
