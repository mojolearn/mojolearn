# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host metrics binding's input checks, typed and Python free.

Moved word for word from `bindings/_mojolearn_metrics_host.mojo` (the
thin-shell rule: a shell unpacks arguments, calls typed code and copies the
result back; every scan over rows lives here). Same checks, same order,
same error sentences.
"""
from std.math import isfinite
from core.host_finite_scan import host_all_finite_f32, host_all_finite_list
from metrics.label_pair_admission import metrics_check_pair

from metrics.host.classification_oracle import (
    MAX_CONFUSION_CLASSES,
    MAX_PRF_CLASSES,
    host_binary_ranking,
)


def require_finite_f32(values: List[Float32], message: String) raises:
    """Refuse with `message` when any value is non-finite."""
    # parallel: the one host finite scan (was a serial isfinite loop)
    if not host_all_finite_list(values):
        raise Error(message)


def regression_inputs_finite(
    y: MutPointer[Float32, MutUntrackedOrigin],
    prediction: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
) raises:
    # parallel: the one host finite scan over each input (was a serial isfinite loop over both)
    if not host_all_finite_f32(y, n) or not host_all_finite_f32(prediction, n):
        raise Error("regression_error: inputs must be finite Float32")


def binary_ranking_checked(
    y: List[Int32], scores: List[Float32], n: Int, curve: Bool,
) raises -> Tuple[List[Float32], Int]:
    """`binary_ranking_host[curve]`, `metrics/estimator.mojo`: the length,
    label and finiteness checks, the both-classes refusal for the AUC."""
    if n <= 0 or n > 2147483647 or len(y) < n or len(scores) < n:
        raise Error("binary ranking: invalid input length")
    var has_zero = False
    var has_one = False
    for i in range(n):
        if y[i] == 0:
            has_zero = True
        elif y[i] == 1:
            has_one = True
        else:
            raise Error("binary ranking: labels must encode 0 or 1")
        if not isfinite(scores[i]):
            raise Error("binary ranking: scores must be finite")
    if not curve:
        if not has_zero or not has_one:
            raise Error("roc_auc_score: both classes required")
    return host_binary_ranking(y, scores, n, curve)


def log_loss_inputs_check(
    y: List[Int32], probability: List[Float32], n: Int, k: Int
) raises:
    for i in range(n):
        if Int(y[i]) < 0 or Int(y[i]) >= k:
            raise Error("log_loss: encoded label out of range")
    for i in range(n * k):
        if not isfinite(probability[i]) or probability[i] < 0 or probability[i] > 1:
            raise Error("log_loss: probabilities must be finite and within [0,1]")


def check_classification(y: List[Int32], p: List[Int32], n: Int, k: Int, matrix: Bool) raises:
    """`_check_classification[matrix]`, `metrics/estimator.mojo`."""
    metrics_check_pair(y, p, n)
    var cap = MAX_CONFUSION_CLASSES if matrix else MAX_PRF_CLASSES
    if n > 2147483647 or k <= 0 or k > cap:
        raise Error("classification metrics: count or class allocation bound exceeded")
    var minimum = -1 if matrix else 0
    for i in range(n):
        if Int(y[i]) < minimum or Int(y[i]) >= k or Int(p[i]) < minimum or Int(p[i]) >= k:
            raise Error("classification metrics: encoded label out of range")


def check_classification_ptr(
    y: MutPointer[Int32, MutUntrackedOrigin],
    p: MutPointer[Int32, MutUntrackedOrigin],
    n: Int, k: Int, matrix: Bool,
) raises:
    if n <= 0:
        raise Error("metrics: n must be positive, got " + String(n))
    var cap = MAX_CONFUSION_CLASSES if matrix else MAX_PRF_CLASSES
    if n > 2147483647 or k <= 0 or k > cap:
        raise Error("classification metrics: count or class allocation bound exceeded")
    var minimum = -1 if matrix else 0
    for i in range(n):
        if Int(y.unsafe_load(i)) < minimum or Int(y.unsafe_load(i)) >= k or Int(p.unsafe_load(i)) < minimum or Int(p.unsafe_load(i)) >= k:
            raise Error("classification metrics: encoded label out of range")
