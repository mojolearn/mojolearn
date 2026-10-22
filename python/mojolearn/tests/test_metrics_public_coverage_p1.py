# SPDX-License-Identifier: Apache-2.0
"""Real-path and refusal tests for the public metrics changed since 0.8.55.

Every metric is called through ``mojolearn.metrics`` on small data and checked
against scikit-learn; every metric refuses inputs whose lengths disagree, and
the float metrics refuse float64 scores (callers cast explicitly).
"""
import warnings

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn import metrics as reference
from sklearn.metrics import cluster as reference_cluster

import mojolearn.metrics as mt
from mojolearn import _expansion_metrics

_rng = np.random.default_rng(0)
_N = 40
YT = _rng.integers(0, 3, _N).astype(np.int32)
YP = np.where(_rng.random(_N) < 0.7, YT, _rng.integers(0, 3, _N)).astype(np.int32)
BT = (_rng.random(_N) < 0.5).astype(np.int32)
BT[:2] = [0, 1]
BP = np.where(_rng.random(_N) < 0.8, BT, 1 - BT).astype(np.int32)
S = (BT + _rng.normal(0, 0.7, _N)).astype(np.float32)
P = _rng.random((_N, 3))
P = (P / P.sum(1, keepdims=True)).astype(np.float32)
PR = (1 / (1 + np.exp(-S))).astype(np.float32)
_rt = _rng.random(_N) * 3 + 0.5
RP = (np.abs(_rt + _rng.normal(0, 0.2, _N)) + 0.1).astype(np.float32)
RT = _rt.astype(np.float32)
X = _rng.normal(size=(_N, 3))
X[YT == 1] += 3
ML = (_rng.random((_N, 4)) < 0.4).astype(np.int32)
ML[:, 0] |= 1 - ML.max(1)
MS = _rng.random((_N, 4)).astype(np.float32)
REL = _rng.random((5, 6)).astype(np.float32)
RS = _rng.random((5, 6)).astype(np.float32)

LABELS = (YT, YP)
BINARY = (BT, BP)
SCORES = (BT, S)
PROBA = (BT, PR)
REGRESSION = (RT, RP)
CLUSTERS = (X, YT)
RANKING = (ML, MS)
GAINS = (REL, RS)

CASES = {
    "adjusted_mutual_info_score": LABELS,
    "average_precision_score": SCORES,
    "balanced_accuracy_score": LABELS,
    "brier_score_loss": PROBA,
    "calinski_harabasz_score": CLUSTERS,
    "class_likelihood_ratios": BINARY,
    "classification_report": LABELS,
    "cohen_kappa_score": LABELS,
    "contingency_matrix": LABELS,
    "coverage_error": RANKING,
    "d2_absolute_error_score": REGRESSION,
    "d2_brier_score": PROBA,
    "d2_log_loss_score": (YT, P),
    "d2_pinball_score": REGRESSION,
    "d2_tweedie_score": REGRESSION,
    "davies_bouldin_score": CLUSTERS,
    "dcg_score": GAINS,
    "det_curve": SCORES,
    "explained_variance_score": REGRESSION,
    "fbeta_score": BINARY,
    "hamming_loss": LABELS,
    "hinge_loss": SCORES,
    "jaccard_score": BINARY,
    "label_ranking_average_precision_score": RANKING,
    "label_ranking_loss": RANKING,
    "matthews_corrcoef": LABELS,
    "max_error": REGRESSION,
    "mean_absolute_percentage_error": REGRESSION,
    "mean_gamma_deviance": REGRESSION,
    "mean_pinball_loss": REGRESSION,
    "mean_poisson_deviance": REGRESSION,
    "mean_squared_log_error": REGRESSION,
    "mean_tweedie_deviance": REGRESSION,
    "median_absolute_error": REGRESSION,
    "multilabel_confusion_matrix": LABELS,
    "ndcg_score": GAINS,
    "normalized_mutual_info_score": LABELS,
    "pair_confusion_matrix": LABELS,
    "precision_recall_fscore_support": LABELS,
    "roc_curve": SCORES,
    "root_mean_squared_log_error": REGRESSION,
    "top_k_accuracy_score": (YT, P),
    "zero_one_loss": LABELS,
}
KWARGS = {"fbeta_score": {"beta": 0.5}, "precision_recall_fscore_support": {"average": "macro"}}
# Metrics whose score or target argument must be float32.
FLOAT_SCORE = sorted(n for n, a in CASES.items() if a[1].dtype == np.float32)


def _reference(name):
    return getattr(reference, name, None) or getattr(reference_cluster, name)


def _close(ours, theirs):
    if isinstance(theirs, str):
        assert isinstance(ours, str) and ours.strip()
        return
    if isinstance(theirs, tuple):
        assert len(ours) >= len([t for t in theirs if t is not None])
        for a, b in zip(ours, theirs):
            if b is not None:
                _close(a, b)
        return
    np.testing.assert_allclose(np.asarray(ours, np.float64), np.asarray(theirs, np.float64),
                               rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("name", sorted(CASES))
def test_metric_matches_reference_on_public_path(name):
    fn = getattr(mt, name)
    assert fn is getattr(_expansion_metrics, name)
    args, kw = CASES[name], KWARGS.get(name, {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _close(fn(*args, **kw), _reference(name)(*args, **kw))


@pytest.mark.parametrize("name", sorted(CASES))
def test_metric_refuses_length_mismatch(name):
    fn = getattr(mt, name)
    assert fn is getattr(_expansion_metrics, name)
    first, second = CASES[name]
    with pytest.raises(ValueError):
        fn(first, second[:-1], **KWARGS.get(name, {}))


@pytest.mark.parametrize("name", FLOAT_SCORE)
def test_metric_refuses_float64_scores(name):
    fn = getattr(mt, name)
    assert fn is getattr(_expansion_metrics, name)
    first, second = CASES[name]
    if first.dtype == np.float32:
        first = first.astype(np.float64)
    with pytest.raises(TypeError, match="float32"):
        fn(first, second.astype(np.float64), **KWARGS.get(name, {}))
