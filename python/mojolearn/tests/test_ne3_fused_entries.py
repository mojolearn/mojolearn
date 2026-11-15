"""Fused whole-step entries of lane ne3-g-svc-gp-neighbors: the Python
boundary cases.

SVC.score on codes compares predict's Python labels (int and str) with the
targets, and an empty NearestCentroid query keeps scikit-learn's refusal.
The fused-entry relations (OneClassSVM predict / decision sign,
NearestCentroid predict / proba / log proba, homogeneity_completeness_v_measure,
LabelPropagation rbf predict / proba and the kNN fit) are the native check
`checks/native_tests/core_ne3_fused_entries.mojo`.
"""
import numpy as np
import pytest

import mojolearn as ml


def _blobs(seed, n=96, d=5, k=3):
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % k).astype(np.int32)
    x = (rng.standard_normal((n, d)) + 2.0 * y[:, None]).astype(np.float32)
    q = (rng.standard_normal((37, d)) + 1.5).astype(np.float32)
    return x, y, q


def test_nearest_centroid_empty_query_keeps_working():
    x, y, _ = _blobs(14)
    model = ml.NearestCentroid().fit(x, y)
    empty = np.zeros((0, x.shape[1]), dtype=np.float32)
    # An empty query takes the op chain, whose refusal is scikit-learn's
    # (no fused-session result for zero rows); the model keeps working after.
    for method in ("predict", "predict_proba", "predict_log_proba"):
        with pytest.raises(ValueError, match="minimum of 1 is required"):
            getattr(model, method)(empty)
    assert np.asarray(model.predict(x[:4])).shape == (4,)
    assert np.asarray(model.predict_proba(x[:4])).shape == (4, 3)


@pytest.mark.parametrize("labels", ["int", "str"])
def test_svc_score_equals_the_accuracy_of_predict(labels):
    x, y, _ = _blobs(16, n=120)
    target = y if labels == "int" else np.array(["a", "b", "c"], dtype=object)[y].tolist()
    for decision in ("ovr", "ovo"):
        model = ml.SVC(decision_function_shape=decision).fit(x, target)
        predicted = model.predict(x)
        expected = float(np.mean(np.asarray(predicted, dtype=object) == np.asarray(target, dtype=object)))
        assert model.score(x, target) == expected


def test_svc_score_binary_and_unknown_labels():
    x, y, _ = _blobs(17, n=80, k=2)
    model = ml.SVC().fit(x, y)
    expected = float(np.mean(np.asarray(model.predict(x)) == y))
    assert model.score(x, y) == expected
    # a label the model never saw never matches
    shifted = np.where(np.arange(len(y)) % 7 == 0, 9, y).astype(np.int32)
    expected = float(np.mean(np.asarray(model.predict(x)) == shifted))
    assert model.score(x, shifted) == expected
