"""Fused whole-step entries of lane ne3-g-svc-gp-neighbors.

Each public method below now runs its device steps in one native session
(OneClassSVM.predict, NearestCentroid predict / predict_proba /
predict_log_proba, LabelPropagation rbf predict / predict_proba,
homogeneity_completeness_v_measure, SVC.score on codes, the host column's
resident LabelPropagation kNN loop). The fused entries run the same kernels
in the same order as the per-op chains, so each result must equal, bit for
bit, the value the public surface gives for the same quantity through the
methods that were not fused.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import metrics


def _blobs(seed, n=96, d=5, k=3):
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % k).astype(np.int32)
    x = (rng.standard_normal((n, d)) + 2.0 * y[:, None]).astype(np.float32)
    q = (rng.standard_normal((37, d)) + 1.5).astype(np.float32)
    return x, y, q


def test_ocsvm_predict_is_the_sign_of_decision_function():
    x, _, q = _blobs(11)
    for kernel in ("rbf", "linear", "poly", "sigmoid"):
        model = ml.OneClassSVM(kernel=kernel, nu=0.3).fit(x)
        decision = np.asarray(model.decision_function(q))
        labels = np.asarray(model.predict(q))
        assert labels.dtype == np.int64
        np.testing.assert_array_equal(labels, np.where(decision > 0, 1, -1))
        np.testing.assert_array_equal(np.asarray(model.fit_predict(x)),
                                      np.where(np.asarray(model.decision_function(x)) > 0, 1, -1))


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_nearest_centroid_uniform_predict_is_the_nearest_centroid(metric):
    x, y, q = _blobs(12)
    model = ml.NearestCentroid(metric=metric).fit(x, y)
    centers = np.asarray(model.centroids_, dtype=np.float64)
    diff = q.astype(np.float64)[:, None, :] - centers[None, :, :]
    dist = (diff ** 2).sum(-1) if metric == "euclidean" else np.abs(diff).sum(-1)
    np.testing.assert_array_equal(np.asarray(model.predict(q)), np.asarray(model.classes_)[dist.argmin(1)])


def test_nearest_centroid_decision_routes_agree_bitwise():
    x, y, q = _blobs(13)
    y = np.where(np.arange(len(y)) % 5 == 0, 0, y).astype(np.int32)  # non-uniform class counts
    # Empirical priors: predict is then the argmax of decision_function. With
    # the default uniform priors predict is the nearest centroid instead
    # (scikit-learn's rule), which need not be that argmax.
    model = ml.NearestCentroid(priors="empirical").fit(x, y)
    scores = np.asarray(model.decision_function(q))
    np.testing.assert_array_equal(np.asarray(model.predict(q)), np.asarray(model.classes_)[scores.argmax(1)])
    proba = np.asarray(model.predict_proba(q), dtype=np.float64)
    expected = np.exp(scores.astype(np.float64) - scores.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(proba, expected, rtol=2e-5, atol=2e-6)
    log_proba = np.asarray(model.predict_log_proba(q), dtype=np.float64)
    np.testing.assert_allclose(log_proba, np.log(expected), rtol=2e-4, atol=2e-5)
    # repeated calls give the same bits
    np.testing.assert_array_equal(np.asarray(model.predict_proba(q)), np.asarray(model.predict_proba(q)))


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


def test_homogeneity_completeness_v_measure_equals_the_three_functions():
    rng = np.random.default_rng(15)
    a = rng.integers(0, 4, 200).astype(np.int32)
    b = rng.integers(0, 6, 200).astype(np.int32)
    for beta in (1.0, 0.5, 2.0):
        h, c, v = metrics.homogeneity_completeness_v_measure(a, b, beta=beta)
        assert h == metrics.homogeneity_score(a, b)
        assert c == metrics.completeness_score(a, b)
        assert v == metrics.v_measure_score(a, b, beta=beta)
    same = metrics.homogeneity_completeness_v_measure(a, a)
    # The fused pass equals the three functions bit for bit; a perfect
    # labelling scores one up to the last float64 ulps (mutual information
    # over entropy, both summed in float64).
    assert same == (metrics.homogeneity_score(a, a), metrics.completeness_score(a, a),
                    metrics.v_measure_score(a, a))
    np.testing.assert_allclose(same, (1.0, 1.0, 1.0), rtol=0, atol=1e-12)


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


@pytest.mark.parametrize("cls", ["LabelPropagation", "LabelSpreading"])
def test_label_propagation_rbf_predict_is_the_argmax_of_predict_proba(cls):
    x, y, q = _blobs(18)
    y = y.copy()
    y[::4] = -1  # every class keeps labelled rows (y is arange % 3, so ::3 unlabels all of class 0)
    model = getattr(ml, cls)(kernel="rbf", gamma=0.5).fit(x, y)
    proba = np.asarray(model.predict_proba(q))
    assert proba.dtype == np.float32 and proba.shape == (len(q), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(model.predict(q)), np.asarray(model.classes_)[proba.argmax(1)])


@pytest.mark.parametrize("cls", ["LabelPropagation", "LabelSpreading"])
def test_label_propagation_knn_fit_is_deterministic_and_consistent(cls):
    x, y, _ = _blobs(19)
    y = y.copy()
    y[::2] = -1
    first = getattr(ml, cls)(kernel="knn", n_neighbors=5, max_iter=200).fit(x, y)
    second = getattr(ml, cls)(kernel="knn", n_neighbors=5, max_iter=200).fit(x, y)
    np.testing.assert_array_equal(np.asarray(first.label_distributions_), np.asarray(second.label_distributions_))
    assert first.n_iter_ == second.n_iter_
    dist = np.asarray(first.label_distributions_)
    np.testing.assert_array_equal(np.asarray(first.transduction_), np.asarray(first.classes_)[dist.argmax(1)])
