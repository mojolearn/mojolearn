# SPDX-License-Identifier: Apache-2.0
"""Scorer options lane mf-score6 (2026-11-23) moved off the search's Python units onto the typed RUN, and the
public scorer call's per-label answer.

- ``average=None``: calling the scorer returns the signed per-label array, as scikit-learn's ``_Scorer`` does; in
  a search, a per-label array of more than one value is scikit-learn's ``_score`` refusal ("scoring must return a
  number"), raised whatever the ``error_score``; one label scores its value.
- ``sample_weight`` given to ``make_scorer``: the weighted metric on each side, refused per side in the metric
  door's words when its length is not the side's.
- ``average='samples'`` on 1-D targets and a float ``pos_label`` on the binary label scorers: the metric doors'
  refusals, per side.
- A Python callable scorer is refused, naming the native alternative.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import metrics
from mojolearn import model_selection as ms


def _data():
    X = np.array([[0.0, 0.1], [0.2, 0.0], [1.0, 1.1], [1.2, 0.9], [2.0, 2.1], [2.2, 1.9],
                  [0.1, 0.3], [1.1, 1.0], [2.1, 2.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2], dtype=np.int32)
    return X, y


def _search(scoring, X, y, cv=None, error_score=np.nan):
    every = np.arange(len(X))
    est = ml.LogisticRegression(max_iter=50)
    return ms.GridSearchCV(est, {'C': [1.0]}, cv=cv if cv is not None else [(every, every)], scoring=scoring,
                           error_score=error_score, refit=False).fit(X, y)


def test_scorer_call_average_none_returns_the_signed_per_label_array():
    X, y = _data()
    est = ml.LogisticRegression(max_iter=50).fit(X, y)
    want = np.asarray(metrics.precision_score(y, est.predict(X), average=None))
    got = ms.make_scorer(metrics.precision_score, average=None)(est, X, y)
    np.testing.assert_array_equal(np.asarray(got), want)
    neg = ms.make_scorer(metrics.precision_score, average=None, greater_is_better=False)(est, X, y)
    np.testing.assert_array_equal(np.asarray(neg), -want)


def test_search_average_none_of_several_labels_is_sklearns_refusal_whatever_error_score():
    X, y = _data()
    with pytest.raises(ValueError, match=r"scoring must return a number, got .* instead\. \(scorer="):
        _search(ms.make_scorer(metrics.f1_score, average=None), X, y, error_score=np.nan)


def test_search_average_none_of_one_label_scores_its_value():
    X, y = _data()
    search = _search(ms.make_scorer(metrics.f1_score, average=None, labels=[1]), X, y)
    est = ml.LogisticRegression(max_iter=50).fit(X, y)
    want = float(np.asarray(metrics.f1_score(y, est.predict(X), average=None, labels=[1]))[0])
    assert np.isclose(search.cv_results_['mean_test_score'][0], want, rtol=1e-6, atol=0.0)


def test_search_scorer_sample_weight_is_the_weighted_metric():
    X, y = _data()
    w = np.array([1.0, 2.0, 0.5, 1.0, 3.0, 1.0, 2.0, 1.0, 0.5], dtype=np.float32)
    est = ml.LogisticRegression(max_iter=50).fit(X, y)
    for metric in (metrics.accuracy_score, metrics.balanced_accuracy_score):
        search = _search(ms.make_scorer(metric, sample_weight=w), X, y)
        want = float(metric(y, est.predict(X), sample_weight=w))
        assert np.isclose(search.cv_results_['mean_test_score'][0], want, rtol=1e-6, atol=0.0), metric.__name__


def test_search_scorer_sample_weight_of_another_length_refuses_per_side():
    X, y = _data()
    w = np.ones(len(X), dtype=np.float32)
    with pytest.raises(ValueError, match=r"sample_weight has 9 entries for 3 samples"):
        _search(ms.make_scorer(metrics.accuracy_score, sample_weight=w), X, y, cv=3, error_score='raise')
    search = _search(ms.make_scorer(metrics.accuracy_score, sample_weight=w), X, y, cv=3, error_score=np.nan)
    assert np.isnan(search.cv_results_['mean_test_score'][0])


def test_search_average_samples_on_1d_targets_is_the_doors_refusal():
    X, y = _data()
    with pytest.raises(NotImplementedError, match=r"average='samples' applies to multilabel targets"):
        _search(ms.make_scorer(metrics.precision_score, average='samples'), X, y, error_score='raise')


def test_search_float_pos_label_on_a_binary_label_scorer_is_the_doors_refusal():
    X, y = _data()
    keep = y < 2
    with pytest.raises(TypeError, match=r"pos_label must contain only strings or only integers"):
        _search(ms.make_scorer(metrics.f1_score, pos_label=1.0), X[keep], y[keep], error_score='raise')


def test_callable_scorer_refusal_names_the_native_alternative():
    X, y = _data()
    with pytest.raises(TypeError, match=r"scoring string .* or make_scorer over a native mojolearn metric"):
        _search(lambda est, X, y: 0.0, X, y)
