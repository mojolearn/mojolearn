# SPDX-License-Identifier: Apache-2.0
"""The last search scorer rows, typed (lane mf-rows8a, 2026-11-23).

A search over native members now admits, and scores in its typed RUN, the scorers its admission used to hand to
the door's units:

- the binary label scorers (``f1``, ``precision``, ``recall``, ``jaccard`` with ``average='binary'``) on a
  regressor: each side reads its targets and predictions as scikit-learn's ``_check_targets`` does and finds the
  pos_label among them by value; one that is not among two present labels refuses in
  ``_check_set_wise_labels``' words, the labels printed as NumPy prints a float array;
- pos_label kinds: a bool is an integer label; a pos_label that is not one of a binary classifier's classes on
  ``brier_score_loss`` refuses in ``_get_response_values``' words;
- a ``labels`` kwarg the metric doors refuse (empty, a float label, a repeated label) refuses in their words;
- a make_scorer kwarg the metric does not take refuses in the native scorer door's words;
- ``v_measure_score``'s ``beta`` read as ``float(beta)``.

Each refusal is raised per side, so a numeric ``error_score`` replaces it.
"""
import warnings

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import metrics
from mojolearn import model_selection as ms

sk = pytest.importorskip('sklearn.metrics')

_TESTS = [[0, 2, 4], [1, 3, 5], [6, 7, 8]]


def _folds(n):
    every = np.arange(n)
    return [(np.setdiff1d(every, np.array(t)), np.array(t)) for t in _TESTS]


def _reg01(high=1.0):
    """A regressor whose test predictions are training targets (a fully grown tree over distinct rows): integral
    float labels {0, high}, both on every test side."""
    X = np.arange(12, dtype=np.float32).reshape(12, 1)
    y = np.array([0, high, 0, high, 0, high, high, 0, high, 0, high, 0], dtype=np.float32)
    return X, y


def _bin_data():
    X = np.array([[0.0, 0.1], [0.2, 0.0], [1.0, 1.1], [1.2, 0.9], [0.1, 0.2], [1.1, 1.0],
                  [0.3, 0.1], [1.3, 1.2], [0.2, 0.3], [1.2, 1.1], [0.0, 0.3], [1.0, 1.3]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    return X, y


def _cls_data():
    X = np.array([[0.0, 0.1], [0.2, 0.0], [1.0, 1.1], [1.2, 0.9], [2.0, 2.1], [2.2, 1.9],
                  [0.1, 0.3], [1.1, 1.0], [2.1, 2.0], [0.3, 0.2], [1.3, 1.2], [2.3, 2.2]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
    return X, y


def _scores(estimator, grid, X, y, scorer, error_score='raise'):
    folds = _folds(len(X))
    search = ms.GridSearchCV(estimator, grid, cv=folds, scoring=scorer, error_score=error_score,
                             refit=False).fit(X, y)
    return [search.cv_results_[f'split{f}_test_score'][0] for f in range(3)], folds


def _tree():
    return ml.DecisionTreeRegressor(random_state=0)


_TREE_GRID = {'max_depth': [None]}


@pytest.mark.parametrize('name,metric', [
    ('f1', sk.f1_score), ('precision', sk.precision_score), ('recall', sk.recall_score),
    ('jaccard', sk.jaccard_score),
])
def test_binary_label_scorer_on_a_regressor_equals_sklearn(name, metric):
    X, y = _reg01()
    got, folds = _scores(_tree(), _TREE_GRID, X, y, name)
    want = []
    for train, test in folds:
        est = _tree().fit(X[train], y[train])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            want.append(float(metric(y[test], est.predict(X[test]))))
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_binary_label_scorer_on_a_regressor_absent_pos_label_is_sklearns_words():
    X, y = _reg01(high=2.0)
    with pytest.raises(ValueError, match=r"pos_label=1 is not a valid label\. It should be one of \[0\. 2\.\]"):
        _scores(_tree(), _TREE_GRID, X, y, 'f1')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got, _ = _scores(_tree(), _TREE_GRID, X, y, 'f1', error_score=-7.0)
    assert got == [-7.0, -7.0, -7.0]


def test_bool_pos_label_is_the_integer_label():
    X, y = _bin_data()
    lr = ml.LogisticRegression(max_iter=50)
    got, _ = _scores(lr, {'C': [1.0]}, X, y, ms.make_scorer(metrics.f1_score, pos_label=True))
    want, _ = _scores(lr, {'C': [1.0]}, X, y, ms.make_scorer(metrics.f1_score, pos_label=1))
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_brier_pos_label_not_a_class_is_get_response_values_words():
    X, y = _bin_data()
    scorer = ms.make_scorer(metrics.brier_score_loss, response_method='predict_proba', greater_is_better=False,
                            pos_label=5)
    with pytest.raises(ValueError, match=r"pos_label=5 is not a valid label: It should be one of \[0 1\]"):
        _scores(ml.LogisticRegression(max_iter=50), {'C': [1.0]}, X, y, scorer)


@pytest.mark.parametrize('labels,error,words', [
    ([], ValueError, r"labels must contain at least one label"),
    ([0.5], TypeError, r"labels must contain only strings or only integers"),
    ([0, 0], ValueError, r"labels must be unique"),
    ([7, 7], ValueError, r"labels must be unique"),
    (['a'], TypeError, r"labels must use the same type as the targets"),
])
def test_labels_kwarg_refusals_are_the_doors_words(labels, error, words):
    X, y = _cls_data()
    scorer = ms.make_scorer(metrics.f1_score, average='macro', labels=labels)
    with pytest.raises(error, match=words):
        _scores(ml.LogisticRegression(max_iter=50), {'C': [1.0]}, X, y, scorer)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got, _ = _scores(ml.LogisticRegression(max_iter=50), {'C': [1.0]}, X, y, scorer, error_score=-7.0)
    assert got == [-7.0, -7.0, -7.0]


def test_unexpected_kwarg_is_the_native_scorer_doors_words():
    X, y = _reg01()
    scorer = ms.make_scorer(metrics.r2_score, not_a_parameter=1)
    with pytest.raises(TypeError, match=r"native metric received an unexpected keyword argument"):
        _scores(ml.Ridge(), {'alpha': [1.0]}, X, y, scorer)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got, _ = _scores(ml.Ridge(), {'alpha': [1.0]}, X, y, scorer, error_score=-7.0)
    assert got == [-7.0, -7.0, -7.0]


def test_v_measure_beta_is_read_as_float():
    X, y = _cls_data()
    km = ml.KMeans(n_clusters=3, n_init=1, random_state=0)
    grid = {'n_clusters': [3]}
    text, _ = _scores(km, grid, X, y, ms.make_scorer(metrics.v_measure_score, beta='0.5'))
    real, _ = _scores(km, grid, X, y, ms.make_scorer(metrics.v_measure_score, beta=0.5))
    np.testing.assert_allclose(text, real, rtol=0, atol=0)
