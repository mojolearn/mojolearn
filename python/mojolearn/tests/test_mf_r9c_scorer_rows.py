# SPDX-License-Identifier: Apache-2.0
"""The last scorer rows of a search over native members, typed (lane mf-r9c, 2026-11-23).

- a probability scorer on a multi-output classifier reads the fold estimator's ``classes_`` list as scikit-learn's
  ``_get_response_values`` does and the metric's handling of the per-output list in scikit-learn's words; a
  multilabel-indicator target stays NOT IMPLEMENTED in the doors' words; a numeric ``error_score`` replaces each;
- the dual ``roc_auc`` scorer with ``multi_class='ovr'`` / ``'ovo'`` scores the response the class has;
- a regression metric over a ``predict_proba`` / ``decision_function`` response scores the column scikit-learn
  hands it, and refuses a several-column block in ``_check_reg_targets``' words.

Each test compares against scikit-learn's own search where scikit-learn serves the case.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import metrics
from mojolearn import model_selection as ms

sk_ms = pytest.importorskip('sklearn.model_selection')
sk_metrics = pytest.importorskip('sklearn.metrics')
sk_neighbors = pytest.importorskip('sklearn.neighbors')

_TESTS = [[0, 2, 4, 9], [1, 3, 5, 10], [6, 7, 8, 11]]


def _folds(n):
    every = np.arange(n)
    return [(np.setdiff1d(every, np.array(t)), np.array(t)) for t in _TESTS]


def _x(n=12):
    return np.linspace(0.0, 1.0, 2 * n, dtype=np.float32).reshape(n, 2)


def _search(estimator, grid, X, y, scoring, error_score='raise'):
    return ms.GridSearchCV(estimator, grid, cv=_folds(len(X)), scoring=scoring, error_score=error_score,
                           refit=False).fit(X, y)


def test_roc_auc_on_a_multilabel_multi_output_classifier_is_not_implemented():
    # Fail-proof: drop the multilabel branch of _multi_proba_value's roc_auc arm (the side then raises the
    # format words).
    X = _x()
    y = np.stack([np.arange(12) % 2, (np.arange(12) // 2) % 2], axis=1)
    with pytest.raises(NotImplementedError, match='multilabel-indicator targets are NOT IMPLEMENTED'):
        _search(ml.KNeighborsClassifier(), {'n_neighbors': [3]}, X, y, 'roc_auc')


def test_log_loss_on_a_multiclass_multi_output_classifier_is_sklearns_words():
    # Fail-proof: in _multi_classes drop the unequal-widths return (the words become the dim-3 words).
    X = _x()
    y = np.stack([np.arange(12) % 3, np.arange(12) % 4], axis=1)
    with pytest.raises(ValueError, match='inhomogeneous shape after 2 dimensions') as ours:
        _search(ml.KNeighborsClassifier(), {'n_neighbors': [3]}, X, y, 'neg_log_loss')
    with pytest.raises(ValueError, match='inhomogeneous shape after 2 dimensions'):
        sk_ms.GridSearchCV(sk_neighbors.KNeighborsClassifier(), {'n_neighbors': [3]}, cv=_folds(12),
                           scoring='neg_log_loss', error_score='raise', refit=False).fit(X, y)
    assert 'The detected shape was (2, 4)' in str(ours.value)


def test_top_k_on_a_multi_output_classifier_takes_the_error_score():
    # Fail-proof: drop the top_k branch of _multi_proba_value (the side raises the list words, same score).
    X = _x()
    y = np.stack([np.arange(12) % 3, np.arange(12) % 3], axis=1)
    got = _search(ml.KNeighborsClassifier(), {'n_neighbors': [3]}, X, y, 'top_k_accuracy', error_score=-3.0)
    assert np.all(got.cv_results_['mean_test_score'] == -3.0)
    with pytest.raises(ValueError, match=r'y should be a 1d array, got an array of shape \(4, 2\) instead'):
        _search(ml.KNeighborsClassifier(), {'n_neighbors': [3]}, X, y, 'top_k_accuracy')


def test_dual_roc_auc_with_multi_class_ovr_matches_sklearn():
    # Fail-proof: in _multi_class_row leave kw.rename empty (the binary row raises multi_class must be in ...).
    X = _x()
    y = np.arange(12) % 3
    mine = ms.make_scorer(metrics.roc_auc_score, response_method=('decision_function', 'predict_proba'),
                          multi_class='ovr')
    theirs = sk_metrics.make_scorer(sk_metrics.roc_auc_score,
                                    response_method=('decision_function', 'predict_proba'), multi_class='ovr')
    got = _search(ml.KNeighborsClassifier(), {'n_neighbors': [3, 5]}, X, y, mine)
    want = sk_ms.GridSearchCV(sk_neighbors.KNeighborsClassifier(), {'n_neighbors': [3, 5]}, cv=_folds(12),
                              scoring=theirs, error_score='raise', refit=False).fit(X, y)
    np.testing.assert_allclose(got.cv_results_['mean_test_score'], want.cv_results_['mean_test_score'],
                               rtol=1e-5)


def test_dual_roc_auc_ovr_over_decision_rows_is_sklearns_words():
    # Fail-proof: in _dual drop the multi_class name test (the predict_proba row scores instead).
    X = _x()
    y = np.arange(12) % 3
    mine = ms.make_scorer(metrics.roc_auc_score, response_method=('decision_function', 'predict_proba'),
                          multi_class='ovr')
    with pytest.raises(ValueError, match='Target scores need to be probabilities for multiclass roc_auc'):
        ms.GridSearchCV(ml.LogisticRegression(max_iter=30), {'C': [1.0]}, cv=_folds(12), scoring=mine,
                        error_score='raise', refit=False).fit(X, y)


def test_r2_over_a_binary_classifiers_predict_proba_matches_sklearn():
    # Fail-proof: the sabotage patch bindings/sabotage/coverage_msel_r9c_scorers.patch (classes[0]'s column).
    X = _x()
    y = np.arange(12) % 2
    mine = ms.make_scorer(metrics.r2_score, response_method='predict_proba')
    theirs = sk_metrics.make_scorer(sk_metrics.r2_score, response_method='predict_proba')
    got = _search(ml.KNeighborsClassifier(), {'n_neighbors': [3, 5]}, X, y, mine)
    want = sk_ms.GridSearchCV(sk_neighbors.KNeighborsClassifier(), {'n_neighbors': [3, 5]}, cv=_folds(12),
                              scoring=theirs, error_score='raise', refit=False).fit(X, y)
    np.testing.assert_allclose(got.cv_results_['mean_test_score'], want.cv_results_['mean_test_score'],
                               rtol=1e-5)


def test_r2_over_a_multiclass_predict_proba_is_sklearns_words():
    # Fail-proof: in _response_regression drop the width check (the program reads the first column).
    X = _x()
    y = np.arange(12) % 3
    mine = ms.make_scorer(metrics.r2_score, response_method='predict_proba')
    with pytest.raises(ValueError, match=r'different number of output \(1!=3\)'):
        _search(ml.KNeighborsClassifier(), {'n_neighbors': [3]}, X, y, mine)
