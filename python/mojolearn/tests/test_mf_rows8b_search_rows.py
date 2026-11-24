# SPDX-License-Identifier: Apache-2.0
"""The multi-output and refusal search rows, typed (lane mf-rows8b, 2026-11-23).

A search (and the cv tools fitted as one) over native members now admits what its admission used to hand to the
door's units:

- a label or clustering scorer on a multi-output target refuses per side in scikit-learn's words (a mix of target
  types, ``multiclass-multioutput`` / ``continuous-multioutput`` not supported, ``labels_true must be 1D``);
  multilabel-indicator targets stay NOT IMPLEMENTED in the label doors' words; a numeric ``error_score`` replaces
  each refusal;
- a regression scorer on a multi-output classifier scores its labels' values, as scikit-learn does;
- StratifiedKFold over a multi-output target refuses in its words before any fit;
- a member that may take a multi-output target (KNeighbors, PLS) on a 1-D target is scored as any member;
- ``cross_val_predict`` of a non-classifier's ``predict_proba`` (GaussianMixture) is its own columns;
- a regressor's float predictions under a clustering scorer refuse in the public cluster metric's words.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import model_selection as ms

sk = pytest.importorskip('sklearn.metrics')

_TESTS = [[0, 2, 4], [1, 3, 5], [6, 7, 8]]


def _folds(n):
    every = np.arange(n)
    return [(np.setdiff1d(every, np.array(t)), np.array(t)) for t in _TESTS]


def _x(n=12):
    return np.linspace(0.0, 1.0, 2 * n, dtype=np.float32).reshape(n, 2)


def _search(estimator, X, y, scoring, error_score='raise', cv=None):
    return ms.GridSearchCV(estimator, {'n_neighbors': [3]}, cv=_folds(len(X)) if cv is None else cv,
                           scoring=scoring, error_score=error_score, refit=False).fit(X, y)


def test_label_scorer_on_a_continuous_multi_output_regressor_is_sklearns_words():
    X = _x()
    y = np.stack([np.linspace(0.1, 2.3, 12), np.linspace(1.7, 0.2, 12)], axis=1).astype(np.float32)
    with pytest.raises(ValueError, match='continuous-multioutput is not supported'):
        _search(ml.KNeighborsRegressor(), X, y, 'accuracy')
    got = _search(ml.KNeighborsRegressor(), X, y, 'accuracy', error_score=-3.0)
    assert [got.cv_results_[f'split{f}_test_score'][0] for f in range(3)] == [-3.0, -3.0, -3.0]


def test_clustering_scorer_on_a_multi_output_target_is_sklearns_words():
    X = _x()
    y = np.stack([np.arange(12) % 3, np.arange(12) % 2], axis=1).astype(np.float32)
    with pytest.raises(ValueError, match=r'labels_true must be 1D: shape is \(3, 2\)'):
        _search(ml.KNeighborsRegressor(), X, y, 'adjusted_rand_score')


def test_multilabel_target_stays_not_implemented():
    X = _x()
    y = np.stack([np.arange(12) % 2, (np.arange(12) // 2) % 2], axis=1).astype(np.int64)
    with pytest.raises(NotImplementedError, match='multilabel-indicator targets are NOT IMPLEMENTED'):
        _search(ml.KNeighborsClassifier(), X, y, 'accuracy')
    with pytest.raises(ValueError, match='multilabel-indicator is not supported'):
        _search(ml.KNeighborsClassifier(), X, y, 'matthews_corrcoef')


def test_regression_scorer_on_a_multi_output_classifier_equals_sklearn():
    X = _x()
    y = np.stack([np.arange(12) % 3 * 10, np.arange(12) % 4], axis=1).astype(np.int64)
    got = _search(ml.KNeighborsClassifier(), X, y, 'r2')
    want = []
    for train, test in _folds(len(X)):
        est = ml.KNeighborsClassifier(n_neighbors=3).fit(X[train], y[train])
        want.append(float(sk.r2_score(y[test], est.predict(X[test]))))
    np.testing.assert_allclose([got.cv_results_[f'split{f}_test_score'][0] for f in range(3)], want, rtol=1e-5)


def test_stratified_kfold_over_a_multi_output_target_is_sklearns_words():
    X = _x()
    y = np.stack([np.arange(12) % 3, np.arange(12) % 4], axis=1).astype(np.int64)
    with pytest.raises(ValueError, match=r"Supported target types are: \('binary', 'multiclass'\)\. "
                                         r"Got 'multiclass-multioutput' instead\."):
        _search(ml.KNeighborsClassifier(), X, y, 'accuracy', cv=ms.StratifiedKFold(3))


def test_multi_capable_member_on_a_one_output_target_scores_natively():
    X = _x()
    y = (np.arange(12) % 3).astype(np.int64)
    got = _search(ml.KNeighborsClassifier(), X, y, 'accuracy')
    want = []
    for train, test in _folds(len(X)):
        est = ml.KNeighborsClassifier(n_neighbors=3).fit(X[train], y[train])
        want.append(float(sk.accuracy_score(y[test], est.predict(X[test]))))
    np.testing.assert_allclose([got.cv_results_[f'split{f}_test_score'][0] for f in range(3)], want, rtol=1e-6)


def test_cross_val_predict_of_a_mixture_models_proba_is_its_own_columns():
    # Nine rows: `_TESTS` is a partition of 0..8 only; over twelve rows scikit-learn's cross_val_predict itself
    # refuses ("cross_val_predict only works for partitions", sklearn/model_selection/_validation.py).
    X = _x(9)
    # an in-package caller receives the Array (no fancy indexing): read it as NumPy
    got = np.asarray(ms.cross_val_predict(ml.GaussianMixture(n_components=2, random_state=0), X, None,
                                          cv=_folds(len(X)), method='predict_proba'))
    assert got.shape == (9, 2)
    for train, test in _folds(len(X)):
        est = ml.GaussianMixture(n_components=2, random_state=0).fit(X[train])
        np.testing.assert_allclose(got[test], est.predict_proba(X[test]), rtol=1e-5, atol=1e-7)


def test_float_predictions_under_a_clustering_scorer_are_the_cluster_doors_words():
    X = _x()
    y = (np.arange(12) % 3).astype(np.int64)
    # The cluster doors' refusal is a failed SCORE: with error_score='raise' it raises; under the default
    # numeric error_score scikit-learn's `_score` scores error_score and warns "Scoring failed" with the words
    # (lane w3f3-b d9b1fae936 made the search follow that rule; this test expected a raise under NaN).
    with pytest.raises(Exception, match='labels_pred must be an integer label array'):
        ms.GridSearchCV(ml.KNeighborsRegressor(), {'n_neighbors': [3]}, cv=_folds(12),
                        scoring='adjusted_rand_score', refit=False, error_score='raise').fit(X, y)
    import warnings
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        got = ms.GridSearchCV(ml.KNeighborsRegressor(), {'n_neighbors': [3]}, cv=_folds(12),
                              scoring='adjusted_rand_score', refit=False).fit(X, y)
    assert all(np.isnan(got.cv_results_[f'split{f}_test_score'][0]) for f in range(3))
    failed = [str(w.message) for w in seen if str(w.message).startswith('Scoring failed.')]
    assert len(failed) == 3 and all('labels_pred must be an integer label array' in m for m in failed), failed
