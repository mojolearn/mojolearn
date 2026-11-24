# SPDX-License-Identifier: Apache-2.0
"""The last search admission rows, typed (lane mf-rows8d, 2026-11-23).

- StratifiedShuffleSplit over a multi-output target, inside the search member (a search nested in a search):
  each row's joined words are its class, as scikit-learn's ``_iter_indices`` makes them; the nested fit equals
  the same search fitted on its own (the door's split of the same joined labels).
- ``cross_val_predict`` of a multi-output classifier with bool labels answers its labels per row.
- ``GridSearchCV(None, ...)`` raises scikit-learn's parameter validation words; a RandomizedSearchCV whose
  ``random_state`` is a NumPy integer or an int beyond 2**52 is drawn by the sampler as given.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import model_selection as ms


def _x(n=16):
    return np.linspace(0.0, 1.0, 2 * n, dtype=np.float32).reshape(n, 2)


def _multi_labels(n=16):
    return np.stack([np.arange(n) % 2, (np.arange(n) // 2) % 2 + 3], axis=1).astype(np.int64)


def _inner(estimator, grid):
    return ms.GridSearchCV(estimator, grid, cv=ms.StratifiedShuffleSplit(n_splits=2, test_size=8, random_state=0),
                           scoring='r2', refit=True)


def _nested_scores(inner, X, y):
    every = np.arange(len(X))
    outer = ms.GridSearchCV(inner, {'refit': [True]}, cv=[(every, every)], scoring='r2', refit=True).fit(X, y)
    return outer.best_estimator_.cv_results_['mean_test_score']


def test_stratified_shuffle_over_a_multi_output_classifier_in_the_search_member():
    X = _x()
    y = _multi_labels()
    est = ml.MultiOutputClassifier(ml.KNeighborsClassifier())
    own = _inner(est, {'estimator__n_neighbors': [1, 3]}).fit(X, y).cv_results_['mean_test_score']
    nested = _nested_scores(_inner(est, {'estimator__n_neighbors': [1, 3]}), X, y)
    np.testing.assert_array_equal(nested, own)


def test_stratified_shuffle_over_a_multi_output_regressor_in_the_search_member():
    X = _x()
    y = np.stack([np.arange(16) % 2 * 0.5, (np.arange(16) // 2) % 2 + 1.25], axis=1)
    est = ml.KNeighborsRegressor()
    own = _inner(est, {'n_neighbors': [1, 3]}).fit(X, y).cv_results_['mean_test_score']
    nested = _nested_scores(_inner(est, {'n_neighbors': [1, 3]}), X, y)
    np.testing.assert_array_equal(nested, own)


def test_stratified_shuffle_multi_output_lone_row_refuses_in_sklearns_words():
    X = _x()
    y = _multi_labels()
    y[0] = [1, 9]  # a joined row no other row has
    est = ml.MultiOutputClassifier(ml.KNeighborsClassifier())
    with pytest.raises(ValueError, match='The least populated classes in y have only 1 member'):
        _nested_scores(_inner(est, {'estimator__n_neighbors': [1]}), X, y)


def test_cross_val_predict_of_bool_multi_output_labels():
    X = _x()
    y = _multi_labels() % 2 == 1
    folds = [(np.arange(8, 16), np.arange(8)), (np.arange(8), np.arange(8, 16))]
    got = ms.cross_val_predict(ml.MultiOutputClassifier(ml.KNeighborsClassifier(n_neighbors=3)), X, y, cv=folds)
    want = np.column_stack([ms.cross_val_predict(ml.KNeighborsClassifier(n_neighbors=3), X, y[:, j], cv=folds)
                            for j in range(2)])
    assert [[bool(v) for v in row] for row in got] == want.astype(bool).tolist()


def test_search_without_an_estimator_is_sklearns_words():
    with pytest.raises(ValueError, match="The 'estimator' parameter of GridSearchCV must be an object "
                                         "implementing 'fit'. Got None instead."):
        ms.GridSearchCV(None, {'n_neighbors': [1]}).fit(_x(), np.arange(16) % 2)


@pytest.mark.parametrize('state', [np.int64(7), 2 ** 60 + 3])
def test_randomized_search_seed_as_given(state):
    X = _x()
    y = np.arange(16) % 2
    draws = {'n_neighbors': [1, 2, 3, 4, 5]}
    first = ms.RandomizedSearchCV(ml.KNeighborsClassifier(), draws, n_iter=3, random_state=state, cv=2).fit(X, y)
    again = ms.RandomizedSearchCV(ml.KNeighborsClassifier(), draws, n_iter=3, random_state=state, cv=2).fit(X, y)
    assert first.cv_results_['params'] == again.cv_results_['params']
    assert len(first.cv_results_['params']) == 3
