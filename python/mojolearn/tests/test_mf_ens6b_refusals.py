# SPDX-License-Identifier: Apache-2.0
"""Search kind refusals lane mf-ens6b (2026-11-23) moved from the Python door units to the kind's native
admission, in scikit-learn's or the door's words, plus the integral NumPy ``cv`` the kind now serves.

- An empty multi-metric ``scoring``: scikit-learn's ``_check_multimetric_scoring`` words.
- A refit callback: the search door's words (after its candidate and scorer checks).
- ``cross_val_predict`` with a ``method`` outside scikit-learn's options: its parameter words.
- ``cv=np.int64(k)``: the same folds and scores as ``cv=k``; below two, KFold's words.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import model_selection as ms


def _data():
    X = np.array([[0.0, 0.1], [0.2, 0.0], [1.0, 1.1], [1.2, 0.9], [2.0, 2.1], [2.2, 1.9],
                  [0.1, 0.3], [1.1, 1.0], [2.1, 2.0], [0.3, 0.2]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=np.int32)
    return X, y


def _grid(**kw):
    return ms.GridSearchCV(ml.LogisticRegression(max_iter=50), {'C': [1.0, 0.5]}, **kw)


@pytest.mark.parametrize('scoring, words', [
    ([], r"The list/tuple elements must be unique strings of predefined scorers\.  Empty list was given\. \[\]"),
    ({}, r"An empty dict was passed\. \{\}"),
])
def test_empty_multi_metric_scoring_is_sklearns_refusal(scoring, words):
    X, y = _data()
    with pytest.raises(ValueError, match=words):
        _grid(scoring=scoring, cv=2).fit(X, y)


def test_refit_callback_is_the_doors_refusal():
    X, y = _data()
    with pytest.raises(TypeError, match=r"Python refit callbacks are not supported"):
        _grid(cv=2, refit=lambda results: 0).fit(X, y)


def test_cross_val_predict_unknown_method_is_sklearns_refusal():
    X, y = _data()
    with pytest.raises(ValueError, match=r"The 'method' parameter of cross_val_predict must be a str among"):
        ms.cross_val_predict(ml.LogisticRegression(max_iter=50), X, y, cv=2, method='transform')


def test_numpy_integer_cv_scores_as_its_int():
    X, y = _data()
    want = _grid(cv=2).fit(X, y).cv_results_['mean_test_score']
    got = _grid(cv=np.int64(2)).fit(X, y).cv_results_['mean_test_score']
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_numpy_integer_cv_below_two_is_kfolds_refusal():
    X, y = _data()
    with pytest.raises(ValueError, match=r"k-fold cross-validation requires at least one train/test split .* "
                                         r"got n_splits=1\."):
        _grid(cv=np.int64(1)).fit(X, y)
