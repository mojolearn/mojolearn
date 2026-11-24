# SPDX-License-Identifier: Apache-2.0
"""The scorer rows of a search over native members that lane mf-r9d typed (2026-11-23).

- a make_scorer whose ``numeric_mode`` names the tier that is not the process default scores on that tier's
  metrics image, as the native scorer called on the fold estimator does;
- a regression metric over ``predict_log_proba`` scores the log of the positive column, and a metric row with a
  ``response_method`` tuple scores the first method the class has.

Each value is compared with the same scorer called on the search's own fold fits (the public API). The refusals of
these scorer rows in scikit-learn's words (roc_auc and log_loss over a mixture model's columns, a label metric over
predict_proba, a bad top-k ``k``) are in test_meta_boundary.py.
"""
import re

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import metrics
from mojolearn import model_selection as ms

_TESTS = [[0, 2, 4, 9], [1, 3, 5, 10], [6, 7, 8, 11]]


def _folds(n):
    every = np.arange(n)
    return [(np.setdiff1d(every, np.array(t)), np.array(t)) for t in _TESTS]


def _x(n=12):
    return np.linspace(0.0, 1.0, 2 * n, dtype=np.float32).reshape(n, 2)


def _search(estimator, grid, X, y, scoring, error_score='raise'):
    return ms.GridSearchCV(estimator, grid, cv=_folds(len(X)), scoring=scoring, error_score=error_score,
                           refit=False).fit(X, y)


def _fold_scores(make, X, y, scorer):
    out = []
    for train, test in _folds(len(X)):  # glue: the three folds of the reference, a fresh estimator each
        fitted = make().fit(X[train], y[train])
        out.append(scorer(fitted, X[test], y[test]))
    return np.asarray(out, dtype=np.float64)


def _split_scores(search):
    return np.asarray([search.cv_results_[f'split{f}_test_score'][0] for f in range(3)], dtype=np.float64)


def test_r2_over_predict_log_proba_scores_the_log_of_the_positive_column():
    # Fail-proof: in core/msel_search_run.mojo `_respond` drop the log (the scores become r2 of the probabilities).
    X = _x()
    y = (np.arange(12) % 2).astype(np.int64)
    scorer = ms.make_scorer(metrics.r2_score, response_method='predict_log_proba')
    search = _search(ml.LogisticRegression(max_iter=30), {'C': [1.0]}, X, y, scorer)
    want = _fold_scores(lambda: ml.LogisticRegression(max_iter=30, C=1.0), X, y, scorer)
    np.testing.assert_allclose(_split_scores(search), want, rtol=1e-5, atol=1e-6)


def test_a_metric_row_with_a_response_tuple_scores_the_first_method_the_class_has():
    # Fail-proof: in the door's `_scorer_name` return "" for a tuple response (the search keeps no typed row).
    X = _x()
    y = (np.arange(12) % 2).astype(np.int64)
    scorer = ms.make_scorer(metrics.hinge_loss, response_method=('decision_function', 'predict'),
                            greater_is_better=False)
    search = _search(ml.LogisticRegression(max_iter=30), {'C': [1.0]}, X, y, scorer)
    want = _fold_scores(lambda: ml.LogisticRegression(max_iter=30, C=1.0), X, y, scorer)
    np.testing.assert_allclose(_split_scores(search), want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('mode', ['fast', 'identical'])
def test_numeric_mode_of_either_tier_scores_on_that_tier(mode):
    # Fail-proof: in core/msel_search_run.mojo route every scorer by `plan.metric` (the other tier's values, or
    # its image's refusal, are lost).
    X = _x()
    # float32 targets: r2_score documents float32 only (float64 raises "the float64 overloads ... are NOT
    # implemented"), so a float64 y made the reference fail for a reason that is not the tier's.
    y = np.linspace(0.0, 2.0, 12, dtype=np.float32)
    scorer = ms.make_scorer(metrics.r2_score, numeric_mode=mode)
    estimator = ml.KNeighborsRegressor(n_neighbors=3, numeric_mode='identical')
    try:
        want = _fold_scores(lambda: ml.KNeighborsRegressor(n_neighbors=3, numeric_mode='identical'), X, y, scorer)
    except Exception as error:  # the tier's image is not in this installation: the same words from the search
        with pytest.raises(type(error), match=re.escape(str(error)[:40])):
            _search(estimator, {'n_neighbors': [3]}, X, y, scorer)
        return
    search = _search(estimator, {'n_neighbors': [3]}, X, y, scorer)
    np.testing.assert_allclose(_split_scores(search), want, rtol=1e-5, atol=1e-6)
