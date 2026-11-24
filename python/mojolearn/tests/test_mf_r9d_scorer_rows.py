# SPDX-License-Identifier: Apache-2.0
"""The scorer rows of a search over native members that lane mf-r9d typed (2026-11-23).

- a make_scorer whose ``numeric_mode`` names the tier that is not the process default scores on that tier's
  metrics image, as the native scorer called on the fold estimator does;
- a regression metric over ``predict_log_proba`` scores the log of the positive column, and a metric row with a
  ``response_method`` tuple scores the first method the class has;
- ``roc_auc`` over a non-classifier's own columns (GaussianMixture's ``predict_proba``) scores one column against
  the float targets and refuses two in scikit-learn's ``column_or_1d`` words.

Each value is compared with the same scorer called on the search's own fold fits (the public API), and a refusal
with scikit-learn's own search where scikit-learn serves the case.
"""
import re

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import metrics
from mojolearn import model_selection as ms

sk_ms = pytest.importorskip('sklearn.model_selection')
sk_mixture = pytest.importorskip('sklearn.mixture')

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
    y = np.linspace(0.0, 2.0, 12)
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


def test_roc_auc_over_a_mixture_models_columns():
    # Fail-proof: in core/msel_scorer_run.mojo `_float_roc_value` drop the two-column refusal (two components
    # score instead of raising).
    X = _x()
    y = (np.arange(12) % 2).astype(np.float64)
    one = _search(ml.GaussianMixture(random_state=0, numeric_mode='identical'), {'n_components': [1]}, X, y,
                  'roc_auc')
    np.testing.assert_allclose(_split_scores(one), [0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match=r'y should be a 1d array, got an array of shape \(4, 2\) instead'):
        _search(ml.GaussianMixture(random_state=0, numeric_mode='identical'), {'n_components': [2]}, X, y,
                'roc_auc')
    with pytest.raises(ValueError, match=r'y should be a 1d array, got an array of shape \(4, 2\) instead'):
        sk_ms.GridSearchCV(sk_mixture.GaussianMixture(random_state=0), {'n_components': [2]}, cv=_folds(12),
                           scoring='roc_auc', error_score='raise', refit=False).fit(X, y)
    tolerated = _search(ml.GaussianMixture(random_state=0, numeric_mode='identical'), {'n_components': [2]}, X, y,
                        'roc_auc', error_score=-3.0)
    np.testing.assert_array_equal(_split_scores(tolerated), [-3.0, -3.0, -3.0])


def test_roc_auc_over_a_mixture_model_with_continuous_targets_is_sklearns_words():
    # Fail-proof: in `_float_roc_value` code a continuous y as labels (the side scores).
    X = _x()
    y = np.linspace(0.0, 1.0, 12) + 0.25
    with pytest.raises(ValueError, match='continuous format is not supported'):
        _search(ml.GaussianMixture(random_state=0, numeric_mode='identical'), {'n_components': [1]}, X, y,
                'roc_auc')


def test_log_loss_over_a_mixture_model_with_more_target_values_than_components_is_sklearns_words():
    # ROUND b. Fail-proof: in `_float_proba_value` drop log_loss's class-count check (the side scores).
    X = _x()
    y = (np.arange(12) % 3).astype(np.float64)
    with pytest.raises(ValueError, match='y_true and y_prob contain different number of classes: 3 vs 2'):
        _search(ml.GaussianMixture(random_state=0, numeric_mode='identical'), {'n_components': [2]}, X, y,
                'neg_log_loss')
    with pytest.raises(ValueError, match='y_true and y_prob contain different number of classes: 3 vs 2'):
        sk_ms.GridSearchCV(sk_mixture.GaussianMixture(random_state=0), {'n_components': [2]}, cv=_folds(12),
                           scoring='neg_log_loss', error_score='raise', refit=False).fit(X, y)


def test_a_label_metric_whose_tuple_resolves_to_predict_proba_is_sklearns_mix_words():
    # ROUND b. Fail-proof: in `_wrong_response_value` drop the binary column pick (the words name
    # 'continuous-multioutput').
    sk_linear = pytest.importorskip('sklearn.linear_model')
    sk_m = pytest.importorskip('sklearn.metrics')
    X = _x()
    y = (np.arange(12) % 2).astype(np.int64)
    ours = ms.make_scorer(metrics.accuracy_score, response_method=('predict_proba', 'predict'))
    with pytest.raises(ValueError, match="can't handle a mix of binary and continuous targets"):
        _search(ml.LogisticRegression(max_iter=30), {'C': [1.0]}, X, y, ours)
    theirs = sk_ms.make_scorer if hasattr(sk_ms, 'make_scorer') else None
    if theirs is not None:
        with pytest.raises(ValueError, match="can't handle a mix of binary and continuous targets"):
            sk_ms.GridSearchCV(sk_linear.LogisticRegression(max_iter=30), {'C': [1.0]}, cv=_folds(12),
                               scoring=sk_m.make_scorer(sk_m.accuracy_score,
                                                        response_method=('predict_proba', 'predict')),
                               error_score='raise', refit=False).fit(X, y)


def test_top_k_with_a_bad_k_is_the_doors_words_per_side():
    # ROUND b. Fail-proof: in `_scorer_name` drop the k refusal (the search keeps no row for it).
    X = _x()
    y = (np.arange(12) % 3).astype(np.int64)
    scorer = ms.make_scorer(metrics.top_k_accuracy_score, response_method=('decision_function', 'predict_proba'),
                            k=0)
    with pytest.raises(Exception, match='k must be a positive integer'):
        _search(ml.LogisticRegression(max_iter=30), {'C': [1.0]}, X, y, scorer)
    tolerated = _search(ml.LogisticRegression(max_iter=30), {'C': [1.0]}, X, y, scorer, error_score=-3.0)
    np.testing.assert_array_equal(_split_scores(tolerated), [-3.0, -3.0, -3.0])
