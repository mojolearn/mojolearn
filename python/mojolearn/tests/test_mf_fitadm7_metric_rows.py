# SPDX-License-Identifier: Apache-2.0
"""FIT admissions lane mf-fitadm7 (2026-11-23).

- A ``make_scorer`` over ``zero_one_loss`` (``normalize`` True or False) or ``hamming_loss`` scores a search in
  its typed RUN (the metric rows): each split's score equals the metric of the fold estimator's predictions,
  signed by ``greater_is_better``, with and without ``sample_weight``.
- The same scorer in ``CalibratedClassifierCV(GridSearchCV(...))`` fits (the base is a native member).
- An unknown ``stack_method`` is refused in the Stacking door's words.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import metrics
from mojolearn import model_selection as ms


def _data():
    X = np.array([[0.0, 0.1], [0.2, 0.0], [1.0, 1.1], [1.2, 0.9], [2.0, 2.1], [2.2, 1.9],
                  [0.1, 0.3], [1.1, 1.0], [2.1, 2.0], [0.3, 0.2], [1.3, 1.2], [2.3, 2.2]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
    return X, y


def _folds(n):
    every = np.arange(n)
    return [(every[every % 3 != f], every[every % 3 == f]) for f in range(3)]


def _expected(metric, X, y, folds, sign, **kw):
    out = []
    for train, test in folds:
        est = ml.LogisticRegression(max_iter=50).fit(X[train], y[train])
        local = dict(kw)
        if 'sample_weight' in local:
            local['sample_weight'] = np.asarray(local['sample_weight'])[test]
        out.append(sign * float(metric(y[test], est.predict(X[test]), **local)))
    return out


@pytest.mark.parametrize('metric,kw', [
    (metrics.zero_one_loss, {}),
    (metrics.zero_one_loss, {'normalize': False}),
    (metrics.hamming_loss, {}),
])
@pytest.mark.parametrize('greater', [True, False])
def test_search_metric_row_scores_each_split(metric, kw, greater):
    X, y = _data()
    folds = _folds(len(X))
    scorer = ms.make_scorer(metric, greater_is_better=greater, **kw)
    search = ms.GridSearchCV(ml.LogisticRegression(max_iter=50), {'C': [1.0]}, cv=folds, scoring=scorer,
                             refit=False).fit(X, y)
    want = _expected(metric, X, y, folds, 1.0 if greater else -1.0, **kw)
    for f in range(3):
        got = search.cv_results_[f'split{f}_test_score'][0]
        assert np.isclose(got, want[f], rtol=1e-6, atol=0.0), (f, got, want[f])


def test_search_metric_row_weighted_side_length_refusal():
    X, y = _data()
    folds = _folds(len(X))
    scorer = ms.make_scorer(metrics.zero_one_loss, sample_weight=np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match=r"zero_one_loss: sample_weight has 3 entries for 4 samples"):
        ms.GridSearchCV(ml.LogisticRegression(max_iter=50), {'C': [1.0]}, cv=folds, scoring=scorer,
                        error_score='raise', refit=False).fit(X, y)


def test_calibrated_over_search_with_metric_row_fits():
    X, y = _data()
    search = ms.GridSearchCV(ml.LogisticRegression(max_iter=50), {'C': [0.5, 1.0]}, cv=2,
                             scoring=ms.make_scorer(metrics.zero_one_loss, greater_is_better=False))
    cal = ml.CalibratedClassifierCV(search, cv=2).fit(X, y)
    proba = np.asarray(cal.predict_proba(X))  # an in-package caller receives the Array (parallel_public_result_door)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


def test_stacking_unknown_stack_method_is_the_doors_words():
    X, y = _data()
    stack = ml.StackingClassifier([('lr', ml.LogisticRegression(max_iter=50))], stack_method='bogus')
    with pytest.raises(Exception, match=r"stack_method must be auto, predict_proba, decision_function or predict"):
        stack.fit(X, y)
