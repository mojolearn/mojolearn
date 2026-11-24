# SPDX-License-Identifier: Apache-2.0
"""Every remaining search scorer, typed (lane mf-score8, 2026-11-23).

A search scored by each newly typed scorer runs its typed RUN (no door units), and every split's score equals
scikit-learn's metric of the fold estimator's predictions on the same rows, signed by ``greater_is_better``:

- the regression metrics with an alpha or a power (``mean_pinball_loss``, ``d2_pinball_score``,
  ``mean_tweedie_deviance``, ``d2_tweedie_score``), ``multioutput`` on a 1-D target and ``force_finite``;
- ``accuracy_score(normalize=False)``, ``balanced_accuracy_score(adjusted=True)``, ``cohen_kappa_score`` (with
  ``weights`` and ``labels``), ``log_loss(normalize=False)``, ``top_k_accuracy_score(normalize=False)`` and
  ``brier_score_loss(scale_by_half=True)``;
- a regression scorer on a classifier (its predicted labels scored as numbers).

A kwarg value the metric refuses is raised per side in the metric's own words (``error_score='raise'``), and
becomes the error_score with a numeric ``error_score``.
"""
import warnings

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import metrics
from mojolearn import model_selection as ms

sk = pytest.importorskip('sklearn.metrics')


def _cls_data():
    X = np.array([[0.0, 0.1], [0.2, 0.0], [1.0, 1.1], [1.2, 0.9], [2.0, 2.1], [2.2, 1.9],
                  [0.1, 0.3], [1.1, 1.0], [2.1, 2.0], [0.3, 0.2], [1.3, 1.2], [2.3, 2.2]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
    return X, y


def _reg_data():
    rng = np.random.RandomState(0)
    X = rng.rand(12, 3).astype(np.float32)
    y = (X @ np.array([1.0, 2.0, 0.5], dtype=np.float32) + 1.0).astype(np.float32)
    return X, y


# Three folds whose test rows hold every class of `_cls_data` (and train rows too), so each side's metric sees
# the labels the estimator's columns name.
_TESTS = [[0, 2, 4], [1, 3, 5], [6, 7, 8]]


def _folds(n):
    every = np.arange(n)
    return [(np.setdiff1d(every, np.array(t)), np.array(t)) for t in _TESTS]


def _grid(estimator):
    return {'alpha': [1.0]} if isinstance(estimator, ml.Ridge) else {'C': [1.0]}


def _scores(estimator, X, y, scorer, error_score='raise'):
    folds = _folds(len(X))
    search = ms.GridSearchCV(estimator, _grid(estimator), cv=folds, scoring=scorer, error_score=error_score,
                             refit=False).fit(X, y)
    return [search.cv_results_[f'split{f}_test_score'][0] for f in range(3)], folds


def _expected(make, X, y, folds, response, metric, sign, **kw):
    out = []
    for train, test in folds:
        est = make().fit(X[train], y[train])
        answer = getattr(est, response)(X[test])
        if response == 'predict_proba' and answer.shape[1] == 2 and metric is not sk.top_k_accuracy_score:
            answer = answer[:, 1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out.append(sign * float(metric(y[test], answer, **kw)))
    return out


def _check(got, want, rtol=1e-5):
    for f in range(3):
        assert np.isclose(got[f], want[f], rtol=rtol, atol=1e-6, equal_nan=True), (f, got[f], want[f])


REGRESSION = [
    ('mean_pinball_loss', {'alpha': 0.3}, False),
    ('mean_pinball_loss', {'alpha': 0.9}, True),
    ('d2_pinball_score', {'alpha': 0.25}, True),
    ('mean_tweedie_deviance', {'power': 0}, False),
    ('mean_tweedie_deviance', {'power': 3}, False),
    ('d2_tweedie_score', {'power': 0}, True),
    ('r2_score', {'force_finite': False}, True),
    ('r2_score', {'multioutput': 'raw_values'}, True),
    ('r2_score', {'multioutput': 'variance_weighted'}, True),
    ('explained_variance_score', {'force_finite': False}, True),
    ('mean_absolute_error', {'multioutput': 'raw_values'}, False),
    ('median_absolute_error', {'multioutput': 'uniform_average'}, False),
]


@pytest.mark.parametrize('name,kw,greater', REGRESSION)
def test_regression_metric_rows_equal_sklearn(name, kw, greater):
    X, y = _reg_data()
    y = np.abs(y) + 0.5  # positive targets: every Tweedie power's domain
    scorer = ms.make_scorer(getattr(metrics, name), greater_is_better=greater, **kw)
    got, folds = _scores(ml.Ridge(), X, y, scorer)
    want = _expected(ml.Ridge, X, y, folds, 'predict', getattr(sk, name), 1.0 if greater else -1.0, **kw)
    _check(got, want)


CLASSIFICATION = [
    ('accuracy_score', {'normalize': False}, True, 'predict'),
    ('balanced_accuracy_score', {'adjusted': True}, True, 'predict'),
    ('cohen_kappa_score', {}, True, 'predict'),
    ('cohen_kappa_score', {'weights': 'linear'}, True, 'predict'),
    ('cohen_kappa_score', {'weights': 'quadratic', 'labels': [2, 0, 1]}, True, 'predict'),
    ('log_loss', {'normalize': False}, False, 'predict_proba'),
    ('top_k_accuracy_score', {'k': 2, 'normalize': False}, True, 'predict_proba'),
    ('brier_score_loss', {'scale_by_half': True}, False, 'predict_proba'),
]


@pytest.mark.parametrize('name,kw,greater,response', CLASSIFICATION)
def test_classification_metric_rows_equal_sklearn(name, kw, greater, response):
    X, y = _cls_data()
    scorer = ms.make_scorer(getattr(metrics, name), greater_is_better=greater, response_method=response, **kw)
    make = lambda: ml.LogisticRegression(max_iter=50)
    got, folds = _scores(make(), X, y, scorer)
    want = _expected(make, X, y, folds, response, getattr(sk, name), 1.0 if greater else -1.0, **kw)
    _check(got, want)


@pytest.mark.parametrize('name', ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error', 'max_error'])
def test_regression_scorer_on_a_classifier_scores_the_labels_as_numbers(name):
    X, y = _cls_data()
    y = (y * 10 + 5).astype(np.int64)
    got, folds = _scores(ml.LogisticRegression(max_iter=50), X, y, name)
    metric = {'neg_mean_squared_error': sk.mean_squared_error, 'r2': sk.r2_score,
              'neg_mean_absolute_error': sk.mean_absolute_error, 'max_error': sk.max_error}[name]
    sign = 1.0 if name == 'r2' else -1.0
    want = _expected(lambda: ml.LogisticRegression(max_iter=50), X, y, folds, 'predict', metric, sign)
    _check(got, want)


def test_regression_scorer_on_string_labels_refuses_in_numpys_words():
    X, y = _cls_data()
    labels = np.array(['a', 'b', 'c'])[y]
    with pytest.raises(ValueError, match=r"could not convert string to float: 'a'"):
        _scores(ml.LogisticRegression(max_iter=50), X, labels, 'neg_mean_squared_error')
    with pytest.raises(ValueError, match=r"dtype='numeric' is not compatible with arrays of bytes/strings"):
        _scores(ml.LogisticRegression(max_iter=50), X, labels, 'neg_median_absolute_error')


@pytest.mark.parametrize('name,kw,words', [
    ('mean_pinball_loss', {'alpha': 1.5}, r"alpha must be a real number in \[0, 1\]"),
    ('mean_tweedie_deviance', {'power': 0.5}, r"power in \(0, 1\) is not a Tweedie distribution"),
    ('r2_score', {'multioutput': 'bogus'}, r"Allowed 'multioutput' string values are"),
    ('r2_score', {'multioutput': [1.0]}, r"Custom weights are useful only in multi-output cases"),
])
def test_refused_kwargs_raise_the_metrics_words_per_side(name, kw, words):
    X, y = _reg_data()
    scorer = ms.make_scorer(getattr(metrics, name), **kw)
    with pytest.raises(ValueError, match=words):
        _scores(ml.Ridge(), X, y, scorer)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got, _ = _scores(ml.Ridge(), X, y, scorer, error_score=-7.0)
    assert got == [-7.0, -7.0, -7.0]


def test_kappa_weights_refusal_is_the_doors_words():
    X, y = _cls_data()
    scorer = ms.make_scorer(metrics.cohen_kappa_score, weights='cubic')
    with pytest.raises(ValueError, match=r"weights must be None, 'linear' or 'quadratic'"):
        _scores(ml.LogisticRegression(max_iter=50), X, y, scorer)
