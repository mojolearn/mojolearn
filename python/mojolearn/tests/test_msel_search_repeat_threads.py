# SPDX-License-Identifier: Apache-2.0
"""A search's scores do not depend on what ran before it or beside it (lane w3f3-g, 2026-11-24).

In the full suite (pytest -n 32, many files per worker) about 50 search / cv tests returned other numbers than
alone, e.g. GridSearchCV(Ridge(), cv=<three explicit folds>, scoring=r2) split 0 = 0.5543 where the fold's own
Ridge (and scikit-learn) gives -4.0477, and two Python threads running cross_val_score at once got values that
differ from one call alone (tests/proving/test_proving_threads.py). The red count moved between runs of the same
binaries (29, 6, 28), so the cause is state shared between operations or between the units of one operation.

This file is the reproducer: the same search, on FRESH arrays of the same shape each time (so a buffer address or
an id() is reused, as between the suite's tests), many times one after another and then from four threads at
once. Every call must return the scores of the first call. Fold 0's test rows hold every class, as in
test_mf_score8_scorers.
"""
import threading

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import metrics
from mojolearn import model_selection as ms

_TESTS = [[0, 2, 4], [1, 3, 5], [6, 7, 8]]
REPEATS = 12
THREADS = 4


def _reg():
    rng = np.random.RandomState(0)
    X = rng.rand(12, 3).astype(np.float32)
    y = (X @ np.array([1.0, 2.0, 0.5], dtype=np.float32) + 1.0).astype(np.float32)
    return X, np.abs(y) + 0.5


def _cls():
    X = np.array([[0.0, 0.1], [0.2, 0.0], [1.0, 1.1], [1.2, 0.9], [2.0, 2.1], [2.2, 1.9],
                  [0.1, 0.3], [1.1, 1.0], [2.1, 2.0], [0.3, 0.2], [1.3, 1.2], [2.3, 2.2]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
    return X, y


def _folds(n):
    every = np.arange(n)
    return [(np.setdiff1d(every, np.array(t)), np.array(t)) for t in _TESTS]


def _search_ridge_r2(_):
    X, y = _reg()  # fresh arrays every call
    scorer = ms.make_scorer(metrics.r2_score)
    search = ms.GridSearchCV(ml.Ridge(), {'alpha': [1.0]}, cv=_folds(len(X)), scoring=scorer,
                             error_score='raise', refit=False).fit(X, y)
    return np.array([search.cv_results_[f'split{f}_test_score'][0] for f in range(3)])


def _search_logistic_accuracy(_):
    X, y = _cls()
    scorer = ms.make_scorer(metrics.accuracy_score, normalize=False)
    search = ms.GridSearchCV(ml.LogisticRegression(max_iter=50), {'C': [1.0]}, cv=_folds(len(X)),
                             scoring=scorer, error_score='raise', refit=False).fit(X, y)
    return np.array([search.cv_results_[f'split{f}_test_score'][0] for f in range(3)])


def _cvs_linear_svc(_):
    X, y = _cls()
    return np.asarray(ms.cross_val_score(ml.LinearSVC(), X, y, cv=3))


def _expected_ridge_r2():
    # the fold's own Ridge, scored by the metric: what every search call must give
    X, y = _reg()
    out = []
    for train, test in _folds(len(X)):
        est = ml.Ridge(alpha=1.0).fit(X[train], y[train])
        out.append(float(metrics.r2_score(y[test], est.predict(X[test]))))
    return np.array(out)


def _threads(call):
    barrier = threading.Barrier(THREADS)
    results = [None] * THREADS
    errors = []

    def worker(i):
        try:
            barrier.wait()
            results[i] = [call(i) for _ in range(3)]
        except BaseException as error:  # re-raised below
            errors.append(error)

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(THREADS)]
    for t in workers:
        t.start()
    for t in workers:
        t.join(timeout=600)
        assert not t.is_alive(), "a thread did not finish within 600 s (deadlock?)"
    if errors:
        raise errors[0]
    return [r for rows in results for r in rows]


CALLS = [_search_ridge_r2, _search_logistic_accuracy, _cvs_linear_svc]


@pytest.mark.parametrize('call', CALLS, ids=[c.__name__ for c in CALLS])
def test_repeated_calls_give_the_first_calls_scores(call):
    first = call(0)
    for i in range(REPEATS):
        got = call(i)
        assert np.array_equal(got, first, equal_nan=True), (i, got, first)


@pytest.mark.parametrize('call', CALLS, ids=[c.__name__ for c in CALLS])
def test_concurrent_calls_give_one_calls_scores(call):
    first = call(0)
    for i, got in enumerate(_threads(call)):
        assert np.array_equal(got, first, equal_nan=True), (i, got, first)


def test_ridge_search_scores_are_the_folds_own_fits():
    np.testing.assert_allclose(_search_ridge_r2(0), _expected_ridge_r2(), rtol=1e-5, atol=1e-6)
