# SPDX-License-Identifier: Apache-2.0
"""Settings lane mf-lin6 (2026-11-23) moved from the classes' own doors into their member routes, against the
estimators' own doors.

Each estimator is fitted twice: by its own fit (the door) and as a GridSearchCV member (one given split, train =
every row, a one-point grid, refit=True: the refit runs the member). The refit estimator's fitted attributes and
its outputs equal the door's. Refusals keep the door's words (scikit-learn's own where scikit-learn refuses).
"""
import numpy as np
import pytest

import mojolearn
from mojolearn.model_selection import GridSearchCV


def _data(n=40, d=3, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)
    return X, y


def _member(est, X, y=None, scoring=None):
    """The refit estimator of a one-split, one-point search (fitted as a member)."""
    every = np.arange(len(X))
    params = est.get_params()
    name = sorted(params)[0]
    # scoring=None needs the class's own `score` (scikit-learn 1.9 metrics/_scorer.py `check_scoring`, a TypeError
    # before any fit; 9b4bc3f04b, providers/python/member_ensemble_boundary_door.mojo `_no_score_refusal`). A class
    # without one is searched by a scorer whose response it lacks: every score is error_score (nan) and the best is
    # refitted through the member, as scikit-learn does (the test_mf_nbr6_members / test_mf_enc6_members helper).
    if scoring is None and getattr(est, "score", None) is None:
        scoring = "r2"
    search = GridSearchCV(est, {name: [params[name]]}, cv=[(every, every)], scoring=scoring, refit=True)
    search.fit(X, y)
    return search.best_estimator_


def _same(a, b, names):
    for name in names:
        left, right = getattr(a, name), getattr(b, name)
        if isinstance(right, dict):
            # A per-class table (LogisticRegressionCV's scores_, as scikit-learn's: {class: array}): the same
            # keys, each value equal (np.asarray of a dict is one object cell, compared by dict ==).
            assert isinstance(left, dict) and sorted(left) == sorted(right), name
            for key in right:
                np.testing.assert_array_equal(np.asarray(left[key]), np.asarray(right[key]),
                                              err_msg=name + "[" + repr(key) + "]")
            continue
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right), err_msg=name)


def _pairs(n, k=3):
    rows = np.arange(n)
    return [(rows[rows % k != f], rows[rows % k == f]) for f in range(k)]


@pytest.mark.parametrize("name,attrs,scoring", [
    ("RidgeCV", ["coef_", "intercept_", "alpha_", "best_score_"], "r2"),
    ("LassoCV", ["coef_", "intercept_", "alpha_", "mse_path_"], "r2"),
    ("ElasticNetCV", ["coef_", "intercept_", "alpha_", "l1_ratio_", "mse_path_"], "r2"),
])
def test_listed_cv_member_equals_door(name, attrs, scoring):
    X, y = _data()
    est = getattr(mojolearn, name)
    door = est(cv=_pairs(len(X))).fit(X, y)
    member = _member(est(cv=_pairs(len(X))), X, y, scoring=scoring)
    _same(member, door, attrs)
    np.testing.assert_array_equal(np.asarray(member.predict(X)), np.asarray(door.predict(X)))


def test_listed_cv_logistic_member_equals_door():
    X, y = _data()
    labels = (y > 0).astype(np.int64)
    door = mojolearn.LogisticRegressionCV(cv=_pairs(len(X))).fit(X, labels)
    member = _member(mojolearn.LogisticRegressionCV(cv=_pairs(len(X))), X, labels, scoring="accuracy")
    _same(member, door, ["coef_", "intercept_", "C_", "scores_"])


@pytest.mark.parametrize("cv,words", [
    ([], "cv produced no \\(train, test\\) split"),
    ([(np.arange(10), np.arange(10, 20), np.arange(5))], "too many values to unpack \\(expected 2, got 3\\)"),
    ([(np.arange(10),)], "not enough values to unpack \\(expected 2, got 1\\)"),
])
def test_listed_cv_malformed_keeps_door_words(cv, words):
    X, y = _data()
    with pytest.raises(Exception, match=words):
        mojolearn.LassoCV(cv=cv).fit(X, y)
    with pytest.raises(Exception, match=words):
        _member(mojolearn.LassoCV(cv=cv), X, y, scoring="r2")


def test_lda_n_components_zero_is_sklearn_refusal():
    X, y = _data()
    labels = (y > 0).astype(np.int64)
    words = "The 'n_components' parameter of LinearDiscriminantAnalysis must be an int in the range \\[1, inf\\)"
    with pytest.raises(Exception, match=words):
        mojolearn.LinearDiscriminantAnalysis(n_components=0).fit(X, labels)
    with pytest.raises(Exception, match=words):
        _member(mojolearn.LinearDiscriminantAnalysis(n_components=0), X, labels, scoring="accuracy")


@pytest.mark.parametrize("name,kwargs", [
    ("Nystroem", {"kernel": "rbf", "gamma": 0.5, "n_components": 8, "random_state": 3}),
    ("RBFSampler", {"gamma": 0.5, "n_components": 8, "random_state": 3}),
])
def test_fast_kernel_approx_member_equals_door(name, kwargs):
    X, _ = _data()
    est = getattr(mojolearn, name)
    try:
        door = est(numeric_mode="fast", **kwargs).fit(X)
    except RuntimeError as error:  # no fast kernel methods binding in this installation
        pytest.skip(str(error))
    member = _member(est(numeric_mode="fast", **kwargs), X)
    np.testing.assert_array_equal(np.asarray(member.transform(X)), np.asarray(door.transform(X)))


def test_host_lasso_member_fit_keeps_door_words():
    from mojolearn._classical_host import HostLasso
    X, y = _data()
    words = "no CPU implementation of the HostLasso fit on the host class"
    with pytest.raises(Exception, match=words):
        HostLasso().fit(X, y)
    with pytest.raises(Exception, match=words):
        _member(HostLasso(), X, y, scoring="r2")
