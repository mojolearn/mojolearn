# SPDX-License-Identifier: Apache-2.0
"""The search admission rows lane mf-rows8c (2026-11-23) closed (providers/python/member_ensemble_boundary_door.mojo).

- scoring=None over a member kind with no native score operation (the forests' and the neighbors' regressors and
  classifiers, the GP, kernel ridge, PLS, EllipticEnvelope, ...): scored by the typed scorer the class's own
  ``score`` is, 'accuracy' or 'r2' of its predict, so each split's result equals the class's own ``score`` on that
  split. Since lane w3f3-c (f305ae8281) a named ``scoring='accuracy'`` / ``'r2'`` is the public metric's float32
  kernel, while scoring=None keeps the class's own score program, so the two differ in the last bits.
- GaussianMixture's member publishes its score (the public ``score``'s own entry words).
- scoring=None on a class with no ``score``: scikit-learn's ``check_scoring`` TypeError.
- a base that is not a Mojolearn estimator: the meta door's native refusal.
- a nested meta-estimator whose own fit refuses: its words (error_score='raise'), or scikit-learn's
  all-fits-failed words (a numeric error_score).
Before, each answered False and the search kept its Python units. Each test names its fail-proof edit.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import model_selection as ms


def _regression(n=40, d=3, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1] + 0.1 * rng.normal(size=n)).astype(np.float32)
    return X, y


def _classification(n=40, d=3, seed=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.int64)
    return X, y


def _split_scores(est, grid, X, y, cv):
    # scoring=None: every split's score, candidates x folds.
    s = ms.GridSearchCV(est, grid, cv=cv, refit=False).fit(X, y)
    folds = len([k for k in s.cv_results_ if k.startswith("split") and k.endswith("_test_score")])
    return np.stack([np.asarray(s.cv_results_["split%d_test_score" % f]) for f in range(folds)], axis=1)


def _own_scores(make, grid, X, y, folds):
    # The class's own ``score`` per candidate and fold (scikit-learn's scoring=None is ``estimator.score``).
    (name, values), = grid.items()
    out = []
    for v in values:
        row = []
        for train, test in folds:
            train, test = np.asarray(train), np.asarray(test)
            row.append(make().set_params(**{name: v}).fit(X[train], y[train]).score(X[test], y[test]))
        out.append(row)
    return np.asarray(out, dtype=np.float64)


@pytest.mark.parametrize("make, grid", [
    (lambda: ml.RandomForestRegressor(n_estimators=4, random_state=0), {"max_depth": [2, 3]}),
    (lambda: ml.KNeighborsRegressor(), {"n_neighbors": [3, 5]}),
    (lambda: ml.KernelRidge(), {"alpha": [0.5, 1.0]}),
])
def test_scoring_none_regressor_without_member_score_is_r2(make, grid):
    # Fail-proof: in `_own_score` answer _OWN_ACCURACY for a regressor: the scores differ from 'r2'.
    X, y = _regression()
    folds = list(ms.KFold(3).split(X, y))
    np.testing.assert_array_equal(_split_scores(make(), grid, X, y, 3), _own_scores(make, grid, X, y, folds))


@pytest.mark.parametrize("make, grid", [
    (lambda: ml.KNeighborsClassifier(), {"n_neighbors": [3, 5]}),
    (lambda: ml.GaussianNB(), {"var_smoothing": [1e-9, 1e-6]}),
])
def test_scoring_none_classifier_without_member_score_is_accuracy(make, grid):
    # Fail-proof: in the search admission append no scorer code when the kind lacks MEMBER_CAN_SCORE (`lacking`):
    # the RUN asks the member's own score and the search raises the member's refusal.
    X, y = _classification()
    folds = list(ms.StratifiedKFold(3).split(X, y))
    np.testing.assert_array_equal(_split_scores(make(), grid, X, y, 3), _own_scores(make, grid, X, y, folds))


def test_scoring_none_gaussian_mixture_is_its_own_score():
    # Fail-proof: drop MEMBER_CAN_SCORE from core/mixture_member.mojo's capability answer: the search refuses
    # (NotImplementedError naming GaussianMixture) instead of scoring.
    X, _ = _regression(n=60)
    folds = [(np.arange(0, 40), np.arange(40, 60))]
    s = ms.GridSearchCV(ml.GaussianMixture(random_state=0), {"n_components": [1, 2]}, cv=folds,
                        refit=False).fit(X)
    for i, k in enumerate([1, 2]):
        own = ml.GaussianMixture(n_components=k, random_state=0).fit(X[:40]).score(X[40:])
        np.testing.assert_allclose(s.cv_results_["split0_test_score"][i], own, rtol=1e-6)


def test_scoring_none_elliptic_envelope_is_accuracy():
    # Fail-proof: in `_own_score` drop the "outlier_detector" case: the search refuses with NotImplementedError.
    X, _ = _regression(n=40)
    y = np.where(np.abs(X[:, 0]) < 1.5, 1, -1)
    folds = [(np.arange(14, 40), np.arange(0, 14)), (np.r_[0:14, 27:40], np.arange(14, 27)),
             (np.arange(0, 27), np.arange(27, 40))]
    make = lambda: ml.EllipticEnvelope(random_state=0)  # noqa: E731
    grid = {"contamination": [0.1, 0.2]}
    np.testing.assert_array_equal(_split_scores(make(), grid, X, y, folds), _own_scores(make, grid, X, y, folds))


def test_scoring_none_class_without_score_raises_sklearn_words():
    # Fail-proof: delete the `_no_score_refusal` call in `_admit_search`.
    X, _ = _regression()
    with pytest.raises(TypeError, match="If no scoring is specified, the estimator passed should have a 'score' method"):
        ms.GridSearchCV(ml.PCA(), {"n_components": [1, 2]}, cv=3).fit(X)


def test_foreign_base_is_refused_natively():
    # Fail-proof: answer False again at the base-plan check of `_admit_search` (row :3442).
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    X, y = _regression()
    with pytest.raises(TypeError, match="is not a Mojolearn estimator"):
        ms.GridSearchCV(sklearn_linear.Ridge(), {"alpha": [0.5, 1.0]}, cv=3, scoring="r2").fit(X, y)


class _UserSplitter:
    """A user's own splitter object: a Python callback the Stacking admission refuses before it is called."""

    def split(self, X, y=None, groups=None):  # never called
        raise AssertionError("a user's splitter is never called")

    def get_n_splits(self, X=None, y=None, groups=None):
        return 2


def _nested_stack():
    return ml.StackingRegressor([("r", ml.Ridge())], cv=_UserSplitter())


def test_nested_meta_refusal_raises_its_words():
    # Fail-proof: drop the failure record in `ensemble_member_admit`'s work-list (`done.failed_words.append`).
    X, y = _regression()
    with pytest.raises(TypeError, match="Python callbacks cannot execute"):
        ms.GridSearchCV(_nested_stack(), {"passthrough": [False, True]}, cv=3, scoring="r2",
                        error_score="raise").fit(X, y)


def test_nested_meta_refusal_every_fit_fails_in_sklearn_words():
    # Fail-proof: in `_search_nested_refusal` return True when every candidate is refused.
    X, y = _regression()
    with pytest.raises(ValueError, match="All the 6 fits failed"):
        ms.GridSearchCV(_nested_stack(), {"passthrough": [False, True]}, cv=3, scoring="r2").fit(X, y)
