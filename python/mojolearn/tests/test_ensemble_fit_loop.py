# SPDX-License-Identifier: Apache-2.0
"""`fit_loop_` at the public boundary of the bagging and AdaBoost ensembles
(lane alm-left, 2026-11-24): after `fit` it is a tuple `(int, float)` whose
count is the members built (bagging: `n_estimators`; the tree bag is one
forest fit and reports its tree loop, a bag of another member kind clocks its
member waves) or the boosting rounds run (AdaBoost: `n_estimators` unless a
round stops the fit early, so the count equals `len(estimators_)` plus at most
the one refused round), and before `fit` it raises AttributeError like the
other fitted attributes."""
import numpy as np
import pytest
from mojolearn import (AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor,
                       DecisionTreeClassifier, DecisionTreeRegressor)

K = 3


def _data(classifier):
    rng = np.random.default_rng(0)
    X = rng.random((60, 4), dtype=np.float32)
    if classifier:
        return X, (X[:, 0] + 0.3 * rng.random(60) > 0.6).astype(np.int32)
    return X, (2.0 * X[:, 0] + X[:, 1] + 0.1 * rng.random(60)).astype(np.float32)


def _check(loop):
    assert type(loop) is tuple and len(loop) == 2
    assert type(loop[0]) is int and type(loop[1]) is float
    assert loop[1] >= 0.0


@pytest.mark.parametrize('cls,classifier', [(BaggingClassifier, True), (BaggingRegressor, False)])
def test_bagging_tree_bag_reports_members(cls, classifier):
    X, y = _data(classifier)
    model = cls(n_estimators=K, random_state=0).fit(X, y)
    _check(model.fit_loop_)
    assert model.fit_loop_[0] == K


@pytest.mark.parametrize('cls,member,classifier', [(BaggingClassifier, DecisionTreeClassifier, True),
                                                    (BaggingRegressor, DecisionTreeRegressor, False)])
def test_bagging_member_waves_report_members(cls, member, classifier):
    # max_features below every column: not the one-forest bag, so the member waves run
    X, y = _data(classifier)
    model = cls(estimator=member(max_depth=3, random_state=0), n_estimators=K, max_features=2,
                random_state=0).fit(X, y)
    _check(model.fit_loop_)
    assert model.fit_loop_[0] == K == len(model.estimators_)


@pytest.mark.parametrize('cls,member,classifier', [(AdaBoostClassifier, DecisionTreeClassifier, True),
                                                    (AdaBoostRegressor, DecisionTreeRegressor, False)])
def test_adaboost_reports_rounds(cls, member, classifier):
    X, y = _data(classifier)
    model = cls(estimator=member(max_depth=1, random_state=0), n_estimators=K, random_state=0).fit(X, y)
    _check(model.fit_loop_)
    kept = len(model.estimators_)
    assert kept <= model.fit_loop_[0] <= min(K, kept + 1)


@pytest.mark.parametrize('cls', [BaggingClassifier, BaggingRegressor, AdaBoostClassifier, AdaBoostRegressor])
def test_fit_loop_before_fit_raises(cls):
    model = cls(n_estimators=K, random_state=0)
    with pytest.raises(AttributeError):
        model.fit_loop_
    assert not hasattr(model, 'fit_loop_')
