# SPDX-License-Identifier: Apache-2.0
"""The meta-estimators' member fits with no fit door left (lane mf-del8, 2026-11-23).

Before, a member the native member set did not take (a draw or a cv fold lacking a class, a non-finite X, a y the
float32 admission refused, a member the set refused) was fitted by its member's Python fit door, which raised the
estimator's words or fitted the classes it was given. Now:
  * a bagging draw or a cv fold that lacks a class is fitted natively on the classes it holds (THE UNIT CLASSES,
    core/member_unit_classes.mojo), its member labelled with the ensemble codes it holds, as scikit-learn fits it;
  * every refusal is raised natively before any fit, in its refuser's words (scikit-learn's check_array /
    check_consistent_length words, the forest door's TREES_NONFINITE, the member's own words).
Each test names the edit that must make it fail (its fail-proof).
"""
import numpy as np
import pytest

import mojolearn as ml


def _imbalanced(n=60, d=4, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    y[:3] = 2  # three rows of class 2: small draws often lack it
    y[3:20] = 1
    X[:, 0] += y
    return X, y


#: Bagging's max_features for the per-member route: a bag of best-splitter trees over EVERY feature, unweighted and
#: without out-of-bag scoring, is fitted as ONE forest (`_bagging_batched_ok`, bindings/trees_public_boost_door.mojo),
#: whose trees hold every class. These tests are about the per-member route (`_bagging_fit_bags`' native set), which a
#: bag drawing fewer features than X has reaches.
PER_MEMBER = 3


def test_bagging_draws_lacking_a_class_fit_the_classes_they_hold():
    # Fail-proof: make member_unit_classes return an empty list always (core/member_unit_classes.mojo): the set's
    # cover pass is skipped, the lacking draws are fitted with k classes and their classes_ become [0, 1, 2].
    X, y = _imbalanced()
    m = ml.BaggingClassifier(ml.DecisionTreeClassifier(random_state=0), n_estimators=12, max_samples=6,
                             max_features=PER_MEMBER, bootstrap=True, random_state=0).fit(X, y)
    narrowed = 0
    for est in m.estimators_:
        held = np.asarray(est.classes_)
        assert set(held.tolist()) <= {0, 1, 2}
        assert list(held) == sorted(held.tolist())
        narrowed += int(len(held) < 3)
    assert narrowed > 0, "no draw lacked a class: the fixture no longer exercises the re-code"
    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    assert set(np.unique(m.predict(X)).tolist()) <= {0, 1, 2}


def test_bagging_lacking_draws_same_answer_every_n_jobs():
    # Fail-proof: drop `set.run.skip_lacking = classifier` in _bagging_fit_bags: the set's own cover pass then
    # raises MEMBER_SET_LACKING for the first lacking draw.
    X, y = _imbalanced()
    one = ml.BaggingClassifier(ml.DecisionTreeClassifier(random_state=0), n_estimators=8, max_samples=6,
                               max_features=PER_MEMBER, random_state=1, n_jobs=1).fit(X, y)
    many = ml.BaggingClassifier(ml.DecisionTreeClassifier(random_state=0), n_estimators=8, max_samples=6,
                                max_features=PER_MEMBER, random_state=1, n_jobs=4).fit(X, y)
    np.testing.assert_array_equal(one.predict_proba(X), many.predict_proba(X))


def test_bagging_nonfinite_x_is_the_forest_doors_words():
    # Fail-proof: delete the `raise Error(TREES_NONFINITE)` in _bagging_fit_bags: the tree members read NaN words.
    X, y = _imbalanced()
    X[5, 1] = np.nan
    with pytest.raises(Exception, match="contains NaN or infinity"):
        ml.BaggingClassifier(ml.DecisionTreeClassifier(), n_estimators=3, random_state=0).fit(X, y)


def test_voting_regressor_nonfinite_x_is_refused_before_any_fit():
    # Fail-proof: delete the admitted_finite_refuse for X in _fit_units (bindings/trees_public_meta_door.mojo).
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3)).astype(np.float32)
    y = rng.normal(size=30).astype(np.float32)
    X[2, 0] = np.inf
    m = ml.VotingRegressor([("t", ml.DecisionTreeRegressor(random_state=0)),
                            ("u", ml.DecisionTreeRegressor(random_state=1))])
    with pytest.raises(Exception, match="Input X contains NaN or infinity"):
        m.fit(X, y)
    assert not hasattr(m, "estimators_") or len(m.estimators_) == 0


def test_voting_regressor_inconsistent_lengths_are_sklearns_words():
    # Fail-proof: delete the inconsistent-length raise in _voting_fit: the set reads y past its end.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3)).astype(np.float32)
    y = rng.normal(size=29).astype(np.float32)
    m = ml.VotingRegressor([("t", ml.DecisionTreeRegressor(random_state=0))])
    with pytest.raises(ValueError, match=r"inconsistent numbers of samples: \[30, 29\]"):
        m.fit(X, y)


def test_voting_regressor_nonfinite_y_is_refused():
    # Fail-proof: delete the admitted_finite_refuse for y in _fit_units.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 3)).astype(np.float32)
    y = rng.normal(size=30).astype(np.float32)
    y[4] = np.nan
    m = ml.VotingRegressor([("t", ml.DecisionTreeRegressor(random_state=0))])
    with pytest.raises(Exception, match="Input y contains NaN or infinity"):
        m.fit(X, y)


def test_calibrated_has_no_door_units():
    # The door fold units are gone: a served base still calibrates (the typed RUN), and the module exposes no
    # door-unit entry. Fail-proof: restore `_CalUnits` and route a base to it: this test does not see it, the
    # digests (tools/mf_del8_digests.py) do; this case guards the RUN path after the deletion.
    rng = np.random.default_rng(4)
    X = rng.normal(size=(90, 4)).astype(np.float32)
    y = (np.arange(90) % 3).astype(np.int64)
    X[:, 0] += y
    m = ml.CalibratedClassifierCV(ml.LogisticRegression(), cv=3).fit(X, y)
    proba = np.asarray(m.predict_proba(X))  # an in-package caller receives the Array (parallel_public_result_door)
    assert proba.shape == (90, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
