# SPDX-License-Identifier: Apache-2.0
"""The meta-estimators' member fits with no fit door left (lane mf-del8, 2026-11-23).

Before, a member the native member set did not take (a draw or a cv fold lacking a class, a non-finite X, a y the
float32 admission refused, a member the set refused) was fitted by its member's Python fit door, which raised the
estimator's words or fitted the classes it was given. Now:
  * a cv fold that lacks a class is fitted natively on the classes it holds (THE UNIT CLASSES,
    core/member_unit_classes.mojo), as scikit-learn fits it; a bagging member is fitted on the whole y with its
    draw's counts as weights and holds every class (scikit-learn 1.9.1 _bagging.py:164-172; lane w3f3-h);
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
    y[:3] = 2  # three rows of class 2: small draws often lack it (a cv fold's or a bag's)
    y[3:20] = 1
    X[:, 0] += y
    return X, y


#: Bagging's max_features for the per-member route: a bag of best-splitter trees over EVERY feature, unweighted and
#: without out-of-bag scoring, is fitted as ONE forest (`_bagging_batched_ok`, bindings/trees_public_boost_door.mojo).
#: A bag drawing fewer features than X has reaches the per-member route (`_bagging_fit_bags`' native set).
PER_MEMBER = 3


@pytest.mark.parametrize("max_features", [PER_MEMBER, 1.0])
def test_bagging_members_hold_every_class(max_features):
    # scikit-learn 1.9.1 sklearn/ensemble/_bagging.py:164-172: a member that takes sample_weight (every tree member)
    # is fitted on the WHOLE y with sample_weight = np.bincount(draw), so its classes_ is every class of y even when
    # its draw lacks one (sklearn's own BaggingClassifier(DecisionTreeClassifier) over this fixture: [0, 1, 2] for
    # all 12 members). Both routes: the per-member set (PER_MEMBER) and the one forest (1.0).
    # Fail-proof: in _bagging_fit_bags hand a weight-taking member its gathered draw (row_draw / row_count kept)
    # instead of every row with the draw's counts: the set's cover pass then refuses the first draw lacking class 2.
    X, y = _imbalanced()
    m = ml.BaggingClassifier(ml.DecisionTreeClassifier(random_state=0), n_estimators=12, max_samples=6,
                             max_features=max_features, bootstrap=True, random_state=0).fit(X, y)
    for est in m.estimators_:
        assert np.asarray(est.classes_).tolist() == [0, 1, 2]
    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    assert set(np.unique(m.predict(X)).tolist()) <= {0, 1, 2}


def test_bagging_sample_weight_drives_the_draw():
    # scikit-learn 1.9.1 sklearn/ensemble/_bagging.py:82-93: a weighted fit draws row i with probability
    # w[i] / sum(w), so a zero-weight row is never drawn; the member gets the draw's counts only (:169-170). With
    # every class-2 row weighted 0, no member sees class 2 with weight: its probability column is 0 everywhere, and
    # every member still holds [0, 1, 2] (sklearn's own fit of this fixture: max proba[:, 2] = 0.0, both routes).
    # Fail-proof: in _bagging_fit_bags draw a weighted fit's rows with `_indices` (uniform) instead of
    # `_weighted_indices`: some member draws a class-2 row and proba[:, 2] is no longer 0.
    X, y = _imbalanced()
    sw = np.where(y == 2, 0.0, 1.0)
    m = ml.BaggingClassifier(ml.DecisionTreeClassifier(random_state=0), n_estimators=8, max_samples=0.5,
                             max_features=PER_MEMBER, random_state=0).fit(X, y, sample_weight=sw)
    for est in m.estimators_:
        assert np.asarray(est.classes_).tolist() == [0, 1, 2]
    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 3)
    assert float(np.max(proba[:, 2])) == 0.0


def test_bagging_max_samples_as_sklearn_resolves_it():
    # scikit-learn 1.9.1 _bagging.py:299 (max_samples=None), _bootstrap.py:44-47 (None: n; an int as given) and
    # _bagging.py:468-472: an int above n draws with replacement; without it, the refusal (sklearn's words, checked
    # against BaggingClassifier(DecisionTreeClassifier()) on this fixture).
    # Fail-proof: refuse k > n in core/bag_draws.mojo `bag_rows_count` whatever bootstrap is (the first fit raises).
    X, y = _imbalanced()
    assert ml.BaggingClassifier().max_samples is None
    m = ml.BaggingClassifier(ml.DecisionTreeClassifier(random_state=0), n_estimators=4, max_samples=100,
                             max_features=PER_MEMBER, random_state=0).fit(X, y)
    assert len(m.estimators_) == 4
    with pytest.raises(ValueError, match=r"Effective max_samples=100 must be <= n_samples=60 to be able to sample "
                                         r"without replacement\."):
        ml.BaggingClassifier(ml.DecisionTreeClassifier(), max_samples=100, bootstrap=False).fit(X, y)


def test_bagging_weighted_without_bootstrap_warns():
    # scikit-learn 1.9.1 _bagging.py:377-384: a weighted fit with bootstrap=False warns, in these words.
    # Fail-proof: drop the MEMBER_NOTICE_BAG_NO_BOOTSTRAP notice in _bagging_fit_bags: no warning.
    X, y = _imbalanced()
    with pytest.warns(UserWarning, match=r"When fitting BaggingClassifier with sample_weight it is recommended to "
                                         r"use bootstrap=True, got False\."):
        ml.BaggingClassifier(ml.DecisionTreeClassifier(random_state=0), n_estimators=3, max_samples=0.5,
                             bootstrap=False, random_state=0).fit(X, y, sample_weight=np.ones(len(y)))


def test_bagging_same_answer_every_n_jobs():
    # Fail-proof: draw member k's rows on stream 2 * (k - done) instead of 2 * k in _bagging_fit_bags: the waves
    # differ in width between n_jobs=1 and n_jobs=4 and so do the draws.
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
