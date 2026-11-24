# SPDX-License-Identifier: Apache-2.0
"""IterativeImputer over mojolearn's own DecisionTreeRegressor(splitter="random") (lane w3f2-cal).

Before, the imputer refused it ("requires a mojolearn estimator; got DecisionTreeRegressor"): the tree family's
member admission does not publish a random-splitter tree. scikit-learn accepts it. Now the step member is the tree's
inner one-tree ExtraTrees forest (what the tree's own fit grows; bindings/prep_iterimp_native_door.mojo `_member`),
each step packs into a tree clone's inner forest, which the clone adopts, and transform predicts through it.
Fail-proof: drop the `tree_mark.family == MEMBER_FAMILY_TREE` branch in `_member`: every test here raises the old
TypeError.
"""
import numpy as np

import mojolearn as ml


def _two_columns(n=200, seed=3):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n).astype(np.float32)
    b = (2.0 * a + 0.1 * rng.standard_normal(n)).astype(np.float32)
    X = np.stack([a, b], axis=1)
    missing = rng.random(n) < 0.25
    X[missing, 1] = np.nan
    return np.ascontiguousarray(X), missing


def test_one_step_equals_the_tree_fitted_alone():
    # skip_complete: only column 1 gets a step; max_iter=1: one round. Its estimator is fitted on column 0 of the
    # rows where column 1 is observed and predicts the missing rows: the same rows, settings and seed as the tree
    # fitted on its own, so the same bits.
    X, missing = _two_columns()
    imp = ml.IterativeImputer(estimator=ml.DecisionTreeRegressor(splitter="random", random_state=0), max_iter=1,
                              skip_complete=True, random_state=0)
    out = np.asarray(imp.fit_transform(X))
    alone = ml.DecisionTreeRegressor(splitter="random", random_state=0).fit(X[~missing, :1], X[~missing, 1])
    expect = np.asarray(alone.predict(X[missing, :1])).astype(np.float32)
    np.testing.assert_array_equal(out[missing, 1].astype(np.float32), expect)
    np.testing.assert_array_equal(out[~missing], X[~missing])
    assert len(imp.imputation_sequence_) == 1
    step = imp.imputation_sequence_[0][2]
    assert type(step) is ml.DecisionTreeRegressor and step.splitter == "random"


def test_transform_repeats_fit_transform():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((300, 2))
    X = (Z @ rng.standard_normal((2, 4)) + 0.1 * rng.standard_normal((300, 4))).astype(np.float32)
    X[rng.random(X.shape) < 0.15] = np.nan
    X = np.ascontiguousarray(X)
    imp = ml.IterativeImputer(estimator=ml.DecisionTreeRegressor(splitter="random", random_state=0), max_iter=3,
                              random_state=0)
    first = np.asarray(imp.fit_transform(X))
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(np.asarray(imp.transform(X)), first)
    for _, _, est in imp.imputation_sequence_:
        assert type(est) is ml.DecisionTreeRegressor


def test_random_tree_classifier_step():
    # A classifier step on an integer column: the random-splitter DecisionTreeClassifier through its inner forest.
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 3, 240)
    X = np.stack([labels, labels + 1, labels % 2], axis=1).astype(np.float32)
    X[rng.random(240) < 0.3, 0] = np.nan
    X = np.ascontiguousarray(X)
    imp = ml.IterativeImputer(estimator=ml.DecisionTreeClassifier(splitter="random", random_state=0), max_iter=2,
                              skip_complete=True, random_state=0)
    out = np.asarray(imp.fit_transform(X))
    assert set(np.unique(out[:, 0]).tolist()) <= {0.0, 1.0, 2.0}
    np.testing.assert_array_equal(np.asarray(imp.transform(X)), out)
