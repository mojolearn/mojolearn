# SPDX-License-Identifier: Apache-2.0
"""Lane mf-enc6 (2026-11-23): the OrdinalEncoder and OneHotEncoder settings that moved from the classes' own doors
into their member routes, against the estimators' own doors.

Each encoder is fitted twice: by its own fit (the door) and as a GridSearchCV member (one given split, train = every
row, a one-point grid, refit=True: the refit runs the member, then PACK writes the fitted attributes). The refit
estimator's fitted attributes, its transform and its inverse_transform equal the door's. Covered: given categories,
the infrequent grouping (min_frequency, max_categories), a drop given per column (also with the grouping),
handle_unknown='warn' (and 'ignore' / 'infrequent_if_exist' with a drop, whose transforms warn), and the FAST image's
packed category route. Refusals keep the door's words.
"""
import warnings

import numpy as np
import pytest

import mojolearn
from mojolearn.model_selection import GridSearchCV


def _data(n=60, seed=3):
    rng = np.random.default_rng(seed)
    a = rng.choice([0.0, 1.0, 2.0, 3.0, 4.0], size=n, p=[0.4, 0.3, 0.2, 0.06, 0.04])
    b = rng.choice([10.0, 20.0, 30.0], size=n, p=[0.6, 0.35, 0.05])
    c = rng.choice([-1.0, 5.0], size=n)
    return np.stack([a, b, c], axis=1).astype(np.float32)


def _member(est, X):
    """The refit estimator of a one-split, one-point search (fitted as a member)."""
    every = np.arange(len(X))
    params = est.get_params()
    name = sorted(params)[0]
    # scoring=None needs the class's own `score` (scikit-learn 1.9 metrics/_scorer.py `check_scoring`, a TypeError
    # before any fit; providers/python/member_ensemble_boundary_door.mojo `_no_score_refusal`). A class without one
    # is searched by a scorer whose response it lacks: every score is error_score (nan) and the best is refitted
    # through the member, as scikit-learn does (lane w3f2-cv).
    scoring = None if getattr(est, "score", None) is not None else "r2"
    search = GridSearchCV(est, {name: [params[name]]}, cv=[(every, every)], scoring=scoring, refit=True)
    search.fit(X)
    return search.best_estimator_


def _lists(a, b):
    assert (a is None) == (b is None)
    if a is None:
        return
    assert len(a) == len(b)
    for left, right in zip(a, b):
        assert (left is None) == (right is None)
        if left is not None:
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def _same(member, door, X, inverse=True):
    _lists(member.categories_, door.categories_)
    for name in ("infrequent_categories_", "drop_idx_"):
        if hasattr(door, name):
            _lists(getattr(member, name), getattr(door, name))
    assert member.n_features_in_ == door.n_features_in_
    got = np.asarray(member.transform(X))
    np.testing.assert_array_equal(got, np.asarray(door.transform(X)))
    if inverse:
        np.testing.assert_array_equal(np.asarray(member.inverse_transform(got)),
                                      np.asarray(door.inverse_transform(got)))


def _both(make, X, inverse=True):
    door = make().fit(X)
    member = _member(make(), X)
    _same(member, door, X, inverse)
    return member, door


GIVEN = [[0.0, 1.0, 2.0, 3.0, 4.0, 7.0], [10.0, 20.0, 30.0], [-1.0, 5.0]]


def test_ordinal_given_categories():
    X = _data()
    _both(lambda: mojolearn.OrdinalEncoder(categories=GIVEN), X)
    # a training value outside its list is the door's refusal, under 'error'
    with pytest.raises(Exception, match=r"found unknown categories in column\(s\) \[0\] during fit"):
        _member(mojolearn.OrdinalEncoder(categories=[[0.0, 1.0], GIVEN[1], GIVEN[2]]), X)
    with pytest.raises(Exception, match="unsorted categories are not supported"):
        _member(mojolearn.OrdinalEncoder(categories=[[1.0, 0.0, 2.0, 3.0, 4.0], GIVEN[1], GIVEN[2]]), X)
    _both(lambda: mojolearn.OrdinalEncoder(categories=[[0.0, 1.0], GIVEN[1], GIVEN[2]],
                                           handle_unknown="use_encoded_value", unknown_value=-1), X, inverse=False)


def test_onehot_given_categories():
    X = _data()
    _both(lambda: mojolearn.OneHotEncoder(categories=GIVEN, sparse_output=False), X)


@pytest.mark.parametrize("kwargs", [{"min_frequency": 4}, {"min_frequency": 0.1}, {"max_categories": 2},
                                    {"min_frequency": 3, "max_categories": 3}])
def test_ordinal_infrequent(kwargs):
    X = _data()
    member, door = _both(lambda: mojolearn.OrdinalEncoder(**kwargs), X, inverse=False)
    _lists(member._grouping, door._grouping)
    _lists(member._infrequent, door._infrequent)


@pytest.mark.parametrize("kwargs", [{"min_frequency": 4}, {"max_categories": 2},
                                    {"min_frequency": 4, "handle_unknown": "infrequent_if_exist"},
                                    {"min_frequency": 4, "categories": GIVEN}])
def test_onehot_infrequent(kwargs):
    X = _data()
    member, door = _both(lambda: mojolearn.OneHotEncoder(sparse_output=False, **kwargs), X, inverse=False)
    _lists(member._grouping, door._grouping)
    _lists(member._drop_after, door._drop_after)


@pytest.mark.parametrize("kwargs", [{"drop": [1.0, 20.0, -1.0]}, {"drop": np.array([0.0, 10.0, 5.0])},
                                    {"drop": [0.0, 10.0, 5.0], "min_frequency": 4},
                                    {"drop": "first", "min_frequency": 4}, {"drop": "if_binary", "max_categories": 2}])
def test_onehot_drop(kwargs):
    X = _data()
    grouped = "min_frequency" in kwargs or "max_categories" in kwargs
    member, door = _both(lambda: mojolearn.OneHotEncoder(sparse_output=False, **kwargs), X, inverse=not grouped)
    assert member._drop_after == door._drop_after


def test_onehot_drop_refusals_are_the_doors():
    X = _data()
    with pytest.raises(Exception, match=r"were not found in the training data.\nCategory: 9.0, Feature: 0"):
        _member(mojolearn.OneHotEncoder(drop=[9.0, 10.0, 5.0], sparse_output=False), X)
    with pytest.raises(Exception, match="because it is infrequent"):
        _member(mojolearn.OneHotEncoder(drop=[4.0, 10.0, 5.0], min_frequency=4, sparse_output=False), X)
    with pytest.raises(Exception, match="should have length equal to the number of features"):
        _member(mojolearn.OneHotEncoder(drop=[0.0, 10.0], sparse_output=False), X)


@pytest.mark.parametrize("kwargs", [{"handle_unknown": "warn"}, {"handle_unknown": "ignore", "drop": "first"},
                                    {"handle_unknown": "infrequent_if_exist", "drop": "first", "min_frequency": 4}])
def test_onehot_warning_settings(kwargs):
    X = _data()
    member, door = _both(lambda: mojolearn.OneHotEncoder(sparse_output=False, **kwargs), X, inverse=False)
    unknown = X[:4].copy()
    unknown[:, 0] = 99.0
    with warnings.catch_warnings(record=True) as seen_door:
        warnings.simplefilter("always")
        want = np.asarray(door.transform(unknown))
    with warnings.catch_warnings(record=True) as seen_member:
        warnings.simplefilter("always")
        got = np.asarray(member.transform(unknown))
    np.testing.assert_array_equal(got, want)
    assert [str(w.message) for w in seen_member] == [str(w.message) for w in seen_door]
    assert any("Found unknown categories in columns [0]" in str(w.message) for w in seen_door)


def test_fast_packed_category_route():
    # The FAST image built with x_prep_cls2_cat runs the packed category route first; the member and the door
    # read the same route word and fall back to the scan alike.
    try:
        previous = mojolearn.set_numeric_mode("fast")
    except Exception as error:  # no fast set in this install
        pytest.skip("no fast set: " + str(error))
    try:
        X = _data()
        _both(lambda: mojolearn.OrdinalEncoder(), X)
        _both(lambda: mojolearn.OneHotEncoder(sparse_output=False), X)
    finally:
        mojolearn.set_numeric_mode(previous)
