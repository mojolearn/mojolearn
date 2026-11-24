# SPDX-License-Identifier: Apache-2.0
"""resident() scope (lane ma-core round 6, arena): inside `with resident():` a cross-validation over a member
uploads X once for every fold, gives the scores it gives outside the scope, and leaves no pin on X.

The resident store is device state, so every test here is `gpu`. The scope's counters are read with
`resident.stats()` ({prefix: (uploads, hits)} of the active scope).
"""
import sys

import numpy as np
import pytest

from _proving import assert_same, binary, cls, data

pytestmark = pytest.mark.gpu

X, Y, C = data(rows=240)


def _resident():
    from mojolearn import _arena_io

    return _arena_io.resident


def _scores_in_scope(estimator, X_, y, cv, **more):
    resident = _resident()
    with resident():
        scores = np.asarray(cls("cross_val_score")(estimator, X_, y, cv=cv, **more))
        stats = resident.stats()
    return scores, stats


def test_gaussian_nb_cross_val_score_in_resident_equals_outside_and_uploads_once():
    outside = np.asarray(cls("cross_val_score")(cls("GaussianNB")(), X, C, cv=5))
    before = sys.getrefcount(X)
    inside, stats = _scores_in_scope(cls("GaussianNB")(), X, C, 5)
    assert_same(inside, outside, "scores in resident()")
    assert sys.getrefcount(X) == before, "resident() left a pin on X"
    assert stats, "the scope recorded no cache: the member did not go through resident()"
    uploads = sum(u for u, _ in stats.values())
    hits = sum(h for _, h in stats.values())
    assert 1 <= uploads <= 2, f"X (and at most y) are uploaded once for every fold: {stats}"
    assert hits >= 4, f"the later folds read the uploaded X: {stats}"
    # The number of folds does not change the uploads.
    _, three = _scores_in_scope(cls("GaussianNB")(), X, C, 3)
    assert sum(u for u, _ in three.values()) == uploads, (three, stats)


def _validate_in_scope(estimator, X_, y, cv, **more):
    resident = _resident()
    with resident():
        scores = np.asarray(cls("cross_validate")(estimator, X_, y, cv=cv, **more)["test_score"])
        stats = resident.stats()
    return scores, stats


def test_failing_unit_in_resident_gives_error_score_and_releases_the_pin():
    # A number for error_score ("error_score : 'raise' or numeric", cross_validate's and cross_val_score's
    # docstrings): the failing units score it, with scikit-learn's warnings (lane w3f3-b).
    bad = X.copy()
    bad[5, 2] = np.nan
    outside = np.asarray(cls("cross_validate")(cls("GaussianNB")(), bad, binary(C), cv=3,
                                               error_score=0.0)["test_score"])
    before = sys.getrefcount(bad)
    inside, _ = _validate_in_scope(cls("GaussianNB")(), bad, binary(C), 3, error_score=0.0)
    assert_same(inside, outside, "error_score in resident()")
    assert np.any(inside == 0.0)
    assert sys.getrefcount(bad) == before, "a failing unit left a pin on X"


def test_numeric_error_score_forgives_a_failing_fit_and_a_failing_score_and_raise_raises():
    # cross_validate's docstring: "'raise' propagates a fit or scoring error; a number is recorded as the fold's
    # score instead" (scikit-learn's _fit_and_score). Split A has the NaN row in its training rows (the fit fails),
    # split B in its test rows (the fit works, the score fails), split C has no NaN row (a real score).
    bad = X.copy()
    bad[5, 2] = np.nan
    rows = np.arange(len(bad))
    clean = rows[rows != 5]
    splits = [(rows[:160], rows[160:]), (rows[80:], rows[:80]), (clean[80:], clean[:80])]
    for name in ("GaussianNB", "Ridge"):
        y = binary(C) if name == "GaussianNB" else Y
        scores = np.asarray(cls("cross_validate")(cls(name)(), bad, y, cv=splits, error_score=-7.0)["test_score"])
        assert scores[0] == -7.0 and scores[1] == -7.0, (name, scores)
        assert np.isfinite(scores[2]) and scores[2] != -7.0, (name, scores)
        with pytest.raises(ValueError):
            cls("cross_validate")(cls(name)(), bad, y, cv=splits[1:2], error_score="raise")


def test_nested_resident_scope_shares_the_outer_uploads():
    resident = _resident()
    with resident():
        first = np.asarray(cls("cross_val_score")(cls("GaussianNB")(), X, C, cv=3))
        with resident():
            second = np.asarray(cls("cross_val_score")(cls("GaussianNB")(), X, C, cv=3))
        stats = resident.stats()
    assert_same(second, first, "nested scope")
    assert sum(u for u, _ in stats.values()) <= 2, f"the nested scope uploaded X again: {stats}"
