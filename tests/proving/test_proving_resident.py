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


def test_failing_unit_in_resident_gives_error_score_and_releases_the_pin():
    bad = X.copy()
    bad[5, 2] = np.nan
    outside = np.asarray(cls("cross_val_score")(cls("GaussianNB")(), bad, binary(C), cv=3, error_score=0.0))
    before = sys.getrefcount(bad)
    inside, _ = _scores_in_scope(cls("GaussianNB")(), bad, binary(C), 3, error_score=0.0)
    assert_same(inside, outside, "error_score in resident()")
    assert np.any(inside == 0.0)
    assert sys.getrefcount(bad) == before, "a failing unit left a pin on X"


def test_nested_resident_scope_shares_the_outer_uploads():
    resident = _resident()
    with resident():
        first = np.asarray(cls("cross_val_score")(cls("GaussianNB")(), X, C, cv=3))
        with resident():
            second = np.asarray(cls("cross_val_score")(cls("GaussianNB")(), X, C, cv=3))
        stats = resident.stats()
    assert_same(second, first, "nested scope")
    assert sum(u for u, _ in stats.values()) <= 2, f"the nested scope uploaded X again: {stats}"
