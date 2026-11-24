# SPDX-License-Identifier: Apache-2.0
"""Settings lane mf-ens6 (2026-11-23) moved from the classes' own doors into their member routes, and the
refusals it raises natively where scikit-learn itself fails.

A member is reached as the refit estimator of a one-split, one-point GridSearchCV (train = every row). Its
fitted attributes and outputs equal the estimator's own fit. Refusals are checked by their words.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn.model_selection import GridSearchCV


def _member(est, X, y, scoring=None, error_score=np.nan):
    every = np.arange(len(X))
    params = est.get_params()
    name = sorted(params)[0]
    search = GridSearchCV(est, {name: [params[name]]}, cv=[(every, every)], scoring=scoring, refit=True,
                          error_score=error_score)
    search.fit(X, y)
    return search.best_estimator_


def test_recurrent_builtin_schedule_member_equals_door():
    # A built-in lr_schedule is DATA: the member fills each step's rate with the schedule's native series.
    x = np.arange(48, dtype=np.float32).reshape(8, 3, 2) / 48
    y = x[:, -1, 0].copy()

    def make():
        return mojolearn.RNNRegressor(hidden_size=4, max_epochs=2, batch_size=4, random_state=7,
                                      lr_schedule=mojolearn.StepLR(0.01, step_size=2, gamma=0.5),
                                      numeric_mode="identical")

    door = make().fit(x, y)
    member = _member(make(), x, y, scoring="r2")
    np.testing.assert_array_equal(np.asarray(member.params_), np.asarray(door.params_))
    np.testing.assert_array_equal(np.asarray(member.loss_curve_), np.asarray(door.loss_curve_))
    np.testing.assert_array_equal(np.asarray(member.predict(x)), np.asarray(door.predict(x)))


def test_recurrent_callable_schedule_refused_natively():
    x = np.arange(48, dtype=np.float32).reshape(8, 3, 2) / 48
    y = x[:, -1, 0].copy()
    est = mojolearn.RNNRegressor(hidden_size=4, max_epochs=1, batch_size=4, random_state=7)
    est.lr_schedule = lambda t: 0.01
    # The member's admission refuses the callback before it is called. Since fc40f1e06e (w3f5-b) a candidate
    # whose own admission refuses is a failed fit under a numeric error_score, as scikit-learn's _fit_and_score
    # scores a fit that raises: error_score='raise' raises the refusal itself, the default nan scikit-learn's
    # all-fits-failed words over it.
    with pytest.raises(TypeError, match="built-in native schedule"):
        _member(est, x, y, scoring="r2", error_score="raise")
    with pytest.raises(ValueError, match=r"(?s)All the 1 fits failed.*built-in native schedule"):
        _member(est, x, y, scoring="r2")


def test_quantile_subsample_below_one_is_sklearns_refusal():
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    with pytest.raises(ValueError, match=r"The 'subsample' parameter of QuantileTransformer must be an int in the "
                                         r"range \[1, inf\) or None\. Got 0 instead\."):
        mojolearn.QuantileTransformer(n_quantiles=5, subsample=0).fit(X)


def test_kbins_subsample_and_random_state():
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    with pytest.raises(ValueError, match=r"The 'subsample' parameter of KBinsDiscretizer must be an int"):
        mojolearn.KBinsDiscretizer(n_bins=3, subsample=2.5).fit(X)
    with pytest.raises(ValueError, match=r"The 'random_state' parameter of KBinsDiscretizer"):
        mojolearn.KBinsDiscretizer(n_bins=3, subsample=10, random_state=1.5).fit(X)
    # A RandomState is read only when the rows are drawn: no draw, a fit; a draw, the door's words.
    rng = np.random.RandomState(0)
    mojolearn.KBinsDiscretizer(n_bins=3, subsample=100, random_state=rng).fit(X)
    with pytest.raises(TypeError, match="pass an integer random_state"):
        mojolearn.KBinsDiscretizer(n_bins=3, subsample=10, random_state=rng).fit(X)


def test_calibrated_label_propagation_binary_marker_is_sklearns_failure():
    X = np.arange(40, dtype=np.float32).reshape(20, 2) / 40
    y = np.where(np.arange(20) % 2 == 0, 1, -1)
    est = mojolearn.CalibratedClassifierCV(mojolearn.LabelPropagation(), ensemble=False, cv=2)
    with pytest.raises(IndexError, match="index 1 is out of bounds for axis 1 with size 1"):
        est.fit(X, y)

