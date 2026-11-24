# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RandomForest fit notices are Python UserWarnings and never touch stdout.

0.8.63 printed cuML's three fit notices ("WARN: ...") to stdout from the
native fit, which corrupted any caller's own stdout (it broke
`verify --self-test --json`). The fit now records them as a bit mask
(`ensemble/rf_notices.mojo`) and `bindings/trees_public_forest_door.mojo` raises
each through the native warnings path (`bindings/classical_warnings_unpack.mojo`).

cuML's third notice, the `n_bins` clamp ("n_bins is greater than the number
of samples"), is not raised (Andrew, 2026-11-24: noise the user cannot act
on, it fired in every nested fit, scikit-learn never emits it); the fit
still clamps the bins to the rows.
"""

import warnings

import numpy as np
import pytest

import mojolearn as ml

SAMPLES = ("If bootstrap sampling is disabled, max_samples value is ignored and whole dataset is "
           "used for building each tree")
OOB = ("Some inputs do not have OOB scores. This probably means too few trees were used to "
       "compute any reliable OOB estimates.")

ESTIMATORS = (ml.RandomForestClassifier, ml.RandomForestRegressor)


def _data(cls, n=40):
    rs = np.random.RandomState(0)
    X = rs.rand(n, 4).astype("float32")
    if cls is ml.RandomForestClassifier:
        return X, (X[:, 0] + X[:, 1] > 1).astype("int64")
    return X, (X @ np.arange(1, 5)).astype("float32")


def _fit_messages(capfd, cls, **params):
    X, y = _data(cls)
    capfd.readouterr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls(n_estimators=params.pop("n_estimators", 3), random_state=0, **params).fit(X, y)
    out, err = capfd.readouterr()
    assert out == "", out
    assert "WARN" not in err, err
    return [(w.category, str(w.message)) for w in caught]


@pytest.mark.parametrize("cls", ESTIMATORS + (ml.DecisionTreeClassifier, ml.DecisionTreeRegressor))
def test_n_bins_above_rows_is_silent(capfd, cls):
    # 128 bins over 40 rows: the fit clamps the bins to the rows and raises nothing.
    X, y = _data(ml.RandomForestClassifier if cls in (ml.RandomForestClassifier, ml.DecisionTreeClassifier)
                 else ml.RandomForestRegressor)
    capfd.readouterr()
    params = {"n_estimators": 3} if cls in ESTIMATORS else {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        cls(n_bins=128, random_state=0, **params).fit(X, y)
    out, err = capfd.readouterr()
    assert out == "", out
    assert "WARN" not in err, err


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_max_samples_without_bootstrap_warns(capfd, cls):
    messages = _fit_messages(capfd, cls, n_bins=16, bootstrap=False, max_samples=0.5)
    assert messages == [(UserWarning, SAMPLES)]


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_clamped_bins_leave_only_max_samples_notice(capfd, cls):
    messages = _fit_messages(capfd, cls, n_bins=128, bootstrap=False, max_samples=0.5)
    assert messages == [(UserWarning, SAMPLES)]


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_quiet_fit_has_no_notice(capfd, cls):
    assert _fit_messages(capfd, cls, n_bins=16) == []


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_pytest_warns_matches(capfd, cls):
    X, y = _data(cls)
    capfd.readouterr()
    with pytest.warns(UserWarning, match="max_samples value is ignored"):
        cls(n_estimators=3, n_bins=16, bootstrap=False, max_samples=0.5, random_state=0).fit(X, y)
    assert capfd.readouterr().out == ""


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_partial_oob_warns(capfd, cls):
    # One bootstrapped tree leaves about a third of the rows out of bag; the
    # rest have no OOB score.
    messages = _fit_messages(capfd, cls, n_estimators=1, n_bins=16, oob_score=True)
    assert messages == [(UserWarning, OOB)]


NESTED = (
    # AdaBoost's own rounds (DecisionTree splitter='best', host or device rounds)
    lambda: ml.AdaBoostClassifier(n_estimators=3, random_state=0),
    lambda: ml.AdaBoostRegressor(n_estimators=3, random_state=0),
    # AdaBoost's member route (a RandomForest base) and the member set behind bagging
    lambda: ml.AdaBoostClassifier(ml.RandomForestClassifier(n_estimators=2, n_bins=128), n_estimators=2,
                                  random_state=0),
    lambda: ml.BaggingRegressor(ml.DecisionTreeRegressor(n_bins=128), n_estimators=2, random_state=0),
)


@pytest.mark.parametrize("make", NESTED)
def test_nested_fits_raise_no_n_bins_notice(capfd, make):
    # Every member fits fewer rows than its bins; no member route reports the clamp.
    model = make()
    regression = isinstance(model, (ml.AdaBoostRegressor, ml.BaggingRegressor))
    X, y = _data(ml.RandomForestRegressor if regression else ml.RandomForestClassifier)
    capfd.readouterr()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.fit(X, y)
    assert capfd.readouterr().out == ""
