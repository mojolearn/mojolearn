# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RandomForest fit notices are Python UserWarnings and never touch stdout.

0.8.63 printed cuML's three fit notices ("WARN: ...") to stdout from the
native fit, which corrupted any caller's own stdout (it broke
`verify --self-test --json`). The fit now records them as a bit mask
(`ensemble/rf_notices.mojo`) and `bindings/trees_public_forest.mojo` raises
each through the native warnings path (`bindings/classical_warnings.mojo`).
"""

import warnings

import numpy as np
import pytest

import mojolearn as ml

BINS = ("The number of bins, `n_bins` is greater than the number of samples used for training. "
        "Changing `n_bins` to number of training samples.")
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


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_n_bins_above_rows_warns(capfd, cls):
    messages = _fit_messages(capfd, cls, n_bins=128)
    assert messages == [(UserWarning, BINS)]


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_max_samples_without_bootstrap_warns(capfd, cls):
    messages = _fit_messages(capfd, cls, n_bins=16, bootstrap=False, max_samples=0.5)
    assert messages == [(UserWarning, SAMPLES)]


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_both_notices_in_fit_order(capfd, cls):
    messages = _fit_messages(capfd, cls, n_bins=128, bootstrap=False, max_samples=0.5)
    assert messages == [(UserWarning, BINS), (UserWarning, SAMPLES)]


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
