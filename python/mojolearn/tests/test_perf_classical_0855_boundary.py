"""Refusals pinned by the 2026-11-13 classical performance fixes (lane
fable-perf-classical-0855): the finite tests that moved off the host onto the
kernel entries still refuse in scikit-learn's wording.

- LinearRegression.fit: X and y are tested by the centered-TSQR entry
  (x_decomp ols_tsqr_factor, device or host column) on the intercept route;
  the door scans X on the host only for the other routes.
- ARDRegression.predict: the decision entry's status word now rides home
  after the result copy (x_linear/device.mojo _decision_status_home).
"""
import numpy as np
import pytest

import mojolearn as ml

__mojolearn_user_boundary__ = True


def _xy(rows=96, cols=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((rows, cols), dtype=np.float32)
    y = (x @ np.arange(1, cols + 1, dtype=np.float32) + 0.5).astype(np.float32)
    return x, y


# rows: one TSQR block, and several blocks (the tree combine runs). The fit
# takes whichever column the build selects (Apple Metal on the Mac wheel,
# NVIDIA or AMD on the Linux GPU wheels, the host column on the CPU wheel);
# every column refuses with the admission message before the factor's own
# "R entry ... is NaN" refusal.
@pytest.mark.parametrize("rows", [96, 3 * 4096 + 17])
@pytest.mark.parametrize("bad", ["X", "y"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_linear_regression_fit_refuses_non_finite_on_the_entry(bad, value, rows):
    x, y = _xy(rows=rows)
    if bad == "X":
        x[7, 2] = value
    else:
        y[7] = value
    with pytest.raises(ValueError, match="Input %s contains NaN or infinity" % bad):
        ml.LinearRegression().fit(x, y)


def test_linear_regression_fit_refuses_non_finite_x_before_y():
    # Both bad: X's status word is read first, as the host column scans X first.
    x, y = _xy()
    x[5, 1] = np.nan
    y[9] = np.nan
    with pytest.raises(ValueError, match="Input X contains NaN or infinity"):
        ml.LinearRegression().fit(x, y)


def test_linear_regression_fit_refuses_non_finite_x_without_intercept():
    # The no-intercept route reads X on the host: the door's own scan.
    x, y = _xy()
    x[3, 0] = np.nan
    with pytest.raises(ValueError, match="Input X contains NaN or infinity"):
        ml.LinearRegression(fit_intercept=False).fit(x, y)


def test_linear_regression_fit_refuses_non_finite_x_with_weights():
    x, y = _xy()
    x[3, 0] = np.inf
    w = np.ones(len(y), dtype=np.float32)
    with pytest.raises(ValueError, match="Input X contains NaN or infinity"):
        ml.LinearRegression().fit(x, y, sample_weight=w)


def test_linear_regression_fit_finite_input_is_fitted():
    x, y = _xy()
    model = ml.LinearRegression().fit(x, y)
    assert model.coef_.shape == (x.shape[1],)
    assert np.all(np.isfinite(np.asarray(model.coef_)))


def test_ard_predict_refuses_non_finite_x_after_the_result_copy():
    x, y = _xy(rows=64, cols=3)
    model = ml.ARDRegression().fit(x, y)
    xq = x[:8].copy()
    xq[2, 1] = np.nan
    with pytest.raises(ValueError, match="contains NaN or infinity"):
        model.predict(xq)
    finite = np.asarray(model.predict(x[:8]))
    assert finite.shape == (8,)
    assert np.all(np.isfinite(finite))
