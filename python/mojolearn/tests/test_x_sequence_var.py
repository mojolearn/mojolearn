# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""VAR against a float64 NumPy least-squares restatement of statsmodels'
estimator; parity with statsmodels itself (within 1e-5 on params, sigma_u and
forecasts) was checked on the lane's pod."""
import numpy as np

import mojolearn as ml


def _y(n=300, seed=0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.2, 0.3]])
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t] = 0.5 + A @ y[t - 1] + rng.standard_normal(2)
    return y.astype(np.float32)


def _ref(y, p, trend):
    y = y.astype(np.float64)
    rows = [np.concatenate(([np.ones(1)] if trend == "c" else []) + [y[t - l] for l in range(1, p + 1)])
            for t in range(p, len(y))]
    Z = np.array(rows)
    B = np.linalg.lstsq(Z, y[p:], rcond=None)[0]
    R = y[p:] - Z @ B
    return B, R.T @ R / (len(Z) - Z.shape[1])


def test_var_matches_least_squares():
    y = _y()
    for p, trend in ((1, "c"), (2, "c"), (2, "n")):
        r = ml.VAR(y).fit(maxlags=p, trend=trend)
        B, S = _ref(y, p, trend)
        np.testing.assert_allclose(r.params, B, atol=2e-4)
        np.testing.assert_allclose(r.sigma_u, S, rtol=1e-3)
        assert r.forecast(y, 4).shape == (4, 2)


def test_rank_deficient_design_is_refused():
    y = _y()
    y[:, 1] = y[:, 0]
    try:
        ml.VAR(y).fit(maxlags=1)
    except np.linalg.LinAlgError:
        return
    raise AssertionError("a collinear design was not refused")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
