# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Theta family: agreement with statsforecast when it is installed
(forecasts within 2e-3 relative on the pod), else the model's own identities:
a batch fit equals the single-series fits, and a seasonal series is
decomposed."""
import numpy as np

import mojolearn as ml


def _Y():
    rng = np.random.default_rng(0)
    t = np.arange(96)
    return np.stack([30 + 5 * np.sin(2 * np.pi * t / 12) + 0.1 * t + rng.standard_normal(96),
                     np.cumsum(rng.standard_normal(96)) + 50]).astype(np.float32)


def test_batch_equals_single_and_seasonality():
    Y = _Y()
    m = ml.AutoTheta(season_length=12).fit(Y)
    fb = m.predict(6)["mean"]
    assert m.info_[0, 5] == 1.0          # the seasonal series was decomposed
    for i in range(2):
        np.testing.assert_array_equal(fb[i], ml.AutoTheta(season_length=12).fit(Y[i]).predict(6)["mean"])


def test_against_statsforecast():
    try:
        from statsforecast import models as M
    except ImportError:
        return
    Y = _Y()
    for name in ("Theta", "DynamicTheta"):
        ours = getattr(ml, name)(season_length=12).fit(Y).predict(6)["mean"]
        for i in range(2):
            ref = getattr(M, name)(season_length=12).forecast(Y[i].astype(np.float64), 6)["mean"]
            np.testing.assert_allclose(ours[i], ref, rtol=2e-3)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
