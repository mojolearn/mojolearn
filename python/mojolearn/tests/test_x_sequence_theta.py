# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Theta family agrees with statsforecast when it is installed
(forecasts within 2e-3 relative on the pod). The model's own identities (a
batch fit equals the single-series fits, a seasonal series is decomposed)
are checked natively in checks/native_tests/sequence_theta_{cpu,gpu}.mojo."""
import numpy as np
import pytest

import mojolearn as ml


def _Y():
    rng = np.random.default_rng(0)
    t = np.arange(96)
    return np.stack([30 + 5 * np.sin(2 * np.pi * t / 12) + 0.1 * t + rng.standard_normal(96),
                     np.cumsum(rng.standard_normal(96)) + 50]).astype(np.float32)


def test_against_statsforecast():
    M = pytest.importorskip("statsforecast.models")
    Y = _Y()
    for name in ("Theta", "DynamicTheta"):
        ours = getattr(ml, name)(season_length=12).fit(Y).predict(6)["mean"]
        for i in range(2):
            ref = getattr(M, name)(season_length=12).forecast(Y[i].astype(np.float64), 6)["mean"]
            np.testing.assert_allclose(ours[i], ref, rtol=2e-3)
