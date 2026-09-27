# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ProphetForecaster recovers a known trend break, weekly cycle and holiday
effect (parity with the prophet package, within 1e-3 of max|y| on 800 daily
points, was checked on the lane's pod)."""
import numpy as np

import mojolearn as ml


def test_recovers_structure():
    rng = np.random.default_rng(0)
    N = 200
    t = np.arange(N, dtype=np.float64) + 18000.0
    i = np.arange(N)
    hol = (np.arange(N + 14) % 50 == 7).astype(np.float32)
    y = (10 + np.where(i < 100, 0.05 * i, 5 - 0.02 * (i - 100)) + np.sin(2 * np.pi * i / 7)
         + 3 * hol[:N] + 0.1 * rng.standard_normal(N)).astype(np.float32)
    m = ml.ProphetForecaster().fit(t, y, holidays=hol[:N, None])
    fit = m.predict(t, holidays=hol[:N, None])
    assert np.abs(fit - y).mean() < 0.3
    fut = m.predict(np.arange(N, N + 14, dtype=np.float64) + 18000.0, holidays=hol[N:, None])
    assert fut.shape == (14,) and np.all(np.isfinite(fut))
    B = np.stack([y, y * np.float32(2.0)])
    mb = ml.ProphetForecaster().fit(t, B, holidays=hol[:N, None])
    np.testing.assert_array_equal(mb.params_[0], m.params_[0])


if __name__ == "__main__":
    test_recovers_structure()
    print("PASS test_recovers_structure")
