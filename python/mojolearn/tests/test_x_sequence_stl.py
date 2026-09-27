# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""STL's decomposition identities and batch behaviour. Agreement with
statsmodels' STL (within 4e-5 on robust and jumped configurations) was checked
on the lane's pod (docs/lanes/progress/sequence.md)."""
import numpy as np

import mojolearn as ml


def _y(n=96, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (np.sin(2 * np.pi * t / 12) * 3 + 0.05 * t + rng.standard_normal(n) * 0.3).astype(np.float32)


def test_components_add_up_and_weights():
    y = _y()
    r = ml.STL(y, period=12).fit()
    np.testing.assert_allclose(r.seasonal + r.trend + r.resid, y, atol=1e-5)
    assert np.all(r.weights == 1.0)
    rr = ml.STL(y, period=12, robust=True).fit()
    assert np.all((rr.weights >= 0) & (rr.weights <= 1)) and np.any(rr.weights < 1)


def test_batch_rows_equal_single_fits():
    Y = np.stack([_y(seed=s) for s in range(3)])
    rb = ml.STL(Y, period=12).fit()
    for i in range(3):
        np.testing.assert_array_equal(rb.trend[i], ml.STL(Y[i], period=12).fit().trend)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
