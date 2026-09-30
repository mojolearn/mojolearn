# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""AutoARIMA's search on series whose order is known. The chosen orders and
AICs were held against statsmodels' ARIMA on the lane's pod."""
import numpy as np

import mojolearn as ml


def _series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((3, n)).astype(np.float32)
    y = np.zeros_like(e)
    for t in range(1, n):
        y[0, t] = 0.7 * y[0, t - 1] + e[0, t]
        y[1, t] = e[1, t] + 0.6 * e[1, t - 1]
        y[2, t] = y[2, t - 1] + e[2, t]
    return (y + np.float32(3.0)).astype(np.float32)


def test_search_finds_the_simulated_orders():
    m = ml.AutoARIMA(_series()).search(d=range(2), p=range(2), q=range(2), ic="aic")
    assert m.d_.tolist() == [0, 0, 1]
    assert [tuple(o[:3]) for o in m.order_.tolist()] == [(1, 0, 0), (0, 0, 1), (0, 1, 0)]
    fc = m.fit().forecast(4)
    assert fc.shape == (3, 4) and np.all(np.isfinite(fc))


if __name__ == "__main__":
    test_search_finds_the_simulated_orders()
    print("PASS test_search_finds_the_simulated_orders")
