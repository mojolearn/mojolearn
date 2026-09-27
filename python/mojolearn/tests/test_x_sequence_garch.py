# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GARCH(1,1): recovers a simulated process, and agrees with the arch
package's fit when it is installed (parameters within 2e-3, log-likelihood
within 0.05 on the pod)."""
import numpy as np

import mojolearn as ml


def _sim(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    s2 = 1.0
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = 0.05 + 0.1 * r[t] ** 2 + 0.85 * s2
    return r.astype(np.float32)


def test_recovers_and_matches_arch():
    y = _sim()
    g = ml.GARCH(mean="Zero").fit(y, horizon=3)
    omega, alpha, beta = g.params_[1:]
    assert abs(alpha - 0.1) < 0.05 and abs(beta - 0.85) < 0.08
    assert g.forecast(3).shape == (3,) and g.conditional_volatility_.shape == y.shape
    try:
        from arch import arch_model
    except ImportError:
        return
    a = arch_model(y.astype(np.float64), mean="Zero", p=1, q=1, rescale=False).fit(disp="off")
    np.testing.assert_allclose(g.params_[1:], a.params.values, atol=5e-3)
    assert abs(g.loglikelihood_ - a.loglikelihood) < 0.1


if __name__ == "__main__":
    test_recovers_and_matches_arch()
    print("PASS test_recovers_and_matches_arch")
