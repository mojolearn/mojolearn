# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GARCH(1,1) agrees with the arch package's fit when it is installed
(parameters within 5e-3, log-likelihood within 0.1). The recovery of the
simulated process is checked natively in
checks/native_tests/sequence_garch_{cpu,gpu}.mojo on the same stored series
(tools/sequence_check_fixtures.py)."""
import numpy as np
import pytest

import mojolearn as ml


def _sim(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    s2 = 1.0
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = 0.05 + 0.1 * r[t] ** 2 + 0.85 * s2
    return r.astype(np.float32)


def test_matches_arch():
    arch = pytest.importorskip("arch")
    y = _sim()
    g = ml.GARCH(mean="Zero").fit(y, horizon=3)
    a = arch.arch_model(y.astype(np.float64), mean="Zero", p=1, q=1, rescale=False).fit(disp="off")
    np.testing.assert_allclose(g.params_[1:], a.params.values, atol=5e-3)
    assert abs(g.loglikelihood_ - a.loglikelihood) < 0.1
