# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Croston against a float64 restatement of statsforecast's
_croston_classic / _croston_sba (the optimized variant agreed with
statsforecast within 2e-4 relative on the pod)."""
import numpy as np

import mojolearn as ml


def _ses(x, a):
    f = x[0]
    for v in x[:-1]:
        f = a * v + (1 - a) * f
    return a * x[-1] + (1 - a) * f if len(x) > 1 else x[0]


def test_classic_and_sba():
    rng = np.random.default_rng(1)
    Y = ((rng.random((4, 60)) < 0.3) * rng.integers(1, 9, (4, 60))).astype(np.float32)
    Y[3] = 0
    c = ml.CrostonClassic().fit(Y).predict(2)["mean"]
    s = ml.CrostonSBA().fit(Y).predict(2)["mean"]
    for i, y in enumerate(Y.astype(np.float64)):
        nz = np.flatnonzero(y != 0)
        if not len(nz):
            assert c[i, 0] == y[-1]
            continue
        d = _ses(y[y > 0], 0.1)
        itv = _ses(np.diff(nz + 1, prepend=0).astype(float), 0.1)
        np.testing.assert_allclose(c[i], d / itv, rtol=1e-5)
        np.testing.assert_allclose(s[i], 0.95 * d / itv, rtol=1e-5)


if __name__ == "__main__":
    test_classic_and_sba()
    print("PASS test_classic_and_sba")
