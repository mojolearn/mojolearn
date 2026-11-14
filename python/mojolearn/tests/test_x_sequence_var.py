# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""VAR's public refusal of a rank-deficient design. The estimates against a
float64 least-squares restatement of statsmodels' estimator are checked
natively in checks/native_tests/sequence_var_{cpu,gpu}.mojo; parity with
statsmodels itself (within 1e-5 on params, sigma_u and forecasts) was
checked on the lane's pod."""
import numpy as np
import pytest

import mojolearn as ml


def _y(n=300, seed=0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.2, 0.3]])
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t] = 0.5 + A @ y[t - 1] + rng.standard_normal(2)
    return y.astype(np.float32)


@pytest.mark.parametrize("rows,scale", [(96, 1.0), (300, 1.0), (300, 1e-4)])
def test_rank_deficient_design_is_refused(rows, scale):
    y = _y(n=rows) * np.float32(scale)
    y[:, 1] = y[:, 0]
    try:
        ml.VAR(y).fit(maxlags=1)
    except np.linalg.LinAlgError:
        return
    raise AssertionError("a collinear design was not refused")
