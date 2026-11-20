# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Non-seasonal ETS: agreement with statsforecast AutoETS when installed
(forecasts within 1.2e-3 relative on the pod). The structural checks (the
damped forecast flattens, the batch equals single-series fits) are native:
checks/native_tests/tsa_sequence_ets_{cpu,gpu}.mojo."""
import numpy as np

import mojolearn as ml


def _Y():
    rng = np.random.default_rng(0)
    t = np.arange(80)
    return np.stack([20 + 0.5 * t + rng.standard_normal(80), 50 + rng.standard_normal(80)]).astype(np.float32)


def test_against_statsforecast():
    try:
        from statsforecast.models import AutoETS
    except ImportError:
        return
    Y = _Y()
    ours = ml.DampedETS().fit(Y).predict(8)["mean"]
    for i in range(2):
        ref = AutoETS(model="AAN", damped=True).forecast(Y[i].astype(np.float64), 8)["mean"]
        np.testing.assert_allclose(ours[i], ref, rtol=5e-3)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
