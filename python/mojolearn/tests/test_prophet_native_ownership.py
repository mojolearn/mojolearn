"""The complete public Prophet fixture retains native fit/predict inputs."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("fixture", ["base", "odd"])
def test_full_prophet_fixture_repeated_forecast_keeps_input_owners(fixture):
    script = r'''
import sys
import numpy as np
import mojolearn as ml
from mojolearn import _identity_break as h

fixture = sys.argv[1]
x, yc, yr = h.fixture(fixture)
result = h.LANES["sequence-prophet"](ml, x, yc, yr, h.heldout(fixture))
probe = result.est[0][2]
models = [cell.cell_contents for cell in probe.__closure__
          if type(cell.cell_contents) is ml.ProphetForecaster]
assert len(models) == 2
parameters = [np.asarray(model.params_).tobytes() for model in models]
expected = [result["a_fc"], result["m_fc"]]
for horizon in (21, 7, 21, 33, 21):
    # Ordinary heap reuse must not change the forecast or fitted state.
    scratch = [np.full((horizon, 4), i, dtype=np.float32) for i in range(24)]
    forecasts = probe(horizon)
    assert all(np.isfinite(np.asarray(value)).all() for value in forecasts)
    if horizon == 21:
        assert [h._h(value) for value in forecasts] == expected
    assert [np.asarray(model.params_).tobytes() for model in models] == parameters
    del scratch
'''
    result = subprocess.run(
        [sys.executable, "-c", script, fixture],
        env=dict(os.environ), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
