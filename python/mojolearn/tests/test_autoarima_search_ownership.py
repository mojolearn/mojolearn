"""AutoARIMA group selection retains its native scratch until writes finish."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("observations", [96, 200])
def test_autoarima_search_fit_forecast_survive_group_scratch_reuse(observations):
    script = r'''
import sys
import numpy as np
import mojolearn as ml

n = int(sys.argv[1])
raw = np.random.default_rng(518).standard_normal((2, n)).astype(np.float32)
walk = np.cumsum(raw, axis=1, dtype=np.float32)
y = np.ascontiguousarray(np.concatenate([raw, walk]), dtype=np.float32)
model = ml.AutoARIMA(y).search(d=range(2), p=range(2), q=range(2))
assert model.d_.shape == (4,)
assert model.order_.shape == (4, 8)
assert np.isfinite(model.ic_).all()
model.fit()
forecast = model.forecast(12).copy()
assert forecast.shape == (4, 12)
assert np.isfinite(forecast).all()
scratch = [np.full(4, i, dtype=np.int64) for i in range(8)]
assert sum(int(value[0]) for value in scratch) == 28
assert forecast.tobytes() == model.forecast(12).tobytes()
print("autoarima-search-output-owner-complete", flush=True)
'''
    environment = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    result = subprocess.run([sys.executable, "-c", script, str(observations)],
                            env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert "autoarima-search-output-owner-complete" in result.stdout
