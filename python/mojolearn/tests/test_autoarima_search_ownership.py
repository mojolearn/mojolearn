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


def test_autoarima_single_d_option_keeps_gather_indices_alive():
    script = r'''
import numpy as np
import mojolearn as ml

y = np.array([1, 1.25, 1.5, 1.125, 1.75, 1.375, 2, 1.625], np.float32)
for d in ([0], [1]):
    model = ml.AutoARIMA(y).search(d=d, p=[1], q=[0], P=[0], Q=[0], D=[0], fit_intercept=False)
    assert model.d_.tolist() == d
    model.fit(maxiter=2)
    assert np.asarray(model.predict(start=0, end=2)).shape == (1, 2)
print("autoarima-single-d-complete", flush=True)
'''
    environment = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    result = subprocess.run([sys.executable, "-c", script],
                            env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert "autoarima-single-d-complete" in result.stdout
