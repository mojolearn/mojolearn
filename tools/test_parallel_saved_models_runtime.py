"""Small native CPU roundtrips; no device or historical-column qualification."""
import re
from pathlib import Path

import numpy as np
import pytest

import identity_break as identity
import mojolearn as ml
from mojolearn import _backend
from mojolearn.parallel_classical import fit_arima, fit_exponential_smoothing
from mojolearn.parallel_preprocessing import fit_scaler


@pytest.fixture(autouse=True)
def cpu_bindings(monkeypatch):
    host = Path(ml.__file__).parent / 'host'
    needed = ('arima', 'tsa', 'preprocessing', 'core', 'forecast')
    if _backend._CPU_ONLY is None or any(not (host / f'_mojolearn_{name}_host.so').exists() for name in needed):
        pytest.skip('requires installed native CPU bindings for these four model families')
    monkeypatch.delenv(identity.HOST_INFER_ENV, raising=False)
    monkeypatch.setattr(identity, '_par_devices', lambda: (0,))


@pytest.mark.parametrize('lane', ['par-arima', 'par-holtwinters', 'par-scaler', 'par-queries-radius'])
def test_parallel_models_record_real_save_reload_and_batch(lane, tmp_path):
    rng = np.random.default_rng(71)
    x = rng.normal(size=(48, 4)).astype(np.float32)
    heldout = rng.normal(size=(12, 4)).astype(np.float32)
    if lane == 'par-arima':
        model = fit_arima(ml.ARIMA(order=(1, 0, 0), maxiter=8), x.T.copy(), series_per_shard=3)
        state = model.params_
        probe = lambda e: (e.forecast(7),)
    elif lane == 'par-holtwinters':
        series = (x.T + np.float32(8)).copy()
        model = fit_exponential_smoothing(ml.ExponentialSmoothing(series, ts_num=4, seasonal_periods=4), series_per_shard=3)
        state = model._comps
        probe = lambda e: (e.forecast(7),)
    elif lane == 'par-scaler':
        model = fit_scaler(ml.StandardScaler(), x, columns_per_shard=3)
        state = model.mean_
        probe = lambda e: (e.transform(heldout),)
    else:
        model = ml.RadiusNeighbors(radius=2.0).fit(x)
        state = model._index
        probe = lambda e: identity._ragged(identity._pq(e, heldout, 'radius_neighbors', sort_results=True))
    fit = identity._fit({'state': identity._h(state)}, model, probe)
    infer, saved, reloaded, error = identity._probe_fit(fit, lane)
    assert error is None
    assert re.fullmatch('[0-9a-f]{16}', fit['state'])
    assert re.fullmatch('[0-9a-f]{16}', saved)
    assert infer == reloaded
    path = tmp_path / 'direct.npz'
    model.save(path)
    assert saved == identity._hfile(path)
    batch, error = identity._probe_batch(fit, lane, ml, heldout, alone=2, sabotage=False)
    assert error is None
    assert re.fullmatch('[0-9a-f]{16}', batch), batch
