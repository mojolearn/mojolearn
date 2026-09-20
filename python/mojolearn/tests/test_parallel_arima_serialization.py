# SPDX-License-Identifier: Apache-2.0
"""Test partition assembly and archive identity, not native arithmetic."""
import numpy as np
import pytest

from mojolearn._arima_impl import ARIMA, _series_major
from mojolearn._buffer import Array
from mojolearn import parallel_classical


def fitted_state(params, data):
    """Series-local deterministic state so partitioning cannot change a fit."""
    model = ARIMA(**params)
    data, batch, observations, copied = _series_major(data, 'y')
    model._y = data
    model.batch_size_, model.n_obs_, model.input_copied_ = batch, observations, copied
    values = np.asarray(data)[:, 0]
    for name in ('params_', 'x_', 'x0_'):
        rows = [[float(v) + j for j in range(model.complexity_)] for v in values]
        setattr(model, name, Array.from_list(rows, '<f4'))
    for name in ('fx_', 'n_iter_', 'retcode_', 'llf_', 'aic_', 'bic_'):
        dtype = '<i4' if name in ('n_iter_', 'retcode_') else '<f4' if name == 'fx_' else '<f8'
        setattr(model, name, Array.from_list(values.astype(dtype).tolist(), dtype))
    return model


@pytest.mark.parametrize('order,trend', [((1, 0, 0), None), ((0, 1, 1), None),
                                        ((1, 0, 0), 'n'), ((1, 0, 0), 'c')])
def test_parallel_arima_preserves_configuration_and_model_bytes(monkeypatch, tmp_path, order, trend):
    class Pool:
        def __init__(self, devices):
            pass
        def map(self, requests):
            return [fitted_state(params, args[0]) for operation, params, args in requests]
        def close(self):
            pass

    monkeypatch.setattr(parallel_classical, 'DevicePool', Pool)
    params = dict(order=order, trend=trend, numeric_mode='identical')
    data = Array.from_list(np.arange(5 * 16).reshape(5, 16).tolist(), '<f4')
    plain = fitted_state(params, data)
    model = ARIMA(**params)
    returned = parallel_classical.fit_arima(model, data, series_per_shard=2)
    assert returned is model
    assert model.trend == trend
    assert model.k_ == plain.k_
    parallel_path, plain_path = tmp_path / 'parallel.npz', tmp_path / 'plain.npz'
    model.save(parallel_path)
    plain.save(plain_path)
    assert parallel_path.read_bytes() == plain_path.read_bytes()
    loaded = ARIMA.load(parallel_path)
    assert loaded.trend == trend
    assert loaded.k_ == model.k_
    for name in ('params_', 'x_', 'x0_', 'fx_', 'n_iter_', 'retcode_', 'llf_', 'aic_', 'bic_', '_y'):
        assert np.asarray(getattr(loaded, name)).tobytes() == np.asarray(getattr(model, name)).tobytes()
