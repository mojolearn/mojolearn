# SPDX-License-Identifier: Apache-2.0
"""fit_arima's native partition and merge reproduce the plain fit's saved model.

fit_arima is one native call (bindings/parallel_forecast_native.mojo
`fit_arima`): whole series in `series_per_shard` ranges, each fitted by the
plain native fit in a worker process, every fitted array placed at its
series offset. Every series is fitted alone, so the merged model must save
to the same bytes as one plain fit of the whole batch.
"""
import numpy as np
import pytest

from mojolearn._arima_impl import ARIMA
from mojolearn._buffer import Array
from mojolearn import parallel_classical


@pytest.mark.parametrize('order,trend', [((1, 0, 0), None), ((0, 1, 1), None),
                                        ((1, 0, 0), 'n'), ((1, 0, 0), 'c')])
def test_parallel_arima_preserves_configuration_and_model_bytes(tmp_path, order, trend):
    params = dict(order=order, trend=trend, numeric_mode='identical')
    rng = np.random.default_rng(7)
    data = Array.from_list(np.cumsum(rng.standard_normal((5, 32)), axis=1).astype(np.float32).tolist(), '<f4')
    plain = ARIMA(**params).fit(data)
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
