# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Saved Holt-Winters models, and on a CPU with no GPU the forecast host binding.

Runtime checks, skipped (and said to be) when nothing in this process fits
Holt-Winters: a fit through whatever binding this install has (a GPU set, or
the reference tsa host binding inside the internal reference context) saves
and loads to the same bytes and the same answers; a public prediction
straddling `n` is the in-sample tail then the forecast head; and, when the
forecast host binding is built, `mojolearn.host_model` answers the same bytes
through it as a `HostExponentialSmoothing`. The bit claim against the GPU
columns is the verification's, not this file's. The warm-up NaNs, the stored
level against the smoothing update and the in-sample prediction against its
inputs are checked natively in checks/native_tests/decomp_holtwinters_inference.mojo.

    cd python && python3 -m pytest mojolearn/tests/test_holtwinters_inference.py
"""
import os
from pathlib import Path

import pytest

from mojolearn import _backend


def _series(np, n, b, f):
    t = np.arange(n, dtype=np.float64)
    rows = [20.0 + (s + 1) * 3.0 * np.sin(2 * np.pi * t / f + s) + 0.05 * t + 0.5 * np.cos(0.37 * t * (s + 1))
            for s in range(b)]
    return np.ascontiguousarray(np.stack(rows).astype(np.float32))


def _fit_or_skip(np, seasonal, n=96, b=2, f=12):
    import mojolearn
    y = _series(np, n, b, f)
    try:
        return y, mojolearn.ExponentialSmoothing(y, seasonal=seasonal, seasonal_periods=f, ts_num=b).fit()
    except (ImportError, NotImplementedError, RuntimeError) as exc:
        pytest.skip(f"no binding fits Holt-Winters in this process: {type(exc).__name__}: {exc}")


@pytest.mark.parametrize("seasonal", ["additive", "multiplicative"])
def test_saved_model_forecasts_and_predicts(tmp_path, seasonal):
    np = pytest.importorskip("numpy")
    from mojolearn import ExponentialSmoothing
    f = 12
    y, m = _fit_or_skip(np, seasonal, f=f)
    n, b = m.n, m.ts_num

    path = str(tmp_path / "hw.npz")
    m.save(path)
    back = ExponentialSmoothing.load(path)
    again = str(tmp_path / "again.npz")
    back.save(again)
    assert Path(path).read_bytes() == Path(again).read_bytes(), "save(load(file)) wrote different bytes"

    def answers(e):
        return [np.asarray(v).tobytes() for v in (
            e.forecast(24), e.forecast(24, index=1), e.predict(0, n), e.predict(n - 8, n + 8),
            e.predict(n + 3, n + 9), e.level_, e.trend_, e.season_, e.sse_, e.alpha_, e.beta_,
            e.gamma_, e.n_iter_, e.criterion_, e.get_level(), e.score())]

    assert answers(m) == answers(back), "a loaded model answers differently from the fitted one"

    ins = np.asarray(m.predict(0, n))
    assert ins.shape == (n, b)
    straddle = np.asarray(m.predict(n - 8, n + 8))
    assert straddle[:8].tobytes() == ins[-8:].tobytes()
    assert straddle[8:].tobytes() == np.asarray(m.forecast(8)).tobytes()
    assert np.asarray(m.predict(n, n + 8)).tobytes() == np.asarray(m.forecast(8)).tobytes()
    assert np.asarray(m.predict(0, n, index=1)).tobytes() == np.ascontiguousarray(ins[:, 1]).tobytes()

    with pytest.raises(ValueError, match="start < end"):
        m.predict(5, 5)

    host_path = _backend.host_module_path("_mojolearn_forecast_host")
    if not os.path.exists(host_path):
        return
    from mojolearn import host_model
    host = host_model(path)
    assert type(host).__name__ == "HostExponentialSmoothing"
    assert os.path.realpath(host._bind().__file__) == os.path.realpath(host_path)
    assert answers(host) == answers(m), "the forecast host binding answers differently"
