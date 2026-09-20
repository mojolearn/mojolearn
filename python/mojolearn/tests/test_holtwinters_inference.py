# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Saved Holt-Winters models on a CPU with no GPU (lane/inference-holtwinters,
2026-09-15).

Source checks, which run anywhere: the forecast inference family serves the
`_mojolearn_tsa` route and the two Holt-Winters lanes; the reference tsa host
binding, the forecast inference binding and the GPU binding register
`holtwinters_predict` from ONE source; that source reaches no fit; and the host
class is registered for the new format.

Runtime checks, skipped (and said to be) when nothing in this process fits
Holt-Winters: a fit through whatever binding this install has (a GPU set, or
the reference tsa host binding inside the internal reference context) saves
and loads to the same bytes and the same answers; `predict(0, n)` is NaN
before `2 * seasonal_periods` and finite after; a prediction straddling `n`
is the in-sample tail then the forecast head; the level the fit stored agrees,
within float32 rounding, with the smoothing update applied to the in-sample
prediction's inputs, which pins the time index of those inputs; and, when the
forecast host binding is built, `mojolearn.host_model` answers the same bytes
through it. The bit claim against the GPU columns is the gates', not this
file's.

    cd python && python3 -m pytest mojolearn/tests/test_holtwinters_inference.py
"""
import os
import re
from pathlib import Path

import pytest

from mojolearn import _backend, host_surface

ROOT = Path(__file__).resolve().parents[3]
SHARED = "bindings/holtwinters_host_predict.mojo"


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _code(text):
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    return re.sub(r"#[^\n]*", "", text)


def test_forecast_family_serves_holtwinters():
    f = host_surface.family("forecast")
    assert "_mojolearn_tsa" in f["serves"]
    assert {"holtwinters", "holtwinters-multiplicative"} <= set(f["inference_lanes"])
    assert "ExponentialSmoothing" in f["classes"]
    assert {"tsa_vendor", "holtwinters_forecast", "holtwinters_predict"} <= set(f["exports"])
    assert "holtwinters_fit" not in f["exports"]
    # Both ship since lane/ship-cpu-host-families (2026-09-16): this binding
    # so a saved model predicts from a binary with no fit in it, the tsa
    # binding so the holtwinters and kpss lanes can be checked on a CPU.
    assert f["ships_in_wheel"] and host_surface.family("tsa")["ships_in_wheel"]
    assert host_surface.inference_routes()["_mojolearn_tsa"] == "_mojolearn_forecast_host"


def test_three_bindings_register_predict_from_one_source():
    for src in (host_surface.binding_source("tsa"), host_surface.binding_source("forecast"),
                "bindings/_mojolearn_tsa.mojo"):
        code = _code(_read(src))
        assert "from bindings.holtwinters_host_predict import" in code, src
        assert 'def_function[holtwinters_predict_binding]("holtwinters_predict")' in code, src
        assert "def holtwinters_predict_binding" not in code, f"{src} defines its own predict"
    for src in (host_surface.binding_source("tsa"), host_surface.binding_source("forecast")):
        code = _code(_read(src))
        assert 'def_function[holtwinters_forecast_binding]("holtwinters_forecast")' in code, src
        assert "def holtwinters_forecast_binding" not in code, f"{src} defines its own forecast"
    # The oracle the device is held to forecasts through the same body.
    assert "hw_forecast_from_state[dt](" in _code(_read("holtwinters/host/hw_oracle.mojo"))


def test_prediction_sources_reach_no_fit():
    allowed = {
        "std.math", "std.memory", "std.python", "std.python._cpython", "std.sys.compile",
        "max.algorithm", "core.host_predict_threads",
        "checks.numerics", "bindings.hostptr", "holtwinters.host.hw_predict",
        "holtwinters.impl.tsa.holtwinters_params",
    }
    for src in ("holtwinters/host/hw_predict.mojo", SHARED):
        imports = set(re.findall(r"^from ([\w.]+) import", _code(_read(src)), re.M))
        assert imports <= allowed, f"{src} imports {sorted(imports - allowed)}"
    params = _code(_read("holtwinters/impl/tsa/holtwinters_params.mojo"))
    assert not re.findall(r"^from ([\w.]+) import", params, re.M), "holtwinters_params now imports"


def test_host_class_is_registered():
    from mojolearn._classical_host import _FORMATS, _HOST_BASENAMES, HostExponentialSmoothing
    from mojolearn._tsa_impl import _HW_FORMAT
    assert _FORMATS[_HW_FORMAT] == {"ExponentialSmoothing": HostExponentialSmoothing}
    assert _HOST_BASENAMES["_mojolearn_tsa"] == "_mojolearn_forecast_host"
    assert HostExponentialSmoothing._BINDING == "_mojolearn_tsa"


def _series(np, n, b, f):
    t = np.arange(n, dtype=np.float64)
    rows = [20.0 + (s + 1) * 3.0 * np.sin(2 * np.pi * t / f + s) + 0.05 * t + 0.5 * np.cos(0.37 * t * (s + 1))
            for s in range(b)]
    return np.ascontiguousarray(np.stack(rows).astype(np.float32))


def _fit_or_skip(np, seasonal, n=96, b=2, f=12):
    import mojolearn
    from mojolearn._cpu_reference import reference_training
    y = _series(np, n, b, f)
    try:
        with reference_training():
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
    assert np.isnan(ins[:2 * f]).all() and not np.isnan(ins[2 * f:]).any()
    straddle = np.asarray(m.predict(n - 8, n + 8))
    assert straddle[:8].tobytes() == ins[-8:].tobytes()
    assert straddle[8:].tobytes() == np.asarray(m.forecast(8)).tobytes()
    assert np.asarray(m.predict(n, n + 8)).tobytes() == np.asarray(m.forecast(8)).tobytes()
    assert np.asarray(m.predict(0, n, index=1)).tobytes() == np.ascontiguousarray(ins[:, 1]).tobytes()

    # The in-sample prediction's inputs at time t are level[t - f - 1],
    # trend[t - f - 1] and season[t - 2f]; the smoothing update from them and
    # y[t] must land, within rounding, on the stored level[t - f].
    level, trend, season = (np.asarray(a, dtype=np.float64) for a in (m.level_, m.trend_, m.season_))
    alpha = np.asarray(m.alpha_, dtype=np.float64)
    for t in range(2 * f, n):
        i = t - f
        lt = level[:, i - 1] + trend[:, i - 1]
        stmp = season[:, i - f]
        if seasonal == "additive":
            want = alpha * (y[:, t] - stmp) + (1 - alpha) * lt
            pred = lt + stmp
        else:
            want = alpha * (y[:, t] / stmp) + (1 - alpha) * lt
            pred = lt * stmp
        np.testing.assert_allclose(level[:, i], want, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(ins[t], pred, rtol=1e-5, atol=1e-5)

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
