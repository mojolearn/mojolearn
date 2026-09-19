# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream E, CPU training for the arima, arima-011 and arima-seasonal-c
lanes (2026-09-14), checked from SOURCE so it runs on a box with nothing
built, plus a runtime check that runs only where the arima host binding is
built and the package took the CPU-only path.

What the source checks hold: the manifest declares the arima family, routes
`_mojolearn_arima`, covers the three ARIMA lanes (and par-arima since
lane/cpu-training-par-classical, 2026-09-15) and no
longer names ARIMA as having no CPU path; the binding registers the GPU
binding's fit, predict and forecast names; the oracle imports no GPU module
and nothing from `arima/` or any other package beyond the
`checks/numerics.mojo` seams; the oracle spells the device constructs a bit
claim rests on (DEVIATION 687's step, the two-rounding Jones association,
test_invparams' one-rounding association, the perturbation and the reset by
copy, the refusals by name); the sabotage define reaches the step and the
binding reads it back.

The runtime check (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded): the three lane orders fit twice on a small draw
through the host binding and return the same bytes, `forecast(h)` equals
`predict(n_obs, n_obs + h)` byte for byte, an in-sample prediction agrees
with a prediction straddling the end of the series (since 2026-09-15), and a
second AR coefficient refuses by name. It is a plumbing check. The bit claim
against the GPU columns is the CPU identity gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_arima
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import re
import sys
from pathlib import Path

import mojolearn
from mojolearn._cpu_reference import reference_training
from mojolearn import _backend, host_surface

ROOT = Path(__file__).resolve().parents[3]

ORACLE = "arima/host/arima_oracle.mojo"
LANES = ("arima", "arima-011", "arima-seasonal-c")
#: lane/arima-exog (2026-09-15), declared after par-arima.
EXOG_LANES = ("arima-exog", "arima-exog-seasonal")
#: lane/forecast-predict-cpu-route (2026-09-19), declared last: the driver
#: behind `parallel_forecasting.forecast_arima` / `.predict_arima`, which had
#: no lane at all until `forecast_predict` joined `_parallel_pool
#: .CPU_OPERATIONS`. Pinned here because this tuple is the arima family's
#: CONTRACT, and a lane appearing in it without a test moving is how a family
#: grows a member nobody checked.
FORECAST_LANES = ("par-forecast-arima",)
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_manifest_declares_the_arima_family():
    fam = host_surface.family("arima")
    assert fam["routes"] == "_mojolearn_arima"
    assert fam["training_lanes"] == LANES + ("par-arima",) + EXOG_LANES + FORECAST_LANES
    assert ORACLE in fam["host_modules"] and (ROOT / ORACLE).is_file()
    assert (ROOT / host_surface.build_shim("arima")).is_file()
    assert (ROOT / host_surface.binding_source("arima")).is_file()
    assert host_surface.routed_modules()["_mojolearn_arima"] == "_mojolearn_arima_host"
    for lane in LANES:
        assert lane in host_surface.covered_lanes(), f"{lane} is not a covered training lane"
    # par-arima is declared since lane/cpu-training-par-classical (2026-09-15):
    # fit_arima's series shards run as host fits through the pool's CPU route.
    assert "par-arima" in host_surface.covered_lanes(), "par-arima is not a covered training lane"
    assert "ARIMA" not in host_surface.no_cpu_path_sentence(), host_surface.no_cpu_path_sentence()
    assert "seasonal ARIMA" in host_surface.training_sentence()


def test_binding_registers_the_gpu_names():
    # arima_predict and arima_forecast live in bindings/arima_host_predict.mojo
    # since lane/inference-forecast-umap-pca (2026-09-15), shared with the
    # forecast inference binding; the reference binding imports them.
    src = _read(host_surface.binding_source("arima")) + _read("bindings/arima_host_predict.mojo")
    exports = host_surface.family("arima")["exports"]
    gpu = _read("bindings/_mojolearn_arima.mojo")
    for name in ("arima_fit", "arima_predict", "arima_forecast", "arima_vendor", "arima_numeric_mode"):
        assert f'("{name}")' in src, f"the arima host binding does not register {name}"
        assert f'("{name}")' in gpu, f"{name} is not a GPU binding name"
        assert name in exports, f"the manifest does not list {name} for arima"
    for slots in ("arima_fit: params must contain 13 values (batch_size, n_obs, p,",
                  " n_obs, start, end, p, d, q, P, D, Q, s, k, n_exog), got ",
                  "params[12] is reserved and must be 0"):
        assert slots in src, f"the host binding does not carry the GPU slot check {slots!r}"
        assert slots in gpu, f"the GPU binding no longer carries {slots!r}"
    assert "validate_order(order)" in src, "the order is not validated by the device's own function"


def test_oracle_imports_no_gpu_and_no_device_module():
    text = _read(ORACLE)
    assert not GPU_IMPORTS.search(text), f"{ORACLE} imports a GPU module"
    assert not re.search(r"^\s*from .*import.*DeviceContext", text, re.M), f"{ORACLE} imports DeviceContext"
    imports = re.findall(r"^from\s+([\w.]+)\s+import", text, re.M)
    assert sorted(set(imports)) == ["checks.numerics", "std.math", "std.memory", "std.sys.compile"], imports


def test_oracle_spells_the_bit_carrying_constructs():
    text = _read(ORACLE)
    assert "comptime AH_FIT_H = Float32(0.0009765625)" in text, "DEVIATION 687's step"
    assert "var prod = ftz(a * mine[j - k - 1])" in text, "the Jones recursion rounds a * x on its own"
    assert "identical_mul_add(coef_a, new_params[j - k - 1], new_params[k])" in text, (
        "test_invparams fuses (coef * a) * x, one rounding"
    )
    assert "x_pert[idx] = ftz(ftz(xin[idx]) + h)" in text, "the perturbation"
    assert "x_pert[idx] = xin[idx]" in text, "the reset is a copy, not x + 0.0"
    assert "fout[b] = ftz(ftz(-base[b]) / scale)" in text
    assert "for i in range(q, m1):" in text, "the sigma2 fold starts at q"
    assert "Float32(n_obs - 1)" in text, "the objective scale uses the undifferenced length"
    assert "var neg = (bitcast[DType.uint32](v) >> 31) != 0" in text, "the intercept nudge reads the sign bit"
    assert "identical_mul_add(n_obs_ll_f, inner, b_sum_logFs)" in text
    assert "no CPU implementation of ARIMA with" in text


def test_sabotage_define_moves_the_step():
    text = _read(ORACLE)
    define = host_surface.sabotage_define("arima")
    assert define == "MOJOLEARN_HOST_SABOTAGE"
    assert f'is_defined["{define}"]()' in text
    assert "comptime if ARIMA_ORACLE_HOST_SABOTAGE:" in text
    assert "comptime AH_FIT_H_SABOTAGE = Float32(0.001953125)" in text
    assert "ARIMA_ORACLE_HOST_SABOTAGE" in _read(host_surface.binding_source("arima"))



@reference_training()
def test_arima_fits_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_arima_host" not in _backend.host_families_built():
        print("SKIP: the arima host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_arima_host")
    assert not bool(module.arima_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.standard_normal((3, 160)), axis=1).astype(np.float32) * 0.1
    series += rng.standard_normal((3, 160)).astype(np.float32)
    kinds = (
        dict(order=(1, 0, 0)),
        dict(order=(0, 1, 1)),
        dict(order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), trend="c"),
    )
    for kw in kinds:
        outs = []
        for _ in range(2):
            m = mojolearn.ARIMA(**kw).fit(series)
            fc = np.asarray(m.forecast(12))
            pr = np.asarray(m.predict(m.n_obs_, m.n_obs_ + 12))
            assert fc.tobytes() == pr.tobytes(), f"{kw}: forecast(h) and predict(n_obs, n_obs + h) differ"
            outs.append((np.asarray(m.params_).tobytes(), fc.tobytes()))
        assert outs[0] == outs[1], f"{kw}: two host fits returned different bytes"
        # In-sample prediction runs on the host since 2026-09-15: the tail of
        # predict(0, n_obs) is the head of a prediction straddling n_obs,
        # whose tail is the forecast.
        ins = np.asarray(m.predict(0, m.n_obs_))
        st = np.asarray(m.predict(m.n_obs_ - 4, m.n_obs_ + 4))
        assert st[:, :4].tobytes() == ins[:, -4:].tobytes(), f"{kw}: the straddle's in-sample half moved"
        assert st[:, 4:].tobytes() == np.asarray(m.forecast(4)).tobytes(), f"{kw}: the straddle's forecast half moved"
        nan_prefix = kw["order"][1] + kw.get("seasonal_order", (0, 0, 0, 0))[1] * kw.get("seasonal_order", (0, 0, 0, 0))[3]
        assert np.isnan(ins[:, :nan_prefix]).all() and not np.isnan(ins[:, nan_prefix:]).any(), kw
    try:
        mojolearn.ARIMA(order=(2, 0, 0)).fit(series)
    except Exception as exc:
        assert "no CPU implementation of ARIMA with p=2" in str(exc), exc
    else:
        raise AssertionError("p=2 did not refuse by name")


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
