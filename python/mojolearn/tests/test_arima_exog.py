# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Exogenous regressors on `ARIMA` (lane/arima-exog, 2026-09-15).

Source checks, which run anywhere: the order validator no longer refuses a
regressor count and refuses a negative one and one above `EXOG_MAX`; the three
doors that take an exogenous address permute it through ONE file
(`bindings/arima_exog_layout.mojo`), which is where a non-finite regressor is
refused by name; the host oracle restates the constructs a bit claim rests on
(the observation intercept's fold, the start-value regression, the future
differencing, `beta` through the packing and the Jones copy); the manifest
declares the two new lanes on both families; and `arima/NOT_IMPLEMENTED.tsv`
no longer carries the exog row.

Runtime checks, skipped (and said to be) when nothing in this process fits
ARIMA: a fit with regressors answers `beta_`, `forecast(h, exog)` equals
`predict(n_obs, n_obs + h, exog)` byte for byte, the four shape and
presence refusals fire by name, a model WITH regressors saves as
`mojolearn-arima-2` and reloads to the same bytes, and a model WITHOUT them
still saves as `mojolearn-arima-1` with no exog member, which is what keeps
every ARIMA saved-model hash recorded before this lane valid. The bit claim
against the GPU columns is the gates', not this file's.

    cd python && python3 -m pytest mojolearn/tests/test_arima_exog.py
"""
import re
from pathlib import Path

import mojolearn
from mojolearn import _backend, host_surface
from mojolearn._cpu_reference import reference_training

ROOT = Path(__file__).resolve().parents[3]

ORACLE = "arima/host/arima_oracle.mojo"
LAYOUT = "bindings/arima_exog_layout.mojo"
LANES = ("arima-exog", "arima-exog-seasonal")


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _code(text):
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    return re.sub(r"#[^\n]*", "", text)


# ---------------------------------------------------------------- source


def test_validate_order_takes_a_regressor_count():
    code = _code(_read("arima/impl/tsa/arima_common.mojo"))
    assert "n_exog != 0" not in code, "the order validator still refuses every regressor count"
    assert "order.n_exog < 0" in code, "a negative n_exog is not refused"
    assert "order.n_exog > EXOG_MAX" in code, "n_exog is not bounded by EXOG_MAX"
    assert "comptime EXOG_MAX = 17" in code, "EXOG_MAX is not the QR solver's column bound"
    # beta is packed after mu, as arima_common.cu:24-36 packs it.
    assert "beta.unsafe_load(n_exog * bid + i)" in code, "beta is not packed"
    assert "beta.unsafe_store(n_exog * bid + i, param_vec.unsafe_load(o + i))" in code


def test_one_file_permutes_the_exog_layout_and_refuses_a_non_finite_one():
    layout = _read(LAYOUT)
    assert "def exog_filter_layout(" in layout
    assert "non-finite value at series " in layout, "the refusal does not name the cell"
    for door in ("arima/estimator.mojo", "bindings/arima_host_predict.mojo",
                 "bindings/_mojolearn_arima_host.mojo"):
        code = _code(_read(door))
        assert "from bindings.arima_exog_layout import exog_filter_layout" in code, door
        assert "def exog_filter_layout" not in code, f"{door} spells the permutation itself"


def test_the_device_reaches_the_exog_arms():
    kal = _code(_read("arima/impl/batched_kalman.mojo"))
    assert "def obs_intercept_kernel(" in kal, "the observation intercept has no kernel"
    assert "identical_mul_add(xv, bv, acc)" in kal, "DEVIATION 995's fold"
    assert "if has_exog_in != 0:" in kal, "the loop kernel does not add the intercept"
    x0 = _code(_read("arima/impl/estimate_x0.mojo"))
    assert "def exog_regression_kernel(" in x0, "the start-value regression is missing"
    assert "householder_qr_solve(scratch, sb, m, n, scratch, bb)" in x0
    helpers = _code(_read("arima/impl/timeSeries/arima_helpers.mojo"))
    assert "def prepare_future_data(" in helpers, "the future regressors are not differenced"
    assert "t_params.beta" in helpers, "beta is not copied through the Jones transform"


def test_the_host_oracle_restates_them():
    text = _read(ORACLE)
    assert "def _obs_intercept(" in text, "the oracle has no observation intercept"
    assert "identical_mul_add(xv, bv, acc)" in text, "the oracle's fold is not DEVIATION 995's"
    assert "def _exog_regression(" in text, "the oracle has no start-value regression"
    assert "def _prepare_future(" in text, "the oracle does not difference the future regressors"
    assert "ARIMA_ORACLE_EXOG_SABOTAGE" in text, "the exog arithmetic has no negative control"
    # The oracle stays host only: the same import set the CPU training lane pins.
    imports = re.findall(r"^from\s+([\w.]+)\s+import", text, re.M)
    assert sorted(set(imports)) == ["checks.numerics", "std.math", "std.memory", "std.sys.compile"], imports


def test_the_manifest_declares_the_lanes():
    fam = host_surface.family("arima")
    assert set(LANES) <= set(fam["training_lanes"]), fam["training_lanes"]
    forecast = host_surface.family("forecast")
    assert set(LANES) <= set(forecast["inference_lanes"]), forecast["inference_lanes"]
    for lane in LANES:
        assert lane in host_surface.covered_lanes(), f"{lane} is not covered"


def test_the_unimplemented_row_is_gone():
    rows = _read("arima/NOT_IMPLEMENTED.tsv")
    assert "exogenous regressors everywhere" not in rows, "the exog row is still listed as unimplemented"


def test_the_saved_format_is_versioned():
    code = _code(_read("python/mojolearn/_arima_impl.py"))
    assert '_ARIMA_FORMAT = "mojolearn-arima-1"' in code
    assert '_ARIMA_FORMAT_EXOG = "mojolearn-arima-2"' in code
    # A model with no regressors keeps the old tag and the old members.
    assert "_ARIMA_FORMAT_EXOG if n_exog else _ARIMA_FORMAT" in code
    assert 'if n_exog:\n            arrays["exog"]' in code


# ---------------------------------------------------------------- runtime


def _can_fit():
    """Whether something in this process fits ARIMA: a GPU set, or the
    reference arima host binding on a CPU-only install."""
    if _backend._CPU_ONLY is None:
        return True
    return "_mojolearn_arima_host" in _backend.host_families_built()


@reference_training()
def test_exog_fit_forecast_and_save_when_a_binding_can_fit():
    if not _can_fit():
        print("SKIP: nothing in this process fits ARIMA (no GPU set, no arima host binding)")
        return
    import tempfile

    import numpy as np

    rng = np.random.default_rng(0)
    b, n, h = 3, 96, 8
    exog = rng.standard_normal((b, n, 2)).astype(np.float32)
    fut = rng.standard_normal((b, h, 2)).astype(np.float32)
    series = rng.standard_normal((b, n)).astype(np.float32)
    for i in range(b):
        # a real regression component, so beta is not a fit of noise on noise
        series[i] += (2.0 * exog[i, :, 0] - 0.5 * exog[i, :, 1]).astype(np.float32)

    m = mojolearn.ARIMA(order=(1, 0, 0), trend="c").fit(series, exog)
    assert m.n_exog_ == 2 and np.asarray(m.beta_).shape == (b, 2), m.n_exog_
    assert np.asarray(m.params_).shape == (b, 1 + 2 + 1 + 1), np.asarray(m.params_).shape
    fc = np.asarray(m.forecast(h, exog=fut))
    pr = np.asarray(m.predict(m.n_obs_, m.n_obs_ + h, exog=fut))
    assert fc.tobytes() == pr.tobytes(), "forecast(h, exog) and predict(n_obs, n_obs + h, exog) differ"
    assert np.isfinite(fc).all(), fc

    # the four refusals, each by name
    for call, needle in (
        (lambda: m.forecast(h), "future values must be provided"),
        (lambda: m.predict(0, m.n_obs_, exog=fut), "only in-sample"),
        (lambda: m.forecast(h, exog=fut[:, :, :1]), "regressor column"),
        (lambda: m.forecast(h, exog=rng.standard_normal((b, h + 1, 2)).astype(np.float32)),
         "dimensions mismatch"),
    ):
        try:
            call()
        except ValueError as exc:
            assert needle in str(exc), f"{needle!r} not in {exc}"
        else:
            raise AssertionError(f"a call that should have been refused ({needle}) was accepted")

    bad = fut.copy()
    bad[1, 2, 1] = np.nan
    try:
        m.forecast(h, exog=bad)
    except Exception as exc:
        assert "non-finite value at series 1, row 2, regressor 1" in str(exc), exc
    else:
        raise AssertionError("a non-finite regressor was accepted")

    plain = mojolearn.ARIMA(order=(1, 0, 0), trend="c").fit(series)
    try:
        plain.forecast(h, exog=fut)
    except ValueError as exc:
        assert "without any regression component" in str(exc), exc
    else:
        raise AssertionError("exog on a model fit without one was accepted")
    try:
        plain.beta_
    except AttributeError as exc:
        assert "without exogenous regressors" in str(exc), exc
    else:
        raise AssertionError("beta_ answered on a model fit without regressors")

    with tempfile.TemporaryDirectory(prefix="arima_exog_") as tmp:
        p_exog = str(Path(tmp) / "exog.npz")
        p_plain = str(Path(tmp) / "plain.npz")
        m.save(p_exog)
        plain.save(p_plain)
        import zipfile

        from mojolearn import _serialize
        with zipfile.ZipFile(p_exog) as zf:
            assert "exog.npy" in zf.namelist(), zf.namelist()
        with zipfile.ZipFile(p_plain) as zf:
            assert "exog.npy" not in zf.namelist(), "a model with no regressors saved an exog member"
        assert _serialize.scalar_str(_serialize.read_npz(p_exog, "mojolearn-arima-2"), "format") == "mojolearn-arima-2"
        assert _serialize.scalar_str(_serialize.read_npz(p_plain, "mojolearn-arima-1"), "format") == "mojolearn-arima-1"
        back = mojolearn.ARIMA.load(p_exog)
        assert back.n_exog_ == 2
        assert np.asarray(back.beta_).tobytes() == np.asarray(m.beta_).tobytes()
        assert np.asarray(back.forecast(h, exog=fut)).tobytes() == fc.tobytes(), "a loaded model forecast differently"
        if "_mojolearn_forecast_host" in _backend.host_families_built():
            host = mojolearn.host_model(p_exog)
            assert np.asarray(host.forecast(h, exog=fut)).tobytes() == fc.tobytes(), (
                "the forecast host binding answered different bytes for a saved exog model"
            )


if __name__ == "__main__":
    for name in [n for n in sorted(globals()) if n.startswith("test_")]:
        globals()[name]()
        print("ok", name)
