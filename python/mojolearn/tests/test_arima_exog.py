# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Exogenous regressors on `ARIMA`, through the public estimator.

Runtime checks, skipped (and said to be) when nothing in this process fits
ARIMA: a fit with regressors answers `beta_`, `forecast(h, exog)` equals
`predict(n_obs, n_obs + h, exog)` byte for byte, the four shape and
presence refusals fire by name, a non-finite regressor is refused by its
cell, a regressor count above the solver's bound is refused, a model WITH
regressors saves as `mojolearn-arima-2` and reloads to the same bytes, and a
model WITHOUT them saves as `mojolearn-arima-1` with no exog member. The bit
claim against the GPU columns is the verification's, not this file's.

    cd python && python3 -m pytest mojolearn/tests/test_arima_exog.py
"""
from pathlib import Path

import mojolearn
from mojolearn import _backend


# ---------------------------------------------------------------- runtime


def _can_fit():
    """Whether something in this process fits ARIMA: a GPU set, or the
    reference arima host binding on a CPU-only install."""
    if _backend._CPU_ONLY is None:
        return True
    return "_mojolearn_arima_host" in _backend.host_families_built()


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

    # one regressor above the QR solver's column bound (EXOG_MAX = 17) is refused
    wide = rng.standard_normal((b, n, 18)).astype(np.float32)
    try:
        mojolearn.ARIMA(order=(1, 0, 0), trend="c").fit(series, wide)
    except Exception as exc:
        assert "EXOG_MAX" in str(exc), exc
    else:
        raise AssertionError("18 regressors were accepted above EXOG_MAX")

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


def test_save_load_round_trip_over_every_input_kind_the_verify_lanes_pass():
    """The verify lanes arima, arima-011, arima-seasonal-c, arima-exog and
    arima-exog-seasonal save a model fit on a C-contiguous float32 ndarray
    (the zero-copy input), which `save` refused with "Sequential requires a
    memoryview over a native buffer owner" (AMD verify of 7c6e0dade). Each
    input kind and order is fit, saved, loaded, and must forecast the same
    bytes and carry the same fitted arrays."""
    if not _can_fit():
        print("SKIP: nothing in this process fits ARIMA (no GPU set, no arima host binding)")
        return
    import tempfile

    import numpy as np

    rng = np.random.default_rng(7)
    b, n, h = 4, 128, 6
    base = np.ascontiguousarray(rng.standard_normal((n, b)).astype(np.float32).T)
    exog = np.ascontiguousarray(rng.standard_normal((b, n, 2)).astype(np.float32))
    fut = np.ascontiguousarray(rng.standard_normal((b, h, 2)).astype(np.float32))
    inputs = (
        ("float32 C ndarray", base),
        ("float64 ndarray", base.astype(np.float64)),
        ("float32 F ndarray", np.asfortranarray(base)),
        ("mojolearn Array", mojolearn.Array.from_buffer(base)),
        ("nested list", base.tolist()),
        ("one series", np.ascontiguousarray(base[0])),
    )
    orders = (
        dict(order=(1, 0, 0)),
        dict(order=(0, 1, 1)),
        dict(order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), trend="c"),
    )
    fields = ("params_", "x_", "x0_", "fx_", "n_iter_", "retcode_", "llf_", "aic_", "bic_")
    with tempfile.TemporaryDirectory(prefix="arima_round_trip_") as tmp:
        for label, y in inputs:
            for k, kw in enumerate(orders):
                for with_exog in (False, True):
                    if with_exog and label == "one series":
                        continue
                    m = mojolearn.ARIMA(**kw).fit(y, exog if with_exog else None)
                    path = str(Path(tmp) / f"m{len(label)}_{k}_{int(with_exog)}.npz")
                    m.save(path)
                    back = mojolearn.ARIMA.load(path)
                    where = f"{label}, {kw}, exog={with_exog}"
                    for name in fields:
                        assert (np.asarray(getattr(back, name)).tobytes()
                                == np.asarray(getattr(m, name)).tobytes()), f"{name} differs after load ({where})"
                    future = fut if with_exog else None
                    want = np.asarray(m.forecast(h, exog=future))
                    got = np.asarray(back.forecast(h, exog=future))
                    assert got.tobytes() == want.tobytes(), f"a loaded model forecast differently ({where})"
                    # a loaded model saves again, to the same bytes
                    again = path + ".again.npz"
                    back.save(again)
                    assert Path(again).read_bytes() == Path(path).read_bytes(), f"re-save differs ({where})"


def test_save_accepts_a_fitted_array_in_the_native_output_store():
    """`Array.astype` (and the label encoders and float64 to float32
    converters) return Arrays over a raw ctypes output store, a ctypes array
    class; the 0.8.55 ARIMA kept `llf_ = stats[:batch].astype("<f8")`, and
    `save` refused it ("Sequential requires a memoryview over a native buffer
    owner", AMD verify of 7c6e0dade). A fitted field held in that store saves
    and loads to the same bytes."""
    if not _can_fit():
        print("SKIP: nothing in this process fits ARIMA (no GPU set, no arima host binding)")
        return
    import ctypes
    import tempfile

    import numpy as np

    rng = np.random.default_rng(11)
    series = np.ascontiguousarray(rng.standard_normal((4, 96)).astype(np.float32))
    m = mojolearn.ARIMA(order=(1, 0, 0)).fit(series)
    # a fitted field held in a native output store (ctypes.Array), built
    # directly: astype only returns one above the cast helper's size cut
    from mojolearn._array import Array
    from mojolearn.tests._typed_input import output_store
    src = np.ascontiguousarray(np.asarray(m.llf_), dtype="<f8")
    store = output_store("d", src.size)
    ctypes.memmove(store, src.ctypes.data, src.nbytes)
    held = Array._owned(store, src.shape, "<f8", "C")
    assert isinstance(held._store, ctypes.Array)
    m.llf_ = held
    with tempfile.TemporaryDirectory(prefix="arima_output_store_") as tmp:
        path = str(Path(tmp) / "m.npz")
        m.save(path)
        back = mojolearn.ARIMA.load(path)
        assert np.asarray(back.llf_).tobytes() == np.asarray(held).tobytes()
        assert np.asarray(back.forecast(5)).tobytes() == np.asarray(m.forecast(5)).tobytes()


if __name__ == "__main__":
    for name in [n for n in sorted(globals()) if n.startswith("test_")]:
        globals()[name]()
        print("ok", name)
