# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`ExponentialSmoothing(initialization_method=...)` (2026-09-22).

"estimated" (the default) estimates the initial level, trend and seasonal
states jointly with alpha, beta and gamma over all `n` points
(`holtwinters/impl/internal/hw_estimate.mojo`); "heuristic" and its alias
"cuml" are cuML's fit.

Surface checks run anywhere. Runtime checks skip, by name, when nothing in
this process fits Holt-Winters. The estimated method's refit bits, parameter
ranges and clean-series forecast quality are native:
checks/native_tests/tsa_holtwinters_initialization.mojo.

    cd python && python3 -m pytest mojolearn/tests/test_holtwinters_initialization.py
"""
import pytest


def _series(np, n, b, f):
    t = np.arange(n, dtype=np.float64)
    rows = [20.0 + (s + 1) * 3.0 * np.sin(2 * np.pi * t / f + s) + 0.05 * t + 0.5 * np.cos(0.37 * t * (s + 1))
            for s in range(b)]
    return np.ascontiguousarray(np.stack(rows).astype(np.float32))


def _fit_or_skip(y, **kw):
    import mojolearn
    try:
        return mojolearn.ExponentialSmoothing(y, **kw).fit()
    except (ImportError, NotImplementedError, RuntimeError) as exc:
        pytest.skip(f"no binding fits Holt-Winters in this process: {type(exc).__name__}: {exc}")



# -- surface ------------------------------------------------------------------

def test_default_is_estimated_and_names_are_checked():
    import inspect
    from mojolearn import ExponentialSmoothing
    sig = inspect.signature(ExponentialSmoothing)
    assert sig.parameters["initialization_method"].default == "estimated"
    for name in ("estimated", "heuristic", "cuml"):
        assert ExponentialSmoothing([1.0] * 8, initialization_method=name).initialization_method == name
    for bad in ("known", "legacy-heuristic", None, 1):
        with pytest.raises(ValueError, match="initialization_method"):
            ExponentialSmoothing([1.0] * 8, initialization_method=bad)


# -- runtime ------------------------------------------------------------------

# The 0.8.13 bit-pin test was deleted on 2026-11-15 (Andrew): bits are never held to an older version's bits
# (AGENTS.md "Identity and bits"); the method's correctness is covered by the Mojo check tsa_holtwinters_initialization.

def test_saved_model_keeps_the_method(tmp_path):
    np = pytest.importorskip("numpy")
    from mojolearn import ExponentialSmoothing, _serialize
    from mojolearn._tsa_impl import _HW_FORMAT
    y = _series(np, 48, 2, 4)
    for name in ("estimated", "heuristic"):
        m = _fit_or_skip(y, seasonal_periods=4, ts_num=2, initialization_method=name)
        path = str(tmp_path / f"{name}.npz")
        m.save(path)
        back = ExponentialSmoothing.load(path)
        assert back.initialization_method == name
        assert np.asarray(back.forecast(8)).tobytes() == np.asarray(m.forecast(8)).tobytes()
    # a file written before the field existed loads as the only method there was
    arrays = _serialize.read_npz(str(tmp_path / "heuristic.npz"), _HW_FORMAT)
    arrays.pop("initialization_method")
    old = str(tmp_path / "old.npz")
    _serialize.write_npz(old, arrays)
    assert ExponentialSmoothing.load(old).initialization_method == "heuristic"
