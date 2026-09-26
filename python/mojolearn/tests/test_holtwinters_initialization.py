# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`ExponentialSmoothing(initialization_method=...)` (2026-09-22).

"estimated" (the default) estimates the initial level, trend and seasonal
states jointly with alpha, beta and gamma over all `n` points
(`holtwinters/impl/internal/hw_estimate.mojo`); "heuristic" and its alias
"cuml" are cuML's fit, which must stay BIT FOR BIT what 0.8.13 returned.

Source and surface checks run anywhere. Runtime checks skip, by name, when
nothing in this process fits Holt-Winters. The 0.8.13 digests below were
computed from the PyPI 0.8.13 wheel on an Apple M4 under IDENTICAL; the
heuristic fit is bitwise identical across columns, so they hold on every
column that fits.

    cd python && python3 -m pytest mojolearn/tests/test_holtwinters_initialization.py
"""
import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

#: sha256 over forecast(24), level_, trend_, season_, sse_, alpha_, beta_,
#: gamma_, n_iter_, criterion_ (that order), from pip mojolearn==0.8.13.
DIGESTS_0813 = {
    ("additive", 96, 2, 12, 2): "0918dea8d4662aa5b6494625d3640c1f0d092f5740da98fb8ef109cd1e6b2aa0",
    ("additive", 40, 3, 4, 3): "974ab399ec881de27d166a23c03be79c6288df3fc1b05a825110f5004da76f57",
    ("multiplicative", 96, 2, 12, 2): "fb8895af9424d729a52b29d6a39e7320952e1dd59eb20670a15ab0a7400209b2",
    ("multiplicative", 40, 3, 4, 3): "05368f46b3185986e7943897f1826933201db1f8e3dabcc6eaa4aa9bb5aedfbd",
}


def _series(np, n, b, f):
    t = np.arange(n, dtype=np.float64)
    rows = [20.0 + (s + 1) * 3.0 * np.sin(2 * np.pi * t / f + s) + 0.05 * t + 0.5 * np.cos(0.37 * t * (s + 1))
            for s in range(b)]
    return np.ascontiguousarray(np.stack(rows).astype(np.float32))


def _fit_or_skip(y, **kw):
    import mojolearn
    from mojolearn._cpu_reference import reference_training
    try:
        with reference_training():
            return mojolearn.ExponentialSmoothing(y, **kw).fit()
    except (ImportError, NotImplementedError, RuntimeError) as exc:
        pytest.skip(f"no binding fits Holt-Winters in this process: {type(exc).__name__}: {exc}")


def _digest(np, m):
    h = hashlib.sha256()
    for v in (m.forecast(24), m.level_, m.trend_, m.season_, m.sse_, m.alpha_, m.beta_, m.gamma_,
              m.n_iter_, m.criterion_):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


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


def test_codes_cross_the_binding_in_one_order():
    """The sixth params entry is the method code, on both sides."""
    from mojolearn._tsa_impl import _HW_INIT_CODES
    assert _HW_INIT_CODES == {"estimated": 1, "heuristic": 0, "cuml": 0}
    est = (ROOT / "holtwinters/impl/internal/hw_estimate.mojo").read_text(encoding="utf-8")
    assert re.search(r"^comptime HW_INIT_HEURISTIC = 0$", est, re.M)
    assert re.search(r"^comptime HW_INIT_ESTIMATED = 1$", est, re.M)
    for src in ("bindings/_mojolearn_tsa.mojo", "bindings/_mojolearn_tsa_host.mojo"):
        text = (ROOT / src).read_text(encoding="utf-8")
        assert "5  init_method" in text, src
        assert "if len(params) == 6 else 0" in text, src


def test_parallel_driver_forwards_the_method():
    text = (ROOT / "python/mojolearn/parallel_classical.py").read_text(encoding="utf-8")
    assert "initialization_method=estimator.initialization_method" in text


def test_device_and_host_share_one_estimate_function():
    """The CPU column and both device arms call the same per-element
    helpers, and nothing sums across threads."""
    oracle = (ROOT / "holtwinters/host/hw_oracle.mojo").read_text(encoding="utf-8")
    kernel = (ROOT / "holtwinters/impl/internal/hw_estimate.mojo").read_text(encoding="utf-8")
    assert "hw_estimate_series(" in oracle
    assert re.search(r"def holtwinters_estimate_gpu_kernel\([\s\S]*?hw_estimate_series\(", kernel)
    blk = kernel[kernel.index("def _blk_eval_jac["):kernel.index("def holtwinters_estimate_finish_kernel(")]
    for helper in ("_est_step(", "_sse_add(", "_est_dx(", "_est_dln(", "_est_dbn(", "_est_dsn(",
                   "_est_eval_plain(", "_seed(", "_hold("):
        assert helper in blk, f"the parallel arm does not call {helper}"
    assert "hw_est_finish(" in kernel[kernel.index("def holtwinters_estimate_finish_kernel("):]
    # no libm, no atomics, no warp or block reductions in the estimated path
    imports = set(re.findall(r"^from ([\w.]+) import", kernel, re.M))
    assert imports <= {"std.gpu", "std.memory", "std.sys.compile", "max.gpu.host", "max.gpu.memory",
                       "max.gpu.sync", "holtwinters.impl.internal.hw_utils",
                       "holtwinters.impl.tsa.holtwinters_params", "checks.numerics"}, sorted(imports)
    assert not re.search(r"\bwarp\.|block_reduce|Atomic", kernel)


# -- runtime ------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(DIGESTS_0813))
@pytest.mark.parametrize("name", ["heuristic", "cuml"])
def test_heuristic_reproduces_0813_bits(key, name):
    np = pytest.importorskip("numpy")
    import mojolearn as _ml
    if _ml._backend.requested_mode() != "identical":
        pytest.skip("DIGESTS_0813 pin IDENTICAL-tier bits; FAST is not bitwise")
    seasonal, n, b, f, sp = key
    m = _fit_or_skip(_series(np, n, b, f), seasonal=seasonal, seasonal_periods=f, start_periods=sp,
                     ts_num=b, initialization_method=name)
    assert _digest(np, m) == DIGESTS_0813[key]


@pytest.mark.parametrize("seasonal", ["additive", "multiplicative"])
def test_estimated_is_the_default_and_refits_to_the_same_bits(seasonal):
    np = pytest.importorskip("numpy")
    y = _series(np, 96, 2, 12)
    a = _fit_or_skip(y, seasonal=seasonal, seasonal_periods=12, ts_num=2)
    b = _fit_or_skip(y, seasonal=seasonal, seasonal_periods=12, ts_num=2, initialization_method="estimated")
    h = _fit_or_skip(y, seasonal=seasonal, seasonal_periods=12, ts_num=2, initialization_method="heuristic")
    assert _digest(np, a) == _digest(np, b)
    assert _digest(np, a) != _digest(np, h)
    for m in (a, h):
        assert np.asarray(m.level_).shape == (2, 96 - 12)
        assert np.isfinite(np.asarray(m.forecast(12))).all()
        alpha, beta, gamma = (np.asarray(v) for v in (m.alpha_, m.beta_, m.gamma_))
        assert ((alpha >= 0) & (alpha <= 1) & (beta >= 0) & (beta <= 1) & (gamma >= 0) & (gamma <= 1)).all()
    crit = np.asarray(a.criterion_)
    assert set(crit.tolist()) <= {0, 1, 2}


@pytest.mark.parametrize("seasonal", ["additive", "multiplicative"])
def test_estimated_forecasts_at_least_as_well_on_a_clean_series(seasonal):
    """A noiseless trend + season: the estimated fit, which scores every
    point and chooses its initial states, forecasts the continuation at
    least as well as the heuristic one, and well in absolute terms."""
    np = pytest.importorskip("numpy")
    f, n, h = 12, 72, 12
    t = np.arange(n + h, dtype=np.float64)
    season = 3.0 * np.sin(2 * np.pi * t / f) + np.cos(4 * np.pi * t / f)
    level = 40.0 + 0.3 * t
    y = level + season if seasonal == "additive" else level * (1.0 + 0.05 * season)
    y = y.astype(np.float32)
    est = _fit_or_skip(y[:n], seasonal=seasonal, seasonal_periods=f)
    heu = _fit_or_skip(y[:n], seasonal=seasonal, seasonal_periods=f, initialization_method="heuristic")
    rmse = lambda m: float(np.sqrt(np.mean((np.asarray(m.forecast(h), np.float64) - y[n:]) ** 2)))
    assert rmse(est) <= rmse(heu) * 1.001 + 1e-6
    assert rmse(est) < 0.05 * float(np.std(y))


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
