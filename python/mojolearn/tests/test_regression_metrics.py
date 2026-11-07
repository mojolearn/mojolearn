"""Host-only public ABI/validation tests; device arithmetic has its own gate."""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from mojolearn import metrics
from mojolearn import _backend

NAMES = ("mean_squared_error", "mean_absolute_error", "root_mean_squared_error")
MODES = {"fast": 0, "identical": 1}


class Binding:
    def __init__(self, mode="fast", result=2.5):
        self.mode, self.result, self.calls = mode, result, []

    def metrics_numeric_mode(self):
        return MODES[self.mode]

    def __getattr__(self, name):
        if name not in (*NAMES, "accuracy_score", "r2_score"):
            raise AttributeError(name)

        def call(a, b, params):
            n = int(params[0])
            dtype = ctypes.c_int32 if name == "accuracy_score" else ctypes.c_float
            arrays = [np.ctypeslib.as_array((dtype * n).from_address(int(p))).copy()
                      for p in (a, b)]
            self.calls.append((name, arrays, params))
            return np.float32(self.result)
        return call


@pytest.fixture
def binding(monkeypatch):
    """Plant the fake for `_mojolearn_metrics` ONLY. Every other name goes to
    the real resolver: the input path's finiteness check asks
    `_backend.binding("_mojolearn", mode="identical")` for `all_finite_f32`
    (`_buffer._native`), and since 97d087fc6 removed its Python fallback a
    fake answering that lookup refused every call that reached it. Through
    the real resolver it is the identical base binding on a GPU install and
    the core host binding on a CPU-only one."""
    fake = Binding()
    real = _backend.binding
    monkeypatch.setattr(_backend, "default_mode", lambda: "fast")
    monkeypatch.setattr(
        _backend, "binding",
        lambda name, mode=None: fake if name == "_mojolearn_metrics" else real(name, mode))
    return fake


# The public call resolves its binding inside the native operation
# (d7415403cd, "Move public metric binding selection inside native
# operations"): a Python fake planted in `_backend.binding` is never reached,
# so the value and overflow tests below check the real binding's answer.
_EXPECTED = {
    "mean_squared_error": lambda t, p: float(np.mean((t - p) ** 2, dtype=np.float32)),
    "mean_absolute_error": lambda t, p: float(np.mean(np.abs(t - p), dtype=np.float32)),
    "root_mean_squared_error":
        lambda t, p: float(np.sqrt(np.mean((t - p) ** 2, dtype=np.float32))),
}


@pytest.mark.parametrize("name", NAMES)
def test_public_export_and_borrowed_array_contents(name):
    assert name in metrics.__all__
    source = np.arange(20, dtype=np.float32)
    source.flags.writeable = False
    true, pred = source[::3], source[::-3].reshape(-1, 1)
    before = source.copy()
    result = getattr(metrics, name)(true, pred)
    assert type(result) is float
    assert result == pytest.approx(_EXPECTED[name](true, pred.ravel()), rel=1e-6)
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.int32, np.uint8,
                                   np.bool_, np.complex64, object, str])
def test_strict_float32_dtype(name, side, dtype, binding):
    args = [np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)]
    args[side] = np.ones(2, dtype=dtype)
    with pytest.raises(TypeError, match="float32"):
        getattr(metrics, name)(*args)
    assert not binding.calls


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bad", [np.array(1, dtype=np.float32),
    np.ones((2, 2), dtype=np.float32), np.ones((1, 2), dtype=np.float32),
    np.ones((2, 1, 1), dtype=np.float32), np.empty(0, dtype=np.float32),
    np.array([np.nan, 0], dtype=np.float32),
    np.array([np.inf, 0], dtype=np.float32),
    np.array([-np.inf, 0], dtype=np.float32), np.ones(3, dtype=np.float32)])
@pytest.mark.parametrize("side", [0, 1])
def test_invalid_shape_length_or_value(name, bad, side, binding):
    args = [np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)]
    args[side] = bad
    with pytest.raises(ValueError):
        getattr(metrics, name)(*args)
    assert not binding.calls


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("kwargs", [
    {"sample_weight": []}, {"multioutput": np.array([0.5, 0.5])},
    {"multioutput": [1.0]},
])
def test_invalid_weight_or_multioutput(name, kwargs, binding):
    x = np.ones(2, dtype=np.float32)
    with pytest.raises(ValueError):
        getattr(metrics, name)(x, x, **kwargs)
    assert not binding.calls


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("kwargs", [
    {"sample_weight": np.ones(2)}, {"multioutput": "raw_values"},
    {"multioutput": None},
])
def test_supported_options_reach_expansion(name, kwargs, binding):
    # The options reach the one native metric call (the Python expansion
    # routing they used to pass through is gone): weights and multioutput are
    # honoured and the plain binding is never called.
    yt = np.asarray([1.0, 2.0], dtype=np.float32)
    yp = np.asarray([1.5, 1.0], dtype=np.float32)
    w = np.asarray(kwargs.get("sample_weight", np.ones(2)), dtype=np.float64)
    err = (yp - yt).astype(np.float64)
    want = {"mean_squared_error": float((w * err * err).sum() / w.sum()),
            "mean_absolute_error": float((w * np.abs(err)).sum() / w.sum()),
            "root_mean_squared_error": float(np.sqrt((w * err * err).sum() / w.sum()))}.get(name)
    got = getattr(metrics, name)(yt, yp, **kwargs)
    if kwargs.get("multioutput") == "raw_values":
        got = np.asarray(got)
        assert got.shape == (1,)
        got = float(got[0])
    if want is not None:
        assert float(got) == pytest.approx(want, rel=1e-6)
    assert not binding.calls


@pytest.mark.parametrize("name", NAMES)
def test_singleton_and_overflow_result_are_preserved(name):
    # float32 residuals with no wider accumulator: max - (-max) overflows to
    # inf, as the public docstring says, and a single sample is accepted.
    top = np.finfo(np.float32).max
    x = np.array([top], dtype=np.float32)
    assert getattr(metrics, name)(x, -x) == float("inf")
    assert getattr(metrics, name)(x, x) == 0.0


@pytest.mark.parametrize("mode", ["", "bogus", False, 1, [], np.array(["fast"])])
def test_invalid_mode_refused(mode, binding):
    x = np.ones(2, dtype=np.float32)
    with pytest.raises(ValueError, match="numeric_mode"):
        metrics.mean_squared_error(x, x, numeric_mode=mode)
    assert not binding.calls


def test_shared_loader_resolves_live_default_and_explicit_mode(monkeypatch):
    """`_mojolearn_metrics` ships fast and identical. The native operation
    reads the live `_backend._DEFAULT_MODE` when `numeric_mode` is None and
    selects that mode's import-connected set from `_backend._SETS`; an
    explicit mode selects its own set. No Python loader runs at call time.

    This is the GPU install's resolution (a CPU-only install serves the host
    module, see the test below)."""
    if _backend._CPU_ONLY is not None:
        pytest.skip("GPU install resolution; this install is CPU-only")

    def load(mode):
        raise AssertionError(f"load_set({mode!r}) reached at operation time")

    monkeypatch.setattr(_backend, "load_set", load)
    x = np.ones(2, dtype=np.float32)
    monkeypatch.setattr(_backend, "_DEFAULT_MODE", "identical")
    assert metrics.mean_squared_error(x, x) == 0.0
    assert metrics.mean_squared_error(x, x, numeric_mode="identical") == 0.0
    # An explicit mode overrides the live default.
    monkeypatch.setattr(_backend, "_DEFAULT_MODE", "fast")
    assert metrics.mean_squared_error(x, x, numeric_mode="identical") == 0.0
    # The set is looked up per call, not captured at import: without the
    # identical set the live default and the explicit mode are both refused,
    # and putting it back serves the next call.
    saved = _backend._SETS["identical"]
    monkeypatch.setattr(_backend, "_DEFAULT_MODE", "identical")
    monkeypatch.delitem(_backend._SETS, "identical")
    with pytest.raises(TypeError, match="mode set"):
        metrics.mean_squared_error(x, x)
    with pytest.raises(TypeError, match="mode set"):
        metrics.mean_squared_error(x, x, numeric_mode="identical")
    _backend._SETS["identical"] = saved
    assert metrics.mean_squared_error(x, x) == 0.0


def test_cpu_only_install_serves_the_installed_module_without_load_set(monkeypatch):
    """On a CPU-only install (`_backend._CPU_ONLY` set) the native operation
    reads the module installed under the canonical name from `sys.modules`
    on every call and never opens a tier directory. Only the exact extension
    module is served: a replaced module is refused by type and a removed one
    by name."""
    import sys
    monkeypatch.setattr(_backend, "_CPU_ONLY", "no GPU binding on this box (test)")
    monkeypatch.setattr(_backend, "_DEFAULT_MODE", "identical")

    def load(mode):
        raise AssertionError(f"load_set({mode!r}) reached on a CPU-only install")

    monkeypatch.setattr(_backend, "load_set", load)
    x = np.arange(3, dtype=np.float32)
    monkeypatch.setitem(sys.modules, "mojolearn._mojolearn_metrics", Binding("identical"))
    with pytest.raises(TypeError, match="exact extension module"):
        metrics.mean_squared_error(x, x)
    monkeypatch.delitem(sys.modules, "mojolearn._mojolearn_metrics", raising=False)
    with pytest.raises(ImportError, match="removed after import"):
        metrics.mean_squared_error(x, x, numeric_mode="identical")
