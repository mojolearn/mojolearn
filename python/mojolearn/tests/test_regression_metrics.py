"""Host-only public ABI/validation tests; device arithmetic has its own gate."""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from mojolearn import metrics
from mojolearn import _backend

NAMES = ("mean_squared_error", "mean_absolute_error", "root_mean_squared_error")
MODES = {"fast": 0, "identical": 1, "deterministic": 2}


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


@pytest.mark.parametrize("name", NAMES)
def test_public_export_and_borrowed_array_contents(name, binding):
    assert name in metrics.__all__
    source = np.arange(20, dtype=np.float32)
    source.flags.writeable = False
    true, pred = source[::3], source[::-3].reshape(-1, 1)
    before = source.copy()
    result = getattr(metrics, name)(true, pred)
    assert type(result) is float and result == 2.5
    called, arrays, params = binding.calls.pop()
    assert called == name and params == [7]
    np.testing.assert_array_equal(arrays[0], true)
    np.testing.assert_array_equal(arrays[1], pred.ravel())
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
    {"sample_weight": []}, {"sample_weight": np.ones(2)},
    {"multioutput": "raw_values"}, {"multioutput": None},
    {"multioutput": np.array([0.5, 0.5])}, {"multioutput": [1.0]},
])
def test_unsupported_weight_or_multioutput(name, kwargs, binding):
    x = np.ones(2, dtype=np.float32)
    with pytest.raises(NotImplementedError, match="sample_weight|multioutput"):
        getattr(metrics, name)(x, x, **kwargs)
    assert not binding.calls


@pytest.mark.parametrize("name", NAMES)
def test_singleton_and_overflow_result_are_preserved(name, binding):
    binding.result = np.inf
    x = np.array([np.finfo(np.float32).max], dtype=np.float32)
    assert getattr(metrics, name)(x, x) == float("inf")
    assert binding.calls[0][2] == [1]


@pytest.mark.parametrize("mode", ["", "bogus", False, 1, [], np.array(["fast"])])
def test_invalid_mode_refused(mode, binding):
    x = np.ones(2, dtype=np.float32)
    with pytest.raises(ValueError, match="numeric_mode"):
        metrics.mean_squared_error(x, x, numeric_mode=mode)
    assert not binding.calls


def test_shared_loader_resolves_live_default_and_explicit_mode(monkeypatch):
    """`_mojolearn_metrics` ships fast and identical (classical FAST,
    2026-09-25): the live default resolves the identical set, an explicit
    'identical' does too, 'fast' explicit or as the process default resolves
    the fast set, and 'deterministic' (tree lanes only) refuses BY NAME
    before `load_set` is reached.

    This is the GPU install's resolution, so the CPU-only marker is cleared:
    on a CPU-only install `binding()` never reaches `load_set` (it serves
    the host proxy, see the test below)."""
    fakes = {m: Binding(m) for m in ("identical", "fast")}
    loaded = []
    current = ["identical"]
    monkeypatch.setattr(_backend, "_CPU_ONLY", None)
    monkeypatch.setattr(_backend, "default_mode", lambda: current[0])
    # The input path's finiteness check resolves from the identical base
    # binding through the same `binding()`; plant it so this test sees only
    # the metrics binding's resolution.
    from mojolearn import _buffer
    monkeypatch.setitem(_buffer._NATIVE, "all_finite_f32", lambda addr, n: 1)

    def load(mode):
        loaded.append(mode)
        return SimpleNamespace(mode=mode, _mojolearn_metrics=fakes[mode])

    monkeypatch.setattr(_backend, "load_set", load)
    x = np.ones(2, dtype=np.float32)
    metrics.mean_squared_error(x, x)
    metrics.mean_squared_error(x, x, numeric_mode="identical")
    assert loaded == ["identical", "identical"]
    for explicit, default in [("fast", "identical"), (None, "fast")]:
        current[0] = default
        metrics.mean_squared_error(x, x, numeric_mode=explicit)
    assert loaded == ["identical", "identical", "fast", "fast"]
    for explicit, default in [("deterministic", "identical"), (None, "deterministic")]:
        current[0] = default
        with pytest.raises(ValueError, match="tree lanes"):
            metrics.mean_squared_error(x, x, numeric_mode=explicit)
    assert len(loaded) == 4, "the deterministic tier reached load_set"


def test_cpu_only_install_serves_the_installed_module_without_load_set(monkeypatch):
    """On a CPU-only install (`_backend._CPU_ONLY` set by `_select_cpu_only`)
    `binding()` answers with the module installed under the canonical name,
    the host proxy of the metrics host binding (e3dcae470), and never opens a
    tier directory; a lower tier still refuses by name before anything
    resolves."""
    import sys
    from mojolearn import _buffer
    fake = Binding("identical")
    monkeypatch.setattr(_backend, "_CPU_ONLY", "no GPU binding on this box (test)")
    monkeypatch.setattr(_backend, "default_mode", lambda: "identical")
    monkeypatch.setitem(sys.modules, "mojolearn._mojolearn_metrics", fake)
    monkeypatch.setitem(_buffer._NATIVE, "all_finite_f32", lambda addr, n: 1)

    def load(mode):
        raise AssertionError(f"load_set({mode!r}) reached on a CPU-only install")

    monkeypatch.setattr(_backend, "load_set", load)
    x = np.arange(3, dtype=np.float32)
    assert metrics.mean_squared_error(x, x) == 2.5
    assert metrics.mean_squared_error(x, x, numeric_mode="identical") == 2.5
    assert [c[0] for c in fake.calls] == ["mean_squared_error"] * 2
    for explicit in ("fast", "deterministic"):
        with pytest.raises(ValueError, match="IDENTICAL only"):
            metrics.mean_squared_error(x, x, numeric_mode=explicit)
    assert len(fake.calls) == 2
