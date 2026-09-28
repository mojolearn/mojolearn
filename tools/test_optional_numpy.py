"""The optional dependency guard must preserve NumPy and explain its absence."""
import builtins
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'python/mojolearn/_optional_numpy.py'


def load_guard():
    spec = importlib.util.spec_from_file_location('optional_numpy_under_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importing_guard_does_not_need_numpy_and_missing_feature_is_actionable(monkeypatch):
    original = builtins.__import__
    def without_numpy(name, *args, **kwargs):
        if name == 'numpy':
            raise ModuleNotFoundError("No module named 'numpy'", name='numpy')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_numpy)
    guard = load_guard()
    with pytest.raises(ModuleNotFoundError, match=r'mojolearn\[numpy\]') as raised:
        guard.require_numpy('TSNE')
    assert raised.value.name == 'numpy'
    assert 'TSNE' in str(raised.value)


def test_guard_returns_existing_numpy_module_without_changing_rng():
    import numpy as np
    state = np.random.get_state()
    assert load_guard().require_numpy('optional API') is np
    after = np.random.get_state()
    assert state[0] == after[0]
    assert (state[1] == after[1]).all()
    assert state[2:] == after[2:]


@pytest.mark.parametrize('error', [
    ModuleNotFoundError("missing internal dependency", name='numpy_internal'),
    ImportError('NumPy binary incompatible'),
])
def test_broken_numpy_installation_is_not_hidden(monkeypatch, error):
    guard = load_guard()
    original = builtins.__import__
    def broken_numpy(name, *args, **kwargs):
        if name == 'numpy':
            raise error
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', broken_numpy)
    with pytest.raises(type(error)) as raised:
        guard.require_numpy('optional API')
    assert raised.value is error
