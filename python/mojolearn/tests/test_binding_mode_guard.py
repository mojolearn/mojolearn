"""Call-time compiled-mode checks; no native learner execution."""
from types import SimpleNamespace
import pytest
from mojolearn import _backend


@pytest.mark.parametrize('name', ['_mojolearn', '_mojolearn_gbdt', '_mojolearn_rf', '_mojolearn_trees'])
@pytest.mark.parametrize('mode,code', [('fast', 0), ('identical', 1), ('deterministic', 2)])
def test_actual_module_mode(monkeypatch, name, mode, code):
    prefix = _backend._vendor_fn(name).removesuffix('_vendor')
    module = SimpleNamespace(**{prefix + '_numeric_mode': lambda: code})
    selected = SimpleNamespace(mode=mode, **{name: module})
    monkeypatch.setattr(_backend, 'load_set', lambda _: selected)
    assert _backend.binding(name, mode) is module
    # Sabotage the called extension inside an already cached, otherwise
    # correctly labeled set. A sibling GBDT witness cannot detect this.
    setattr(module, prefix + '_numeric_mode', lambda: 99)
    with pytest.raises(RuntimeError, match='compiled for unknown'):
        _backend.binding(name, mode)


def test_identical_refuses_fast_binary(monkeypatch):
    module = SimpleNamespace(rf_numeric_mode=lambda: 0)
    monkeypatch.setattr(_backend, 'load_set', lambda _: SimpleNamespace(
        mode='identical', _mojolearn_rf=module))
    with pytest.raises(RuntimeError, match='compiled for fast.*requested identical'):
        _backend.binding('_mojolearn_rf', 'identical')


def test_legacy_without_readback(monkeypatch):
    module = SimpleNamespace()
    monkeypatch.setattr(_backend, 'load_set', lambda _: SimpleNamespace(
        mode='fast', _mojolearn_rf=module))
    assert _backend.binding('_mojolearn_rf', 'fast') is module


@pytest.mark.parametrize('mode', ['', 0, False, 1, 'unknown', []])
def test_invalid_mode_never_falls_back(monkeypatch, mode):
    def unexpected_load(_):
        raise AssertionError('invalid mode reached loader')
    monkeypatch.setattr(_backend, 'load_set', unexpected_load)
    with pytest.raises(ValueError, match='numeric_mode'):
        _backend.binding('_mojolearn_rf', mode)
