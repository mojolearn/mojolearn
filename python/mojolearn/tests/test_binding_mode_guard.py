"""Call-time compiled-mode checks; no native learner execution."""
from types import SimpleNamespace
import pytest
from mojolearn import _backend


@pytest.mark.parametrize('name', ['_mojolearn', '_mojolearn_gbdt', '_mojolearn_rf', '_mojolearn_trees'])
@pytest.mark.parametrize('mode,code', [('fast', 0), ('identical', 1), ('deterministic', 2)])
def test_actual_module_mode(monkeypatch, name, mode, code):
    if name in _backend._IDENTICAL_ONLY and mode != 'identical':
        # DEVIATION 2490: only the tree lanes have a lower tier. The base
        # binding refuses BY NAME before any set is loaded, so the sabotage
        # below is unreachable there; that refusal is its own test.
        monkeypatch.setattr(_backend, 'load_set', lambda _: pytest.fail('load_set reached'))
        with pytest.raises(ValueError, match='has no .* tier'):
            _backend.binding(name, mode)
        return
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


@pytest.mark.parametrize('mode', ['fast', 'deterministic'])
def test_identical_only_lanes_refuse_lower_tiers_by_name(monkeypatch, mode):
    """DEVIATION 2490: every binding but the three tree lanes refuses a
    lower tier at the choke point, before any set is loaded, with a sentence
    that names the rule. The tree lanes are the allowlist."""
    monkeypatch.setattr(_backend, 'load_set', lambda _: pytest.fail('load_set reached'))
    assert _backend._TIERED == frozenset({'_mojolearn_gbdt', '_mojolearn_rf', '_mojolearn_trees'})
    assert _backend._IDENTICAL_ONLY == frozenset(_backend._MODULES) - _backend._TIERED
    for name in _backend._IDENTICAL_ONLY:
        with pytest.raises(ValueError, match='tree lanes') as info:
            _backend.binding(name, mode)
        assert name in str(info.value) and mode in str(info.value)
