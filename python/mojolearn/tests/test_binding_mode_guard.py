"""Call-time compiled-mode checks; no native learner execution."""
from types import SimpleNamespace
import pytest
from mojolearn import _backend


@pytest.mark.parametrize('name', ['_mojolearn', '_mojolearn_gbdt', '_mojolearn_rf', '_mojolearn_trees', '_mojolearn_estimators', '_mojolearn_svm'])
@pytest.mark.parametrize('mode,code', [('fast', 0), ('identical', 1), ('deterministic', 2)])
def test_actual_module_mode(monkeypatch, name, mode, code):
    if not _backend._offers(name, mode):
        # The base binding (and every neural family) has no lower tier and
        # refuses BY NAME before any set is loaded, so the sabotage below is
        # unreachable there; that refusal is its own test.
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
def test_lanes_refuse_tiers_they_do_not_ship_by_name(monkeypatch, mode):
    """Trees ship fast and deterministic, classical ML ships fast, the neural
    families and the base binding ship identical only. Every refusal happens
    at the choke point, before any set is loaded, with a sentence that names
    the rule and the binding."""
    monkeypatch.setattr(_backend, 'load_set', lambda _: pytest.fail('load_set reached'))
    assert _backend._TIERED == frozenset({'_mojolearn_gbdt', '_mojolearn_rf', '_mojolearn_trees'})
    assert _backend._FAST_TIERED == _backend._TIERED | _backend._CLASSICAL_FAST
    assert _backend._IDENTICAL_ONLY == frozenset(_backend._MODULES) - _backend._FAST_TIERED
    for neural in ('_mojolearn_training', '_mojolearn_byte_lm', '_mojolearn_mamba',
                   '_mojolearn_transformer', '_mojolearn_embedding'):
        assert neural in _backend._IDENTICAL_ONLY, neural
    refused = [n for n in _backend._MODULES if not _backend._offers(n, mode)]
    assert refused
    for name in refused:
        with pytest.raises(ValueError, match='tree lanes') as info:
            _backend.binding(name, mode)
        assert name in str(info.value) and mode in str(info.value)
    if mode == 'fast':
        assert set(refused) == set(_backend._IDENTICAL_ONLY)
    else:
        assert set(refused) == set(_backend._MODULES) - _backend._TIERED
