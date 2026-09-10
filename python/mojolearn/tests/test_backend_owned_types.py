"""Exact-file module reuse avoids duplicate Mojo type registration."""
import sys
from types import ModuleType

import pytest
from mojolearn import _backend


@pytest.mark.parametrize('same_binary', [True, False])
def test_mode_set_reuses_only_the_canonical_binary(monkeypatch, tmp_path, same_binary):
    name = '_mojolearn_byte_lm'
    binary = tmp_path / (name + '.so')
    binary.touch()
    canonical = ModuleType('mojolearn.' + name)
    canonical.__file__ = str(binary if same_binary else tmp_path / 'other.so')
    alias = 'mojolearn._sets.identical.' + name
    monkeypatch.setitem(sys.modules, canonical.__name__, canonical)
    monkeypatch.delitem(sys.modules, alias, raising=False)
    monkeypatch.setattr(_backend, '_MODULES', (name,))
    monkeypatch.setattr(_backend, '_SETS', {})
    monkeypatch.setattr(_backend, 'tier_dir', lambda _: str(tmp_path))
    checked = []
    monkeypatch.setattr(_backend, '_check_vendor', lambda *args: checked.append(args))

    def loader(*args):
        raise RuntimeError('separate binary must initialize')

    monkeypatch.setattr(_backend.importlib.machinery, 'ExtensionFileLoader', loader)
    try:
        if same_binary:
            result = _backend.load_set('identical')
            assert getattr(result, name) is canonical
            assert sys.modules[alias] is canonical
            assert checked == [(canonical, name, str(binary))]
        else:
            with pytest.raises(RuntimeError, match='separate binary'):
                _backend.load_set('identical')
    finally:
        sys.modules.pop(alias, None)
