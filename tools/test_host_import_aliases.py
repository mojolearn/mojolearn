"""Native import aliases with inert modules/files; never loads a shared library."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def connection(monkeypatch, tmp_path):
    package = ModuleType('alias_test_package')
    package.__path__ = [str(ROOT / 'python/mojolearn')]
    manifest = ModuleType('alias_test_package.host_surface')
    manifest.HOST_IMPORT_ALIASES = {'legacy': ({
        'binding': 'owner', 'marker': 'owner_abi', 'abi': sys.getrecursionlimit(),
        'functions': ('first', 'second'), 'types': ('NativeType',),
    },)}
    legacy, owner = ModuleType('legacy'), ModuleType('owner')
    legacy.owner_abi, legacy.first, legacy.second = sys.getrecursionlimit, print, print
    owner.first, owner.second, owner.NativeType = len, abs, int
    artifact = tmp_path / 'owner.so'
    artifact.write_bytes(b'inert; must never be loaded')
    loads = []
    backend = ModuleType('alias_test_package._backend')
    backend.host_module_path = lambda name: str(artifact)

    def load(name):
        loads.append(name)
        return owner

    backend.load_host_module = load
    for module in (package, manifest, backend):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    package.host_surface, package._backend = manifest, backend
    spec = importlib.util.spec_from_file_location('alias_test_package._host_import_aliases',
                                                 ROOT / 'python/mojolearn/_host_import_aliases.py')
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    return bridge, legacy, owner, artifact, loads


def test_aliases_are_original_native_objects_and_one_existing_type(monkeypatch, tmp_path):
    bridge, legacy, owner, _, loads = connection(monkeypatch, tmp_path)
    bridge.connect('legacy', legacy)
    assert loads == ['owner']
    assert legacy.first is owner.first
    assert legacy.second is owner.second
    assert legacy.NativeType is owner.NativeType
    assert legacy._mojolearn_missing_import_owners == ()


def test_missing_owner_preserves_native_refusal_entries_without_loading(monkeypatch, tmp_path):
    bridge, legacy, _, artifact, loads = connection(monkeypatch, tmp_path)
    artifact.unlink()
    bridge.connect('legacy', legacy)
    assert loads == []
    assert legacy.first is print and legacy.second is print
    assert legacy._mojolearn_missing_import_owners == ('owner',)
    assert 'NativeType' not in vars(legacy)


@pytest.mark.parametrize('bad', ['missing', 'python', 'wrong_version'])
def test_old_or_unverified_abi_refuses_before_loading_another_type_owner(monkeypatch, tmp_path, bad):
    bridge, legacy, _, _, loads = connection(monkeypatch, tmp_path)
    callbacks = []
    if bad == 'missing':
        del legacy.owner_abi
    elif bad == 'python':
        legacy.owner_abi = lambda: callbacks.append(True)
    else:
        legacy.owner_abi = sys.getdefaultencoding
    with pytest.raises(ImportError, match='ownership ABI'):
        bridge.connect('legacy', legacy)
    assert loads == []
    assert callbacks == []


@pytest.mark.parametrize('bad', ['missing', 'python', 'type'])
def test_invalid_owner_inventory_never_publishes_partial_aliases(monkeypatch, tmp_path, bad):
    bridge, legacy, owner, _, _ = connection(monkeypatch, tmp_path)
    callbacks = []
    owner.__getattr__ = lambda name: callbacks.append(name)
    if bad == 'missing':
        del owner.second
    elif bad == 'python':
        owner.second = lambda *args: callbacks.append(args)
    else:
        owner.NativeType = object()
    with pytest.raises(ImportError, match='native alias owner'):
        bridge.connect('legacy', legacy)
    assert legacy.first is print and legacy.second is print
    assert 'NativeType' not in vars(legacy)
    assert callbacks == []


def test_non_module_target_is_refused_before_its_protocol(monkeypatch, tmp_path):
    bridge, _, _, _, loads = connection(monkeypatch, tmp_path)
    callbacks = []

    class Target:
        def __getattribute__(self, name):
            callbacks.append(name)
            return super().__getattribute__(name)

    with pytest.raises(ImportError, match='exact extension module'):
        bridge.connect('legacy', Target())
    assert callbacks == []
    assert loads == []
