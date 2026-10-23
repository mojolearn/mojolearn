"""Import connection/refusal contract with isolated native-loader test doubles."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "python/mojolearn"


def connection(monkeypatch, loader):
    # Import only the two connection modules. No mojolearn package operation,
    # optional dependency, native binary or compiler is needed by this test.
    name = "partition_connection_test"
    package = ModuleType(name)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, name, package)
    backend = SimpleNamespace(load_host_module=loader)
    monkeypatch.setattr(package, "_backend", backend, raising=False)

    def load(member):
        full = name + "." + member
        spec = importlib.util.spec_from_file_location(full, PACKAGE / (member + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full, module)
        monkeypatch.setattr(package, member, module, raising=False)
        spec.loader.exec_module(module)
        return module

    family = load("_native_family_bridge")
    context = {"import_metadata": object()}
    monkeypatch.setattr(family, "shared_context", lambda: context)
    return load("_native_model_selection_bridge"), context


def test_missing_owner_keeps_metadata_connection_and_only_refuses(monkeypatch):
    loads = []

    def missing(name):
        loads.append(name)
        raise ModuleNotFoundError("native owner is not built", name=name)

    bridge, context = connection(monkeypatch, missing)
    entry, found = bridge.connect()
    assert found is context
    assert bridge.connect() == (entry, context)
    assert loads == ["_mojolearn_runtime_model_selection_host"]
    with pytest.raises(ModuleNotFoundError, match="native owner is not built") as caught:
        entry(None, "family_msel_split", (), context)
    assert caught.value.name == "_mojolearn_runtime_model_selection_host"


def test_present_owner_retains_the_verified_entry_once(monkeypatch):
    loads = []

    def native_module(name):
        loads.append(name)
        return SimpleNamespace(model_selection_native_call=len)

    bridge, context = connection(monkeypatch, native_module)
    assert bridge.connect() == (len, context)
    assert bridge.connect() == (len, context)
    assert loads == ["_mojolearn_runtime_model_selection_host"]


def test_python_callable_is_not_treated_as_a_missing_native_owner(monkeypatch):
    bridge, _ = connection(monkeypatch, lambda name: SimpleNamespace(
        model_selection_native_call=lambda *args: None))
    with pytest.raises(TypeError, match="compiled model_selection_native_call"):
        bridge.connect()
    assert bridge._ENTRY is None
