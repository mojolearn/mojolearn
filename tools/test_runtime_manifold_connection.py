"""Manifold import connections retain native entries and missing-owner refusal."""
from types import SimpleNamespace

import pytest

from test_model_selection_connection import connection


def test_manifold_missing_owner_keeps_shared_metadata_and_refuses_operations(monkeypatch):
    calls = []

    def missing(name):
        calls.append(name)
        raise ModuleNotFoundError('manifold owner is not built', name=name)

    bridge, context = connection(monkeypatch, missing, '_native_runtime_manifold_bridge')
    entry, found = bridge.connect()
    assert found is context
    assert context['family_mani_door'] is entry
    assert bridge.connect() == (entry, context)
    assert calls == ['_mojolearn_runtime_manifold_host']
    with pytest.raises(ModuleNotFoundError, match='manifold owner is not built'):
        entry(None, 'family_mani_cmds_fit', (), context)


def test_manifold_modules_reuse_the_exact_native_door(monkeypatch):
    calls = []

    def native(name):
        calls.append(name)
        return SimpleNamespace(manifold_native_call=len)

    bridge, context = connection(monkeypatch, native, '_native_runtime_manifold_bridge')
    assert bridge.connect() == (len, context)
    assert bridge.connect() == (len, context)
    assert context['family_mani_door'] is len
    assert calls == ['_mojolearn_runtime_manifold_host']


def test_manifold_python_entry_is_rejected_before_publishing_a_door(monkeypatch):
    bridge, context = connection(monkeypatch, lambda name: SimpleNamespace(
        manifold_native_call=lambda *args: None), '_native_runtime_manifold_bridge')
    with pytest.raises(TypeError, match='compiled manifold_native_call'):
        bridge.connect()
    assert bridge._ENTRY is None
    assert 'family_mani_door' not in context
