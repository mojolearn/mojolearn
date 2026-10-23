"""Sequential import connection with inert native substitutes, no product loads."""
from types import BuiltinFunctionType, SimpleNamespace

import pytest

from test_model_selection_connection import connection


def test_sequential_connects_one_native_entry_and_retains_legacy_context(monkeypatch):
    calls = []

    def native(name):
        calls.append(name)
        return SimpleNamespace(sequential_native_call=len)

    bridge, context = connection(monkeypatch, native, '_native_runtime_sequential_bridge')
    assert bridge.connect(context) is len
    assert calls == ['_mojolearn_runtime_sequential_host']
    assert context['sequential_owner_entry'] is len
    assert context['sequential_builtin_type'] is BuiltinFunctionType


def test_sequential_rejects_python_entry_without_connecting_it(monkeypatch):
    bridge, context = connection(monkeypatch, lambda name: SimpleNamespace(
        sequential_native_call=lambda *args: None), '_native_runtime_sequential_bridge')
    with pytest.raises(TypeError, match='compiled sequential_native_call'):
        bridge.connect(context)
    assert 'sequential_owner_entry' not in context


def test_sequential_missing_owner_preserves_explicit_import_refusal(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError('Sequential owner is not built', name=name)

    bridge, context = connection(monkeypatch, missing, '_native_runtime_sequential_bridge')
    with pytest.raises(ModuleNotFoundError, match='Sequential owner is not built'):
        bridge.connect(context)
    assert 'sequential_owner_entry' not in context
