"""Causal model source controls; compilation and runtime checks are separate."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = (ROOT / "bindings/causal_lm_public.mojo").read_text()


def test_compiled_entry_resolution_admits_raw_storage_before_keyed_reads():
    fields = PUBLIC.split("def _native_entry_fields", 1)[1].split("def _native_has_entry", 1)[0]
    assert 'py_type(entries) is types.ModuleType' in fields
    assert 'py_type(entries) is types.SimpleNamespace' in fields
    assert '"PyModule_GetDict"' in fields and '"PyObject_GenericGetDict"' in fields
    assert 'py_type(keys[i]) is not bi.str' in fields
    function = PUBLIC.split("def _native_function", 1)[1].split("def causal_lm_admit_binding", 1)[0]
    assert 'py_get(_native_entry_fields(module), name)' in function
    assert 'getattr' not in function and '.getattr' not in fields
    optional = PUBLIC.split("def _native_has_entry", 1)[1].split("def _native_function", 1)[0]
    assert 'py_contains(_native_entry_fields(entries), name)' in optional
