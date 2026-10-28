"""Bounded public helper probes, authored for the final compiled Mac validation."""
import json
from pathlib import Path
import pytest


def test_framework_artifact_helpers_native_and_refresh_files(tmp_path, monkeypatch):
    from mojolearn.adapters import _shipped
    from mojolearn._verify_native_runtime import audit_native_call
    doc = {'schema': _shipped.SCHEMA, 'abi': _shipped.ABI, 'max': {'version': 'fixture'}}
    (tmp_path / _shipped.MANIFEST).write_text(json.dumps(doc))
    for name in _shipped.FILES.values():
        (tmp_path / name).write_bytes(b'fixture')
    context = dict(_shipped._CONTEXT, cache={}, directory=str(tmp_path), layout_error=None)
    monkeypatch.setattr(_shipped, '_CONTEXT', context)
    assert audit_native_call(_shipped.directory).assert_native() == str(tmp_path)
    assert audit_native_call(_shipped.manifest).assert_native() == doc
    assert audit_native_call(_shipped.path, 'native').assert_native() == str(tmp_path / _shipped.FILES['native'])
    assert audit_native_call(_shipped.series, '2.13.0+rocm7.2').assert_native() == '2.13'
    extra = {'model': '1,1,1', 'library': 'discard', 'opaque': object()}
    params = _shipped._NATIVE('parameters', (extra,), context)
    assert params['opaque'] is extra['opaque'] and params['library'] == str(tmp_path / _shipped.FILES['native'])
    paths = _shipped._NATIVE('custom_extensions', (), context)
    assert paths == [Path(tmp_path / _shipped.FILES['max'])]
    assert paths[0] is not _shipped._NATIVE('custom_extensions', (), context)[0]
    with pytest.raises(KeyError):
        _shipped.path('unknown')
    (tmp_path / _shipped.FILES['native']).unlink()
    with pytest.raises(ImportError, match='is missing'):
        _shipped.path('native')
