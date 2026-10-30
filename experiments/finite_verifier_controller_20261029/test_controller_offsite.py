import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest
from controller_offsite import defer_archive, pending_count, upload_one


def queue(tmp_path):
    archive = tmp_path / 'a.tar.gz'
    raw = b'original complete archive'
    archive.write_bytes(raw)
    return defer_archive(tmp_path, tmp_path / 'unit', archive, raw, 'key', 'source', {'vendor': 'apple'}), raw


def test_queue_is_explicit_pending_and_idempotent(tmp_path):
    first, raw = queue(tmp_path)
    again, _ = queue(tmp_path)
    assert first == again and pending_count(tmp_path) == 1
    assert not (tmp_path / 'unit/r2-retention.json').exists()
    assert Path(json.loads(first.read_text())['archive']).read_bytes() == raw


def test_corruption_stops_before_any_remote_operation(tmp_path):
    path, raw = queue(tmp_path)
    Path(json.loads(path.read_text())['archive']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        upload_one(path, lambda *a, **k: pytest.fail('must not contact R2'))


def test_failed_put_keeps_pending_without_receipt(tmp_path):
    path, _ = queue(tmp_path)
    with pytest.raises(RuntimeError, match='PUT 500'):
        upload_one(path, lambda method, key, **kw: (404 if method == 'GET' else 500, b''))
    assert pending_count(tmp_path) == 1 and not (tmp_path / 'unit/r2-retention.json').exists()


def test_readback_mismatch_never_becomes_retained(tmp_path):
    path, _ = queue(tmp_path)
    with pytest.raises(RuntimeError, match='readback'):
        upload_one(path, lambda *a, **k: (200, b'wrong bytes'))
    assert pending_count(tmp_path) == 1


def test_exact_readback_reuses_object_and_records_retention(tmp_path):
    path, raw = queue(tmp_path)
    calls = []
    assert upload_one(path, lambda method, key, **kw: (calls.append(method) or (200, raw))) == 'retained'
    assert calls == ['GET'] and pending_count(tmp_path) == 0
    receipt = json.loads((tmp_path / 'unit/r2-retention.json').read_text())
    assert receipt['readback_verified'] and receipt['sha256'] == hashlib.sha256(raw).hexdigest()


def test_standard_library_404_response_uploads_then_checks_exact_bytes(tmp_path):
    path, raw = queue(tmp_path)
    calls = []
    def request(method, key, **kw):
        calls.append(method)
        if len(calls) == 1:
            raise HTTPError('https://example.invalid/key', 404, 'not found', {}, None)
        if method == 'PUT':
            assert kw['data'] == raw
            return 200, b''
        return 200, raw
    assert upload_one(path, request) == 'retained'
    assert calls == ['GET', 'PUT', 'GET'] and pending_count(tmp_path) == 0


def test_partial_raw_failure_archive_remains_explicitly_partial(tmp_path):
    raw = b'original failed unit evidence'
    archive = tmp_path / 'failure.partial.tar.gz'
    archive.write_bytes(raw)
    path = defer_archive(tmp_path, tmp_path / 'unit', archive, raw, 'key', 'source', {'vendor': 'cpu'})
    upload_one(path, lambda *a, **kw: (200, raw))
    assert json.loads((tmp_path / 'unit/r2-retention.json').read_text())['partial'] is True
