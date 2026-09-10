"""Authored host-only capture admission fixtures; root runs, never agents."""
import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / 'byte_lm_real_text_capture.py'
_spec = importlib.util.spec_from_file_location('byte_capture_portability', _path)
capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture)


@pytest.mark.parametrize('system,vendor', [('linux', 'cuda'), ('linux', 'hip'), ('darwin', 'metal')])
def test_explicit_matching_platform_vendor(monkeypatch, system, vendor):
    monkeypatch.setattr(capture.sys, 'platform', system)
    monkeypatch.setenv('MOJOLEARN_NUMERIC_MODE', 'identical')
    capture.validate_platform_vendor(vendor)
    monkeypatch.setenv('MOJOLEARN_NUMERIC_MODE', 'fast')
    with pytest.raises(ValueError, match='IDENTICAL'):
        capture.validate_platform_vendor(vendor)


@pytest.mark.parametrize('system,vendor', [('linux', 'metal'), ('darwin', 'cuda'),
                                         ('darwin', 'hip'), ('win32', 'metal')])
def test_mismatched_platform_vendor_refuses(monkeypatch, system, vendor):
    monkeypatch.setattr(capture.sys, 'platform', system)
    monkeypatch.setenv('MOJOLEARN_NUMERIC_MODE', 'identical')
    with pytest.raises(ValueError):
        capture.validate_platform_vendor(vendor)


@pytest.mark.parametrize('resident', [False, True])
def test_transfer_decodes_and_retains_same_bytes_after_source_replacement(tmp_path, resident):
    source = tmp_path / 'incoming.json'
    target = tmp_path / 'retained.json'
    raw = b'{"capture":"fixed"}'
    source.write_bytes(raw)
    result = object()
    class Loader:
        @classmethod
        def from_checkpoint_bytes(cls, encoded, **kwargs):
            assert kwargs == {'resident': resident}
            assert type(encoded) is bytes and encoded == raw
            source.write_bytes(b'changed after capture')
            return result
    restored, receipt = capture.load_foreign_checkpoint(Loader, source, target, resident=resident)
    assert restored is result and target.read_bytes() == raw
    assert receipt['sha256'] == capture.sha(raw)
    assert receipt['loaded_from_immutable_bytes'] is True
    assert receipt['loaded_from_sealed_capture'] is False
    assert receipt['capture_method'] == 'bounded-read-once-immutable-bytes.v1'


def test_transfer_refuses_symlink_and_oversize_before_decode(tmp_path):
    class Loader:
        @classmethod
        def from_checkpoint_bytes(cls, encoded, **kwargs):
            raise AssertionError('invalid incoming file reached decoder')
    source = tmp_path / 'source'
    source.write_bytes(b'x')
    alias = tmp_path / 'alias'
    alias.symlink_to(source)
    with pytest.raises(OSError):
        capture.load_foreign_checkpoint(Loader, alias, tmp_path / 'out1')
    with source.open('wb') as stream:
        stream.truncate(2 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match='bounded regular file'):
        capture.load_foreign_checkpoint(Loader, source, tmp_path / 'out2')
