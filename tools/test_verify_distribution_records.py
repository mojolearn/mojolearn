"""Split-wheel RECORD qualification without importing native bindings."""
import base64
import importlib.util
from importlib import metadata
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def verifier(monkeypatch):
    root = Path(__file__).resolve().parents[1] / 'python/mojolearn'
    package = ModuleType('record_fixture')
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, 'record_fixture', package)
    spec = importlib.util.spec_from_file_location(
        'record_fixture._verify_distributed', root / '_verify_distributed.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Record:
    def __init__(self, name, sha):
        self.name = name
        self.hash = SimpleNamespace(mode='sha256', value=base64.urlsafe_b64encode(
            bytes.fromhex(sha)).decode().rstrip('='))

    def __str__(self):
        return self.name


@pytest.mark.parametrize('fault', [None, 'changed-payload', 'missing-record',
                                  'wrong-version', 'wrong-location', 'missing-owner',
                                  'changed-core', 'unknown-architecture', 'external-native'])
def test_split_wheel_ownership_and_negative_controls(verifier, monkeypatch, tmp_path, fault):
    package = tmp_path / 'mojolearn'
    relative = 'mojolearn/cuda_ptx/sm_80/identical/native.so'
    core = SimpleNamespace(version='0.8.60', files=[Record('mojolearn/_verify_distributed.py', 'a' * 64)])
    payload = SimpleNamespace(version='0.8.60', files=[Record(relative, 'b' * 64)],
                              locate_file=lambda name: tmp_path / name)
    source = {'_verify_distributed.py': 'a' * 64}
    native = {'binding': dict(path=str(tmp_path / relative), sha256='b' * 64)}
    calls = []

    def distribution(name):
        calls.append(name)
        assert name == 'mojolearn-nvidia-ptx'
        if fault == 'missing-owner':
            raise metadata.PackageNotFoundError(name)
        return payload

    monkeypatch.setattr(metadata, 'distribution', distribution)
    if fault == 'changed-payload': payload.files = [Record(relative, 'c' * 64)]
    if fault == 'missing-record': payload.files = []
    if fault == 'wrong-version': payload.version = '0.8.59'
    if fault == 'wrong-location': payload.locate_file = lambda name: tmp_path / 'other' / name
    if fault == 'changed-core': core.files = [Record('mojolearn/_verify_distributed.py', 'c' * 64)]
    if fault == 'unknown-architecture': native['binding']['path'] = str(tmp_path / relative.replace('sm_80', 'sm_999'))
    if fault == 'external-native': native['binding']['path'] = str(tmp_path / 'external.so')
    if fault:
        with pytest.raises(ValueError):
            verifier.verify_distribution_records(core, package, source, native)
    else:
        verifier.verify_distribution_records(core, package, source, native)
        assert calls == ['mojolearn-nvidia-ptx']


def test_monolithic_wheel_needs_no_plugin_lookup(verifier, monkeypatch, tmp_path):
    package = tmp_path / 'mojolearn'
    relative = 'mojolearn/identical/native.so'
    dist = SimpleNamespace(files=[Record(relative, 'b' * 64)])
    monkeypatch.setattr(metadata, 'distribution', lambda name: pytest.fail('unexpected plugin lookup'))
    verifier.verify_distribution_records(
        dist, package, {}, {'binding': dict(path=str(tmp_path / relative), sha256='b' * 64)})
