"""Installed CLI routing and evidence preservation, without native execution."""
import json
import pytest
from mojolearn import __main__ as cli, _verify_causal_lm as proof


def test_capture_routes_formats_and_preserves_output(tmp_path, monkeypatch):
    calls = []
    def capture(device, formats):
        calls.append((device, formats))
        return {'status': 'CAPTURED_UNQUALIFIED'}
    monkeypatch.setattr(proof, 'capture', capture)
    path = tmp_path / 'capture.json'
    args = cli.build_parser().parse_args(['verify-causal-lm', '--output', str(path)])
    assert args.func(args) == 0
    assert calls == [('cpu', ('float32', 'bfloat16', 'int8'))]
    before = path.read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        args.func(args)
    assert path.read_bytes() == before
    assert len(calls) == 1


@pytest.mark.parametrize('equal,exit_code', [(True, 0), (False, 1)])
def test_compare_never_runs_model(tmp_path, monkeypatch, equal, exit_code):
    def unexpected(*args):
        raise AssertionError('comparison executed a model')
    monkeypatch.setattr(proof, 'capture', unexpected)
    monkeypatch.setattr(proof, 'compare', lambda a, b: equal)
    path = tmp_path / 'capture.json'
    path.write_text(json.dumps({'status': 'example'}))
    args = cli.build_parser().parse_args(['verify-causal-lm', '--compare', str(path), str(path)])
    assert args.func(args) == exit_code


def test_requires_an_explicit_action():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(['verify-causal-lm'])


def test_distributed_capture_routes_explicit_layer_map(tmp_path, monkeypatch):
    calls = []
    def capture(device, formats, **kwargs):
        calls.append((device, kwargs))
        return {'status': 'CAPTURED_UNQUALIFIED'}
    monkeypatch.setattr(proof, 'capture', capture)
    args = cli.build_parser().parse_args(['verify-causal-lm', '--device', 'gpu',
        '--layer-devices', '1', '0', '--output', str(tmp_path/'distributed.json')])
    assert args.func(args) == 0
    assert calls == [('gpu', {'layer_devices': [1, 0]})]
