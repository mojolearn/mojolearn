"""Installed command routing; hardware checks live in the capture runner."""
import pytest
from mojolearn import __main__ as cli, _verify_distributed as proof


def test_capture_routes_installed_requirement(monkeypatch):
    calls = []
    monkeypatch.setattr(proof, 'main', lambda argv: calls.append(argv) or 0)
    args = cli.build_parser().parse_args(['verify-distributed', '--devices', '1,0',
        '--out', 'capture.json', '--require-installed'])
    assert args.cpu_threads == 1
    assert args.func(args) == 0
    assert calls == [['--devices', '1,0', '--out', 'capture.json', '--require-installed']]


def test_compare_routes_without_capture(monkeypatch):
    calls = []
    monkeypatch.setattr(proof, 'main', lambda argv: calls.append(argv) or 1)
    args = cli.build_parser().parse_args(['verify-distributed', '--compare', 'a.json', 'b.json'])
    assert args.func(args) == 1
    assert calls == [['--compare', 'a.json', 'b.json']]


def test_compare_rejects_capture_options():
    args = cli.build_parser().parse_args(['verify-distributed', '--compare', 'a', 'b', '--devices', '0,1'])
    with pytest.raises(ValueError, match='cannot be combined'):
        args.func(args)


def test_cross_validation_routes_installed_backend(monkeypatch):
    from mojolearn import _verify_parallel_cv as cv
    calls = []
    monkeypatch.setattr(cv, 'main', lambda argv: calls.append(argv) or 0)
    args = cli.build_parser().parse_args(['verify-cross-validation', '--devices', '0,1',
        '--out', 'evidence', '--require-backend', 'hip', '--require-installed'])
    assert args.func(args) == 0
    assert calls == [['--devices', '0,1', '--out', 'evidence',
                      '--require-backend', 'hip', '--require-installed']]


def test_cross_validation_compare_is_read_only(monkeypatch):
    from mojolearn import _verify_parallel_cv as cv
    calls = []
    monkeypatch.setattr(cv, 'main', lambda argv: calls.append(argv) or 1)
    args = cli.build_parser().parse_args(['verify-cross-validation', '--compare', 'a', 'b'])
    assert args.func(args) == 1
    assert calls == [['--compare', 'a', 'b']]
