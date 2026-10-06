"""Complete cross-check selection and failure propagation without GPU work."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mojolearn import _backend, _forest_host
from mojolearn import _verify_all as va


def cells(lanes, fixtures):
    return [dict(lane=lane, fixture=fixture, part=part, gpu='a' * 16,
                 cpu='a' * 16, agree=True, na=False, seconds=0)
            for lane in lanes for fixture in fixtures for part in ('infer', 'batch')]


@pytest.mark.parametrize('fault', ['none', 'missing', 'crash', 'mismatch', 'nonzero', 'spawn'])
def test_all_batches_cover_every_lane_and_fixture_and_propagate_failures(monkeypatch, fault):
    lanes = [f'lane-{i}' for i in range(53)]
    fixtures = ['base', 'ties', 'wide']
    calls = []
    monkeypatch.setattr(_backend, 'vendor', lambda: 'metal')

    def run(command, **kwargs):
        assert command[command.index('--cross-check') + 1] == 'default'  # no recursion
        assert command[command.index('--cpu-threads') + 1] == '1'
        selected = command[command.index('--lanes') + 1].split(',')
        fixture = command[command.index('--fixtures') + 1]
        calls.append((selected, fixture))
        rows = cells(selected, [fixture])
        rc = 0
        if len(calls) == 2:
            if fault == 'missing':
                rows.pop()  # a partial child report must not pass
            elif fault == 'crash':
                return SimpleNamespace(returncode=-9, stdout='')
            elif fault == 'spawn':
                raise OSError('cannot start child')
            elif fault == 'nonzero':
                rc = 4
            elif fault == 'mismatch':
                rows[0].update(cpu='b' * 16, agree=False)
                rc = 1
        doc = va._cross_check_result('metal', selected, [fixture], rows, {})
        doc['format'] = 'mojolearn.verify-cross-check.v1'
        return SimpleNamespace(returncode=rc, stdout=json.dumps(doc))

    monkeypatch.setattr(va.subprocess, 'run', run)
    result = va._cross_check_batched(lanes, fixtures, 1, lambda _: None)
    assert len(calls) == 9  # failure must not discard the remaining batches
    assert all(len(selected) <= va.CROSS_CHECK_BATCH_LANES for selected, _ in calls)
    assert [(lane, fx) for selected, fx in calls for lane in selected] == [
        (lane, fx) for fx in fixtures for lane in lanes]
    assert result['passed'] is (fault == 'none')
    assert result['requested_lanes'] == lanes
    assert result['requested_fixtures'] == fixtures
    if fault == 'mismatch':
        assert result['differ'] == 1
    elif fault != 'none':
        if fault != 'nonzero':
            assert result['missing']
        assert 'INCOMPLETE' in va.format_cross_check(result)


@pytest.mark.parametrize('scope,requested,expected', [
    ('all', '', ['base', 'wide']),
    ('default', '', ['base']),
    ('quick', '', ['base']),
    ('all', 'wide', ['wide']),
])
def test_cli_tiers_choose_all_fixtures_only_for_all(monkeypatch, capsys, scope, requested, expected):
    harness = SimpleNamespace(FIXTURES=['base', 'wide'])
    monkeypatch.setattr(va, 'load_harness', lambda: harness)
    monkeypatch.setattr(va, 'cross_check_lanes', lambda h, s: (['ols'], ['ols'], {}))
    monkeypatch.setattr(_backend, 'vendor', lambda: 'metal')

    def result(lanes, fixtures):
        assert fixtures == expected
        return va._cross_check_result('metal', lanes, fixtures, cells(lanes, fixtures), {})

    monkeypatch.setattr(va, '_cross_check_batched', lambda lanes, fixtures, threads, log: result(lanes, fixtures))
    monkeypatch.setattr(va, 'cross_check', lambda h, ml, lanes, fixtures, log: result(lanes, fixtures))
    args = SimpleNamespace(json=True, cross_check=scope, lanes='', fixtures=requested, cpu_threads=1)
    assert va._cmd_cross_check(args, None) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['scope'] == ('custom' if requested else scope)
    assert report['available_fixtures'] == 2


def test_missing_fixture_cannot_hide_behind_another_fixture_passing(monkeypatch, capsys):
    monkeypatch.setattr(va, 'load_harness', lambda: SimpleNamespace(FIXTURES=['base', 'wide']))
    monkeypatch.setattr(va, 'cross_check_lanes', lambda h, s: (['ols'], ['ols'], {}))
    monkeypatch.setattr(_backend, 'vendor', lambda: 'metal')
    result = va._cross_check_result('metal', ['ols'], ['base', 'wide'], cells(['ols'], ['base']), {})
    monkeypatch.setattr(va, '_cross_check_batched', lambda *a: result)
    args = SimpleNamespace(json=True, cross_check='all', lanes='', fixtures='', cpu_threads=1)
    assert va._cmd_cross_check(args, None) == va.EXIT_CANNOT_RUN
    report = json.loads(capsys.readouterr().out)
    assert report['passed'] is False and len(report['missing']) == 2


@pytest.mark.parametrize('batch_error', [None, 'GPU batch failed'])
def test_spectral_batch_uses_identical_fixture_input_on_both_devices(monkeypatch, batch_error):
    affinity = np.arange(12, dtype=np.float32).reshape(3, 4)
    est = SimpleNamespace(_identity_heldout_affinity=affinity, save=lambda p: None)
    host = SimpleNamespace()
    fit = SimpleNamespace(est=est, probe=lambda e: ('answer',))
    seen = []

    def batch(fit, *args):
        if batch_error and fit.est is est:
            return None, batch_error
        seen.append(fit.est)
        assert fit.est._identity_heldout_affinity is affinity
        return 'a' * 16, None

    harness = SimpleNamespace(
        LANES={'spectral-precomputed': lambda *a: fit},
        fixture=lambda fx: (affinity, None, None), heldout=lambda fx: affinity,
        _save_load=lambda est: ('save', None, '.model'), _public_est=lambda lane, est: est,
        _h=lambda *a: 'b' * 16, _probe_batch=batch, BATCH_ALONE=1,
        Fit=lambda *a: SimpleNamespace())
    monkeypatch.setattr(_backend, 'vendor', lambda: 'metal')
    monkeypatch.setattr(_forest_host, 'host_model', lambda p: host)
    result = va.cross_check(harness, None, ['spectral-precomputed'], ['base'])
    if batch_error:
        assert result['passed'] is not True
        assert 'GPU batch failed' in result['skipped']['spectral-precomputed/base']
    else:
        assert seen == [est, host]
        assert result['passed'] is True and result['agree'] == 2
