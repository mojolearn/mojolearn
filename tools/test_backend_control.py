# SPDX-License-Identifier: Apache-2.0
"""Backend selection and parallel jobs must fail closed without real GPUs."""
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
import identity_iterate
import verify_lanes


@pytest.mark.parametrize('backend', ['cpu', 'metal', 'cuda', 'hip'])
def test_iteration_requires_selected_backend(backend, tmp_path):
    command = identity_iterate.command('python', 'ridge', 'base', tmp_path / 'r.json',
                                       60, backend, False, 'core', deadline=100)
    assert command[command.index('--require-backend') + 1] == backend
    assert '--no-batch' in command and '--no-rlpair' in command
    assert '--deadline' in command


@pytest.mark.parametrize('backend', ['metal', 'cuda', 'hip'])
def test_gpu_parallelism_on_same_host_refuses(backend):
    with pytest.raises(SystemExit):
        verify_lanes.main(['--lane', 'ridge', '--backend', backend, '--jobs', '2', '--plan'])


def test_pod_runner_is_plan_only(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(verify_lanes.lane_select, 'shard', lambda *a: ([['ridge']], [1]))
    monkeypatch.setattr(verify_lanes, '_run_local', lambda *a: pytest.fail('pod plan ran locally'))
    assert verify_lanes.main(['--lane', 'ridge', '--runner', 'pods', '--out', str(tmp_path / 'out')]) == 0
    output = capsys.readouterr().out
    assert '--budget' in output and '--backend cpu' in output
    assert not (tmp_path / 'out').exists()


@pytest.fixture
def local(monkeypatch, tmp_path):
    monkeypatch.setenv('MOJOLEARN_MAC_SLOT_BASE', str(tmp_path / 'slots'))
    monkeypatch.setenv('MOJOLEARN_METAL_LOCK', str(tmp_path / 'gpu'))
    monkeypatch.setenv('MOJOLEARN_METAL_QUEUE', str(tmp_path / 'queue'))
    monkeypatch.setenv('MAC_SLOTS', '2')
    # Scheduling mechanics, not this machine's load (tools/mac_slot.py room).
    monkeypatch.setenv('MOJOLEARN_SLOT_CAPACITY', '0')
    monkeypatch.setattr(identity_iterate, 'cpu_package', lambda *a: (tmp_path, tmp_path))
    started = time.monotonic()
    args = SimpleNamespace(backend='cpu', host_dir=None, jobs=2, started=started,
                           deadline=started+5, budget=5, timeout=3, wait_timeout=2)
    return args, tmp_path


def test_cpu_jobs_actually_overlap_and_report_completion(local, monkeypatch):
    args, out = local
    code = """import pathlib,sys,time
root=pathlib.Path(sys.argv[1]); mine=sys.argv[2]; other=sys.argv[3]
(root/mine).touch()
end=time.monotonic()+2
while not (root/other).exists():
    if time.monotonic()>end: raise SystemExit(8)
    time.sleep(.01)
"""
    monkeypatch.setattr(verify_lanes, '_identity_break_cmd',
        lambda group, *a: [sys.executable, '-c', code, str(out), group[0], 'b' if group[0]=='a' else 'a'])
    _, codes, _ = verify_lanes._run_local([['a'], ['b']], [1,1], args, str(out))
    assert codes == {0: 0, 1: 0}
    result = json.loads((out / 'run-summary.json').read_text())
    assert result['execution_complete'] and not result['complete']
    assert not result['pending'] and not result['running']


def test_deadline_leaves_pending_shards_and_releases_slots(local, monkeypatch):
    args, out = local
    args.jobs = 1
    args.deadline = time.monotonic() + .2
    monkeypatch.setattr(verify_lanes, '_identity_break_cmd',
                        lambda *a: [sys.executable, '-c', 'import time; time.sleep(30)'])
    _, codes, elapsed = verify_lanes._run_local([['a'], ['b']], [1,1], args, str(out))
    assert codes[0] != 0 and 1 not in codes and elapsed < 4
    result = json.loads((out / 'run-summary.json').read_text())
    assert not result['complete'] and result['pending'] == [1]
    assert not list(out.glob('slots.[0-9]*'))


@pytest.mark.parametrize('option,value', [('--budget','nan'), ('--timeout','inf'),
                                         ('--wait-timeout','0'), ('--jobs','0'), ('--shards','0')])
def test_invalid_resource_controls_refuse(option, value):
    with pytest.raises(SystemExit):
        verify_lanes.main(['--lane', 'ridge', option, value, '--plan'])


@pytest.mark.parametrize('field,value', [('infer_verdict', 'MOVED'), ('verdict', 'N/A'), ('verdict', 'MOVED')])
def test_failed_probe_cannot_be_hidden_by_stable_training(monkeypatch, tmp_path, field, value):
    cell = dict(verdict='STABLE', infer_verdict='STABLE')
    cell[field] = value
    record = dict(complete=True, cells={'ridge/base': cell})
    part = tmp_path / 'part0.json'
    part.write_text(json.dumps(record))
    (tmp_path / 'run-summary.json').write_text('{}')
    def merge(cmd, **kwargs):
        Path(cmd[-1]).write_text(json.dumps(record))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(verify_lanes.subprocess, 'run', merge)
    assert verify_lanes._verdict(['ridge'], [str(part)], {0: 0}, str(tmp_path), .1) == 1
    assert not (tmp_path / 'column.json').exists()
    assert not json.loads((tmp_path / 'run-summary.json').read_text())['complete']


def test_explicit_unsupported_probe_refuses():
    with pytest.raises(SystemExit):
        verify_lanes.main(['--lane', 'ridge', '--probe-group', 'rlpair', '--plan'])


@pytest.mark.parametrize('record,success', [
    ({'complete': True, 'cells': {'ridge/base': {'verdict': 'STABLE'}, 'ridge/odd': {'verdict': 'STABLE'}}}, True),
    ({'complete': True, 'cells': {'ridge/base': {'verdict': 'STABLE'}}}, False),
    ({'complete': False, 'cells': {'ridge/base': {'verdict': 'STABLE'}, 'ridge/odd': {'verdict': 'STABLE'}}}, False),
    ({'complete': True, 'cells': {'ridge/base': {}, 'ridge/odd': {'verdict': 'STABLE'}}}, False),
    ({'complete': True, 'cells': {'ridge/base': {'verdict': 'STABLE'}, 'ridge/odd': {'verdict': 'STABLE'}, 'ridge/ties': {'verdict': 'STABLE'}}}, False),
    ('broken-json', False),
])
def test_verdict_requires_exact_fixture_coverage(monkeypatch, tmp_path, record, success):
    part = tmp_path / 'part0.json'
    part.write_text('{}')
    (tmp_path / 'run-summary.json').write_text('{}')
    def merge(cmd, **kwargs):
        Path(cmd[-1]).write_text(record if isinstance(record, str) else json.dumps(record))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(verify_lanes.subprocess, 'run', merge)
    result = verify_lanes._verdict(['ridge'], [str(part)], {0: 0}, str(tmp_path), .1, fixtures=['base', 'odd'])
    assert (result == 0) == success
    assert (tmp_path / 'column.json').exists() == success
    assert json.loads((tmp_path / 'run-summary.json').read_text())['complete'] == success


def test_resume_rejects_scope_change_before_touching_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(verify_lanes.lane_select, 'shard', lambda *a: ([['ridge']], [1]))
    monkeypatch.setattr(verify_lanes, '_run_local', lambda *a: pytest.fail('changed scope launched'))
    manifest = dict(commit=verify_lanes._commit(), lanes=['ridge'], shards=[['ridge']],
                    fixtures='base', repeats=2, backend='cpu', probe_group='core')
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'column.json').write_text('previous evidence')
    (tmp_path / 'part0.json').write_text('previous part')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(SystemExit):
        verify_lanes.main(['--lane', 'ridge', '--resume', '--fixtures', 'odd', '--out', str(tmp_path)])
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_resume_can_change_budget_and_concurrency(tmp_path):
    manifest = dict(commit='same', lanes=['ridge'], shards=[['ridge']], fixtures='base',
                    repeats=2, backend='cpu', probe_group='core', budget=1, jobs=1)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    verify_lanes.validate_resume(tmp_path, dict(manifest, budget=300, jobs=2))
    (tmp_path / 'manifest.json').unlink()
    (tmp_path / 'part0.json').write_text('{}')
    with pytest.raises(ValueError, match='without their manifest'):
        verify_lanes.validate_resume(tmp_path, manifest)
