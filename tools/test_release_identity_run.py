"""Guarded launch tests use fake producer processes, never Mojo/product fits."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import release_identity_run as G
import release_identity_evidence as E
import test_release_identity_evidence as fixtures


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(G, 'source_preflight', lambda root, source: source)
    monkeypatch.setattr(G, 'applicability', lambda root, vendor, source: {})
    monkeypatch.setattr(E, 'declarations', lambda revision: (frozenset(), ()))
    ledger = fixtures.ledger_with(tmp_path / 'ledger')
    root = tmp_path / 'tree'
    root.mkdir()
    return ledger, root, tmp_path / 'evidence'


def test_plan_does_not_begin_or_execute(setup):
    ledger, _, _ = setup
    before = ledger.read()
    result = G.plan(ledger, 'amd')
    assert result['counts'] == {'run': 1}
    assert result['selected']['identity-amd/ols']['reasons']
    assert ledger.read() == before


def test_active_selection_refuses_whole_launch(setup):
    ledger, root, out = setup
    ledger.begin('identity-amd/ols')
    with pytest.raises(ValueError, match='active'):
        G.launch(ledger, root, 'amd-mi325x-gfx942', out, environment={},
                 runner=lambda *a, **k: pytest.fail('producer must not execute'))
    assert len(ledger.read()['runs']) == 1


def test_success_is_pending_admission_and_claim_cannot_execute_twice(setup):
    ledger, root, out = setup
    seen = []
    def runner(command, **kwargs):
        env = kwargs['env']
        path = env['MOJOLEARN_IDENTITY_LAUNCH']
        record = G.validate_launch(path, root, command[-2], out, environment=env, consume=True)
        assert record['lanes'] == ['ols']
        assert record['execution']['all_parts'] == list(E.ALL_PARTS)
        with pytest.raises(ValueError, match='already started'):
            G.validate_launch(path, root, command[-2], out, environment=env, consume=True)
        seen.append(command[-2])
        return SimpleNamespace(returncode=0)
    result = G.launch(ledger, root, 'amd-mi325x-gfx942', out, environment={}, runner=runner)
    assert seen == ['amd-mi325x-gfx942']
    assert result['status'] == 'columns-recorded-awaiting-cross-vendor-admission'
    assert ledger.plan()['identity-amd/ols']['action'] == 'resume'
    assert not ledger.read()['attempts']


def test_nvidia_native_and_ptx_share_one_begin(setup):
    ledger, root, out = setup
    seen = []
    def runner(command, **kwargs):
        seen.append(command[-2])
        G.validate_launch(kwargs['env']['MOJOLEARN_IDENTITY_LAUNCH'], root, command[-2], out,
                          environment=kwargs['env'], consume=True)
        return SimpleNamespace(returncode=0)
    result = G.launch(ledger, root, 'nvidia-l40s-sm89', out, environment={}, runner=runner)
    assert seen == ['nvidia-l40s-sm89', 'nvidia-ptx-l40s-sm80']
    assert result['launched'] == 1
    assert len(ledger.read()['runs']) == 1


@pytest.mark.parametrize('environment', [{'FIXTURES': 'base'}, {'REPEATS': '2'},
    {'PARTIAL_COLUMN': 'train'}, {'MOJOLEARN_IDENTITY_N': '32'}, {'ALLOW_DIRTY': '1'},
    {'MOJOLEARN_X_SABOTAGE': '1'}, {'ONLY_CHANGED': 'table'}, {'SHARD': '0/2'}])
def test_partial_or_diagnostic_environment_refused_before_begin(setup, environment):
    ledger, root, out = setup
    with pytest.raises(ValueError):
        G.launch(ledger, root, 'amd-mi325x-gfx942', out, environment=environment,
                 runner=lambda *a, **k: pytest.fail('producer must not execute'))
    assert not ledger.read()['runs']


def test_failure_and_unstarted_are_retained_without_success(setup):
    ledger, root, out = setup
    result = G.launch(ledger, root, 'amd-mi325x-gfx942', out, environment={},
                      runner=lambda *a, **k: SimpleNamespace(returncode=7))
    outcome = json.loads(Path(result['outcome']).read_text())
    assert outcome['status'] == 'producer-failed-awaiting-admission'
    assert outcome['columns'][0]['exit_code'] == 7
    assert not ledger.read()['attempts']
    def unavailable(*a, **k):
        raise FileNotFoundError('producer unavailable')
    with pytest.raises(FileNotFoundError):
        G.launch(ledger, root, 'apple-m4', out, environment={}, runner=unavailable)
    saved = [json.loads(p.read_text()) for p in out.glob('identity-execution-apple-*.json')]
    assert len(saved) == 1 and not saved[0]['columns']
    assert saved[0]['status'] == 'unstarted'
    assert saved[0]['error'] == 'producer unavailable'
    attempts = ledger.read()['attempts']
    assert len(attempts) == 1 and attempts[0]['status'] == 'failed'
    assert attempts[0]['coverage'] == dict(expected=0, completed=0, failed=0, skipped=0)
    assert ledger.plan()['identity-apple/ols']['action'] == 'repair'


def test_lock_refuses_concurrent_controller(setup):
    ledger, root, out = setup
    with G.exclusive(out / 'identity-producer-amd.lock'):
        with pytest.raises(ValueError, match='another identity producer'):
            G.launch(ledger, root, 'amd-mi325x-gfx942', out, environment={})
    assert not ledger.read()['runs']


def test_completed_pass_reused_without_new_launch(setup, monkeypatch):
    ledger, root, out = setup
    launches = fixtures.begin_all(ledger, out)
    E.admit(ledger, [fixtures.columns(out)], launches, out / 'admit', check=fixtures.fake_check)
    before = len(ledger.read()['runs'])
    result = G.launch(ledger, root, 'amd-mi325x-gfx942', out, environment={},
                      runner=lambda *a, **k: pytest.fail('retained PASS must not rerun'))
    assert result['status'] == 'reused' and result['launched'] == 0
    assert len(ledger.read()['runs']) == before


def test_supplied_release_claim_refuses_before_import(tmp_path):
    root = tmp_path / 'tree'
    (root / 'tools').mkdir(parents=True)
    script = root / 'tools/record_identity_column.sh'
    script.write_text((Path(__file__).parent / script.name).read_text())
    result = subprocess.run(['bash', str(script), 'amd-mi325x-gfx942', str(tmp_path / 'out')],
                            env={'PATH': os.environ['PATH'], 'PYTHON': sys.executable,
                                 'MOJOLEARN_IDENTITY_LAUNCH': str(tmp_path / 'missing-launch.json'),
                                 'MOJOLEARN_IDENTITY_GUARD': G.__file__}, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'release identity launch validation failed' in result.stderr
    assert 'ModuleNotFoundError' not in result.stderr


def test_legacy_controller_keeps_existing_source_preflight(tmp_path):
    root = tmp_path / 'tree'
    (root / 'tools').mkdir(parents=True)
    script = root / 'tools/record_identity_column.sh'
    script.write_text((Path(__file__).parent / script.name).read_text())
    result = subprocess.run(['bash', str(script), 'amd-mi325x-gfx942', str(tmp_path / 'out')],
                            env={'PATH': os.environ['PATH'], 'PYTHON': sys.executable}, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'not a git checkout' in result.stderr
    assert 'guarded launch is required' not in result.stderr


def test_mismatched_source_refuses_before_begin(setup, monkeypatch):
    ledger, root, out = setup
    monkeypatch.setattr(G, 'source_preflight', lambda *a: (_ for _ in ()).throw(ValueError('wrong source')))
    with pytest.raises(ValueError, match='wrong source'):
        G.launch(ledger, root, 'amd-mi325x-gfx942', out, environment={})
    assert not ledger.read()['runs']


def test_default_scope_uses_existing_applicability_reasons(setup, monkeypatch):
    ledger, root, out = setup
    doc = ledger.read()
    extra = fixtures.unit('amd', 'tokenizer')
    doc['requirements']['identity-amd/tokenizer'] = extra
    ledger.sync(doc)
    monkeypatch.setattr(G, 'applicability', lambda root, vendor, source: {'tokenizer': 'host-only arithmetic; no GPU proposition'})
    result = G.plan(ledger, 'amd')
    assert list(result['selected']) == ['identity-amd/ols']
    assert result['excluded']['identity-amd/tokenizer'] == 'host-only arithmetic; no GPU proposition'
    assert not ledger.read()['runs']


def test_missing_provenance_is_blocked_not_silently_omitted(setup):
    ledger, root, out = setup
    doc = ledger.read()
    doc['requirements']['identity-amd/ols']['missing_inputs'] = ['compiler observation']
    ledger.sync(doc)
    result = G.plan(ledger, 'amd')
    assert result['blocked'] == ['identity-amd/ols']
    assert result['counts'] == {'complete-provenance': 1}
    with pytest.raises(ValueError, match='restored provenance'):
        G.launch(ledger, root, 'amd-mi325x-gfx942', out, environment={})
    assert not ledger.read()['runs']
