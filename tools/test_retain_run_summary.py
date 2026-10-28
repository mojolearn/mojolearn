"""Retention must preserve verdicts, originals and reference admission boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

import retain_run_summary as R


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / 'mojolearn'
    (root / '.git').mkdir(parents=True)
    (root / 'tools').mkdir()
    monkeypatch.setenv('MOJOLEARN_SUMMARY_REPO', str(root))
    return root


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')
    return path


def saved(repo, kind, stream):
    return json.loads((repo / 'docs' / kind / 'latest/auto' / (stream + '.json')).read_text())['results']


def board():
    return dict(config=dict(vendor='nvidia', rows='full', smoke=False, token='do-not-copy'),
        box=dict(gpu='fixture-gpu', env={'SECRET': 'do-not-copy'}),
        races={'example': dict(status='failed', rc=1, audit={'identical/fit': {'verdict': 'INSUFFICIENT'}},
            evidence_receipt={'r2_key': 'measurement/receipt', 'sha256': 'a'*64},
            cells=[dict(arm='ours', status='ok', mode='identical', median_ms=4.2,
                times_ms=[4.2], rounds=1, quality={'accuracy': 0.1},
                comparability={'inside_clock': 'fit plus transfer'},
                measurement=dict(source={'commit': 'b'*40, 'sha256': 'c'*64},
                                 dataset={'identity': 'd'*64}, completeness='incomplete',
                                 missing=['artifact'], workers={'large-detail': 'external'}))])})


def test_summary_updates_are_idempotent_and_preserve_originals_and_verdicts(repo, tmp_path):
    source = write(tmp_path / 'evidence/result.json', {'status': 'FAILED', 'pending': ['gpu']})
    before = source.read_bytes()
    path = R.retain_verification('test', [source])
    stamp = path.stat().st_mtime_ns
    assert R.retain_verification('test', [source]) == path
    assert path.stat().st_mtime_ns == stamp
    assert source.read_bytes() == before
    result = saved(repo, 'verification', 'test')
    assert result['reports']['result.json'] == {'status': 'FAILED', 'pending': ['gpu']}
    assert result['originals'][0]['sha256'] == hashlib.sha256(before).hexdigest()
    write(source, {'status': 'running', 'pending': ['gpu', 'cpu']})
    R.retain_verification('test', [source])
    assert saved(repo, 'verification', 'test')['reports']['result.json']['status'] == 'running'


def test_size_failure_does_not_replace_latest_or_remove_original(repo, tmp_path):
    path = R.retain('verification', 'test', {'status': 'failed'})
    previous = path.read_bytes()
    with pytest.raises(ValueError, match='exceeds'):
        R.atomic_json(path, {'large': 'x'*200}, limit=100)
    assert path.read_bytes() == previous
    assert list(path.parent.iterdir()) == [path]
    large = tmp_path / 'large.json'
    large.write_bytes(b' ' * (R.MAX_SUMMARY_BYTES + 1))
    with pytest.raises(ValueError, match='limit'):
        R.retain_verification('test', [large])
    assert large.exists() and path.read_bytes() == previous


def test_retention_failure_is_visible_and_not_a_verification_verdict(repo, capsys):
    assert R.try_retain(R.retain_verification, 'missing', [repo/'missing.json']) is None
    assert 'RETENTION FAILED' in capsys.readouterr().err


def test_measurements_keep_clocks_quality_and_missing_evidence_without_promotion(repo, tmp_path):
    original = board()
    before = copy.deepcopy(original)
    R.retain_board(tmp_path, original, exit_code=1)
    result = saved(repo, 'measurements', 'bench-board-nvidia')
    cell = result['board']['races']['example']['cells'][0]
    assert result['exit_code'] == 1
    assert cell['median_ms'] == 4.2 and cell['quality'] == {'accuracy': 0.1}
    assert cell['comparability']['inside_clock'] == 'fit plus transfer'
    context = result['board']['contexts'][cell['provenance_ref']]
    assert context['source']['commit'] == 'b'*40
    assert cell['measurement']['missing'] == ['artifact']
    assert result['board']['races']['example']['audit']['identical/fit']['verdict'] == 'INSUFFICIENT'
    assert original == before
    assert 'do-not-copy' not in json.dumps(result)
    assert 'large-detail' not in json.dumps(result)


def test_in_progress_smoke_does_not_inherit_old_pass_and_does_not_replace_full_board(repo, tmp_path):
    R.retain_board(tmp_path, board(), exit_code=1)
    write(tmp_path / 'smoke.json', {'passed': 12})
    R.retain_board(tmp_path, board(), smoke=True)
    result = saved(repo, 'measurements', 'bench-board-nvidia-smoke')
    assert 'smoke' not in result and result['execution_state'] == 'in-progress'
    write(tmp_path / 'smoke.json', {'passed': 0, 'failures': ['example']})
    R.retain_board(tmp_path, board(), smoke=True, exit_code=1)
    assert saved(repo, 'measurements', 'bench-board-nvidia-smoke')['smoke']['passed'] == 0
    assert saved(repo, 'measurements', 'bench-board-nvidia')['run_kind'] == 'measurement'


def test_unbounded_parameters_and_failed_nonfinite_scores_are_preserved(repo, tmp_path):
    original = board()
    cell = original['races']['example']['cells'][0]
    cell.update(params={'limit': [0, float('inf')]}, quality={'loss': float('nan')})
    R.retain_board(tmp_path, original, exit_code=1)
    saved_cell = saved(repo, 'measurements', 'bench-board-nvidia')['board']['races']['example']['cells'][0]
    assert saved_cell['params']['limit'] == [0, {'$float': 'Infinity'}]
    assert saved_cell['quality']['loss'] == {'$float': 'NaN'}
    assert original['races']['example']['status'] == 'failed'


def test_campaign_retains_partial_results_and_excludes_controller_secrets(repo, tmp_path):
    out = tmp_path / 'evidence'
    path = write(out / 'nvidia/boards/one.json', board())
    journal = {'status': 'needs-attention', 'vendors': {'nvidia': {'status': 'running', 'completed': 1}}}
    config = dict(repo=str(repo), out=str(out), source_commit='b'*40,
        vendors=[dict(vendor='nvidia'), dict(vendor='amd')],
        r2_credentials='secret-path', provider_env={'TOKEN': 'secret-value'})
    assert len(R.retain_campaign(config, journal)) == 1
    result = saved(repo, 'measurements', 'campaign-nvidia')
    assert result['controller']['status'] == 'needs-attention'
    assert result['vendor']['status'] == 'running'
    assert result['boards']['one.json']['original_sha256'] == R.digest(path)
    assert 'secret' not in json.dumps(result)
    assert not (repo/'docs/measurements/latest/auto/campaign-amd.json').exists()


def test_identity_bytes_are_immutable_reusable_and_do_not_regenerate_references(repo, tmp_path):
    reference = write(repo/'python/mojolearn/verify_reference/table.json', {'unchanged': True})
    column = write(tmp_path/'cpu.identical.json', {'commit': 'b'*40, 'vendor': 'cpu',
                                                'partial_column': True, 'lanes': {}})
    sidecar = tmp_path/'cpu.provenance.txt'
    sidecar.write_text('first receipt\n')
    target = R.retain_identity(column)
    assert target.read_bytes() == column.read_bytes()
    assert target.with_name('cpu.provenance.txt').read_bytes() == sidecar.read_bytes()
    assert R.retain_identity(column) == target
    sidecar.write_text('later receipt\n')
    assert R.retain_identity(column) == target
    assert len(list(target.parent.glob('*.identical.json'))) == 1
    assert len(list(target.parent.glob('*.provenance.txt'))) == 2
    assert saved(repo, 'verification', 'identity-cpu')['metadata']['partial_column']
    assert json.loads(reference.read_text()) == {'unchanged': True}
    target.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='overwrite immutable'):
        R.retain_identity(column)
    assert target.read_bytes() == b'corrupt'


def test_symlink_cannot_redirect_repository_writes(repo, tmp_path):
    (repo/'docs').symlink_to(tmp_path)
    with pytest.raises(ValueError, match='symlinks'):
        R.retain('verification', 'test', {})
    assert not (tmp_path/'verification').exists()


def test_controller_collects_failure_without_provisioning_or_running_a_campaign(repo, tmp_path):
    import run_full_do_measurements as controller
    state = tmp_path/'state'
    state.mkdir()
    (state/'state.env').write_text('POD_ID=another-owner\n')
    config = dict(repo=str(repo), out=str(tmp_path/'out'), source_commit='b'*40,
                  vendors=[dict(vendor='nvidia', state_dir=str(state), expected_id='our-id')])
    with pytest.raises(RuntimeError, match='owned box identity changed'):
        controller.run(config)
    result = saved(repo, 'measurements', 'campaign-nvidia')
    assert result['controller']['status'] == 'needs-attention'
    assert result['boards'] == {}


def test_neural_runner_keeps_failed_result_and_exit_code(repo, tmp_path, monkeypatch):
    import verify_neural_provider as runner
    out = tmp_path/'neural'
    def fake_verify(args):
        write(args.output/'result.json', dict(status='failed', pending=['runtime']))
        return 7
    monkeypatch.setattr(runner, 'verify', fake_verify)
    monkeypatch.setattr(sys, 'argv', ['verify_neural_provider.py', '--output', str(out)])
    assert runner.main() == 7
    result = saved(repo, 'verification', 'neural-provider-source-only')
    assert result['reports']['result.json']['status'] == 'failed'


def test_api_runner_keeps_pending_scope_and_original_exit(repo, tmp_path, monkeypatch):
    import verify_api_endpoints as runner
    module = types.ModuleType('mojolearn._verify_api_runner')
    module.registry = lambda: {'one': ('estimators',)}
    module.TARGETS = [dict(endpoint='one', public='Example')]
    module.capture = lambda *args, **kwargs: dict(endpoints={}, pending=['one'])
    module.differences = lambda *args: []
    monkeypatch.setitem(sys.modules, 'mojolearn', types.ModuleType('mojolearn'))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    out = tmp_path/'api.json'
    monkeypatch.setattr(sys, 'argv', ['verify_api_endpoints.py', '--out', str(out)])
    assert runner.main() == 1
    result = saved(repo, 'verification', 'api-all')
    assert result['metadata']['exit_code'] == 1
    assert result['reports']['api.json']['pending'] == ['one']


def test_light_runner_retains_capture_without_claiming_qualification(repo, tmp_path, monkeypatch):
    import light_verifier as runner
    verifier = types.SimpleNamespace(EXPECTED_CASES=['one'],
        capture=lambda cases: dict(vendor='cpu', status='CAPTURED', cases=cases), compare=lambda *args: [])
    package = types.ModuleType('mojolearn')
    package._verify_light = verifier
    faults = types.ModuleType('light_verifier_faults')
    faults.FAULTS, faults.inject = {}, None
    monkeypatch.setitem(sys.modules, 'mojolearn', package)
    monkeypatch.setitem(sys.modules, 'light_verifier_faults', faults)
    monkeypatch.setattr(sys, 'argv', ['light_verifier.py', '--out', str(tmp_path/'light'), '--vendor', 'cpu'])
    assert runner.main() == 0
    result = saved(repo, 'verification', 'light-cpu')
    assert result['reports']['result.json']['status'] == 'CAPTURED_UNQUALIFIED'


def test_opt_in_campaign_publish_excludes_unrelated_staged_edits(repo, tmp_path):
    from run_full_do_measurements import publish_campaign
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo, stderr=subprocess.STDOUT, text=True).strip()
    git('init', '-b', 'main')
    git('config', 'user.name', 'Retention test')
    git('config', 'user.email', 'retention@example.invalid')
    (repo/'source.py').write_text('original\n')
    board = write(repo/'bench/results/bench_board/main-nvidia/board.json', {'status': 'old'})
    git('add', '.')
    git('commit', '-m', 'fixture')
    remote = tmp_path/'remote.git'
    git('init', '--bare', str(remote))
    git('remote', 'add', 'origin', str(remote))
    (repo/'source.py').write_text('unrelated staged change\n')
    git('add', 'source.py')
    write(board, {'status': 'complete'})
    summary = R.retain('measurements', 'campaign-nvidia', {'status': 'complete'})
    config = dict(repo=str(repo), source_commit='b'*40, vendors=[dict(board_column='nvidia')])
    def command(argv, log):
        subprocess.run(argv, cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    publish_campaign(config, [summary], command)
    assert git('diff', '--cached', '--name-only') == 'source.py'
    assert git('show', 'HEAD:source.py') == 'original'
    assert json.loads(git('show', 'HEAD:' + str(summary.relative_to(repo))))['results']['status'] == 'complete'
    assert git('rev-parse', 'HEAD') == git('--git-dir', str(remote), 'rev-parse', 'main')
    git('switch', '-c', 'development')
    with pytest.raises(RuntimeError, match='requires the main branch'):
        publish_campaign(config, [summary], command)
