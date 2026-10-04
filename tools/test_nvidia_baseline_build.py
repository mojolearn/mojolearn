"""Planning must reject unpushed sources and never rent implicitly."""
import subprocess
from types import SimpleNamespace

import pytest

import nvidia_baseline_build as build


@pytest.fixture
def local_git(monkeypatch):
    calls = []

    def git(*args):
        calls.append(args)
        if args[0] == 'rev-parse':
            return 'a' * 40
        if args[0] == 'remote':
            return 'https://example.test/team/project.git'
        if args[0] == 'ls-remote':
            return 'a' * 40 + '\trefs/heads/finished-baseline'
        return ''

    monkeypatch.setattr(build, 'git', git)
    return calls


def test_plan_requires_remote_frozen_tip_and_preserves_bounds(tmp_path, local_git):
    plan = build.plan('a' * 40, tmp_path / 'new')
    assert plan['source_commit'] == 'a' * 40
    assert plan['build_seconds'] == 6000 and plan['lease_minutes'] == 120
    assert plan['advertised_refs'] == ['refs/heads/finished-baseline']
    assert plan['code_format'] == 'ptx-baseline' and not plan['release_qualified']
    assert any(call[0] == 'ls-remote' for call in local_git)
    assert not any(call[0] == 'worktree' for call in local_git)
    assert '--rent' not in build.body_command(plan)


def test_unpushed_commit_refuses_before_runner(tmp_path, local_git, monkeypatch):
    original = build.git
    monkeypatch.setattr(build, 'git', lambda *args: '' if args[0] == 'ls-remote' else original(*args))
    with pytest.raises(ValueError, match='push the finished branch'):
        build.plan('a' * 40, tmp_path / 'new')
    assert not any(call[0] == 'worktree' for call in local_git)


@pytest.mark.parametrize('options', [dict(lease=10), dict(lease=181), dict(seconds=6001),
    dict(seconds=119), dict(lease=60, seconds=3000), dict(vcpu=1), dict(jobs='17'), dict(jobs=';echo bad')])
def test_invalid_or_unbounded_jobs_refuse_before_network(tmp_path, local_git, options):
    with pytest.raises(ValueError):
        build.plan('a' * 40, tmp_path / 'new', **options)
    assert not local_git


@pytest.mark.parametrize('url', ['https://token@example.test/team/repo.git',
    'https://user:secret@example.test/team/repo.git', 'https://example.test/repo?secret=x',
    'https://example.test/repo#secret', '/local/repo', 'file:///local/repo',
    'https://example.test/team/../repo', 'https://example.test/team/repo;echo'])
def test_credentials_and_unsafe_origins_are_not_transported(url):
    with pytest.raises(ValueError, match='credential-free'):
        build.transport_origin(url)


def test_ssh_origin_is_translated_without_copying_credentials():
    assert build.transport_origin('git@example.test:team/repo.git') == 'https://example.test/team/repo.git'


def test_default_cli_is_plan_only(tmp_path, local_git, monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['baseline-build', 'a' * 40, '--out', str(tmp_path / 'new')])
    monkeypatch.setattr(build, 'run', lambda spec: pytest.fail('dry run attempted rental'))
    assert build.main() == 0
    assert 'DRY RUN: no pod created' in capsys.readouterr().out


def test_explicit_run_uses_guarded_runner_and_cleans_worktree(tmp_path, local_git, monkeypatch):
    spec = build.plan('a' * 40, tmp_path / 'new')
    commands = []

    def run(args, **kwargs):
        commands.append(args)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(build.subprocess, 'run', run)
    assert build.run(spec) == 7
    command = commands[0]
    assert command[-1] == '--rent' and 'runpod_cpu_leg.sh' in command[1]
    assert command[command.index('--lease') + 1] == '120'
    assert command[command.index('--envs') + 1] == 'default,pkg'
    assert command[command.index('--image') + 1] == build.IMAGE
    assert any(call[:3] == ('worktree', 'remove', '--force') for call in local_git)


def test_remote_body_parses_and_keeps_experimental_build_separate():
    path = build.ROOT / build.BODY
    subprocess.run(['bash', '-n', str(path)], check=True)
    text = path.read_text()
    assert 'tools/cpu_build_guard.py --seconds "$seconds"' in text
    assert 'MOJOLEARN_CUDA_CODE_FORMAT=ptx-baseline' in text
    assert 'git fetch --filter=blob:none --depth=1 origin "$commit"' in text
    assert 'git diff --quiet HEAD --' in text
    assert 'release061_remote_build.sh' not in text
    assert 'experimental-build.json' in text
