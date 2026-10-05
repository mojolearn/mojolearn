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


@pytest.mark.parametrize('git_present,install_fails', [(False, False), (True, False), (False, True)])
def test_bootstrap_missing_git_before_unchanged_frozen_body(tmp_path, git_present, install_fails):
    """Execute the real generated shell with fake package tools; never apt or rent."""
    import os
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    out = tmp_path / 'out'
    out.mkdir()
    (tmp_path / 'tools').mkdir()

    def executable(path, text):
        path.write_text('#!/bin/bash\n' + text)
        path.chmod(0o755)

    executable(bin_dir / 'bash', 'exec /bin/bash "$@"\n')
    executable(bin_dir / 'timeout', '[[ "$1 $2 $3" == "-k 10 180" ]] || exit 90\nshift 3\nexec "$@"\n')
    executable(bin_dir / 'apt-get', '''printf '%s\\n' "$*" >> "$LEG_OUT/apt-calls"
if [[ "$*" == *install* ]]; then
  [[ "${INSTALL_FAIL:-0}" == 0 ]] || exit 7
  printf '#!/bin/bash\\nprintf "git version fake\\\\n"\\n' > "$FAKE_BIN/git"
  /bin/chmod +x "$FAKE_BIN/git"
fi
''')
    if git_present:
        executable(bin_dir / 'git', 'echo "git version preinstalled"\n')
    executable(tmp_path / build.BODY, 'printf "%s\\n" "$@" > "$LEG_OUT/body-args"\n')
    spec = dict(source_commit='a' * 40, origin='https://example.test/team/project.git',
                build_seconds=6000, jobs='auto')
    env = dict(os.environ, PATH=str(bin_dir), LEG_OUT=str(out), FAKE_BIN=str(bin_dir),
               INSTALL_FAIL=str(int(install_fails)))
    result = subprocess.run(['/bin/bash', '-c', build.body_command(spec)], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    if install_fails:
        assert result.returncode == 7 and not (out / 'body-args').exists()
        return
    assert result.returncode == 0, result.stderr
    assert (out / 'body-args').read_text().splitlines() == [spec['source_commit'], spec['origin'], '6000', 'auto']
    assert (out / 'ptx-prerequisites-readback.txt').read_text().startswith('git version ')
    if git_present:
        assert not (out / 'apt-calls').exists()
    else:
        calls = (out / 'apt-calls').read_text().splitlines()
        assert calls == ['-o Acquire::Retries=2 update',
                         '-o Acquire::Retries=2 install -y --no-install-recommends --no-upgrade git ca-certificates']
