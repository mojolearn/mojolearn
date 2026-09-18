"""No native work: test budgets, import-time isolation and exit propagation."""
from types import SimpleNamespace

import pytest

from mojolearn import _verify_resources as resources


@pytest.mark.parametrize('cpus,requested,want', [
    (1, 5, 1), (2, 5, 1), (4, 5, 3), (6, 5, 5), (16, 5, 5), (16, 1, 1)])
def test_budget_leaves_headroom_and_never_assumes_five_cpus(cpus, requested, want):
    effective, env = resources.budget_environment(requested, {'KEEP': 'value'}, cpus)
    assert effective == want
    assert env['KEEP'] == 'value'
    assert all(env[key] == str(want) for key in resources.THREAD_VARIABLES)
    assert env['OMP_MAX_ACTIVE_LEVELS'] == '1'


@pytest.mark.parametrize('requested', [0, -1])
def test_invalid_thread_budget_is_refused(requested):
    with pytest.raises(ValueError, match='positive integer'):
        resources.budget_environment(requested, {}, 8)


def test_process_affinity_limits_reported_host_capacity(monkeypatch):
    monkeypatch.setattr(resources.os, 'cpu_count', lambda: 128)
    monkeypatch.setattr(resources.os, 'process_cpu_count', lambda: 16, raising=False)
    monkeypatch.setattr(resources.os, 'sched_getaffinity', lambda _: {2, 3}, raising=False)
    assert resources.available_cpus() == 2


def test_unknown_cpu_capacity_falls_back_to_one(monkeypatch):
    monkeypatch.setattr(resources.os, 'cpu_count', lambda: None)
    monkeypatch.setattr(resources.os, 'process_cpu_count', lambda: None, raising=False)
    def unavailable(_):
        raise OSError('unavailable')
    monkeypatch.setattr(resources.os, 'sched_getaffinity', unavailable, raising=False)
    assert resources.available_cpus() == 1


def test_budget_uses_fresh_interpreter_and_propagates_failure(monkeypatch):
    monkeypatch.setattr(resources.os, 'environ', {'OPENBLAS_NUM_THREADS': '64'})
    monkeypatch.setattr(resources, 'available_cpus', lambda: 8)
    calls = []
    def run(command, env):
        calls.append((command, env))
        return SimpleNamespace(returncode=4)
    monkeypatch.setattr(resources.subprocess, 'run', run)
    args = SimpleNamespace(command='verify', cpu_threads=5,
                           argv=['verify', '--inference', '--cpu-threads', '5'])
    assert resources.run_with_budget(args) == 4
    command, env = calls[0]
    assert command == [resources.sys.executable, '-m', 'mojolearn', *args.argv]
    assert env['OPENBLAS_NUM_THREADS'] == env['MOJOLEARN_CPU_THREADS'] == '5'
    monkeypatch.setattr(resources.os, 'environ', env)
    assert resources.run_with_budget(args) is None  # no recursive relaunch
    assert len(calls) == 1


@pytest.mark.parametrize('flag', ['coverage', 'compare', 'commitment'])
def test_read_only_commands_do_not_launch_a_numerical_process(monkeypatch, flag):
    def unexpected(*args, **kwargs):
        raise AssertionError('read-only command relaunched')
    monkeypatch.setattr(resources.subprocess, 'run', unexpected)
    args = SimpleNamespace(command='verify', cpu_threads=1, **{flag: True})
    assert resources.run_with_budget(args) is None
