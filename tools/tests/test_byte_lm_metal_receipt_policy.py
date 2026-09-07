"""Authored file-policy tests. Root execution only; no GPU or subprocess calls."""
import importlib.util
from pathlib import Path

import pytest

_tools = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('receipt_policy', _tools / 'root_job_receipt.py')
policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policy)


def terminal():
    gib = 2**30
    memory = dict(total_bytes=16 * gib, conservative_reserve_bytes=5 * gib,
                  pressure_level=1, swap_used_bytes=0, compressed_bytes=gib)
    return dict(guard='macos-root-serial-v1', reason=None, error=None, returncode=0,
        thread_limit=2, cpu_affinity=None, cpu_enforcement='sampled, not hard affinity',
        gpu_memory_accounting='system unified-memory reserve/pressure; no per-process Metal VRAM counter',
        watchdog_errors=[], watchdog_expired=False,
        cleanup=dict(verified=True, quarantined=False, errors=[]), initial_memory=memory,
        samples=[dict(memory, elapsed_seconds=1., rss_bytes=1024, sampled_cpu_cores=1.)],
        policy_limits=dict(deadline_seconds=60, rss_bytes=2 * gib,
            entry_reserve_bytes=4 * gib, runtime_reserve_bytes=2 * gib,
            normal_pressure_level=1, swap_growth_bytes=128 * 2**20,
            compressed_growth_bytes=256 * 2**20, sampled_cpu_limit=3,
            cpu_grace_seconds=4, sample_interval_seconds=2.0))


def test_truthful_metal_capture_admission():
    policy.validate_guard_terminal(terminal(), 'metal', 0)


def user_tiny_terminal():
    record = terminal()
    record['guard'] = 'macos-root-user-tiny-v1'
    record['policy_limits'].update(entry_reserve_bytes=0, runtime_reserve_bytes=0)
    record['initial_memory']['conservative_reserve_bytes'] = 100 * 2**20
    record['samples'][0]['conservative_reserve_bytes'] = 50 * 2**20
    return record


def test_explicit_user_policy_has_no_free_reserve_minimum():
    policy.validate_guard_terminal(user_tiny_terminal(), 'metal', 0)


@pytest.mark.parametrize('change', ('rss', 'pressure', 'swap', 'legacy-label'))
def test_user_policy_preserves_other_limits_and_distinct_provenance(change):
    record = user_tiny_terminal()
    if change == 'rss':
        record['policy_limits']['rss_bytes'] = 3 * 2**30
    elif change == 'pressure':
        record['samples'][0]['pressure_level'] = 2
    elif change == 'swap':
        record['samples'][0]['swap_used_bytes'] = 129 * 2**20
    else:
        record['guard'] = 'macos-root-serial-v1'
    with pytest.raises(ValueError):
        policy.validate_guard_terminal(record, 'metal', 0)


@pytest.mark.parametrize('key,value', [('cpu_affinity', '0,1'), ('error', 'failed'),
    ('watchdog_errors', ['monitor failed']), ('watchdog_expired', True),
    ('samples', []), ('policy_limits', {}), ('returncode', False)])
def test_metal_missing_or_false_safety_witness_refuses(key, value):
    value_terminal = terminal()
    value_terminal[key] = value
    with pytest.raises(ValueError):
        policy.validate_guard_terminal(value_terminal, 'metal', 0)


@pytest.mark.parametrize('key,value', [('verified', False), ('quarantined', True), ('errors', ['failed'])])
def test_metal_cleanup_must_be_complete(key, value):
    value_terminal = terminal()
    value_terminal['cleanup'][key] = value
    with pytest.raises(ValueError, match='cleanup'):
        policy.validate_guard_terminal(value_terminal, 'metal', 0)


def test_metal_reserve_and_sample_limits_cannot_be_loosened():
    for mutate in (
        lambda t: t['policy_limits'].__setitem__('entry_reserve_bytes', 2**30),
        lambda t: t['initial_memory'].__setitem__('conservative_reserve_bytes', 2**30),
        lambda t: t['samples'][0].__setitem__('rss_bytes', 3 * 2**30),
        lambda t: t['samples'][0].__setitem__('pressure_level', 2),
        lambda t: t['samples'][0].__setitem__('sampled_cpu_cores', float('nan')),
    ):
        value_terminal = terminal()
        mutate(value_terminal)
        with pytest.raises(ValueError):
            policy.validate_guard_terminal(value_terminal, 'metal', 0)


def test_metal_sustained_cpu_violation_refuses():
    value_terminal = terminal()
    first = dict(value_terminal['samples'][0], sampled_cpu_cores=4.)
    value_terminal['samples'] = [first, dict(first, elapsed_seconds=5.)]
    with pytest.raises(ValueError, match='CPU'):
        policy.validate_guard_terminal(value_terminal, 'metal', 0)


def test_metal_oracle_is_not_admitted():
    with pytest.raises(ValueError, match='oracle'):
        policy.validate_guard_terminal(terminal(), 'metal', 0, 'oracle')


def test_linux_terminal_remains_accepted_with_additive_telemetry():
    value_terminal = dict(guard='amd-root-serial-v1', reason=None, returncode=0,
                         thread_limit=2, cpu_affinity='0,1', telemetry={'samples': 1})
    policy.validate_guard_terminal(value_terminal, 'hip', 0)
