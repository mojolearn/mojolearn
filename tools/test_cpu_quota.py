"""Prevent diagnostic CPU caps leaking into measurement subprocesses."""
from unittest.mock import mock_open, patch
import os
import subprocess
import sys
import cpu_quota as cq
import bench_board as board


def test_unrestricted_worker_drops_all_diagnostic_caps():
    dirty = {k: '1' for k in cq.INHERITED_CAPS}
    dirty['KEEP_ME'] = 'yes'
    with patch.dict(os.environ, dirty), patch('builtins.open', side_effect=FileNotFoundError):
        env = cq.apply_cpu_quota(dict(os.environ))
    assert not set(cq.INHERITED_CAPS).intersection(env)
    assert env['KEEP_ME'] == 'yes'
    assert env['MOJOLEARN_BENCH_CPU_POLICY'] == 'full-allocation'
    # Verify the actual child receives the sanitized environment.
    subprocess.run([sys.executable, '-c', 'import os; assert "MOJOLEARN_BENCH_THREADS" not in os.environ; assert "OMP_NUM_THREADS" not in os.environ'], env=env, check=True)


def test_real_quota_wins_over_one_core_override():
    with patch.dict(os.environ, {'MOJOLEARN_BENCH_THREADS': '1'}), patch('builtins.open', mock_open(read_data='1360000 100000')), patch.object(cq.os, 'cpu_count', return_value=128):
        env = cq.apply_cpu_quota({'OMP_THREAD_LIMIT': '1', 'MOJOLEARN_CPU_THREADS': '1'})
    assert env['OMP_NUM_THREADS'] == env['MOJOLEARN_CPU_THREADS'] == '13'
    assert 'OMP_THREAD_LIMIT' not in env


def test_unlimited_cgroup_does_not_manufacture_cap():
    with patch('builtins.open', mock_open(read_data='max 100000')):
        assert cq.cpu_quota_threads() is None


def test_board_extra_cannot_restore_cpu_caps():
    with patch('builtins.open', side_effect=FileNotFoundError):
        env = board.child_env({'data_root':'/tmp/data', 'vendor':'apple'}, {'OMP_NUM_THREADS':'1', 'MOJOLEARN_BENCH_THREADS':'1', 'MOJOLEARN_CPU_THREADS':'1'})
    assert not set(cq.INHERITED_CAPS).intersection(env)
    assert env['MOJOLEARN_BENCH_CPU_POLICY'] == 'full-allocation'


def test_env_arm_explicit_value_is_let_through():
    # lane fx-harness (2026-11-18): an A/B env arm's declared key keeps its value over the quota
    with patch('builtins.open', mock_open(read_data='1400000 100000')), patch.object(cq.os, 'cpu_count', return_value=128):
        env = cq.apply_cpu_quota({'MOJOLEARN_CPU_THREADS': '7', 'OMP_NUM_THREADS': '1',
                                  cq.ARM_ENV_KEYS: 'MOJOLEARN_CPU_THREADS'})
    assert env['MOJOLEARN_CPU_THREADS'] == '7'
    assert env['OMP_NUM_THREADS'] == '14'          # undeclared caps still follow the quota
    assert env['MOJOLEARN_BENCH_ARM_ENV_KEPT'] == 'MOJOLEARN_CPU_THREADS=7'


def test_without_declaration_default_is_unchanged():
    with patch('builtins.open', mock_open(read_data='1400000 100000')), patch.object(cq.os, 'cpu_count', return_value=128):
        env = cq.apply_cpu_quota({'MOJOLEARN_CPU_THREADS': '7'})
    assert env['MOJOLEARN_CPU_THREADS'] == '14'
    assert 'MOJOLEARN_BENCH_ARM_ENV_KEPT' not in env
