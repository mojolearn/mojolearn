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
    with patch.dict(os.environ, {'MOJOLEARN_BENCH_THREADS': '1'}), patch('builtins.open', mock_open(read_data='1360000 100000')), patch.object(cq, '_visible_cores', return_value=128):
        env = cq.apply_cpu_quota({'OMP_THREAD_LIMIT': '1', 'MOJOLEARN_CPU_THREADS': '1'})
    # 13.6 cores round up to 14 (the product's core/host_predict_threads.mojo rule, tools/numa_place.py)
    assert env['OMP_NUM_THREADS'] == env['MOJOLEARN_CPU_THREADS'] == '14'
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
    with patch('builtins.open', mock_open(read_data='1400000 100000')), patch.object(cq, '_visible_cores', return_value=128):
        env = cq.apply_cpu_quota({'MOJOLEARN_CPU_THREADS': '7', 'OMP_NUM_THREADS': '1',
                                  cq.ARM_ENV_KEYS: 'MOJOLEARN_CPU_THREADS'})
    assert env['MOJOLEARN_CPU_THREADS'] == '7'
    assert env['OMP_NUM_THREADS'] == '14'          # undeclared caps still follow the quota
    assert env['MOJOLEARN_BENCH_ARM_ENV_KEPT'] == 'MOJOLEARN_CPU_THREADS=7'


def test_without_declaration_default_is_unchanged():
    with patch('builtins.open', mock_open(read_data='1400000 100000')), patch.object(cq, '_visible_cores', return_value=128):
        env = cq.apply_cpu_quota({'MOJOLEARN_CPU_THREADS': '7'})
    assert env['MOJOLEARN_CPU_THREADS'] == '14'
    assert 'MOJOLEARN_BENCH_ARM_ENV_KEPT' not in env


def test_cgroup_v1_quota_sizes_the_pools(tmp_path, monkeypatch):
    # lane fx-numa-bind-smoke (2026-11-19): the RunPod L40S pod's quota is cgroup v1 (14 of 128 visible cores); only
    # v2's root cpu.max was read, so no pool was sized and every worker's pools followed the visible cores
    np_ = cq._numa()
    root = tmp_path / 'cgroup'
    (root / 'cpu').mkdir(parents=True)
    (root / 'cpu' / 'cpu.cfs_quota_us').write_text('1400000\n')
    (root / 'cpu' / 'cpu.cfs_period_us').write_text('100000\n')
    proc = tmp_path / 'proc_self_cgroup'
    proc.write_text('5:cpu,cpuacct:/\n')
    monkeypatch.setattr(np_, 'CGROUP_ROOT', str(root))
    monkeypatch.setattr(np_, 'PROC_CGROUP', str(proc))
    monkeypatch.setattr(cq, '_visible_cores', lambda: 128)
    env = cq.apply_cpu_quota({'OMP_NUM_THREADS': '128'})
    assert env['OMP_NUM_THREADS'] == env['MOJOLEARN_CPU_THREADS'] == env['OPENBLAS_NUM_THREADS'] == '14'
    monkeypatch.setattr(cq, '_visible_cores', lambda: 14)       # the allocation is every visible core: no cap
    assert cq.cpu_quota_threads() is None
