# SPDX-License-Identifier: Apache-2.0
"""DevicePool sizes its workers' host threads from the CPUs the process may
really use: affinity, the cgroup CPU quota and MOJOLEARN_CPU_THREADS.

The RunPod L4 pod reports 128 CPUs but its cgroup grants 13.6
(cpu.max "1360000 100000"); sizing from the core count oversubscribed the
quota once the pool parent and its worker both ran.
"""
import os

import pytest

from mojolearn import _parallel_pool as pool


def _cgroup(tmp_path, v2=None, v1=None):
    if v2 is not None:
        (tmp_path / "cpu.max").write_text(v2 + "\n")
    if v1 is not None:
        (tmp_path / "cpu").mkdir()
        (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text(f"{v1[0]}\n")
        (tmp_path / "cpu" / "cpu.cfs_period_us").write_text(f"{v1[1]}\n")
    return str(tmp_path)


@pytest.fixture
def many_cpus(monkeypatch):
    monkeypatch.setattr(pool.os, "sched_getaffinity", lambda pid: set(range(128)), raising=False)
    monkeypatch.setattr(pool.os, "cpu_count", lambda: 128)


def test_cgroup_v2_quota_caps_the_budget(tmp_path, many_cpus):
    root = _cgroup(tmp_path, v2="1360000 100000")
    assert pool._cgroup_cpu_quota(root) == pytest.approx(13.6)
    assert pool.effective_cpu_budget({}, root) == 13
    assert pool.worker_cpu_threads(1, {}, root) == 13
    assert pool.worker_cpu_threads(4, {}, root) == 3
    assert pool.worker_cpu_threads(32, {}, root) == 1


def test_unlimited_quota_and_v1_quota(tmp_path, many_cpus):
    unlimited = tmp_path / "a"
    unlimited.mkdir()
    assert pool.effective_cpu_budget({}, _cgroup(unlimited, v2="max 100000")) == 128
    root = tmp_path / "b"
    root.mkdir()
    assert pool.effective_cpu_budget({}, _cgroup(root, v1=(250000, 100000))) == 2
    empty = tmp_path / "c"
    empty.mkdir()
    assert pool._cgroup_cpu_quota(str(empty)) is None
    assert pool.effective_cpu_budget({}, str(empty)) == 128


def test_explicit_thread_ceiling_and_fractional_quota(tmp_path, many_cpus):
    root = _cgroup(tmp_path, v2="1360000 100000")
    assert pool.effective_cpu_budget({"MOJOLEARN_CPU_THREADS": "4"}, root) == 4
    assert pool.effective_cpu_budget({"MOJOLEARN_CPU_THREADS": "64"}, root) == 13
    assert pool.effective_cpu_budget({"MOJOLEARN_CPU_THREADS": "junk"}, root) == 13
    small = tmp_path / "small"
    small.mkdir()
    assert pool.effective_cpu_budget({}, _cgroup(small, v2="50000 100000")) == 1


def test_device_pool_hands_each_worker_its_share(monkeypatch, tmp_path, many_cpus):
    root = _cgroup(tmp_path, v2="1360000 100000")
    monkeypatch.setattr(pool, "worker_cpu_threads",
                        lambda n, environ=None, cgroup_root=root: pool.effective_cpu_budget(
                            {}, cgroup_root) // max(1, n))
    launched = []

    class Fake:
        def __init__(self, args, env, **kwargs):
            launched.append(env)

        def poll(self):
            return 0

    monkeypatch.setattr(pool.subprocess, "Popen", Fake)
    from mojolearn import _backend
    monkeypatch.setattr(_backend, "vendor", lambda: "cpu")
    devices = pool.DevicePool((0, 1))
    devices._start()
    assert [env["MOJOLEARN_CPU_THREADS"] for env in launched] == ["6", "6"]
    one = pool.DevicePool((0,))
    launched.clear()
    one._start()
    assert [env["MOJOLEARN_CPU_THREADS"] for env in launched] == ["13"]
    assert len(one._workers) == 1
    one._start()  # created once: a later map reuses the same worker
    assert len(launched) == 1
