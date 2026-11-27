# SPDX-License-Identifier: Apache-2.0
"""python/mojolearn/_allocation.py: the one allocation function verify, the recorder, the GPU smoke, the proving
workers and the job scripts' `alloc_cores` read (rel-fix-3, 2026-11-26). Fake cgroup trees under tmp_path; nothing
reads the real /sys/fs/cgroup."""
import os
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATH = ROOT / "python" / "mojolearn" / "_allocation.py"


def _load():
    import importlib.util
    spec = importlib.util.spec_from_file_location("allocation_under_test", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = _load()


@pytest.fixture
def affinity_128(monkeypatch):
    """The L40S pod: 128 visible CPUs in the affinity set."""
    monkeypatch.setattr(A.os, "sched_getaffinity", lambda pid: set(range(128)), raising=False)
    monkeypatch.delenv(A.ENV_CORES, raising=False)


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_cgroup_v1_quota_caps_the_affinity(tmp_path, affinity_128):
    """The record job of 3ce09e341f: a v1 quota of 14 cores under 128 visible; verify read v2 only and ran 128."""
    write(tmp_path, "cpu/cpu.cfs_quota_us", "1400000\n")
    write(tmp_path, "cpu/cpu.cfs_period_us", "100000\n")
    n, why = A.cores(str(tmp_path), environ={})
    assert n == 14 and "cgroup v1" in why and "affinity 128" in why


def test_cgroup_v1_under_cpu_cpuacct_and_a_partial_core_rounds_up(tmp_path, affinity_128):
    write(tmp_path, "cpu,cpuacct/cpu.cfs_quota_us", "1360000\n")
    write(tmp_path, "cpu,cpuacct/cpu.cfs_period_us", "100000\n")
    assert A.cores(str(tmp_path), environ={})[0] == 14       # 13.6 cores: the whole allocation is used


def test_cgroup_v1_without_a_quota_leaves_the_affinity(tmp_path, affinity_128):
    write(tmp_path, "cpu/cpu.cfs_quota_us", "-1\n")
    write(tmp_path, "cpu/cpu.cfs_period_us", "100000\n")
    n, why = A.cores(str(tmp_path), environ={})
    assert n == 128 and "no quota" in why


def test_cgroup_v2_quota_caps_the_affinity_and_max_does_not(tmp_path, affinity_128):
    write(tmp_path, "cpu.max", "800000 100000\n")
    assert A.cores(str(tmp_path), environ={})[0] == 8
    write(tmp_path, "cpu.max", "max 100000\n")
    assert A.cores(str(tmp_path), environ={})[0] == 128


def test_affinity_below_the_quota_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(A.os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    write(tmp_path, "cpu.max", "1400000 100000\n")
    n, why = A.cores(str(tmp_path), environ={})
    assert n == 3 and why.startswith("affinity 3")


def test_the_jobs_measured_count_overrides_everything(tmp_path, affinity_128):
    write(tmp_path, "cpu.max", "800000 100000\n")
    n, why = A.cores(str(tmp_path), environ={A.ENV_CORES: "14"})
    assert n == 14 and A.ENV_CORES in why
    # a malformed or zero value is not a count: the measurement is used
    assert A.cores(str(tmp_path), environ={A.ENV_CORES: "zero"})[0] == 8
    assert A.cores(str(tmp_path), environ={A.ENV_CORES: "0"})[0] == 8


def test_workers_are_bounded_by_memory_gpu_tasks_and_say_why(tmp_path, affinity_128, monkeypatch):
    write(tmp_path, "cpu/cpu.cfs_quota_us", "1400000\n")
    write(tmp_path, "cpu/cpu.cfs_period_us", "100000\n")
    monkeypatch.setattr(A, "mem_available_bytes", lambda root=None: 1313 * 2 ** 30)
    n, why = A.workers(host_gib_per_worker=4.0, gpu_mb_per_worker=2048, gpu_free=45000, root=str(tmp_path),
                       environ={})
    assert n == 14 and "cores 14" in why and "gpu memory 19" in why and "host memory 328" in why
    n, why = A.workers(host_gib_per_worker=4.0, gpu_mb_per_worker=2048, gpu_free=10000, root=str(tmp_path),
                       environ={})
    assert n == 4 and "gpu memory 4" in why
    write(tmp_path, "pids.max", "8704\n")
    write(tmp_path, "pids.current", "704\n")
    n, why = A.workers(threads_per_worker=200, root=str(tmp_path), environ={})
    assert n == 14
    n, why = A.workers(threads_per_worker=1000, root=str(tmp_path), environ={})
    assert n == 6 and "tasks 6" in why                      # 8000 left x 0.75 // 1000
    assert A.workers(requested=128, root=str(tmp_path), environ={}) == (128, "128 requested")
    assert A.workers(units=3, root=str(tmp_path), environ={})[0] == 3


def test_cgroup_v1_memory_limit_caps_and_its_no_limit_value_does_not(tmp_path):
    write(tmp_path, "memory/memory.limit_in_bytes", str(8 * 2 ** 30))
    write(tmp_path, "memory/memory.usage_in_bytes", str(2 * 2 ** 30))
    got = A.mem_available_bytes(str(tmp_path))
    if got is None:
        pytest.skip("no MemAvailable or available-pages count on this OS")
    assert got <= 6 * 2 ** 30
    write(tmp_path, "memory/memory.limit_in_bytes", "9223372036854771712")
    unlimited = A.mem_available_bytes(str(tmp_path))
    assert unlimited is None or unlimited >= got


def test_a_lane_without_a_cost_takes_its_familys():
    hints = {"gbdt-a": 5.5, "gbdt-b": 1.0, "x-decomp-eigh-wide-pca": 0.9, "x-decomp-ica": 1.2, "pca": 0.4,
             "@public": 300.0}
    assert A.lane_cost(hints, "gbdt-a") == (5.5, "measured")
    assert A.lane_cost(hints, "gbdt-multiclass") == (5.5, "family gbdt-*")
    assert A.lane_cost(hints, "x-decomp-eigh-wide-fastica") == (0.9, "family x-decomp-eigh-wide-*")
    seconds, source = A.lane_cost(hints, "bootstrap")
    assert source == "p90 of all lanes" and seconds == 5.5          # never an '@' unit's seconds
    assert A.lane_cost({}, "x") == (1.0, "no hints")


def test_the_shipped_table_gives_every_reference_lane_a_cost():
    import json
    hints = json.loads((ROOT / "python/mojolearn/verify_reference/cell_seconds.json").read_text())["seconds"]
    lanes = {key.split("/")[0] for key in json.loads(
        (ROOT / "python/mojolearn/verify_reference/table.json").read_text())["cells"]}
    for lane in lanes:
        seconds, source = A.lane_cost(hints, lane)
        assert seconds > 0 and source != "no hints"
    assert A.lane_cost(hints, "gbdt-multiclass")[0] >= max(v for k, v in hints.items() if k.startswith("gbdt-"))


def test_the_shell_alloc_cores_runs_this_function_and_exports_it_once():
    lib = ROOT / "tools" / "ops" / "box_cores_lib.sh"
    env = {k: v for k, v in os.environ.items() if k != A.ENV_CORES}
    out = subprocess.run(["bash", "-c", '. "$1"; echo "$MOJOLEARN_ALLOC_CORES $(alloc_cores)"', "_", str(lib)],
                         capture_output=True, text=True, env=env, timeout=60)
    exported, called = out.stdout.split()
    assert out.returncode == 0 and exported == called == str(A.cores(environ={})[0])
    env[A.ENV_CORES] = "14"
    out = subprocess.run(["bash", "-c", '. "$1"; alloc_cores', "_", str(lib)], capture_output=True, text=True,
                         env=env, timeout=60)
    assert out.stdout.strip() == "14"
    # box_job_lib.sh has no copy of its own
    assert "cfs_quota_us" not in (ROOT / "tools" / "ops" / "box_job_lib.sh").read_text()


def test_busy_cores_read_the_pods_own_counter_v2_then_v1(tmp_path):
    """rel-fix-3: every sampler differences this counter; on cgroup v1 it is cpuacct.usage, never the host's
    /proc/stat (the record's "40.7 of 14" was its host's busy CPUs)."""
    write(tmp_path, "cpuacct/cpuacct.usage", "9800000000\n")             # ns
    assert A.cpu_usage_seconds(str(tmp_path)) == (9.8, "cgroup v1 cpuacct/cpuacct.usage")
    write(tmp_path, "cpu.stat", "usage_usec 25000000\nuser_usec 1\n")
    assert A.cpu_usage_seconds(str(tmp_path)) == (25.0, "cgroup v2 cpu.stat")
    assert A.busy_cores((100.0, 25.0), (110.0, 123.0)) == 9.8
    assert A.busy_cores((100.0, 25.0), (100.0, 30.0)) is None and A.busy_cores(None, (1.0, 2.0)) is None
    assert A.busy_cores((100.0, 30.0), (110.0, 25.0)) is None             # a counter that went backwards


def test_the_shell_samplers_difference_the_shared_counter():
    lib = ROOT / "tools" / "ops" / "box_cores_lib.sh"
    out = subprocess.run(["bash", "-c", '. "$1"; busy_between "100000000 5000000 x" "110000000 103000000 x"; '
                          'busy_between none "1 2 x"; echo end', "_", str(lib)], capture_output=True, text=True,
                         timeout=60)
    assert out.stdout.split() == ["9.8", "end"]
    for job in ("tools/ops/release_gate_job.sh", "tools/runpod_cpu_leg.sh", "tools/bisect_hang_group.sh"):
        assert "/proc/stat)" not in (ROOT / job).read_text() and "< /proc/stat" not in (ROOT / job).read_text()
