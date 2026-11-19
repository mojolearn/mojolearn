# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/numa_place.py (lane fx-harness, 2026-11-18): worker NUMA placement, its record and per-call host usage,
on fake sysfs trees (no box, no GPU)."""
import importlib.util
import json
import os
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("t_numa_place", os.path.join(HERE, "numa_place.py"))
NP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NP)


def test_cpulist_round_trip():
    assert NP.parse_cpulist("0-3,8,10-11") == {0, 1, 2, 3, 8, 10, 11}
    assert NP.cpulist({0, 1, 2, 3, 8, 10, 11}) == "0-3,8,10-11"
    assert NP.parse_cpulist("") == set() and NP.parse_cpulist(None) == set()


@pytest.fixture
def two_nodes(tmp_path, monkeypatch):
    root = tmp_path / "node"
    for n, cpus, free in ((0, "0-15", 200 * 1048576), (1, "16-31", 3 * 1048576)):
        d = root / ("node%d" % n)
        d.mkdir(parents=True)
        (d / "cpulist").write_text(cpus + "\n")
        (d / "meminfo").write_text("Node %d MemTotal: %d kB\nNode %d MemFree: %d kB\nNode %d FilePages: 1048576 kB\n"
                                   % (n, 400 * 1048576, n, free, n))
    monkeypatch.setattr(NP, "NODE_ROOT", str(root))
    monkeypatch.setattr(NP.os, "sched_getaffinity", lambda pid: set(range(32)), raising=False)
    monkeypatch.setattr(NP, "mems_allowed", lambda: {0, 1})
    monkeypatch.setattr(NP, "allocation_cores", lambda: 14)
    monkeypatch.delenv(NP.BIND_ENV, raising=False)
    monkeypatch.delenv(NP.MIN_FREE_ENV, raising=False)
    monkeypatch.setattr(NP.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(NP, "_try_prefix", lambda how, timeout=20: None)     # every form starts
    return root


def test_binds_to_the_gpu_node_cores_and_memory(two_nodes, monkeypatch):
    monkeypatch.setenv(NP.GPU_NODE_ENV, "0")
    p = NP.plan()
    assert p["node"] == 0 and p["cpus"] == "0-15" and p["mem"] == "bind"
    assert p["how"] == ["/usr/bin/numactl", "--physcpubind=0-15", "--membind=0"]
    env = {}
    NP._PLAN.clear()
    cmd, rec = NP.wrap(["python", "w.py"], env)
    assert cmd[:3] == p["how"] and cmd[3:] == ["python", "w.py"]
    assert json.loads(env[NP.PLACEMENT_ENV])["node"] == 0 and "how" not in rec
    NP._PLAN.clear()


def test_full_node_is_preferred_not_bound(two_nodes, monkeypatch):
    monkeypatch.setenv(NP.GPU_NODE_ENV, "1")         # node 1 has 3 GB free: never membind a full node
    p = NP.plan()
    assert p["node"] == 1 and p["mem"] == "preferred" and p["how"][-1] == "--preferred=1"


def test_never_binds_away_cores_of_the_allocation(two_nodes, monkeypatch):
    monkeypatch.setenv(NP.GPU_NODE_ENV, "0")
    monkeypatch.setattr(NP, "allocation_cores", lambda: 24)    # 24 cores allocated, the node has 16
    p = NP.plan()
    assert p["node"] is None and "16 of the allocation's 24" in p["why"]


def test_off_switch_and_unknown_gpu(two_nodes, monkeypatch):
    monkeypatch.setenv(NP.BIND_ENV, "0")
    assert NP.plan()["node"] is None
    monkeypatch.delenv(NP.BIND_ENV)
    monkeypatch.setenv(NP.GPU_NODE_ENV, "none")
    assert NP.plan()["why"] == "the GPU's NUMA node is unknown"


def test_without_numactl_cores_only(two_nodes, monkeypatch):
    monkeypatch.setenv(NP.GPU_NODE_ENV, "0")
    monkeypatch.setattr(NP.shutil, "which", lambda name: "/usr/bin/taskset" if name == "taskset" else None)
    p = NP.plan()
    assert p["how"] == ["/usr/bin/taskset", "-c", "0-15"] and p["mem"].startswith("unbound")


def test_refused_memory_policy_falls_back_to_cores(two_nodes, monkeypatch):
    # lane fx-pod-job2 (2026-11-18): the RunPod container refuses set_mempolicy; `numactl --membind` exited 1 and
    # every worker-protocol race failed its smoke. The refused form is never used: cores only, and the cell says why
    monkeypatch.setenv(NP.GPU_NODE_ENV, "0")
    tried = []

    def refuse_mem(how, timeout=20):
        tried.append(list(how))
        return "set_mempolicy: Operation not permitted" if any(a.startswith(("--membind", "--preferred"))
                                                                for a in how) else None
    monkeypatch.setattr(NP, "_try_prefix", refuse_mem)
    p = NP.plan()
    assert p["node"] == 0 and p["how"] == ["/usr/bin/numactl", "--physcpubind=0-15"]
    assert "set_mempolicy" in p["mem"] and p["mem"].startswith("unbound")
    assert tried == [["/usr/bin/numactl", "--physcpubind=0-15", "--membind=0"], ["/usr/bin/numactl", "--physcpubind=0-15"]]
    monkeypatch.setattr(NP, "_try_prefix", lambda how, timeout=20: "Operation not permitted")
    p = NP.plan()
    assert p["node"] is None and "no placement command starts" in p["why"]


def test_try_prefix_runs_the_command():
    assert NP._try_prefix([]) is None               # `true` alone: present on every POSIX host
    assert NP._try_prefix(["/bin/sh", "-c", "echo refused >&2; exit 1", "x"]) == "refused"


def test_box_facts_and_thp(two_nodes, monkeypatch):
    monkeypatch.setenv(NP.GPU_NODE_ENV, "0")
    f = NP.box_facts()
    assert f["gpu_numa_node"] == 0 and f["numa"]["1"]["mem_free_gb"] == 3.0 and f["numa"]["0"]["cpus"] == "0-15"
    assert set(f["thp"]) == {"enabled", "defrag"}


def test_usage_summary_flags_a_host_stall():
    a = types.SimpleNamespace(ru_stime=1.0, ru_utime=1.0, ru_minflt=10, ru_majflt=0)
    b = types.SimpleNamespace(ru_stime=13.0, ru_utime=1.5, ru_minflt=650010, ru_majflt=0)
    d = NP.delta(a, b, 13021.0)
    assert d["stime_ms"] == 12000.0 and d["minflt"] == 650000
    s = NP.phase_summary([{"fit": [d], "infer": None, "round": 1}], "fit")
    assert s["host_stall"] is True and s["calls"] == 1 and s["per_call"][0]["minflt"] == 650000
    calm = NP.delta(a, types.SimpleNamespace(ru_stime=1.1, ru_utime=1.7, ru_minflt=20, ru_majflt=0), 750.0)
    assert NP.summarize([calm])["host_stall"] is False
    assert NP.phase_summary([{"fit": [d], "infer": None}], "infer") is None


def _cgroup_tree(tmp_path, monkeypatch, proc, files, affinity=128):
    root = tmp_path / "cgroup"
    for rel, text in files.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    root.mkdir(exist_ok=True)
    p = tmp_path / "proc_self_cgroup"
    p.write_text(proc)
    monkeypatch.setattr(NP, "CGROUP_ROOT", str(root))
    monkeypatch.setattr(NP, "PROC_CGROUP", str(p))
    monkeypatch.setattr(NP.os, "sched_getaffinity", lambda pid: set(range(affinity)), raising=False)


def test_allocation_reads_a_cgroup_v1_quota(tmp_path, monkeypatch):
    # lane fx-cagra-taxi (2026-11-19): the RunPod L40S pod's quota is cgroup v1 (14 cores of 128 visible). Only v2's
    # root cpu.max was read, the allocation read as 128, and no worker was bound to the GPU's node.
    _cgroup_tree(tmp_path, monkeypatch, "5:cpu,cpuacct:/\n1:name=systemd:/\n",
                 {"cpu/cpu.cfs_quota_us": "1400000\n", "cpu/cpu.cfs_period_us": "100000\n"})
    assert NP.cgroup_quota_cores() == 14 and NP.allocation_cores() == 14


def test_allocation_reads_v2_root_own_path_and_no_quota(tmp_path, monkeypatch):
    _cgroup_tree(tmp_path, monkeypatch, "0::/\n", {"cpu.max": "2750000 100000\n"})
    assert NP.allocation_cores() == 28                      # rounded up
    _cgroup_tree(tmp_path, monkeypatch, "0::/kubepods/pod1\n",
                 {"cpu.max": "max 100000\n", "kubepods/cpu.max": "max 100000\n",
                  "kubepods/pod1/cpu.max": "800000 100000\n"})
    assert NP.cgroup_quota_cores() == 8                     # the limit sits on the leaf, not the mount root
    _cgroup_tree(tmp_path, monkeypatch, "0::/\n", {"cpu.max": "max 100000\n"}, affinity=32)
    assert NP.cgroup_quota_cores() is None and NP.allocation_cores() == 32
    _cgroup_tree(tmp_path, monkeypatch, "4:cpu,cpuacct:/\n",
                 {"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"}, affinity=32)
    assert NP.cgroup_quota_cores() is None and NP.allocation_cores() == 32
    _cgroup_tree(tmp_path, monkeypatch, "0::/\n", {"cpu.max": "6400000 100000\n"}, affinity=16)
    assert NP.allocation_cores() == 16                      # never above the affinity


def test_v1_quota_binds_the_workers_to_the_gpu_node(tmp_path, monkeypatch):
    # the pod of the pair job: 14 quota cores, the GPU's node holds 16 of the 32 visible: bound, no core lost
    root = tmp_path / "node2"
    for n, cpus in ((0, "0-15"), (1, "16-31")):
        d = root / ("node%d" % n)
        d.mkdir(parents=True)
        (d / "cpulist").write_text(cpus + "\n")
        (d / "meminfo").write_text("Node %d MemTotal: 419430400 kB\nNode %d MemFree: 209715200 kB\n" % (n, n))
    monkeypatch.setattr(NP, "NODE_ROOT", str(root))
    monkeypatch.setattr(NP, "mems_allowed", lambda: {0, 1})
    monkeypatch.delenv(NP.BIND_ENV, raising=False)
    monkeypatch.delenv(NP.MIN_FREE_ENV, raising=False)
    monkeypatch.setattr(NP.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(NP, "_try_prefix", lambda how, timeout=20: None)
    monkeypatch.setenv(NP.GPU_NODE_ENV, "1")
    _cgroup_tree(tmp_path, monkeypatch, "5:cpu,cpuacct:/\n",
                 {"cpu/cpu.cfs_quota_us": "1400000\n", "cpu/cpu.cfs_period_us": "100000\n"}, affinity=32)
    p = NP.plan()
    assert p["node"] == 1 and p["cpus"] == "16-31" and p["allocation_cores"] == 14
