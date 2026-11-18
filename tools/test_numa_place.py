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
