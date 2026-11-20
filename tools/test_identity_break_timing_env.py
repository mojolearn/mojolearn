# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""identity_break.timing_environment: the wall-clock environment a record states
(timing record only; nothing hashed reads it). No Mojolearn import, < 1 s."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import identity_break as ib  # noqa: E402


def test_timing_environment_states_threads_cores_and_concurrency(monkeypatch):
    monkeypatch.setenv("MOJOLEARN_CPU_THREADS", "8")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv(ib.TIMING_CONCURRENCY_ENV, "1/4")
    env = ib.timing_environment()
    assert env["schema"] == ib.TIMING_ENV_SCHEMA
    assert env["threads"]["MOJOLEARN_CPU_THREADS"] == "8"
    assert env["threads"]["OMP_NUM_THREADS"] is None
    assert set(env["threads"]) == set(ib.TIMING_THREAD_ENV)
    assert env["concurrency"] == "1/4"
    assert env["logical_cpus"] == os.cpu_count()
    assert env["cgroup_cpu_quota"] is None or env["cgroup_cpu_quota"] > 0
    assert env["other_identity_processes"] is None or env["other_identity_processes"] >= 0
    assert env["started_utc"].endswith("Z")


def test_concurrency_unset_is_not_assumed(monkeypatch):
    monkeypatch.delenv(ib.TIMING_CONCURRENCY_ENV, raising=False)
    assert ib.timing_environment()["concurrency"] is None


def test_timing_env_is_outside_every_merge_and_host_check():
    """Shards of one column run side by side and state different `i/N`; the
    environment must never make `--merge` refuse them."""
    assert "timing_env" not in ib.MERGE_SAME
    part = dict(host=dict(cpu_model="x", families={}), timing_env=dict(concurrency="0/2"))
    other = dict(host=dict(cpu_model="x", families={}), timing_env=dict(concurrency="1/2"))
    assert ib._host_without_digests(part) == ib._host_without_digests(other)
