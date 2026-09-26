# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Scheduler regression tests use isolated directories and no GPU."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import mac_slot
from mac_slot import Scheduler


@pytest.fixture
def env(tmp_path):
    return dict(os.environ, MOJOLEARN_MAC_SLOT_BASE=str(tmp_path / "slot"),
                MOJOLEARN_METAL_LOCK=str(tmp_path / "metal"),
                MOJOLEARN_METAL_QUEUE=str(tmp_path / "queue"), MAC_SLOTS="1",
                MOJOLEARN_SLOT_CAPACITY="0")


def test_new_arrival_cannot_jump_waiter(env):
    first, waiter, newcomer = [Scheduler(env) for _ in range(3)]
    first.enqueue(["first"])
    assert first.attempt(True, ["first"])
    waiter.enqueue(["waiter"])
    newcomer.enqueue(["newcomer"])
    assert int(waiter.ticket.name[1:]) < int(newcomer.ticket.name[1:])
    first.release()
    assert not newcomer.attempt(True, ["newcomer"])
    assert waiter.attempt(True, ["waiter"])
    waiter.release()
    assert newcomer.attempt(True, ["newcomer"])
    newcomer.release()


def test_waiting_for_cpu_does_not_hold_metal(env):
    cpu, gpu = Scheduler(env), Scheduler(env)
    assert cpu.attempt(False, ["cpu"])
    gpu.enqueue(["gpu"])
    assert not gpu.attempt(True, ["gpu"])
    assert not gpu.metal.exists()
    cpu.release()
    assert gpu.attempt(True, ["gpu"])
    gpu.release()


def test_multi_slot_job_takes_all_or_nothing(env):
    env = dict(env, MAC_SLOTS="3")
    one, many = Scheduler(env), Scheduler(env)
    assert one.attempt(False, ["one"])
    assert not many.attempt(False, ["many"], slots=3)
    assert not many.held and sum(p.exists() for p in many.slots) == 1
    assert many.attempt(False, ["many"], slots=2)
    assert len(many.held) == 2 and all(p.exists() for p in many.slots)
    many.release()
    one.release()
    assert not any(p.exists() for p in one.slots)
    with pytest.raises(ValueError):
        many.attempt(True, ["gpu"], slots=2)


def test_slots_sets_the_build_jobs_and_nothing_else(env, tmp_path):
    env = dict(env, MAC_SLOTS="4")
    out = tmp_path / "env.txt"
    body = ("import os,sys;open(sys.argv[1],'w').write(os.environ['MOJOLEARN_BUILD_JOBS']+' '"
            "+os.environ['MOJOLEARN_COMPILE_JOBS']+' '+os.environ['OMP_NUM_THREADS'])")
    p = launch(env, "--slots", "4", "run", sys.executable, "-c", body, str(out))
    p.communicate(timeout=10)
    assert p.returncode == 0 and out.read_text() == "4 1 1"
    for bad in (("--slots", "5", "run"), ("--slots", "2", "metal"), ("--slots", "0", "run")):
        p = launch(env, *bad, sys.executable, "-c", "pass")
        p.communicate(timeout=10)
        assert p.returncode == 2, bad


def test_live_legacy_and_unpublished_locks_not_stolen(env):
    s = Scheduler(env)
    s.metal.mkdir()
    s.enqueue(["gpu"])
    assert not s.attempt(True, ["gpu"])
    (s.metal / "pid").write_text(str(os.getpid()))
    assert not s.attempt(True, ["gpu"])
    s.release()
    assert s.metal.exists()


def test_stale_lock_recovered_but_live_child_preserved(env):
    s = Scheduler(env)
    s.metal.mkdir()
    (s.metal / "pid").write_text("999999999")
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
    try:
        (s.metal / "pgid").write_text(str(child.pid))
        s.enqueue(["gpu"])
        assert not s.attempt(True, ["gpu"])
        child.terminate()
        child.wait(timeout=5)
        assert s.attempt(True, ["gpu"])
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        s.release()


def launch(env, *args):
    return subprocess.Popen([sys.executable, str(Path(__file__).with_name("mac_slot.py")),
                             "--poll", "0.01", *args], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_timeout_releases_slots_and_preserves_exit_code(env):
    p = launch(env, "--timeout", "0.1", "metal", sys.executable, "-c", "import time;time.sleep(30)")
    _, err = p.communicate(timeout=10)
    assert p.returncode == 124
    assert '"wait_seconds"' in err and '"run_seconds"' in err
    assert not Path(env["MOJOLEARN_METAL_LOCK"]).exists()
    p = launch(env, "metal", sys.executable, "-c", "raise SystemExit(7)")
    p.communicate(timeout=10)
    assert p.returncode == 7


def test_cancelled_waiter_removes_ticket(env):
    owner = Scheduler(env)
    owner.enqueue(["owner"])
    assert owner.attempt(True, ["owner"])
    p = launch(env, "metal", sys.executable, "-c", "pass")
    try:
        deadline = time.monotonic() + 5
        while not list(owner.queue.glob("t[0-9]*")) and time.monotonic() < deadline:
            time.sleep(.01)
        assert list(owner.queue.glob("t[0-9]*"))
        p.send_signal(signal.SIGTERM)
        p.communicate(timeout=5)
        assert p.returncode == 143
        assert not list(owner.queue.glob("t[0-9]*"))
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        owner.release()


@pytest.mark.parametrize("mode", ["metal", "cuda", "hip"])
def test_concurrent_jobs_never_overlap(env, tmp_path, mode):
    marker = str(tmp_path / "exclusive")
    body = "import os,time,sys;p=sys.argv[1];f=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY);time.sleep(.05);os.close(f);os.unlink(p)"
    jobs = [launch(env, mode, sys.executable, "-c", body, marker) for _ in range(5)]
    for p in jobs:
        out, err = p.communicate(timeout=10)
        assert p.returncode == 0, (out, err)


@pytest.mark.parametrize("flag", ["--timeout", "--wait-timeout", "--poll", "--deadline"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_limits_refuse_before_scheduler(monkeypatch, flag, value):
    import mac_slot
    monkeypatch.setattr(mac_slot, "Scheduler", lambda: pytest.fail("invalid limit reached scheduler"))
    with pytest.raises(SystemExit):
        mac_slot.main([f"{flag}={value}", "run", "true"])


def test_shared_deadline_includes_queue_and_execution(env, tmp_path):
    owner = Scheduler(env)
    assert owner.attempt(False, ["owner"])
    marker = tmp_path / "started"
    deadline = time.monotonic() + .8
    p = launch(env, "--deadline", str(deadline), "run", sys.executable, "-c",
               "import pathlib,time,sys;pathlib.Path(sys.argv[1]).touch();time.sleep(30)", str(marker))
    try:
        time.sleep(.2)
        owner.release()
        p.communicate(timeout=5)
        assert marker.exists(), "child never ran; execution budget was not exercised"
        assert p.returncode == 124
        assert time.monotonic() < deadline + 2
        assert not any(Path(env["MOJOLEARN_MAC_SLOT_BASE"]).parent.glob("slot.[0-9]*"))
    finally:
        owner.release()
        if p.poll() is None:
            p.kill()
            p.wait()


def test_expired_deadline_never_launches(env, tmp_path):
    marker = tmp_path / "must-not-exist"
    p = launch(env, "--deadline", str(time.monotonic() - 1), "metal", sys.executable,
               "-c", "import pathlib,sys;pathlib.Path(sys.argv[1]).touch()", str(marker))
    p.communicate(timeout=5)
    assert p.returncode == 124 and not marker.exists()
    assert not Path(env["MOJOLEARN_METAL_LOCK"]).exists()


# ---- capacity (2026-09-26): a ceiling sized to the machine, admission by room

def room_env(env, **extra):
    return dict(env, MOJOLEARN_SLOT_CAPACITY="1", MAC_SLOTS="4", **extra)


def test_slot_count_is_mac_slots_else_half_the_cores(monkeypatch):
    assert mac_slot.slot_count({"MAC_SLOTS": "7"}) == 7
    monkeypatch.setattr(mac_slot.os, "cpu_count", lambda: 10)
    assert mac_slot.slot_count({}) == 5
    monkeypatch.setattr(mac_slot.os, "cpu_count", lambda: 1)
    assert mac_slot.slot_count({}) == 1
    monkeypatch.setattr(mac_slot.os, "cpu_count", lambda: None)
    assert mac_slot.slot_count({}) == 1


def test_first_job_is_admitted_whatever_the_readings(env):
    s = Scheduler(room_env(env), loadavg=lambda: 99.0, available=lambda: 0)
    assert s.attempt(False, ["cpu"])
    s.release()


def test_high_load_holds_the_next_cpu_job_and_says_why(env, monkeypatch):
    monkeypatch.setattr(mac_slot.os, "cpu_count", lambda: 10)
    load = [0.0]
    e = room_env(env)
    first = Scheduler(e, loadavg=lambda: load[0], available=lambda: 64 * 2**30)
    second = Scheduler(e, loadavg=lambda: load[0], available=lambda: 64 * 2**30)
    assert first.attempt(False, ["first"])
    load[0] = 7.5                      # 7.5 + 1 > 0.8 * 10
    assert not second.attempt(False, ["second"])
    assert "load 7.5" in second.why and not second.held
    load[0] = 6.9                      # 6.9 + 1 <= 8
    assert second.attempt(False, ["second"])
    first.release()
    second.release()


def test_multi_slot_job_counts_every_slot_against_the_load(env, monkeypatch):
    monkeypatch.setattr(mac_slot.os, "cpu_count", lambda: 10)
    e = room_env(env)
    one = Scheduler(e, loadavg=lambda: 5.5, available=lambda: None)
    many = Scheduler(e, loadavg=lambda: 5.5, available=lambda: None)
    assert one.attempt(False, ["one"])
    assert not many.attempt(False, ["many"], slots=3)   # 5.5 + 3 > 8
    assert many.attempt(False, ["many"], slots=2)       # 5.5 + 2 <= 8
    many.release()
    one.release()


def test_metal_is_exempt_from_load_but_not_from_memory(env, monkeypatch):
    monkeypatch.setattr(mac_slot.os, "cpu_count", lambda: 10)
    free = [64 * 2**30]
    e = room_env(env)
    cpu = Scheduler(e, loadavg=lambda: 50.0, available=lambda: free[0])
    gpu = Scheduler(e, loadavg=lambda: 50.0, available=lambda: free[0])
    assert cpu.attempt(False, ["cpu"])
    gpu.enqueue(["gpu"])
    free[0] = 2 * 2**30
    assert not gpu.attempt(True, ["gpu"])
    assert "available memory" in gpu.why and not gpu.metal.exists()
    free[0] = 64 * 2**30
    assert gpu.attempt(True, ["gpu"])
    gpu.release()
    cpu.release()


def test_unreadable_memory_does_not_block(env):
    e = room_env(env)
    a = Scheduler(e, loadavg=lambda: 0.0, available=lambda: None)
    b = Scheduler(e, loadavg=lambda: 0.0, available=lambda: None)
    assert a.attempt(False, ["a"]) and b.attempt(False, ["b"])
    a.release()
    b.release()


def test_capacity_off_leaves_the_plain_slot_count(env):
    e = dict(room_env(env), MOJOLEARN_SLOT_CAPACITY="0")
    a = Scheduler(e, loadavg=lambda: 99.0, available=lambda: 0)
    b = Scheduler(e, loadavg=lambda: 99.0, available=lambda: 0)
    assert a.attempt(False, ["a"]) and b.attempt(False, ["b"])
    a.release()
    b.release()


def test_available_bytes_reads_this_machine():
    free = mac_slot.available_bytes()
    assert free is None or free > 0
    if sys.platform == "darwin" or Path("/proc/meminfo").exists():
        assert free is not None and free > 2**20


def test_waiting_log_names_the_reason(env, tmp_path):
    e = dict(room_env(env), MOJOLEARN_SLOT_LOAD_FRACTION="0.0001")
    holder = Scheduler(e)
    assert holder.attempt(False, ["holder"])
    p = launch(e, "--wait-timeout", "1", "run", sys.executable, "-c", "pass")
    _, err = p.communicate(timeout=20)
    holder.release()
    assert p.returncode == 124
    assert "waiting" in err and "(load " in err


def test_a_gpu_job_alone_does_not_stall_one_cpu_job(env):
    e = room_env(env)
    gpu = Scheduler(e, loadavg=lambda: 99.0, available=lambda: None)
    cpu1 = Scheduler(e, loadavg=lambda: 99.0, available=lambda: None)
    cpu2 = Scheduler(e, loadavg=lambda: 99.0, available=lambda: None)
    gpu.enqueue(["gpu"])
    assert gpu.attempt(True, ["gpu"])
    assert cpu1.attempt(False, ["cpu1"])       # only the GPU holder: one CPU job runs
    assert not cpu2.attempt(False, ["cpu2"])   # a second waits on the load
    assert "load 99.0" in cpu2.why
    cpu1.release()
    gpu.release()
