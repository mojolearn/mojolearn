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
from mac_slot import Scheduler


@pytest.fixture
def env(tmp_path):
    return dict(os.environ, MOJOLEARN_MAC_SLOT_BASE=str(tmp_path / "slot"),
                MOJOLEARN_METAL_LOCK=str(tmp_path / "metal"),
                MOJOLEARN_METAL_QUEUE=str(tmp_path / "queue"), MAC_SLOTS="1")


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
