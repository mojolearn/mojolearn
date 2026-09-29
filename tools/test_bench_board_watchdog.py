"""The bench board's host-memory watchdog (tools/bench_board_watchdog.py).

2026-09-29: a scikit-learn DBSCAN arm took 157 GB on do-amd and the OOM kill
stopped the whole steward service. Every driver the board starts is now
watched; the arm that grows past the limit is killed and named.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_board as bb                 # noqa: E402
import bench_board_watchdog as wd        # noqa: E402

GROW = ("import time\nblocks = []\n"
        "for _ in range(60):\n"
        "    b = bytearray(16 * 1024 * 1024)\n"
        "    b[::4096] = b'x' * len(b[::4096])\n"
        "    blocks.append(b)\n"
        "    time.sleep(0.05)\n"
        "time.sleep(60)\n")


def test_a_growing_arm_is_killed_and_named(tmp_path):
    """A driver whose worker (in its own session) grows past the limit: the
    worker is killed, the driver survives it, the kill names the worker."""
    worker = tmp_path / "grow.py"
    worker.write_text(GROW)
    driver = ("import subprocess, sys\n"
              "p = subprocess.Popen([sys.executable, %r, '--arm', 'sklearn-cpu'], start_new_session=True)\n"
              "print('worker rc', p.wait(), flush=True)\n" % str(worker))
    log = open(tmp_path / "race.log", "w")
    proc = subprocess.Popen([sys.executable, "-c", driver], stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    # a 2 GB "box" with a 0.15 limit: 300 MB; the worker reaches about 960 MB
    dog = wd.HostMemoryWatchdog(proc.pid, log, limit_frac=0.15, ram=2 * 1024 ** 3,
                                available=lambda: None).start()
    try:
        rc = proc.wait(timeout=120)
    finally:
        kills = dog.stop()
        log.close()
    assert rc == 0
    assert len(kills) >= 1
    k = kills[0]
    assert "--arm sklearn-cpu" in k["command"] and k["mb"] > 250
    assert wd.arm_of(k["command"], ["ours", "sklearn-cpu"]) == "sklearn-cpu"
    text = (tmp_path / "race.log").read_text()
    assert "HOST MEMORY: killed pid %d" % k["pid"] in text
    assert "worker rc -9" in text


def test_check_once_picks_the_heaviest_of_the_tree_only(monkeypatch):
    procs = {100: (1, 50 * 2 ** 20, "driver"),
             101: (100, 900 * 2 ** 20, "python w.py worker --arm sklearn-cpu"),
             102: (100, 10 * 2 ** 20, "python w.py worker --arm ours"),
             200: (1, 5000 * 2 ** 20, "someone else")}
    killed = []
    monkeypatch.setattr(wd.os, "kill", lambda pid, sig: killed.append(pid))
    dog = wd.HostMemoryWatchdog(100, None, limit_frac=0.5, ram=2 * 2 ** 30,
                                sampler=lambda: procs, available=lambda: None)
    assert dog.check_once() is None                 # 960 MB < 1 GB: nothing
    procs[101] = (100, 1100 * 2 ** 20, procs[101][2])
    rec = dog.check_once()
    assert killed == [101] and rec["pid"] == 101    # never pid 200, outside the tree
    # MemAvailable under 3% of RAM fires even under the tree limit
    procs[101] = (100, 100 * 2 ** 20, procs[101][2])
    dog2 = wd.HostMemoryWatchdog(100, None, limit_frac=0.9, ram=2 * 2 ** 30,
                                 sampler=lambda: procs, available=lambda: 10 * 2 ** 20)
    assert dog2.check_once()["pid"] == 101 and "available" in dog2.kills[0]["why"]


def test_arm_of():
    arms = ["ours", "sklearn-cpu", "torch-gpu", "torch-gpu-eigh"]
    assert wd.arm_of("py tools/x.py worker --arm torch-gpu-eigh --lane ols", arms) == "torch-gpu-eigh"
    assert wd.arm_of("py tools/x.py worker --arm=ours", arms) == "ours"
    assert wd.arm_of("py bench/speed/forest_speed_arm.py --lane rf", arms) is None


def test_the_race_fails_by_name():
    rec = {"status": "done", "cells": [{"arm": "ours", "status": "REFUSED(x)"},
                                       {"arm": "sklearn-cpu", "status": "UNKNOWN(no race json)"}]}
    kills = [{"command": "py c.py worker --arm sklearn-cpu", "mb": 150000.0,
              "why": "the driver's process tree held 157 GB, over 90% of the box's 168 GB"}]
    bb.note_host_memory(rec, kills, ["ours", "sklearn-cpu"])
    assert rec["status"] == "failed" and rec["failure"].startswith("HOST MEMORY: the watchdog killed sklearn-cpu")
    assert rec["cells"][1]["status"].startswith("HOST-MEMORY(killed at 146.5 GB")
    assert rec["cells"][0]["status"] == "REFUSED(x)"
    rec2 = {"status": "done", "cells": []}
    bb.note_host_memory(rec2, [], ["ours"])
    assert rec2 == {"status": "done", "cells": []}


def test_run_logged_runs_the_watchdog(tmp_path, monkeypatch):
    monkeypatch.setenv(wd.LIMIT_ENV, "0.9")
    n = len(bb.HOST_MEMORY_KILLS)
    rc = bb.run_logged([sys.executable, "-c", "print('hi')"], None, str(tmp_path / "l.log"), 60)
    assert rc == 0 and len(bb.HOST_MEMORY_KILLS) == n
    assert "watchdog not started" not in (tmp_path / "l.log").read_text()
