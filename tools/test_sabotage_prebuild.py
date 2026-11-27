# SPDX-License-Identifier: Apache-2.0
"""Scheduling, cancellation and artifact ownership of the sabotage prebuilder (no compiler required)."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

import sabotage_prebuild as pre


def test_reuse_requires_clean_bytes_and_the_complete_patched_set(tmp_path):
    specification = {"builds": ["build"]}
    rel = "python/mojolearn/sample.so"
    patched = tmp_path / "out" / rel
    patched.parent.mkdir(parents=True)
    patched.write_bytes(b"patched")
    receipt = dict(prepared=True, clean_sha256="clean", rc={"build": 0}, changed={rel: pre.sha(patched)})
    (tmp_path / "ready").write_text(json.dumps(receipt))
    assert pre.reusable(tmp_path, specification, "clean")
    assert not pre.reusable(tmp_path, specification, "different clean build")
    patched.write_bytes(b"truncated")
    assert not pre.reusable(tmp_path, specification, "clean")
    patched.unlink()
    assert not pre.reusable(tmp_path, specification, "clean")


def test_every_plan_build_maps_to_a_development_group():
    groups = json.load(open(pre.Path(__file__).resolve().parent.parent / "bindings" / "development_groups.json"))
    # the development stems the base build records (identical-_mojolearn_dev_core, host-_mojolearn_dev_core_host)
    stems = {(mode, "_mojolearn_dev_" + name + ("_host" if mode == "host" else ""))
             for mode, table in groups["groups"].items() for name in table}
    plan = json.load(open(pre.Path(__file__).resolve().parent / "identity_lanes" / "coverage_sabotage_gpu_plan.json"))
    builds = {b for batch in plan["batches"] for b in batch["builds"]}
    assert builds and {b for b in builds if pre.dev_group(b) not in stems} == set()


def test_recorded_reads_only_a_passed_peak(tmp_path):
    (tmp_path / "peaks").mkdir()
    (tmp_path / "peaks" / "host-_mojolearn_dev_runtime_prep_host.json").write_text(
        json.dumps({"status": "passed", "peak_rss_bytes": 3 << 30, "elapsed_seconds": 685.0}))
    (tmp_path / "peaks" / "identical-_mojolearn_dev_core.json").write_text(
        json.dumps({"status": "lower-bound", "lower_bound_rss_bytes": 1 << 30}))
    assert pre.recorded(tmp_path, "build_runtime_prep_host") == (3 << 30, 685.0)
    assert pre.recorded(tmp_path, "build") == (0, 0.0)
    assert pre.recorded("", "build") == (0, 0.0)


def test_fallback_keeps_running_and_late_prebuilds(tmp_path):
    directory = tmp_path / "batch"
    directory.mkdir()
    (directory / "build.log").touch()
    staged_receipt = tmp_path / "consumer" / "prebuilt-consumed.json"
    pre.retire_consumed(directory, staged_receipt)
    assert (directory / "build.log").is_file()
    (directory / "ready").write_bytes(b"published")
    pre.retire_consumed(directory, staged_receipt)
    assert (directory / "ready").is_file()

    staged_receipt.parent.mkdir()
    staged_receipt.write_bytes(b"other attempt")
    with pytest.raises(ValueError, match="receipt changed"):
        pre.retire_consumed(directory, staged_receipt)
    staged_receipt.write_bytes(b"published")
    pre.retire_consumed(directory, staged_receipt)
    assert not directory.exists()


@pytest.mark.parametrize(("stop", "expected_rc"), [("timeout", 124), ("cancel", 128 + signal.SIGTERM)])
def test_prebuild_releases_descendants_before_returning(tmp_path, stop, expected_rc):
    # Simulate a compiler that outlives its launcher and ignores TERM. The lock
    # tests whether it can still write, even when its PID survives as a zombie.
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import fcntl, os, pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "with open('lock', 'w') as lock:\n"
        "    fcntl.flock(lock, fcntl.LOCK_EX)\n"
        "    pathlib.Path('pid.tmp').write_text(str(os.getpid()))\n"
        "    pathlib.Path('pid.tmp').rename('worker.pid')\n"
        "    time.sleep(30)\n"
    )
    launcher = "import subprocess, sys; subprocess.run([sys.executable, 'worker.py'])"
    # Use a separate controller so SIGTERM exercises the real signal handler
    # while the compile is running, without changing pytest's own handlers.
    controller = (
        "import signal, sys; import sabotage_prebuild as pre; "
        "signal.signal(signal.SIGTERM, pre.request_stop); "
        "sys.exit(pre.run([sys.executable, '-c', sys.argv[1]], '.', 'build.log', timeout=float(sys.argv[2])))"
    )
    proc = subprocess.Popen([sys.executable, "-c", controller, launcher, "5" if stop == "timeout" else "30"],
                            cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(Path(pre.__file__).parent)))
    pid_file = tmp_path / "worker.pid"
    lock_file = tmp_path / "lock"
    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.02)
        assert pid_file.exists()
        if stop == "cancel":
            proc.terminate()
        assert proc.wait(timeout=10) == expected_rc
        with lock_file.open() as lock:
            deadline = time.monotonic() + 2
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.02)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if pid_file.is_file():
            # Only kill a surviving worker while it still holds our lock.
            with lock_file.open() as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    try:
                        os.kill(int(pid_file.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
