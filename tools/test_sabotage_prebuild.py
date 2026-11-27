# SPDX-License-Identifier: Apache-2.0
"""Scheduling, cancellation and artifact ownership of the sabotage prebuilder (no compiler required)."""
import fcntl
import json
import os
import shlex
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

import pytest

import sabotage_prebuild as pre
from sabotage_natives import configured_dirs, natives, stage_wheels


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


@pytest.mark.parametrize(("stop", "expected_rc"), [
    ("timeout", 124), ("cancel", 128 + signal.SIGTERM), ("fallback", 128 + signal.SIGTERM),
])
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
    command = [sys.executable, "-c", controller, launcher, "5" if stop == "timeout" else "30"]
    if stop == "fallback":
        binding = tmp_path / "bindings" / "build.sh"
        binding.parent.mkdir()
        binding.write_text(f"exec {shlex.quote(sys.executable)} -c {shlex.quote(launcher)}\n")
        command = [sys.executable, pre.__file__, "build-in-place", str(tmp_path), str(tmp_path), "1", "build"]
    proc = subprocess.Popen(command, cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(Path(pre.__file__).parent)))
    pid_file = tmp_path / "worker.pid"
    lock_file = tmp_path / "lock"
    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.02)
        assert pid_file.exists()
        if stop in {"cancel", "fallback"}:
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


def test_wheel_staging_snapshot_and_reuse_share_native_inventory(tmp_path):
    wheel = tmp_path / "candidate.whl"
    compiled = {
        "mojolearn/pkg/extension.so", "mojolearn/pkg/macos.dylib", "mojolearn/_mojolearn_merged.json",
        "mojolearn/runtime/libmojo.so.1.2", "mojolearn.libs/libsupport-a1b2.so.9",
    }
    ignored = {"unrelated/extension.so", "mojolearn/libs.so.info", "mojolearn/module.py"}
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in compiled | ignored:
            archive.writestr(name, name.encode())
    root = tmp_path / "batch" / "out"
    sibling = root / "python/mojolearn.libs"
    sibling.mkdir(parents=True)
    (sibling / "old.so.1").touch()
    (sibling / "keep.txt").touch()
    assert stage_wheels(root, [wheel]) == (1, len(compiled))
    expected = sorted("python/" + rel for rel in compiled)
    assert natives(root) == pre.natives(root) == expected
    assert (sibling / "keep.txt").is_file()
    assert all(not (root / "python" / rel).exists() for rel in ignored)
    hashes = {rel: pre.sha(root / rel) for rel in expected}
    receipt = dict(prepared=True, clean_sha256="clean", rc={"build": 0}, changed=hashes)
    (root.parent / "ready").write_text(json.dumps(receipt))
    assert pre.reusable(root.parent, {"builds": ["build"]}, "clean")
    (sibling / "libsupport-a1b2.so.9").write_bytes(b"corrupt")
    assert not pre.reusable(root.parent, {"builds": ["build"]}, "clean")


def test_extra_native_roots_keep_the_wheel_inventory(monkeypatch, tmp_path):
    monkeypatch.setenv("NATIVE_DIRS", "extra/host")
    expected = ["extra/host/host.so", "python/mojolearn/ext.so", "python/mojolearn.libs/dep.so.1"]
    for name in expected:
        path = tmp_path / name
        path.parent.mkdir(parents=True)
        path.touch()
    assert natives(tmp_path, configured_dirs()) == expected


def test_fallback_records_independent_build_statuses(tmp_path, monkeypatch):
    def fake_build(name, tree, prefix, cores, directory):
        assert tree == str(tmp_path) and directory == tmp_path
        assert prefix == ["tool", "with spaces"] and cores == 2
        return (3, 7) if name == "build_a" else (0, 4)

    monkeypatch.setattr(pre, "build_binding", fake_build)
    assert pre.build_in_place([str(tmp_path), str(tmp_path), "2", "build_a,build_b", "tool", "with spaces"]) == 0
    assert (tmp_path / "build_build_a.rc").read_text() == "3\n"
    assert (tmp_path / "build_build_a.secs").read_text() == "7\n"
    assert (tmp_path / "build_build_b.rc").read_text() == "0\n"
