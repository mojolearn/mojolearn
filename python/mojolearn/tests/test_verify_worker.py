# SPDX-License-Identifier: Apache-2.0
"""Worker lifecycle tests. Subprocesses here execute Python only, never a GPU fit."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mojolearn import _verify_worker as worker
from mojolearn import _verify_all as verifier


def substitute_child(monkeypatch, script):
    original = subprocess.Popen
    processes = []

    def start(argv, **kwargs):
        child = original([sys.executable, "-c", script, argv[-2], argv[-1]], **kwargs)
        processes.append(child)
        return child

    monkeypatch.setattr(worker.subprocess, "Popen", start)
    return processes


def test_native_stdout_cannot_corrupt_transport(monkeypatch):
    children = substitute_child(monkeypatch, '''
import json, sys
print('native diagnostic, not JSON')
with open(sys.argv[2], 'w') as stream:
    json.dump({'value': {'train': ['abcd', None]}, 'bindings': []}, stream)
''')
    assert worker.run(dict(action="cell"), timeout=5)["value"]["train"] == ["abcd", None]
    assert children[0].poll() == 0


def test_timeout_reaps_worker_group(monkeypatch):
    children = substitute_child(monkeypatch, "import time; time.sleep(30)")
    killed = []
    if os.name == "posix":
        original = os.killpg

        def killpg(pid, signum):
            killed.append(pid)
            return original(pid, signum)

        monkeypatch.setattr(worker.os, "killpg", killpg)
    with pytest.raises(worker.WorkerFailure, match="exceeded"):
        worker.run(dict(action="cell"), timeout=0.1)
    assert children[0].poll() is not None
    if os.name == "posix":
        assert killed == [children[0].pid]


@pytest.mark.parametrize("script, message", [
    ("import sys; print('native crash'); sys.exit(7)", "exited 7"),
    ("print('successful exit without result')", "missing or invalid"),
    ("import sys; open(sys.argv[2], 'w').write('broken json')", "missing or invalid"),
    ("import sys; open(sys.argv[2], 'w').write('{\"error\": \"fixture changed\"}')", "fixture changed"),
    ("import sys; open(sys.argv[2], 'w').write('{\"value\": {}, \"bindings\": []}'); print('dead/saturated-device signature')", "unhealthy-device"),
    ("import sys; open(sys.argv[2], 'w').write('{\"value\": \"device lost\", \"bindings\": []}')", "unhealthy-device"),
])
def test_worker_failures_are_never_answers(monkeypatch, script, message):
    children = substitute_child(monkeypatch, script)
    with pytest.raises(worker.WorkerFailure, match=message):
        worker.run(dict(action="cell"), timeout=5)
    assert children[0].poll() is not None


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_timeout_must_be_bounded(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        worker.run(dict(action="cell"), timeout=timeout)


def test_checkpoint_atomic_replace(tmp_path):
    path = tmp_path / "progress.json"
    worker.atomic_json(path, dict(complete=False, completed_cells=1))
    worker.atomic_json(path, dict(complete=False, completed_cells=2))
    assert json.loads(path.read_text())["completed_cells"] == 2
    assert list(tmp_path.iterdir()) == [path]


def test_device_failure_stops_remaining_probes():
    class Rows:
        def copy(self):
            return self

    harness = SimpleNamespace(
        LANES={"test": lambda *args: object()},
        _train_hash=lambda fit: "abc",
        _probe_fit=lambda *args: (None, None, None, "dead/saturated-device signature"),
        _probe_batch=lambda *args: pytest.fail("must not run another probe on unhealthy device"),
    )
    parts = verifier.run_cell(harness, None, "test", "base", (None, None, None), Rows(), 2)
    assert parts
    assert all(value is None and "dead/saturated" in error for value, error in parts.values())
