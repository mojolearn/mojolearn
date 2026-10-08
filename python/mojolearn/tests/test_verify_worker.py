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


@pytest.mark.parametrize("failure", ["timeout", "preflight", "mismatch", None])
def test_controller_checkpoints_and_stops_without_false_pass(monkeypatch, tmp_path, failure):
    """Exercise actual controller/report assembly with disposable workers mocked.

    The timeout case must not dispatch the second lane, while a legitimate
    comparison mismatch must still dispatch it. Neither may return VERIFIED.
    """
    from mojolearn import _backend, _verification_coverage, _verify
    reference = verifier.vref
    harness_file = tmp_path / "harness.py"
    harness_file.write_text("# fixture witness\n")
    table_file = tmp_path / "table.json"
    table_file.write_text("{}")
    output = tmp_path / "evidence.json"
    table = dict(fixtures={}, heldout={}, records=[], format="test", harness_sha256="test")
    harness = SimpleNamespace(LANES={"one": None, "two": None}, FIXTURES=("base",),
                              fixture=lambda f: ([], [], []), heldout=lambda f: [], _h=lambda value: "hash")
    monkeypatch.setattr(_backend, "numeric_mode", lambda: "identical")
    monkeypatch.setattr(_backend, "vendor", lambda: "cpu")
    monkeypatch.setattr(_verify, "binding_artifacts", lambda: [])
    monkeypatch.setattr(verifier, "host_binding_artifacts", lambda: [])
    monkeypatch.setattr(reference, "load_table", lambda path: table)
    monkeypatch.setattr(reference, "stale_reference_lanes", lambda *args: [])
    monkeypatch.setattr(verifier, "harness_path", lambda: (str(harness_file), "test"))
    monkeypatch.setattr(verifier, "load_harness", lambda path: harness)
    monkeypatch.setattr(verifier, "select_lanes", lambda *args, **kwargs: (["one", "two"], ["base"]))
    monkeypatch.setattr(verifier, "family_map", lambda lanes: {lane: "test" for lane in lanes})
    monkeypatch.setattr(verifier, "_device_block", lambda *args: {"numpy": "test"})
    monkeypatch.setattr(verifier, "verification_contract", lambda *args: {"harness_sha256": "test"})
    monkeypatch.setattr(_verification_coverage, "inventory", lambda *args: {"lanes": {}})
    monkeypatch.setattr(verifier, "host_surface", lambda: SimpleNamespace(lane_exposure=lambda *args: {}, FAMILIES=()))
    monkeypatch.setattr(verifier, "lane_accounting", lambda *args, **kwargs: {"lanes": {}})
    monkeypatch.setattr(verifier, "lane_scope", lambda *args: {"gaps": {}, "checked": 2})
    monkeypatch.setattr(verifier, "lane_scope_clause", lambda *args: "")
    monkeypatch.setattr(verifier, "reference_evidence", lambda rows: [])

    def judge(rows, *args, **kwargs):
        return [dict(row, state=reference.REFUSED if row["error"] else
                     reference.DIVERGENT if failure == "mismatch" and row["lane"] == "one" else reference.IDENTICAL,
                     detail=row["error"] or "", reference=None) for row in rows]

    monkeypatch.setattr(verifier, "judge_rows", judge)
    calls = []

    def run(request, timeout, log):
        calls.append(request)
        if request["action"] == "self-test":
            return dict(value=dict(passed=failure != "preflight"), bindings=[])
        if failure == "timeout":
            raise worker.WorkerFailure("test worker exceeded limit")
        return dict(value={part: ["abcdef0123456789", None] for part in reference.PARTS}, bindings=[])

    monkeypatch.setattr(worker, "run", run)
    args = SimpleNamespace(all=True, quick=False, lanes="one,two", repeats=1, no_models=True,
                           json=True, json_out=str(output), cell_timeout=1, reference_table=str(table_file))
    code = verifier.cmd_verify_all(args)
    report = json.loads(output.read_text())
    progress = json.loads((tmp_path / "evidence.json.progress.json").read_text())
    assert progress["complete"] is True
    assert progress["exit"] == code == report["exit"]
    cells_dispatched = [r for r in calls if r["action"] == "cell"]
    if failure == "timeout":
        assert len(cells_dispatched) == 1
        assert code == verifier.EXIT_CANNOT_RUN
        assert {r["lane"] for r in report["cells"]} == {"one", "two"}
        assert all(r["state"] == reference.REFUSED for r in report["cells"])
    elif failure == "preflight":
        assert not cells_dispatched
        assert code == verifier.EXIT_MISMATCH
    else:
        assert len(cells_dispatched) == 2
        assert code == (verifier.EXIT_MISMATCH if failure else verifier.EXIT_VERIFIED)
        journal = (tmp_path / "evidence.json.cells.jsonl").read_text().splitlines()
        assert len(journal) == 2 * len(reference.PARTS)


@pytest.mark.parametrize("message", ["Failed to create Metal command queue", "DEVIATION 2002", "GPU context is not trustworthy"])
def test_metal_health_markers(message):
    assert worker.unhealthy({"error": message})
