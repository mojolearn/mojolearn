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
    ("import sys; print('device lost'); sys.exit(7)", "unhealthy-device"),
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


def controller(monkeypatch, tmp_path, lanes=("one", "two"), divergent=()):
    """Mock everything around cmd_verify_all except the cell scheduling."""
    from mojolearn import _backend, _verification_coverage, _verify
    reference = verifier.vref
    harness_file = tmp_path / "harness.py"
    harness_file.write_text("# fixture witness\n")
    table_file = tmp_path / "table.json"
    table_file.write_text("{}")
    table = dict(fixtures={}, heldout={}, records=[], format="test", harness_sha256="test")
    harness = SimpleNamespace(LANES={lane: None for lane in lanes}, FIXTURES=("base",),
                              fixture=lambda f: ([], [], []), heldout=lambda f: [], _h=lambda value: "hash")
    monkeypatch.setattr(_backend, "numeric_mode", lambda: "identical")
    monkeypatch.setattr(_backend, "vendor", lambda: "cpu")
    monkeypatch.setattr(_verify, "binding_artifacts", lambda: [])
    monkeypatch.setattr(verifier, "host_binding_artifacts", lambda: [])
    monkeypatch.setattr(reference, "load_table", lambda path: table)
    monkeypatch.setattr(reference, "stale_reference_lanes", lambda *args: [])
    monkeypatch.setattr(verifier, "harness_path", lambda: (str(harness_file), "test"))
    monkeypatch.setattr(verifier, "load_harness", lambda path: harness)
    monkeypatch.setattr(verifier, "select_lanes", lambda *args, **kwargs: (list(lanes), ["base"]))
    monkeypatch.setattr(verifier, "family_map", lambda lanes: {lane: "test" for lane in lanes})
    monkeypatch.setattr(verifier, "_device_block", lambda *args: {"numpy": "test"})
    monkeypatch.setattr(verifier, "verification_contract", lambda *args: {"harness_sha256": "test"})
    monkeypatch.setattr(_verification_coverage, "inventory", lambda *args: {"lanes": {}})
    monkeypatch.setattr(verifier, "host_surface", lambda: SimpleNamespace(lane_exposure=lambda *args: {}, FAMILIES=()))
    monkeypatch.setattr(verifier, "lane_accounting", lambda *args, **kwargs: {"lanes": {}})
    monkeypatch.setattr(verifier, "lane_scope", lambda *args: {"gaps": {}, "checked": len(lanes)})
    monkeypatch.setattr(verifier, "lane_scope_clause", lambda *args: "")
    monkeypatch.setattr(verifier, "reference_evidence", lambda rows: [])

    def judge(rows, *args, **kwargs):
        return [dict(row, state=reference.REFUSED if row["error"] else
                     reference.DIVERGENT if row["lane"] in divergent else reference.IDENTICAL,
                     detail=row["error"] or "", reference=None) for row in rows]

    monkeypatch.setattr(verifier, "judge_rows", judge)
    output = tmp_path / "evidence.json"

    def run_verify(**extra):
        args = SimpleNamespace(all=True, quick=False, lanes=",".join(lanes), repeats=1, no_models=True,
                               json=True, json_out=str(output), cell_timeout=1,
                               reference_table=str(table_file), **extra)
        code = verifier.cmd_verify_all(args)
        report = json.loads(output.read_text())
        progress = json.loads((tmp_path / "evidence.json.progress.json").read_text())
        journal = [json.loads(line) for line in
                   (tmp_path / "evidence.json.cells.jsonl").read_text().splitlines()]
        return code, report, progress, journal

    return run_verify


def good_value(lane):
    return {part: [lane.ljust(16, "0")[:16], None] for part in verifier.vref.PARTS}


@pytest.mark.parametrize("failure", ["timeout", "crash", "preflight", "mismatch", None])
def test_controller_checkpoints_and_stops_without_false_pass(monkeypatch, tmp_path, failure):
    """Exercise actual controller/report assembly with disposable workers mocked
    (fresh-process-per-cell mode).

    The timeout case must not dispatch the second lane, while a legitimate
    comparison mismatch must still dispatch it. Neither may return VERIFIED.
    """
    reference = verifier.vref
    run_verify = controller(monkeypatch, tmp_path, divergent=("one",) if failure == "mismatch" else ())
    monkeypatch.setattr(worker, "run_batch", lambda *args, **kwargs: pytest.fail("fresh mode must not batch"))
    calls = []

    def run(request, timeout, log):
        calls.append(request)
        if request["action"] == "self-test":
            return dict(value=dict(passed=failure != "preflight"), bindings=[])
        if failure == "timeout":
            raise worker.WorkerFailure("test worker exceeded limit")
        if failure == "crash" and request["lane"] == "one":
            raise worker.WorkerCrash("cell/one/base: worker exited -5; ")
        return dict(value={part: ["abcdef0123456789", None] for part in reference.PARTS}, bindings=[])

    monkeypatch.setattr(worker, "run", run)
    code, report, progress, journal = run_verify(fresh_process_per_cell=True)
    assert progress["complete"] is True
    assert progress["execution"]["mode"] == "fresh-process-per-cell"
    assert progress["exit"] == code == report["exit"]
    cells_dispatched = [r for r in calls if r["action"] == "cell"]
    if failure == "timeout":
        assert len(cells_dispatched) == 1
        assert code == verifier.EXIT_CANNOT_RUN
        assert {r["lane"] for r in report["cells"]} == {"one", "two"}
        assert all(r["state"] == reference.REFUSED for r in report["cells"])
    elif failure == "crash":
        # one crashed worker refuses its own cell; the sweep goes on
        assert len(cells_dispatched) == 2
        assert code == verifier.EXIT_CANNOT_RUN
        assert {r["lane"] for r in report["cells"] if r["state"] == reference.REFUSED} == {"one"}
        assert [c["lane"] for c in progress["execution"]["crashed_cells"]] == ["one"]
        assert progress["execution"]["interrupted"] is None
    elif failure == "preflight":
        assert not cells_dispatched
        assert code == verifier.EXIT_MISMATCH
    else:
        assert len(cells_dispatched) == 2
        assert code == (verifier.EXIT_MISMATCH if failure else verifier.EXIT_VERIFIED)
        assert len(journal) == 2 * len(reference.PARTS)


@pytest.mark.parametrize("message", ["Failed to create Metal command queue", "DEVIATION 2002", "GPU context is not trustworthy"])
def test_metal_health_markers(message):
    assert worker.unhealthy({"error": message})


# ---------------------------------------------------------------------------
# Batched cell workers: one process answers several cells; a crash, timeout
# or unhealthy-device signal stops the batch at the first unanswered cell.
# ---------------------------------------------------------------------------

BATCH_CHILD = '''
import json, os, sys, time
requests = json.load(open(sys.argv[1]))["batch"]
results = sys.argv[2]
def answer(i, value, log=""):
    open(os.path.join(results, "cell-%d.log" % i), "w").write(log)
    tmp = os.path.join(results, ".tmp-%d" % i)
    json.dump({"value": value, "bindings": [{"module": "m", "sha256": "s"}]}, open(tmp, "w"))
    os.replace(tmp, os.path.join(results, "result-%d.json" % i))
'''


def batch_requests(n):
    return [dict(action="cell", lane=f"lane{i}", fixture="base") for i in range(n)]


def test_batch_streams_every_result_in_order(monkeypatch):
    children = substitute_child(monkeypatch, BATCH_CHILD + '''
for i, r in enumerate(requests):
    print("native diagnostic, not JSON")
    answer(i, r["lane"])
''')
    seen = []
    stop = worker.run_batch(batch_requests(5), timeout=5, on_result=lambda i, r: seen.append((i, r["value"])))
    assert stop is None
    assert seen == [(i, f"lane{i}") for i in range(5)]
    assert len(children) == 1 and children[0].poll() == 0


@pytest.mark.parametrize("tail, kind, index", [
    ("answer(0, 'a'); answer(1, 'b'); sys.exit(9)", "crash", 2),
    ("answer(0, 'a'); open(os.path.join(results, 'cell-1.log'), 'w').write('device lost'); sys.exit(9)", "unhealthy", 1),
    ("answer(0, 'a'); answer(1, 'b', log='dead/saturated-device signature')", "unhealthy", 1),
    ("answer(0, 'device lost')", "unhealthy", 0),
    ("answer(0, 'a'); json.dump({'error': 'boom'}, open(os.path.join(results, 'result-1.json'), 'w'))", "failure", 1),
    ("answer(0, 'a')", "failure", 1),
])
def test_batch_stops_at_first_unanswered_cell(monkeypatch, tail, kind, index):
    children = substitute_child(monkeypatch, BATCH_CHILD + tail)
    seen = []
    stop = worker.run_batch(batch_requests(4), timeout=5, on_result=lambda i, r: seen.append(i))
    assert (stop.kind, stop.index) == (kind, index)
    assert seen == list(range(index))
    assert f"lane{index}" in stop.message
    assert children[0].poll() is not None


def test_batch_timeout_is_per_cell_and_reaps_worker(monkeypatch):
    # Each cell takes 0.3 s, under the 1 s limit, though the batch takes
    # longer than 1 s in total; the third cell hangs and times out.
    children = substitute_child(monkeypatch, BATCH_CHILD + '''
for i in range(2):
    time.sleep(0.3); answer(i, i)
time.sleep(0.3); answer(2, 2)
time.sleep(0.3); answer(3, 3)
time.sleep(30)
''')
    seen = []
    stop = worker.run_batch(batch_requests(6), timeout=1, on_result=lambda i, r: seen.append(i))
    assert (stop.kind, stop.index) == ("timeout", 4)
    assert seen == [0, 1, 2, 3]
    assert children[0].poll() is not None


def test_batch_worker_main_writes_one_log_and_result_per_cell(tmp_path):
    """The real worker entry point: per-cell output files, and it stops after
    the first failing cell (the controller runs the rest fresh)."""
    requests = tmp_path / "requests.json"
    results = tmp_path / "results"
    results.mkdir()
    requests.write_text(json.dumps(dict(batch=[dict(lane="a"), dict(lane="bad"), dict(lane="c")])))
    script = '''
import os, sys
import mojolearn._verify_worker as w
def execute(request):
    if request["lane"] == "bad":
        raise RuntimeError("cell failed")
    os.write(1, ("native says " + request["lane"]).encode())
    return dict(value=request["lane"], bindings=[])
w.execute = execute
w.batch_main(sys.argv[1], sys.argv[2])
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(worker.__file__)))
    subprocess.run([sys.executable, "-c", script, str(requests), str(results)], env=env, check=True,
                   timeout=60)
    assert json.loads((results / "result-0.json").read_text())["value"] == "a"
    assert (results / "cell-0.log").read_text() == "native says a"
    assert "cell failed" in json.loads((results / "result-1.json").read_text())["error"]
    assert not (results / "result-2.json").exists()


def fake_batch(answer, stops=None):
    """A run_batch stand-in: answers each request with `answer(request)` or
    stops at `stops(request)` -> kind."""
    calls = []

    def run_batch(requests, timeout, log=None, on_result=None):
        calls.append([r["lane"] for r in requests])
        for index, request in enumerate(requests):
            kind = stops(request) if stops else None
            if kind:
                return worker.BatchStop(index, kind, f"cell/{request['lane']}/base: {kind}")
            on_result(index, dict(value=answer(request), bindings=[dict(module="m", sha256="s")]))
        return None

    return run_batch, calls


def fresh_runs(fail=None):
    calls = []

    def run(request, timeout, log):
        calls.append(request.get("lane", request["action"]))
        if request["action"] == "self-test":
            return dict(value=dict(passed=True), bindings=[])
        if fail:
            fail(request)
        return dict(value=good_value(request["lane"]), bindings=[])

    return run, calls


LANES = ("one", "two", "three", "four", "five")


def test_batched_sweep_one_process_per_batch(monkeypatch, tmp_path):
    run_verify = controller(monkeypatch, tmp_path, lanes=LANES)
    batch, batch_calls = fake_batch(lambda r: good_value(r["lane"]))
    run, fresh_calls = fresh_runs()
    monkeypatch.setattr(worker, "run_batch", batch)
    monkeypatch.setattr(worker, "run", run)
    code, report, progress, journal = run_verify(batch_cells=2)
    execution = progress["execution"]
    assert code == verifier.EXIT_VERIFIED
    assert batch_calls == [["one", "two"], ["three", "four"], ["five"]]
    assert fresh_calls == ["self-test"]
    assert (execution["mode"], execution["batch_cells"], execution["batches"]) == ("batched-worker", 2, 3)
    assert (execution["batched_cells"], execution["fresh_cells"], execution["fallback_cells"]) == (5, 0, [])
    assert execution["completed_cells"] == execution["total_cells"] == 5
    # the journal is per cell and in order
    assert [row["lane"] for row in journal[::len(verifier.vref.PARTS)]] == list(LANES)
    assert len(journal) == 5 * len(verifier.vref.PARTS)
    assert report["bindings"] == [dict(module="m", sha256="s")]


def test_cell_values_do_not_depend_on_execution_mode(monkeypatch, tmp_path):
    run_verify = controller(monkeypatch, tmp_path, lanes=LANES)
    batch, _ = fake_batch(lambda r: good_value(r["lane"]))
    run, _ = fresh_runs()
    monkeypatch.setattr(worker, "run_batch", batch)
    monkeypatch.setattr(worker, "run", run)
    keep = ("lane", "fixture", "part", "value", "error", "state")
    _, batched, _, _ = run_verify(batch_cells=2)
    _, fresh, _, _ = run_verify(fresh_process_per_cell=True)
    assert [{k: r[k] for k in keep} for r in batched["cells"]] == [{k: r[k] for k in keep} for r in fresh["cells"]]
    assert batched["exit"] == fresh["exit"] == verifier.EXIT_VERIFIED


@pytest.mark.parametrize("kind", ["crash", "timeout", "unhealthy", "failure"])
def test_batch_stop_runs_remaining_cells_fresh(monkeypatch, tmp_path, kind):
    run_verify = controller(monkeypatch, tmp_path, lanes=LANES)
    batch, batch_calls = fake_batch(lambda r: good_value(r["lane"]),
                                    stops=lambda r: kind if r["lane"] == "two" else None)
    run, fresh_calls = fresh_runs()
    monkeypatch.setattr(worker, "run_batch", batch)
    monkeypatch.setattr(worker, "run", run)
    code, report, progress, journal = run_verify(batch_cells=3)
    execution = progress["execution"]
    assert code == verifier.EXIT_VERIFIED
    # batch 1 answered "one", stopped at "two"; "two" and "three" ran fresh;
    # the next batch is batched again
    assert batch_calls == [["one", "two", "three"], ["four", "five"]]
    assert fresh_calls == ["self-test", "two", "three"]
    assert execution["fallback_cells"] == ["two/base", "three/base"]
    assert [(s["lane"], s["kind"], s["index"]) for s in execution["batch_stops"]] == [("two", kind, 1)]
    assert (execution["batched_cells"], execution["fresh_cells"]) == (3, 2)
    assert execution["interrupted"] is None and execution["crashed_cells"] == []
    assert [row["lane"] for row in journal[::len(verifier.vref.PARTS)]] == ["one", "two", "three", "four", "five"]
    assert all(r["state"] == verifier.vref.IDENTICAL for r in report["cells"])


def test_fallback_crash_refuses_cell_and_streak_stops(monkeypatch, tmp_path):
    run_verify = controller(monkeypatch, tmp_path, lanes=LANES)
    batch, batch_calls = fake_batch(lambda r: good_value(r["lane"]), stops=lambda r: "crash")

    def crash(request):
        raise worker.WorkerCrash(f"cell/{request['lane']}/base: worker exited -5; ")

    run, fresh_calls = fresh_runs(crash)
    monkeypatch.setattr(worker, "run_batch", batch)
    monkeypatch.setattr(worker, "run", run)
    code, report, progress, journal = run_verify(batch_cells=24)
    execution = progress["execution"]
    assert code == verifier.EXIT_CANNOT_RUN
    assert batch_calls == [list(LANES)]
    assert fresh_calls == ["self-test", "one", "two", "three"]
    assert [c["lane"] for c in execution["crashed_cells"]] == ["one", "two", "three"]
    assert "3 consecutive cell workers crashed" in execution["interrupted"]
    assert {r["lane"] for r in report["cells"]} == set(LANES)
    assert all(r["state"] == verifier.vref.REFUSED for r in report["cells"])


def test_fallback_timeout_stops_sweep(monkeypatch, tmp_path):
    run_verify = controller(monkeypatch, tmp_path, lanes=LANES)
    batch, batch_calls = fake_batch(lambda r: good_value(r["lane"]),
                                    stops=lambda r: "timeout" if r["lane"] == "two" else None)

    def hang(request):
        raise worker.WorkerFailure(f"cell/{request['lane']}/base: exceeded 1s worker limit")

    run, fresh_calls = fresh_runs(hang)
    monkeypatch.setattr(worker, "run_batch", batch)
    monkeypatch.setattr(worker, "run", run)
    code, report, progress, journal = run_verify(batch_cells=2)
    assert code == verifier.EXIT_CANNOT_RUN
    assert batch_calls == [["one", "two"]]
    assert fresh_calls == ["self-test", "two"]
    assert "exceeded" in progress["execution"]["interrupted"]
    refused = {r["lane"] for r in report["cells"] if r["state"] == verifier.vref.REFUSED}
    assert refused == {"two", "three", "four", "five"}


@pytest.mark.parametrize("bad", [0, -2])
def test_batch_cells_must_be_positive(monkeypatch, tmp_path, bad):
    run_verify = controller(monkeypatch, tmp_path)
    run, _ = fresh_runs()
    monkeypatch.setattr(worker, "run", run)
    args = SimpleNamespace(all=True, quick=False, lanes="one,two", repeats=1, no_models=True, json=True,
                           json_out=None, cell_timeout=1, reference_table=None, batch_cells=bad)
    assert verifier.cmd_verify_all(args) == verifier.EXIT_USAGE
