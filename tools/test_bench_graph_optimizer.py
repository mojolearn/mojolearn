# SPDX-License-Identifier: Apache-2.0
"""Controller-only controls; these do not qualify native kernels/performance."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import bench_graph_optimizer as graph
import bench_board_neural as board


def inspection(enabled, active=None):
    return {"optimization": "structural_pass_pipeline" if enabled else "none",
            "pass_report": {"passes": [{"name": name, "rewrites": int(name == active)}
                                        for name in graph.PASSES] if enabled else []}}


def fixture(tmp_path):
    np.savez(tmp_path / "inputs.npz", X=np.arange(6, dtype=np.float32).reshape(3, 2),
             y=np.array([0, 1, 0], np.int64))
    with np.load(tmp_path / "inputs.npz") as archive:
        arrays = {k: archive[k] for k in archive.files}
    recipe = {"schema": "mojolearn.graph-optimizer-input-recipe.v1", "shape": "full",
              "recipe": "controller-only-fixture", "artifact": {"path": "inputs.npz", **graph.file_record(tmp_path / "inputs.npz")},
              "arrays": graph.arrays_record(arrays)}
    graph.write(tmp_path / "recipe.json", recipe)
    (tmp_path / "program.mjl").write_bytes(b"controller-only-not-a-native-model")
    np.savez(tmp_path / "expected.npz", output_0=arrays["X"])
    manifest = {"schema": "mojolearn.graph-optimizer-workload.v1", "name": "controller-test",
                "coverage": "unit_test_not_performance", "model": "fake fixture", "inputs": ["X"],
                "recipe": "recipe.json", "program": "program.mjl", "expected": "expected.npz",
                "files": {p.name: graph.file_record(p) for p in tmp_path.iterdir()}}
    graph.write(tmp_path / "workload.json", manifest)
    return tmp_path / "workload.json", recipe, arrays


@pytest.mark.parametrize("enabled", [False, True])
def test_board_argument_reaches_every_native_call(monkeypatch, enabled):
    received = []
    runner = object.__new__(board.OursProgramRun)
    runner.eager = False
    runner.optimize = enabled
    runner.X = np.ones((3, 2), np.float32)
    runner.program = SimpleNamespace(run=lambda x, *, optimize: received.append((x, optimize)))
    runner.call()
    runner.call()
    assert len(received) == 2
    assert all(x is runner.X and toggle is enabled for x, toggle in received)


@pytest.mark.parametrize("enabled", [False, True])
def test_board_configuration_reaches_runner(monkeypatch, enabled):
    monkeypatch.setenv("MOJOLEARN_NUMERIC_MODE", "identical")
    calls = []
    monkeypatch.setattr(board, "OursProgramRun", lambda *args, **kwargs: calls.append(kwargs))
    board.build_runner("program-run", "ours", "full", {},
                       {"runtime": {"operation": "program_optimize", "optimize": enabled}})
    assert calls == [{"eager": False, "optimize": enabled}]


@pytest.mark.parametrize("runtime", [{"operation": "program_optimize"},
                                     {"operation": "program_optimize", "optimize": 1},
                                     {"operation": "program_eager", "optimize": True},
                                     {"optimize": False}])
def test_board_never_silently_drops_optimize(monkeypatch, runtime):
    monkeypatch.setenv("MOJOLEARN_NUMERIC_MODE", "identical")
    with pytest.raises(ValueError, match="optimize"):
        board.build_runner("program-run", "ours", "full", {}, {"runtime": runtime})


def test_complete_recipe_and_native_checkpoint_intake(tmp_path):
    path, recipe, arrays = fixture(tmp_path)
    manifest, loaded, files, values, expected = graph.load_workload(path)
    assert loaded == recipe
    assert graph.arrays_record(values) == graph.arrays_record(arrays)
    assert manifest["inputs"] == ["X"]
    assert expected == graph.outputs_record(arrays["X"])
    assert set(files) == {"inputs.npz", "recipe.json", "program.mjl", "expected.npz"}


@pytest.mark.parametrize("corrupt", ["last_word", "shape", "dtype", "unconsumed_labels", "reduced_shape"])
def test_receipt_corruption_and_reduced_inputs_refused(tmp_path, corrupt):
    _, recipe, arrays = fixture(tmp_path)
    if corrupt == "last_word":
        arrays["X"].flat[-1] += 1
    elif corrupt == "shape":
        arrays["X"] = arrays["X"].reshape(2, 3)
    elif corrupt == "dtype":
        arrays["X"] = arrays["X"].astype(np.float64)
    elif corrupt == "unconsumed_labels":
        arrays["y"][-1] = 1
    else:
        recipe["shape"] = "small"
    np.savez(tmp_path / "inputs.npz", **arrays)
    # Updating the outer archive checksum must not bypass individual full array
    # identity. In particular labels remain part of the immutable dataset.
    recipe["artifact"].update(graph.file_record(tmp_path / "inputs.npz"))
    graph.write(tmp_path / "recipe.json", recipe)
    with pytest.raises(ValueError, match="inventory|full-workload"):
        graph.read_recipe(tmp_path / "recipe.json")


def test_native_readback_and_zero_hits_are_distinct():
    assert not any(graph.pass_coverage(inspection(True), True).values())
    assert graph.pass_coverage(inspection(True, graph.PASSES[0]), True)[graph.PASSES[0]] == 1
    with pytest.raises(ValueError, match="readback"):
        graph.pass_coverage(inspection(False), True)
    missing = inspection(True)
    missing["pass_report"]["passes"].pop()
    with pytest.raises(ValueError, match="pipeline"):
        graph.pass_coverage(missing, True)


@pytest.mark.parametrize("mutate", [False, True])
def test_actual_loaded_native_bytes_must_match_build_receipt(tmp_path, mutate):
    path = tmp_path / "binary.json"
    graph.write(path, {"source_commit": "retained-source", "binaries": [
        {"path": "/build/_mojolearn_x_sequence.so", "sha256": "a" * 64}]})
    runtime = {"native_extensions": {"mojolearn._mojolearn_x_sequence": {
        "path": "/installed/_mojolearn_x_sequence.so", "sha256": ("b" if mutate else "a") * 64}}}
    if mutate:
        with pytest.raises(ValueError, match="native closure"):
            graph.check_binary_provenance(path, runtime)
    else:
        assert graph.check_binary_provenance(path, runtime)["source_commit"] == "retained-source"


def test_whole_call_timer_consumes_every_output(monkeypatch):
    calls = []
    value = (np.array([-0.0, 2], np.float32), np.array([1, 2, 3], np.int32))
    def run(*inputs, optimize):
        calls.append((inputs, optimize))
        return value
    clock = iter([10.0, 10.25, 11.0])
    result, measured = graph.timed_call(SimpleNamespace(run=run), ("full-input",), True, lambda: next(clock))
    assert result is value
    assert calls == [(("full-input",), True)]
    assert measured["native_call_ms"] == 250
    assert measured["output_consumption_ms"] == 750
    assert measured["whole_operation_ms"] == 1000
    altered = (value[0], value[1].copy())
    altered[1][-1] += 1
    with pytest.raises(ValueError, match="words"):
        graph.require_equal(measured["outputs"], graph.outputs_record(altered), "last output")


def report():
    output = graph.outputs_record(np.arange(8, dtype=np.float32))
    result = {"status": "validation_only", "backend": {"vendor": "cpu", "numeric_mode": 1},
              "workload": "fixture", "recipe": "full fixture", "expected_outputs": output,
              "active_passes": [], "samples_per_arm": 1, "arms": {}}
    for arm, optimize in (("baseline", False), ("candidate", True)):
        result["arms"][arm] = {"status": "passed", "requested_optimize": optimize,
            "inspection": inspection(optimize), "pass_rewrites": dict.fromkeys(graph.PASSES, 0),
            "native_audit": {"passed": True}, "cold": {"outputs": output},
            "excluded_warmup": {"outputs": output}, "rounds": [{"outputs": output}]}
    return result


@pytest.mark.parametrize("change", ["none", "digest", "mode", "toggle", "missing_round", "audit", "failed"])
def test_comparator_positive_and_deliberate_failures(tmp_path, change):
    original = report()
    modified = copy.deepcopy(original)
    if change == "digest":
        modified["arms"]["candidate"]["rounds"][0]["outputs"]["output_0"]["sha256"] = "0" * 64
    elif change == "mode":
        modified["backend"]["numeric_mode"] = 0
    elif change == "toggle":
        modified["arms"]["candidate"]["requested_optimize"] = False
    elif change == "missing_round":
        modified["arms"]["baseline"]["rounds"] = []
    elif change == "audit":
        modified["arms"]["baseline"]["native_audit"]["python_frames"] = ["callback"]
    elif change == "failed":
        modified["status"] = "failed"
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, data in zip(paths, (original, modified)):
        graph.write(path, data)
    if change == "none":
        assert graph.compare(paths)["status"] == "complete_output_words_match"
    else:
        with pytest.raises(ValueError):
            graph.compare(paths)


def test_quality_uses_all_rows_and_refuses_short_output():
    values = np.array([[.8, .2], [.3, .7], [.6, .4]], np.float32)
    arrays = {"labels": np.array([0, 1, 1])}
    config = {"kind": "softmax", "labels": "labels"}
    actual = graph.quality(values, arrays, config)
    assert actual["rows"] == 3
    assert actual["accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="every returned row"):
        graph.quality(values[:2], arrays, config)
