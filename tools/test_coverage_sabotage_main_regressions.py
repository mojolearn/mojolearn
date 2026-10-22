"""Reject false evidence while retaining actual value-fault detection.

These are controller tests; no product or GPU binaries are executed.
"""
import copy
import hashlib
import json

import pytest

import coverage_sabotage as gate


ARM = {"arm": "test-arm", "lanes": ["demo"], "patch": "unit.patch", "bindings": ["build_core_host"]}


def record():
    return dict(commit="a" * 40, vendor="apple", platform="test-machine",
                mode="identical", complete=True, repeats=1,
                package={"bindings": [{"module": "_mojolearn_core_host", "sha256": "a" * 64, "size": 10}]},
                resume_signature={"source_sha256": "c" * 64, "environment_sha256": "d" * 64,
                                  "options": {"lanes": "demo", "fixtures": "base,tail", "repeats": 1}},
                fixtures={"base": {"X": "input"}, "tail": {"X": "tail-input"}},
                heldout={"base": {"X": "held"}, "tail": {"X": "tail-held"}},
                lane_revisions={"demo": 1}, batch_protocol={"alone": 16},
                stepfull_protocol={"steps": 4},
                cells={f"demo/{f}": dict(verdict="STABLE", hashes=["a" * 16],
                                         batch_verdict="STABLE", batch=["c" * 16])
                       for f in ("base", "tail")})


@pytest.fixture
def columns():
    clean = record()
    sab = copy.deepcopy(clean)
    sab["package"]["bindings"][0]["sha256"] = "b" * 64
    sab["resume_signature"]["source_sha256"] = "e" * 64
    sab["cells"]["demo/base"]["hashes"] = ["b" * 16]
    return clean, sab, copy.deepcopy(clean)


def run(tmp_path, columns, refusals=None):
    paths = []
    for i, column in enumerate(columns):
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps(column))
        paths.append(str(path))
    patch = tmp_path / "unit.patch"
    patch.write_text("retained test fault\n")
    evidence_path = None
    if refusals is not None:
        review = dict(arm=ARM["arm"], patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
                      records={key: hashlib.sha256(__import__('pathlib').Path(path).read_bytes()).hexdigest()
                               for key, path in zip(("clean", "sabotaged", "restored"), paths)},
                      refusals={key: dict(error=value["error"], product_check="test invariant",
                                          why_fault_reaches_check=value["reason"])
                                for key, value in refusals.items()})
        evidence_path = tmp_path / "review.json"
        evidence_path.write_text(json.dumps(review))
    return gate.check(ARM, *paths, evidence_path=evidence_path, root=tmp_path)


def test_changed_value_and_restoration_pass(tmp_path, columns):
    assert run(tmp_path, columns) == 0


def test_unchanged_build_does_not_count(tmp_path, columns):
    columns[1]["cells"] = copy.deepcopy(columns[0]["cells"])
    assert run(tmp_path, columns) == 1


@pytest.mark.parametrize("verdict", ["DIVERGENT", "MOVED", "REFUSED", "N/A"])
def test_bad_clean_verdict_never_passes(tmp_path, columns, verdict):
    columns[0]["cells"]["demo/base"]["verdict"] = verdict
    assert run(tmp_path, columns) == 1


@pytest.mark.parametrize("field,value", [
    ("commit", "b" * 40), ("vendor", "nvidia"), ("mode", "fast"),
    ("complete", False), ("fixtures", {}), ("heldout", {}),
    ("lane_revisions", {"demo": 2}), ("batch_protocol", {"alone": 32}),
    ("platform", "another-machine"), ("repeats", 2),
])
def test_incomparable_or_incomplete_runs_fail(tmp_path, columns, field, value):
    columns[1][field] = value
    assert run(tmp_path, columns) == 1


@pytest.mark.parametrize("index", [0, 1, 2])
def test_missing_fixture_cannot_hide_behind_one_caught_cell(tmp_path, columns, index):
    del columns[index]["cells"]["demo/tail"]
    assert run(tmp_path, columns) == 1


def test_missing_restoration_fails(tmp_path, columns):
    assert run(tmp_path, columns[:2]) == 1


def test_restoration_must_restore_bytes(tmp_path, columns):
    columns[2]["cells"]["demo/base"]["hashes"] = ["d" * 16]
    assert run(tmp_path, columns) == 1


def test_missing_binary_refusal_is_not_fault_detection(tmp_path, columns):
    error = "ImportError: missing native binding"
    columns[1]["cells"]["demo/base"] = dict(verdict="REFUSED", error=error)
    review = {"demo/base": dict(error=error, reason="mistaken approval")}
    assert run(tmp_path, columns, review) == 1


def test_fault_refusal_requires_exact_review(tmp_path, columns):
    error = "RuntimeError: select_fold_i64 disagrees with the group counts"
    columns[1]["cells"]["demo/base"] = dict(verdict="REFUSED", error=error)
    assert run(tmp_path, columns) == 1
    review = {"demo/base": dict(error=error, reason="patched labels violate the count invariant")}
    assert run(tmp_path, columns, review) == 0
    review["demo/base"]["error"] = "another failure"
    assert run(tmp_path, columns, review) == 1


def test_all_numerical_parts_are_checked(tmp_path, columns):
    for column in columns:
        column["cells"]["demo/base"].update(stepfull=["d" * 16], stepfull_verdict="STABLE")
    columns[1]["cells"]["demo/base"]["hashes"] = ["a" * 16]
    columns[1]["cells"]["demo/base"]["stepfull"] = ["e" * 16]
    assert run(tmp_path, columns) == 0
    columns[2]["cells"]["demo/base"]["stepfull"] = ["f" * 16]
    assert run(tmp_path, columns) == 1


def test_failed_clean_property_is_not_a_valid_baseline(tmp_path, columns):
    columns[0]["cells"]["demo/base"]["batch_verdict"] = "BATCH_MOVED"
    assert run(tmp_path, columns) == 1


@pytest.mark.parametrize("index", [0, 2])
def test_clean_or_restored_repeat_count_is_complete(tmp_path, columns, index):
    columns[index]["cells"]["demo/base"]["hashes"] *= 2
    assert run(tmp_path, columns) == 1


@pytest.mark.parametrize("mutation", ["missing", "refused", "malformed", "repeat"])
def test_partial_sabotaged_output_cannot_hide_behind_changed_train(tmp_path, columns, mutation):
    cell = columns[1]["cells"]["demo/base"]
    if mutation == "missing":
        del cell["batch"]
    elif mutation == "refused":
        cell["batch_verdict"] = "REFUSED"
    elif mutation == "repeat":
        cell["batch"] *= 2
    else:
        cell["batch"] = ["invalid"]
    assert run(tmp_path, columns) == 1


def test_named_inapplicable_property_is_preserved(tmp_path, columns):
    for column in columns:
        column["cells"]["demo/base"].update(model=["n/a:no-save"], model_verdict="N/A")
    assert run(tmp_path, columns) == 0
    columns[1]["cells"]["demo/base"]["model"] = ["n/a:no-longer-ran"]
    assert run(tmp_path, columns) == 1


def test_loaded_import_path_names_match_the_arm_binding(tmp_path, columns):
    # Real loaded-native-v2 records name modules by import path.
    for column in columns:
        column["package"]["bindings"][0]["module"] = "mojolearn._host._mojolearn_core_host"
    assert run(tmp_path, columns) == 0
    columns[1]["package"]["bindings"].append(
        {"module": "mojolearn._mojolearn_rf", "sha256": "f" * 64, "size": 10})
    for column in (columns[0], columns[2]):
        column["package"]["bindings"].append(
            {"module": "mojolearn._mojolearn_rf", "sha256": "9" * 64, "size": 10})
    assert run(tmp_path, columns) == 1
