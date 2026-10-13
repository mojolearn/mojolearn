"""Reuse execution evidence without turning missing or failed work into a reference."""
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("identity_columns_for_report", Path(__file__).with_name("identity_columns.py"))
columns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(columns)


def example(tmp_path):
    source = tmp_path / "harness.py"
    source.write_text("# executed test harness\n")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    h = SimpleNamespace(__file__=str(source), LANES={"lane": None}, FIXTURES=["base"],
                        LANE_REVISIONS={"lane": "revision"}, fixture=lambda f: (1, 2, 3),
                        heldout=lambda f: 4, _h=lambda value: str(value))
    parts = columns.vref().PARTS + columns.vref().OPTIONAL_PARTS
    report = dict(format="mojolearn.verify-all-report.v1", repeats=2, verdict="MISMATCH",
                  device=dict(numeric_mode="identical", device_class="nvidia", commit="a" * 40),
                  self_test=dict(passed=True), lanes=["lane"], fixtures=["base"],
                  execution=dict(completed_cells=1, total_cells=1, interrupted=None),
                  bindings=[dict(module="native", size=16, sha256="b" * 64)],
                  harness=dict(sha256=sha), verification_contract=dict(harness_sha256=sha,
                      fixtures=dict(base=dict(X="1", y_clf="2", y_reg="3")),
                      heldout=dict(base=dict(X="4")), protocols={}),
                  cells=[dict(lane="lane", fixture="base", part=p, value="c" * 16,
                              state="DIVERGENT", detail="this box differs from old reference") for p in parts])
    return report, h


def test_reference_mismatch_retains_one_observation_and_original_evidence(tmp_path):
    report, h = example(tmp_path)
    before = copy.deepcopy(report)
    result = columns.record_from_verify(report, h, "d" * 64)
    assert result["cells"]["lane/base"]["hashes"] == ["c" * 16]
    assert result["cells"]["lane/base"]["verdict"] == "STABLE"
    assert result["repeats"] == 1
    assert result["verifier_observation"]["reported_repeats"] == 2
    assert result["verifier_observation"]["report_sha256"] == "d" * 64
    assert report == before


@pytest.mark.parametrize("damage", ["incomplete", "duplicate", "missing", "harness", "input", "binding", "selftest"])
def test_unwitnessed_or_incomplete_report_is_refused(tmp_path, damage):
    report, h = example(tmp_path)
    if damage == "incomplete": report["execution"]["completed_cells"] = 0
    if damage == "duplicate": report["cells"].append(report["cells"][0])
    if damage == "missing": report["cells"].pop()
    if damage == "harness": report["harness"]["sha256"] = "0" * 64
    if damage == "input": report["verification_contract"]["fixtures"]["base"]["X"] = "other"
    if damage == "binding": report["bindings"][0]["sha256"] = "missing"
    if damage == "selftest": report["self_test"]["passed"] = False
    with pytest.raises(ValueError): columns.record_from_verify(report, h, "d" * 64)


@pytest.mark.parametrize("state,value,local", [("REFUSED", None, None),
    ("DIVERGENT", "MOVED", None), ("DIVERGENT", "c" * 16, "failed")])
def test_local_failure_is_never_admissible(tmp_path, state, value, local):
    report, h = example(tmp_path)
    report["cells"][0].update(state=state, value=value, local_check=local)
    result = columns.record_from_verify(report, h, "d" * 64)
    assert columns.vref()._part_value(result["cells"]["lane/base"], "train") is None
