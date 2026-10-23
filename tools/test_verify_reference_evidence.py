"""Inert fixtures for static coverage versus retained reference evidence."""
import json
import sys
from types import SimpleNamespace

import pytest

import verify_coverage_audit as audit_tool
from verify_reference_evidence import reference_label, reference_snapshot


def write_table(tmp_path, *, cells=None, records=None):
    path = tmp_path / "table.json"
    path.write_text(json.dumps({"cells": cells or {}, "records": records or []}))
    return path


def linked_record(column):
    return {"class": column, "commit": "a" * 40, "dir": "retained-run",
            "file": column + ".json", "vendor": column}


def test_retained_parts_keep_actual_columns_without_claiming_admission(tmp_path):
    columns = ("apple", "nvidia", "amd")
    path = write_table(tmp_path, cells={
        "existing/base": {"train": {"ref": "1" * 16, "cols": dict(zip(columns, range(3)))}},
        "partial/base": {"train": {"ref": "2" * 16, "cols": {"cpu": 3}}},
        "na/base": {"batch": {"ref": "n/a:dataset reduction"}},
    }, records=[linked_record(c) for c in (*columns, "cpu")])
    original = path.read_bytes()
    snapshot = reference_snapshot(path, ["existing", "partial", "na", "new-outside-legacy-list"])
    assert snapshot["candidate_admission"] == "not assessed"
    assert snapshot["lanes"]["existing"]["status"] == "RETAINED"
    assert snapshot["lanes"]["existing"]["fixtures"] == ["base"]
    assert snapshot["lanes"]["partial"]["status"] == "RETAINED"
    assert snapshot["lanes"]["partial"]["columns"] == ["cpu"]
    assert snapshot["lanes"]["partial"]["numeric_parts"]["base/train"]["columns"]["cpu"] == {
        "index": 3, "commit": "a" * 40, "dir": "retained-run", "file": "cpu.json", "vendor": "cpu"}
    assert snapshot["lanes"]["na"]["status"] == "OWED"
    assert snapshot["lanes"]["new-outside-legacy-list"]["status"] == "OWED"
    assert path.read_bytes() == original


@pytest.mark.parametrize("fault", ["index", "bool-index", "class", "identity", "columns", "hash"])
def test_bad_numeric_record_metadata_is_unknown_not_admitted(tmp_path, fault):
    records = [linked_record("cpu")]
    part = {"ref": "1" * 16, "cols": {"cpu": 0}}
    if fault == "index":
        part["cols"]["cpu"] = 7
    elif fault == "bool-index":
        part["cols"]["cpu"] = False
    elif fault == "class":
        records[0]["class"] = "apple"
    elif fault == "identity":
        del records[0]["commit"]
    elif fault == "columns":
        del part["cols"]
    else:
        part["ref"] = "not-a-reference"
    path = write_table(tmp_path, cells={"lane/base": {"train": part}}, records=records)
    item = reference_snapshot(path, ["lane"])["lanes"]["lane"]
    assert item["status"] == "UNKNOWN"
    assert item["issues"]


@pytest.mark.parametrize("content", [None, "not json", "[]", '{"cells": {}, "records": {}}'])
def test_unavailable_or_malformed_table_is_unknown(tmp_path, content):
    path = tmp_path / "table.json"
    if content is not None:
        path.write_text(content)
    snapshot = reference_snapshot(path, ["lane"])
    assert snapshot["table_status"] == "UNKNOWN"
    assert snapshot["lanes"]["lane"]["status"] == "UNKNOWN"


def audit_fixture(lanes):
    return {"base": "base", "head": "candidate", "lane_inventory": lanes,
            "recorded": lanes, "native": [{
                "entry": "inert_test_entry", "src": "bindings/inert_test.mojo",
                "lanes_precise": lanes[:1], "py_files": ["inert_test.py"],
                "sabotage_defines": [], "sabotage_patches": [], "changed_kernel": [],
                "new_anywhere": True, "new": True, "kinds": [],
            }], "python": [{"public": "inert.Public", "op": "predict", "where": "inert.py:1",
                            "lanes": lanes[:1]}]}


def inert_renderer(monkeypatch):
    monkeypatch.setattr(audit_tool, "_proven_arms", lambda: [])
    monkeypatch.setitem(sys.modules, "coverage_sabotage", SimpleNamespace(arms=lambda: {}))


def test_static_mapping_never_infers_recording_from_legacy_new_lane_list(tmp_path, monkeypatch):
    inert_renderer(monkeypatch)
    lanes = ["new-outside-legacy-list", "inventory-only-new-lane"]
    audit = audit_fixture(lanes)
    audit["reference_evidence"] = reference_snapshot(write_table(tmp_path), lanes)
    out = tmp_path / "audit.md"
    audit_tool.render_markdown(audit, str(out))
    report = out.read_text()
    assert "| static reach | reference evidence |" in report
    assert "| covered | OWED 1 |" in report
    assert "| OWED | 2 |" in report
    assert "- `inventory-only-new-lane`:" in report
    assert "**not assessed**" in report
    assert "is recorded unless" not in report
    assert "Each ran 9 of 9" not in report
    assert "`apply-check` passes on this commit" not in report


def test_old_report_recorded_list_has_unknown_scope_and_does_not_read_today_table(tmp_path, monkeypatch):
    inert_renderer(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("must not read current table for a retained audit")
    monkeypatch.setattr(audit_tool, "reference_snapshot", forbidden)
    audit = audit_fixture(["historically-listed-as-recorded"])
    path = tmp_path / "old.json"
    path.write_text(json.dumps(audit))
    out = tmp_path / "old.md"
    assert audit_tool.main(["--from-json", str(path), "--markdown", str(out)]) == 0
    assert "| covered | UNKNOWN 1 |" in out.read_text()
    assert audit_tool.audit_references(audit)["candidate_admission"] == "not assessed"


def test_reference_label_keeps_unknown_lane_and_empty_reach_explicit(tmp_path):
    snapshot = reference_snapshot(write_table(tmp_path), ["owed"])
    assert reference_label(["owed", "owed", "unseen"], snapshot) == "OWED 1; UNKNOWN 1"
    assert reference_label([], snapshot) == "UNKNOWN: no mapped lane"


def test_row_missing_from_retained_snapshot_is_included_as_unknown(tmp_path):
    audit = audit_fixture(["owed", "row-added-after-snapshot"])
    audit["reference_evidence"] = reference_snapshot(write_table(tmp_path), ["owed"])
    references = audit_tool.audit_references(audit)
    assert references["lanes"]["row-added-after-snapshot"]["status"] == "UNKNOWN"
    assert "row-added-after-snapshot" not in audit["reference_evidence"]["lanes"]
