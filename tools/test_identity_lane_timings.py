# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Per-lane timings are informational: the reader sorts and totals them, and a
record admits exactly the same with or without them."""
import copy
import json
import sys
from pathlib import Path

import identity_break as ib
import identity_columns
import identity_lane_timings as lt
from test_identity_runtime import run_fixture  # noqa: F401  (pytest fixture)


def _record(vendor, cells):
    return dict(vendor=vendor, commit="a" * 40, mode="identical", complete=True,
                fixtures={"base": {}, "ties": {}}, cells=cells)


def _cell(total, train):
    return dict(verdict="STABLE", hashes=["x"],
                timing=dict(total_seconds=total, stages={"repeat=1/train": train,
                                                         "repeat=1/infer-save-reload": total - train}))


def _write(path, data):
    path.write_text(json.dumps(data))
    return str(path)


def test_reader_sorts_totals_and_counts_cells_once(tmp_path):
    nv = _write(tmp_path / "nv.identical.json", _record("nvidia-l40s", {
        "slow/base": _cell(30.0, 20.0), "slow/ties": _cell(10.0, 5.0),
        "fast/base": _cell(1.0, 0.5), "fast/ties": _cell(2.0, 1.0),
        "old/base": dict(verdict="STABLE", hashes=["x"])}))         # a record older than the cell timer
    amd = _write(tmp_path / "amd.identical.json", _record("amd-mi325x", {
        "slow/base": _cell(25.0, 15.0), "slow/ties": _cell(5.0, 2.0)}))
    # The same column's cell again (a resumed checkpoint): counted once, last input wins.
    dup = _write(tmp_path / "nv-resumed.identical.json", _record("nvidia-l40s", {"fast/base": _cell(4.0, 3.0)}))
    _write(tmp_path / "unrelated.json", {"hello": 1})

    report = lt.summarize([nv, amd, dup, str(tmp_path / "unrelated.json")], top=2)
    order = [(r["lane"], r["column"], r["wall_s"]) for r in report["lanes"]]
    assert order == [("slow", "nvidia-l40s", 40.0), ("slow", "amd-mi325x", 30.0), ("fast", "nvidia-l40s", 6.0)]
    slow = report["lanes"][0]
    assert (slow["train_s"], slow["other_s"], slow["timed_cells"]) == (25.0, 15.0, 2)
    assert report["total_wall_s"] == 76.0
    assert report["column_wall_s"] == {"amd-mi325x": 30.0, "nvidia-l40s": 46.0}
    assert report["top"] == [dict(lane="slow", column="nvidia-l40s", wall_s=40.0),
                             dict(lane="slow", column="amd-mi325x", wall_s=30.0)]
    assert report["untimed_lanes"] == [dict(lane="old", column="nvidia-l40s")]
    assert len(report["files_read"]) == 3 and len(report["files_skipped"]) == 1


def test_reader_takes_directories_verifier_reports_and_json(tmp_path, capsys):
    _write(tmp_path / "nv.identical.json", _record("nvidia-l40s", {"a/base": _cell(3.0, 1.0)}))
    sub = tmp_path / "verify"
    sub.mkdir()
    _write(sub / "report.json", dict(format=lt.VERIFY_FORMAT, device=dict(vendor="nvidia-l40s"),
                                     lane_seconds={"a": 2.5, "b": 7.0}))
    assert lt.main([str(tmp_path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [(r["lane"], r["column"], r["wall_s"]) for r in report["lanes"]] == [
        ("b", "nvidia-l40s (verify)", 7.0), ("a", "nvidia-l40s", 3.0), ("a", "nvidia-l40s (verify)", 2.5)]
    assert report["total_wall_s"] == 12.5
    assert lt.main([str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "TOTAL 12.500 s over 3 timed lane(s)" in text and "TOP 3:" in text
    assert lt.main([str(tmp_path / "nothing-here")]) == 1


def test_reader_on_a_harness_record(run_fixture):  # noqa: F811
    args, _, _ = run_fixture
    assert ib._run_reference(args) == 0
    report = lt.summarize([args.json])
    (row,) = report["lanes"]
    assert row["lane"] == "tiny" and row["column"] == "fixture-cpu" and row["timed_cells"] == 2
    assert row["wall_s"] >= row["train_s"] >= 0


def _strip_timing(value):
    if isinstance(value, dict):
        return {k: _strip_timing(v) for k, v in value.items() if k != "timing"}
    if isinstance(value, list):
        return [_strip_timing(v) for v in value]
    return value


def test_record_admits_the_same_with_and_without_timings(run_fixture, monkeypatch, tmp_path):  # noqa: F811
    args, _, _ = run_fixture
    assert ib._run_reference(args) == 0
    timed = json.loads(Path(args.json).read_text())
    assert all("timing" in c for c in timed["cells"].values())
    # One build for both copies, so the merge below may join them.
    timed.setdefault("package", {})["bindings"] = [{"module": "native", "sha256": "same"}]
    untimed = _strip_timing(copy.deepcopy(timed))
    assert untimed != timed
    a, b = tmp_path / "timed", tmp_path / "untimed"
    a.mkdir()
    b.mkdir()
    pa, pb = Path(_write(a / "col.identical.json", timed)), Path(_write(b / "col.identical.json", untimed))
    # The admission code lives in the package; load it as identity_columns does
    # (namespace package, no native bindings), not through the fixture's stand-in.
    monkeypatch.delitem(sys.modules, "mojolearn", raising=False)
    monkeypatch.delitem(sys.modules, "mojolearn._verify_reference", raising=False)
    v = identity_columns.vref()
    why_a = v.admit(timed, str(pa), known_lanes=ib.LANES)
    why_b = v.admit(untimed, str(pb), known_lanes=ib.LANES)
    norm = lambda why, d: None if why is None else str(why).replace(str(d), "<dir>")
    assert norm(why_a, a) == norm(why_b, b)
    assert v.record_device_class(timed, str(pa))[0] == v.record_device_class(untimed, str(pb))[0]
    # Merging parts ignores timings too (tools/identity_break.py merge).
    out = tmp_path / "merged.json"
    assert ib.merge([str(pa), str(pb)], str(out)) == 0
