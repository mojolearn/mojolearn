"""tools/test_timing_ledger.py -- the read-only timing report over existing evidence files (pure Python, no
device, no binding): python3 -m pytest -q tools/test_timing_ledger.py"""
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import timing_ledger as tl  # noqa: E402


def write(path, obj, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_read_tests(tmp_path):
    write(tmp_path / "run1" / "durations-1.json", {"tests": {
        "a/test_x.py::t1": {"seconds": 2.0, "outcome": "passed"},
        "a/test_x.py::t2": {"seconds": 1.0, "outcome": "failed"}}}, mtime=1000)
    rows = sorted(tl.read_tests([tmp_path]), key=lambda r: r["name"])
    assert [(r["kind"], r["name"], r["start"], r["end"]) for r in rows] == [
        ("test", "a/test_x.py::t1", 998.0, 1000.0), ("test", "a/test_x.py::t2", 999.0, 1000.0)]
    assert rows[0]["extra"]["file"] == "a/test_x.py" and tl.failed(rows[1])
    assert tl.read_tests([tmp_path / "missing"]) == []


def test_read_builds_skips_execution_receipts_and_other_json(tmp_path):
    write(tmp_path / "b" / "gpu-trees.json", {"mode": "gpu", "stem": "trees", "key": "k", "state": "built",
                                              "exit_code": 0, "seconds": 30.0, "peak_rss_bytes": 5}, mtime=500)
    write(tmp_path / "b" / "support-portable_math.json", {"support": "portable_math", "key": "k2",
                                                         "seconds": 2.0, "exit_code": 1}, mtime=500)
    write(tmp_path / "b" / "gpu-trees.log.execution.json", {"seconds": 99.0, "key": "x", "stem": "x"})
    write(tmp_path / "b" / "plan.json", {"groups": []})
    rows = {r["name"]: r for r in tl.read_builds([tmp_path])}
    assert set(rows) == {"gpu-trees", "support-portable_math"}
    assert rows["gpu-trees"]["seconds"] == 30.0 and rows["gpu-trees"]["extra"]["peak_rss_bytes"] == 5
    assert tl.failed(rows["support-portable_math"])


def test_read_release_state(tmp_path):
    state = write(tmp_path / "0.8.66" / "state.json", {"steps": {
        "cross-compile": {"at": "2026-11-07T04:58:11Z", "seconds": 2111.2, "result": "PASS"},
        "freeze-commit": {"at": "2026-11-07T04:00:00Z"}}})
    for source in (state, state.parent):
        (row,) = tl.read_release([source])
        assert row["kind"] == "step" and row["name"] == "cross-compile" and row["seconds"] == pytest.approx(2111.2)


def test_read_identity_lanes(tmp_path):
    rec = write(tmp_path / "nvidia-l40s.identical.json", {"vendor": "nvidia", "fixtures": {}, "cells": {
        "par-knn/f0": {"timing": {"total_seconds": 3.0, "stages": {"f0/train": 1.0}}},
        "par-knn/f1": {"timing": {"total_seconds": 2.0, "stages": {}}},
        "par-svm/f0": {}}})
    (row,) = tl.read_identity([rec])
    assert row["kind"] == "lane" and row["name"] == "par-knn" and row["seconds"] == pytest.approx(5.0)
    assert row["extra"]["column"] == "nvidia" and row["extra"]["train_s"] == 1.0
    assert tl.read_identity([]) == []


def test_read_jobs_from_persist_log(tmp_path):
    write(tmp_path / "box-out" / "t1" / "persist.log",
          "2026-11-07T10:00:00Z START tag=t1 prefix=p pid=1 every=600s\n"
          "2026-11-07T10:01:00Z UPLOADED name=00.tgz\n"
          "2026-11-07T10:10:00Z EXIT rc=3\n"
          "2026-11-07T11:00:00Z START tag=t1 prefix=p pid=2\n")   # a job still running: no line
    (row,) = tl.read_jobs([tmp_path])
    assert row["kind"] == "job" and row["name"] == "t1" and row["seconds"] == 600.0 and row["extra"]["rc"] == 3


def test_read_workflows(tmp_path):
    run = write(tmp_path / "run.json", {"databaseId": 42, "workflowName": "release-build", "jobs": [
        {"name": "shard 1", "startedAt": "2026-11-07T10:00:00Z", "completedAt": "2026-11-07T10:05:00Z",
         "conclusion": "success"},
        {"name": "shard 2", "startedAt": "2026-11-07T10:00:00Z", "completedAt": "2026-11-07T10:01:00Z",
         "conclusion": "failure"},
        {"name": "queued", "startedAt": None, "completedAt": None}]})
    rows = {r["name"]: r for r in tl.read_workflows([run])}
    assert rows["release-build: shard 1"]["seconds"] == 300.0 and not tl.failed(rows["release-build: shard 1"])
    assert tl.failed(rows["release-build: shard 2"]) and len(rows) == 2


def test_union_seconds_counts_parallel_once():
    assert tl.union_seconds([(0, 10), (5, 15), (20, 25)]) == 20
    assert tl.union_seconds([]) == 0


def test_parse_since():
    assert tl.parse_since("2h", now=10000) == 10000 - 7200
    with pytest.raises(ValueError):
        tl.parse_since("yesterday")


def test_summarize_kinds_top_and_filters():
    rows = [tl.unit("test", "a/test_x.py::t1", 0, 4, file="a/test_x.py", outcome="passed"),
            tl.unit("test", "a/test_x.py::t2", 0, 6, file="a/test_x.py", outcome="failed"),
            tl.unit("test", "a/test_y.py::t1", 0, 1, file="a/test_y.py"),
            tl.unit("job", "box", 100, 200, rc=0)]
    rep = tl.summarize(rows)
    assert rep["records"] == 4 and rep["sum_s"] == 111.0 and rep["wall_s"] == 106.0
    assert rep["kinds"]["test"] == dict(count=3, sum_s=11.0, wall_s=6.0, wall_share=round(6 / 106, 3))
    assert rep["top"][0]["key"] == "box"
    by_file = tl.summarize(rows, kind="test", by="file")
    assert [(i["key"], i["sum_s"], i["failed"]) for i in by_file["top"]] == [("a/test_x.py", 10.0, 1),
                                                                             ("a/test_y.py", 1.0, 0)]
    assert tl.summarize(rows, since=150)["records"] == 1
    assert len(tl.summarize(rows, top=1)["top"]) == 1


def test_report_cli(tmp_path, capsys):
    write(tmp_path / "t" / "durations-9.json", {"tests": {"t.py::a": {"seconds": 1.5, "outcome": "passed"}}})
    write(tmp_path / "j" / "persist.log", "2026-11-07T10:00:00Z START tag=box1\n2026-11-07T10:00:30Z EXIT rc=0\n")
    args = ["report", "--no-defaults", "--tests", str(tmp_path / "t"), "--jobs", str(tmp_path / "j")]
    assert tl.main(args + ["--top", "5"]) == 0
    out = capsys.readouterr().out
    assert "2 unit(s)" in out and "box1" in out and "t.py::a" in out
    assert tl.main(args + ["--kind", "test", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["records"] == 1


def test_report_cli_nothing_found_exits_1(tmp_path):
    assert tl.main(["report", "--no-defaults", "--tests", str(tmp_path / "none")]) == 1


def test_report_writes_nothing(tmp_path):
    write(tmp_path / "t" / "durations-1.json", {"tests": {"t.py::a": {"seconds": 1.0}}})
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    tl.main(["report", "--no-defaults", "--tests", str(tmp_path / "t")])
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
