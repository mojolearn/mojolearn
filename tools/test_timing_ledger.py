"""tools/test_timing_ledger.py -- the timing ledger's record, feeds and report (pure Python + bash, no device,
no binding): python3 -m pytest -q tools/test_timing_ledger.py"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import timing_ledger as tl  # noqa: E402


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("MOJOLEARN_TIMING_LEDGER", str(path))
    monkeypatch.setenv("MOJOLEARN_TIMING_COMMIT", "c" * 40)
    return path


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_record_writes_one_schema_line_with_commit_and_host(ledger):
    row = tl.record("step", "freeze-commit", 100.0, 160.5, rc=0, extra=dict(outcome="done"))
    (got,) = lines(ledger)
    assert got == row
    assert got["schema"] == tl.SCHEMA and got["kind"] == "step" and got["seconds"] == 60.5
    assert got["commit"] == "c" * 40 and got["host"] and got["cores"] >= 1
    assert got["extra"] == {"outcome": "done"}


def test_numeric_busy_becomes_a_summary(ledger):
    tl.record("job", "j", 0, 1, busy=7.5, cores=8)
    assert lines(ledger)[0]["busy"] == {"mean": 7.5, "max": 7.5, "samples": 1}


def test_record_seconds_derives_start(ledger):
    tl.record_seconds("group", "gpu-trees", 12.0, end=1000.0)
    got = lines(ledger)[0]
    assert (got["start"], got["end"], got["seconds"]) == (988.0, 1000.0, 12.0)
    assert got["extra"]["start_derived"] is True


def test_ledger_path_rules(tmp_path):
    assert tl.ledger_path({"MOJOLEARN_TIMING_LEDGER": "/x/l.jsonl"}) == Path("/x/l.jsonl")
    box = {"MOJOLEARN_BOX_PERSIST": "1", "MOJOLEARN_RESULTS_DIR": str(tmp_path)}
    assert tl.ledger_path(box) == tmp_path / "timing-ledger.jsonl"
    assert tl.ledger_path({}) == tl.DEFAULT_LEDGER
    # a tool's unit test never writes the real ledger
    assert tl.ledger_path({"PYTEST_CURRENT_TEST": "t"}) is None


def test_safe_swallows_errors(capsys):
    def boom():
        raise RuntimeError("x")
    assert tl.safe(boom) is None
    assert "not recorded" in capsys.readouterr().err


def test_ingest_tests_once_per_file_and_again_when_rewritten(ledger, tmp_path):
    run = tmp_path / "durations"
    run.mkdir()
    f = run / "durations-1.json"
    f.write_text(json.dumps({"tests": {"a/test_x.py::t1": {"seconds": 2.0, "outcome": "passed"},
                                       "a/test_x.py::t2": {"seconds": 1.0, "outcome": "failed"}}}))
    assert tl.ingest_tests(run) == 2
    assert tl.ingest_tests(run) == 0
    got = lines(ledger)
    assert {r["name"] for r in got} == {"a/test_x.py::t1", "a/test_x.py::t2"}
    assert all(r["kind"] == "test" and r["extra"]["file"] == "a/test_x.py" for r in got)
    os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 10**9))   # conftest rewrote 'latest'
    assert tl.ingest_tests(run) == 2
    assert tl.ingest_tests(run, force=True) == 2


def test_ingest_lanes_from_an_identity_record(ledger, tmp_path):
    rec = tmp_path / "nvidia-l40s.identical.json"
    rec.write_text(json.dumps({"vendor": "nvidia", "fixtures": {}, "cells": {
        "par-knn/f0": {"timing": {"total_seconds": 3.0, "stages": {"f0/train": 1.0}}},
        "par-knn/f1": {"timing": {"total_seconds": 2.0, "stages": {}}},
        "par-svm/f0": {}}}))
    assert tl.ingest_lanes([rec], column="nvidia-l40s") == 1
    (got,) = lines(ledger)
    assert got["kind"] == "lane" and got["name"] == "par-knn" and got["seconds"] == 5.0
    assert got["extra"]["column"] == "nvidia-l40s" and got["extra"]["train_s"] == 1.0


def test_ingest_workflow_jobs(ledger):
    run = {"databaseId": 42, "workflowName": "release-build", "headSha": "d" * 40, "jobs": [
        {"name": "shard 1", "startedAt": "2026-11-07T10:00:00Z", "completedAt": "2026-11-07T10:05:00Z",
         "conclusion": "success"},
        {"name": "shard 2", "startedAt": "2026-11-07T10:00:00Z", "completedAt": "2026-11-07T10:01:00Z",
         "conclusion": "failure"},
        {"name": "queued", "startedAt": None, "completedAt": None}]}
    assert tl.ingest_workflow(run) == 2
    got = {r["name"]: r for r in lines(ledger)}
    assert got["release-build: shard 1"]["seconds"] == 300.0 and got["release-build: shard 1"]["rc"] == 0
    assert got["release-build: shard 2"]["rc"] == 1 and got["release-build: shard 2"]["commit"] == "d" * 40


def test_merge_dedupes_by_id(ledger, tmp_path):
    other = tmp_path / "box" / "timing-ledger.jsonl"
    tl.record("job", "box job", 0, 10, ledger=other)
    assert tl.merge([other]) == 1
    assert tl.merge([other]) == 0
    assert len(lines(ledger)) == 1


def test_union_seconds_counts_parallel_once():
    assert tl.union_seconds([(0, 10), (5, 15), (20, 25)]) == 20
    assert tl.union_seconds([]) == 0


def test_parse_since():
    assert tl.parse_since("2h", now=10000) == 10000 - 7200
    with pytest.raises(ValueError):
        tl.parse_since("yesterday")


def test_summarize_kinds_top_and_filters(ledger):
    tl.record("test", "a/test_x.py::t1", 0, 4, extra=dict(file="a/test_x.py", outcome="passed"))
    tl.record("test", "a/test_x.py::t2", 0, 6, extra=dict(file="a/test_x.py", outcome="failed"))
    tl.record("test", "a/test_y.py::t1", 0, 1, extra=dict(file="a/test_y.py"))
    tl.record("job", "box", 100, 200, cores=48, busy=dict(mean=40.0, max=47.5, samples=3), rc=0)
    rows = tl.load([ledger])
    rep = tl.summarize(rows)
    assert rep["records"] == 4 and rep["sum_s"] == 111.0 and rep["wall_s"] == 106.0
    assert rep["kinds"]["test"] == dict(count=3, sum_s=11.0, wall_s=6.0, wall_share=round(6 / 106, 3))
    assert rep["top"][0]["key"] == "box" and rep["top"][0]["busy_max"] == 47.5 and rep["top"][0]["cores"] == 48
    by_file = tl.summarize(rows, kind="test", by="file")
    assert [(i["key"], i["sum_s"], i["failed"]) for i in by_file["top"]] == [("a/test_x.py", 10.0, 1),
                                                                             ("a/test_y.py", 1.0, 0)]
    assert tl.summarize(rows, since=150)["records"] == 1
    assert tl.summarize(rows, top=1)["top"][0]["key"] == "box"


def test_report_cli_prints_and_ingests_test_dirs(ledger, tmp_path, capsys):
    tests = tmp_path / "test-durations" / "20261107-000000"
    tests.mkdir(parents=True)
    (tests / "durations-9.json").write_text(json.dumps({"tests": {"t.py::a": {"seconds": 1.5, "outcome": "passed"}}}))
    tl.record("job", "box", 0, 30, cores=8, busy=6.0)
    assert tl.main(["report", "--tests-dir", str(tmp_path / "test-durations"), "--top", "5"]) == 0
    out = capsys.readouterr().out
    assert "2 record(s)" in out and "busy 6.0 of 8" in out and "t.py::a" in out
    assert tl.main(["report", "--kind", "test", "--json", "--tests-dir", ""]) == 0
    assert json.loads(capsys.readouterr().out)["records"] == 1


def test_report_cli_empty_ledger_exits_1(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_TIMING_LEDGER", str(tmp_path / "none.jsonl"))
    assert tl.main(["report", "--tests-dir", ""]) == 1


def test_shell_helper_writes_a_line_the_report_reads(ledger, tmp_path):
    env = dict(os.environ, MOJOLEARN_TIMING_LEDGER=str(ledger), TL_STATE_DIR=str(tmp_path / "open"),
               TL_SAMPLE_SEC="0", MOJOLEARN_TIMING_COMMIT="e" * 40)
    env.pop("PYTEST_CURRENT_TEST", None)
    sh = str(HERE / "timing_ledger.sh")
    subprocess.run(["bash", sh, "start", "compile", 'build "x" y'], env=env, check=True)
    subprocess.run(["bash", sh, "end", "compile", 'build "x" y', "3", "slot=2"], env=env, check=True)
    (got,) = tl.load([ledger])
    assert got["kind"] == "compile" and got["name"] == 'build "x" y' and got["rc"] == 3
    assert got["commit"] == "e" * 40 and got["extra"] == {"slot": "2"} and got["seconds"] >= 0
    assert got["cores"] >= 1 and got["busy"] is None
    assert not list((tmp_path / "open").iterdir())


def test_shell_helper_sourced_with_samples(ledger, tmp_path):
    state = tmp_path / "open"
    script = f"""
set -u
. {HERE / 'timing_ledger.sh'}
export TL_STATE_DIR={state} TL_SAMPLE_SEC=0
tl_start job box-tag
st=$(tl_state job box-tag); printf '40\\n80\\n' > "$st.samples"
tl_end job box-tag 0 why=exit=0
"""
    env = dict(os.environ, MOJOLEARN_TIMING_LEDGER=str(ledger))
    env.pop("PYTEST_CURRENT_TEST", None)
    subprocess.run(["bash", "-c", script], env=env, check=True)
    (got,) = tl.load([ledger])
    assert got["busy"] == {"mean": 6.0, "max": 8.0, "samples": 2}
    assert got["rc"] == 0 and got["extra"] == {"why": "exit=0"}


def test_ingest_release_state_once(ledger, tmp_path):
    state = tmp_path / "0.8.66" / "state.json"
    state.parent.mkdir()
    state.write_text(json.dumps({"commit": "f" * 40, "steps": {
        "cross-compile": {"done": True, "at": "2026-11-07T04:58:11Z", "seconds": 2111.2, "result": "PASS"},
        "freeze-commit": {"done": True, "at": "2026-11-07T04:00:00Z"}}}))
    assert tl.ingest_release(state.parent) == 1
    assert tl.ingest_release(state) == 0
    (got,) = lines(ledger)
    assert got["kind"] == "step" and got["name"] == "cross-compile" and got["seconds"] == 2111.2
    assert got["commit"] == "f" * 40 and got["extra"]["from_state"] is True
