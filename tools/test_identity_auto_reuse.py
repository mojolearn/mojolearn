# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""REUSE GOOD OLD RUNS, AUTOMATICALLY (Andrew 2026-10-31).

`record_identity_column.sh` with no LANES/SHARD/ONLY_CHANGED/FULL records only
the lanes whose closure changed since THIS device class's newest good evidence;
unchanged lanes are reused, FULL=1 records every lane. Admission accepts older
incoming records whose lanes are superseded or whose closure is unchanged.
Nothing here imports a binding or runs a fit: the harness, git and the recorder's
interpreter are stand-ins.
"""
import json
import os
import stat
import subprocess
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import identity_columns as ic  # noqa: E402
import lane_select  # noqa: E402

FIX = ("base", "ties")
LANES = ["lane-a", "lane-b", "lane-c", "lane-d"]
TIMES = {"c1" * 20: 100, "c2" * 20: 200, "c3" * 20: 300}


def _harness():
    return types.SimpleNamespace(LANES=list(LANES), FIXTURES=FIX, LANE_REVISIONS={},
                                 _h=lambda x: "h", fixture=lambda f: (1, 2, 3), heldout=lambda f: 4)


def _record(path, cls, commit, lanes, bad=()):
    cells = {}
    for lane in lanes:
        for f in FIX:
            verdict = "DIVERGENT" if lane in bad else "STABLE"
            cells[f"{lane}/{f}"] = {"verdict": verdict, "hashes": ["0" * 16], "infer_verdict": "N/A"}
    doc = dict(mode="identical", complete=True, vendor=cls, commit=commit, cells=cells,
               fixtures={f: dict(X="h", y_clf="h", y_reg="h") for f in FIX},
               heldout={f: dict(X="h") for f in FIX}, lane_revisions={})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture
def world(tmp_path, monkeypatch):
    v = types.SimpleNamespace(
        record_device_class=lambda j, p: (j.get("vendor"), None),
        admit=lambda j, p, known_lanes=(): None,
        stale_reference_lanes=lambda t, h: [])
    monkeypatch.setattr(ic, "harness", _harness)
    monkeypatch.setattr(ic, "vref", lambda: v)
    monkeypatch.setattr(ic, "profile_lanes", lambda h, scope: list(LANES))
    monkeypatch.setattr(ic, "_commit_time", lambda c, cache: TIMES.get(c))
    monkeypatch.setattr(lane_select, "lane_sources", lambda: ({l: [] for l in LANES}, {}))
    state = {"changed": {}, "diffs": []}

    def changed_paths(ref):
        state["diffs"].append(ref)
        return ["x.mojo"]

    def select(paths, ref=None, sources=None, backend=None, stop_when=None):
        return {"lanes": sorted(state["changed"].get(ref, ())), "unattributed": [], "stopped_early": False}

    monkeypatch.setattr(lane_select, "changed_paths", changed_paths)
    monkeypatch.setattr(lane_select, "select", select)
    table = tmp_path / "table.json"
    table.write_text(json.dumps({"cells": {}, "records": []}))
    return tmp_path, table, state


def _args(table, records, out=""):
    return types.SimpleNamespace(table=str(table), since="", scope="routine", no_release_floor=True,
                                 backend="cuda", floor_desc="", cls="nvidia", records=[str(r) for r in records],
                                 summary_json=str(out) if out else "")


def test_unchanged_lane_is_reused_changed_lane_is_recorded(world, capsys):
    tmp, table, state = world
    recs = tmp / "records"
    _record(recs / "old" / "nvidia-x.identical.json", "nvidia", "c1" * 20, ["lane-a", "lane-b", "lane-c"])
    _record(recs / "amd" / "amd-x.identical.json", "amd", "c3" * 20, LANES)  # another class: never reused here
    state["changed"]["c1" * 20] = {"lane-b"}
    summary = tmp / "reuse.json"
    assert ic.cmd_changed_lanes(_args(table, [recs], summary)) == 0
    out, err = capsys.readouterr()
    # lane-b's closure moved; lane-d has no nvidia evidence at all
    assert out.strip().split(",") == ["lane-b", "lane-d"]
    assert "REUSED 2 RECORD 2 KNOWN-BAD 0" in err
    doc = json.loads(summary.read_text())
    assert sorted(doc["reused"]) == ["lane-a", "lane-c"]
    assert doc["reused"]["lane-a"]["commit"] == "c1" * 20
    assert state["diffs"] == ["c1" * 20]


def test_newest_evidence_of_the_class_wins(world, capsys):
    tmp, table, state = world
    recs = tmp / "records"
    _record(recs / "a.identical.json", "nvidia", "c1" * 20, LANES)
    _record(recs / "b" / "nvidia-inc.identical.json", "nvidia", "c2" * 20, ["lane-b"])
    # lane-b moved between c1 and c2, then was re-recorded at c2: reused from c2
    state["changed"]["c1" * 20] = {"lane-b"}
    assert ic.cmd_changed_lanes(_args(table, [recs])) == 0
    out, err = capsys.readouterr()
    assert out.strip() == ""
    assert "REUSED 4 RECORD 0" in err


def test_bad_result_with_unchanged_closure_is_known_bad_not_rerun(world, capsys):
    tmp, table, state = world
    recs = tmp / "records"
    _record(recs / "a.identical.json", "nvidia", "c1" * 20, LANES, bad={"lane-c"})
    assert ic.cmd_changed_lanes(_args(table, [recs])) == 0
    out, err = capsys.readouterr()
    assert out.strip() == ""
    assert "KNOWN-BAD 1" in err and "known-bad lane-c" in err
    # once its code changes, the bad lane is recorded again
    state["changed"]["c1" * 20] = {"lane-c"}
    assert ic.cmd_changed_lanes(_args(table, [recs])) == 0
    assert capsys.readouterr()[0].strip() == "lane-c"


def test_table_rows_of_the_class_count_as_evidence(world, capsys):
    tmp, table, state = world
    cells = {f"{lane}/{f}": {"train": {"ref": "f" * 16, "cols": {"nvidia": 0, "amd": 1}}}
             for lane in ("lane-a", "lane-b") for f in FIX}
    cells["lane-c/base"] = {"train": {"ref": "f" * 16, "cols": {"amd": 1}}}       # no nvidia column
    cells["lane-c/ties"] = {"train": {"ref": "f" * 16, "cols": {"amd": 1}}}
    table.write_text(json.dumps({"cells": cells, "records": [
        {"commit": "c2" * 20, "commit_time": 200, "class": "nvidia"},
        {"commit": "c1" * 20, "commit_time": 100, "class": "amd"}]}))
    assert ic.cmd_changed_lanes(_args(table, [])) == 0
    out, err = capsys.readouterr()
    assert out.strip().split(",") == ["lane-c", "lane-d"]
    assert "REUSED 2 RECORD 2" in err


def test_reuse_check_admits_superseded_and_unchanged_refuses_changed(world, capsys):
    tmp, table, state = world
    old = _record(tmp / "in" / "nvidia-old.identical.json", "nvidia", "c1" * 20, LANES)
    new = _record(tmp / "in" / "nvidia-new.identical.json", "nvidia", "c2" * 20, ["lane-b"])
    amd = _record(tmp / "in" / "amd-new.identical.json", "amd", "c2" * 20, LANES)
    # lane-b moved, but the newer nvidia record supersedes the old row
    state["changed"]["c1" * 20] = {"lane-b"}
    args = types.SimpleNamespace(commit="c2" * 20, files=[str(old), str(new), str(amd)])
    assert ic.cmd_reuse_check(args) == 0
    assert "ok=yes" in capsys.readouterr()[0]
    # lane-a moved too and nothing newer carries it: refused
    state["changed"]["c1" * 20] = {"lane-a", "lane-b"}
    assert ic.cmd_reuse_check(args) == 1
    out = capsys.readouterr()[0]
    assert "REUSE-REFUSED" in out and "lane-a" in out and "ok=NO" in out


# ------------------------------------------------ the recorder's lane choice

FAKE_PY = r'''#!/bin/bash
# stand-in interpreter for record_identity_column.sh: no binding, no fit
echo "$*" >> "$FAKE_LOG"
case "$*" in
  *"import mojolearn"*) echo "cpu identical"; exit 0 ;;
  "-c "*|"-") exec python3 "$@" ;;
  *"identity_columns.py changed-lanes"*)
      prev=""; for a in "$@"; do [ "$prev" = --summary-json ] && echo "$FAKE_SUMMARY_JSON" > "$a"; prev=$a; done
      echo "# REUSE-SUMMARY class=cpu $FAKE_SUMMARY of 3 (routine)" >&2
      echo "# CHANGED-LANES" >&2
      echo "$FAKE_CHANGED"; exit 0 ;;
  *"identity_columns.py lanes"*) echo "lane-a,lane-b,lane-c"; exit 0 ;;
  *"identity_columns.py check"*) echo "IDCOLUMN x admissible=yes"; exit 0 ;;
  *"retain_run_summary.py"*) exit 0 ;;
  *"identity_break.py"*)
      prev=""; for a in "$@"; do [ "$prev" = --json ] && echo '{"complete": true, "commit": "x"}' > "$a"; prev=$a; done
      exit 0 ;;
esac
echo "unexpected: $*" >&2; exit 9
'''


def _run_recorder(tmp_path, changed, summary="REUSED 2 RECORD 1 KNOWN-BAD 0", **env):
    fake = tmp_path / "fakepy"
    fake.write_text(FAKE_PY)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "py.log"
    out = tmp_path / "out"
    full_env = dict(os.environ, PYTHON=str(fake), FAKE_LOG=str(log), FAKE_CHANGED=changed,
                    FAKE_SUMMARY=summary, ALLOW_DIRTY="1",
                    FAKE_SUMMARY_JSON=json.dumps({"reused": {"lane-a": {"source": "table", "committed": True}}}))
    for k in ("LANES", "SHARD", "ONLY_CHANGED", "FULL", "MOJOLEARN_IDENTITY_LAUNCH",
              "MOJOLEARN_IDENTITY_REUSE_RECORDS"):
        full_env.pop(k, None)
    full_env.update(env)
    res = subprocess.run(["bash", os.path.join(ROOT, "tools", "record_identity_column.sh"), "cpu", str(out)],
                         env=full_env, capture_output=True, text=True, timeout=120)
    calls = log.read_text().splitlines() if log.exists() else []
    harness = [c for c in calls if "identity_break.py" in c]
    return res, calls, harness


def _lanes_of(call):
    words = call.split()
    return words[words.index("--lanes") + 1]


def test_recorder_default_records_only_changed_lanes(tmp_path):
    res, calls, harness = _run_recorder(tmp_path, "lane-b")
    assert res.returncode == 0, res.stderr[-800:]
    assert "REUSED 2 / RECORD 1 cpu" in res.stdout
    assert any("changed-lanes" in c and "--class cpu" in c for c in calls)
    assert len(harness) == 1 and _lanes_of(harness[0]) == "lane-b"


def test_recorder_nothing_owed_runs_nothing(tmp_path):
    res, calls, harness = _run_recorder(tmp_path, "", summary="REUSED 3 RECORD 0 KNOWN-BAD 0")
    assert res.returncode == 0, res.stderr[-800:]
    assert "NOTHING-CHANGED cpu" in res.stdout
    assert harness == []


def test_recorder_extra_reuse_dirs_reach_the_selection(tmp_path):
    extra = tmp_path / "old-records"
    extra.mkdir()
    res, calls, _ = _run_recorder(tmp_path, "lane-b", MOJOLEARN_IDENTITY_REUSE_RECORDS=str(extra))
    assert res.returncode == 0, res.stderr[-800:]
    assert any(f"--records {extra}" in c for c in calls if "changed-lanes" in c)


def test_recorder_full_records_every_lane(tmp_path):
    res, calls, harness = _run_recorder(tmp_path, "lane-b", FULL="1")
    assert res.returncode == 0, res.stderr[-800:]
    assert not any("changed-lanes" in c for c in calls)
    assert len(harness) == 1 and _lanes_of(harness[0]) == "lane-a,lane-b,lane-c"


def test_recorder_full_refuses_only_changed(tmp_path):
    res, _, harness = _run_recorder(tmp_path, "lane-b", FULL="1", ONLY_CHANGED="table")
    assert res.returncode == 2 and "FULL=1 excludes" in res.stderr and harness == []


def test_recorder_only_changed_table_keeps_its_old_meaning(tmp_path):
    res, calls, harness = _run_recorder(tmp_path, "lane-c", ONLY_CHANGED="table")
    assert res.returncode == 0, res.stderr[-800:]
    sel = [c for c in calls if "changed-lanes" in c]
    assert len(sel) == 1 and "--class" not in sel[0]
    assert _lanes_of(harness[0]) == "lane-c"


def test_recorder_explicit_lanes_are_not_narrowed(tmp_path):
    res, calls, harness = _run_recorder(tmp_path, "", LANES="lane-a,lane-c")
    assert res.returncode == 0, res.stderr[-800:]
    assert not any("changed-lanes" in c for c in calls)
    assert _lanes_of(harness[0]) == "lane-a,lane-c"
