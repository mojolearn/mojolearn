# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""tools/identity_timing_compare.py over tiny synthetic records (no Mojolearn import, < 1 s)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import identity_timing_compare as itc  # noqa: E402


def _cell(train, infer=0.0, batch=0.0, repeat2=None, verdict="STABLE", error=None):
    stages = {"repeat=1/train": train, "repeat=1/infer-save-reload": infer, "repeat=1/batch": batch}
    if repeat2 is not None:
        stages["repeat=2/train"] = repeat2
    cell = {"verdict": verdict, "timing": {"total_seconds": train + infer + batch + (repeat2 or 0.0), "stages": stages}}
    if error:
        cell["error"] = error
    return cell


def _record(path, cells, vendor="amd-test", commit="a" * 40, revs=None, fixtures=None):
    path.write_text(json.dumps(dict(
        vendor=vendor, commit=commit, repeats=1, cells=cells, lane_revisions=revs or {},
        fixtures=fixtures or {"base": {"X": "x0"}, "ties": {"X": "x1"}})))
    return str(path)


def test_regression_flag_stage_and_host_factor(tmp_path):
    before = _record(tmp_path / "b.json", {
        "slow/base": _cell(0.2, batch=0.1), "slow/ties": _cell(0.2, batch=0.1),
        "steady/base": _cell(1.0), "steady2/base": _cell(2.0),
    })
    after = _record(tmp_path / "a.json", {
        "slow/base": _cell(2.0, batch=0.1), "slow/ties": _cell(2.0, batch=0.1),
        "steady/base": _cell(1.0), "steady2/base": _cell(2.0),
    })
    rep = itc.compare([before], [after])
    rows = {r["lane"]: r for r in rep["rows"]}
    assert rep["rows"][0]["lane"] == "slow"
    slow = rows["slow"]
    assert slow["fixtures"] == 2
    assert abs(slow["before_s"] - 0.6) < 1e-9 and abs(slow["after_s"] - 4.2) < 1e-9
    assert slow["grew"] == "train" and slow["regressed"] and slow["outlier"]
    assert not rows["steady"]["regressed"]
    assert rep["host_factor"] == 1.0
    assert rep["regressed"] == ["slow"] and rep["improved"] == []


def test_repeat2_refused_and_error_cells_are_not_timed(tmp_path):
    before = _record(tmp_path / "b.json", {
        "x/base": _cell(1.0, repeat2=5.0),
        "r/base": _cell(0.0, verdict="REFUSED"),
        "e/base": _cell(0.0, error="ImportError: not built"),
    })
    after = _record(tmp_path / "a.json", {
        "x/base": _cell(1.0), "r/base": _cell(9.0), "e/base": _cell(9.0),
    })
    rep = itc.compare([before], [after])
    assert [r["lane"] for r in rep["rows"]] == ["x"]
    assert rep["rows"][0]["before_s"] == 1.0  # the repeat=2 stage is not summed
    assert rep["regressed"] == []


def test_changed_lane_revision_or_fixture_skips_the_lane(tmp_path):
    before = _record(tmp_path / "b.json", {"a/base": _cell(1.0), "b/base": _cell(1.0), "c/ties": _cell(1.0)},
                     revs={"a": "rows-1"})
    after = _record(tmp_path / "a.json", {"a/base": _cell(5.0), "b/base": _cell(5.0), "c/ties": _cell(5.0)},
                    revs={"a": "rows-2"}, fixtures={"base": {"X": "x0"}, "ties": {"X": "changed"}})
    rep = itc.compare([before], [after])
    assert set(rep["skipped"]) == {"a", "c"}
    assert [r["lane"] for r in rep["rows"]] == ["b"]


def test_last_record_of_a_side_supplies_the_cell_and_vendors_are_listed(tmp_path):
    old = _record(tmp_path / "old.json", {"x/base": _cell(9.0)}, vendor="nv-a")
    new = _record(tmp_path / "new.json", {"x/base": _cell(1.0)}, vendor="nv-b")
    after = _record(tmp_path / "a.json", {"x/base": _cell(1.0)}, vendor="nv-b")
    rep = itc.compare([old, new], [after])
    assert rep["rows"][0]["before_s"] == 1.0
    assert rep["before_vendors"] == ["nv-a", "nv-b"]


def test_cli_text_and_json(tmp_path, capsys):
    before = _record(tmp_path / "b.json", {"x/base": _cell(1.0)})
    after = _record(tmp_path / "a.json", {"x/base": _cell(3.0)})
    assert itc.main(["--before", before, "--after", after]) == 0
    out = capsys.readouterr().out
    assert "REGRESSED" in out and "TOTAL before" in out
    assert itc.main(["--before", before, "--after", after, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["regressed"] == ["x"]
    assert itc.main(["--before", str(tmp_path / "missing.json"), "--after", after]) == 1
