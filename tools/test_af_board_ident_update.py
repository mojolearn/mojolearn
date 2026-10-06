"""Regression coverage for retained sweep reconciliation, without GPU work."""
import copy
import json
import runpy
import sys
from pathlib import Path

import af_board_render as render
import bench_board as bb


def cell(lane, median=9):
    return dict(library="mojolearn", arm="ours", mode="identical", phase="fit",
                device="gpu", lane=lane, dataset="taxi", family="algos", status="ok",
                median_ms=median, min_ms=median, max_ms=median, rounds=1, hash="old")


def test_sweep_extra_failure_and_newer_result(tmp_path, monkeypatch):
    records = {lane: dict(lane=lane, dataset="taxi", family="algos", cells=[cell(lane)])
               for lane in ("poisson", "pca", "ols")}
    records["ols"]["cells"][0]["source"] = dict(finished="2026-10-06T00:00:00Z")
    board = dict(races={"pca": records["pca"], "ols": records["ols"]},
                 extra_races={"poisson": records["poisson"]})
    (tmp_path / "board.json").write_text(json.dumps(board))
    for lane in records:
        dest = tmp_path / ("race-" + lane) / "A-1" / "res"
        dest.mkdir(parents=True)
        arm = dict(status="error") if lane == "pca" else dict(status="ok", median_ms=2, ms=[1, 3], digests=["new"])
        (dest / "result.json").write_text(json.dumps(dict(lane=lane, dataset="taxi", arms=dict(ours=arm),
                                                       finished="2026-10-05T00:00:00Z")))
    monkeypatch.setattr(render, "write_all", lambda *args: None)
    monkeypatch.setattr(render, "check", lambda *args: [])
    monkeypatch.setattr(sys, "argv", ["af_board_ident_update.py", str(tmp_path), str(tmp_path / "race-*"), "frozen"])
    runpy.run_path(str(Path(__file__).with_name("af_board_ident_update.py")), run_name="__main__")
    result = json.loads((tmp_path / "board.json").read_text())
    assert result["extra_races"]["poisson"]["cells"][0]["median_ms"] == 2
    failed = result["races"]["pca"]["cells"][0]
    assert failed["median_ms"] is None and failed["status"].startswith("stale:")
    assert failed["source"]["previous_measurement"]["median_ms"] == 9
    assert result["races"]["ols"] == board["races"]["ols"]


def test_main_board_includes_extra_without_inventing_row_count():
    board = dict(updated="2026-10-05", races={}, extra_races={
        "poisson": dict(lane="poisson", dataset="taxi", family="algos", cells=[cell("poisson")])})
    original = copy.deepcopy(board)
    text = bb.render_board(board)
    assert "### poisson / taxi (rows unrecorded" in text
    assert board == original
