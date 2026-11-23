"""The composed-model consistency rows (Andrew, 2026-11-23): the row-class field, who plans them, the staleness rule
(re-measure only when a module's source changed) and the module-time sum beside the row. Plans only; runs nothing."""
import glob
import importlib.util
import json
import os
import shutil

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("bench_board_composed_test", os.path.join(HERE, "bench_board.py"))
bb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bb)
N = bb.NEURAL
S = N.SEQFIT
FUSED = ("transformer-forward", "mamba1-forward", "mamba2-forward", "mamba3-forward",
         "transformer-decode", "mamba1-decode", "mamba2-decode", "mamba3-decode")


def test_row_class_marks_exactly_the_composed_rows_and_no_fused_block():
    assert set(N.ROW_CLASS) == set(S.LANES) == set(S.MODULES)
    assert set(N.ROW_CLASS.values()) == {S.ROW_CLASS}
    for lane in FUSED:
        assert lane in N.LANES and lane not in N.ROW_CLASS


def test_every_module_has_sources_and_the_files_exist():
    for lane, mods in S.MODULES.items():
        assert mods and all(m in S.SOURCES for m in mods), lane
        for rel in S.module_files(lane):
            assert os.path.isfile(os.path.join(bb.REPO, rel)), rel


@pytest.mark.parametrize("vendor", ["apple", "nvidia", "amd"])
def test_routine_board_never_plans_composed_rows(vendor):
    races = bb.plan_races(vendor, bb.modes_for(vendor), ["neural"])
    assert not {r["lane"] for r in races} & set(N.ROW_CLASS)
    assert {r["lane"] for r in races} & set(FUSED)       # the fused blocks stay ordinary rows


def test_paper_set_holds_no_composed_row():
    for path in glob.glob(os.path.join(HERE, "ops", "paper_compare_*ids*.txt")):
        text = open(path).read()
        assert not any("/%s/" % lane in text for lane in N.ROW_CLASS), path


def test_composed_plan_skips_stored_rows_until_a_module_changes(tmp_path):
    store = tmp_path / "composed_rows.jsonl"
    races = bb.plan_races("nvidia", ["identical"], ["neural"], composed=True, composed_store=str(store))
    assert {r["lane"] for r in races} == set(N.ROW_CLASS)
    assert all(r["row_class"] == S.ROW_CLASS and r["arms"][0] == "ours" for r in races)
    # every row stored with the tree's module hash: nothing to measure
    with open(store, "w") as fh:
        for lane in N.ROW_CLASS:
            fh.write(json.dumps({"lane": lane, "status": "ok",
                                 "modules_source_sha256": S.modules_source_sha256(lane, bb.REPO)}) + "\n")
    assert bb.plan_races("nvidia", ["identical"], ["neural"], composed=True, composed_store=str(store)) == []
    # a failed cell is measured again; a stale hash is measured again
    cells = [json.loads(l) for l in open(store)]
    cells[0]["status"] = "failed"
    cells[1]["modules_source_sha256"] = "0" * 64
    with open(store, "w") as fh:
        fh.write("".join(json.dumps(c) + "\n" for c in cells))
    again = bb.plan_races("nvidia", ["identical"], ["neural"], composed=True, composed_store=str(store))
    assert {r["lane"] for r in again} == {cells[0]["lane"], cells[1]["lane"]}


def test_a_changed_module_source_changes_only_the_rows_that_use_it(tmp_path):
    repo = tmp_path / "repo"
    files = sorted({f for lane in S.MODULES for f in S.module_files(lane)})
    for rel in files:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(os.path.join(bb.REPO, rel), repo / rel)
    before = {lane: S.modules_source_sha256(lane, str(repo)) for lane in S.MODULES}
    with open(repo / "autograd/kernels/mamba2_ssd_body.mojo", "a") as fh:
        fh.write("\n# changed\n")
    after = {lane: S.modules_source_sha256(lane, str(repo)) for lane in S.MODULES}
    assert {lane for lane in S.MODULES if before[lane] != after[lane]} == {"seq-mamba2-fit"}
    stored = {"status": "ok", "modules_source_sha256": before["seq-mamba2-fit"]}
    assert S.needs_measure("seq-mamba2-fit", stored, str(repo))
    assert not S.needs_measure("seq-cnn1d-fit", {"status": "ok", "modules_source_sha256": before["seq-cnn1d-fit"]},
                               str(repo))


def test_module_sum_is_na_unless_every_module_was_measured():
    lane = "seq-gru-lengths-fit"
    shown = S.composition_overhead(lane, 50.0, {"gru": 30.0, "narrow": 1.0})
    assert shown["modules_sum_ms"] == "n/a" and shown["composition_overhead_ms"] == "n/a"
    assert shown["modules_unmeasured"] == ["flatten", "linear"]
    full = {m: 2.0 for m in S.MODULES[lane]}
    shown = S.composition_overhead(lane, 50.0, full)
    assert shown["modules_sum_ms"] == 8.0 and shown["composition_overhead_ms"] == 42.0
