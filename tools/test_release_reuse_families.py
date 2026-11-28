# SPDX-License-Identifier: Apache-2.0
"""tools/release_families.py and its use by tools/release_reuse.py (AGENTS.md part 4, "A release rebuilds and
re-records only the families that changed", Andrew 2026-11-28): the map covers every binding, the plan is whole per
family, merged files tie families, shared reaches every lane, and a kept family's columns must rest on the shipped
bytes. Synthetic plans, tables and records only: nothing is built, archived or recorded."""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release_families as RF  # noqa: E402
import release_reuse as RR  # noqa: E402


def fam(lanes=None, rules=(), families=None):
    doc = dict(RF.load().doc)
    if families is not None:
        doc["families"] = families
    doc.update(lanes=dict(lanes or {}), lane_rules=[list(r) for r in rules])
    return RF.Families(doc)


def row(name, decision="REUSE", target=RR.LINUX, vendor="cuda", arch="sm_89", tier="identical", reason=None):
    key = "/".join((target, vendor, arch or "-", tier, name))
    return dict(key=key, target=target, vendor=vendor, arch=arch, tier=tier, name=name, decision=decision,
                set_rel="%s/%s.so" % (tier, name),
                reason=reason or ("identity unchanged" if decision == "REUSE" else "changed: closure (gbdt/boost.mojo)"),
                identity_digest="d-" + name, reuse_from=dict(version="0.8.67") if decision == "REUSE" else None)


def plan_of(rows, runtime="REUSE"):
    return dict(rows=rows, runtime=dict(decision=runtime, reason="identity unchanged" if runtime == "REUSE"
                                        else "changed: toolchain"))


NO_MERGED = {"dirs": {}}


# ---------------------------------------------------------------- the map
def test_every_binding_the_release_and_the_builds_know_has_exactly_one_family():
    f = RF.load()
    names = RF.known_bindings()
    assert names and RF.check(f, names, []) == []
    assert {f.binding(n) for n in names} <= set(f.families)
    # a host binding sits with its device twin, the Metal 1 build with its binding
    assert f.binding("_mojolearn_gbdt") == f.binding("_mojolearn_gbdt_host") == "trees"
    assert f.binding("_mojolearn_estimators_m1") == f.binding("_mojolearn_estimators")


def test_the_map_refuses_a_binding_in_two_families_an_unknown_family_and_no_shared():
    doc = dict(RF.load().doc)
    with pytest.raises(SystemExit, match="in two families"):
        RF.Families(dict(doc, families=dict(doc["families"], extra=["_mojolearn_gbdt"])))
    with pytest.raises(SystemExit, match="unknown family"):
        RF.Families(dict(doc, lanes={"knn": "nope"}))
    with pytest.raises(SystemExit, match="no 'shared' family"):
        RF.Families(dict(doc, families={k: v for k, v in doc["families"].items() if k != "shared"}))


def test_check_refuses_a_binding_or_a_lane_with_no_family_and_a_lane_loading_another_family():
    f = fam(lanes={"gbdt-rmse": "trees", "knn": "cluster"})
    problems = RF.check(f, ["_mojolearn_gbdt", "_mojolearn_nope"], ["gbdt-rmse", "orphan"],
                        {"gbdt-rmse": ["_mojolearn_gbdt", "_mojolearn"], "knn": ["_mojolearn_x_neighbors",
                                                                                "_mojolearn_gbdt"]})
    assert any("binding _mojolearn_nope has no family" in p for p in problems)
    assert any("lane orphan has no family" in p for p in problems)
    assert any(p.startswith("lane knn is in family cluster but loads _mojolearn_gbdt (trees)") for p in problems)
    assert not any("gbdt-rmse" in p for p in problems)            # its own family and shared only
    # the committed rule `*` -> shared until derive --write: every lane has a family, every lane is shared
    assert RF.load().lane("anything") == "shared"


def test_derive_gives_a_lane_its_one_family_else_shared():
    lanes, problems = RF.derive(fam(), {"gbdt-rmse": ["_mojolearn_gbdt", "_mojolearn_gbdt_host", "_mojolearn"],
                                        "pipeline": ["_mojolearn_gbdt", "_mojolearn_x_prep"],
                                        "core-only": ["_mojolearn"], "odd": ["_mojolearn_unknown"]})
    assert lanes == {"gbdt-rmse": "trees", "pipeline": "shared", "core-only": "shared", "odd": "shared"}
    assert problems == ["lane odd loads binding(s) with no family: _mojolearn_unknown"]


# ---------------------------------------------------------------- the plan, whole per family
def test_one_changed_binding_builds_its_family_whole_on_every_target_and_nothing_else():
    rows = [row("_mojolearn_gbdt", "BUILD"), row("_mojolearn_rf"), row("_mojolearn_rf", vendor="hip", arch="gfx942"),
            row("_mojolearn_rf", target=RR.MACOS, vendor="metal", arch="apple"),
            row("_mojolearn_gbdt_host", vendor="host", arch="", tier="host"),
            row("_mojolearn_svm"), row("_mojolearn_training")]
    p = plan_of(rows)
    out = RF.apply(p, fam(), merged=NO_MERGED)
    by = RF.by_name(out)
    assert by["trees"]["decision"] == "BUILD" and "_mojolearn_gbdt" in by["trees"]["reason"]
    assert "changed: closure (gbdt/boost.mojo)" in by["trees"]["reason"]     # which file's digest changed
    assert by["trees"]["record"] and by["trees"]["built"] == 1 and by["trees"]["bindings"] == 5
    assert all(r["decision"] == "BUILD" for r in rows if r["family"] == "trees")
    assert all(r["reuse_from"] is None for r in rows if r["family"] == "trees")
    assert any("family trees builds whole" in r["reason"] for r in rows if r["name"] == "_mojolearn_rf")
    assert by["kernel"]["decision"] == "REUSE" and not by["kernel"]["record"]
    assert by["neural"]["decision"] == "REUSE" and not by["neural"]["record"]
    # shared's lanes load several families: they record whenever one family records
    assert by["shared"]["decision"] == "REUSE" and by["shared"]["record"]
    assert p["families"] is out and out["schema"] == RF.PLAN_SCHEMA


def test_a_merged_file_ties_the_families_it_serves():
    merged = {"dirs": {"identical": {"path": "identical", "groups": {
        "_mojolearn_native_trees": ["_mojolearn_gbdt", "_mojolearn_svm"],
        "_mojolearn_native_sequence": ["_mojolearn_training"]}, "optional": {}}}}
    rows = [row("_mojolearn_gbdt", "BUILD"), row("_mojolearn_svm"), row("_mojolearn_training")]
    by = RF.by_name(RF.apply(plan_of(rows), fam(), merged=merged))
    assert by["kernel"]["decision"] == "BUILD" and by["kernel"]["record"]
    assert "shares the merged file identical/_mojolearn_native_trees with family trees" in by["kernel"]["reason"]
    assert by["neural"]["decision"] == "REUSE" and not by["neural"]["record"]


def test_shared_or_the_runtime_building_records_every_family_and_keeps_their_bytes():
    rows = [row("_mojolearn", "BUILD"), row("_mojolearn_svm"), row("_mojolearn_training")]
    by = RF.by_name(RF.apply(plan_of(rows), fam(), merged=NO_MERGED))
    assert by["shared"]["decision"] == "BUILD"
    assert all(f["record"] for f in by.values())
    assert by["kernel"]["decision"] == "REUSE" and "shared builds" in by["kernel"]["record_reason"]
    assert next(r for r in rows if r["name"] == "_mojolearn_svm")["decision"] == "REUSE"
    rows = [row("_mojolearn"), row("_mojolearn_svm")]
    by = RF.by_name(RF.apply(plan_of(rows, runtime="BUILD"), fam(), merged=NO_MERGED))
    assert by["shared"]["decision"] == "BUILD" and "runtime .libs" in by["shared"]["reason"]
    assert by["kernel"]["record"]


def test_nothing_changed_keeps_every_family_and_records_nothing():
    rows = [row("_mojolearn"), row("_mojolearn_svm"), row("_mojolearn_gbdt")]
    out = RF.apply(plan_of(rows), fam(), merged=NO_MERGED)
    assert not RF.recording(out) and all(f["decision"] == "REUSE" for f in out["families"])
    assert RF.lanes_to_record(out, ["a", "b"], fam(lanes={"a": "trees", "b": "shared"})) == []


def test_rebuild_all_builds_and_records_everything():
    rows = [row("_mojolearn"), row("_mojolearn_svm")]
    p = plan_of(rows)
    out = RF.apply(p, fam(), merged=NO_MERGED, rebuild_all=True)
    assert all(f["decision"] == "BUILD" and f["record"] and f["reason"] == "--rebuild-all" for f in out["families"])
    assert all(r["decision"] == "BUILD" for r in rows) and p["runtime"]["decision"] == "BUILD" and out["rebuild_all"]


def test_a_row_with_no_family_stops_the_plan():
    with pytest.raises(SystemExit, match="no family for binding"):
        RF.apply(plan_of([row("_mojolearn_brand_new")]), fam(), merged=NO_MERGED)


def test_the_lanes_of_the_recording_families_in_order():
    f = fam(lanes={"gbdt-rmse": "trees", "svc": "kernel", "cross-val": "shared"})
    fplan = dict(families=[dict(family="shared", record=True), dict(family="trees", record=True),
                           dict(family="kernel", record=False)])
    assert RF.lanes_to_record(fplan, ["svc", "gbdt-rmse", "cross-val"], f) == ["gbdt-rmse", "cross-val"]
    with pytest.raises(SystemExit, match="no family"):
        RF.lanes_to_record(fplan, ["orphan"], fam(lanes={}, rules=()))


def test_render_states_per_family_bytes_lanes_and_the_reason():
    rows = [row("_mojolearn_gbdt", "BUILD"), row("_mojolearn_svm")]
    text = "\n".join(RF.render(RF.apply(plan_of(rows), fam(), merged=NO_MERGED)))
    assert "release families:" in text and "BUILD" in text and "KEEP" in text
    assert any(l.split()[:3] == ["trees", "BUILD", "RECORD"] for l in text.splitlines()[2:])
    assert any(l.split()[:3] == ["kernel", "REUSE", "KEEP"] for l in text.splitlines()[2:])


def test_release_reuse_render_prints_the_family_table():
    rows = [row("_mojolearn_gbdt", "BUILD"), row("_mojolearn_svm")]
    p = dict(plan_of(rows), commit="c" * 40, previous=dict(version="0.8.67", channel="pypi", source_commit="p" * 40),
             previous_identities_from="record", considered=[], legs=["cuda-sm_89"],
             leg_reasons={"cuda-sm_89": "a binding builds"})
    p["runtime"].update(identity_digest="r")
    RF.apply(p, fam(), merged=NO_MERGED)
    text = RR.render(p)
    assert "release families: " in text and "trees" in text


# ---------------------------------------------------------------- kept columns rest on the shipped bytes
def world(tmp_path, wheels=True, cells=True):
    src = tmp_path / "src"
    table = src / RF.TABLE
    table.parent.mkdir(parents=True)
    rec_dir = src / RF.RECORDS / "2026-11-20" / "amd"
    rec_dir.mkdir(parents=True)
    (rec_dir / "kernel.identical.json").write_text(json.dumps(dict(run=dict(wheels=[
        dict(name="mojolearn_amd-0.8.67.whl", sha256="a" * 64, version="0.8.67")] if wheels else []))))
    doc = dict(records=[dict(dir="2026-11-20/amd", file="kernel.identical.json", commit="k" * 40, vendor="amd",
                             **{"class": "amd"})],
               cells={"svc/base": {"train": dict(ref="h", cols={"amd": 0})}} if cells else {})
    table.write_text(json.dumps(doc))
    return src


def kept_plan():
    rows = [row("_mojolearn_gbdt", "BUILD"), row("_mojolearn_svm")]
    p = dict(plan_of(rows), host_toolchain=None)
    RF.apply(p, fam(), merged=NO_MERGED)
    return p


def test_a_kept_family_whose_cells_rest_on_todays_identity_from_release_wheels_is_kept(tmp_path, monkeypatch):
    p = kept_plan()
    f = fam(lanes={"svc": "kernel", "gbdt-rmse": "trees"})
    monkeypatch.setattr(RF, "_family_digests", lambda rows, commit, cache, root, tc: {r["key"]: r["identity_digest"]
                                                                                     for r in rows})
    assert RF.kept_refusals(p, ["svc", "gbdt-rmse"], f, source_root=world(tmp_path)) == {}
    assert RF.settle(p, ["svc", "gbdt-rmse"], f, source_root=world(tmp_path / "again")) == ["gbdt-rmse"]


@pytest.mark.parametrize("case", ["moved", "no-wheels", "no-cell"])
def test_a_kept_family_whose_columns_do_not_rest_on_the_shipped_bytes_records(tmp_path, monkeypatch, case):
    p = kept_plan()
    f = fam(lanes={"svc": "kernel", "gbdt-rmse": "trees"})
    monkeypatch.setattr(RF, "_family_digests", lambda rows, commit, cache, root, tc: {
        r["key"]: ("other" if case == "moved" else r["identity_digest"]) for r in rows})
    src = world(tmp_path, wheels=case != "no-wheels", cells=case != "no-cell")
    why = RF.kept_refusals(p, ["svc", "gbdt-rmse"], f, source_root=src)
    assert set(why) == {"kernel"}
    assert {"moved": "had another identity", "no-wheels": "names no installed wheels",
            "no-cell": "has no recorded cell"}[case] in why["kernel"]
    lanes = RF.settle(p, ["svc", "gbdt-rmse"], f, source_root=src)
    assert lanes == ["svc", "gbdt-rmse"]
    assert "do not rest on the shipped bytes" in RF.by_name(p["families"])["kernel"]["record_reason"]


# ---------------------------------------------------------------- the CLI the recorders call
def test_the_lanes_command_maps_families_to_lanes(tmp_path):
    lanes = tmp_path / "lanes.txt"
    lanes.write_text("knn,ols\n")
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "tools" / "release_families.py"), "lanes", *a,
                                     "--lanes-file", str(lanes)], capture_output=True, text=True, timeout=120)
    p = run("--families", "shared")          # the committed rule sends every lane to shared until derive --write
    assert p.returncode == 0 and p.stdout.strip().splitlines()[-1] == "knn,ols"
    p = run("--families", "none")
    assert p.returncode == 0 and p.stdout.strip() == ""
    p = run("--families", "nope")
    assert p.returncode != 0 and "unknown famil" in p.stderr
