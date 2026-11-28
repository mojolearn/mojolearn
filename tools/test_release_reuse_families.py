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


# ---------------------------------------------------------------- one family per merged file (lane family-groups-1)
def test_no_merged_file_of_this_checkout_holds_two_families():
    f = RF.load()
    assert RF.merged_span_problems(f) == []
    # every directory set, the optional byte LM in its group, and so each macOS <stem>_m1 file (a subset of a group)
    for key, members in RF.merged_files(f.merged):
        assert len({f.binding(m) for m in members}) == 1, key
    # the regroup moved no member in or out of a directory set and kept every module name
    names = {d: sorted(m for k, ms in RF.merged_files(f.merged) if k.startswith(d + "/") for m in ms)
             for d in ("fast", "identical", "host")}
    assert len(names["host"]) == len(set(names["host"])) == 56
    assert names["fast"] == names["identical"] and len(names["identical"]) == len(set(names["identical"])) == 34
    assert "_mojolearn_byte_lm" in names["identical"]


def test_check_refuses_a_merged_file_that_spans_families():
    merged = {"dirs": {"host": {"path": "host", "groups": {
        "_mojolearn_native_host": ["_mojolearn_core_host", "_mojolearn_linalg_host"],
        "_mojolearn_native_forest_host": ["_mojolearn_gbdt_host", "_mojolearn_rf_host"]},
        "optional": {"_mojolearn_svm_host": {"group": "_mojolearn_native_forest_host", "platforms": ["linux"]}}}}}
    problems = RF.merged_span_problems(fam(), merged)
    assert len(problems) == 2
    assert any(p.startswith("merged file host/_mojolearn_native_host holds bindings of 2 families") and
               "linear: _mojolearn_linalg_host" in p and "shared: _mojolearn_core_host" in p for p in problems)
    # an optional member counts in the group it joins
    assert any("host/_mojolearn_native_forest_host" in p and "kernel: _mojolearn_svm_host" in p for p in problems)
    merged["dirs"]["host"]["groups"]["_mojolearn_native_host"] = ["_mojolearn_core_host"]
    merged["dirs"]["host"]["optional"] = {}
    assert RF.merged_span_problems(fam(), merged) == []


def test_the_tokenizer_host_owner_is_shared():
    f = RF.load()
    assert f.binding("_mojolearn_tokenizer_host") == "shared"
    assert f.binding("_mojolearn_x_tokenizer") == f.binding("_mojolearn_runtime_catalog_host") == "tokenizer"


# ---------------------------------------------------------------- the Python glue is in a family's identity
def py_fam(rules):
    doc = dict(RF.load().doc)
    doc["python"] = dict(root="python/mojolearn", rules=[list(r) for r in rules])
    return RF.Families(doc)


def test_every_python_file_of_this_checkout_has_a_family_and_no_family_file_is_imported_across():
    f = RF.load()
    assert RF.python_problems(f) == []
    assert f.python("_expansion_cnn.py") == "neural" and f.python("linalg_fast.py") == "linear"
    assert f.python("_backend.py") == f.python("host_surface.py") == f.python("models/causal_lm.py") == "shared"
    # per-file families (lane py-glue-1, 2026-11-28): leaf modules have a family by name; what a shared file imports,
    # a composition, and a new file under a pattern stay shared
    assert f.python("ops.py") == f.python("importers/_program.py") == f.python("models/_catalog.py") == "neural"
    assert f.python("_x_sequence_ets.py") == "neural" and f.python("_x_sequence_autoarima.py") == "series"
    assert f.python("_expansion_ann.py") == "cluster" and f.python("_surface_trees.py") == "trees"
    assert f.python("_x_sequence_model.py") == f.python("_training_impl.py") == f.python("_expansion_trees.py") == "shared"
    assert f.python("_x_sequence_new.py") == f.python("importers/new.py") == f.python("models/__init__.py") == "shared"
    assert f.python("_surface_new.py") is None                # a new lane fragment must be given its family
    # verification tooling, tests and the version string are not product glue
    for rel in ("_verify.py", "_verify_light.py", "_verification_catalog.py", "__main__.py", "tests/test_x.py",
                "verify_reference/table.json", "_version.py", "ALPHA_API.md"):
        assert f.python(rel) == RF.PY_NOT_GLUE, rel
    assert f.python("_brand_new_module.py") is None           # a new runtime file must be given a family


def test_python_problems_refuse_an_unmapped_file_a_cross_family_import_and_another_familys_binding(tmp_path):
    base = tmp_path / "python/mojolearn"
    (base / "sub").mkdir(parents=True)
    files = {"_trees_glue.py": "from . import _common\nNAME = '_mojolearn_gbdt'\n",
             "_common.py": "X = 1\n",
             "_linear_glue.py": "from ._trees_glue import (\n    NAME,\n)\nB = '_mojolearn_estimators'\n",
             "_kernel_glue.py": "import os\nB = '_mojolearn_gbdt_host'\n",
             "_verify_thing.py": "from . import _trees_glue\n",
             "sub/__init__.py": "",
             "sub/inner.py": "from .. import _trees_glue\nfrom mojolearn import _common\n",
             "orphan.py": "X = 2\n"}
    for rel, text in files.items():
        (base / rel).write_text(text)
    f = py_fam([("_verify*.py", "-"), ("_trees_glue.py", "trees"), ("_linear_glue.py", "linear"),
                ("_kernel_glue.py", "kernel"), ("_common.py", "shared"), ("sub/*", "shared")])
    problems = RF.python_problems(f, root=tmp_path, files=sorted(files), workers=1)
    assert "python file python/mojolearn/orphan.py has no family (bindings/release_families.json python.rules)" in problems
    assert any(p.startswith("python glue _trees_glue.py is in family trees but _linear_glue.py (linear) imports it")
               for p in problems)
    assert any(p.startswith("python glue _trees_glue.py is in family trees but sub/inner.py (shared) imports it")
               for p in problems)
    assert any(p == "python glue _kernel_glue.py is in family kernel but names _mojolearn_gbdt_host (trees)"
               for p in problems)
    # verification tooling may import anything; a family file may import shared glue and name its own bindings
    assert not any("_verify_thing.py" in p or "_common.py is in family" in p for p in problems)
    assert len(problems) == 4
    assert RF.python_problems(py_fam([("*", "nope")]), root=tmp_path, files=["orphan.py"], workers=1)[0] == \
        "python rule * names the unknown family 'nope'"


def test_python_digests_and_owner_follow_the_first_matching_rule():
    f = py_fam([("tests/*", "-"), ("_a.py", "trees"), ("_*.py", "shared")])
    assert [RF.python_owner(f, r) for r in ("tests/t.py", "_a.py", "_b.py", "new.py")] == [None, "trees", "shared", "shared"]
    one = RF.python_digests(f, {"_a.py": "1" * 40, "_b.py": "2" * 40, "tests/t.py": "3" * 40})
    two = RF.python_digests(f, {"_a.py": "9" * 40, "_b.py": "2" * 40, "tests/t.py": "4" * 40})
    assert one["trees"] != two["trees"] and one["shared"] == two["shared"] and one["kernel"] == two["kernel"]


def test_python_glue_names_the_changed_files_per_family(tmp_path):
    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                                     *a], check=True, capture_output=True, text=True).stdout.strip()
    base = tmp_path / "python/mojolearn"
    (base / "tests").mkdir(parents=True)
    git("init", "-q")
    for rel, text in (("_a.py", "1"), ("_b.py", "1"), ("_version.py", "1"), ("tests/t.py", "1"), ("gone.py", "1")):
        (base / rel).write_text(text)
    git("add", "-A")
    git("commit", "-q", "-m", "one")
    first = git("rev-parse", "HEAD")
    f = py_fam([("tests/*", "-"), ("_version.py", "-"), ("_a.py", "trees"), ("_b.py", "shared")])
    assert RF.python_glue(f, first, first, root=tmp_path)["changed"] == {}
    for rel in ("_a.py", "_version.py", "tests/t.py"):
        (base / rel).write_text("2")
    (base / "gone.py").unlink()
    git("add", "-A")
    git("commit", "-q", "-m", "two")
    glue = RF.python_glue(f, "HEAD", first, root=tmp_path)
    # the version string and the tests are not glue; a file no rule matches is shared's (the safe side)
    assert glue["changed"] == {"trees": ["_a.py"], "shared": ["gone.py"]}
    assert glue["digests"]["trees"] != RF.python_glue(f, first, first, root=tmp_path)["digests"]["trees"]
    # no previous source commit, or one this clone lacks: nothing is known unchanged, every lane records
    for prev in (None, "0" * 40):
        assert list(RF.python_glue(f, "HEAD", prev, root=tmp_path)["changed"]) == ["shared"]


def test_changed_python_glue_records_its_family_and_reuses_its_binaries():
    rows = [row("_mojolearn"), row("_mojolearn_svm"), row("_mojolearn_gbdt"), row("_mojolearn_training")]
    p = plan_of(rows)
    out = RF.apply(p, fam(), merged=NO_MERGED, python=dict(changed={"kernel": ["_svm_glue.py"]},
                                                          digests={"kernel": "k", "shared": "s"}))
    by = RF.by_name(out)
    assert by["kernel"]["decision"] == "REUSE" and by["kernel"]["record"]
    assert "its Python glue changed (_svm_glue.py)" in by["kernel"]["record_reason"]
    assert all(r["decision"] == "REUSE" for r in rows)              # no binary reads a Python file
    assert not by["trees"]["record"] and not by["neural"]["record"]
    assert by["shared"]["record"] and "one records" in by["shared"]["record_reason"]
    assert out["python_digests"] == {"kernel": "k", "shared": "s"}
    assert RF.lanes_to_record(out, ["svc", "gbdt-rmse", "cv"], fam(lanes={"svc": "kernel", "gbdt-rmse": "trees",
                                                                          "cv": "shared"})) == ["svc", "cv"]
    text = "\n".join(RF.render(out))
    assert any(l.split()[:3] == ["kernel", "REUSE", "RECORD"] for l in text.splitlines()[2:])


def test_changed_shared_python_glue_records_every_family():
    rows = [row("_mojolearn"), row("_mojolearn_svm"), row("_mojolearn_gbdt")]
    out = RF.apply(plan_of(rows), fam(), merged=NO_MERGED, python=dict(changed={"shared": ["_backend.py"]}))
    assert all(f["record"] and f["decision"] == "REUSE" for f in out["families"])
    assert "the shared Python glue changed (_backend.py)" in RF.by_name(out)["trees"]["record_reason"]
    # a family another map names is shared's
    out = RF.apply(plan_of([row("_mojolearn_svm")]), fam(), merged=NO_MERGED, python=dict(changed={"nope": ["x.py"]}))
    assert all(f["record"] for f in out["families"])


def test_a_kept_family_whose_python_glue_cannot_be_read_at_its_records_commit_records(tmp_path, monkeypatch):
    p = kept_plan()
    p["families"]["python_digests"] = {"kernel": "k", "shared": "s"}
    f = fam(lanes={"svc": "kernel", "gbdt-rmse": "trees"})
    monkeypatch.setattr(RF, "_family_digests", lambda rows, commit, cache, root, tc: {r["key"]: r["identity_digest"]
                                                                                     for r in rows})
    # the fixture tree is no git clone: the glue at the record's commit is unreadable, so the family records
    why = RF.kept_refusals(p, ["svc", "gbdt-rmse"], f, source_root=world(tmp_path))
    assert set(why) == {"kernel"} and "the Python glue at" in why["kernel"]
    monkeypatch.setattr(RF, "python_tree", lambda commit, root, py_root: {})
    monkeypatch.setattr(RF, "python_digests", lambda fam_, tree: {"kernel": "k", "shared": "other"})
    why = RF.kept_refusals(p, ["svc", "gbdt-rmse"], f, source_root=world(tmp_path / "b"))
    assert "where the Python glue of shared was different" in why["kernel"]
    monkeypatch.setattr(RF, "python_digests", lambda fam_, tree: {"kernel": "k", "shared": "s"})
    assert RF.kept_refusals(p, ["svc", "gbdt-rmse"], f, source_root=world(tmp_path / "c")) == {}


def test_loaders_counts_the_lanes_a_family_of_its_own_would_turn_shared():
    loads = {"rf-fit": ["_mojolearn_rf", "_mojolearn_runtime_trees_host", "_mojolearn"],
             "kde": ["_mojolearn_runtime_trees_host", "_mojolearn_x_neighbors"],
             "stack": ["_mojolearn_runtime_trees_host", "_mojolearn_svm", "_mojolearn_x_prep"],
             "ols": ["_mojolearn_estimators"]}
    (row,) = RF.loaders(fam(), loads, ["_mojolearn_runtime_trees_host"])
    assert row == dict(binding="_mojolearn_runtime_trees_host", family="trees", lanes=3, alone=1,
                       by_family={"cluster": 1}, mixed=1)
