# SPDX-License-Identifier: Apache-2.0
"""tools/release_plan.py: one fake-input test per DECISION TABLE row, plus the tools/release.py hooks
(gate-boxes rents nothing when the plan owes no device work; publication refuses while the plan's
evidence is missing; waste.json). Nothing here reads git, a box or the release directory."""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release_plan as rp  # noqa: E402

PREV = "a" * 40
NEW = "b" * 40


def groups():
    return {
        "linux/identical/_mojolearn_native_trees": dict(closure=["gbdt/boost.mojo", "core/device_plan.mojo"]),
        "linux/identical/_mojolearn_native": dict(closure=["linalg/qr.mojo", "core/device_plan.mojo"]),
        "linux/host/_mojolearn_native_host": dict(closure=["core/device_plan.mojo", "host/thing.mojo"]),
    }


def lanes(owed_since_after_previous=True, device="nvidia"):
    return {"nvidia": {
        "gbdt-fit": dict(owed=True, reason="closure changed since " + NEW[:12], since=NEW,
                         since_after_previous=owed_since_after_previous, device=device),
        "qr": dict(owed=False, since=PREV, device=device)}}


def gates(**over):
    out = {
        "freeze-version": dict(kind="metadata", state="owed"),
        "linux-pack": dict(kind="package", state="owed"),
        "macos-smoke": dict(kind="loading", state="owed"),
        "linux-builds": dict(kind="artifact", state="owed"),
        "gate-apple": dict(kind="numerical", state="owed",
                           previous_evidence=dict(local=True, local_path="/prev/gate-apple")),
    }
    out.update(over)
    return out


def facts(changes, **kw):
    base = dict(version="0.8.66", commit=NEW, previous=PREV, changes=changes, groups=groups(),
                lanes=lanes(), gates=gates(), leases={"nvidia": 210, "amd": 240})
    base.update(kw)
    return base


def by_id(rows):
    return {r["id"]: r for r in rows}


# ---------------------------------------------------------------- row 1: docs/version/changelog only
def test_metadata_only_change_runs_metadata_and_package_checks_and_reruns_nothing_numerical():
    plan = rp.decide(facts(["CHANGELOG.md", "python/mojolearn/_version.py", "docs/release/NOTES.md",
                            "tools/release.py"]))
    assert plan["change_class"] == "metadata"
    assert {r["action"] for r in plan["compile"]} == {"hit"}
    ident = by_id(plan["identity"])
    # 0.8.65: _version.py "reached" the hf lanes; the version bump is metadata, never a re-record
    assert ident["nvidia/gbdt-fit"]["action"] == "reuse" and "decision table metadata" in ident["nvidia/gbdt-fit"]["reason"]
    g = by_id(plan["gates"])
    assert g["freeze-version"]["action"] == "run" and g["linux-pack"]["action"] == "run"
    assert g["gate-apple"]["action"] == "reuse" and "previous release" in g["gate-apple"]["reason"]
    assert g["linux-builds"]["action"] == "reuse" and "every compile group hits" in g["linux-builds"]["reason"]
    # a check the class does not run, with no previous evidence to reuse, still runs
    assert g["macos-smoke"]["action"] == "run"
    assert plan["boxes"]["action"] == "none" and "rents nothing" in plan["boxes"]["reason"]


def test_a_lane_owed_since_before_the_previous_release_is_still_recorded_in_a_metadata_release():
    plan = rp.decide(facts(["CHANGELOG.md"], lanes=lanes(owed_since_after_previous=False)))
    row = by_id(plan["identity"])["nvidia/gbdt-fit"]
    assert row["action"] == "record" and plan["boxes"]["action"] == "rent"
    assert plan["boxes"]["boxes"]["nvidia"]["minutes"] == 210


# ---------------------------------------------------------------- row 2: unchanged binaries repackaged
def test_repackage_runs_artifact_byte_checks_and_the_changed_loading_boundary_only():
    plan = rp.decide(facts(["packaging/linux/pack_wheel.py", "python/mojolearn/_merged.py"],
                           gates=gates(**{"linux-builds": dict(kind="artifact", state="owed")})))
    assert plan["change_class"] == "repackage"
    assert {r["action"] for r in plan["compile"]} == {"hit"}
    g = by_id(plan["gates"])
    assert g["linux-builds"]["action"] == "run" and g["macos-smoke"]["action"] == "run"
    assert g["gate-apple"]["action"] == "reuse"
    assert by_id(plan["identity"])["nvidia/gbdt-fit"]["action"] == "reuse"


# ---------------------------------------------------------------- row 3: one algorithm implementation
def test_one_algorithm_compiles_its_groups_and_records_the_lanes_its_closure_reaches():
    plan = rp.decide(facts(["gbdt/boost.mojo"]))
    assert plan["change_class"] == "algorithm"
    c = by_id(plan["compile"])
    assert c["linux/identical/_mojolearn_native_trees"]["action"] == "compile"
    assert "gbdt/boost.mojo" in c["linux/identical/_mojolearn_native_trees"]["reason"]
    assert c["linux/identical/_mojolearn_native"]["action"] == "hit"
    assert c["linux/host/_mojolearn_native_host"]["action"] == "hit"
    assert by_id(plan["identity"])["nvidia/gbdt-fit"]["action"] == "record"
    assert by_id(plan["identity"])["nvidia/qr"]["action"] == "reuse"
    assert by_id(plan["gates"])["gate-apple"]["action"] == "run"
    assert plan["invalidates"] == {"gbdt/boost.mojo": ["linux/identical/_mojolearn_native_trees"]}
    assert any("gbdt/boost.mojo invalidates these 1 groups" in line for line in rp.lines(plan))


# ---------------------------------------------------------------- row 4: shared arithmetic / toolchain
def test_shared_arithmetic_rebuilds_every_dependent_justified_by_its_closure():
    plan = rp.decide(facts(["core/device_plan.mojo"]))
    assert plan["change_class"] == "shared"
    assert all(r["action"] == "compile" and "core/device_plan.mojo" in r["reason"] for r in plan["compile"])
    assert "reaches all 3 compile groups" in plan["class_reasons"][0]


def test_a_toolchain_pin_rebuilds_everything_and_says_so():
    plan = rp.decide(facts(["pixi.lock"]))
    assert plan["change_class"] == "shared"
    assert all(r["reason"] == "toolchain pin changed: pixi.lock" for r in plan["compile"])


def test_a_comment_only_build_script_edit_is_judged_per_group_by_its_executed_inputs():
    g = {k: dict(v, inputs_changed=[]) for k, v in groups().items()}
    plan = rp.decide(facts(["bindings/merged_build.sh"], groups=g))
    assert {r["action"] for r in plan["compile"]} == {"hit"}
    g = {k: dict(v, inputs_changed=["bindings/merged_build.sh"]) for k, v in groups().items()}
    plan = rp.decide(facts(["bindings/merged_build.sh"], groups=g))
    assert {r["action"] for r in plan["compile"]} == {"compile"}


# ---------------------------------------------------------------- row 5: missing local evidence
def test_missing_local_evidence_is_restored_before_any_execution():
    g = groups()
    g["linux/identical/_mojolearn_native"]["evidence"] = dict(local=False, retained="r2:bincache/linux-64/sm_89")
    plan = rp.decide(facts(["gbdt/boost.mojo"], groups=g))
    (row,) = plan["restore"]
    assert row["id"] == "linux/identical/_mojolearn_native" and row["source"] == "r2:bincache/linux-64/sm_89"
    assert by_id(plan["compile"])["linux/identical/_mojolearn_native"]["action"] == "hit"
    assert plan["publish"]["status"] == "blocked"
    assert any(b["reason"].startswith("evidence to restore first") for b in plan["publish"]["blocked"])
    text = rp.lines(plan)
    assert text.index(next(t for t in text if t.startswith("restore:"))) < \
        text.index(next(t for t in text if t.startswith("compile:")))


def test_reuse_with_no_copy_anywhere_becomes_owed_work():
    g = groups()
    g["linux/identical/_mojolearn_native"]["evidence"] = dict(local=False, retained=None)
    plan = rp.decide(facts(["gbdt/boost.mojo"], groups=g))
    row = by_id(plan["compile"])["linux/identical/_mojolearn_native"]
    assert row["action"] == "compile" and row["reason"].startswith("no evidence on this machine and none retained")
    assert plan["restore"] == []


# ---------------------------------------------------------------- row 6: genuine unresolved failure
def test_an_unresolved_failure_blocks_publication_and_a_resolved_one_does_not():
    quiet = dict(lanes={}, gates={})
    plan = rp.decide(facts(["CHANGELOG.md"], failures=[dict(id="gate-amd", reason="verdict FAIL", resolved=False)],
                           **quiet))
    assert plan["publish"]["status"] == "blocked"
    assert plan["publish"]["blocked"] == [dict(id="gate-amd", reason="unresolved failure: verdict FAIL")]
    plan = rp.decide(facts(["CHANGELOG.md"], failures=[dict(id="gate-amd", reason="x", resolved=True)], **quiet))
    assert plan["publish"]["status"] == "ready" and plan["launches_nothing"] is True


# ---------------------------------------------------------------- boxes
def test_no_owed_device_work_rents_nothing_and_says_so():
    plan = rp.decide(facts(["linalg/qr.mojo"], lanes={"apple": {"qr": dict(owed=True, reason="closure changed",
                                                                           device=None)}}))
    assert plan["boxes"] == dict(action="none", reason="no owed device work: gate-boxes rents nothing", boxes={})


def test_a_self_checked_box_step_that_owes_work_is_never_claimed_as_reuse():
    g = gates(**{"gate-nvidia": dict(kind="numerical", state="owed", self_checked=True, device="nvidia",
                                     reason="no gate report", previous_evidence=dict(local=True))})
    plan = rp.decide(facts(["CHANGELOG.md"], gates=g, lanes={}))
    row = by_id(plan["gates"])["gate-nvidia"]
    assert row["action"] == "run" and "no gate report" in row["reason"]
    assert plan["boxes"]["boxes"]["nvidia"]["minutes"] == 210


# ---------------------------------------------------------------- waste
def test_waste_counts_and_the_delta_against_the_previous_release(tmp_path):
    prev = dict(version="0.8.65", groups_reused=0, groups_rebuilt=3, groups_rebuilt_seconds=900.0,
                lanes_reused=0, lanes_recorded=2, box_wait_seconds=None, admission_seconds=50)
    plan = rp.decide(facts(["gbdt/boost.mojo"], group_seconds={"linux/identical/_mojolearn_native_trees": 120},
                           step_seconds={"record": 40.0}, admission_seconds=40.0, previous_waste=prev))
    w = plan["waste"]
    assert (w["groups_reused"], w["groups_rebuilt"], w["groups_rebuilt_seconds"]) == (2, 1, 120)
    assert (w["lanes_reused"], w["lanes_recorded"]) == (1, 1) and w["steps"] == {"record": 40.0}
    d = plan["waste_delta"]
    assert d["groups_rebuilt"] == -2 and d["lanes_recorded"] == -1 and d["against"] == "0.8.65"
    assert any(line.startswith("waste vs 0.8.65") for line in rp.lines(plan))
    root = tmp_path / "release"
    rp.write_waste(root / "0.8.64" / "aaa" / "waste.json", dict(prev, version="0.8.64"))
    rp.write_waste(root / "0.8.65" / "bbb" / "waste.json", prev)
    rp.write_waste(root / "0.8.66" / "ccc" / "waste.json", dict(prev, version="0.8.66"))
    assert rp.previous_waste(root, "0.8.66")["version"] == "0.8.65"
    assert rp.previous_waste(root, "0.8.64") is None


def test_kinds_and_lane_facts_parse_the_selector_summary():
    assert rp.kind("python/mojolearn/_version.py") == "metadata"
    assert rp.kind("gbdt/boost.mojo") == "native" and rp.kind("python/mojolearn/linear.py") == "python"
    assert rp.kind("packaging/linux/pack_wheel.py") == "packaging" and rp.kind("tools/x.py") == "tooling"
    summary = {"record": {"a": "closure changed since " + NEW[:12]}, "reused": {"b": {"commit": PREV}}}
    out = rp.lane_facts(summary, PREV, "nvidia", "nvidia", is_ancestor=lambda a, b: (a, b) == (PREV, NEW[:12]))
    assert out["a"] == dict(owed=True, reason="closure changed since " + NEW[:12], since=NEW[:12],
                            since_after_previous=True, device="nvidia")
    assert out["b"]["owed"] is False


def test_decide_cli_writes_plan_json_and_exits_3_when_blocked(tmp_path):
    f = tmp_path / "facts.json"
    f.write_text(json.dumps(facts(["gbdt/boost.mojo"])))
    assert rp.main(["decide", "--facts", str(f), "--json", str(tmp_path / "plan.json")]) == 3
    assert json.loads((tmp_path / "plan.json").read_text())["schema"] == rp.SCHEMA


# ---------------------------------------------------------------- tools/release.py hooks
def test_gate_boxes_rents_nothing_when_the_saved_plan_owes_no_device_work(tmp_path, monkeypatch):
    import release
    from test_release_no_hands import box_release
    monkeypatch.setenv("MOJOLEARN_ORCHESTRATOR", "1")
    r, boxes = box_release(tmp_path, monkeypatch)
    out = tmp_path / "plan.json"
    out.write_text(json.dumps(dict(schema=rp.SCHEMA, commit=r.commit, boxes=dict(action="none"))))
    r.args.plan_json = str(out)
    with pytest.raises(release.StepFailed, match="owes no device work"):
        r.step_gate_boxes()
    assert boxes.calls == []


def test_publication_refuses_an_unresolved_failure_the_plan_names(tmp_path, monkeypatch):
    from test_release_no_hands import make
    r = make(tmp_path)
    out = tmp_path / "plan.json"
    out.write_text(json.dumps(dict(schema=rp.SCHEMA, commit=r.commit, restore=[],
                                   publish=dict(blocked=[dict(id="gate-amd", reason="unresolved failure: FAIL")]))))
    r.args.plan_json = str(out)
    assert r.plan_blocks_publication() == ["gate-amd: unresolved failure: FAIL"]
    r.state["steps"]["gate-amd"] = dict(done=True, commit=r.state["commit"], at="x")
    assert r.plan_blocks_publication() == []
    out.write_text(json.dumps(dict(schema=rp.SCHEMA, commit="0" * 40, publish=dict(blocked=[dict(id="x",
                                   reason="unresolved failure: y")]))))
    assert r.plan_blocks_publication() == []   # another commit's plan says nothing about this one


def test_waste_json_is_written_after_a_real_run_with_step_seconds(tmp_path):
    from test_release_no_hands import make, record
    r = make(tmp_path)
    record(r, "linux-pack", seconds=12.5)
    record(r, "record", seconds=40.0)
    path = r.write_waste()
    doc = json.loads(path.read_text())
    assert doc["steps"] == {"linux-pack": 12.5, "record": 40.0} and doc["admission_seconds"] == 40.0


# ---------------------------------------------------------------- interruption recovery
def test_say_survives_a_disconnected_terminal(tmp_path, monkeypatch):
    import release
    from test_release_no_hands import make
    r = make(tmp_path)
    del r.say   # the instance fake; use the real method

    class Gone:
        def write(self, _):
            raise OSError(5, "Input/output error")

        def flush(self):
            raise OSError(5, "Input/output error")
    monkeypatch.setattr(sys, "stdout", Gone())
    release.Release.say(r, "a line nobody can read")   # no exception


def test_a_real_run_ignores_sighup(monkeypatch):
    import signal
    import release
    seen = {}
    monkeypatch.setattr(signal, "signal", lambda sig, handler: seen.setdefault(sig, handler))
    assert release.ignore_hangup() is True and seen == {signal.SIGHUP: signal.SIG_IGN}
