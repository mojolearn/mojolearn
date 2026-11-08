"""The requirements ledger evaluated once per freeze and inputs (lane post-ledger-once, 2026-11-07): the second
read computes nothing, a new input computes only its rows, the evaluation fans out over worker processes,
gate-boxes refuses to rent with an input missing, and the idle-box watchdog's accounting on a fake clock.
Pure Python: nothing is built, rented or run."""
import argparse
import json
import pathlib
import sys
from unittest import mock

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402
import release_ledger_once as once  # noqa: E402
import release_requirements as rr  # noqa: E402
import release_requirements_retained as retained  # noqa: E402

COMMIT = "c" * 40
TOOLING = "d" * 40


def args(state_dir, **kw):
    base = dict(version="0.8.66", dry_run=False, publish=None, only="", redo="", build_backend="gpu-legs",
                amd_expect_from="", smoke_gpu="", state_dir=str(state_dir), amd_build_provider=None,
                amd_provider="auto", smoke_via="rent", hopper_box="", smoke_branch="main", cpu_column=False)
    base.update(kw)
    return argparse.Namespace(**base)


def wanted(vendor, wheel):
    inputs = dict(contract=dict(kind="candidate", stage="pre"), wheels={vendor: wheel})
    return dict(source_revision=COMMIT, fingerprint=rr.digest(inputs), inputs=inputs, required=True,
                stage="pre", kind="candidate", missing_inputs=[])


def manifest(wheels):
    return dict(version="0.8.66", source_revision=COMMIT,
                configuration={"artifacts:" + v: {v + ".whl": sha} for v, sha in wheels.items()},
                requirements={"candidate-" + v: wanted(v, sha) for v, sha in wheels.items()})


@pytest.fixture
def counted(monkeypatch):
    """Every row evaluate computes, by name (a fixture counter on the one per-row function)."""
    calls = []
    original = rr.RequirementsLedger._evaluate_row

    def row(*a, **kw):
        calls.append(a[1])
        return original(*a, **kw)
    monkeypatch.setattr(rr.RequirementsLedger, "_evaluate_row", staticmethod(row))
    return calls


def make(tmp_path, monkeypatch, wheels):
    r = release.Release(args(tmp_path / "state"), runner=lambda *a, **k: 0)
    r.state["commit"] = COMMIT
    r.said = []
    r.say = r.said.append
    monkeypatch.setattr(release.Release, "tooling_commit", property(lambda self: TOOLING))
    monkeypatch.setattr(r, "retained_producers", lambda: [])
    admitted = []
    monkeypatch.setattr(retained, "admit_existing_evidence",
                        lambda *a, **k: admitted.append(1) or ([], []))
    r.admitted = admitted

    def configuration():
        conf = {"artifacts:" + v: {v + ".whl": sha} for v, sha in wheels.items()}
        key, inputs = once.evaluation_key(COMMIT, TOOLING, "cat", conf, version="0.8.66")
        return conf, {}, "cat", ("k", json.dumps(conf, sort_keys=True)), key, inputs
    monkeypatch.setattr(r, "readiness_configuration", configuration)
    analyses = []

    def requirements(root, commit, version, configuration=None, **kw):
        analyses.append(dict(configuration))
        return manifest({k.split(":", 1)[1]: next(iter(v.values())) for k, v in configuration.items()})
    monkeypatch.setattr(rr, "requirements", requirements)
    monkeypatch.setattr(r, "optional_native_presence", lambda manifest, context: None)
    r.analyses = analyses
    return r


def test_the_second_read_reads_the_document_and_computes_nothing(tmp_path, monkeypatch, counted):
    wheels = dict(nvidia="1" * 64, amd="2" * 64)
    r = make(tmp_path, monkeypatch, wheels)
    first = r.readiness_status(write=True)
    assert sorted(counted) == ["candidate-amd", "candidate-nvidia"]
    assert len(r.analyses) == 1 and len(r.admitted) == 1
    key, stored = r.ledger_document()
    path = r.rel / "requirements-eval" / (key + ".json")
    assert stored is not None and path.is_file()
    assert stored["stats"]["units"] == 2 and stored["stats"]["computed"] == 2 and "seconds" in stored["stats"]
    assert stored["inputs"]["wheels"] == sorted(wheels.values()) and stored["inputs"]["source_commit"] == COMMIT
    # every later caller (the gates, the publish preflight, the record, --status): read, nothing computed
    r2 = make(tmp_path, monkeypatch, wheels)   # a resumed release process too
    for caller in (r, r2):
        counted.clear()
        assert caller.readiness_status(write=True, box_step="gate-nvidia") == first
        assert caller.readiness_status(write=False) == first
        assert counted == []
        assert not any("LEDGER EVALUATED IN BOX STEP" in line for line in caller.said)
        assert any("nothing computed" in line for line in caller.said)
    assert r2.analyses == [] and r2.admitted == []   # no dependency round, no reconstruction


def test_a_new_wheel_computes_only_the_rows_it_changes(tmp_path, monkeypatch, counted):
    wheels = dict(nvidia="1" * 64)
    r = make(tmp_path, monkeypatch, wheels)
    r.readiness_status(write=True)
    old_key, _ = r.ledger_document()
    counted.clear()
    wheels["amd"] = "2" * 64          # linux-pack's AMD wheel: a new key
    r.readiness_status(write=True)
    new_key, stored = r.ledger_document()
    assert new_key != old_key and stored is not None
    assert counted == ["candidate-amd"]                    # the nvidia row is read from the memo
    assert stored["stats"]["reused"] >= 1 and stored["stats"]["computed"] == 1
    assert (r.rel / "requirements-eval" / (old_key + ".json")).is_file()   # the earlier document is kept


def test_a_box_step_that_has_to_compute_says_so(tmp_path, monkeypatch, counted):
    r = make(tmp_path, monkeypatch, dict(nvidia="1" * 64))
    r.readiness_status(write=True, box_step="gate-nvidia")
    assert any("LEDGER EVALUATED IN BOX STEP gate-nvidia" in line and "owed it" in line for line in r.said)


def test_a_stored_manifest_runs_no_dependency_analysis(tmp_path, monkeypatch):
    wheels = dict(nvidia="1" * 64)
    r = make(tmp_path, monkeypatch, wheels)
    r.readiness_status(write=True)
    r2 = make(tmp_path, monkeypatch, wheels)
    manifest_, key, _ = r2.readiness_manifest_keyed()
    assert r2.analyses == [] and key == r.ledger_document()[0]
    assert set(manifest_["requirements"]) == {"candidate-nvidia"}


def test_the_prefetch_hook_writes_the_document_once(tmp_path, monkeypatch, counted):
    import release_requirement_dependencies as dependencies
    r = make(tmp_path, monkeypatch, dict(nvidia="1" * 64))
    monkeypatch.setattr(dependencies, "prefetch", lambda compute, **kw: compute())
    doc = r.prefetch_ledger_rounds("linux-pack")
    assert doc["error"] is None and pathlib.Path(doc["document"]).is_file()
    assert "evaluated once" in doc["summary"]
    counted.clear()
    r.readiness_status(write=True, box_step="gate-nvidia")
    assert counted == []


# ------------------------------------------------------------ the evaluation itself
def ledger_doc(n):
    requirements = {f"candidate-v{i}": wanted(f"v{i}", f"{i:064d}") for i in range(n)}
    return dict(schema=rr.SCHEMA, version="0.8.66", requirements=requirements, attempts=[], runs=[])


def test_evaluate_reads_unchanged_rows_from_the_memo(counted):
    doc = ledger_doc(3)
    memo = rr.EvaluationMemo()
    stats = {}
    rows = rr.RequirementsLedger.evaluate(doc, memo=memo, jobs=1, stats=stats)
    assert len(counted) == 3 and stats["computed"] == 3 and stats["units"] == 3
    counted.clear()
    again = rr.RequirementsLedger.evaluate(doc, memo=memo, jobs=1, stats=stats)
    assert again == rows and counted == [] and stats["reused"] == 3
    doc["runs"].append(dict(id="run-1", requirement="candidate-v1", source_revision=COMMIT,
                            fingerprint=doc["requirements"]["candidate-v1"]["fingerprint"], reasons=[]))
    rows = rr.RequirementsLedger.evaluate(doc, memo=memo, jobs=1)
    assert counted == ["candidate-v1"] and rows["candidate-v1"]["action"] == "resume"


def test_evaluate_fans_rows_out_over_worker_processes(monkeypatch):
    doc = ledger_doc(6)
    serial = rr.RequirementsLedger.evaluate(doc, jobs=1)
    monkeypatch.setattr(rr, "POOL_MIN_ROWS", 2)
    stats = {}
    parallel = rr.RequirementsLedger.evaluate(doc, jobs=2, stats=stats)
    assert parallel == serial and list(parallel) == list(doc["requirements"])
    assert stats.get("pool_error") is None and stats["jobs"] == 2


def test_the_discovery_rounds_evaluate_in_process_and_remember_nothing_they_could_not_finish(monkeypatch):
    import release_requirement_dependencies as dependencies
    doc = ledger_doc(3)
    memo = rr.EvaluationMemo()
    monkeypatch.setattr(rr, "POOL_MIN_ROWS", 1)
    original = rr.RequirementsLedger._evaluate_row

    def asks_for_a_snapshot(*a, **kw):
        if a[1] == "candidate-v0":
            dependencies._DISCOVERY.get()["snapshot"] = ("root", COMMIT, "request", None)
        return original(*a, **kw)
    monkeypatch.setattr(rr.RequirementsLedger, "_evaluate_row", staticmethod(asks_for_a_snapshot))
    token = dependencies._DISCOVERY.set({})
    try:
        rr.RequirementsLedger.evaluate(doc, memo=memo, jobs=4)
    finally:
        dependencies._DISCOVERY.reset(token)
    assert sorted(memo.rows) == ["candidate-v1", "candidate-v2"]


# ------------------------------------------------------------ before the bill
def test_ready_to_rent_names_every_missing_input():
    missing = once.ready_to_rent({"nvidia": ["gate-nvidia: owed"], "amd": ["gate-amd: owed"]},
                                 wheels=lambda v: "absent linux.whl" if v == "amd" else None,
                                 payload=lambda v: None, plan=lambda: None,
                                 ledger=lambda: (_ for _ in ()).throw(OSError("no key")))
    assert missing == ["ledger document: OSError: no key", "amd wheels: absent linux.whl"]
    assert once.ready_to_rent({"nvidia": []}, wheels=lambda v: None, payload=lambda v: None,
                              plan=lambda: None, ledger=lambda: None) == []


def test_gate_boxes_refuses_before_the_bill_with_the_missing_input_named(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_ORCHESTRATOR", "1")
    r = release.Release(args(tmp_path / "state"), runner=lambda *a, **k: 0)
    r.state["commit"] = COMMIT
    r.say = lambda line: None
    monkeypatch.setattr(r, "box_needs", lambda: {"nvidia": ["gate-nvidia: no gate report"]})
    monkeypatch.setattr(r, "saved_plan", lambda: None)
    monkeypatch.setattr(r, "box_plan", lambda needs: [])
    monkeypatch.setattr(r, "ledger_document", lambda: ("k" * 24, None))
    monkeypatch.setattr(r, "source_checkout", lambda create=True: tmp_path / "src")
    rented = []
    monkeypatch.setattr(r, "nvidia_box_up", lambda: rented.append("nvidia"))
    with pytest.raises(release.StepFailed) as error:
        r.step_gate_boxes()
    text = str(error.value)
    assert rented == []
    assert "rents nothing" in text and "no reuse plan" in text and "ledger document: no evaluation document" in text
    assert "nvidia wheels: StepFailed: candidate gate requires the packed linux wheel" in text
    assert "nvidia payload" in text and "release_gate.py" in text


# ------------------------------------------------------------ the idle-box watchdog
def test_the_watchdog_reports_and_totals_idle_minutes_on_a_fake_clock():
    now = [0.0]
    state, said = {}, []
    watch = once.IdleBoxWatch(say=said.append, clock=lambda: now[0],
                              rate=lambda v: 0.86 if v == "nvidia" else None, state=state, save=lambda: None)
    held = {"nvidia": dict(pod="nvc1"), "amd": dict(droplet="555")}
    for t in (0, 60, 120):
        now[0] = t
        watch.tick(held, busy={"amd"})
    assert said == ["  IDLE BOX nvidia nvc1 2 min (~$0.03)"]
    assert state["idle_boxes"]["nvidia"]["minutes"] == 2.0 and "amd" not in state["idle_boxes"]
    for t in (180, 240):                       # a box job runs: busy, nothing counted
        now[0] = t
        watch.tick(held, busy={"nvidia", "amd"})
    for t in (300, 360, 420):                  # idle again from the last busy tick (240)
        now[0] = t
        watch.tick(held, busy=set())
    assert state["idle_boxes"]["nvidia"]["minutes"] == 5.0          # 2 + 3
    assert state["idle_boxes"]["amd"]["minutes"] == 3.0 and state["idle_boxes"]["amd"]["rate_unknown"]
    assert state["idle_box_minutes"] == 8.0
    assert state["idle_box_cost"] == pytest.approx(round(0.86 * 5 / 60, 2))
    # a line every two minutes of a streak, each naming the streak so far
    assert said[1:] == ["  IDLE BOX amd 555 2 min (~$?)", "  IDLE BOX nvidia nvc1 2 min (~$0.03)"]
    assert once.summary_line(state).startswith("idle boxes: 8 min")


def test_the_release_counts_a_box_busy_only_while_its_job_runs(tmp_path):
    r = release.Release(args(tmp_path / "state"), runner=lambda *a, **k: 0)
    r.state["commit"] = COMMIT
    r.state["boxes"] = {"nvidia": dict(pod="nvc1", rented=True)}
    said, now = [], [0.0]
    r.say = said.append
    r.clock = lambda: now[0]
    with mock.patch.dict("os.environ", {"MOJOLEARN_BOX_HOURLY_NVIDIA": "1.2"}):
        with r.box_job("nvidia"):
            for t in (0, 120, 240):
                now[0] = t
                r.watch_idle_boxes()
        assert said == [] and r.state.get("idle_box_minutes", 0) == 0
        for t in (300, 420):
            now[0] = t
            r.watch_idle_boxes()
    assert said == ["  IDLE BOX nvidia nvc1 3 min (~$0.06)"]
    assert r.state["idle_box_minutes"] == 3.0 and r.state["idle_box_cost"] == 0.06
    saved = json.loads(r.state_path.read_text())
    assert saved["idle_box_minutes"] == 3.0


def test_the_hourly_rate_comes_from_the_provider_state(tmp_path):
    env = tmp_path / "state.env"
    env.write_text("POD_ID=x\nCOST_HR='0.86'\n")
    assert once.hourly_rate("nvidia", env, environ={}) == 0.86
    env.write_text("DROPLET_ID=1\nPRICE=1.99\n")
    assert once.hourly_rate("amd", env, environ={}) == 1.99
    assert once.hourly_rate("amd", env, environ={"MOJOLEARN_BOX_HOURLY_AMD": "2.5"}) == 2.5
    assert once.hourly_rate("amd", tmp_path / "absent", environ={}) is None


# ------------------------------------------------------------ after identity-admit: no ledger at all
@pytest.fixture
def no_ledger(monkeypatch):
    """Any ledger consultation fails the test (readiness_status, the manifest, release_gate, a row)."""
    def never(*a, **k):
        raise AssertionError("a step after identity-admit consulted the requirements ledger")
    for name in ("readiness_status", "readiness_manifest_keyed", "readiness_configuration", "release_gate",
                 "readiness"):
        monkeypatch.setattr(release.Release, name, never)
    monkeypatch.setattr(rr.RequirementsLedger, "_evaluate_row", staticmethod(never))


def admitted_release(tmp_path, monkeypatch, wheels=None, **kw):
    r = release.Release(args(tmp_path / "state", identity_on_wheels=True, **kw), runner=lambda *a, **k: 0)
    r.state["commit"] = COMMIT
    r.say = lambda line: None
    monkeypatch.setattr(r, "candidate_verdict", lambda vendor: (True, "PASS"))
    if wheels is not None:
        r.state["identity_admitted"] = dict(source_commit=COMMIT, commit="e" * 40, wheels=wheels)
    return r


def test_the_publication_gate_after_identity_admit_reads_its_record_only(tmp_path, monkeypatch, no_ledger):
    wheel = tmp_path / "mojolearn_core.whl"
    wheel.write_bytes(b"admitted bytes")
    r = admitted_release(tmp_path, monkeypatch, wheels={"linux": release.sha256(wheel)})
    assert r.publication_gate(stage="pre", platform="linux", wheel=wheel) == (True, [])
    wheel.write_bytes(b"other bytes")
    ready, reasons = r.publication_gate(stage="pre", platform="linux", wheel=wheel)
    assert not ready and "is not the wheel identity-admit admitted for linux" in reasons[0]
    r.state.pop("identity_admitted")
    ready, reasons = r.publication_gate(vendors=["nvidia"])
    assert not ready and reasons == [f"identity-admit has no admission of {COMMIT[:12]}: run identity-admit"]


def test_the_record_after_identity_admit_consults_no_ledger(tmp_path, monkeypatch, no_ledger):
    r = admitted_release(tmp_path, monkeypatch, publish="pypi")
    monkeypatch.setattr(r, "published_platforms", lambda: ["linux", "nvidia"])
    with pytest.raises(release.StepFailed, match="identity-admit's admission: identity-admit has no admission"):
        r.step_record()


def test_identity_admit_is_not_followed_by_a_ledger_prefetch():
    assert "identity-admit" not in release.Release.LEDGER_PREFETCH_AFTER
