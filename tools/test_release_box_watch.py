"""The release path without the reuse machinery (lane post-ledger-once, 2026-11-07): by default (identity on the
wheels, no --reuse-unchanged) no dependency analysis, reconstruction or ledger evaluation runs; after
identity-admit no step consults the ledger; gate-boxes refuses to rent with an input missing; the idle-box
watchdog's accounting on a fake clock. Pure Python: nothing is built, rented or run."""
import argparse
import json
import pathlib
import sys
from unittest import mock

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402
import release_box_watch as watch_  # noqa: E402
import release_requirements as rr  # noqa: E402

COMMIT = "c" * 40
TOOLING = "d" * 40


def args(state_dir, **kw):
    base = dict(version="0.8.66", dry_run=False, publish=None, only="", redo="", build_backend="gpu-legs",
                amd_expect_from="", smoke_gpu="", state_dir=str(state_dir), amd_build_provider=None,
                amd_provider="auto", smoke_via="rent", hopper_box="", smoke_branch="main", cpu_column=False)
    base.update(kw)
    return argparse.Namespace(**base)


# ------------------------------------------------------------ before the bill
def test_ready_to_rent_names_every_missing_input():
    missing = watch_.ready_to_rent({"nvidia": ["gate-nvidia: owed"], "amd": ["gate-amd: owed"]},
                                   wheels=lambda v: "absent linux.whl" if v == "amd" else None,
                                   payload=lambda v: (_ for _ in ()).throw(OSError("no tree")) if v == "nvidia" else None,
                                   plan=lambda: None)
    assert missing == ["amd wheels: absent linux.whl", "nvidia payload: OSError: no tree"]
    assert watch_.ready_to_rent({"nvidia": []}, wheels=lambda v: None, payload=lambda v: None,
                                plan=lambda: None) == []


def test_gate_boxes_refuses_before_the_bill_with_the_missing_input_named(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_ORCHESTRATOR", "1")
    r = release.Release(args(tmp_path / "state"), runner=lambda *a, **k: 0)
    r.state["commit"] = COMMIT
    r.say = lambda line: None
    monkeypatch.setattr(r, "box_needs", lambda: {"nvidia": ["gate-nvidia: no gate report"]})
    monkeypatch.setattr(r, "saved_plan", lambda: None)
    monkeypatch.setattr(r, "box_plan", lambda needs: [])
    monkeypatch.setattr(r, "source_checkout", lambda create=True: tmp_path / "src")
    rented = []
    monkeypatch.setattr(r, "nvidia_box_up", lambda: rented.append("nvidia"))
    with pytest.raises(release.StepFailed) as error:
        r.step_gate_boxes()
    text = str(error.value)
    assert rented == []
    assert "rents nothing" in text and "no reuse plan" in text
    assert "nvidia wheels: StepFailed: candidate gate requires the packed linux wheel" in text
    assert "nvidia payload" in text and "release_gate.py" in text


# ------------------------------------------------------------ the idle-box watchdog
def test_the_watchdog_reports_and_totals_idle_minutes_on_a_fake_clock():
    now = [0.0]
    state, said = {}, []
    watch = watch_.IdleBoxWatch(say=said.append, clock=lambda: now[0],
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
    assert watch_.summary_line(state).startswith("idle boxes: 8 min")


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
    assert watch_.hourly_rate("nvidia", env, environ={}) == 0.86
    env.write_text("DROPLET_ID=1\nPRICE=1.99\n")
    assert watch_.hourly_rate("amd", env, environ={}) == 1.99
    assert watch_.hourly_rate("amd", env, environ={"MOJOLEARN_BOX_HOURLY_AMD": "2.5"}) == 2.5
    assert watch_.hourly_rate("amd", tmp_path / "absent", environ={}) is None


# ------------------------------------------------------------ after identity-admit: no ledger at all
@pytest.fixture
def no_ledger(monkeypatch):
    """Any ledger consultation fails the test (readiness_status, the manifest, release_gate, a row)."""
    def never(*a, **k):
        raise AssertionError("a step after identity-admit consulted the requirements ledger")
    for name in ("readiness_status", "readiness_manifest", "release_gate", "readiness"):
        monkeypatch.setattr(release.Release, name, never)
    monkeypatch.setattr(rr.RequirementsLedger, "evaluate", staticmethod(never))


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


# ------------------------------------------------------------ the default path runs no reuse machinery
def test_the_default_path_runs_no_dependency_analysis_and_the_reuse_flag_does(tmp_path, monkeypatch):
    import release_requirement_dependencies as dependencies
    ran = []
    monkeypatch.setattr(dependencies, "prefetch", lambda compute, **kw: ran.append(1) or dict(requirements={}))
    default = admitted_release(tmp_path / "a", monkeypatch)
    monkeypatch.setattr(default, "prefetch_ledger_rounds", lambda after: ran.append(after) or dict(summary="ok"))
    assert not default.ledger_on()
    assert default.step_readiness() == release.Release.LEDGER_OFF and ran == []
    assert default.find_requirement_leg("cuda-sm_89", "cuda", "sm_89")[0] is None
    assert default.record_compile_requirements(None) is None and default.close_stopped_compile_producers(None) == []
    reuse = admitted_release(tmp_path / "b", monkeypatch, reuse_unchanged=True)
    monkeypatch.setattr(reuse, "readiness_manifest", lambda: dict(requirements={}))
    monkeypatch.setattr(reuse, "prefetch_ledger_rounds", lambda after: ran.append(after) or dict(summary="ok"))
    assert reuse.ledger_on()
    assert "requirement unit(s)" in reuse.step_readiness()[0] and ran == [1, "readiness"]
    # a release without --identity-on-wheels has no identity-admit: the ledger stays its gate
    legacy = release.Release(args(tmp_path / "c"), runner=lambda *a, **k: 0)
    assert legacy.ledger_on()


def test_the_default_gate_consults_no_ledger(tmp_path, monkeypatch, no_ledger):
    r = admitted_release(tmp_path, monkeypatch)
    monkeypatch.setattr(r, "gate_ok", lambda vendor, measure=True: (True, "PASS"))
    assert r.run_gate("nvidia") == "PASS"


def test_publish_binds_identity_admits_facts(tmp_path, monkeypatch):
    r = admitted_release(tmp_path, monkeypatch, wheels={"linux": "a" * 64, "macos": "b" * 64})
    r.state["identity_admitted"].update(table_sha256="t" * 64)
    facts = r.admitted_facts()
    assert facts["wheels"] == {"linux": "a" * 64, "macos": "b" * 64}
    assert facts["admitted_commit"] == "e" * 40 and facts["source_commit"] == COMMIT
    assert facts["table_sha256"] == "t" * 64 and facts["version"] == "0.8.66"
    script = (ROOT / "tools" / "release_linux_publish.sh").read_text()
    assert 'if [ -n "${MOJOLEARN_ADMITTED_FACTS:-}" ]' in script
    assert "admitted by identity-admit at" in script
