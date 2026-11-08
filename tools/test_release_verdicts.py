# SPDX-License-Identifier: Apache-2.0
"""Fail soft (tools/release_verdicts.py, Andrew 2026-11-08): a gate passes on its HARD parts, SOFT failures are
recorded, notified and summarized, --strict restores the old verdict, a HARD failure always stops, and the
identity admission refuses a disagreeing column whatever the mode. Fakes only: no box, no GitHub, no lane."""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402
import release_evidence_set as ES  # noqa: E402
import release_verdicts as RV  # noqa: E402
from test_release_identity_wheels import COMMIT, VERSION, make, summary  # noqa: E402


# ---------------------------------------------------------------- the table
def test_the_table_classes_verification_hard_and_harness_soft():
    for name in ("identity", "identity-admit", "install-load", "shardcheck", "cpu/install", "cpu/correctness",
                 "apple/identity", "rehearsal/sha256", "publish", "record", "freeze", "build", "pack"):
        assert RV.check_class(name) == RV.HARD, name
    for name in ("autograd/linear-regression", "adapters", "runtime", "measuresmoke", "cpu/runtime", "cpu/autograd",
                 "cpu/adapters", "apple/adapters", "apple/runtime", "box-exit", "artifacts", "rehearsal/secondary"):
        assert RV.check_class(name) == RV.SOFT and RV.is_soft(name) and not RV.is_soft(name, strict=True), name
    assert RV.check_class("a-new-check") == RV.HARD and RV.check_class("new/thing") == RV.HARD   # unknown: HARD
    assert RV.rel_name("0.8.66") == "rel0866"


def test_a_soft_failure_is_recorded_once_logged_and_noticed(tmp_path):
    lines = []
    log = tmp_path / "x.log"
    log.write_text("first\nsecond\n")
    assert RV.record(tmp_path / "rel", VERSION, "amd", "adapters", "unit adapters did not pass", log, lines.append,
                     evidence_root=tmp_path / "ev")
    assert not RV.record(tmp_path / "rel", VERSION, "amd", "adapters", "unit adapters did not pass", log,
                         lines.append, evidence_root=tmp_path / "ev")
    (entry,) = RV.load(tmp_path / "rel")
    assert entry["first_lines"] == ["first", "second"] and entry["vendor"] == "amd"
    assert lines == ["SOFT FAIL amd adapters: unit adapters did not pass"]
    assert "| amd | adapters |" in (tmp_path / "ev" / "rel0866" / "SOFT-FAILURES.md").read_text()


# ---------------------------------------------------------------- the one box job
def box_job(r, out, units, shard=True, smoke_ok=False, runtime_ok=False, monkeypatch=None):
    import logical_shard_identity as LS
    import release_check_units as U
    import release_measurement_smoke as MS
    plan = dict(schema=U.PLAN_SCHEMA, units=[dict(unit=u, decision="run") for u in units])
    r.units_plan_path("amd").parent.mkdir(parents=True, exist_ok=True)
    r.units_plan_path("amd").write_text(json.dumps(plan))
    summary(out / "identity", ["amd"])
    (out / "units").mkdir(parents=True, exist_ok=True)
    (out / "units" / U.SUMMARY).write_text(json.dumps(dict(schema=U.SCHEMA + "+summary", units={
        u: dict(decision="run", status=s) for u, s in units.items()})))
    if shard:
        (out / "shardcheck").mkdir(parents=True, exist_ok=True)
        lanes = {n: dict(verdict="MATCH", reason="") for n in LS.lane_names()}
        (out / "shardcheck" / LS.SUMMARY).write_text(json.dumps(dict(schema=LS.SCHEMA, ok=True, counts=list(LS.COUNTS),
                                                                     lanes=lanes)))
    ids = ["algos/a/taxi/rows=2000"]
    smoke = out / "measuresmoke-job" / "measuresmoke"
    smoke.mkdir(parents=True, exist_ok=True)
    verdict = MS.PASS if smoke_ok else MS.FAIL
    (smoke / MS.SUMMARY).write_text(json.dumps(MS.summarize("amd", ids, {ids[0]: dict(verdict=verdict, reason="x",
                                                                                       batch=0)}, {})))
    monkeypatch.setattr(r, "gate_runtime_error", lambda vendor, output: (None, 7) if runtime_ok else (
        release.Release.RUNTIME_GATE_FAILED + "status failed in runtime-report.json: timer", None))


SOFT_UNITS = {"install-load": "passed", "autograd/linear-regression": "failed", "adapters": "failed"}


def test_a_soft_failure_passes_the_gate_and_reaches_the_summary(tmp_path, monkeypatch):
    r = make(tmp_path, identity_on_wheels=True)
    out = tmp_path / "gate"
    box_job(r, out, SOFT_UNITS, monkeypatch=monkeypatch)
    ok, why = r.single_gate_ok("amd", out)
    assert ok and "SOFT FAIL (fix lanes owed)" in why
    names = sorted(f["name"] for f in RV.load(r.rel))
    assert names == ["adapters", "autograd/linear-regression", "measuresmoke", "runtime"]
    assert sum(l.startswith("SOFT FAIL amd ") for l in r.lines) == 4
    assert (r.base / "rel0866" / "SOFT-FAILURES.md").is_file()   # a --state-dir run keeps the notice under it
    # read again (resume, status, the record): nothing is recorded twice
    assert r.single_gate_ok("amd", out)[0] and len(RV.load(r.rel)) == 4
    # the evidence set carries the failures and SUMMARY.md lists them
    ctx = ES.Context(version=VERSION, commit=COMMIT, tooling_commit=COMMIT, date="2026-11-08", rel=r.rel,
                     staged=tmp_path / "staged", published=["linux", "amd"], single=True)
    text = ES.derived_summary(ctx)
    assert "## Soft failures (fix lanes owed)" in text and "| amd | measuresmoke |" in text
    assert ES.soft_missing(ctx, "gates/amd/measuresmoke.json") and not ES.soft_missing(ctx, "gates/amd/shardcheck.json")


def test_strict_stops_on_a_soft_failure(tmp_path, monkeypatch):
    r = make(tmp_path, identity_on_wheels=True, strict=True)
    out = tmp_path / "gate"
    box_job(r, out, SOFT_UNITS, monkeypatch=monkeypatch)
    ok, why = r.single_gate_ok("amd", out)
    assert not ok and "unit adapters did not pass" in why and "measuresmoke" in why
    assert RV.load(r.rel) == []


@pytest.mark.parametrize("strict", [False, True])
def test_a_hard_failure_always_stops(tmp_path, monkeypatch, strict):
    r = make(tmp_path, identity_on_wheels=True, strict=strict)
    out = tmp_path / "gate"
    box_job(r, out, {"install-load": "failed"}, smoke_ok=True, runtime_ok=True, monkeypatch=monkeypatch)
    ok, why = r.single_gate_ok("amd", out)
    assert not ok and "unit install-load did not pass" in why
    out2 = tmp_path / "gate2"
    box_job(r, out2, {"install-load": "passed"}, shard=False, smoke_ok=True, runtime_ok=True, monkeypatch=monkeypatch)
    ok, why = r.single_gate_ok("amd", out2)
    assert not ok and "shardcheck" in why


def test_a_box_job_that_exited_1_on_a_soft_part_passes_the_gate(tmp_path, monkeypatch):
    r = make(tmp_path, identity_on_wheels=True)
    verdicts = iter([(False, "no report"), (False, "no report"), (True, "PASS job"), (True, "PASS")])
    monkeypatch.setattr(r, "gate_ok", lambda vendor, measure=True: next(verdicts))
    for name, value in (("candidate_gate_wheels", lambda v: [tmp_path / "core.whl"]),
                        ("source_checkout", lambda create=True: tmp_path / "src"),
                        ("gate_out", lambda v: tmp_path / "gate"), ("gate_autograd_prior", lambda v: None),
                        ("gate_runtime_prior", lambda v: None), ("gate_identity_args", lambda v: []),
                        ("refuse_device_lanes_after_publish", lambda step: None),
                        ("readiness_status", lambda write=False: None),
                        ("readiness", lambda: type("R", (), {"begin": lambda self, n: None})()),
                        ("launch_measure_smoke_job", lambda *a: None),
                        ("record_candidate_requirement", lambda *a, **k: None)):
        monkeypatch.setattr(r, name, value)

    def must(cmd, **kw):
        raise release.StepFailed("the amd release gate box job exited 1")
    monkeypatch.setattr(r, "must", must)
    assert r.run_gate("amd") == "PASS"
    assert [f["name"] for f in RV.load(r.rel)] == ["box-exit"]
    # --strict: the exit stops it as before
    r2 = make(tmp_path / "s", identity_on_wheels=True, strict=True)
    for name in ("candidate_gate_wheels", "source_checkout", "gate_out", "gate_autograd_prior", "gate_runtime_prior",
                 "gate_identity_args", "refuse_device_lanes_after_publish", "readiness_status", "readiness"):
        monkeypatch.setattr(r2, name, getattr(r, name))
    monkeypatch.setattr(r2, "gate_ok", lambda vendor, measure=True: (False, "no report"))
    monkeypatch.setattr(r2, "must", must)
    with pytest.raises(release.StepFailed, match="exited 1"):
        r2.run_gate("amd")


# ---------------------------------------------------------------- the CPU checks on GitHub
def cpu_run(r, tmp_path, monkeypatch, groups, conclusion="failure"):
    core = tmp_path / "core.whl"
    core.write_bytes(b"core")
    monkeypatch.setattr(r, "split_final", lambda package: core)
    out = r.cpu_out()
    (out / "artifacts" / "cpu-wheel-checks").mkdir(parents=True, exist_ok=True)
    (out / "artifacts" / "cpu-wheel-checks" / "checks.json").write_text(json.dumps(dict(checks=groups)))
    summary(out / "artifacts" / "cpu-wheel-identity-0", ["cpu"])
    (out / "run.json").write_text(json.dumps(dict(conclusion=conclusion, source_commit=COMMIT, run_id="9",
                                                  wheel_sha256=release.sha256(core), shards=1)))


def test_cpu_checks_pass_on_install_and_correctness(tmp_path, monkeypatch):
    r = make(tmp_path, identity_on_wheels=True)
    cpu_run(r, tmp_path, monkeypatch, dict(install=0, correctness=0, runtime=1, autograd=0, adapters=0))
    ok, why = r.cpu_ok()
    assert ok and "SOFT FAIL" in why and r._cpu_soft_groups == {"runtime"}
    assert sorted(f["name"] for f in RV.load(r.rel)) == ["cpu/job-exit", "cpu/runtime"]
    r.args.strict = True
    assert not r.cpu_ok()[0]
    r.args.strict = False
    cpu_run(r, tmp_path, monkeypatch, dict(install=0, correctness=1, runtime=0))
    ok, why = r.cpu_ok()
    assert not ok and "cpu correctness rc 1" in why


# ---------------------------------------------------------------- identity admission
@pytest.mark.parametrize("strict", [False, True])
def test_identity_admit_refuses_a_disagreeing_column_in_every_mode(tmp_path, monkeypatch, strict):
    r = make(tmp_path, identity_on_wheels=True, strict=strict)
    RV.record(r.rel, VERSION, "amd", "adapters", "unit adapters did not pass", evidence_root=tmp_path / "ev",
              say=lambda m: None)
    monkeypatch.setattr(r, "recorded", lambda step: dict(done=True))
    monkeypatch.setattr(r, "all_record_dirs", lambda: [str(tmp_path / "rec")])

    def refuse(dirs):
        raise release.StepFailed("the admission refused (rc 1; NVIDIA != AMD)")
    monkeypatch.setattr(r, "admit_records", refuse)
    with pytest.raises(release.StepFailed, match="admission refused"):
        r.step_identity_admit()
    assert r.admitted() is None


def test_identity_admit_carries_the_soft_failures(tmp_path, monkeypatch):
    import release_identity_wheels as RIW
    monkeypatch.delenv(RIW.REFERENCE_ENV, raising=False)   # the step sets it; restored after the test
    r = make(tmp_path, identity_on_wheels=True)
    r.soft_fail("amd", "adapters", "unit adapters did not pass")
    monkeypatch.setattr(r, "recorded", lambda step: dict(done=True))
    monkeypatch.setattr(r, "all_record_dirs", lambda: [])
    r.step_identity_admit()
    assert [f["name"] for f in r.admitted()["soft_failures"]] == ["adapters"]


# ---------------------------------------------------------------- the TestPyPI rehearsal
def test_rehearsal_secondary_checks_are_soft_and_the_sha256_hard(tmp_path):
    r = make(tmp_path, identity_on_wheels=True)
    rows = [dict(check="pip install mojolearn==0.8.66 on manylinux py3.12", problems=["pip could not resolve it (exit 1)"]),
            dict(check="pip install mojolearn-x.whl from TestPyPI into a fresh venv and import",
                 problems=["pip install exited 1 (macos-install.log)"]),
            dict(check="TestPyPI bytes of mojolearn-x.whl", problems=[])]
    problems = ["%s: %s" % (row["check"], p) for row in rows for p in row["problems"]]
    assert r.rehearsal_hard_problems(rows, problems, tmp_path / "rehearsal.json") == []
    assert [f["name"] for f in RV.load(r.rel)] == ["rehearsal/secondary"] * 2
    rows[2]["problems"] = ["TestPyPI served x sha256 a, gated b"]
    assert r.rehearsal_hard_problems(rows, problems, tmp_path / "rehearsal.json") == [
        "TestPyPI bytes of mojolearn-x.whl: TestPyPI served x sha256 a, gated b"]
    rows[1]["problems"] = ["import mojolearn exited 1 (macos-import.log)"]   # the import stays HARD
    assert len(r.rehearsal_hard_problems(rows, problems, tmp_path / "rehearsal.json")) == 2
    r.args.strict = True
    assert r.rehearsal_hard_problems(rows, problems, tmp_path / "rehearsal.json") == problems
