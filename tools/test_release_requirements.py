# SPDX-License-Identifier: Apache-2.0
import copy
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

import release_requirements as R


def manifest(revision="a", arithmetic="one", compiler="1", flags="identical"):
    result = dict(schema=R.SCHEMA, version="0.8.57", source_revision=revision, requirements={})
    for name, source in (("identity", arithmetic), ("metadata", "constant")):
        inputs = dict(arithmetic={name + ".mojo": source}, compiler=compiler, flags=flags)
        result["requirements"][name] = dict(source_revision=revision, fingerprint=R.digest(inputs),
                                           inputs=inputs, scope=name, coverage=["whole operation"],
                                           kind="source", stage="pre", missing_inputs=[], required=True)
    return result


@pytest.fixture
def ledger(tmp_path):
    led = R.RequirementsLedger(tmp_path / "requirements.json")
    led.sync(manifest())
    return led


def record(ledger, tmp_path, name="identity", status="passed", reason=None):
    run = ledger.begin(name, reason)
    path = tmp_path / (run + ".log")
    path.write_text(status)
    ledger.record(run, status, [path], "whole operation checked", 0 if status == "passed" else 1,
                  2, 2 if status == "passed" else 1, failed=int(status == "failed"),
                  receipt=receipt_for(ledger, run, path, expected=2) if status == "passed" else None)
    return path


def receipt_for(ledger, run_id, path, expected=1):
    import release_requirement_receipts as P
    run = next(r for r in ledger.read()["runs"] if r["id"] == run_id)
    wanted = run["manifest"]
    return dict(schema=P.SCHEMA, requirement_id=run["requirement"], kind=wanted["kind"],
                source_revision=wanted["source_revision"], input_fingerprint=wanted["fingerprint"],
                producer=dict(tool="fixture", report_type="source"), status="passed", exit_code=0,
                coverage=[dict(id=name, status="passed", exit_code=0, expected=expected,
                               completed=expected, failed=0, skipped=0) for name in wanted["coverage"]],
                evidence=R.evidence_files([path]))


def test_new_commit_only_reuses_pass_and_preserves_tested_revision(ledger, tmp_path):
    record(ledger, tmp_path)
    ledger.sync(manifest(revision="new"))
    row = ledger.plan()["identity"]
    assert row["status"] == "passed" and row["action"] == "reuse"
    assert row["tested_revision"] == "a" and row["source_revision"] == "new"
    assert ledger.begin("identity") is None
    assert len(ledger.read()["runs"]) == 1


@pytest.mark.parametrize("field,value,reason", [("arithmetic", "two", "arithmetic/identity.mojo"),
                                               ("compiler", "2", "compiler"),
                                               ("flags", "other", "flags")])
def test_relevant_inputs_invalidate_with_specific_reason(ledger, tmp_path, field, value, reason):
    record(ledger, tmp_path)
    record(ledger, tmp_path, "metadata")
    ledger.sync(manifest(**{field: value}))
    row = ledger.plan()["identity"]
    assert row["status"] == "untested" and row["reasons"] == ["changed input: " + reason]
    if field == "arithmetic":
        assert ledger.plan()["metadata"]["status"] == "passed"
    assert len(ledger.read()["attempts"]) == 2


def test_missing_and_corrupt_evidence_require_restore_not_blind_rerun(ledger, tmp_path):
    path = record(ledger, tmp_path)
    path.write_text("tampered")
    assert ledger.plan()["identity"]["action"] == "restore-evidence"
    with pytest.raises(ValueError, match="restore provenance"):
        ledger.begin("identity")
    path.unlink()
    assert "evidence unavailable" in ledger.plan()["identity"]["reasons"][0]
    assert not ledger.gate()[0]


def test_later_failure_cannot_be_hidden_by_old_pass(ledger, tmp_path):
    record(ledger, tmp_path)
    record(ledger, tmp_path, status="failed", reason="negative control exposed wrong-device routing")
    assert ledger.plan()["identity"]["status"] == "failed"
    assert ledger.begin("identity") is not None
    assert ledger.read()["runs"][-1]["reasons"][0].startswith("previous attempt failed")


@pytest.mark.parametrize("kwargs", [dict(exit_code=1), dict(expected=0, completed=0),
                                    dict(completed=1), dict(failed=1), dict(skipped=1)])
def test_incomplete_or_failed_coverage_never_passes(ledger, tmp_path, kwargs):
    run = ledger.begin("identity")
    path = tmp_path / "report.log"
    path.write_text("summary")
    values = dict(exit_code=0, expected=2, completed=2)
    values.update(kwargs)
    with pytest.raises(ValueError, match="complete coverage"):
        ledger.record(run, "passed", [path], "checks", **values)


def test_inflight_old_source_result_does_not_qualify_changed_source(ledger, tmp_path):
    run = ledger.begin("identity")
    ledger.sync(manifest(revision="b", arithmetic="two"))
    path = tmp_path / "old.log"
    path.write_text("passed old source")
    ledger.record(run, "passed", [path], "old source", 0, 1, 1, receipt=receipt_for(ledger, run, path))
    assert ledger.plan()["identity"]["status"] == "untested"


def test_readonly_plan_does_not_rewrite_or_create_ledger(ledger, tmp_path):
    before = ledger.path.read_bytes()
    ledger.plan(manifest(revision="b", arithmetic="two"))
    assert ledger.path.read_bytes() == before
    absent = R.RequirementsLedger(tmp_path / "absent.json")
    assert not absent.gate()[0] and not absent.path.exists()


def test_parallel_producers_preserve_both_attempts(ledger, tmp_path):
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda name: record(ledger, tmp_path, name), ["identity", "metadata"]))
    assert len(ledger.read()["attempts"]) == 2
    assert ledger.gate() == (True, [])


def test_missing_configuration_cannot_admit_a_pass(ledger, tmp_path):
    target = manifest()
    target["requirements"]["identity"]["missing_inputs"] = ["compiler"]
    ledger.sync(target)
    assert ledger.plan()["identity"]["action"] == "complete-provenance"
    with pytest.raises(ValueError, match="missing input"):
        ledger.begin("identity")


def test_git_snapshot_ignores_docs_and_tracks_additions_deletions(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (tmp_path / "a.mojo").write_text("a")
    (tmp_path / "README").write_text("old")
    git("add", ".")
    git("commit", "-qm", "initial")
    catalog = {"requirements": [dict(id="id", scope="source", coverage=["source"],
                                     source_inputs={"arithmetic": ["*.mojo"]})]}
    a = R.requirements(tmp_path, git("rev-parse", "HEAD"), "v", catalog)
    (tmp_path / "README").write_text("new docs")
    git("commit", "-qam", "docs")
    b = R.requirements(tmp_path, git("rev-parse", "HEAD"), "v", catalog)
    assert a["requirements"]["id"]["fingerprint"] == b["requirements"]["id"]["fingerprint"]
    (tmp_path / "b.mojo").write_text("b")
    git("add", ".")
    git("commit", "-qm", "new arithmetic")
    c = R.requirements(tmp_path, "HEAD", "v", catalog)
    assert R.changes(b["requirements"]["id"]["inputs"], c["requirements"]["id"]["inputs"]) == ["added input: arithmetic/b.mojo"]


def test_record_is_immutable(ledger, tmp_path):
    path = record(ledger, tmp_path)
    run = ledger.read()["runs"][0]["id"]
    with pytest.raises(ValueError, match="already recorded"):
        ledger.record(run, "passed", [path], "second result", 0, 1, 1)


def test_flags_use_existing_cache_policy_and_include_linker_options():
    flags = R.build_flags({"MOJOLEARN_COMMIT": "new", "MOJOLEARN_COMPILE_JOBS": "2",
                           "MOJOLEARN_BUILD_JOBS": "16", "CXXFLAGS": "-DNEW=1"})
    assert flags == {"MOJOLEARN_COMPILE_JOBS": "2", "CXXFLAGS": "-DNEW=1"}


def test_explicit_redo_requires_a_specific_reason(tmp_path):
    import release
    from test_release import args
    ctx = release.Release(args(state_dir=str(tmp_path), redo="rehearsal", only="rehearsal"), runner=lambda *a: 0)
    with pytest.raises(SystemExit, match="--rerun-reason"):
        ctx.go()
    ctx.args.rerun_reason = "fixed receipt parser rejected the retained complete report"
    ctx.schedule = lambda steps: {"rehearsal": "done"}
    ctx.report = lambda status: True
    ctx.go()
    assert json.loads(ctx.state_path.read_text())["rerun_requests"][0]["reason"] == ctx.args.rerun_reason


def test_raw_log_and_aggregate_counts_cannot_create_a_pass(ledger, tmp_path):
    run = ledger.begin("identity")
    log = tmp_path / "source-only.log"
    log.write_text("one source test passed")
    with pytest.raises(ValueError, match="structured release requirement"):
        ledger.record(run, "passed", [log], "not an execution receipt", 0, 1, 1)
    assert ledger.plan()["identity"]["status"] == "untested"


def test_prepublication_does_not_require_future_postpublication_checks(ledger, tmp_path):
    target = manifest()
    target["requirements"]["metadata"]["stage"] = "post"
    ledger.sync(target)
    record(ledger, tmp_path)
    assert ledger.gate(stage="pre") == (True, [])
    assert not ledger.gate()[0] and not ledger.gate(stage="post")[0]


def test_producer_evidence_is_revalidated_when_reusing_a_pass(ledger, tmp_path):
    path = record(ledger, tmp_path)
    doc = ledger.read()
    doc["attempts"][0]["receipt"]["producer"]["report_type"] = "execution"
    ledger.path.write_text(json.dumps(doc))
    assert ledger.plan()["identity"]["status"] == "untested"


def test_unfinished_run_prevents_duplicate_launches(ledger):
    run = ledger.begin("identity")
    assert ledger.plan()["identity"]["active_run"] == run
    with pytest.raises(ValueError, match="duplicate work"):
        ledger.begin("identity")
    assert len(ledger.read()["runs"]) == 1


def test_explicit_rerun_holds_readiness_until_its_result(ledger, tmp_path):
    record(ledger, tmp_path)
    ledger.begin("identity", reason="new negative control may expose invalid retained result")
    assert ledger.plan()["identity"]["status"] == "untested"
    assert ledger.plan()["identity"]["action"] == "resume"
