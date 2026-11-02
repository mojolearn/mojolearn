"""A GitHub build leg's open begun compile runs are closed only when their
producer stopped: failed, cancelled or never dispatched are interrupted with
retained evidence and the leg relaunches; running refuses; succeeded is
collected, never interrupted. No gh, network or build: gh is a fake."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release  # noqa: E402
import release_compile_evidence as compile_evidence  # noqa: E402
import release_requirements as requirements  # noqa: E402
import release_stopped_producers as producers  # noqa: E402
from test_release_compile_evidence import fixture, write  # noqa: E402

RUN = 41364664603
SOURCE = "a" * 40
TITLE = f"release-linux-build sm_89 @ {SOURCE} [gha-20261101T000000Z-abcdef]"


class FakeGH:
    def __init__(self, view=None, listing=()):
        self.view, self.listing, self.calls = view, list(listing), []

    def __call__(self, *args):
        self.calls.append(args)
        if args[:2] == ("run", "view") and "--log-failed" in args:
            return 0, "build\tcompile\tmemory guard: peak above the runner's limit\n"
        if args[:2] == ("run", "view"):
            return (0, json.dumps(self.view)) if self.view else (1, "")
        if args[:2] == ("run", "list"):
            return 0, json.dumps(self.listing)
        raise AssertionError(f"unexpected gh call {args}")


def view(status, conclusion):
    return dict(databaseId=RUN, displayTitle=TITLE, status=status, conclusion=conclusion, headSha="b" * 40,
                url=f"https://github.com/o/r/actions/runs/{RUN}", createdAt="2026-11-01T00:00:00Z",
                updatedAt="2026-11-01T01:00:00Z", jobs=[])


def controller(tmp_path, gh, *, launched=True, run_id_file=True, monkeypatch=None):
    name, wanted, _, _ = fixture(tmp_path)
    wanted.update(scope="test", required=True, stage="pre", missing_inputs=[])
    manifest = dict(schema=requirements.SCHEMA, requirements={name: wanted}, version="0.8.64",
                    source_revision=wanted["source_revision"], source_root=str(tmp_path))
    ledger = requirements.RequirementsLedger(tmp_path / "requirements.json")
    ledger.sync(manifest)
    run = ledger.begin(name)
    out = tmp_path / "legs" / "cuda-sm_89"
    leg = release.Leg("cuda-sm_89", "cuda", "sm_89",
                      ["python3", "tools/release_github_build.py", "run", "--commit", SOURCE, "--arch", "sm_89",
                       "--out", str(out)], {}, out / "cuda-sm_89" / "release-build", tmp_path / "legs", out)
    leg.workdir.mkdir(parents=True, exist_ok=True)
    leg.log.write_text(f"[00:00:00 gha-build] dispatched https://github.com/o/r/actions/runs/{RUN}\n")
    leg.exit_file.write_text("1\n")
    if launched:
        launch = json.loads((tmp_path / "leg.provenance.json").read_text())
        launch["compile_requirements"]["runs"][name]["run_id"] = run
        launch["compile_requirements"]["selected"] = [name]
        launch["launched_at"] = "2026-11-01T00:00:00Z"
        write(leg.provenance_file, launch)
    if run_id_file:
        write(out / "GHA" / "run-id.json", dict(databaseId=RUN, url=f"https://github.com/o/r/actions/runs/{RUN}"))
    else:
        write(out / "GHA" / "dispatch-intent.json", dict(tag="gha-20261101T000000Z-abcdef"))
    ctx = SimpleNamespace(rel=tmp_path / "release", commit=SOURCE, dry=False, readonly=False, gh=gh,
                          say=lambda *_: None, readiness=lambda: ledger, readiness_manifest=lambda: manifest,
                          admitted_leg=lambda _: False, reused_leg_objects=lambda: [])
    for method in ("close_stopped_compile_producers", "leg_compile_names", "attach_succeeded_producer",
                   "record_compile_requirements", "record_compile_failure"):
        setattr(ctx, method, getattr(release.Release, method).__get__(ctx))
    if monkeypatch is not None:
        monkeypatch.setattr(release, "file_at", lambda *a: compile_evidence.CAPTURE_SCHEMA)
    return ctx, ledger, leg, name, run


def assert_interrupted(ledger, name, run, files):
    doc = ledger.read()
    [item] = doc["interruptions"]
    assert item["run_id"] == run and item["producer_stopped"] is True and item["qualification_claim"] is False
    assert requirements.evidence_errors(item["evidence"]) == []
    names = {Path(e["path"]).name for e in item["evidence"]}
    assert set(files) <= names and "producer.json" in names
    # The leg relaunches: a fresh run for the same requirement was begun.
    assert doc["runs"][-1]["requirement"] == name and doc["runs"][-1]["id"] != run


@pytest.mark.parametrize("conclusion", ["failure", "cancelled"])
def test_stopped_producer_runs_are_interrupted_and_the_leg_relaunches(tmp_path, monkeypatch, conclusion):
    gh = FakeGH(view("completed", conclusion))
    ctx, ledger, leg, name, run = controller(tmp_path, gh, monkeypatch=monkeypatch)
    launch = release.Release.begin_compile_requirements(ctx, leg)
    assert launch["schema"] == compile_evidence.LAUNCH_SCHEMA and launch["runs"][name]["run_id"] != run
    assert_interrupted(ledger, name, run, ["gh-run.json", "failed-log.txt", "launch-provenance.json"])
    excerpt = next(Path(e["path"]) for e in ledger.read()["interruptions"][0]["evidence"]
                   if e["path"].endswith("failed-log.txt"))
    assert "memory guard" in excerpt.read_text()
    assert not any("--run-id" in c for c in leg.command)


def test_never_dispatched_launch_is_interrupted_with_the_listing(tmp_path, monkeypatch):
    other = dict(databaseId=1, displayTitle=f"release-linux-build gfx942 @ {SOURCE} [gha-other]",
                 createdAt="2026-11-01T00:05:00Z", status="completed", conclusion="success")
    gh = FakeGH(listing=[other])
    ctx, ledger, leg, name, run = controller(tmp_path, gh, run_id_file=False, monkeypatch=monkeypatch)
    leg.log.write_text("[00:00:00 gha-build] binding cache map: ...\n")  # killed before the dispatch
    release.Release.begin_compile_requirements(ctx, leg)
    assert_interrupted(ledger, name, run, ["gh-run-list.json", "launch-provenance.json"])
    assert not any(c[:2] == ("run", "view") for c in gh.calls)


def test_begun_but_never_spawned_runs_are_interrupted_after_the_grace(tmp_path, monkeypatch):
    gh = FakeGH(listing=[])
    ctx, ledger, leg, name, run = controller(tmp_path, gh, launched=False, run_id_file=False,
                                             monkeypatch=monkeypatch)
    # Within the grace a launch may still be on its way: refuse.
    with pytest.raises(release.StepFailed, match="duplicate work"):
        release.Release.begin_compile_requirements(ctx, leg)
    assert ledger.read().get("interruptions", []) == []
    monkeypatch.setattr(producers, "GRACE_SECONDS", -1)
    release.Release.begin_compile_requirements(ctx, leg)
    assert_interrupted(ledger, name, run, ["gh-run-list.json"])


@pytest.mark.parametrize("status", ["in_progress", "queued"])
def test_running_producer_still_refuses(tmp_path, monkeypatch, status):
    gh = FakeGH(view(status, None))
    ctx, ledger, leg, name, run = controller(tmp_path, gh, monkeypatch=monkeypatch)
    before = len(ledger.read()["runs"])
    with pytest.raises(release.StepFailed, match=f"duplicate work.*producer.*{status}"):
        release.Release.begin_compile_requirements(ctx, leg)
    doc = ledger.read()
    assert doc.get("interruptions", []) == [] and len(doc["runs"]) == before


def test_succeeded_producer_is_collected_never_interrupted(tmp_path, monkeypatch):
    gh = FakeGH(view("completed", "success"))
    ctx, ledger, leg, name, run = controller(tmp_path, gh, monkeypatch=monkeypatch)
    before = len(ledger.read()["runs"])
    launch = release.Release.begin_compile_requirements(ctx, leg)
    assert launch["runs"][name]["run_id"] == run and launch["attached_github_run"] == RUN
    assert leg.command[-2:] == ["--run-id", str(RUN)]
    doc = ledger.read()
    assert doc.get("interruptions", []) == [] and len(doc["runs"]) == before
    assert not any("--log-failed" in c for c in gh.calls)


def test_unreadable_github_state_refuses(tmp_path, monkeypatch):
    gh = FakeGH(view=None)
    ctx, ledger, leg, name, run = controller(tmp_path, gh, monkeypatch=monkeypatch)
    with pytest.raises(release.StepFailed, match="duplicate work.*unknown"):
        release.Release.begin_compile_requirements(ctx, leg)
    assert ledger.read().get("interruptions", []) == []


def test_wait_path_closes_a_failed_producer_before_it_raises(tmp_path, monkeypatch):
    gh = FakeGH(view("completed", "failure"))
    ctx, ledger, leg, name, run = controller(tmp_path, gh, monkeypatch=monkeypatch)
    monkeypatch.setattr(release, "linux_legs", lambda _: [leg])
    with pytest.raises(release.StepFailed, match="build leg.*failed.*interrupted, producer stopped"):
        release.Release.step_linux_wait(ctx)
    [item] = ledger.read()["interruptions"]
    assert item["run_id"] == run and item["qualification_claim"] is False
    assert ledger.plan()[name]["action"] == "run"


def test_non_github_leg_is_left_to_the_ledger(tmp_path, monkeypatch):
    gh = FakeGH(view("completed", "failure"))
    ctx, ledger, leg, name, run = controller(tmp_path, gh, monkeypatch=monkeypatch)
    leg.command = ["python3", "tools/cpu_box_build.py"]
    with pytest.raises(release.StepFailed, match="duplicate work"):
        release.Release.begin_compile_requirements(ctx, leg)
    assert gh.calls == []
