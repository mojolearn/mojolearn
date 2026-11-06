# SPDX-License-Identifier: Apache-2.0
"""ONE IDENTITY RUN PER RELEASE, BEFORE PUBLISH, NEVER AFTER (Andrew 2026-11-05; lane
identity-once-on-frozen-source). The release step graph and its steps, with fake runners: nothing here
runs a lane, reaches a box or builds.

1. identity-at-freeze runs right after freeze-commit and before anything is built or rented; it adopts a
   column recorded at a source-equal commit and HOLDS when one is owed.
2. No step that can run after a publish executes identity lanes on a device: every DEVICE_LANE_STEPS step
   settles before any publish, the published-smoke steps are gone, and each device step refuses to launch
   once a platform of the release is published."""
import argparse
import inspect
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402

COMMIT = "c" * 40
VERSION = "0.8.66"


def args(**kw):
    base = dict(version=VERSION, dry_run=False, publish="pypi", only="", redo="", build_backend="github",
                amd_expect_from="", smoke_gpu="", state_dir="", amd_build_provider=None, amd_provider="auto",
                smoke_via="lq", hopper_box="", smoke_branch="main", cpu_column=False, gpu_smoke=False,
                owner_waiver=None, status=False, refreeze=False, source_checkout="", published_smoke=[],
                published_smoke_reason="")
    base.update(kw)
    return argparse.Namespace(**base)


def make(tmp_path, runner=None, **kw):
    calls = []

    def default_runner(cmd, env, log, detach=False):
        calls.append([str(c) for c in cmd])
        return 0
    r = release.Release(args(state_dir=str(tmp_path / "state"), **kw), runner=runner or default_runner)
    r.state["commit"] = COMMIT
    r.lines = []
    r.say = r.lines.append
    r.calls = calls
    return r


def published(r, step="publish-macos", result="pypi via alpha-api-0.8.66-macos-20261105"):
    r.state["steps"][step] = dict(done=True, commit=COMMIT, at="2026-11-05T20:00:00Z", result=result)


def downstream(steps):
    """Every step that waits (NEEDS or AFTER), directly or not, for one of STEPS."""
    out, grew = set(steps), True
    while grew:
        grew = False
        for s in release.Release.STEPS:
            if s not in out and (set(release.NEEDS.get(s, ())) | set(release.AFTER.get(s, ()))) & out:
                out.add(s)
                grew = True
    return out - set(steps)


def upstream(step):
    out, todo = set(), [step]
    while todo:
        for n in release.NEEDS.get(todo.pop(), ()):
            if n not in out:
                out.add(n)
                todo.append(n)
    return out


# ---------------------------------------------------------------- 1. identity at the freeze
def test_identity_at_freeze_follows_the_freeze_and_precedes_every_build_rental_and_publish():
    steps = release.Release.STEPS
    assert steps.index("freeze-commit") < steps.index("identity-at-freeze") < steps.index("rehearsal")
    assert release.NEEDS["identity-at-freeze"] == ["freeze-commit"]
    assert release.NEEDS["rehearsal"] == ["identity-at-freeze"]
    assert release.NEEDS["cross-compile"] == ["identity-at-freeze"]
    for later in ("linux-builds", "macos-build", "gate-boxes", "gate-nvidia", "gate-amd", *release.PUBLISH_STEPS):
        assert "identity-at-freeze" in upstream(later), later
    assert "identity-at-freeze" in release.Release.SKIP_IF_RECORDED
    assert not "identity-at-freeze".startswith("freeze-")   # a --refreeze judges the new source again


def at_freeze_runner(report, rc):
    seen = []

    def runner(cmd, env, log, detach=False):
        cmd = [str(c) for c in cmd]
        seen.append(cmd)
        if report is not None:
            out = pathlib.Path(cmd[cmd.index("--json") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report))
        return rc
    return runner, seen


def report_doc(adopted, held=(), state="adopted"):
    return dict(schema="mojolearn.identity-at-freeze.v1", freeze=COMMIT, adopted=adopted, held=list(held),
                held_columns=["nvidia", "amd", "cpu", "apple"],
                columns={"nvidia": dict(column="nvidia", held=True, state=state, recorded_at="a" * 40, owed=0 if adopted else 3,
                                        source_paths=0 if adopted else 1, first_paths=[] if adopted else ["bindings/x.mojo"],
                                        reason="r", lanes=513)})


def test_identity_at_freeze_adopts_a_source_equal_record(tmp_path, monkeypatch):
    runner, seen = at_freeze_runner(report_doc(True), 0)
    r = make(tmp_path, runner=runner)
    monkeypatch.setattr(r, "source_checkout", lambda create=True: tmp_path / "src")
    result, data = r.step_identity_at_freeze()
    assert result.startswith("identity adopted at the freeze: nvidia recorded at aaaaaaaaaaaa")
    (cmd,) = seen
    assert cmd[1].endswith("tools/identity_at_freeze.py") and cmd[2] == "check"
    assert cmd[cmd.index("--root") + 1] == str(tmp_path / "src") and cmd[cmd.index("--freeze") + 1] == COMMIT
    assert data["identity_at_freeze"].endswith("identity-at-freeze/report.json")


def test_identity_at_freeze_holds_loudly_when_a_held_column_owes_lanes(tmp_path, monkeypatch):
    runner, _ = at_freeze_runner(report_doc(False, held=["nvidia"], state="owed"), 4)
    r = make(tmp_path, runner=runner)
    monkeypatch.setattr(r, "source_checkout", lambda create=True: tmp_path / "src")
    with pytest.raises(release.StepHeld, match="identity is owed at the freeze cccccccccccc on nvidia"):
        r.step_identity_at_freeze()
    assert any("IDENTITY RECORDED AT aaaaaaaaaaaa, FREEZE IS cccccccccccc: 3 lane(s) of nvidia owed" in l
               for l in r.lines)
    # held, not failed: run_step leaves it unrecorded, so the next run judges it again
    assert r.run_step("identity-at-freeze") == "held" and not r.recorded("identity-at-freeze")


def test_identity_at_freeze_without_a_report_fails(tmp_path, monkeypatch):
    runner, _ = at_freeze_runner(None, 2)
    r = make(tmp_path, runner=runner)
    monkeypatch.setattr(r, "source_checkout", lambda create=True: tmp_path / "src")
    with pytest.raises(release.StepFailed, match="without a report"):
        r.step_identity_at_freeze()


# ---------------------------------------------------------------- 2. nothing after the first publish
def test_no_identity_lane_step_can_run_after_a_publish():
    assert set(release.DEVICE_LANE_STEPS) <= set(release.Release.STEPS)
    after_publish = downstream(release.PUBLISH_STEPS)
    assert not after_publish & set(release.DEVICE_LANE_STEPS), after_publish & set(release.DEVICE_LANE_STEPS)
    for publish in release.PUBLISH_STEPS:
        # every publish waits for every identity-lane step to settle (in one run, nothing launches after it)
        assert set(release.DEVICE_LANE_STEPS) <= set(release.AFTER[publish]) | upstream(publish), publish
    assert release.PUBLISHED_SMOKE_STEPS == {}
    assert not any(s.startswith("published-smoke-") for s in release.Release.STEPS)
    assert not any(hasattr(release.Release, "step_published_smoke_" + v) for v in ("apple", "nvidia", "amd"))


def test_every_post_publish_step_is_a_read():
    # The steps that may run after a publish: the bytes checks, the finish line, boxes-down and the record.
    # None of them launches a lane: their code names no verifier, smoke or identity recorder.
    banned = ("qualify_verifier_wheel", "wheel_self_test", "record_identity_column", "reference_refresh.sh",
              "identity_box_job", "release_wheel_smoke", "release_identity_run", "gate_smoke_box",
              "tools/ops/published_install")
    for step in sorted(downstream(release.PUBLISH_STEPS)):
        source = inspect.getsource(getattr(release.Release, "step_" + step.replace("-", "_")))
        assert not [b for b in banned if b in source], step


@pytest.mark.parametrize("method,launch", [("step_macos_smoke", "self.must("), ("run_columns", "launch_detached("),
                                           ("launch_gate_job", "self.must(cmd"),
                                           ("launch_measure_smoke_job", "self.must(cmd")])
def test_each_device_lane_launch_is_guarded_before_it_launches(method, launch):
    source = inspect.getsource(getattr(release.Release, method))
    guard = source.find("self.refuse_device_lanes_after_publish(")
    assert guard != -1 and guard < source.find(launch), method


def test_the_guard_refuses_once_any_platform_is_published(tmp_path):
    r = make(tmp_path)
    r.refuse_device_lanes_after_publish("gate-amd")                     # nothing published: no refusal
    published(r, "publish-core-linux", result="none via alpha-api-0.8.66-linux-20261105")
    r.refuse_device_lanes_after_publish("gate-amd")                     # --publish none uploads nothing
    published(r)
    with pytest.raises(release.StepFailed, match="gate-amd would run identity lanes on a device after publish-macos"):
        r.refuse_device_lanes_after_publish("gate-amd")


def test_macos_smoke_refuses_to_launch_after_a_publish(tmp_path, monkeypatch):
    r = make(tmp_path)
    published(r, "publish-core-linux", result="pypi via alpha-api-0.8.66-linux-20261105")
    wheel = tmp_path / "mojolearn-0.8.66-py3-none-macosx_26_0_arm64.whl"
    wheel.write_bytes(b"w")
    monkeypatch.setattr(r, "macos_wheel", lambda: wheel)
    monkeypatch.setattr(r, "macos_self_test_passed", lambda: False)
    monkeypatch.setattr(r, "stale_preflight", lambda what: None)
    with pytest.raises(release.StepFailed, match="macos-smoke would run identity lanes"):
        r.step_macos_smoke()
    assert r.calls == []


def test_macos_smoke_with_a_receipt_is_a_read_after_a_publish(tmp_path, monkeypatch):
    r = make(tmp_path)
    published(r, "publish-core-linux")
    monkeypatch.setattr(r, "macos_wheel", lambda: tmp_path / "w.whl")
    monkeypatch.setattr(r, "macos_self_test_passed", lambda: True)
    assert r.step_macos_smoke() == "WHEEL-SELF-TEST PASSED (receipt exists)" and r.calls == []


def test_a_gpu_column_refuses_to_launch_after_a_publish(tmp_path, monkeypatch):
    r = make(tmp_path)
    published(r)

    class Leg:
        name = "amd"

        def describe(self):
            return "amd column"
    monkeypatch.setattr(r, "linux_final", lambda: tmp_path / "core.whl")
    monkeypatch.setattr(r, "column_legs", lambda names: [Leg()])
    monkeypatch.setattr(release, "launch_detached", lambda *a, **k: pytest.fail("launched after a publish"))
    with pytest.raises(release.StepFailed, match="gpu-column-amd would run identity lanes"):
        r.run_columns(("amd",))


def test_a_gate_refuses_to_launch_after_a_publish(tmp_path, monkeypatch):
    r = make(tmp_path)
    published(r)

    class Ledger:
        def begin(self, name):
            return "run-1"

        def record(self, *a, **k):
            pass
    monkeypatch.setattr(r, "readiness_status", lambda write=False: {})
    monkeypatch.setattr(r, "readiness", lambda: Ledger())
    monkeypatch.setattr(r, "gate_ok", lambda vendor, measure=True: (False, "no gate report"))
    monkeypatch.setattr(r, "gate_out", lambda vendor: tmp_path / "gate-amd" / "x-release-light-v1")
    monkeypatch.setattr(r, "source_checkout", lambda create=True: tmp_path / "src")
    monkeypatch.setattr(r, "candidate_gate_wheels", lambda vendor: [])
    monkeypatch.setattr(r, "gate_autograd_prior", lambda vendor: None)
    monkeypatch.setattr(r, "gate_runtime_prior", lambda vendor: None)
    with pytest.raises(release.StepFailed, match="gate-amd would run identity lanes"):
        r.run_gate("amd")
    assert r.calls == []
