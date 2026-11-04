# SPDX-License-Identifier: Apache-2.0
"""tools/release.py carries a release from freeze to record with no hand steps (lane release-no-hands,
2026-11-04). Nothing here rents, reaches a box or builds: the runner and the box wrappers are fakes.

1. A step recorded done at this commit whose output still checks out returns its "have ..." result
   before its method runs, so no refusal at the top of a step stops a resumed release."""
import argparse
import os
import pathlib
import sys
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402

COMMIT = "c" * 40
OTHER = "d" * 40
VERSION = "0.8.66"


def args(**kw):
    base = dict(version=VERSION, dry_run=False, publish=None, only="", redo="", build_backend="github",
                amd_expect_from="", smoke_gpu="", state_dir="", amd_build_provider=None, amd_provider="auto",
                smoke_via="lq", hopper_box="", smoke_branch="main", cpu_column=False, gpu_smoke=False,
                owner_waiver=None, status=False, refreeze=False, source_checkout="")
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


def wheel(path, commit=COMMIT):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mojolearn/identity_columns/COMMIT", commit + "\n")
    return path


def record(r, step, commit=COMMIT, **data):
    r.state["steps"][step] = dict(done=True, commit=commit, at="2026-11-04T00:00:00Z", **data)


# ---------------------------------------------------------------- 1. done steps short-circuit
def test_a_done_step_with_its_output_returns_have_before_its_method_runs(tmp_path, monkeypatch):
    r = make(tmp_path)
    record(r, "linux-assemble", result="sets assembled")
    monkeypatch.setattr(r, "linux_packed", lambda: False)
    monkeypatch.setattr(r, "assembly_needed", lambda: True)
    monkeypatch.setattr(r, "assembled_ok", lambda: True)

    def refused():
        raise release.StepFailed("the refusal at the top of the method (0.8.65)")
    monkeypatch.setattr(r, "step_linux_assemble", refused)
    assert r.run_step("linux-assemble") == "done"
    assert any(line.startswith("-- linux-assemble: done") and "have " in line for line in r.lines)
    assert not r.state.get("failures")


def test_a_done_step_whose_output_is_gone_runs_its_method(tmp_path, monkeypatch):
    r = make(tmp_path)
    record(r, "macos-build", result="old")
    ran = []
    monkeypatch.setattr(r, "step_macos_build", lambda: ran.append(1) or "built")
    monkeypatch.setattr(r, "mark", lambda step, **d: None)
    assert r.run_step("macos-build") == "done"       # no wheel: the step ran
    assert ran == [1]
    ran.clear()
    wheel(r.rel / "macos" / f"mojolearn-{VERSION}-py3-none-macosx_26_0_arm64.whl")
    assert r.run_step("macos-build") == "done"       # the wheel of this commit: have, not run
    assert ran == []
    assert any("have mojolearn-" in line for line in r.lines)


def test_a_record_of_another_commit_or_a_failing_have_check_runs_the_step(tmp_path, monkeypatch):
    r = make(tmp_path)
    wheel(r.rel / "macos" / f"mojolearn-{VERSION}-py3-none-macosx_26_0_arm64.whl")
    record(r, "macos-build", commit=OTHER)
    ran = []
    monkeypatch.setattr(r, "step_macos_build", lambda: ran.append(1) or "built")
    monkeypatch.setattr(r, "mark", lambda step, **d: None)
    r.run_step("macos-build")
    assert ran == [1]
    record(r, "macos-build")

    def broken(rec):
        raise OSError("unreadable output")
    monkeypatch.setattr(r, "have_macos_build", broken)
    r.run_step("macos-build")
    assert ran == [1, 1]


def test_every_step_is_settled_by_its_record_a_have_check_or_is_a_freeze_step(tmp_path):
    # THE AUDIT AS A TEST: a step that is neither skipped on its record nor has a have_<step> check
    # re-executes on every resumed run; only the freeze steps (which check the frozen source) may.
    r = make(tmp_path)
    loose = [s for s in r.STEPS if s not in r.SKIP_IF_RECORDED
             and not hasattr(r, "have_" + s.replace("-", "_")) and not s.startswith("freeze-")]
    assert loose == []


def test_linux_assemble_returns_have_before_a_composition_refusal(tmp_path, monkeypatch):
    r = make(tmp_path)
    monkeypatch.setattr(r, "assembly_needed", lambda: True)
    monkeypatch.setattr(r, "assembled_ok", lambda: True)

    def composition():
        raise release.StepFailed("the host composition receipt is missing or changed")
    monkeypatch.setattr(r, "host_composition", composition)
    assert r.step_linux_assemble().startswith("have ")


def test_publish_takes_a_file_already_on_pypi_before_the_pre_publication_gate(tmp_path, monkeypatch):
    r = make(tmp_path, publish="pypi")
    w = wheel(r.rel / "macos" / f"mojolearn-{VERSION}-py3-none-macosx_26_0_arm64.whl")
    monkeypatch.setattr(r, "on_pypi", lambda wh: True)

    def gate(stage=None, vendors=None):
        raise AssertionError("the gate decides an upload, never a file the index already serves")
    monkeypatch.setattr(r, "release_gate", gate)
    assert r.publish("macos", w, None) == "already on PyPI"


# ---------------------------------------------------------------- 2. stale artifacts in the source checkout
import subprocess  # noqa: E402

import release_stale_artifacts as stale  # noqa: E402

PLAN = {"identical": ["_mojolearn_identical_0", "_mojolearn_identical_1"], "host": ["_mojolearn_host_0"]}
#: what 0.8.65 found (each cost a 75-minute build), plus a Linux-layout file and a per-binding leftover
STALE = ["_mojolearn_native.so", "_mojolearn_native_m1.so", "identical/_mojolearn_scoped_gemm_probe.so",
         "identical/_mojolearn_svm.so", "cuda/sm_89/_mojolearn_gbdt.so", "_mojolearn_merged.json"]
PRODUCED = ["identical/_mojolearn_identical_0.so", "identical/_mojolearn_merged.json", "host/_mojolearn_host_0.so"]
LEFT_ALONE = [".dylibs/libKGENCompilerRTShared.dylib", "_adapters/libmojolearn_adapter.so", "identical/notes.txt"]


def git(repo, *argv):
    return subprocess.run(["git", "-C", str(repo), *argv], check=True, capture_output=True, text=True).stdout.strip()


def fake_tree(tmp_path):
    """A source checkout: a tracked package (one tracked .so) plus untracked build leftovers."""
    src = tmp_path / "src"
    pkg = src / stale.PKG
    pkg.mkdir(parents=True)
    git(src, "init", "-q")
    (pkg / "__init__.py").write_text("")
    (pkg / "tracked_fixture.so").write_bytes(b"tracked")
    (src / ".gitignore").write_text("*.so\n")
    git(src, "add", "-f", ".gitignore", stale.PKG + "/__init__.py", stale.PKG + "/tracked_fixture.so")
    git(src, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "fixture")
    for rel in STALE + PRODUCED + LEFT_ALONE:
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_bytes(rel.encode())
    return src


def fake_plan(src, directory, byte_lm):
    return PLAN.get(directory, [])


def test_preflight_moves_every_stale_shipping_artifact_and_keeps_what_the_build_produces(tmp_path):
    src = fake_tree(tmp_path)
    said = []
    moved, receipt = stale.preflight(src, tmp_path / "rel/stale-artifacts", ("identical",), plan=fake_plan,
                                     log=said.append, stamp="20261104T000000Z")
    assert sorted(moved) == sorted(STALE)
    pkg = src / stale.PKG
    dest = tmp_path / "rel/stale-artifacts/20261104T000000Z" / stale.PKG
    for rel in STALE:
        assert not (pkg / rel).exists() and (dest / rel).read_bytes() == rel.encode()   # moved, never deleted
        assert any(f"moved {stale.PKG}/{rel}" in line for line in said)                   # each one logged
    for rel in PRODUCED + LEFT_ALONE + ["tracked_fixture.so", "__init__.py"]:
        assert (pkg / rel).exists()
    doc = __import__("json").loads(receipt.read_text())
    assert {row["path"] for row in doc["moved"]} == {f"{stale.PKG}/{rel}" for rel in STALE}
    assert all(len(row["sha256"]) == 64 for row in doc["moved"])
    # a second pre-flight finds nothing more
    assert stale.preflight(src, tmp_path / "rel/stale-artifacts", ("identical",), plan=fake_plan,
                           log=said.append)[0] == []


def test_preflight_dry_run_lists_and_moves_nothing(tmp_path):
    src = fake_tree(tmp_path)
    said = []
    moved, receipt = stale.preflight(src, tmp_path / "dest", ("identical",), plan=fake_plan, dry=True,
                                     log=said.append)
    assert sorted(moved) == sorted(STALE) and receipt is None
    assert all((src / stale.PKG / rel).exists() for rel in STALE)
    assert not (tmp_path / "dest").exists()
    assert sum("would move" in line for line in said) == len(STALE)


def test_a_per_binding_or_unreadable_plan_moves_every_untracked_artifact_before_the_build_only(tmp_path):
    src = fake_tree(tmp_path)

    def unreadable(src, directory, byte_lm):
        raise subprocess.CalledProcessError(2, "merged_bindings.py")
    # after the build (macos-smoke): only the unselected tier and the Linux layout are not this build's
    after, _ = stale.preflight(src, tmp_path / "d1", ("identical",), merged=False, phase="smoke", log=lambda m: None)
    assert sorted(after) == sorted(["_mojolearn_native.so", "_mojolearn_native_m1.so",
                                    "cuda/sm_89/_mojolearn_gbdt.so", "_mojolearn_merged.json"])
    # before the build, with the plan unreadable: every untracked shipping artifact left (the call above moved
    # its four), produced ones included, since the build rebuilds every output it expects
    before, _ = stale.preflight(src, tmp_path / "d2", ("identical",), plan=unreadable, log=lambda m: None)
    assert sorted(before) == sorted(set(STALE + PRODUCED) - set(after))


def release_on(tmp_path, src, monkeypatch, **kw):
    r = make(tmp_path, source_checkout=str(src), **kw)
    r.state["commit"] = git(src, "rev-parse", "HEAD")
    monkeypatch.setattr(stale, "merged_plan", fake_plan)
    return r


def test_macos_build_moves_the_stale_artifacts_before_the_build_starts(tmp_path, monkeypatch):
    src = fake_tree(tmp_path)
    seen = []

    def runner(cmd, env, log, detach=False):
        if "./packaging/macos/build_release_wheel.sh" in [str(c) for c in cmd]:
            seen.append([rel for rel in STALE if (src / stale.PKG / rel).exists()])
        return 0
    r = release_on(tmp_path, src, monkeypatch, runner=runner)
    monkeypatch.setattr(r, "plan", lambda: {"previous": None})
    monkeypatch.setattr(release.release_reuse, "plan_rows", lambda plan, platform=None, decision=None: [])
    monkeypatch.setattr(release.release_reuse, "parse_modes", lambda v: ("identical",))
    r.release_modes = ("identical",)
    with pytest.raises(release.StepFailed, match="expected one macOS wheel"):
        r.step_macos_build()          # the fake build makes no wheel; the pre-flight ran first
    assert seen == [[]]
    moved = r.state["stale_artifacts"]
    assert moved[0]["phase"] == "build" and moved[0]["moved"] == len(STALE)
    assert pathlib.Path(moved[0]["receipt"]).parent.parent == r.rel / "stale-artifacts"


def test_macos_smoke_runs_the_same_preflight_and_a_dry_run_moves_nothing(tmp_path, monkeypatch):
    src = fake_tree(tmp_path)
    r = release_on(tmp_path, src, monkeypatch, dry_run=True)
    r.release_modes = ("identical",)
    w = wheel(r.rel / "macos" / f"mojolearn-{VERSION}-py3-none-macosx_26_0_arm64.whl", r.commit)
    monkeypatch.setattr(r, "macos_self_test_passed", lambda: False)
    r.step_macos_smoke()
    assert all((src / stale.PKG / rel).exists() for rel in STALE)
    assert sum("would move" in line for line in r.lines) == len(STALE)
    r.dry = False
    monkeypatch.setattr(r, "smoke_python", lambda: "python3.12")
    with pytest.raises(release.StepFailed, match="self-test is not PASSED"):
        r.step_macos_smoke()
    assert not any((src / stale.PKG / rel).exists() for rel in STALE)
    assert r.state["stale_artifacts"][-1]["phase"] == "smoke" and w.exists()


# ---------------------------------------------------------------- 3. the gate boxes as steps
class FakeHold:
    """A held `nvidia_central.sh sh` (its process never runs here)."""
    def __init__(self, cmd):
        self.cmd, self.rc = cmd, None

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        self.rc = 0
        return 0

    def terminate(self):
        self.rc = -15


class Boxes:
    """Fake box wrappers behind the release's runner: `up` writes the state file the real one writes."""
    def __init__(self, tmp_path, monkeypatch, fetch_rc=0):
        self.nvc_state = tmp_path / "devpods"
        self.do_state = tmp_path / "do-amd-steward"
        monkeypatch.setenv("MOJOLEARN_DEVPOD_STATE", str(self.nvc_state))
        monkeypatch.setenv("MOJOLEARN_STEWARD_DO_STATE", str(self.do_state))
        monkeypatch.setenv("MOJOLEARN_RELEASE_NVC", "fake/nvidia_central.sh")
        monkeypatch.setenv("MOJOLEARN_RELEASE_DO_AMD", "fake/do_amd_steward.sh")
        self.calls, self.holds, self.fetch_rc = [], [], fetch_rc

    def nvidia_up(self, key="nvc1"):
        (self.nvc_state / key).mkdir(parents=True, exist_ok=True)
        (self.nvc_state / key / "state.env").write_text(f"POD_ID=pod-{key}\n")

    def amd_up(self, droplet="555"):
        self.do_state.mkdir(parents=True, exist_ok=True)
        (self.do_state / "state.env").write_text(f"DROPLET_ID={droplet}\nIP=10.0.0.1\n")

    def runner(self, cmd, env, log, detach=False):
        cmd = [str(c) for c in cmd]
        self.calls.append((cmd, dict(env or {})))
        if cmd[:2] == ["bash", "fake/nvidia_central.sh"] and cmd[2] == "up":
            self.nvidia_up()
        if cmd[:2] == ["bash", "fake/do_amd_steward.sh"] and cmd[2] == "up":
            self.amd_up()
        if cmd[:2] == ["bash", "fake/nvidia_central.sh"] and cmd[2] == "down":
            (self.nvc_state / cmd[3] / "state.env").unlink()
        if cmd[:2] == ["bash", "fake/do_amd_steward.sh"] and cmd[2] == "down":
            (self.do_state / "state.env").unlink()
        if "fetch" in cmd and "--tag" in cmd:
            return self.fetch_rc
        return 0

    def popen(self, cmd, **kw):
        self.holds.append(FakeHold(cmd))
        return self.holds[-1]

    def verbs(self):
        return [" ".join(c[1:3]) for c, _ in self.calls]


def box_release(tmp_path, monkeypatch, gates_ok=False, **kw):
    boxes = Boxes(tmp_path, monkeypatch, fetch_rc=kw.pop("fetch_rc", 0))
    r = make(tmp_path, runner=boxes.runner, **kw)
    r.box_popen = boxes.popen
    monkeypatch.setattr(r, "gate_ok", lambda vendor: (True, "PASS") if gates_ok else (False, "no gate report"))
    monkeypatch.setattr(r, "source_checkout", lambda create=True: tmp_path / "src")
    return r, boxes


def test_the_step_table_brings_the_boxes_up_before_the_gates_and_down_after_every_box_step():
    steps = [s for s, _, _, _ in release.STEP_TABLE]
    assert steps.index("gate-boxes") < steps.index("gate-nvidia") < steps.index("boxes-down")
    assert all("gate-boxes" in release.NEEDS[s] for s in release.BOX_CONSUMERS)
    assert set(release.BOX_CONSUMERS) <= set(release.AFTER["boxes-down"])
    assert release.RESOURCE["gate-boxes"] == release.RESOURCE["boxes-down"] == "boxes"


def test_gate_boxes_dry_run_prints_what_it_would_rent_and_for_how_long(tmp_path, monkeypatch):
    monkeypatch.delenv("MOJOLEARN_ORCHESTRATOR", raising=False)
    r, boxes = box_release(tmp_path, monkeypatch, dry_run=True)
    result = r.step_gate_boxes()
    text = "\n".join(r.lines)
    assert result == "would bring up: amd, nvidia" and boxes.calls == [] and boxes.holds == []
    assert "fake/nvidia_central.sh up 1" in text and "60 idle minutes" in text
    assert f"fake/do_amd_steward.sh up {r.AMD_LEASE_MIN}" in text and f"{r.AMD_LEASE_MIN}-minute lease" in text
    assert f"renewed by {r.AMD_RENEW_MIN} minutes every 85 minutes" in text
    assert "refuses this step without MOJOLEARN_ORCHESTRATOR=1" in text
    assert "mac" not in text.lower().replace("machine", "")   # never an Apple machine


def test_gate_boxes_is_orchestrator_only_and_rents_nothing_otherwise(tmp_path, monkeypatch):
    monkeypatch.delenv("MOJOLEARN_ORCHESTRATOR", raising=False)
    r, boxes = box_release(tmp_path, monkeypatch)
    with pytest.raises(release.StepFailed, match="ORCHESTRATOR ONLY"):
        r.step_gate_boxes()
    assert boxes.calls == []


def test_gate_boxes_is_not_needed_when_every_box_step_is_done(tmp_path, monkeypatch):
    r, boxes = box_release(tmp_path, monkeypatch, gates_ok=True)
    assert r.step_gate_boxes().startswith("not needed") and boxes.calls == []


def test_gate_boxes_rents_records_and_holds_then_boxes_down_pulls_results_and_takes_down(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_ORCHESTRATOR", "1")
    r, boxes = box_release(tmp_path, monkeypatch)
    r.selected = ["gate-boxes", "gate-nvidia", "gate-amd", "boxes-down"]
    r._status = {"gate-nvidia": "pending", "gate-amd": "pending"}
    monkeypatch.setattr(r, "keep_boxes", lambda: None)       # the keeper is driven by hand below
    result, data = r.step_gate_boxes()
    assert boxes.verbs() == ["fake/nvidia_central.sh up", "fake/nvidia_central.sh sync",
                             "fake/do_amd_steward.sh up"]
    assert boxes.calls[0][1].get("MOJOLEARN_ORCHESTRATOR") == "1"
    held = r.state["boxes"]
    assert held["nvidia"]["pod"] == "nvc1" and held["nvidia"]["rented"] is True
    assert held["amd"]["droplet"] == "555" and held["amd"]["rented"] is True
    assert "NVIDIA nvc1 rented" in result and "AMD droplet 555 rented" in result
    record(r, "gate-boxes")
    assert r.have("gate-boxes", r.recorded("gate-boxes")).startswith("have ")

    keeper = release.BoxKeeper(r)
    clock = [r.clock()]
    r.clock = lambda: clock[0]
    assert keeper.tick() == {"nvidia", "amd"}
    assert len(boxes.holds) == 1 and boxes.holds[0].cmd[2:4] == ["sh", "release-gate"]
    assert "RELEASE-HOLD-END" in boxes.holds[0].cmd[4]
    keeper.tick()
    assert len(boxes.holds) == 1                              # one hold while it runs
    boxes.holds[0].rc = 0
    keeper.tick()
    assert len(boxes.holds) == 2                              # an ended hold is started again while needed
    assert "fake/do_amd_steward.sh extend" not in boxes.verbs()
    clock[0] += r.AMD_RENEW_EVERY_S + 1
    keeper.tick()
    assert boxes.verbs().count("fake/do_amd_steward.sh extend") == 1
    r._status = {"gate-nvidia": "done", "gate-amd": "done"}
    assert keeper.tick() == set()                             # nothing pending: the hold is released
    assert any(c[2] == "sh" and "touch /root/release-hold/" in c[4] for c, _ in boxes.calls)
    assert keeper.hold is None

    # boxes-down: the job results that are not home are fetched first, then both rented boxes go down
    (r.rel / "gates").mkdir(parents=True)
    (r.rel / "gates" / "gate-nvidia.log").write_text("queued nvc1-0007, persisted as nvc-release-gate-1-ab\n")
    (r.rel / "gates" / "gate-amd.log").write_text("box_persist: job rel-gate-amd-2-cd persists to R2 ...\n")
    (r.rel / "gates" / "fetch-nvc-release-gate-1-ab" / "final").mkdir(parents=True)
    out = r.step_boxes_down()
    fetches = [c for c, _ in boxes.calls if "fetch" in c]
    assert [c[c.index("--tag") + 1] for c in fetches] == ["rel-gate-amd-2-cd"]
    assert "fake/nvidia_central.sh down" in boxes.verbs() and "fake/do_amd_steward.sh down" in boxes.verbs()
    assert r.held_boxes() == {} and "taken down" in out
    record(r, "boxes-down")
    assert r.have("boxes-down", r.recorded("boxes-down"))


def test_adopted_boxes_are_used_renewed_and_released_never_taken_down(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_ORCHESTRATOR", "1")
    r, boxes = box_release(tmp_path, monkeypatch)
    boxes.nvidia_up("nvc2")
    boxes.amd_up("777")
    monkeypatch.setattr(r, "keep_boxes", lambda: None)
    r.step_gate_boxes()
    assert "fake/do_amd_steward.sh up" not in boxes.verbs()
    assert "fake/do_amd_steward.sh extend" in boxes.verbs()  # an adopted droplet's lease covers the gates
    held = r.state["boxes"]
    assert held["nvidia"] == dict(held["nvidia"], pod="nvc2", rented=False)
    assert held["amd"]["droplet"] == "777" and held["amd"]["rented"] is False
    r.step_boxes_down()
    assert not any(v.endswith(" down") for v in boxes.verbs())
    assert (boxes.nvc_state / "nvc2" / "state.env").is_file() and (boxes.do_state / "state.env").is_file()
    assert r.held_boxes() == {}


def test_boxes_down_never_takes_down_a_box_whose_job_results_are_not_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_ORCHESTRATOR", "1")
    r, boxes = box_release(tmp_path, monkeypatch, fetch_rc=1)
    monkeypatch.setattr(r, "keep_boxes", lambda: None)
    r.step_gate_boxes()
    (r.rel / "gates").mkdir(parents=True)
    (r.rel / "gates" / "gate-nvidia.log").write_text("persisted as nvc-release-gate-9-ff\n")
    with pytest.raises(release.StepFailed, match="not home; nothing is taken down"):
        r.step_boxes_down()
    assert not any(v.endswith(" down") for v in boxes.verbs())
    assert set(r.held_boxes()) == {"nvidia", "amd"}


# ---------------------------------------------------------------- 4. the published-install producers as steps
import hashlib  # noqa: E402
import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

sys.path.insert(0, str(ROOT / "tools" / "ops"))
import published_install as PI  # noqa: E402

JOB = ROOT / "tools" / "ops" / "published_install_job.sh"


def test_the_producers_follow_their_publish_steps_and_record_waits_for_them():
    steps = [s for s, _, _, _ in release.STEP_TABLE]
    for step, (vendor, publish) in release.PUBLISHED_STEPS.items():
        assert steps.index(publish) < steps.index(step) < steps.index("boxes-down")
        assert publish in release.NEEDS[step] and step in release.AFTER["record"]
        if vendor != "apple":
            assert "gate-boxes" in release.NEEDS[step] and step in release.AFTER["boxes-down"]
    assert release.RESOURCE["published-apple"] == "mac" and "published-apple" not in release.BOX_CONSUMERS


def finals(r):
    """The split candidate wheels and the macOS wheel of this release, as linux-pack and macos-build leave them."""
    out = {}
    for package, (_, prefix) in release.SPLIT_PACKAGES.items():
        out[package] = wheel(r.rel / "linux" / "final" / f"{prefix}-{VERSION}-py3-none-manylinux_2_35_x86_64.whl")
    out["macos"] = wheel(r.rel / "macos" / f"mojolearn-{VERSION}-py3-none-macosx_26_0_arm64.whl")
    return out


def report(out, vendor, candidates, status="passed"):
    out.mkdir(parents=True, exist_ok=True)
    (out / "request.json").write_text(json.dumps(dict(vendor=vendor, version=VERSION, candidates={
        w.name: hashlib.sha256(w.read_bytes()).hexdigest() for w in candidates})))
    (out / "published-report.json").write_text(json.dumps(dict(
        schema=release.Release.PUBLISHED_SCHEMA, vendor=vendor, version=VERSION, status=status,
        exit_code=0 if status == "passed" else 1, coverage=[dict(id="index-install", completed=3, expected=3)])))


def test_published_steps_skip_without_pypi_and_take_a_passed_report_for_these_wheels(tmp_path, monkeypatch):
    r = make(tmp_path)
    assert r.step_published_nvidia() == "skipped (not published to PyPI)"
    r = make(tmp_path, publish="pypi")
    w = finals(r)
    report(r.published_out("nvidia"), "nvidia", [w["linux"], w["nvidia"], w["nvidia-ptx"]])
    assert r.published_ok("nvidia")[0] and r.step_published_nvidia().endswith("(report exists)")
    record(r, "published-nvidia")
    assert r.have("published-nvidia", r.recorded("published-nvidia")).startswith("PASSED index-install=3/3")
    # another wheel's report is not this release's
    report(r.published_out("amd"), "amd", [w["linux"], w["nvidia"]])
    assert not r.published_ok("amd")[0]
    needs = r.box_needs()
    assert any(n.startswith("published-amd: ") for n in needs["amd"])      # owed: gate-boxes keeps the droplet
    assert not any(n.startswith("published-nvidia") for n in needs.get("nvidia", []))


def test_published_apple_runs_the_producer_locally_under_the_metal_lock(tmp_path, monkeypatch):
    seen = []

    def runner(cmd, env, log, detach=False):
        cmd = [str(c) for c in cmd]
        seen.append(cmd)
        out = pathlib.Path(cmd[cmd.index("--out") + 1])
        report(out, "apple", [w["macos"]])
        return 0
    r = make(tmp_path, runner=runner, publish="pypi")
    w = finals(r)
    monkeypatch.setattr(r, "smoke_python", lambda: "python3.12")
    assert r.step_published_apple().startswith("PASSED")
    cmd = seen[0]
    assert cmd[2:6] == ["--wait-timeout", "3600", "metal", "--"] and cmd[1].endswith("tools/mac_slot.py")
    assert cmd[cmd.index("--vendor") + 1] == "apple" and cmd[cmd.index("--candidate-wheel") + 1] == str(w["macos"])
    assert cmd[cmd.index("--out") + 1] == str(r.rel / "published-apple")


def test_published_box_step_dry_run_prints_the_wrapper_commands(tmp_path, monkeypatch):
    r = make(tmp_path, publish="pypi", dry_run=True)
    finals(r)
    monkeypatch.setattr(r, "source_checkout", lambda create=True: tmp_path / "src")
    assert r.step_published_amd() is None
    text = "\n".join(r.lines)
    assert "tools/ops/published_install.py amd" in text
    assert "would: bash tools/do_amd_steward.sh run rel-0.8.66-published-amd-" in text
    assert "published_install_job.sh" in text and "sha256sum" in text


class FakeBox:
    """The wrappers and R2 as one fake: uploads land in .files, the job's results come home without wheels."""

    def __init__(self, vendor, candidates):
        self.vendor, self.candidates, self.files, self.calls, self.lists = vendor, candidates, {}, [], 0
        self.ptag = (f"nvc-release-gate-20261104T000000Z-abc123" if vendor == "nvidia"
                     else f"rel-{VERSION}-published-amd-20261104T000000Z-def456")

    def run(self, argv, cwd=None, stdin=None, timeout=None, capture_output=True, text=True):
        self.calls.append(argv)
        out = SimpleNamespace(returncode=0, stdout="", stderr="")
        command = argv[-1]
        if argv[2:3] == ["submit"]:
            out.stdout = "nvc1-0009\n"
            out.stderr = f"box_persist: job {self.ptag} persists to R2 measurements/box-jobs/x/\n"
        elif argv[2] in ("sh", "ssh") and command.startswith("cat > "):
            self.files[command[len("cat > "):]] = stdin.read()
            out.stdout = b""
        elif argv[2] in ("sh", "ssh") and "sha256sum" in command:
            out.stdout = "".join(f"{hashlib.sha256(v).hexdigest()}  {k}\n" for k, v in self.files.items())
        elif argv[2] in ("sh", "ssh") and "READY" in command:
            out.stdout = "READY\n"
        elif argv[2] == "status":
            out.stdout = "job 0009: running\n"
        elif "box_persist.py" in argv[1] and "--list" in argv:
            self.lists += 1
            out.stdout = "MANIFEST state=final exit=0 updated=x uploads=1\n"
        elif "box_persist.py" in argv[1] and "--extract" in argv:
            final = pathlib.Path(argv[argv.index("--dest") + 1]) / "final"
            report(final / "out", self.vendor, self.candidates)
            (final / "out" / "wheel-manifest.json").write_text(json.dumps(dict(files=[
                dict(file=w.name, sha256=hashlib.sha256(w.read_bytes()).hexdigest()) for w in self.candidates])))
            (final / "job").mkdir()
            (final / "job" / "status.json").write_text(json.dumps(dict(phase="published", exit=0)))
        elif "release_published_install.py" in argv[1] and argv[2] == "judge":
            out.stdout = '{"status": "passed"}'
        return out

    def popen(self, argv, **kw):
        self.calls.append(argv)
        lines = [f"box_persist: job {self.ptag} persists to R2 measurements/box-jobs/x/\n", "published: running\n"]
        return SimpleNamespace(stdout=iter(lines), wait=lambda: 0, kill=lambda: None)


@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
def test_a_published_box_job_is_staged_launched_fetched_restored_and_rejudged(tmp_path, monkeypatch, vendor):
    monkeypatch.setattr(PI, "job_script_committed", lambda: None)
    monkeypatch.setattr(PI.rg, "on_origin_main", lambda commit: True)
    r = make(tmp_path, publish="pypi")
    w = finals(r)
    candidates = r.published_wheels(vendor)
    box = FakeBox(vendor, candidates)
    out = r.published_out(vendor)
    a = PI.parse([vendor, VERSION, "--source-commit", COMMIT, "--source-tree", str(tmp_path / "src"),
                  "--out", str(out), "--state", str(r.rel / "published" / f"{vendor}.job.json"),
                  *[x for c in candidates for x in ("--candidate-wheel", str(c))]])
    assert PI.Runner(a, run=box.run, sleep=lambda s: None, popen=box.popen).main() == 0
    launches = [c for c in box.calls if c[2:3] in (["submit"], ["run"])]
    assert len(launches) == 1 and launches[0][1] == PI.rg.VENDORS[vendor]["wrapper"]
    staged = {pathlib.PurePosixPath(k).name: v for k, v in box.files.items()}
    assert staged["published_install_job.sh"] == JOB.read_bytes()     # the committed job script, verbatim
    assert {c.name for c in candidates} <= set(staged)
    env = staged["published.env"].decode()
    assert f"PUB_VENDOR={vendor}" in env and f"PUB_SOURCE_COMMIT={COMMIT}" in env
    # the wheels the persisted results lack are restored from the equal candidates, then re-judged at home
    assert all((out / "wheels" / c.name).read_bytes() == c.read_bytes() for c in candidates)
    assert any(c[1] == "tools/release_published_install.py" and c[2] == "judge" for c in box.calls)
    assert r.published_ok(vendor)[0]
    assert (r.rel / "published" / f"fetch-{box.ptag}" / "final").is_dir()          # boxes-down sees it home
    assert w["macos"].exists()


def test_restore_wheels_never_restores_bytes_that_differ(tmp_path):
    good, bad = wheel(tmp_path / "c" / "a-1.whl"), wheel(tmp_path / "c" / "b-1.whl")
    out = tmp_path / "out"
    out.mkdir()
    (out / "wheel-manifest.json").write_text(json.dumps(dict(files=[
        dict(file="a-1.whl", sha256=hashlib.sha256(good.read_bytes()).hexdigest()),
        dict(file="b-1.whl", sha256="0" * 64)])))
    assert PI.restore_wheels(out, [good, bad], log=lambda m: None) == ["a-1.whl"]
    assert not (out / "wheels" / "b-1.whl").exists()


def fake_bin(tmp_path, *names):
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    for name in names:
        (d / name).write_text("#!/bin/sh\nexit 0\n")
        (d / name).chmod(0o755)
    return d


@pytest.mark.parametrize("tools,want", [(("nvidia-smi",), "nvidia"), (("rocm-smi",), "amd")])
def test_the_job_takes_its_vendor_from_the_box(tmp_path, tools, want):
    env = dict(os.environ, PATH=f"{fake_bin(tmp_path, *tools)}:/usr/bin:/bin")
    got = subprocess.run(["bash", str(JOB), "--detect-vendor"], capture_output=True, text=True, env=env)
    assert got.returncode == 0 and got.stdout.strip() == want


def test_the_job_refuses_a_box_of_another_vendor_or_none(tmp_path):
    env = dict(os.environ, PATH="/usr/bin:/bin")
    if subprocess.run(["sh", "-c", "command -v nvidia-smi || command -v rocm-smi"], env=env).returncode == 0:
        pytest.skip("this machine has a GPU vendor tool on /usr/bin")
    assert subprocess.run(["bash", str(JOB), "--detect-vendor"], capture_output=True, env=env).returncode == 2
    indir = tmp_path / "in"
    indir.mkdir()
    shutil.copy2(JOB, indir / JOB.name)
    (indir / "published.env").write_text(f"PUB_VENDOR=amd\nPUB_VERSION={VERSION}\nPUB_SOURCE_COMMIT={COMMIT}\n"
                                         "PUB_TREE=/root/x\nPUB_WHEELS=()\n")
    env = dict(env, PATH=f"{fake_bin(tmp_path, 'nvidia-smi')}:/usr/bin:/bin",
               MOJOLEARN_RESULTS_DIR=str(tmp_path / "res"))
    got = subprocess.run(["bash", str(indir / JOB.name)], capture_output=True, text=True, env=env)
    assert got.returncode == 3 and "this box is nvidia" in got.stderr
    assert not (tmp_path / "res" / "out").exists()


import shutil  # noqa: E402
