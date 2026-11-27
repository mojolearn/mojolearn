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


def test_macos_build_seeds_from_the_one_mac_cache_and_copies_its_archives_back(tmp_path, monkeypatch):
    # The one Mac cache (MOJOLEARN_BINCACHE_DIR here; ~/mojolearn-evidence/bincache-mac by default)
    # holds what tools/development_build.py --layout release compiled: it is the first seed of the
    # release-scoped cache, and the release's own archives are copied back into it after the build.
    import bincache
    shared = tmp_path / "bincache-mac"
    shared.mkdir()
    (shared / ("1" * 64 + ".tar.gz")).write_bytes(b"an earlier archive")
    monkeypatch.setenv("MOJOLEARN_BINCACHE_DIR", str(shared))
    src = fake_tree(tmp_path)
    seen = {}

    def runner(cmd, env, log, detach=False):
        if "./packaging/macos/build_release_wheel.sh" in [str(c) for c in cmd]:
            seen.update(env)
            built = pathlib.Path(env["MOJOLEARN_BINCACHE_DIR"])
            built.mkdir(parents=True, exist_ok=True)
            (src / "built.so").write_bytes(b"compiled by the release")
            fields = {"schema": bincache.SCHEMA, "script": "bindings/merged_build.sh", "args": ["gpu"]}
            key = bincache.key_of(fields)
            bincache.pack(built / (key + ".tar.gz"), key, fields, src, ["built.so"], {})
            seen["key"] = key
        return 0
    r = release_on(tmp_path, src, monkeypatch, runner=runner)
    monkeypatch.setattr(r, "plan", lambda: {"previous": None})
    monkeypatch.setattr(release.release_reuse, "plan_rows", lambda plan, platform=None, decision=None: [])
    monkeypatch.setattr(release.release_reuse, "parse_modes", lambda v: ("identical",))
    r.release_modes = ("identical",)
    with pytest.raises(release.StepFailed, match="expected one macOS wheel"):
        r.step_macos_build()          # the fake build makes no wheel; the write-back ran before that check
    assert seen["MOJOLEARN_BINCACHE_DIR"] == str(r.macos_bincache()), "the release keeps its own provenance cache"
    assert seen["MOJOLEARN_BINCACHE_SEED_DIRS"].split(os.pathsep)[0] == str(shared)
    assert (seen["MOJOLEARN_COMPILE_JOBS"], seen["MOJOLEARN_PACKAGE_BYTE_LM"]) == ("1", "1")
    copied = shared / (seen["key"] + ".tar.gz")
    assert copied.is_file() and copied.read_bytes() == (r.macos_bincache() / copied.name).read_bytes()
    assert any("one Mac cache" in line and "1 archive(s) copied" in line for line in r.lines)


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
    # the rental mechanics; what gate-boxes checks before it rents: test_release_ledger_once.py
    monkeypatch.setattr(r, "ready_to_rent", lambda needs: [])
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


def held_release(tmp_path, monkeypatch):
    """A release that brought both boxes up and has both gates pending, its keeper driven by hand."""
    monkeypatch.setenv("MOJOLEARN_ORCHESTRATOR", "1")
    r, boxes = box_release(tmp_path, monkeypatch)
    r.selected = ["gate-boxes", "gate-nvidia", "gate-amd", "boxes-down"]
    r._status = {"gate-nvidia": "pending", "gate-amd": "pending"}
    monkeypatch.setattr(r, "keep_boxes", lambda: None)
    r.step_gate_boxes()
    record(r, "gate-boxes")
    clock = [r.clock()]
    r.clock = lambda: clock[0]
    return r, boxes, clock


def test_the_keeper_holds_both_boxes_through_a_three_hour_mac_side_stretch(tmp_path, monkeypatch):
    # 0.8.66 run h: the Mac spent about two hours before the first box job; the held boxes must outlive that
    r, boxes, clock = held_release(tmp_path, monkeypatch)
    keeper = release.BoxKeeper(r)
    renewals = []
    for minute in range(0, 3 * 60 + 1):
        clock[0] = r.state["boxes"]["amd"]["lease_extended_at"] if minute == 0 else clock[0] + 60
        if minute % 50 == 49 and boxes.holds:
            boxes.holds[-1].rc = 0                       # a hold that ran out is started again
        assert keeper.tick() == {"nvidia", "amd"}
        n = boxes.verbs().count("fake/do_amd_steward.sh extend")
        if n > len(renewals):
            renewals.append(minute)
    # the NVIDIA hold is an `nvidia_central.sh sh` (nvq hold: the pod's lease counts it busy) for the whole stretch
    assert all(h.cmd[2:4] == ["sh", "release-gate"] for h in boxes.holds) and len(boxes.holds) >= 3
    assert sum(1 for h in boxes.holds if h.rc is None) == 1
    # the droplet's lease (now + AMD_RENEW_MIN at each extend) never comes within reach of running out
    assert renewals and renewals[0] <= r.AMD_RENEW_EVERY_S // 60 + 1
    marks = [0] + renewals + [3 * 60]
    assert max(b - a for a, b in zip(marks, marks[1:])) * 60 < r.AMD_RENEW_MIN * 60
    assert not any("BOX LOST" in line for line in r.lines)


def test_a_failed_amd_renewal_is_retried_within_five_minutes(tmp_path, monkeypatch):
    r, boxes, clock = held_release(tmp_path, monkeypatch)
    keeper = release.BoxKeeper(r)
    real = boxes.runner
    rcs = [1]
    monkeypatch.setattr(r, "runner", lambda cmd, env, log, detach=False:
                        rcs.pop(0) if ("extend" in [str(c) for c in cmd] and rcs) else real(cmd, env, log, detach))
    clock[0] += r.AMD_RENEW_EVERY_S + 1
    keeper.tick()                                        # the extend fails
    clock[0] += 301
    keeper.tick()                                        # and is tried again five minutes later
    assert boxes.verbs().count("fake/do_amd_steward.sh extend") >= 1 and rcs == []


def test_a_held_pod_that_dies_is_reported_box_lost_and_resume_rents_a_live_one(tmp_path, monkeypatch):
    r, boxes, clock = held_release(tmp_path, monkeypatch)
    keeper = release.BoxKeeper(r)
    assert keeper.tick() == {"nvidia", "amd"} and len(boxes.holds) == 1
    # the pod is deleted: its ssh hold ends rc 255 and the watcher retires its state
    boxes.holds[0].rc = 255
    (boxes.nvc_state / "nvc1" / "state.env").unlink()
    assert keeper.tick() == {"amd"}
    lost = [line for line in r.lines if "BOX LOST" in line]
    assert len(lost) == 1 and lost[0].lstrip().startswith("BOX LOST nvidia nvc1 at 20")
    assert len(boxes.holds) == 1                         # no hold started on a dead pod
    assert "nvidia" not in r.held_boxes() and r.state["boxes"]["nvidia"]["lost"]
    keeper.tick()
    assert len([line for line in r.lines if "BOX LOST" in line]) == 1   # reported once
    # resume: gate-boxes no longer counts as had, and its run rents a live pod (the AMD droplet is kept)
    assert r.have("gate-boxes", r.recorded("gate-boxes")) is None
    ups = boxes.verbs().count("fake/nvidia_central.sh up")
    r.step_gate_boxes()
    assert boxes.verbs().count("fake/nvidia_central.sh up") == ups + 1
    assert "fake/do_amd_steward.sh up" not in boxes.verbs()[-3:]
    assert r.held_boxes()["nvidia"]["pod"] == "nvc1" and not r.held_boxes()["nvidia"].get("lost")
    assert r.have("gate-boxes", r.recorded("gate-boxes")).startswith("have ")


def test_a_held_droplet_whose_state_is_gone_is_reported_box_lost(tmp_path, monkeypatch):
    r, boxes, clock = held_release(tmp_path, monkeypatch)
    keeper = release.BoxKeeper(r)
    (boxes.do_state / "state.env").unlink()
    assert keeper.tick() == {"nvidia"}
    assert any(line.lstrip().startswith("BOX LOST amd 555 at ") for line in r.lines)
    assert "fake/do_amd_steward.sh extend" not in boxes.verbs()[-1:]


def test_a_keeper_pass_that_raises_is_logged_and_the_keeper_keeps_going(tmp_path, monkeypatch):
    r, boxes, clock = held_release(tmp_path, monkeypatch)
    r.KEEPER_TICK_S = 0
    keeper = release.BoxKeeper(r)
    passes = []

    def tick():
        passes.append(1)
        if len(passes) == 1:
            raise OSError("ssh: connect timed out")
        return set() if len(passes) >= 3 else {"amd"}
    keeper.tick = tick
    keeper.loop()
    assert len(passes) == 3 and any("a pass failed" in line and "connect timed out" in line for line in r.lines)


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


# ---------------------------------------------------------------- 4. the published check (one CPU job on GitHub)
# Andrew 2026-11-27: after publication no step rents or runs a GPU; one CPU job on a free GitHub runner downloads
# every published file (sha256 equal to the candidate's), installs mojolearn from PyPI, imports it and runs
# verify --quick. The per-vendor published-<vendor> steps and the --published-smoke GPU route are removed.
import hashlib  # noqa: E402
import json  # noqa: E402

import release_published_install as RPI  # noqa: E402


def test_one_published_step_waits_for_every_publish_needs_no_box_and_record_waits_for_it():
    steps = [s for s, _, _, _ in release.STEP_TABLE]
    assert "published" in steps and steps.index("published") < steps.index("boxes-down") < steps.index("record")
    assert not any(s.startswith("published-") for s in steps)
    assert release.NEEDS["published"] == ["freeze-commit"]
    assert release.AFTER["published"] == list(release.PUBLISH_STEPS) and "published" in release.AFTER["record"]
    assert "published" not in release.BOX_CONSUMERS and "published" not in release.AFTER["boxes-down"]
    assert release.RESOURCE["published"] is None
    assert not hasattr(release, "PUBLISHED_STEPS") and not hasattr(release, "PUBLISHED_SMOKE_STEPS")
    assert not (ROOT / "tools" / "ops" / "published_install.py").exists()
    assert not (ROOT / "tools" / "ops" / "published_install_job.sh").exists()


def finals(r):
    """The split candidate wheels and the macOS wheel of this release, as linux-pack and macos-build leave them."""
    out = {}
    for package, (_, prefix) in release.SPLIT_PACKAGES.items():
        out[package] = wheel(r.rel / "linux" / "final" / f"{prefix}-{VERSION}-py3-none-manylinux_2_35_x86_64.whl")
    out["macos"] = wheel(r.rel / "macos" / f"mojolearn-{VERSION}-py3-none-macosx_26_0_arm64.whl")
    return out


def publish_all(r):
    for step in release.PUBLISH_STEPS:
        record(r, step, result="pypi via alpha-api")


def digest(w):
    return hashlib.sha256(w.read_bytes()).hexdigest()


def runner_report(out, candidates, published=None, quick=0):
    """The directory the workflow's artifact brings home: request.json and the report judged from it."""
    out.mkdir(parents=True, exist_ok=True)
    cands = {w.name: digest(w) for w in candidates}
    files = published or cands
    request = {"version": VERSION, "index_url": RPI.PYPI_INDEX, "candidates": cands,
               "runner": "https://github.com/o/r/actions/runs/1",
               "index_wait": dict(listed=True, waited_s=0, polls=1, missing=[]),
               "resolved": [dict(candidate=n, distribution=RPI.wheel_tags(n)["distribution"], pip_exit=0, files=[n],
                                 file=n, sha256=files[n], index_sha256=files[n]) for n in cands],
               "install": dict(exit=0, installed={"mojolearn": VERSION, "mojolearn-core": VERSION}),
               "import": dict(exit=0, version=VERSION), "verify_quick": dict(exit=quick)}
    (out / "request.json").write_text(json.dumps(request))
    (out / "published-report.json").write_text(json.dumps(RPI.judge(out)))


def test_the_published_step_skips_without_pypi_and_waits_for_the_linux_mojolearn(tmp_path):
    r = make(tmp_path)
    assert r.step_published() == "skipped (not published to PyPI)"
    r = make(tmp_path, publish="pypi")
    assert r.step_published()[0] == "nothing published yet"
    record(r, "publish-macos", result="pypi via alpha-api")
    with pytest.raises(release.StepFailed, match="cannot run"):
        r.step_published()
    assert r.calls == []


def test_a_passed_report_for_exactly_the_published_wheels_is_taken(tmp_path):
    r = make(tmp_path, publish="pypi")
    w = finals(r)
    publish_all(r)
    runner_report(r.published_out(), list(w.values()))
    ok, why = r.published_ok()
    assert ok and why.startswith("PASSED index-resolution=") and "verify-quick=1/1" in why
    assert r.step_published()[0].endswith("(report exists)") and r.calls == []
    # a report of other files is not this release's; a hand-edited report differs from its re-derivation
    runner_report(r.published_out(), [w["linux"], w["nvidia"]])
    assert not r.published_ok()[0]
    runner_report(r.published_out(), list(w.values()))
    doc = json.loads((r.published_out() / "published-report.json").read_text())
    (r.published_out() / "published-report.json").write_text(json.dumps(dict(doc, coverage=[])))
    assert "re-derivation" in r.published_ok()[1]
    assert not any(n.startswith("published") for v in r.box_needs().values() for n in v)


def test_published_bytes_that_differ_fail_naming_both_hashes(tmp_path):
    r = make(tmp_path, publish="pypi")
    w = finals(r)
    publish_all(r)
    cands = list(w.values())
    runner_report(r.published_out(), cands, published=dict({c.name: digest(c) for c in cands}, **{w["amd"].name: "a" * 64}))
    ok, why = r.published_ok()
    assert not ok and "a" * 64 in why and digest(w["amd"]) in why and "HASH MISMATCH" in why


def test_a_failed_verify_quick_fails_the_step(tmp_path):
    r = make(tmp_path, publish="pypi")
    w = finals(r)
    publish_all(r)
    runner_report(r.published_out(), list(w.values()), quick=1)
    ok, why = r.published_ok()
    assert not ok and "verify-quick" in why and r.have_published(None) is None


def test_the_step_dispatches_the_github_workflow_and_judges_the_artifact(tmp_path, monkeypatch):
    r = make(tmp_path, publish="pypi")
    w = finals(r)
    publish_all(r)
    seen = []

    def gh(*argv, log=None):
        argv = [str(a) for a in argv]
        seen.append(argv)
        if argv[:2] == ["run", "download"]:
            runner_report(pathlib.Path(argv[argv.index("-D") + 1]), list(w.values()))
        return 0, ""
    monkeypatch.setattr(r, "gh", gh)
    monkeypatch.setattr(r, "xcc_find_run", lambda token, workflow=None: ("77", "https://github.com/o/r/actions/runs/77"))
    result, data = r.step_published()
    assert result.startswith("PASSED") and data["run"].endswith("/77") and sorted(data["platforms"]) == sorted(
        r.published_platforms())
    (cmd,) = r.calls                                   # the dispatch; nothing else runs on this Mac
    assert cmd[:4] == ["gh", "workflow", "run", "published-check.yml"]
    sent = json.loads(next(a for a in cmd if a.startswith("candidates="))[len("candidates="):])
    assert sent == {x.name: digest(x) for x in w.values()}
    assert ["run", "watch", "77", "--exit-status", "--interval", "30"] in seen
    assert not any(word in " ".join(cmd) for word in ("nvidia_central", "do_amd_steward", "tools/ops/"))


def test_a_failed_attempt_is_kept_beside_the_next(tmp_path, monkeypatch):
    r = make(tmp_path, publish="pypi")
    w = finals(r)
    publish_all(r)
    runner_report(r.published_out(), list(w.values()), quick=1)
    monkeypatch.setattr(r, "gh", lambda *a, log=None: (1, ""))
    monkeypatch.setattr(r, "xcc_find_run", lambda token, workflow=None: ("78", "u"))
    with pytest.raises(release.StepFailed, match="published did not pass"):
        r.step_published()
    assert list(r.rel.glob("published-check.failed-*"))


def test_boxes_down_waits_only_for_the_gates(tmp_path, monkeypatch):
    r = make(tmp_path, publish="pypi", only="gate-boxes,gate-nvidia,gate-amd,published,boxes-down")
    monkeypatch.setattr(r, "schedule", lambda steps: {s: "skipped" for s in r.STEPS})
    monkeypatch.setattr(r, "report", lambda status: True)
    assert r.go() == 0
    waits = [s for s in release.AFTER["boxes-down"] if s in r.selected]
    assert waits == ["gate-boxes", "gate-nvidia", "gate-amd"]


def test_the_published_smoke_option_is_gone():
    with pytest.raises(SystemExit):
        release.release_parser().parse_args([VERSION, "--published-smoke", "amd"])
