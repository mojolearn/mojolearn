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
