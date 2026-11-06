# SPDX-License-Identifier: Apache-2.0
"""tools/release.py --identity-on-wheels (tools/release_identity_wheels.py): default off and inert; with it one
run per device class, nothing twice, admitted before any publish. Fakes only: no box, no GitHub, no lane."""
import argparse
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402
import release_identity_wheels as RIW  # noqa: E402
import release_requirements  # noqa: E402

COMMIT = "c" * 40
VERSION = "0.8.66"


def args(**kw):
    base = dict(version=VERSION, dry_run=False, publish="pypi", only="", redo="", build_backend="github",
                amd_expect_from="", smoke_gpu="", state_dir="", amd_build_provider=None, amd_provider="auto",
                smoke_via="lq", hopper_box="", smoke_branch="main", cpu_column=False, gpu_smoke=False,
                owner_waiver=None, status=False, refreeze=False, source_checkout="", published_smoke=[],
                published_smoke_reason="", identity_on_wheels=False, nvidia_columns="")
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


# ---------------------------------------------------------------- default off: nothing changes
def test_without_the_option_the_graph_and_steps_are_unchanged(tmp_path):
    r = make(tmp_path)
    assert not r.single() and r.AFTER is release.AFTER
    assert r.step_identity_cpu().startswith("skipped") and r.step_identity_admit().startswith("skipped")
    assert r.gate_identity_args("nvidia") == []
    w = tmp_path / "w.whl"
    assert r.shipped_wheel("linux", w) is w
    assert {"identity-cpu", "identity-admit"} <= release.Release.SKIP_IF_RECORDED
    for publish in release.PUBLISH_STEPS:
        assert "identity-admit" not in release.AFTER[publish]


def test_the_option_orders_every_publish_after_the_one_run(tmp_path):
    r = make(tmp_path, identity_on_wheels=True)
    assert r.single()
    assert set(r.AFTER["identity-admit"]) == {"gate-nvidia", "gate-amd", "gate-apple", "identity-cpu"}
    for publish in release.PUBLISH_STEPS:
        assert {"gate-apple", "identity-cpu", "identity-admit"} <= set(r.AFTER[publish])
    assert r.nvidia_columns() == ["nvidia", "ptx"]                                 # what main records today
    assert make(tmp_path / "b", identity_on_wheels=True, nvidia_columns="nvidia").nvidia_columns() == ["nvidia"]
    with pytest.raises(SystemExit, match="--nvidia-columns"):
        make(tmp_path / "c", identity_on_wheels=True, nvidia_columns="nvidia,amd").nvidia_columns()


def test_the_option_refuses_the_gpu_smoke_columns(tmp_path, monkeypatch):
    r = make(tmp_path, identity_on_wheels=True, gpu_smoke=True)
    monkeypatch.setattr(r, "schedule", lambda steps: pytest.fail("nothing may be scheduled"))
    with pytest.raises(SystemExit, match="replaces the --gpu-smoke"):
        r.go()


def test_one_box_job_per_vendor_records_the_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_IDENTITY_LABEL_NVIDIA", "l40s")
    r = make(tmp_path, identity_on_wheels=True, nvidia_columns="nvidia,ptx")
    assert r.gate_identity_args("nvidia") == ["--identity-columns", "nvidia,ptx", "--label-device", "l40s"]
    assert r.gate_identity_args("amd") == ["--identity-columns", "amd", "--label-device", "mi325x"]


def test_macos_smoke_runs_nothing_under_the_option(tmp_path):
    r = make(tmp_path, identity_on_wheels=True)
    assert r.step_macos_smoke().startswith("SKIPPED (--identity-on-wheels)") and r.calls == []


def summary(path, columns, ok=True):
    path.mkdir(parents=True, exist_ok=True)
    cols = {c: dict(label=c, owed=1, reused=3, rc=0 if ok else 1,
                    chunks=[dict(worker=0, lanes=1, rc=0, records=[dict(admissible=True)])]) for c in columns}
    (path / RIW.SUMMARY).write_text(json.dumps(dict(schema="mojolearn.identity-wheels.v1", ok=ok, columns=cols)))


def test_the_gate_must_match_the_mode(tmp_path):
    out = tmp_path / "gate"
    once = dict(identity_lanes="recorded-once")
    r = make(tmp_path, identity_on_wheels=True)
    assert r.gate_identity_errors("amd", out, {}) == [
        "--identity-on-wheels: the gate ran its identity-lane jobs again (no --no-identity-lanes)"]
    assert r.gate_identity_errors("amd", out, once) == ["identity: no identity-wheels summary"]
    summary(out / "identity", ["amd"])
    assert r.gate_identity_errors("amd", out, once) == []
    default = make(tmp_path / "d")
    assert default.gate_identity_errors("amd", out, {}) == []
    assert default.gate_identity_errors("amd", out, once)   # a dropped-lanes gate never passes the default path


def test_no_rental_when_the_vendor_owes_nothing(tmp_path, monkeypatch):
    r = make(tmp_path, identity_on_wheels=True)
    monkeypatch.setattr(r, "gate_ok", lambda vendor: (True, "PASS"))
    monkeypatch.setattr(r, "lanes_owed", lambda col: [])
    assert r.box_step_owed("gate-amd") is None
    assert any("amd: REUSED 0 / RUN 0 lane(s)" in l and "no rental" in l for l in r.lines)
    monkeypatch.setattr(r, "gate_ok", lambda vendor: (False, "no gate report"))
    monkeypatch.setattr(r, "lanes_owed", lambda col: ["knn"])
    assert r.box_step_owed("gate-nvidia") == "no gate report"


def test_nothing_publishes_before_the_admission_and_the_repack_ships(tmp_path):
    r = make(tmp_path, identity_on_wheels=True)
    gated = tmp_path / "core.whl"
    gated.write_bytes(b"gated")
    with pytest.raises(release.StepFailed, match="identity-admit has not run"):
        r.shipped_wheel("linux", gated)
    out = tmp_path / "repacked.whl"
    out.write_bytes(b"repacked")
    receipt = tmp_path / "core.repack.json"
    receipt.write_text(json.dumps(dict(input=dict(sha256=RIW.sha256(gated)), output=dict(path=str(out), sha256=RIW.sha256(out)))))
    r.state["identity_admitted"] = dict(source_commit=COMMIT, commit="a" * 40, repacked=dict(linux=str(receipt)))
    assert r.shipped_wheel("linux", gated) == out
    plugin = tmp_path / "nvidia.whl"
    assert r.shipped_wheel("nvidia", plugin) is plugin                 # plugins ship as built
    gated.write_bytes(b"other bytes")
    with pytest.raises(release.StepFailed, match="other bytes than the gated wheel"):
        r.shipped_wheel("linux", gated)


def test_admit_needs_the_whole_one_run(tmp_path):
    r = make(tmp_path, identity_on_wheels=True)
    with pytest.raises(release.StepFailed, match="gate-nvidia has not passed"):
        r.step_identity_admit()


def test_identity_at_freeze_reports_but_never_holds_under_the_option(tmp_path, monkeypatch):
    def runner(cmd, env, log, detach=False):
        out = pathlib.Path(cmd[cmd.index("--json") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(schema="mojolearn.identity-at-freeze.v1", adopted=False, held=["nvidia"],
                                       columns={})))
        return 4
    r = make(tmp_path, runner=runner, identity_on_wheels=True)
    monkeypatch.setattr(r, "source_checkout", lambda create=True: tmp_path / "src")
    result, _ = r.step_identity_at_freeze()
    assert "recorded once from the built wheels" in result


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()


def test_the_ledger_judges_identity_at_a_source_equal_admission_only(tmp_path, monkeypatch):
    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q")
    for rel, msg in (("python/mojolearn/a.py", "src"), (RIW.TABLE, "admit"), ("bindings/x.mojo", "fix")):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(msg)
        git(repo, "add", "-A")
        git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    f, a, later = git(repo, "rev-parse", "HEAD~2"), git(repo, "rev-parse", "HEAD~1"), git(repo, "rev-parse", "HEAD")
    L = release_requirements.RequirementsLedger
    assert L._reference_revision(repo, f) == f
    monkeypatch.setenv(RIW.REFERENCE_ENV, "%s:%s" % (f, a))
    assert L._reference_revision(repo, f) == a
    monkeypatch.setenv(RIW.REFERENCE_ENV, "%s:%s" % (f, later))
    with pytest.raises(ValueError, match="not source-equal"):
        L._reference_revision(repo, f)
