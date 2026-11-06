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
                published_smoke_reason="", identity_on_wheels=False, nvidia_columns="", reuse_unchanged=False)
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
    monkeypatch.setattr(r, "units_plan", lambda vendor: None)   # no packed wheels here: no units plan to offer
    assert r.gate_identity_args("nvidia") == ["--identity-columns", "nvidia,ptx", "--label-device", "l40s"]
    reuse = make(tmp_path / "r", identity_on_wheels=True, reuse_unchanged=True)
    monkeypatch.setattr(reuse, "units_plan", lambda vendor: None)
    assert "--reuse-unchanged" in reuse.gate_identity_args("amd")
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
    r = make(tmp_path, identity_on_wheels=True, reuse_unchanged=True)
    monkeypatch.setattr(r, "gate_ok", lambda vendor: (True, "PASS"))
    monkeypatch.setattr(r, "lanes_owed", lambda col: [])
    assert r.box_step_owed("gate-amd") is None
    assert any("amd: REUSED 0 / RUN 0 lane(s)" in l and "no rental" in l for l in r.lines)
    monkeypatch.setattr(r, "gate_ok", lambda vendor: (False, "no gate report"))
    monkeypatch.setattr(r, "lanes_owed", lambda col: ["knn"])
    monkeypatch.setattr(r, "units_plan", lambda vendor: dict(reused=25, owed=["install-load"]))
    assert r.box_step_owed("gate-nvidia") == "no gate report"
    assert any("units REUSED 25 / RUN 1" in l for l in r.lines)


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


def test_the_one_box_job_is_judged_by_identity_units_and_runtime(tmp_path, monkeypatch):
    import release_check_units as U
    r = make(tmp_path, identity_on_wheels=True)
    out = tmp_path / "gate"
    plan = dict(schema=U.PLAN_SCHEMA, units=[dict(unit="install-load", decision="run"), dict(unit="adapters", decision="reuse")])
    r.units_plan_path("amd").parent.mkdir(parents=True, exist_ok=True)
    r.units_plan_path("amd").write_text(json.dumps(plan))
    monkeypatch.setattr(r, "gate_runtime_error", lambda vendor, output: (None, 12))
    ok, why = r.single_gate_ok("amd", out)
    assert not ok and "no identity-wheels summary" in why and "no check-units summary" in why
    summary(out / "identity", ["amd"])
    (out / "units").mkdir(parents=True)
    (out / "units" / U.SUMMARY).write_text(json.dumps(dict(schema=U.SCHEMA + "+summary", units={
        "install-load": dict(decision="run", status="passed"), "adapters": dict(decision="reuse", status="reused")})))
    ok, why = r.single_gate_ok("amd", out)
    assert ok and "12 runtime cases" in why


def test_a_vendor_that_owes_nothing_is_composed_on_the_mac_and_rents_nothing(tmp_path, monkeypatch):
    r = make(tmp_path, identity_on_wheels=True, reuse_unchanged=True)
    verdicts = iter([(False, "no gate report"), (True, "PASS")])
    monkeypatch.setattr(r, "gate_ok", lambda vendor: next(verdicts))
    monkeypatch.setattr(r, "lanes_owed", lambda col: [])
    monkeypatch.setattr(r, "units_plan", lambda vendor: dict(reused=26, owed=[]))
    monkeypatch.setattr(r, "compose_without_box", lambda vendor: "composed")
    assert r.box_step_owed("gate-amd") is None
    assert any("nothing owed, composed on this Mac" in l and "no rental" in l for l in r.lines)


def test_a_vendor_whose_runtime_units_are_owed_rents(tmp_path, monkeypatch):
    r = make(tmp_path, identity_on_wheels=True, reuse_unchanged=True)
    monkeypatch.setattr(r, "gate_ok", lambda vendor: (False, "no gate report"))
    monkeypatch.setattr(r, "lanes_owed", lambda col: [])
    monkeypatch.setattr(r, "units_plan", lambda vendor: dict(reused=26, owed=[]))
    monkeypatch.setattr(r, "compose_without_box", lambda vendor: "runtime units owed")
    assert r.box_step_owed("gate-amd") == "no gate report"
    assert any("runtime units owed" in l for l in r.lines)


def test_cpu_with_nothing_owed_dispatches_nothing(tmp_path, monkeypatch):
    r = make(tmp_path, identity_on_wheels=True, reuse_unchanged=True)
    core = tmp_path / "core.whl"
    core.write_bytes(b"core")
    monkeypatch.setattr(r, "split_final", lambda package: core)
    monkeypatch.setattr(r, "lanes_owed", lambda col: [])
    monkeypatch.setattr(r, "units_plan", lambda vendor: dict(reused=20, owed=[]))

    def composed(vendor, wheels, device, out):
        out.mkdir(parents=True, exist_ok=True)
        (out / "runtime-report.json").write_text(json.dumps(dict(status="passed")))
        return True
    monkeypatch.setattr(r, "compose_runtime", composed)
    monkeypatch.setattr(r, "gh", lambda *a, **k: pytest.fail("no workflow dispatch when nothing is owed"))
    assert "no dispatch" in r.step_identity_cpu()


def test_an_unchanged_apple_adapter_unit_runs_nothing(tmp_path, monkeypatch):
    r = make(tmp_path, identity_on_wheels=True)
    monkeypatch.setattr(r, "units_plan", lambda vendor: dict(vendor="apple", device_class="apple-m2-pro", reused=1,
                        owed=[], units=[dict(unit="adapters", decision="reuse")]))
    monkeypatch.setattr(r, "apple_adapter_gate", lambda: pytest.fail("the adapter smoke must not run"))
    assert r.apple_adapter_unit_gate().startswith("adapters REUSED") and r.apple_adapter_unit_ok()


def test_the_default_runs_everything_and_never_asks_the_selector(tmp_path, monkeypatch):
    # Andrew 2026-11-05: the selector cannot be trusted to narrow; --reuse-unchanged is the only way to reuse
    r = make(tmp_path, identity_on_wheels=True)
    assert r.single() and not r.reuse()
    monkeypatch.setattr(r, "gate_ok", lambda vendor: (False, "no gate report"))
    monkeypatch.setattr(r, "lanes_owed", lambda col: pytest.fail("no lane selection in the full run"))
    monkeypatch.setattr(r, "compose_without_box", lambda vendor: pytest.fail("nothing is composed in the full run"))
    monkeypatch.setattr(r, "units_plan", lambda vendor: dict(reused=0, owed=["install-load", "adapters"]))
    assert r.box_step_owed("gate-nvidia") == "no gate report"
    assert any("REUSED 0 / RUN all lane(s); units RUN all" in l for l in r.lines)
    assert r.vendor_lanes("amd") == (0, "all")
    assert not make(tmp_path / "off").reuse()                    # the option off: nothing of this applies


def test_identity_at_freeze_never_holds_the_full_run(tmp_path, monkeypatch):
    def runner(cmd, env, log, detach=False):
        out = pathlib.Path(cmd[cmd.index("--json") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(schema="mojolearn.identity-at-freeze.v1", adopted=False, held=["nvidia", "amd"],
                                       columns={})))
        return 4
    r = make(tmp_path, runner=runner, identity_on_wheels=True)
    monkeypatch.setattr(r, "source_checkout", lambda create=True: tmp_path / "src")
    result, _ = r.step_identity_at_freeze()
    assert "recorded once from the built wheels" in result
