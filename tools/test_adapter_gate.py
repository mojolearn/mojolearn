# SPDX-License-Identifier: Apache-2.0
"""The release gate's installed-wheel adapter smoke: commands per vendor, the PASSED-only judge, the gate stage,
the Apple gate step and MOJOLEARN_RELEASE_ADAPTERS=0. Pure Python: every launch is faked."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapter_gate as AG  # noqa: E402
import gate_smoke_box as G  # noqa: E402
import release  # noqa: E402
RG = release.release_gate_module()

PHASES_OK = {p: {"exit": 0} for p in ("install-torch", "install-max") + AG.PHASES}


def receipt(status, phases=None, problems=()):
    return {"status": status, "phases": PHASES_OK if phases is None else phases, "problems": list(problems),
            "frameworks": {"torch": "ran", "max": "ran"} if status == "PASSED" else {}}


def fake_launch(status, calls, phases=None, venv_rc=0):
    """A launch that records argv and, for the smoke command, writes a receipt of STATUS to its --out."""
    def launch(argv, log):
        calls.append(list(argv))
        if any(str(a).endswith("adapter_wheel_smoke.py") for a in argv):
            out = Path(argv[argv.index("--out") + 1])
            if status is not None:
                (out / AG.RECEIPT).write_text(json.dumps(receipt(status, phases)))
            return 1 if status == "FAILED" else 0
        return venv_rc
    return launch


# ------------------------------------------------------------------ commands
@pytest.mark.parametrize("vendor,device", [("nvidia", "cuda:0"), ("amd", "cuda:0"), ("apple", "mps")])
def test_smoke_command_per_vendor(tmp_path, vendor, device):
    cmd = AG.smoke_command(vendor, tmp_path / "venv", tmp_path / "out", runner_python="/py")
    assert cmd[0] == "/py" and cmd[1].endswith("tools/adapter_wheel_smoke.py")
    assert cmd[cmd.index("--python") + 1] == str(tmp_path / "venv" / "bin" / "python")
    assert cmd[cmd.index("--device") + 1] == device
    assert cmd[cmd.index("--install") + 1] == "torch,max" and cmd[cmd.index("--require") + 1] == "torch,max"
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "out")
    assert cmd[cmd.index("--install-seconds") + 1] == "2400" and cmd[cmd.index("--seconds") + 1] == "900"


def test_smoke_command_refuses_an_unknown_vendor(tmp_path):
    with pytest.raises(ValueError):
        AG.smoke_command("intel", tmp_path, tmp_path)


def test_venv_commands_make_a_fresh_venv_with_the_given_wheels(tmp_path):
    make, install = AG.venv_commands("/base/python3", tmp_path / "v", ["/w/core.whl", "/w/amd.whl"])
    assert make == ["/base/python3", "-m", "venv", "--clear", str(tmp_path / "v")]
    assert install[0] == str(tmp_path / "v" / "bin" / "python") and install[1:4] == ["-m", "pip", "install"]
    assert "/w/core.whl" in install and "/w/amd.whl" in install and "numpy" in install


# ------------------------------------------------------------------ judge
def test_judge_passes_only_passed_with_every_phase(tmp_path):
    assert AG.judge(receipt("PASSED"))[1]
    status, ok, why = AG.judge(receipt("NO_FRAMEWORK", phases={}))
    assert status == "NO_FRAMEWORK" and not ok and "requires every adapter phase" in why
    status, ok, why = AG.judge(receipt("FAILED", problems=["torch adapter check failed (exit 1)"]))
    assert not ok and "torch adapter check failed" in why
    partial = dict(PHASES_OK)
    partial.pop("max-training")
    status, ok, why = AG.judge(receipt("PASSED", phases=partial))
    assert not ok and "max-training" in why
    assert AG.judge(tmp_path / "missing.json")[:2] == ("NO_RECEIPT", False)


def test_the_gate_phases_are_whole_model_only():
    # Andrew, 2026-10-24: no per-operation phase (the per-op surfaces were removed)
    assert AG.PHASES == ("torch-compile", "torch-training", "max-training")
    old = dict(PHASES_OK, torch={"exit": 0}, max={"exit": 0})
    old.pop("torch-compile")
    status, ok, why = AG.judge(receipt("PASSED", phases=old))
    assert not ok and "torch-compile" in why


def test_enabled_follows_the_release_switch():
    assert AG.enabled({}) and AG.enabled({"MOJOLEARN_RELEASE_ADAPTERS": "1"})
    assert not AG.enabled({"MOJOLEARN_RELEASE_ADAPTERS": "0"})


@pytest.mark.parametrize("status,ok", [("PASSED", True), ("NO_FRAMEWORK", False), ("FAILED", False), (None, False)])
def test_run_records_the_judged_receipt(tmp_path, status, ok):
    calls = []
    rec = AG.run("nvidia", "/base/python3", tmp_path / "a", ["/w/core.whl"], launch=fake_launch(status, calls))
    assert rec["ok"] is ok and rec["status"] == (status or "NO_RECEIPT")
    assert [c[1:3] for c in calls[:2]] == [["-m", "venv"], ["-m", "pip"]] and "--device" in calls[2]


def test_run_keeps_no_venv_in_the_evidence_folder(tmp_path):
    calls = []
    rec = AG.run("nvidia", "/base/python3", tmp_path / "a", ["/w/core.whl"], launch=fake_launch("PASSED", calls))
    venv = Path(rec["python"]).parents[1]
    assert tmp_path not in venv.parents and not venv.parent.exists()  # a temp dir, removed on return
    assert not (tmp_path / "a" / "venv").exists()


def test_run_stops_when_the_venv_does_not_install(tmp_path):
    calls = []
    rec = AG.run("amd", "/base/python3", tmp_path / "a", ["x.whl"], launch=fake_launch("PASSED", calls, venv_rc=1))
    assert rec["status"] == "VENV_FAILED" and not rec["ok"] and len(calls) == 1


# ------------------------------------------------------------------ the box gate (gate-nvidia, gate-amd)
def state(**adapters):
    doc = {"smoke_finished": 1, "smoke_rc": 0, "counts": {"PASSED": 3}, "smoke_cpu_fallback": True,
           "autograd_finished": 1, "autograd_rc": 0, "measure_finished": 1, "measure_rc": 0,
           "measure": {"pending": 0, "failed": [], "expected": 3, "planned_whole_plan": 3, "board": "/b.json",
                       "pairs": {"matched": 3, "refused": [], "unchecked": []}}}
    doc.update(adapters)
    return doc


def test_gate_verdict_requires_a_passed_adapter_smoke():
    passed = {"status": "PASSED", "ok": True, "why": "PASSED"}
    assert G.gate_verdict(state(adapters_enabled=True, adapters_finished=1, adapters=passed)) == ("PASS", [])
    for status in ("NO_FRAMEWORK", "FAILED", "VENV_FAILED"):
        verdict, why = G.gate_verdict(state(adapters_enabled=True, adapters_finished=1,
                                            adapters={"status": status, "ok": False, "why": status + ": x"}))
        assert verdict == "FAIL" and any(w.startswith("adapters: " + status) for w in why)
    assert G.gate_verdict(state())[0] == "INCOMPLETE"  # a gate that never ran the stage


def test_gate_verdict_skips_the_disabled_adapter_stage():
    assert G.gate_verdict(state(adapters_enabled=False, adapters_finished=1)) == ("PASS", [])


def gate(tmp_path, vendor="amd", **kw):
    g = G.Gate(vendor, "0.8.56", tmp_path / "gate", profile="performance-campaign-v1", **kw)
    g.state.update(python="/gate/venv/bin/python")
    return g


@pytest.mark.parametrize("status,ok", [("PASSED", True), ("NO_FRAMEWORK", False), ("FAILED", False)])
def test_gate_stage_runs_the_smoke_on_its_own_venv(tmp_path, monkeypatch, status, ok):
    calls, real = [], AG.run
    monkeypatch.setattr(AG, "run", lambda *a, **k: real(*a, launch=fake_launch(status, calls), **k))
    g = gate(tmp_path, "nvidia")
    g.adapters()
    rec = json.loads(g.state_path.read_text())
    assert rec["adapters_enabled"] is True and rec["adapters"]["ok"] is ok and rec["adapters_finished"]
    # a scratch venv under a temp dir, never inside the gate's evidence folder, and never the measurement venv
    venv = calls[0][-1]
    assert venv.endswith("/venv") and str(tmp_path / "gate") not in venv and "/gate/venv/bin/python" not in calls[2]
    assert calls[1][-3:] == ["mojolearn-core==0.8.56", "mojolearn-nvidia==0.8.56", "numpy"]
    assert calls[2][calls[2].index("--device") + 1] == "cuda:0"
    assert (G.gate_verdict(dict(state(), **{k: rec[k] for k in ("adapters_enabled", "adapters_finished",
                                                                 "adapters")}))[0] == "PASS") is ok


def test_candidate_gate_installs_the_candidate_wheels(tmp_path):
    g = gate(tmp_path)
    g.candidate_wheels = ["/in/core.whl", "/in/amd.whl"]
    assert g.adapter_installs() == ["/in/core.whl", "/in/amd.whl"]


def test_gate_stage_disabled_runs_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(AG, "run", lambda *a, **k: pytest.fail("the disabled stage launched the smoke"))
    g = gate(tmp_path, adapters=False)
    g.adapters()
    assert g.state["adapters_enabled"] is False and g.state["adapters"]["status"] == "DISABLED"


def test_gate_job_carries_no_adapters_and_brings_the_receipt_home():
    argv = RG.gate_argv("amd", "0.8.56", "/r")
    assert "--no-adapters" not in argv
    argv = RG.gate_argv("amd", "0.8.56", "/r", adapters=False)
    assert argv[-1] == "--no-adapters"
    assert "adapter-smoke/adapter-smoke.json" in RG.GATE_FETCH


class FakeRelease(release.Release):
    """Release with its derived properties as plain attributes (no state directory, no git)."""
    rel = commit = version = dry = None


def release_runner(tmp_path):
    r = object.__new__(FakeRelease)
    r.dry, r.version, r.rel, r.commit = True, "0.8.56", tmp_path / "rel", "c" * 40
    r.base = tmp_path / "base"
    r.args = SimpleNamespace(smoke_branch="main", gate_autograd_root=None)
    r.gate_ok = lambda vendor, measure=True: (False, "no gate report")
    r.gate_out = lambda vendor: tmp_path / "out"
    r.source_checkout = lambda create=True: tmp_path / "source"
    r.candidate_gate_wheels = lambda vendor: [tmp_path / "core.whl", tmp_path / (vendor + ".whl")]
    r.say = lambda line: None
    r.calls = []
    r.must = lambda cmd, **kw: r.calls.append([str(c) for c in cmd])
    return r


@pytest.mark.parametrize("switch,flag", [("1", False), ("0", True)])
def test_release_run_gate_passes_no_adapters_only_when_switched_off(tmp_path, monkeypatch, switch, flag):
    monkeypatch.setenv("MOJOLEARN_RELEASE_ADAPTERS", switch)
    r = release_runner(tmp_path)
    r.run_gate("amd")
    assert ("--no-adapters" in r.calls[0]) is flag


def test_release_gate_ok_refuses_a_gate_without_its_adapter_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_RELEASE_ADAPTERS", "1")
    r = release_runner(tmp_path)
    del r.gate_ok
    r.candidate_gate_identity = lambda vendor: {"id": 1}
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "gate-smoke.json").write_text(json.dumps(dict(
        state(adapters_enabled=False, adapters_finished=1), version="0.8.56", vendor="amd", verdict="PASS",
        input_mode="candidate", candidate={"id": 1})))
    ok, why = r.gate_ok("amd")
    assert not ok and "without its adapter smoke" in why


# ------------------------------------------------------------------ the Apple gate (gate-apple)
def apple_runner(tmp_path, status):
    r = release_runner(tmp_path)
    r.dry = False
    wheel = tmp_path / "mojolearn-0.8.56-py3-none-macosx_11_0_arm64.whl"
    wheel.write_bytes(b"wheel")
    r.macos_wheel = lambda: wheel
    r.smoke_python = lambda: "/usr/bin/python3.12"
    launch = fake_launch(status, r.calls)

    def must(cmd, **kw):
        if launch([str(c) for c in cmd], kw.get("log")) != 0:
            raise release.StepFailed("exited 1")
    r.must = must
    # runtime-apple (the installed-runtime qualifier gate-apple also runs) has its own tests in
    # tools/test_runtime_qualifier_units.py; these cover the adapter smoke alone
    r.apple_runtime_gate = lambda: None
    return r


def test_gate_apple_passes_on_passed(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_RELEASE_ADAPTERS", "1")
    r = apple_runner(tmp_path, "PASSED")
    assert r.step_gate_apple().startswith("PASSED")
    smoke = r.calls[-1]
    assert "metal" in smoke and smoke[smoke.index("--device") + 1] == "mps"
    assert r.calls[1][-2].endswith(".whl")  # the built macOS wheel, installed in the fresh venv
    assert r.step_gate_apple().endswith("(receipt exists)") and len(r.calls) == 3


@pytest.mark.parametrize("status", ["NO_FRAMEWORK", "FAILED"])
def test_gate_apple_fails_on_anything_but_passed(tmp_path, monkeypatch, status):
    # --strict: the adapter smoke stops gate-apple (the behavior before 2026-11-08)
    monkeypatch.setenv("MOJOLEARN_RELEASE_ADAPTERS", "1")
    r = apple_runner(tmp_path, status)
    r.args.strict = True
    with pytest.raises(release.StepFailed):
        r.step_gate_apple()
    assert not r.apple_adapters_ok()[0] and not r.apple_adapters_soft_failed()


@pytest.mark.parametrize("status", ["NO_FRAMEWORK", "FAILED"])
def test_gate_apple_records_a_failed_adapter_smoke_as_soft(tmp_path, monkeypatch, status):
    # fail soft (tools/release_verdicts.py apple/adapters): recorded, notified, gate-apple continues
    import release_verdicts as RV
    monkeypatch.setenv("MOJOLEARN_RELEASE_ADAPTERS", "1")
    r = apple_runner(tmp_path, status)
    lines = []
    r.say = lines.append
    assert "SOFT FAIL apple/adapters" in r.step_gate_apple()
    assert [f["name"] for f in RV.load(r.rel)] == ["apple/adapters"] and r.apple_adapters_soft_failed()
    assert any(l.startswith("SOFT FAIL apple apple/adapters: ") for l in lines)
    assert (r.base / RV.rel_name(r.version) / "SOFT-FAILURES.md").is_file()   # the notice stays under the fake state


def test_gate_apple_is_skipped_when_no_adapters_ship(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_RELEASE_ADAPTERS", "0")
    r = apple_runner(tmp_path, "PASSED")
    assert r.step_gate_apple().startswith("adapters skipped") and r.calls == []


def test_publish_macos_needs_a_passed_apple_adapter_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_RELEASE_ADAPTERS", "1")
    r = apple_runner(tmp_path, "NO_FRAMEWORK")
    with pytest.raises(release.StepFailed, match="run gate-apple"):
        r.step_publish_macos()   # no adapter smoke at all, recorded neither as passed nor as a SOFT failure


def test_cpu_column_records_the_adapter_refusal_as_its_expected_outcome():
    why = ("FAILED: the installed wheel carries no usable framework adapters: ImportError: mojolearn: the framework "
           "adapters run on a GPU, and this install selected no GPU set (mojolearn: MOJOLEARN_VENDOR=cpu; ...)")
    rec = AG.cpu_refusal({"vendor": "cpu", "status": "FAILED", "ok": False, "why": why})
    assert (rec["status"], rec["ok"], rec["refused_why"]) == ("refused-by-design", True, why)
    other = {"vendor": "cpu", "status": "FAILED", "ok": False, "why": "FAILED: torch-compile exit 1"}
    assert AG.cpu_refusal(other) is other
    gpu = {"vendor": "nvidia", "status": "FAILED", "ok": False, "why": why}
    assert AG.cpu_refusal(gpu) is gpu
    venv = {"vendor": "cpu", "status": "VENV_FAILED", "ok": False, "why": "adapter venv setup failed (exit 1)"}
    assert AG.cpu_refusal(venv)["ok"] is False


def _torch_venv(tmp_path):
    venv = tmp_path / "venv"
    lib = venv / "lib" / "python3.12" / "site-packages" / "torch" / "lib"
    lib.mkdir(parents=True)
    return venv, lib


def test_framework_env_puts_torch_lib_first_on_amd_only_and_never_changes_its_base(tmp_path):
    venv, lib = _torch_venv(tmp_path)
    base = {"LD_LIBRARY_PATH": "/job/lib", "KEEP": "1"}
    amd = AG.framework_env("amd", venv, base)
    parts = amd["LD_LIBRARY_PATH"].split(":")
    assert parts[0] == str(lib) and parts[-1].startswith("/opt/rocm") and len(parts) == 2
    assert amd["KEEP"] == "1" and base == {"LD_LIBRARY_PATH": "/job/lib", "KEEP": "1"}
    assert AG.framework_env("nvidia", venv, base) == base


def test_env_overrides_round_trip_and_name_only_what_changed():
    base = {"A": "1", "B": "2", "SECRET": "s"}
    env = dict(base, A="9", C="3")
    env.pop("B")
    over = AG.env_overrides(env, base)
    assert over == {"A": "9", "C": "3", "B": None} and "SECRET" not in over
    assert AG.apply_overrides(over, base) == env


def test_run_records_the_smoke_python_and_its_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/job/lib")
    calls = []
    env = dict(__import__("os").environ, LD_LIBRARY_PATH="/torch/lib:/opt/rocm/lib")
    rec = AG.run("amd", "/base/python3", tmp_path / "a", ["/w/core.whl"], env=env, launch=fake_launch("PASSED", calls))
    assert rec["python"] == AG.venv_python(tmp_path / "a" / "venv")
    assert rec["env_overrides"] == {"LD_LIBRARY_PATH": "/torch/lib:/opt/rocm/lib"}
    assert AG.run("amd", "/base/python3", tmp_path / "b", ["/w/core.whl"],
                  launch=fake_launch("PASSED", []))["env_overrides"] == {}
