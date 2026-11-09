"""Smoke orchestration regressions; no compiler or GPU required."""
import json
import sys
import types
from pathlib import Path

import pytest

import autograd_smoke as smoke


def test_registry_covers_public_training_and_all_tape_integrations():
    entries = smoke.ENTRIES
    assert len({e["name"] for e in entries}) == len(entries)
    sources = {e["source"] for e in entries if "source" in e}
    assert {
        "autograd/checks/import_gpu.mojo",
        "mamba/checks/mamba_tape_backward_check.mojo",
        "training/checks/samba_tape_step_check.mojo",
    } <= sources
    tests = {e["pytest"] for e in entries if "pytest" in e}
    assert {"python/mojolearn/tests/" + t for t in (
        "test_autograd_usability.py", "test_autograd_model.py::test_dropout_trains",
        "test_ops_program.py", "test_ops_rankn.py", "test_ops_optimizer.py", "test_ops_higher.py",
    )} <= tests
    assert any(e.get("probe") == "program" for e in entries)
    for entry in entries:
        if "probe" in entry:
            assert entry["probe"] in smoke.PROBES
        else:
            assert (smoke.ROOT / (entry.get("source") or entry["pytest"]).split("::")[0]).is_file()
    for vendor, target in (("metal", None), ("nvidia", "sm_86"), ("amd", "gfx942")):
        commands = smoke.build_commands(vendor, "mojo", "/build", target)
        assert len(commands) == len(sources)
        for command in commands:
            assert command[command.index("-j") + 1] == "1"
            assert "MOJOLEARN_NUMERIC_IDENTICAL=1" in command
            if target:
                assert command[command.index("--target-accelerator") + 1] == target


@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
def test_linux_check_commands_use_the_portable_host_isa(vendor):
    for command in smoke.build_commands(vendor, "mojo", "/build"):
        assert command.count("--target-cpu") == 1
        assert command[command.index("--target-cpu") + 1] == "x86-64-v3"


def test_metal_check_commands_keep_their_existing_host_target():
    assert all("--target-cpu" not in command
               for command in smoke.build_commands("metal", "mojo", "/build"))


@pytest.mark.parametrize("actual,mode", [("cpu", 1), ("hip", 1), ("cuda", 0)])
def test_actual_binding_must_match_vendor_and_mode(monkeypatch, actual, mode):
    binding = types.SimpleNamespace(x_sequence_vendor=lambda: actual, x_sequence_numeric_mode=lambda: mode)
    package = types.SimpleNamespace(__file__="/installed/mojolearn/__init__.py",
                                    _backend=types.SimpleNamespace(binding=lambda *args: binding))
    monkeypatch.setitem(sys.modules, "mojolearn", package)
    with pytest.raises(RuntimeError, match="wrong smoke binding"):
        smoke.check_binding("nvidia")


def _completed_entry(cmd, log, timeout, env=None, cwd=None, *, skipped=0, tests=3, marker=""):
    Path(cmd[cmd.index("--binding-report") + 1]).write_text('{"vendor": "cuda", "numeric_mode": 1}')
    Path(cmd[cmd.index("--junit") + 1]).write_text(
        '<testsuites><testsuite tests="%d" failures="0" errors="0" skipped="%d"/></testsuites>' % (tests, skipped))
    with Path(log).open("a") as stream:
        stream.write(marker)
    return "ok", 0, 0.01


@pytest.mark.parametrize("tests,skipped,status,rc", [
    (3, 0, "ok", 0), (3, 3, "SKIPPED", 1), (3, 1, "PARTIAL", 1), (0, 0, "failed", 1),
])
def test_pytest_coverage_and_isolation(monkeypatch, tmp_path, tests, skipped, status, rc):
    def start(cmd, log, timeout, env, cwd):
        assert cmd[1] == "-I"
        assert "PYTHONPATH" not in env and "MOJOLEARN_VENDOR_FORCE" not in env
        assert env["MOJOLEARN_VENDOR"] == "cuda" and env["MOJOLEARN_NUMERIC_MODE"] == "identical"
        return _completed_entry(cmd, log, timeout, env, cwd, tests=tests, skipped=skipped)
    monkeypatch.setattr(smoke, "start", start)
    monkeypatch.setenv("PYTHONPATH", "/wrong/source")
    monkeypatch.setenv("MOJOLEARN_VENDOR", "cpu")
    monkeypatch.setenv("MOJOLEARN_VENDOR_FORCE", "1")
    out = tmp_path / "report.json"
    assert smoke.run("nvidia", tmp_path, sys.executable, out, entries=["autograd-usability"]) == rc
    report = json.loads(out.read_text())
    assert report["rows"][0]["status"] == status
    assert report["rows"][0]["tests"]["tests"] == tests
    assert report["coverage"] == "selected entries" and report["package_scope"] == "installed"


@pytest.mark.parametrize("marker,status", [("SKIPPED no torch\n", "SKIPPED"),
                                          ("SMOKE_REFUSED unsupported node\n", "REFUSED"), ("", "ok")])
def test_probe_refusals_are_visible_and_old_log_lines_do_not_leak(monkeypatch, tmp_path, marker, status):
    logs = tmp_path / "report"
    logs.mkdir()
    (logs / "import-torch-smoke-metal.log").write_text("SKIPPED previous invocation\n")
    def start(cmd, log, timeout, env, cwd):
        assert "MOJOLEARN_VENDOR" not in env
        assert cmd[-2:] == ["--source-package", str(tmp_path)]
        return _completed_entry(cmd, log, timeout, env, cwd, marker=marker)
    monkeypatch.setattr(smoke, "start", start)
    out = tmp_path / "report.json"
    assert smoke.run("metal", tmp_path, sys.executable, out, entries=["import-torch-smoke"],
                     source_package=tmp_path) == 0
    report = json.loads(out.read_text())
    assert report["rows"][0]["status"] == status and report["package_scope"] == "source"


def test_missing_reports_and_binaries_cannot_pass(monkeypatch, tmp_path):
    logs = tmp_path / "report"
    logs.mkdir()
    (logs / "autograd-usability-metal.binding.json").write_text("{}")
    (logs / "autograd-usability-metal.xml").write_text('<testsuite tests="1"/>')
    monkeypatch.setattr(smoke, "start", lambda *args, **kw: ("ok", 0, 0.01))
    out = tmp_path / "report.json"
    assert smoke.run("metal", tmp_path, sys.executable, out,
                     entries=["autograd-usability", "autograd-lm-tape"]) == 1
    assert json.loads(out.read_text())["counts"] == {"failed": 1, "not built": 1}


def test_timeout_remains_incomplete_and_unknown_selection_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "start", lambda *args, **kw: ("STARTED", None, 5))
    out = tmp_path / "report.json"
    assert smoke.run("amd", tmp_path, sys.executable, out, entries=["autograd-usability"]) == 0
    assert json.loads(out.read_text())["counts"] == {"STARTED": 1}
    with pytest.raises(ValueError, match="unknown smoke selection"):
        smoke.run("amd", tmp_path, sys.executable, out, entries=["misspelled"])


def test_an_entry_timer_replaces_the_callers_and_is_recorded(monkeypatch, tmp_path):
    public = next(e for e in smoke.ENTRIES if e["name"] == "autograd-public")
    assert public["timeout_s"] == 180
    assert [e["name"] for e in smoke.ENTRIES if "timeout_s" in e] == ["autograd-public"]
    seen = {}
    def start(cmd, log, timeout, env, cwd):
        seen[cmd[4]] = timeout
        return _completed_entry(cmd, log, timeout, env, cwd)
    monkeypatch.setattr(smoke, "start", start)
    out = tmp_path / "report.json"
    smoke.run("nvidia", tmp_path, sys.executable, out, 30, entries=["autograd-public", "autograd-usability"])
    assert seen == {"autograd-public": 180, "autograd-usability": 30}
    rows = {r["name"]: r for r in json.loads(out.read_text())["rows"]}
    assert rows["autograd-public"]["timeout_s"] == 180 and rows["autograd-usability"]["timeout_s"] == 30


def test_the_deadline_still_bounds_an_entry_timer(monkeypatch, tmp_path):
    seen = []
    def start(cmd, log, timeout, env, cwd):
        seen.append(timeout)
        return _completed_entry(cmd, log, timeout, env, cwd)
    monkeypatch.setattr(smoke, "start", start)
    monkeypatch.setattr(smoke.time, "time", lambda: 1000.0)
    out = tmp_path / "report.json"
    smoke.run("nvidia", tmp_path, sys.executable, out, 30, entries=["autograd-public"], deadline=1052.0)
    assert seen == [50.0]


def test_completed_mlp_binary_checks_stored_digests_and_coverage(tmp_path):
    rows = [line.split() for line in (smoke.ROOT / "autograd/checks/mlp_cases_digests.tsv").read_text().splitlines()
            if line and not line.startswith(("#", "case"))]
    text = "".join("CASE %s epochs %s\nDIGEST %s %s\n" % (case, epochs, case, digest)
                   for case, epochs, digest in rows)
    log = tmp_path / "mlp.log"
    log.write_text(text)
    assert smoke.check_mlp_digests(log)["cases"] == 48
    log.write_text(text.replace(rows[0][2], "0", 1))
    with pytest.raises(ValueError, match="stored-digest mismatch"):
        smoke.check_mlp_digests(log)
    log.write_text("\n".join(text.splitlines()[:-1]))
    with pytest.raises(ValueError, match="incomplete MLP coverage"):
        smoke.check_mlp_digests(log)


def test_start_preserves_exit_status_and_timeout(tmp_path):
    assert smoke.start([sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "failed.log", 5)[:2] == ("failed", 3)
    assert smoke.start([sys.executable, "-c", "import time; time.sleep(10)"],
                       tmp_path / "timeout.log", 0.05)[:2] == ("STARTED", None)


@pytest.mark.parametrize("vendor,require,rc,marker", [
    ("cpu", None, 0, "SKIPPED no torch"), ("cpu", "1", 1, "SMOKE_FAIL real-model"), ("cuda", None, 1, "SMOKE_FAIL real-model")])
def test_real_model_without_torch_is_skipped_only_on_the_cpu_column(monkeypatch, capsys, vendor, require, rc, marker):
    package = types.ModuleType("mojolearn")
    package.vendor = lambda: vendor
    importers = types.ModuleType("mojolearn.importers")
    importers.UnsupportedModel = type("UnsupportedModel", (Exception,), {})
    package.importers = importers
    monkeypatch.setitem(sys.modules, "mojolearn", package)
    monkeypatch.setitem(sys.modules, "mojolearn.importers", importers)
    monkeypatch.setitem(sys.modules, "torch", None)          # import torch raises ImportError
    if require is None:
        monkeypatch.delenv("MOJOLEARN_REQUIRE_TORCH", raising=False)
    else:
        monkeypatch.setenv("MOJOLEARN_REQUIRE_TORCH", require)
    assert smoke.probe_real_model() == rc
    assert marker in capsys.readouterr().out


def _torch_world(monkeypatch, with_torch):
    """has_module answers True for torch/torchvision only for the interpreters in WITH_TORCH; no install runs."""
    monkeypatch.setattr(smoke, "has_module", lambda python, module: str(python) in with_torch)
    installs = []
    monkeypatch.setattr(smoke, "ensure_torchvision", lambda python, log: installs.append(str(python)) or True)
    monkeypatch.delenv("MOJOLEARN_AUTOGRAD_TORCH_PYTHON", raising=False)
    return installs


def test_the_explicit_torch_python_is_tried_before_the_identity_venv_and_the_glob(tmp_path, monkeypatch):
    adapters_py = tmp_path / "adapters" / "venv" / "bin" / "python"
    adapters_py.parent.mkdir(parents=True)
    adapters_py.write_text("")
    logs = tmp_path / "units" / "autograd" / "autograd-smoke"
    logs.mkdir(parents=True)
    assert not _torch_world(monkeypatch, {str(adapters_py)})
    assert smoke.torch_python_for("/identity/python", logs, str(adapters_py)) == str(adapters_py)
    # a torch_python without torch falls back to the identity venv (or the env var and the glob)
    _torch_world(monkeypatch, {"/identity/python"})
    assert smoke.torch_python_for("/identity/python", logs, str(adapters_py)) == "/identity/python"
    # a missing path is never chosen; with no python holding torch the probes keep their SKIPPED
    _torch_world(monkeypatch, {str(tmp_path / "gone")})
    assert smoke.torch_python_for("/identity/python", logs, str(tmp_path / "gone")) is None


def test_the_env_var_remains_a_fallback_and_no_venv_is_globbed_from_the_evidence(tmp_path, monkeypatch):
    units = tmp_path / "units"
    glob_py = units / "adapters" / "venv" / "bin" / "python"
    glob_py.parent.mkdir(parents=True)
    glob_py.write_text("")
    logs = units / "autograd" / "autograd-smoke"
    logs.mkdir(parents=True)
    _torch_world(monkeypatch, {str(glob_py.resolve())})
    assert smoke.torch_python_for("/identity/python", logs) is None
    env_py = tmp_path / "env-python"
    env_py.write_text("")
    _torch_world(monkeypatch, {str(env_py)})
    monkeypatch.setenv("MOJOLEARN_AUTOGRAD_TORCH_PYTHON", str(env_py))
    assert smoke.torch_python_for("/identity/python", logs) == str(env_py)


def test_run_cli_hands_the_torch_python_to_run(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(smoke, "run", lambda *a, **k: seen.update(k) or 0)
    assert smoke.main(["run", "--vendor", "nvidia", "--bin-dir", str(tmp_path), "--python", "/id/python",
                       "--out", str(tmp_path / "o.json"), "--torch-python", "/adapters/venv/bin/python"]) == 0
    assert seen["torch_python"] == "/adapters/venv/bin/python"
    smoke.main(["run", "--vendor", "nvidia", "--bin-dir", str(tmp_path), "--out", str(tmp_path / "o.json")])
    assert seen["torch_python"] is None


def _wheel_venv(tmp_path):
    py = tmp_path / "identity" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    libs = tmp_path / "identity" / "lib" / "python3.12" / "site-packages" / "mojolearn" / ".libs"
    libs.mkdir(parents=True)
    return py, libs


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_compiled_checks_run_with_the_wheel_runtime_first_never_only_the_inherited_path(monkeypatch, tmp_path,
                                                                                        platform):
    # 0.8.66 run t: reused executables' RUNPATH named another checkout's pixi lib; the wheel's own Mojo runtime
    # (mojolearn/.libs) leads LD_LIBRARY_PATH whatever this process inherited
    py, libs = _wheel_venv(tmp_path)
    bins = tmp_path / "bin"
    bins.mkdir()
    (bins / "autograd-gradients-amd").write_text("")
    monkeypatch.setattr(smoke.sys, "platform", platform)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/inherited/lib")
    seen = {}
    monkeypatch.setattr(smoke, "start", lambda cmd, log, timeout, env, cwd: seen.update(env=env) or ("ok", 0, 0.1))
    smoke.run("amd", bins, str(py), tmp_path / "r.json", entries=["autograd-gradients"])
    want = (str(libs) + ":/inherited/lib") if platform == "linux" else "/inherited/lib"
    assert seen["env"] is not None and seen["env"]["LD_LIBRARY_PATH"] == want


def test_the_torch_entries_get_the_adapters_env_and_pytest_in_their_own_python(monkeypatch, tmp_path):
    identity = str(tmp_path / "identity-python")
    adapters = tmp_path / "adapters" / "venv" / "bin" / "python"
    adapters.parent.mkdir(parents=True)
    adapters.write_text("")
    env_file = tmp_path / "torch-env.json"
    env_file.write_text(json.dumps(dict(python=str(adapters),
                                        env={"LD_LIBRARY_PATH": "/torch/lib:/opt/rocm/lib", "GONE": None})))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/job/lib")
    monkeypatch.setenv("GONE", "x")
    monkeypatch.setattr(smoke, "has_module", lambda python, module: str(python) == str(adapters))
    pytests = []
    monkeypatch.setattr(smoke, "ensure_pytest", lambda python, log: pytests.append(str(python)))
    seen = {}

    def start(cmd, log, timeout, env, cwd):
        seen[cmd[4]] = (cmd[0], env)
        return _completed_entry(cmd, log, timeout, env, cwd)
    monkeypatch.setattr(smoke, "start", start)
    # no --torch-python: the env file's python stands in for it
    smoke.run("amd", tmp_path, identity, tmp_path / "r.json",
              entries=["autograd-public", "autograd-usability", "import-torch-smoke"], torch_env=str(env_file))
    assert sorted(pytests) == sorted([identity, str(adapters)])
    for name in ("autograd-public", "import-torch-smoke"):
        python, env = seen[name]
        assert python == str(adapters) and env["LD_LIBRARY_PATH"] == "/torch/lib:/opt/rocm/lib" and "GONE" not in env
    python, env = seen["autograd-usability"]
    assert python == identity and env["LD_LIBRARY_PATH"] == "/job/lib" and env["GONE"] == "x"


def test_run_cli_hands_the_torch_env_to_run(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(smoke, "run", lambda *a, **k: seen.update(k) or 0)
    smoke.main(["run", "--vendor", "amd", "--bin-dir", str(tmp_path), "--out", str(tmp_path / "o.json"),
                "--torch-env", "/units/adapters/torch-env.json"])
    assert seen["torch_env"] == "/units/adapters/torch-env.json"
