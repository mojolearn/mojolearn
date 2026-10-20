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
        "training/checks/lm_tape_backward_check.mojo",
        "training/checks/mlp_tape_step_check.mojo",
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
    (3, 0, "ok", 0), (3, 3, "SKIPPED", 0), (3, 1, "PARTIAL", 0), (0, 0, "failed", 1),
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
