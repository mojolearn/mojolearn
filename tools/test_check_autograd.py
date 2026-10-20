"""The correctness controller's all-selection must include higher-order checks."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import check_autograd as check


@pytest.mark.parametrize("selection", ["all", "higher"])
def test_higher_order_runs_and_missing_case_fails(monkeypatch, tmp_path, selection):
    commands = []
    omit_case = False

    def run(command, *, cwd, stdout, stderr, timeout):
        commands.append(command)
        if "build" in command:
            assert command[command.index("-j") + 1] == "1"
            return SimpleNamespace(returncode=0)
        name = Path(command[-1]).name
        if name.endswith("-gradients"):
            stdout.write("PASS finite-difference cases=35\nPASS lifecycle refusals=10 repeated_graphs=30\n")
        elif name.endswith("-mlp-autograd"):
            for line in check.MLP_DIGESTS.read_text().splitlines():
                if line and not line.startswith(("#", "case")):
                    case, epochs, digest = line.split()
                    stdout.write(f"CASE {case} epochs {epochs}\nDIGEST {case} {digest}\n")
            stdout.write("PASS mlp_cases 48\n")
        elif name.endswith("-infer"):
            stdout.write("PASS infer models=7 refusals=4\n")
        elif name.endswith("-higher"):
            expected = check.higher_expect()
            stdout.write(f"PASS higher-order cases={expected['cases']}\nPASS higher refusals={expected['refusals']}\n"
                         "PASS higher lifecycle repeated_graphs=30 release=1\nFIRST_ORDER_MATCH 1\n")
            for i in range(expected["cases"] - int(omit_case)):
                stdout.write(f"HIGHER_DIGEST case{i} 123\n")
        elif "-rankn-" in name:
            stdout.write(f"RANKN {name.split('-')[-1]} test digest 123\nPASS\n")
        elif name.endswith("-import"):
            stdout.write("IMPORT test case digest 123\nPASS import cases 1\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(check.shutil, "which", lambda name: "/fake/mojo")
    monkeypatch.setattr(check.subprocess, "run", run)
    monkeypatch.setattr(check.subprocess, "check_output", lambda *args, **kw: "test-source\n")
    monkeypatch.setattr("sys.argv", ["check_autograd", "--backend", "cpu", "--only", selection,
                                    "--output", str(tmp_path)])
    assert check.main() == 0
    assert any("higher_cpu.mojo" in " ".join(c) for c in commands)
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "passed"
    # Even though the fake binary exits zero and prints PASS, an absent digest must fail.
    omit_case = True
    assert check.main() == 1
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["status"] == "failed" and "HIGHER_DIGEST lines" in report["error"]
