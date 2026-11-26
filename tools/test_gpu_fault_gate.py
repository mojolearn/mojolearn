"""Tooling test for tools/ops/gpu_fault_gate.py and the pod job's fault gate (tools/ops/box_job_lib.sh fault_gate)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gpu_fault_gate", ROOT / "tools/ops/gpu_fault_gate.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def test_cheap_arms_are_registered_mojo_run_tasks():
    tasks = g.sabotage_tasks(ROOT / "pixi.toml")
    assert len(tasks) >= len(g.CHEAP_ARMS)
    for arm in g.CHEAP_ARMS:
        argv, args = g.build_argv(tasks[arm], "/cache/x")
        assert argv[:2] == ["mojo", "build"] and argv[-3:-1] == ["-o", "/cache/x"] and argv[-1].endswith(".mojo")
        assert "-j" not in argv and any(w.endswith("SABOTAGE=1") for w in argv)


def test_build_argv_keeps_run_args_and_drops_threads():
    argv, args = g.build_argv("mojo run -j 2 -I . -D A=1 checks/x.mojo border_types_check", "/b")
    assert argv == ["mojo", "build", "-I", ".", "-D", "A=1", "-o", "/b", "checks/x.mojo"]
    assert args == ["border_types_check"]


def test_verdict():
    assert g.verdict(1, False, "!! GATE FAILED: check_x") == "detected"
    assert g.verdict(1, False, "3 of 12 cases differ") == "detected"
    assert g.verdict(0, False, "all gates ok") == "passed_silently"   # the fault went unseen: the unit fails
    assert g.verdict(1, False, "segfault-free but no verdict line") == "crashed"
    assert g.verdict(-11, False, "GATE FAILED") == "crashed"
    assert g.verdict(None, True, "") == "timed_out"


def _fakes(tmp: Path, silent_arm: str = "") -> dict:
    """A fake `pixi` on PATH (mojo --version, mojo build -> a script binary, run <binary>) and a fake venv python."""
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    pixi = bin_dir / "pixi"
    pixi.write_text(
        "#!/bin/sh\n[ \"$1\" = run ] && shift\n"
        "if [ \"$1\" = mojo ] && [ \"$2\" = --version ]; then echo mojo 1.2; exit 0; fi\n"
        "if [ \"$1\" = mojo ] && [ \"$2\" = build ]; then\n"
        "  out=; while [ $# -gt 0 ]; do [ \"$1\" = -o ] && out=$2; shift; done\n"
        f"  case \"$out\" in *{silent_arm or 'NO-SUCH-ARM'}) printf '#!/bin/sh\\necho all ok\\nexit 0\\n' > \"$out\";;\n"
        "    *) printf '#!/bin/sh\\necho \"!! GATE FAILED: check_fault\"\\nexit 1\\n' > \"$out\";; esac\n"
        "  chmod +x \"$out\"; exit 0; fi\n"
        "exec \"$@\"\n")
    pixi.chmod(0o755)
    py = tmp / "python"
    py.write_text(
        "#!/bin/sh\ncase \"$*\" in\n"
        "  *vendor*) echo cuda;;\n"
        "  *self-test*) echo '{\"passed\": true, \"format\": \"mojolearn.verify-self-test.v2\", \"sides\": "
        "{\"cpu\": {\"ran\": true, \"clean\": {\"state\": \"IDENTICAL\"}, \"perturbed\": {\"state\": \"DIVERGENT\"}}, "
        "\"gpu\": {\"ran\": true, \"clean\": {\"state\": \"IDENTICAL\"}, \"perturbed\": {\"state\": \"DIVERGENT\"}}}, "
        "\"problems\": []}';;\nesac\n")
    py.chmod(0o755)
    tree = tmp / "tree"
    tree.mkdir()
    (tree / "pixi.toml").write_text((ROOT / "pixi.toml").read_text())
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return dict(py=str(py), tree=str(tree), env=env)


def _run(tmp: Path, f: dict) -> tuple[int, str, dict]:
    out, summary = tmp / "faults", tmp / "FAULTS-nvidia.txt"
    p = subprocess.run([sys.executable, str(ROOT / "tools/ops/gpu_fault_gate.py"), "--python", f["py"], "--out",
                        str(out), "--summary", str(summary), "--vendor", "nvidia", "--tree", f["tree"]],
                       env=f["env"], capture_output=True, text=True, timeout=120)
    return p.returncode, summary.read_text(), json.loads((out / "faults.json").read_text())


def test_every_fault_detected(tmp_path):
    rc, text, report = _run(tmp_path, _fakes(tmp_path))
    assert rc == 0, text
    first = text.splitlines()[0]
    assert first.startswith("FAULTS nvidia: self-test DETECTED on cuda; sabotage 4/4 detected") and '"' not in first
    assert {a["arm"] for a in report["arms"]} == set(g.CHEAP_ARMS)
    assert len(report["not_run"]) == len(report["registered"]) - len(g.CHEAP_ARMS)


def test_a_silent_sabotage_fails_the_unit(tmp_path):
    rc, text, report = _run(tmp_path, _fakes(tmp_path, silent_arm="check-glm-multinomial-sabotage"))
    assert rc == 1
    assert "sabotage 3/4 detected (NOT: glm-multinomial-sabotage=passed_silently)" in text.splitlines()[0]


def test_job_runs_the_gate_once_after_each_install_and_records_its_line():
    job = (ROOT / "tools/ops/paper_compare_job.sh").read_text()
    lib = (ROOT / "tools/ops/box_job_lib.sh").read_text()
    assert "\nfault_gate() {" in lib and "faults/faults.done" in lib and "PC_FAULTS" in lib
    # every flavour (full board, targeted, paper) runs it right after its candidate-manifest check
    assert job.count('fault_gate "${ARM_PY[candidate]}"') == 3
    for line in [ln for ln in job.splitlines() if 'fault_gate "${ARM_PY[candidate]}"' in ln]:
        before = job.splitlines()[job.splitlines().index(line) - 1]
        assert "ARM_MAN[candidate]" in before
    assert '"faults":"%s"' in job and "FAULTS_LINE" in job


def _fault_gate(tmp_path, **env):
    """Run box_job_lib.sh's fault_gate() alone, with `say` and `phase` stubbed and gpu_fault_gate.py not reachable."""
    import os
    import subprocess
    lib = (ROOT / "tools/ops/box_job_lib.sh").read_text()
    body = lib[lib.index("\nfault_gate() {") + 1:]
    body = body[:body.index("\n}\n") + 3]
    script = ('say() { echo "$*"; }; phase() { echo "PHASE $1"; }; phase_end() { :; }\n' + body
              + 'fault_gate /nonexistent/python\necho "LINE $FAULTS_LINE"\n')
    run_env = {k: v for k, v in os.environ.items() if k != "PC_FAULTS"}
    run_env.update(RES=str(tmp_path), V="nvidia", PY="false", **env)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=run_env, cwd=tmp_path).stdout


def test_the_gate_is_release_only_off_unless_asked(tmp_path):
    # AGENTS.md hard rule RELEASE-ONLY WORK (2026-11-26): a round builds and runs no sabotage arm
    out = _fault_gate(tmp_path)
    assert "LINE FAULTS: off (release only; PC_FAULTS=1 runs it)" in out and "PHASE faults" not in out
    on = tmp_path / "on"
    on.mkdir()
    assert "PHASE faults" in _fault_gate(on, PC_FAULTS="1")  # asked for: the gate runs (its tool is absent here)


def test_the_feature_proof_runs_the_verify_options_only_for_the_release():
    text = (ROOT / "tools/ops/feature_proof_job.sh").read_text()
    guard = text.index('if [ "${FP_RELEASE:-0}" = 1 ]; then')
    end = text.index("PROOF verify-options SKIP")
    for step in ("verify-self-test", "verify-cross-check-quick", "verify-check ", "verify-fit-both", "verify-recipe-"):
        assert guard < text.index(f'step {step}' if step != "verify-recipe-" else 'step "verify-recipe-') < end, step
    assert text.index("step digest-families") < guard  # the digest_ feature test stays in every round


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
