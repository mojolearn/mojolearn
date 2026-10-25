"""Crash diagnostics must survive a child that cannot write its final report."""
import json
from pathlib import Path
import subprocess
import sys


TOOLS = Path(__file__).resolve().parent


def test_last_attempted_stage_survives_abnormal_exit(tmp_path):
    # Simulate an uncatchable native exit without loading MAX or starting a GPU.
    # Seed a previous success: the new attempt must never inherit it.
    (tmp_path / "max-side.json").write_text(json.dumps({"complete": True, "stage": "complete"}))
    child = subprocess.run(
        [sys.executable, "-c", """
import os
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from check_max_training_adapters import record_stage
report = {"complete": False, "checks": [], "failures": []}
record_stage(report, Path(sys.argv[2]), "create inference session")
os._exit(7)
""", str(TOOLS), str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert child.returncode == 7
    assert child.stdout == "MAX stage: create inference session\n"
    report = json.loads((tmp_path / "max-side.json").read_text())
    assert report == {
        "complete": False, "checks": [], "failures": [], "stage": "create inference session",
    }
