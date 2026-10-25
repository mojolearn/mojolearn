# SPDX-License-Identifier: Apache-2.0
"""The release gate's framework-adapter smoke against the INSTALLED wheel (test tooling, never product code).

AGENTS.md "Framework adapters are the same API in another format" (Andrew, 2026-10-24): the Torch and MAX
adapters are qualified by loading and running the shipped `_adapters/` files from the installed wheel on each
vendor, and that smoke is part of the release gate. This module is the one place the gate runners take it from:
tools/gate_smoke_box.py (gate-nvidia, gate-amd, ON the held box) and tools/release.py gate-apple (the Mac).

    venv     a FRESH venv (never the gate's measurement venv: its torch is the opponents' pinned build, the
             adapters are built against packaging/adapters/pins.json's) holding the gate's mojolearn wheels
    smoke    tools/adapter_wheel_smoke.py --install torch,max --require torch,max --device <vendor's device>:
             the frameworks the shipped adapters.json names (Torch from its index: PyPI for CUDA and Metal, the
             PyTorch ROCm 7.2 index for HIP), then the three whole-model phases torch-compile (models through
             torch.compile(backend="mojolearn"), bit for bit mojolearn.import_model's output), torch-training
             and max-training (torch.ops.mojolearn.train and MAX mojolearn_train/_resume, bit for bit
             mojolearn.Sequential's fit) on that device. No per-operation phase: the per-op surfaces were removed
             on 2026-10-24 (AGENTS.md "Framework adapters are the same API in another format").
    judge    ONLY a PASSED receipt with all three phases exit 0 passes. NO_FRAMEWORK is a FAILURE here (the
             frameworks must install and every phase must run), as is FAILED or a missing receipt.

MOJOLEARN_RELEASE_ADAPTERS=0 turns it off, the same switch that drops the adapters from the legs and the pack.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAMEWORKS = "torch,max"
PHASES = ("torch-compile", "torch-training", "max-training")
#: the device string each vendor's framework build uses (torch-ROCm keeps the cuda device type)
DEVICE = {"nvidia": "cuda:0", "amd": "cuda:0", "apple": "mps"}
RECEIPT = "adapter-smoke.json"


def enabled(environ=None):
    """False when MOJOLEARN_RELEASE_ADAPTERS=0 (no adapters released: legs, pack and gate alike)."""
    return (os.environ if environ is None else environ).get("MOJOLEARN_RELEASE_ADAPTERS", "1") != "0"


def venv_python(venv):
    return str(Path(venv) / "bin" / "python")


def venv_commands(base_python, venv, installs):
    """The argv list that makes a fresh venv at VENV and installs INSTALLS (wheel files or requirements)."""
    py = venv_python(venv)
    return [[str(base_python), "-m", "venv", "--clear", str(venv)],
            [py, "-m", "pip", "install", "--no-input", "--disable-pip-version-check", *map(str, installs), "numpy"]]


def smoke_command(vendor, venv, out, runner_python=None):
    """The adapter smoke's argv for VENDOR: frameworks installed from the shipped pins, both required."""
    if vendor not in DEVICE:
        raise ValueError("adapter gate: vendor is one of %s, not %r" % (", ".join(sorted(DEVICE)), vendor))
    return [str(runner_python or sys.executable), str(HERE / "adapter_wheel_smoke.py"), "--python", venv_python(venv),
            "--out", str(out), "--device", DEVICE[vendor], "--install", FRAMEWORKS, "--require", FRAMEWORKS]


def judge(receipt):
    """(status, ok, why) of an adapter-smoke.json path or document: ok only for PASSED with every phase exit 0."""
    doc = receipt
    if not isinstance(receipt, dict):
        try:
            doc = json.loads(Path(receipt).read_text())
        except (OSError, ValueError):
            return "NO_RECEIPT", False, "no adapter-smoke.json receipt"
    status = str(doc.get("status") or "NO_RECEIPT")
    if status == "NO_FRAMEWORK":
        return status, False, ("NO_FRAMEWORK: torch and MAX did not install or import, nothing ran; "
                               "the gate requires every adapter phase to run")
    if status != "PASSED":
        return status, False, "%s: %s" % (status, "; ".join(map(str, doc.get("problems") or [])) or "no problems listed")
    phases = doc.get("phases") or {}
    missing = [p for p in PHASES if (phases.get(p) or {}).get("exit") != 0]
    if missing:
        return status, False, "PASSED receipt without every phase passing: " + ", ".join(missing)
    return status, True, "PASSED (%s)" % ", ".join(PHASES)


def summary(record):
    """The one-line form of a gate's adapters record."""
    if not record.get("enabled", True):
        return "adapters=DISABLED"
    return "adapters=%s" % record.get("status")


def run(vendor, base_python, out, installs, env=None, launch=None):
    """Fresh venv, install, smoke, judge. Returns the record the gate keeps:
    {enabled, status, ok, why, rc, report, frameworks, phases}. LAUNCH(argv, log) -> rc (tests swap it)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    venv = out / "venv"

    def default(argv, log):
        with open(log, "ab") as fh:
            return subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    launch = launch or default
    rec = dict(enabled=True, vendor=vendor, device=DEVICE.get(vendor), report=str(out / RECEIPT))
    started = time.monotonic()
    for argv in venv_commands(base_python, venv, installs):
        rc = launch(argv, str(out / "adapter-venv.log"))
        if rc != 0:
            rec.update(status="VENV_FAILED", ok=False, rc=rc,
                       why="adapter venv setup failed (exit %s); see %s" % (rc, out / "adapter-venv.log"))
            return rec
    setup_elapsed = time.monotonic() - started
    rec["rc"] = launch(smoke_command(vendor, venv, out), str(out / "adapter-smoke.log"))
    status, ok, why = judge(out / RECEIPT)
    doc = {}
    try:
        doc = json.loads((out / RECEIPT).read_text())
    except (OSError, ValueError):
        pass
    rec.update(status=status, ok=ok, why=why, frameworks=doc.get("frameworks"),
               phases={n: (p or {}).get("exit") for n, p in (doc.get("phases") or {}).items() if n in PHASES},
               phase_records=doc.get("phases") or {},
               installation_elapsed_seconds=setup_elapsed + sum(float(p.get("seconds", 0)) for n, p in (doc.get("phases") or {}).items() if n.startswith("install-")),
               execution_elapsed_seconds=sum(float(p.get("seconds", 0)) for n, p in (doc.get("phases") or {}).items() if n in PHASES))
    return rec
