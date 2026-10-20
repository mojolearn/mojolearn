# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Release gate: the FULL measurement-board smoke on one box (test tooling).

    python3 tools/gate_smoke_box.py <nvidia|amd> <version> <out-root> [setup|smoke|autograd|all]

setup: the bench venv from the published wheels (mojolearn==<version> and the
vendor plugin, verified against PyPI) plus every pinned opponent, exactly as
tools/bench_board.setup_python builds it for a measurement run.
smoke: tools/gpu_gap_campaign.py --smoke over the vendor's WHOLE plan (no
--manifest, no --smoke-suspects: every line, not a subset), --smoke-verify-data.
autograd (2026-10-19, lane ag-bench): tools/autograd_smoke.py's entries on this vendor, 5 s "does it
start" each: the gradient suite and the MLP identity check from the check binaries the orchestrator built
into <out-root>/autograd-bin (`tools/autograd_smoke.py build-commands`), and the tiny fit test, the public
examples and the Sequential and ops.Session probes (2026-10-19, lane ag-board-twins: the board's
sequential-fit and ops-session-step lanes in miniature) through the installed wheel. A missing binary reads
`not built`, never a pass. The importer probes (import-model-smoke; since 2026-10-20, lane ag-torch-twins,
import-torch-smoke, a torch.export import) run there too; import-torch-smoke reads `SKIPPED` when the
wheel's environment has no torch (set MOJOLEARN_REQUIRE_TORCH=1 in the box environment to fail instead).
The shared registry also includes captured training/replay, dropout, validation, optimizers, rank-N,
higher-order public checks and the Mamba/LM/small-MLP/Samba tape binaries. Python entries verify their
actual compiled binding; incomplete pytest coverage and importer refusals are reported explicitly.
Nothing here times, measures, rents or deletes anything.
"""

import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_board as B  # noqa: E402
import bench_board_evidence as E  # noqa: E402

vendor, version, root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
step = sys.argv[4] if len(sys.argv) > 4 else "all"
root.mkdir(parents=True, exist_ok=True)
out, cache = root / "setup", str(root / "cache")
(out / "logs").mkdir(parents=True, exist_ok=True)
state_path = root / "gate-smoke.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {}
state.update(vendor=vendor, version=version, source_commit=B.repo_commit())


def mark(**fields):
    state.update(fields)
    E.write(state_path, state)


def amd_env():
    if vendor == "amd":
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            glob.glob(cache + "/venv/lib/python*/site-packages/*/lib64") + ["/opt/rocm-7.2.3/lib", "/opt/rocm-6.4.0/lib"])


if step in ("setup", "all"):
    mark(phase="setup", status="running", setup_started=time.time())
    args = B.build_parser().parse_args(["--vendor", vendor, "--out", str(out), "--cache", cache,
                                        "--families", "neural,algos,classical,classical2,trees",
                                        "--mojolearn-version", version])
    python, wheel = B.setup_python(args, vendor, str(out), str(out / "logs/install.log"))
    manifest = getattr(args, "wheel_manifest", None) or str(out / "wheel-manifest.json")
    with (out / "logs/neural-pins.log").open("w") as f:
        subprocess.run([python, "-m", "pip", "install", "--disable-pip-version-check"] + list(B.NEURAL_PINS),
                       stdout=f, stderr=f)
    amd_env()
    refusal = B.gpu_set_refusal(python, vendor)
    if refusal:
        mark(status="failed", error="gpu set: " + refusal)
        raise SystemExit(refusal)
    _, refusal = B.PYPI.verify(manifest, python)
    if refusal:
        mark(status="failed", error="pypi verify: " + refusal)
        raise SystemExit(refusal)
    with (out / "logs/pip-freeze.log").open("w") as f:
        subprocess.run([python, "-m", "pip", "freeze", "--all"], stdout=f, stderr=f)
    mark(phase="setup", status="ready", python=python, wheel=wheel, wheel_manifest=manifest,
         setup_finished=time.time())

if step in ("smoke", "all"):
    amd_env()
    python, manifest = state["python"], state["wheel_manifest"]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    smoke_out = root / ("smoke-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    mark(phase="smoke", status="running", smoke_started=time.time())
    cmd = [python, str(HERE / "gpu_gap_campaign.py"), "--vendor", vendor, "--out", str(smoke_out),
           "--cache", cache, "--python-env", python, "--wheel-manifest", manifest,
           "--smoke", "--smoke-verify-data"]
    with (root / "smoke.log").open("wb") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(HERE.parent)).returncode
    report = smoke_out / "smoke" / "smoke.json"
    doc = json.loads(report.read_text()) if report.exists() else {}
    rows = doc.get("recipes") or doc.get("rows") or []
    if isinstance(rows, dict):
        rows = list(rows.values())
    counts = Counter(str(r.get("status")) for r in rows if isinstance(r, dict))
    mark(phase="smoke", status="done", smoke_rc=rc, smoke_report=str(report), counts=dict(counts),
         smoke_finished=time.time())
    print("GATE_SMOKE vendor=%s rc=%s %s" % (vendor, rc, " ".join("%s=%d" % kv for kv in sorted(counts.items()))))

if step in ("autograd", "all"):
    amd_env()
    import autograd_smoke as AS  # noqa: E402
    report = root / "autograd-smoke.json"
    mark(phase="autograd", status="running", autograd_started=time.time())
    arc = AS.run(vendor, str(root / "autograd-bin"), state["python"], str(report))
    doc = json.loads(report.read_text()) if report.exists() else {}
    mark(phase="autograd", status="done", autograd_rc=arc, autograd_report=str(report),
         autograd_counts=doc.get("counts"), autograd_finished=time.time())
    print("GATE_SMOKE_AUTOGRAD vendor=%s rc=%s %s" % (vendor, arc, " ".join(
        "%s=%d" % kv for kv in sorted((doc.get("counts") or {}).items()))))
