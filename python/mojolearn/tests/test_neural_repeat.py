# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Every neural-family identity lane, all of them in ONE process per column,
each stage run TWICE (`identity_break.py --repeats 2`), on the GPU bindings
and on the CPU host bindings; then the two columns diffed bit for bit.

The directive it answers (CURRENT DIRECTIVES, 2026-09-27): an entry that
builds a DeviceContext per call can hang or fail on a LATER call in the same
process, and the lane checks run each lane once per process
(`--repeats 1`), so they cannot see it. Two such failures were found in this
family (core/neural_context.mojo): `samba/negative ragged` never returned on
do-amd, and every `training-primitives batchgrad` cell (one `linear_backward`
per row) died in the Metal compiler service on the M2 Pro and M3 Ultra.

Running all eleven lanes in one process is the hardest case: the training,
transformer, mamba, embedding and byte LM bindings each open their shared
context once and every later entry, session and stage reuses it. A second
run of a stage that moved reads MOVED, which fails the harness; a hang hits
the timeout. The lanes are the `# lanes:` line of
tools/identity_lanes/neural.checks, and the binding environment is the lane
check's own (tools/algos_lane_check.py arm_env), so the CPU column answers
every lane. Runs on the lane's pod after the lane check built the bindings:
`python -m pytest python/mojolearn/tests/test_neural_repeat.py`."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

HARNESS = ROOT / "tools" / "identity_break.py"
CHECKS = ROOT / "tools" / "identity_lanes" / "neural.checks"
TIMEOUT_S = 3600


def _lanes():
    for line in CHECKS.read_text().splitlines():
        if line.startswith("# lanes:"):
            return [x.strip() for x in line.split(":", 1)[1].split(",") if x.strip()]
    raise AssertionError(f"{CHECKS} has no `# lanes:` line")


def _column(kind, backend, out):
    import algos_lane_check as alc
    cmd = [sys.executable, "-u", str(HARNESS), "--lanes", ",".join(_lanes()), "--repeats", "2",
           "--fail-on-refused", "--json", str(out)]
    cmd += ["--require-cpu", "--require-backend", "cpu"] if kind == "cpu" else ["--require-backend", backend]
    try:
        r = subprocess.run(cmd, cwd=ROOT, env=alc.arm_env(kind), capture_output=True, text=True,
                           timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        tail = (e.stdout or b"")[-3000:]
        tail = tail.decode(errors="replace") if isinstance(tail, bytes) else tail
        raise AssertionError(f"the {kind} column did not finish in {TIMEOUT_S} s (a hang on a later "
                             f"call in the process?); last output:\n{tail}")
    assert r.returncode == 0 and out.is_file(), \
        f"the {kind} column failed (exit {r.returncode}):\n{r.stdout[-3000:]}{r.stderr[-3000:]}"


def test_every_neural_lane_twice_in_one_process_gpu_and_cpu(tmp_path):
    import algos_lane_check as alc
    try:
        backend = alc.gpu_backend()
    except alc.Fail as e:
        pytest.skip(str(e))
    gpu, cpu = tmp_path / "gpu.json", tmp_path / "cpu.json"
    _column("gpu", backend, gpu)
    _column("cpu", backend, cpu)
    r = subprocess.run([sys.executable, str(HARNESS), "--diff", str(gpu), str(cpu), "--lanes",
                        ",".join(_lanes()), "--require-columns", "2"],
                       cwd=ROOT, env=alc.arm_env("gpu"), capture_output=True, text=True)
    assert r.returncode == 0, f"GPU and CPU columns differ:\n{r.stdout[-4000:]}{r.stderr[-2000:]}"
