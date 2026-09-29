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

Running all twelve lanes in one process is the hardest case: the training,
transformer, mamba, embedding and byte LM bindings each open their shared
context once and every later entry, session and stage reuses it. A second
run of a stage that moved reads MOVED, which fails the harness; a hang hits
the timeout. The lane names come from tools/identity_lanes/neural.core. The default uses base and negative fixtures
(the latter retains the known Samba trigger), with a 180-second arm bound.
MOJOLEARN_NEURAL_REPEAT_EXHAUSTIVE=1 explicitly selects all nine fixtures.
Both paths retain every property, including batchgrad and ragged checks.
The subprocesses use the same runtime imported by pytest, including an
installed wheel; source builds also work after their bindings are staged:
`python -m pytest python/mojolearn/tests/test_neural_repeat.py`."""
import json
import re
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import mojolearn

ROOT = Path(__file__).resolve().parents[3]
LANES = ROOT / "tools" / "identity_lanes" / "neural.core"
EXHAUSTIVE = os.environ.get("MOJOLEARN_NEURAL_REPEAT_EXHAUSTIVE") == "1"
FIXTURES = "" if EXHAUSTIVE else "base,negative"
TIMEOUT_S = 3600 if EXHAUSTIVE else 180


def _lanes():
    lanes = [line.strip() for line in LANES.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    assert len(lanes) == 12 and len(set(lanes)) == len(lanes)
    return lanes


def _environment(kind):
    # Use the already imported runtime, including an installed wheel. Never
    # route its subprocess back into an unstaged source package via arm_env.
    pkg = Path(mojolearn.__file__).resolve().parent
    # This explicitly selected continuity test spans the neural family in
    # one process. The Apple guard requires an intentional diagnostic flag;
    # fixture and wall-clock bounds below still prevent an implicit full sweep.
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical",
               MOJOLEARN_APPLE_FULL_DIAGNOSTIC="1")
    for key in list(env):
        if key.startswith("MOJOLEARN_") and ("BINARY" in key or "HOST_DIR" in key
                or key.endswith("_ALLOW_SABOTAGE") or key in
                ("MOJOLEARN_VENDOR", "MOJOLEARN_PAR_DEVICES")):
            env.pop(key)
    env.pop("PYTHONPATH", None)
    witness = pkg / "identity_columns" / "COMMIT"
    if witness.is_file():
        commit = witness.read_text().strip()
        assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
        env["MOJOLEARN_COMMIT"] = commit
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MOJOLEARN_CPU_THREADS"):
        env[key] = "1"
    if kind == "cpu":
        host = pkg / "host"
        env.update(MOJOLEARN_VENDOR="cpu", MOJOLEARN_HOST_DIR=str(host),
                   MOJOLEARN_FOREST_HOST_BINARY=str(host / "_mojolearn_forest_host.so"),
                   MOJOLEARN_BYTE_LM_HOST_BINARY=str(host / "_mojolearn_byte_lm_host.so"))
    return env


def _command():
    # -I prevents cwd/PYTHONPATH from substituting another runtime. Explicitly
    # import the one pytest is holding, then execute its shipped harness.
    pkg = Path(mojolearn.__file__).resolve().parent
    script = pkg / "_identity_break.py"
    if not script.is_file():
        script = ROOT / "tools" / "identity_break.py"
    return [sys.executable, "-I", "-u", "-c",
            "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
            "runpy.run_path(sys.argv.pop(1),run_name='__main__')",
            str(pkg.parent), str(script)]


def _run(cmd, kind, out, timeout):
    log = out.with_suffix(".log")
    proc = subprocess.Popen(cmd, cwd=out.parent, env=_environment(kind),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    try:
        text, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the entire owned process group, including descendants that
        # outlive the leader. Preserve partial output for diagnosis.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            text, _ = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            text = ""
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        tail, _ = proc.communicate()
        log.write_text(text + tail)
        pytest.fail(f"{kind} continuity exceeded {timeout}s; partial log: {log}")
    log.write_text(text)
    assert proc.returncode == 0, f"{kind} failed ({proc.returncode}); {log}:\n{text[-4000:]}"


def _column(kind, backend, out):
    cmd = _command() + ["--lanes", ",".join(_lanes()), "--repeats", "2",
                        "--fail-on-refused", "--json", str(out)]
    if FIXTURES:
        cmd += ["--fixtures", FIXTURES]
    cmd += ["--require-cpu", "--require-backend", "cpu"] if kind == "cpu" else ["--require-backend", backend]
    _run(cmd, kind, out, TIMEOUT_S)
    assert out.is_file()


def _assert_records(gpu_path, cpu_path):
    """Require complete repeated measurements, not merely a permissive diff."""
    docs = [json.loads(path.read_text()) for path in (gpu_path, cpu_path)]
    fixtures = set(FIXTURES.split(",")) if FIXTURES else set(docs[0]["fixtures"])
    assert len(fixtures) == (9 if EXHAUSTIVE else 2)
    expected = {f"{lane}/{fixture}" for lane in _lanes() for fixture in fixtures}
    parts = ("train", "infer", "model", "batch", "stepfull", "batchgrad", "batchscale", "ragged")
    # These six lanes have no sampler/trainer pair in the harness. Absence
    # is structural here; every other neural lane must carry two rlpair hashes.
    no_pair = {"mlp", "optim-sgd", "optim-adam-clip", "optim-maximize",
               "training-primitives", "embedding"}
    values = []
    declared_na = {}
    intentional_notes = {}
    for doc in docs:
        assert doc["complete"] is True and doc["mode"] == "identical"
        assert doc["repeats"] == 2 and not doc.get("partial_column")
        assert doc.get("parts_omitted") == []
        assert set(doc["cells"]) == expected
        assert set(doc["fixtures"]) == fixtures
        assert set(doc["parts_collected"]) == {"batch", "batchgrad", "batchscale", "ragged", "rlpair", "stepfull"}
        column = {}
        for key, cell in doc["cells"].items():
            lane = key.split("/")[0]
            if lane in no_pair:
                assert "rlpair" not in cell and "rlpair_verdict" not in cell
                required = parts
            else:
                required = parts + ("rlpair",)
            for part in required:
                field, verdict = ("hashes", "verdict") if part == "train" else (part, part + "_verdict")
                hashes = cell[field]
                assert isinstance(hashes, list) and len(hashes) == 2, (key, part, "missing repeat")
                assert hashes[0] == hashes[1], (key, part, "MOVED")
                value = hashes[0]
                if cell[verdict] == "N/A":
                    assert part != "train" and isinstance(value, str) and value.startswith("n/a:") and len(value) > 4
                    assert not value.startswith(("n/a:skipped", "n/a:UNDECLARED"))
                    declared_na[key + "/" + part] = value
                else:
                    assert cell[verdict] == "STABLE" and isinstance(value, str) and re.fullmatch(r"[0-9a-f]{16}", value), (key, part, cell[verdict])
                column[key + "/" + part] = value
            if cell.get("batchgrad_notes"):
                intentional_notes.setdefault(key, []).append(cell["batchgrad_notes"])
        values.append(column)
    for field in ("fixtures", "heldout", "lane_revisions", "batch_revisions"):
        assert docs[0][field] == docs[1][field], field
    assert docs[0]["resume_signature"]["options"]["require_backend"] in ("metal", "cuda", "hip")
    assert docs[1]["resume_signature"]["options"]["require_backend"] == "cpu"
    assert values[0] == values[1], "CPU/GPU part mismatch"
    return dict(cells_per_arm=len(expected), repeats_per_part=2,
                numeric_parts=sum(not v.startswith("n/a:") for v in values[0].values()),
                structural_na=declared_na,
                absent_rlpair={f"{lane}/{f}": "no sampler/trainer pair declared" for lane in sorted(no_pair) for f in sorted(fixtures)},
                intentional_batchgrad_notes=intentional_notes,
                fixtures=sorted(fixtures), exhaustive=EXHAUSTIVE)


def test_every_neural_lane_twice_in_one_process_gpu_and_cpu(tmp_path):
    from mojolearn import _backend
    backend = _backend.vendor()
    if backend not in ("metal", "cuda", "hip"):
        pytest.skip("continuity comparison requires a GPU install")
    print(f"neural continuity: fixtures={FIXTURES or 'all9'}, repeats=2, "
          f"all properties including batchgrad/ragged; runtime={mojolearn.__file__}", flush=True)
    gpu, cpu = tmp_path / "gpu.json", tmp_path / "cpu.json"
    _column("gpu", backend, gpu)
    _column("cpu", backend, cpu)
    (tmp_path / "strict-coverage.json").write_text(json.dumps(_assert_records(gpu, cpu), indent=2))
    _run(_command() + ["--diff", str(gpu), str(cpu), "--lanes", ",".join(_lanes()),
                       "--require-columns", "2"], "gpu", tmp_path / "compare.json", 30)
