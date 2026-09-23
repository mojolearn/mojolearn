"""tools/lm_file_evidence.py files a run's small evidence and reduces every
chain to a summary that keeps the digests; a file over the cap is skipped
and listed."""
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "lm_file_evidence.py"
PY = sys.executable


def _hex(x):
    return struct.pack(">f", x).hex()


def _chain_line(step, state, grad, losses, seconds=20.0, hsec=6.5):
    return json.dumps(dict(step=step, route="A", segment="1", label="nvidia-A-1", lr_f32_hex="3a1d482c",
                           losses_f32_hex=[_hex(v) for v in losses], seconds=seconds, hash_seconds=hsec,
                           state_sha256=state, gradient_sha256=grad, hash_scheme="sliced-sha256-8.v2",
                           prev="0" * 64, schema="mojolearn.lm-segment.chain.v1"))


@pytest.fixture
def run(tmp_path):
    """A driver out directory with one landed one-box segment and a live one."""
    out = tmp_path / "out"
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"steps": 20}))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "run": "runs/test", "recipe": str(recipe), "recipe_key": "runs/test/recipe.json", "wheel": "0.8.15",
        "routes": {"A": [{"segment": "1", "vendor": "nvidia", "steps": 10},
                         {"segment": "2", "vendor": "live", "steps": 10, "first": "nvidia", "shards": [44, 20]}]}}))
    # A/1: a one-box leg, the runner's files beside the fetched remote tree
    leg = out / "legs" / "A-1" / "leg-NVIDIA_H100"
    box = leg / "remote" / "lm-segment-A-1"
    (box / "segment").mkdir(parents=True)
    (leg / "commit.txt").write_text("d" * 40 + "\n")
    (leg / "leg.txt").write_text("commit=local\n")
    (box / "leg.txt").write_text("vendor=nvidia\n")
    (box / "gpu.txt").write_text("NVIDIA H100 80GB HBM3, GPU-abc, 550.90, 81559 MiB\n")
    (box / "status.txt").write_text("ready\n")
    (box / "big.log").write_text("x" * 5000)
    (box / "segment" / "chain.jsonl").write_text("\n".join([
        _chain_line(1, "a" * 64, "b" * 64, [10.0] * 64, 40.0, 7.0),
        _chain_line(2, "c" * 64, "d" * 64, [9.0] * 32 + [11.0] * 32, 42.0, 8.0)]) + "\n")
    (box / "segment" / "segment.json").write_text(json.dumps(dict(
        label="nvidia-A-1", first_step=0, last_completed=10, verdict="PASS", utc_start="2026-09-23T00:00:00Z",
        checkpoints=[{"file": "ckpt_00000008.blm", "sha256": "1" * 64}, {"file": "ckpt_00000010.blm", "sha256": "2" * 64}])))
    (box / "segment" / "manifest.tsv").write_text("chain.jsonl\t10\t" + "3" * 64 + "\n")
    # A/2: a live leg, a coordinator and a worker, plus a stale worker from an earlier attempt
    live = out / "legs" / "A-2-live"
    (live / "live.log").parent.mkdir(parents=True)
    (live / "live.log").write_text("orchestrator\n")
    for name, role, label, end in (("nvidia-H100", "coordinator", "nvidia-A-2", "2026-09-23T02:00:00Z"),
                                   ("amd-1", "worker", "amd-A-2", "2026-09-23T02:00:00Z"),
                                   ("amd", "worker", "amd-A-2", "2026-09-22T23:00:00Z")):
        b = live / name / "remote" / "lm-segment-A-2"
        (b / "segment").mkdir(parents=True)
        (b / "gpu.txt").write_text("NVIDIA H100 80GB HBM3, x\n" if role == "coordinator" else
                                   "=== ROCm ===\nGPU[0]\t\t: Card Series: \t\tAMD Instinct Mi325X VF\n")
        (b / "segment" / "chain.jsonl").write_text(_chain_line(11, "e" * 64, "f" * 64, [8.0] * 64, 100.0, 5.0) + "\n")
        (b / "segment" / "segment.json").write_text(json.dumps(dict(
            label=label, first_step=10, last_completed=20, verdict="PASS", utc_start="2026-09-23T01:00:00Z",
            utc_end=end, live={"role": role}, checkpoints=[{"file": "ckpt_00000020.blm", "sha256": "4" * 64}] if role == "coordinator" else [])))
    (out / "ledger.json").write_text(json.dumps({"landed": {
        "A/1": {"verdict": "PASS", "results": str(leg)},
        "A/2": {"verdict": "PASS", "results": str(live)}}}))
    (out / "driver.log").write_text("driver\n")
    return dict(out=out, spec=spec, box=box, tmp=tmp_path)


def _file(run, dest, *extra):
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"))
    r = subprocess.run([PY, str(TOOL), "--spec", str(run["spec"]), "--out", str(run["out"]), "--dest", str(dest),
                        "--cap", "4000", *extra], capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_files_small_files_and_summarizes_the_chain(run):
    dest = run["tmp"] / "filed"
    stdout = _file(run, dest)
    assert (dest / "spec.json").exists() and (dest / "recipe.json").exists()
    assert (dest / "ledger.json").exists() and (dest / "driver.log").exists()
    # the box's own leg.txt wins over the runner's; commit.txt comes from the runner's directory
    assert (dest / "A-1" / "leg.txt").read_text() == "vendor=nvidia\n"
    assert (dest / "A-1" / "commit.txt").read_text().startswith("d" * 40)
    assert (dest / "A-1" / "segment" / "segment.json").exists()
    assert (dest / "A-1" / "segment" / "manifest.tsv").exists()
    assert not (dest / "A-1" / "segment" / "chain.jsonl").exists()
    rows = (dest / "A-1" / "segment" / "chain.summary.tsv").read_text().splitlines()
    assert rows[0].split("\t")[:5] == ["step", "lr_f32_hex", "loss_mean", "seconds", "hash_seconds"]
    assert len(rows) == 3
    r2 = rows[2].split("\t")
    assert r2[0] == "2" and r2[2] == "10.000000" and r2[3] == "42.000" and r2[4] == "8.000"
    assert r2[5] == "c" * 64 and r2[6] == "d" * 64
    losses = [_hex(9.0)] * 32 + [_hex(11.0)] * 32
    assert r2[7] == hashlib.sha256(",".join(losses).encode()).hexdigest()[:16]
    src = run["box"] / "segment" / "chain.jsonl"
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    sha = (dest / "A-1" / "segment" / "chain.sha256").read_text().splitlines()
    assert sha[0] == "%s  chain.jsonl" % digest
    assert sha[1] == "lines 2" and sha[2] == "bytes %d" % src.stat().st_size
    # the table row for the README
    assert "| A/1 | NVIDIA H100 80GB HBM3 | 0 to 10 | 41.0 | 7.5 | none (the seed) | 8, 10 |" in stdout


def test_live_segment_files_both_boxes_and_not_the_stale_worker(run):
    dest = run["tmp"] / "filed"
    stdout = _file(run, dest)
    assert (dest / "A-2-nvidia" / "segment" / "chain.summary.tsv").exists()
    assert (dest / "A-2-amd" / "segment" / "chain.summary.tsv").exists()
    assert (dest / "A-2-live.log").read_text() == "orchestrator\n"
    assert "| A/2 (nvidia) | NVIDIA H100 80GB HBM3 | 10 to 20 | 100.0 | 5.0 | none | 20 |" in stdout
    assert "| A/2 (amd) | AMD Instinct Mi325X VF | 10 to 20 | 100.0 | 5.0 | worker | the coordinator's |" in stdout
    # the stale worker (ended before the coordinator started) is one box, not two
    assert stdout.count("| A/2 (amd) |") == 1


def test_a_file_over_the_cap_is_skipped_and_listed(run):
    dest = run["tmp"] / "filed"
    (run["box"] / "status.txt").write_text("y" * 4500)
    stdout = _file(run, dest)
    assert not (dest / "A-1" / "status.txt").exists()
    assert "A-1/status.txt\t4500 bytes" in (dest / "SKIPPED.txt").read_text()
    assert "skipped 1 file(s)" in stdout


def test_only_passed_segments_and_the_segments_filter(run):
    ledger = json.loads((run["out"] / "ledger.json").read_text())
    ledger["landed"]["A/2"]["verdict"] = "FAIL"
    (run["out"] / "ledger.json").write_text(json.dumps(ledger))
    dest = run["tmp"] / "filed"
    stdout = _file(run, dest)
    assert "filed A/1 into" in stdout and not (dest / "A-2-nvidia").exists()
    dest2 = run["tmp"] / "filed2"
    stdout = _file(run, dest2, "--segments", "A/2")
    assert "filed nothing" in stdout


def test_a_perturbed_digest_changes_the_summary(run):
    dest = run["tmp"] / "filed"
    _file(run, dest)
    before = (dest / "A-1" / "segment" / "chain.summary.tsv").read_text()
    src = run["box"] / "segment" / "chain.jsonl"
    src.write_text(src.read_text().replace("c" * 64, "c" * 63 + "0"))
    dest2 = run["tmp"] / "filed2"
    _file(run, dest2)
    after = (dest2 / "A-1" / "segment" / "chain.summary.tsv").read_text()
    assert before != after and ("c" * 63 + "0") in after
