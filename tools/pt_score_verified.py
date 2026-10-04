#!/usr/bin/env python3
"""Manifest-verified PT quality pair, then strictly gated one-run timings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SOURCE = "bc112b1726f03cf058f71f781575f8659534b7ba"
BINDING = "x_prep"
DEFINE = "MOJOLEARN_PT_SCORE"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("quality", "timing"))
    parser.add_argument("quality_tag")
    parser.add_argument("timing_tag", nargs="?")
    parser.add_argument("dataset", nargs="?", choices=("taxi", "istella"))
    args = parser.parse_args()
    for tag in (args.quality_tag, args.timing_tag):
        if tag is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
            raise SystemExit("invalid tag")
    if args.mode == "timing" and (not args.timing_tag or not args.dataset):
        parser.error("timing requires quality_tag timing_tag dataset")
    if args.mode == "quality" and args.timing_tag:
        parser.error("quality takes only quality_tag")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    home = Path.home()
    py = home / "board-0834/cache/venv/bin/python"
    validator = home / "mq/verified_arms.py"
    if not py.is_file() or not (root / "python/mojolearn/_mojolearn.so").is_file():
        raise SystemExit("provision FAST base binding and board venv first")
    env = os.environ.copy()
    env.update(MOJOLEARN_VENDOR="metal", MOJOLEARN_NUMERIC_MODE="fast",
               MOJOLEARN_BENCH_INSTALLED="0", PYTHONPATH=str(root / "python"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", AB_MULTI_RUN="0")
    out = home / "mq/out" / (args.quality_tag + "-quality")
    marker = out / "PASS.json"
    manifest_path = home / "mq/verified-arms" / SOURCE / BINDING / "manifest.json"
    signature = {
        "source": SOURCE, "binding": BINDING, "define": DEFINE, "rows": 100000,
        "manifest_sha256": sha(manifest_path),
        "checker_sha256": sha(root / "tools/pt_score_quality.py"),
        "fixture_helper_sha256": sha(root / "tools/batchv_quality.py"),
    }
    if args.mode == "timing":
        if not marker.is_file():
            raise SystemExit("SKIP: matching PT quality PASS missing; no scored timing")
        evidence = json.loads(marker.read_text())
        if evidence["signature"] != signature:
            raise SystemExit("SKIP: PT quality source/manifest/checker mismatch")
        for name, digest in evidence["artifacts"].items():
            if sha(out / name) != digest:
                raise SystemExit("SKIP: PT quality artifact changed: " + name)
        cmd = [str(py), str(validator), SOURCE, BINDING, DEFINE, args.timing_tag,
               "bash", "tools/afc_ab_def.sh", args.timing_tag, BINDING,
               "power-transformer", args.dataset, "1", "1", "", "-D " + DEFINE]
        os.execve(str(py), cmd, env)

    # Refuse existing artifacts, even on an interrupted/failed quality job.
    # A retry requires a new explicit tag, so it cannot consume a stale PASS.
    out.mkdir(parents=True, exist_ok=False)
    subprocess.run([str(py), str(validator), SOURCE, BINDING, DEFINE,
                    args.quality_tag, "--stage-only"], check=True, env=env)
    staged = home / "afc-def" / args.quality_tag
    dest = root / "python/mojolearn/_mojolearn_x_prep.so"
    for arm in ("A", "B"):
        shutil.copy2(staged / (arm + ".so"), dest.with_suffix(".so.next"))
        os.replace(dest.with_suffix(".so.next"), dest)
        cmd = [str(py), "tools/pt_score_quality.py", "dump", str(out / (arm + ".npz")),
               "--rows", "100000"]
        if arm == "A":
            cmd.append("--reference")
        with (out / (arm + ".log")).open("w") as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit("PT quality dump failed; inspect " + str(out / (arm + ".log")))
    with (out / "compare.log").open("w") as log:
        result = subprocess.run([str(py), "tools/pt_score_quality.py", "compare",
                                 str(out / "A.npz"), str(out / "B.npz")],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
    report = (out / "compare.log").read_text()
    for line in report.splitlines():
        if line.startswith("PT-SCORE-Q"):
            print(line, flush=True)
    if result.returncode or "PT-SCORE-Q-SUMMARY PASS" not in report.splitlines():
        raise SystemExit("PT quality failed; timings stay blocked; logs=" + str(out))
    evidence = {"signature": signature, "artifacts": {
        name: sha(out / name) for name in ("A.npz", "B.npz", "compare.log")}}
    tmp = marker.with_suffix(".json.next")
    tmp.write_text(json.dumps(evidence, indent=2) + "\n")
    os.replace(tmp, marker)
    print("PT-VERIFIED-QUALITY PASS source=" + SOURCE + " logs=" + str(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
