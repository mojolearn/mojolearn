#!/usr/bin/env python3
"""Run one depthwise A/B from manifest-verified, already compiled M2 arms."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

SOURCE = "2519f4867d76b534cd630c6f00d7152f41fccaf2"
DEFINE = "MOJOLEARN_GBDT_DW_BRIDGE_SCAN"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("dataset", choices=("taxi", "istella"))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise SystemExit("invalid tag")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    home = Path.home()
    py = home / "board-0834/cache/venv/bin/python"
    if not py.is_file() or not (root / "python/mojolearn/_mojolearn.so").is_file():
        raise SystemExit("provision the FAST base binding and board venv before this command")
    out = home / "afc-def" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("run_*.log")) or (out / "aft.log").exists() or (out / "aft-claimed").exists():
        raise SystemExit("refusing to overwrite existing AFT evidence")
    # verified_arms checks source scope, numeric mode, define and both SHA256
    # hashes, then stages A.so/B.so here. It rejects an existing scored race.
    subprocess.run([
        str(py), str(home / "mq/verified_arms.py"), SOURCE, "gbdt",
        DEFINE, args.tag, "--stage-only",
    ], check=True)
    # A permanent claim also rejects accidental reruns where AFT has not yet
    # created a run log. Failed jobs need explicit manager inspection.
    try:
        (out / "aft-claimed").mkdir()
    except FileExistsError:
        raise SystemExit("refusing to replay a claimed AFT job")
    env = os.environ.copy()
    env.update(AFT_OUT=str(out), AFT_SKIP_BUILD="1", AFT_PY=str(py),
               MOJOLEARN_NUMERIC_MODE="fast", MOJOLEARN_VENDOR="metal",
               AB_MULTI_RUN="0", MOJOLEARN_COMPILE_JOBS="1")
    env.pop("AFT_ROWS", None)
    env.pop("MOJOLEARN_BENCH_INSTALLED", None)
    command = ["bash", "tools/aft_ab.sh", "gbdt", "gbdt-depthwise",
               args.dataset, "1", "", "-D " + DEFINE]
    with (out / "aft.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit("AFT failed; inspect " + str(out / "aft.log"))
    # aft_ab itself can exit zero after a failed benchmark. Require one
    # successful timing and both actual held-out prediction metrics per arm.
    metrics = {}
    for arm in ("A", "B"):
        text = (out / ("run_" + arm + "_1.log")).read_text()
        timings = re.findall(r"^FSPEED lane=gbdt-depthwise arm=ours\S* .*? ms=([\d.]+) hash=(\S+)", text, re.M)
        values = re.findall(r"^FSPEED-ACC lane=gbdt-depthwise arm=ours\S* metric=(auc|logloss) value=(\S+)", text, re.M)
        if len(timings) != 1 or len(values) != 2 or {k for k, _ in values} != {"auc", "logloss"}:
            raise SystemExit("missing timing or prediction quality for arm " + arm + "; inspect " + str(out))
        metrics[arm] = {k: float(v) for k, v in values}
        if not all(math.isfinite(v) for v in metrics[arm].values()):
            raise SystemExit("non-finite prediction quality in " + arm)
        print("AFT-VERIFIED arm=" + arm + " ms=" + timings[0][0]
              + " hash=" + timings[0][1] + " quality=" + json.dumps(metrics[arm]), flush=True)
    print("AFT-QUALITY-REVIEW auc_delta=" + str(metrics["B"]["auc"] - metrics["A"]["auc"])
          + " logloss_delta=" + str(metrics["B"]["logloss"] - metrics["A"]["logloss"])
          + " source=" + SOURCE + " logs=" + str(out), flush=True)
    # This confirms evidence completeness, not quality equivalence. Manager
    # reviews both metric deltas against observed noise before promotion.
    return 0


if __name__ == "__main__":
    sys.exit(main())
