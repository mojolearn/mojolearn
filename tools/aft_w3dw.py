#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Quality-gated depthwise A/B for MOJOLEARN_GBDT_DW_FLAT_GRID from verified arms.

  quality SOURCE QUALITY_TAG
  timing  SOURCE QUALITY_TAG TIMING_TAG taxi|istella

Manager-only. No builds, SSH, queue edits or opponent runs. SOURCE is the
compiled full 40-char SHA. Arms come from ~/mq/verified-arms/SOURCE/gbdt/
(manifest + A.so + B.so, built with bindings/build_gbdt.sh, which reads
MOJOLEARN_EXTRA_DEFINES: arm A "", arm B "-D MOJOLEARN_GBDT_DW_FLAT_GRID").
~/mq/verified_arms.py checks the manifest, both SHA-256 hashes and that no
.mojo/bindings/python file changed since SOURCE, then stages A.so/B.so into
~/afc-def/<tag>/. Each job then runs tools/aft_ab.sh once per arm
(AFT_SKIP_BUILD=1, one run per arm).

Quality job: taxi, the first 1,000,000 training rows (AFT_ROWS), the board's
500-tree depth-8 Depthwise config, scored on the fixed 500,000 held-out rows.
Timing job: full taxi or istella, the board row itself, which must pass the
same gate on its own held-out quality.

Gate, fixed before any result (the change is bit-identical by construction,
so the only expected difference is FAST's run-to-run spread):
  * PASS-EXACT when the held-out prediction hashes of A and B are equal;
  * otherwise PASS when B's held-out AUC >= A's - 4e-4 AND B's logloss <=
    A's + 1e-4. The bounds are the largest spread observed between main and
    bit-identical arms on depthwise taxi (AUC .632211 to .632554, logloss
    .527920 to .528002; EXPERIMENTS.md DW_BRIDGE_SCAN, LEDGER DW2/MODE_SKIP
    rows .6322 to .6325). The metrics come from tools/speed_gbdt_arm.py's
    scorer: model.predict_proba(X_test) against y_test.
  * FAIL otherwise. The timing job refuses to start without the quality
    job's PASS.json for the same SOURCE and arm hashes.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

DEFINE = "MOJOLEARN_GBDT_DW_FLAT_GRID"
LANE = "gbdt-depthwise"
QUALITY_ROWS = "1000000"
AUC_TOL = 4e-4
LOGLOSS_TOL = 1e-4


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_pair(root, home, py, source, tag, dataset, rows):
    out = home / "afc-def" / tag
    if (out / "aft.log").exists() or (out / "aft-claimed").exists() or any(out.glob("run_*.log")):
        raise SystemExit("refusing to replay: evidence exists in " + str(out))
    subprocess.run([str(py), str(home / "mq/verified_arms.py"), source, "gbdt",
                    DEFINE, tag, "--stage-only"], check=True)
    (out / "aft-claimed").mkdir()
    hashes = {arm: sha256(out / (arm + ".so")) for arm in ("A", "B")}
    env = os.environ.copy()
    env.update(AFT_OUT=str(out), AFT_SKIP_BUILD="1", AFT_PY=str(py),
               MOJOLEARN_NUMERIC_MODE="fast", MOJOLEARN_VENDOR="metal",
               AB_MULTI_RUN="0", MOJOLEARN_COMPILE_JOBS="1")
    env.pop("MOJOLEARN_BENCH_INSTALLED", None)
    env.pop("AFT_ROWS", None)
    if rows:
        env["AFT_ROWS"] = rows
    with (out / "aft.log").open("x") as log:
        rc = subprocess.run(["bash", "tools/aft_ab.sh", "gbdt", LANE, dataset, "1",
                             "", "-D " + DEFINE], env=env, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise SystemExit("aft_ab.sh failed; inspect " + str(out / "aft.log"))
    res = {}
    for arm in ("A", "B"):
        text = (out / ("run_" + arm + "_1.log")).read_text(errors="replace")
        t = re.findall(r"^FSPEED lane=" + LANE + r" arm=ours\S* .*? ms=([\d.]+) hash=(\S+)", text, re.M)
        q = dict(re.findall(r"^FSPEED-ACC lane=" + LANE + r" arm=ours\S* metric=(auc|logloss) value=(\S+)",
                            text, re.M))
        if len(t) != 1 or set(q) != {"auc", "logloss"}:
            raise SystemExit("missing timing or held-out quality for arm " + arm + " in " + str(out))
        res[arm] = dict(ms=float(t[0][0]), hash=t[0][1], auc=float(q["auc"]), logloss=float(q["logloss"]))
        if not all(math.isfinite(res[arm][k]) for k in ("ms", "auc", "logloss")):
            raise SystemExit("non-finite result for arm " + arm)
        print("W3DW arm=%s tag=%s ds=%s rows=%s ms=%.3f hash=%s auc=%.6f logloss=%.6f"
              % (arm, tag, dataset, rows or "full", res[arm]["ms"], res[arm]["hash"],
                 res[arm]["auc"], res[arm]["logloss"]), flush=True)
    a, b = res["A"], res["B"]
    d_auc, d_ll = b["auc"] - a["auc"], b["logloss"] - a["logloss"]
    if a["hash"] == b["hash"] and a["hash"] != "-":
        status = "PASS-EXACT"
    elif d_auc >= -AUC_TOL and d_ll <= LOGLOSS_TOL:
        status = "PASS"
    else:
        status = "FAIL"
    print("W3DW-GATE tag=%s status=%s auc_delta=%.6f (tol -%g) logloss_delta=%.6f (tol +%g) "
          "ms_A=%.1f ms_B=%.1f speed_delta=%.2f%% source=%s"
          % (tag, status, d_auc, AUC_TOL, d_ll, LOGLOSS_TOL, a["ms"], b["ms"],
             100.0 * (b["ms"] - a["ms"]) / a["ms"], source), flush=True)
    return status, hashes, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("quality", "timing"))
    ap.add_argument("source")
    ap.add_argument("quality_tag")
    ap.add_argument("timing_tag", nargs="?")
    ap.add_argument("dataset", nargs="?", choices=("taxi", "istella"))
    args = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source):
        raise SystemExit("SOURCE must be the full 40-char SHA")
    for tag in (args.quality_tag, args.timing_tag):
        if tag is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
            raise SystemExit("invalid tag")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    home = Path.home()
    py = home / "board-0834/cache/venv/bin/python"
    if not py.is_file() or not (root / "python/mojolearn/_mojolearn.so").is_file():
        raise SystemExit("provision the FAST base binding and the board venv first")
    receipt = home / "afc-def" / args.quality_tag / "PASS.json"
    if args.action == "quality":
        if args.timing_tag or args.dataset:
            raise SystemExit("quality takes SOURCE QUALITY_TAG only")
        status, hashes, res = run_pair(root, home, py, args.source, args.quality_tag, "taxi", QUALITY_ROWS)
        if status.startswith("PASS"):
            with receipt.open("x") as f:
                json.dump(dict(source_sha=args.source, define=DEFINE, hashes=hashes, status=status,
                               fixture="taxi-rows1000000-depthwise500", results=res), f, sort_keys=True)
            print("W3DW-QUALITY status=" + status + " receipt=" + str(receipt), flush=True)
            return 0
        print("W3DW-QUALITY status=FAIL (no receipt)", flush=True)
        return 1
    if not (args.timing_tag and args.dataset):
        raise SystemExit("timing takes SOURCE QUALITY_TAG TIMING_TAG DATASET")
    rec = json.loads(receipt.read_text())
    arms = home / "mq/verified-arms" / args.source / "gbdt"
    hashes = {arm: sha256(arms / (arm + ".so")) for arm in ("A", "B")}
    if (rec.get("source_sha") != args.source or rec.get("define") != DEFINE
            or not str(rec.get("status", "")).startswith("PASS") or rec.get("hashes") != hashes):
        raise SystemExit("quality receipt does not match this SOURCE and these arms")
    status, _, _ = run_pair(root, home, py, args.source, args.timing_tag, args.dataset, None)
    return 0 if status.startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
