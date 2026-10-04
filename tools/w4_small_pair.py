#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Manager-only quality pair / quality-gated timing for lane apple-fast-w4-small
on verified A/B arms (tools/kfeat_pair.py's pattern).

 quality SOURCE QUALITY_TAG CASE
 timing  SOURCE QUALITY_TAG TIMING_TAG CASE ALGO DATASET

CASE -> binding, define (A = main = none, B = -D <define>):
  var          x_sequence   MOJOLEARN_SEQ_FAST_VAR_FUSED       algo var (taxi-hourly, synthetic)
  knn-imputer  x_neighbors  MOJOLEARN_XN_FAST_NAN_FIT_LEAN     algo knn-imputer (taxi)
  maxabs       x_prep       MOJOLEARN_PREP3_MAXABS_POOL        algo maxabs-scaler (istella, taxi)
  resample     resample     MOJOLEARN_RESAMPLE_FAST_ROW_GATHER algo resample (taxi, istella)
SOURCE is the exact full SHA. Arms: ~/mq/verified-arms/SOURCE/BINDING/
{A.so,B.so,manifest.json}. The quality action installs each arm, runs
`tools/w4_small_quality.py dump CASE`, checks reach (B reports the
candidate compiled in, A does not), runs `compare` (byte-identical), writes
PASS.json and restores the installed binding. The timing action refuses
without that receipt, then runs tools/afc_ab_def.sh through
~/mq/verified_arms.py. No builds, SSH, queue edits, or opponent runs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

FIXTURE = "w4-small-v1"
CASES = {
    "var": ("x_sequence", "MOJOLEARN_SEQ_FAST_VAR_FUSED", ("var",), ("taxi-hourly", "synthetic")),
    "knn-imputer": ("x_neighbors", "MOJOLEARN_XN_FAST_NAN_FIT_LEAN", ("knn-imputer",), ("taxi", "istella")),
    "maxabs": ("x_prep", "MOJOLEARN_PREP3_MAXABS_POOL", ("maxabs-scaler",), ("istella", "taxi")),
    "resample": ("resample", "MOJOLEARN_RESAMPLE_FAST_ROW_GATHER", ("resample",), ("taxi", "istella")),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_logged(command, path):
    with path.open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print("\n".join(path.read_text(errors="replace").splitlines()[-12:]))
        raise RuntimeError("Quality command failed: " + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("quality", "timing"))
    parser.add_argument("source")
    parser.add_argument("quality_tag")
    parser.add_argument("rest", nargs="+")
    args = parser.parse_args()
    if args.action == "quality":
        assert len(args.rest) == 1, "quality SOURCE QUALITY_TAG CASE"
        timing_tag, case = None, args.rest[0]
    else:
        assert len(args.rest) == 4, "timing SOURCE QUALITY_TAG TIMING_TAG CASE ALGO DATASET"
        timing_tag, case, algo, dataset = args.rest
    assert case in CASES, case
    binding, define, algos, datasets = CASES[case]
    if args.action == "timing":
        assert algo in algos and dataset in datasets, (algo, dataset)
    assert re.fullmatch("[0-9a-f]{40}", args.source)
    for tag in (args.quality_tag, timing_tag):
        assert tag is None or re.fullmatch("[A-Za-z0-9_.-]+", tag)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.source
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "sequence/", "x_neighbors/", "x_prep/", "resample/",
                    "core/", "bindings/", "python/", "tools/w4_small_quality.py", "tools/w4_small_pair.py"], check=True)
    home = Path.home()
    arms = home / "mq/verified-arms" / args.source / binding
    manifest = json.loads((arms / "manifest.json").read_text())
    assert manifest["source_sha"] == args.source and manifest["binding"] == binding
    assert manifest["numeric_mode"] == "fast"
    assert manifest["defines_A"] == ""
    assert manifest["defines_B"].split() == ["-D", define], manifest["defines_B"]
    hashes = {arm: digest(arms / (arm + ".so")) for arm in ("A", "B")}
    assert hashes == manifest["hashes"]
    out = home / "mq/out" / (args.quality_tag + "-quality")
    receipt_path = out / "PASS.json"
    receipt_want = dict(source_sha=args.source, hashes=hashes, status="PASS", fixture=FIXTURE,
                        case=case, binding=binding, define=define)
    if args.action == "timing":
        assert json.loads(receipt_path.read_text()) == receipt_want
        os.environ["AFC_FAMILY"] = "algos"
        os.execv(sys.executable, [sys.executable, str(home / "mq/verified_arms.py"),
                 args.source, binding, define, timing_tag,
                 "bash", "tools/afc_ab_def.sh", timing_tag, binding, algo, dataset, "1", "1", "", "-D " + define])
    out.mkdir(parents=True, exist_ok=False)  # never silently reuse partial captures
    os.environ.update(MOJOLEARN_NUMERIC_MODE="fast", MOJOLEARN_VENDOR="apple",
                      MOJOLEARN_BENCH_INSTALLED="0", PYTHONPATH=str(root / "python"),
                      OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    installed = root / ("python/mojolearn/_mojolearn_%s.so" % binding)
    shutil.copy2(installed, out / "original.so")
    try:
        for arm in ("A", "B"):
            temporary = installed.with_suffix(".so.next")
            shutil.copy2(arms / (arm + ".so"), temporary)
            os.replace(temporary, installed)
            log = out / (arm + ".log")
            run_logged([sys.executable, "tools/w4_small_quality.py", "dump", case, str(out / (arm + ".npz"))], log)
            records = [json.loads(line.split(" ", 1)[1]) for line in log.read_text().splitlines()
                       if line.startswith("W4SMALL-CAPTURE ")]
            assert len(records) == 1
            rec = records[0]
            assert rec["case"] == case and rec["binding_sha256"] == hashes[arm], rec
            assert rec["reach"] == (1 if arm == "B" else 0), (arm, rec)
            print("W4SMALL-PAIR arm=%s capture=PASS reach=%s" % (arm, rec["reach"]), flush=True)
        cmp_log = out / "compare.log"
        with cmp_log.open("x") as stream:
            rc = subprocess.run([sys.executable, "tools/w4_small_quality.py", "compare", case,
                                 str(out / "A.npz"), str(out / "B.npz")], stdout=stream, stderr=subprocess.STDOUT).returncode
        text = cmp_log.read_text()
        print("\n".join(line for line in text.splitlines() if line.startswith("W4SMALL-")))
        if rc != 0 or "status=PASS" not in text:
            print("W4SMALL-PAIR status=FAIL case=" + case + " source=" + args.source + " log=" + str(cmp_log))
            sys.exit(1)
        with receipt_path.open("x") as stream:
            json.dump(receipt_want, stream, sort_keys=True)
        print("W4SMALL-PAIR status=PASS case=" + case + " source=" + args.source + " receipt=" + str(receipt_path))
    finally:
        temporary = installed.with_suffix(".so.restore")
        shutil.copy2(out / "original.so", temporary)
        os.replace(temporary, installed)


if __name__ == "__main__":
    main()
