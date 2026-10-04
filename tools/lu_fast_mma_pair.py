#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Manager-only LU_FAST_MMA quality pair / quality-gated timing on verified arms
(the tools/target_scratch_pair.py pattern, binding x_decomp).

  quality SOURCE QUALITY_TAG
  timing  SOURCE QUALITY_TAG TIMING_TAG lu-factor|lu-solve

SOURCE is the exact full SHA; ~/mq/verified-arms/SOURCE/x_decomp holds A.so
(main, no define), B.so (-D MOJOLEARN_LU_FAST_MMA) and manifest.json. The
quality job runs tools/lu_fast_mma_quality.py dump on each arm, then compare,
and writes PASS.json only on LU-MMA-AB status=PASS. The timing job refuses
without that receipt. No builds, SSH, queue edits or opponent runs.
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

DEFINE = "MOJOLEARN_LU_FAST_MMA"
FIXTURE = "lu-fast-mma-v1"
BINDING = "x_decomp"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_logged(command, path):
    with path.open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print("\n".join(path.read_text(errors="replace").splitlines()[-12:]))
        raise RuntimeError("Quality command failed: " + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("quality", "timing"))
    parser.add_argument("source")
    parser.add_argument("quality_tag")
    parser.add_argument("timing_tag", nargs="?")
    parser.add_argument("lane", nargs="?", choices=("lu-factor", "lu-solve"))
    args = parser.parse_args()
    assert re.fullmatch("[0-9a-f]{40}", args.source)
    for tag in (args.quality_tag, args.timing_tag):
        assert tag is None or re.fullmatch("[A-Za-z0-9_.-]+", tag)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.source
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "x_decomp/", "bindings/", "python/",
                    "tools/lu_fast_mma_quality.py", "tools/lu_fast_mma_pair.py"], check=True)
    home = Path.home()
    arms = home / "mq/verified-arms" / args.source / BINDING
    manifest = json.loads((arms / "manifest.json").read_text())
    assert manifest["source_sha"] == args.source and manifest["binding"] == BINDING
    assert manifest["numeric_mode"] == "fast"
    assert manifest["defines_A"] == ""
    assert manifest["defines_B"] == "-D " + DEFINE
    hashes = {arm: digest(arms / (arm + ".so")) for arm in ("A", "B")}
    assert hashes == manifest["hashes"]
    out = home / "mq/out" / (args.quality_tag + "-quality")
    receipt_path = out / "PASS.json"
    receipt_want = dict(source_sha=args.source, hashes=hashes, status="PASS", fixture=FIXTURE, define=DEFINE)
    if args.action == "timing":
        assert args.timing_tag and args.lane
        assert json.loads(receipt_path.read_text()) == receipt_want
        os.execv(sys.executable, [sys.executable, str(home / "mq/verified_arms.py"),
                 args.source, BINDING, DEFINE, args.timing_tag,
                 "bash", "tools/afc_ab_def.sh", args.timing_tag, BINDING, args.lane,
                 "synthetic", "1", "1", "", "-D " + DEFINE])
    assert args.timing_tag is None and args.lane is None
    out.mkdir(parents=True, exist_ok=False)  # never silently reuse partial captures
    os.environ.update(MOJOLEARN_NUMERIC_MODE="fast", MOJOLEARN_VENDOR="apple",
                      MOJOLEARN_BENCH_INSTALLED="0", PYTHONPATH=str(root / "python"),
                      OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    installed = root / "python/mojolearn" / ("_mojolearn_" + BINDING + ".so")
    shutil.copy2(installed, out / "original.so")
    try:
        for arm in ("A", "B"):
            temporary = installed.with_suffix(".so.next")
            shutil.copy2(arms / (arm + ".so"), temporary)
            os.replace(temporary, installed)
            log = out / (arm + ".log")
            run_logged([sys.executable, "tools/lu_fast_mma_quality.py", "dump", str(out / (arm + ".npz"))], log)
            records = [json.loads(line.split(" ", 1)[1]) for line in log.read_text().splitlines()
                       if line.startswith("LU-MMA-CAPTURE ")]
            assert len(records) == 1 and records[0]["fixture"] == FIXTURE, records
            assert records[0]["binding_sha256"] in ("", hashes[arm]), (records[0], hashes[arm])
            print("LU-MMA-PAIR arm=" + arm + " capture=PASS", flush=True)
        log = out / "compare.log"
        result = subprocess.run([sys.executable, "tools/lu_fast_mma_quality.py", "compare",
                                 str(out / "A.npz"), str(out / "B.npz")],
                                stdout=log.open("x"), stderr=subprocess.STDOUT)
        text = log.read_text()
        print("\n".join(line for line in text.splitlines() if line.startswith("LU-MMA-AB"))[-3000:])
        if result.returncode or "LU-MMA-AB status=PASS" not in text:
            print("LU-MMA-PAIR status=FAIL source=" + args.source)
            return 1
        with receipt_path.open("x") as stream:
            json.dump(receipt_want, stream, sort_keys=True)
        print("LU-MMA-PAIR status=PASS source=" + args.source + " receipt=" + str(receipt_path))
        return 0
    finally:
        temporary = installed.with_suffix(".so.restore")
        shutil.copy2(out / "original.so", temporary)
        os.replace(temporary, installed)


if __name__ == "__main__":
    sys.exit(main())
