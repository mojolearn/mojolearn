#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Manager-only verified K3 KERNEL quality; never enables scored timings.

  arima_scalar_verified.py SOURCE TAG

SOURCE is the full compiled SHA. A tools/docs-only descendant may reuse
its arms. Validate ~/mq/verified-arms/SOURCE/arima/{A.so,B.so,manifest.json},
install A first if the non-prebuilt arima binding is absent, back up that
binding, install B, run arima_scalar_quality.py, and restore the backup.

The probe runs the existing serial kernel and candidate inside B. A's
hash is pinned but A is NOT independently executed by this kernel gate.
Optimizer, selection, full fits and holdout forecasts need a separate A/B
quality gate before any timing or promotion. No such permission is granted
by KERNEL_PASS.json; this helper has no timing action.
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


DEFINE = "MOJOLEARN_ARIMA_FAST_SCALAR_LL"
FIXTURE = "arima-scalar-k3-v1"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024*1024),b""):
            h.update(data)
    return h.hexdigest()


def install(source,target):
    temp=target.with_suffix(target.suffix+".k3-next")
    shutil.copy2(source,temp)
    os.replace(temp,target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("tag")
    args=parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}",args.source):
        raise SystemExit("Expected exact 40-character source SHA")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+",args.tag):
        raise SystemExit("Invalid quality tag")
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    subprocess.run(["git","merge-base","--is-ancestor",args.source,"HEAD"],check=True)
    scope=["*.mojo","bindings/","python/","pixi.lock","pixi.toml"]
    subprocess.run(["git","diff","--quiet",args.source,"HEAD","--",*scope],check=True)
    subprocess.run(["git","diff","--quiet","HEAD","--",*scope,
                    "tools/arima_scalar_quality.py","tools/arima_scalar_verified.py"],check=True)
    helper_source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    home=Path.home()
    arms=home/"mq/verified-arms"/args.source/"arima"
    manifest_path=arms/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    expected=dict(source_sha=args.source,binding="arima",numeric_mode="fast",
                  defines_A="",defines_B="-D "+DEFINE)
    for key,value in expected.items():
        if manifest.get(key)!=value:
            raise SystemExit("Manifest mismatch: "+key)
    hashes={arm:digest(arms/(arm+".so")) for arm in ("A","B")}
    if hashes!=manifest.get("hashes"):
        raise SystemExit("Verified A/B binary hash mismatch")
    out=home/"mq/out"/(args.tag+"-quality")
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(manifest_path,out/"manifest.json")
    installed=root/"python/mojolearn/_mojolearn_arima.so"
    installed.parent.mkdir(parents=True,exist_ok=True)
    initialized_missing=not installed.exists()
    if initialized_missing:
        install(arms/"A.so",installed)
    backup=out/"original.so"
    shutil.copy2(installed,backup)
    backup_hash=digest(backup)
    report_path=out/"kernel-quality.json"
    log_path=out/"kernel-quality.log"
    env=os.environ.copy()
    env.update(MOJOLEARN_NUMERIC_MODE="fast",MOJOLEARN_VENDOR="apple",
               MOJOLEARN_BENCH_INSTALLED="0",PYTHONPATH=str(root/"python"),
               OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1")
    try:
        install(arms/"B.so",installed)
        if digest(installed)!=hashes["B"]:
            raise RuntimeError("Installed candidate hash mismatch")
        with log_path.open("x") as log:
            completed=subprocess.run([sys.executable,"tools/arima_scalar_quality.py",
                "--output",str(report_path)],stdout=log,stderr=subprocess.STDOUT,env=env)
        print("\n".join(log_path.read_text(errors="replace").splitlines()[-12:]))
        if completed.returncode:
            print("ARIMA-K3-VERIFIED status=HOLD_OR_ERROR log="+str(log_path),flush=True)
            return completed.returncode
        report=json.loads(report_path.read_text())
        if not (report.get("status")=="PASS" and report.get("fixture")==FIXTURE
                and report.get("source")==helper_source and report.get("binding_sha256")==hashes["B"]
                and report.get("scored_timings")==0 and report.get("promotion_authorized") is False):
            raise RuntimeError("Kernel receipt source/hash/status mismatch")
        receipt=dict(status="KERNEL_PASS_ONLY",compiled_source=args.source,helper_source=helper_source,
            hashes=hashes,fixture=FIXTURE,binding="arima",defines_A="",defines_B="-D "+DEFINE,
            baseline="existing serial RD1 kernel inside B; independent A not executed",
            report_sha256=digest(report_path),quality_script_sha256=digest(root/"tools/arima_scalar_quality.py"),
            initialized_missing_binding_with_A=initialized_missing,
            full_fit_forecast_quality_owed=True,timing_authorized=False,promotion_authorized=False)
        receipt_path=out/"KERNEL_PASS.json"
        with receipt_path.open("x") as stream: json.dump(receipt,stream,indent=2)
        print("ARIMA-K3-VERIFIED status=KERNEL_PASS_ONLY receipt="+str(receipt_path),flush=True)
        return 0
    finally:
        install(backup,installed)
        if digest(installed)!=backup_hash:
            raise RuntimeError("Original arima binding restore hash mismatch")


if __name__=="__main__":
    raise SystemExit(main())
