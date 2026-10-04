#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Manager-only quality pair / quality-gated timing for lane apple-fast-w4-decomp,
on verified A/B arms (tools/kfeat_pair.py's pattern).

 quality SOURCE QUALITY_TAG ALGO
 timing  SOURCE QUALITY_TAG TIMING_TAG ALGO DATASET

ALGO: pca | kernel-pca | randomized-svd | lle (binding and define from
tools/w4_decomp_quality.py ALGOS; A = main = no define, B = every w4 define
of that binding: estimators -D MOJOLEARN_PCA_FAST_POOL; x_neighbors
-D MOJOLEARN_KPCA_RESIDENT; x_decomp -D MOJOLEARN_LLE_FAST_DEV_LU
-D MOJOLEARN_RSVD_FAST_DIRECT_IN, one B build for both).
SOURCE is the exact full SHA. Arms: ~/mq/verified-arms/SOURCE/BINDING/
{A.so,B.so,manifest.json}. The quality action installs each arm, runs
`tools/w4_decomp_quality.py dump`, checks the arm's reach (B compiled the
candidate in, A did not) and the binary's sha, runs `compare`, writes
PASS.json and restores the installed binding. The timing action refuses
without that receipt, then runs tools/afc_ab_def.sh through
~/mq/verified_arms.py (pca: AFC_FAMILY=classical, the CTD driver's lane
`pca`, with MOJOLEARN_PCA_STAGE_LOG=~/mq/out/TIMING_TAG-stages.log so each
fit appends its stage times, pooled=0 for arm A and 1 for arm B; the
others: AFC_FAMILY=algos). No builds, SSH, queue edits, or opponent runs.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w4_decomp_quality import ALGOS, FIXTURE  # noqa: E402

FAMILY = {"pca": "classical", "kernel-pca": "algos", "randomized-svd": "algos", "lle": "algos"}
#: the B arm of each binding: x_decomp's two candidates share ONE B build
#: (disjoint paths: randomized_svd's input vs LLE's factor), so each
#: algorithm's timing measures only its own candidate
DEFINES_B = {}
for _algo, (_mod, _def) in ALGOS.items():
    DEFINES_B.setdefault(_mod.removeprefix("_mojolearn_"), []).append(_def)
DEFINES_B = {b: sorted(d) for b, d in DEFINES_B.items()}


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
        assert len(args.rest) == 1, "quality SOURCE QUALITY_TAG ALGO"
        timing_tag, algo, dataset = None, args.rest[0], None
    else:
        assert len(args.rest) == 3, "timing SOURCE QUALITY_TAG TIMING_TAG ALGO DATASET"
        timing_tag, algo, dataset = args.rest
        assert dataset in ("taxi", "istella")
    assert algo in ALGOS, algo
    assert re.fullmatch("[0-9a-f]{40}", args.source)
    for tag in (args.quality_tag, timing_tag):
        assert tag is None or re.fullmatch("[A-Za-z0-9_.-]+", tag)
    modname, define = ALGOS[algo]
    binding = modname.removeprefix("_mojolearn_")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.source
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "decomposition/", "x_decomp/", "x_neighbors/",
                    "bindings/", "python/", "tools/w4_decomp_quality.py", "tools/w4_decomp_pair.py"], check=True)
    home = Path.home()
    arms = home / "mq/verified-arms" / args.source / binding
    manifest = json.loads((arms / "manifest.json").read_text())
    assert manifest["source_sha"] == args.source and manifest["binding"] == binding
    assert manifest["numeric_mode"] == "fast"
    assert manifest["defines_A"] == ""
    defines = DEFINES_B[binding]
    assert sorted(manifest["defines_B"].replace("-D ", "").split()) == defines, manifest["defines_B"]
    hashes = {arm: digest(arms / (arm + ".so")) for arm in ("A", "B")}
    assert hashes == manifest["hashes"]
    out = home / "mq/out" / (args.quality_tag + "-quality")
    receipt_path = out / "PASS.json"
    receipt_want = dict(source_sha=args.source, hashes=hashes, status="PASS", fixture=FIXTURE,
                        binding=binding, defines=defines, algo=algo)
    if args.action == "timing":
        assert json.loads(receipt_path.read_text()) == receipt_want
        os.environ["AFC_FAMILY"] = FAMILY[algo]
        if algo == "pca":
            os.environ["MOJOLEARN_PCA_STAGE_LOG"] = str(home / "mq/out" / (timing_tag + "-stages.log"))
        os.execv(sys.executable, [sys.executable, str(home / "mq/verified_arms.py"),
                 args.source, binding, define, timing_tag,
                 "bash", "tools/afc_ab_def.sh", timing_tag, binding, algo, dataset, "1", "1", "",
                 " ".join("-D " + d for d in defines)])
    out.mkdir(parents=True, exist_ok=False)  # never silently reuse partial captures
    os.environ.update(MOJOLEARN_NUMERIC_MODE="fast", MOJOLEARN_VENDOR="apple",
                      MOJOLEARN_BENCH_INSTALLED="0", PYTHONPATH=str(root / "python"),
                      OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    installed = root / ("python/mojolearn/%s.so" % modname)
    shutil.copy2(installed, out / "original.so")
    try:
        for arm in ("A", "B"):
            temporary = installed.with_suffix(".so.next")
            shutil.copy2(arms / (arm + ".so"), temporary)
            os.replace(temporary, installed)
            log = out / (arm + ".log")
            run_logged([sys.executable, "tools/w4_decomp_quality.py", "dump", algo, str(out / (arm + ".npz"))], log)
            text = log.read_text()
            records = [json.loads(line.split(" ", 1)[1]) for line in text.splitlines()
                       if line.startswith("W4Q-CAPTURE ")]
            assert len(records) == 1
            rec = records[0]
            assert rec["binding_sha256"] == hashes[arm] and rec["fixture"] == FIXTURE, rec
            assert rec["reach"] == (1 if arm == "B" else 0), (arm, rec)
            print("\n".join(line for line in text.splitlines() if line.startswith("W4Q-TIME ")))
            print("W4-PAIR arm=%s capture=PASS reach=%s" % (arm, rec["reach"]), flush=True)
        cmp_log = out / "compare.log"
        with cmp_log.open("x") as stream:
            rc = subprocess.run([sys.executable, "tools/w4_decomp_quality.py", "compare", algo,
                                 str(out / "A.npz"), str(out / "B.npz")], stdout=stream, stderr=subprocess.STDOUT).returncode
        text = cmp_log.read_text()
        print("\n".join(line for line in text.splitlines() if line.startswith("W4Q-")))
        if rc != 0 or "status=PASS" not in text:
            print("W4-PAIR status=FAIL source=" + args.source + " log=" + str(cmp_log))
            sys.exit(1)
        with receipt_path.open("x") as stream:
            json.dump(receipt_want, stream, sort_keys=True)
        print("W4-PAIR status=PASS algo=" + algo + " source=" + args.source + " receipt=" + str(receipt_path))
    finally:
        temporary = installed.with_suffix(".so.restore")
        shutil.copy2(out / "original.so", temporary)
        os.replace(temporary, installed)


if __name__ == "__main__":
    main()
