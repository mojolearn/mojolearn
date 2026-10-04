#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Manager-only quality pair / quality-gated timing for lane apple-fast-w2-kfeat,
on verified A/B arms (tools/target_scratch_pair.py's pattern, any of the
lane's three bindings).

 quality SOURCE QUALITY_TAG BINDING DEFINES
 timing  SOURCE QUALITY_TAG TIMING_TAG BINDING DEFINES ALGO DATASET [FAMILY]

BINDING: kernel_methods | x_neighbors | x_decomp. DEFINES: the B arm's
define names, comma-separated (A = main = none). SOURCE is the exact full
SHA. Arms: ~/mq/verified-arms/SOURCE/BINDING/{A.so,B.so,manifest.json}.
The quality action installs each arm, runs `tools/kfeat_quality.py dump`,
checks the arm's reach (B has the candidate compiled in, A does not), runs
`compare`, writes PASS.json and restores the installed binding. The timing
action refuses without that receipt, then runs tools/afc_ab_def.sh through
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

FIXTURE = "kfeat-v1"
REACH = {  # binding -> (reach key in KFEAT-CAPTURE, B's expected value from the defines)
    "kernel_methods": ("rbf_resident", lambda ds: "MOJOLEARN_KM_FAST_RBF_RESIDENT" in ds),
    "x_neighbors": ("kfeat_flags", lambda ds: (1 if "MOJOLEARN_XN_FAST_ACHI2_DEVSCAN" in ds else 0)
                    | (2 if "MOJOLEARN_XN_FAST_SCHI2_MOJO_MT" in ds else 0)),
    "x_decomp": ("grp_cls2", None),
}
ALLOWED = {
    "kernel_methods": {"MOJOLEARN_KM_FAST_RBF_RESIDENT"},
    "x_neighbors": {"MOJOLEARN_XN_FAST_ACHI2_DEVSCAN", "MOJOLEARN_XN_FAST_SCHI2_MOJO_MT"},
    "x_decomp": {"MOJOLEARN_XD_FAST_SRP_STRAT"},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_logged(command, path):
    with path.open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print("\n".join(path.read_text(errors="replace").splitlines()[-12:]))
        raise RuntimeError("Quality command failed: " + str(path))


def reach_ok(binding, arm, value, defines):
    key, want = REACH[binding]
    if binding == "x_decomp":
        on = bool(int(value) & 16)
        return on == (arm == "B")
    expected = want(defines) if arm == "B" else want([])
    return value == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("quality", "timing"))
    parser.add_argument("source")
    parser.add_argument("quality_tag")
    parser.add_argument("rest", nargs="+")
    args = parser.parse_args()
    if args.action == "quality":
        assert len(args.rest) == 2, "quality SOURCE QUALITY_TAG BINDING DEFINES"
        timing_tag, (binding, defs) = None, args.rest
    else:
        assert len(args.rest) in (5, 6), "timing SOURCE QUALITY_TAG TIMING_TAG BINDING DEFINES ALGO DATASET [FAMILY]"
        timing_tag, binding, defs, algo, dataset = args.rest[:5]
        family = args.rest[5] if len(args.rest) == 6 else "algos"
        assert dataset in ("taxi", "istella") and family in ("algos", "classical2")
        assert re.fullmatch("[a-z0-9-]+", algo)
    assert re.fullmatch("[0-9a-f]{40}", args.source)
    for tag in (args.quality_tag, timing_tag):
        assert tag is None or re.fullmatch("[A-Za-z0-9_.-]+", tag)
    assert binding in ALLOWED
    defines = sorted(defs.split(","))
    assert defines and set(defines) <= ALLOWED[binding], defines
    defines_b = " ".join("-D " + d for d in defines)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.source
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "kernel_methods/", "x_neighbors/", "x_decomp/",
                    "bindings/", "python/", "tools/kfeat_quality.py", "tools/kfeat_pair.py"], check=True)
    home = Path.home()
    arms = home / "mq/verified-arms" / args.source / binding
    manifest = json.loads((arms / "manifest.json").read_text())
    assert manifest["source_sha"] == args.source and manifest["binding"] == binding
    assert manifest["numeric_mode"] == "fast"
    assert manifest["defines_A"] == ""
    assert sorted(manifest["defines_B"].replace("-D ", "").split()) == defines, manifest["defines_B"]
    hashes = {arm: digest(arms / (arm + ".so")) for arm in ("A", "B")}
    assert hashes == manifest["hashes"]
    out = home / "mq/out" / (args.quality_tag + "-quality")
    receipt_path = out / "PASS.json"
    receipt_want = dict(source_sha=args.source, hashes=hashes, status="PASS", fixture=FIXTURE,
                        binding=binding, defines=defines)
    if args.action == "timing":
        assert json.loads(receipt_path.read_text()) == receipt_want
        os.environ["AFC_FAMILY"] = family
        os.execv(sys.executable, [sys.executable, str(home / "mq/verified_arms.py"),
                 args.source, binding, defines[0], timing_tag,
                 "bash", "tools/afc_ab_def.sh", timing_tag, binding, algo, dataset, "1", "1", "", defines_b])
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
            run_logged([sys.executable, "tools/kfeat_quality.py", "dump", binding, str(out / (arm + ".npz"))], log)
            records = [json.loads(line.split(" ", 1)[1]) for line in log.read_text().splitlines()
                       if line.startswith("KFEAT-CAPTURE ")]
            assert len(records) == 1
            rec = records[0]
            assert rec["binding"] == binding and rec["binding_sha256"] == hashes[arm], rec
            key = REACH[binding][0]
            assert reach_ok(binding, arm, rec["reach"][key], defines), (arm, rec["reach"])
            print("KFEAT-PAIR arm=%s capture=PASS reach=%s" % (arm, rec["reach"]), flush=True)
        cmp_log = out / "compare.log"
        with cmp_log.open("x") as stream:
            rc = subprocess.run([sys.executable, "tools/kfeat_quality.py", "compare", binding,
                                 str(out / "A.npz"), str(out / "B.npz")], stdout=stream, stderr=subprocess.STDOUT).returncode
        text = cmp_log.read_text()
        print("\n".join(line for line in text.splitlines() if line.startswith("KFEAT-")))
        if rc != 0 or "status=PASS" not in text:
            print("KFEAT-PAIR status=FAIL source=" + args.source + " log=" + str(cmp_log))
            sys.exit(1)
        with receipt_path.open("x") as stream:
            json.dump(receipt_want, stream, sort_keys=True)
        print("KFEAT-PAIR status=PASS source=" + args.source + " receipt=" + str(receipt_path))
    finally:
        temporary = installed.with_suffix(".so.restore")
        shutil.copy2(out / "original.so", temporary)
        os.replace(temporary, installed)


if __name__ == "__main__":
    main()
