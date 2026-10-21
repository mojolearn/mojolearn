#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The release audit's fault-injection arms (lane verify-coverage-audit, 2026-10-21).

docs/plans/VERIFY_COVERAGE_20261021.md lists the kernel definitions that are
new or changed since the last release and that no existing sabotage arm
reached. Each arm here is a source patch in `<area>/checks/sabotage/` that
injects one VALUE fault into the changed definition (a float output one ulp
off, an integer output permuted or stepped, a byte flipped), so a build with
the patch applied computes a wrong answer through exactly that path. The arm
is CAUGHT when the verifier lanes the manifest names read a different cell
(MOVED, which `verify` reports DIVERGENT against a reference) or REFUSED on
the sabotaged build, and the same lanes read their clean cells on the clean
build. A patch never changes a production build: it is applied only to a
scratch copy of the tree.

THE MANIFEST, tools/identity_lanes/coverage_sabotage.tsv, one arm per line,
tab separated, `#` comments:

    arm    patch    bindings    lanes    why

  arm       a unique name, `coverage-<area>-<what>`
  patch     the patch, relative to the repository root
  bindings  the binding build scripts to rerun with it, comma separated, as
            script stems: `build_core_host` is bindings/build_core_host.sh,
            `build_estimators` is bindings/build_estimators.sh; a stem ending in
            `_host` builds a host binding)
  lanes     the identity_break lanes that must read MOVED or REFUSED,
            comma separated
  why       one sentence: what the fault is and why those lanes reach it

THE RUN (orchestrator; a lane never compiles):

    python3 tools/coverage_sabotage.py list
    python3 tools/coverage_sabotage.py apply-check       # every patch still applies to this tree
    python3 tools/coverage_sabotage.py plan --arm NAME   # the commands for one arm
    python3 tools/coverage_sabotage.py check --arm NAME --clean C.json --sabotaged S.json

`check` reads two identity_break JSON records of the arm's lanes, one from
a clean build and one from the sabotaged build of the same commit, and
prints one line per lane and fixture, then `ARM <name> CAUGHT|MISSED`. It
exits 1 unless every lane moved or refused on at least one fixture and no
lane's clean record refused.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "tools", "identity_lanes", "coverage_sabotage.tsv")
FIELDS = ("arm", "patch", "bindings", "lanes", "why")


def arms(path=MANIFEST):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != len(FIELDS):
                raise SystemExit(f"{path}:{n}: {len(cols)} columns, expected {len(FIELDS)} ({', '.join(FIELDS)})")
            row = dict(zip(FIELDS, cols))
            row["bindings"] = [b for b in row["bindings"].split(",") if b]
            row["lanes"] = [b for b in row["lanes"].split(",") if b]
            if row["arm"] in out:
                raise SystemExit(f"{path}:{n}: arm {row['arm']} twice")
            if not row["lanes"] or not row["bindings"]:
                raise SystemExit(f"{path}:{n}: an arm names its bindings and at least one lane")
            out[row["arm"]] = row
    return out


def _cells(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("cells", {})


def _signature(cell):
    """What a cell answered: its verdict and every hash it carries."""
    sig = {}
    for key in ("hashes", "infer", "model", "batch"):
        if key in cell:
            sig[key] = json.dumps(cell[key], sort_keys=True)
    return cell.get("verdict"), sig


def check(arm, clean_path, sab_path):
    clean, sab = _cells(clean_path), _cells(sab_path)
    ok = True
    for lane in arm["lanes"]:
        keys = sorted(k for k in clean if k.split("/")[0] == lane)
        if not keys:
            print(f"{lane}: no clean cell in {clean_path}")
            ok = False
            continue
        caught = 0
        for key in keys:
            cv, cs = _signature(clean[key])
            if cv == "REFUSED":
                print(f"{key}: the CLEAN build refused; the arm proves nothing until it runs")
                ok = False
                continue
            if key not in sab:
                print(f"{key}: no sabotaged cell")
                continue
            sv, ss = _signature(sab[key])
            if sv == "REFUSED":
                print(f"{key}: REFUSED")
                caught += 1
            elif ss != cs:
                print(f"{key}: MOVED ({', '.join(k for k in cs if ss.get(k) != cs[k])})")
                caught += 1
            else:
                print(f"{key}: unmoved")
        if not caught:
            ok = False
    print(f"ARM {arm['arm']} {'CAUGHT' if ok else 'MISSED'}")
    return 0 if ok else 1


def plan(arm):
    lanes = ",".join(arm["lanes"])
    print(f"# {arm['arm']}: {arm['why']}")
    print("# 1. a scratch worktree at this commit")
    print("git worktree add --detach ../cov-sab-$ARM HEAD && cd ../cov-sab-$ARM")
    host = [b for b in arm["bindings"] if b.endswith("_host")]
    device = [b for b in arm["bindings"] if not b.endswith("_host")]
    print("# 2. the clean column: the lanes on this commit's build (a complete host set in $CLEAN"
          + (", the device bindings built in this tree, on a GPU box" if device else "") + ")")
    print(f"MOJOLEARN_HOST_DIR=$CLEAN python3 tools/identity_break.py --lanes {lanes} --repeats 1 --json clean.json")
    print("# 3. the sabotaged column: the patch applied, the arm's bindings rebuilt over a copy of the set")
    print(f"git apply {arm['patch']}")
    if host:
        print("cp -R $CLEAN $SAB")
        for b in host:
            print(f"MOJOLEARN_HOST_OUTDIR=$SAB MOJOLEARN_COMPILE_JOBS=1 sh bindings/{b}.sh")
    for b in device:
        print(f"MOJOLEARN_COMPILE_JOBS=1 sh bindings/{b}.sh   # device binding, into this scratch tree")
    print(f"MOJOLEARN_HOST_DIR={'$SAB' if host else '$CLEAN'} python3 tools/identity_break.py --lanes {lanes} "
          "--repeats 1 --json sab.json")
    print(f"python3 tools/coverage_sabotage.py check --arm {arm['arm']} --clean clean.json --sabotaged sab.json")


def apply_check(table):
    bad = 0
    for name, arm in table.items():
        r = subprocess.run(["git", "-C", ROOT, "apply", "--check", arm["patch"]], capture_output=True, text=True)
        if r.returncode:
            bad += 1
            print(f"{name}: {arm['patch']} does not apply: {r.stderr.strip().splitlines()[:1]}")
    print(f"apply-check: {len(table) - bad} of {len(table)} patches apply")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("apply-check")
    p = sub.add_parser("plan")
    p.add_argument("--arm", required=True)
    c = sub.add_parser("check")
    c.add_argument("--arm", required=True)
    c.add_argument("--clean", required=True)
    c.add_argument("--sabotaged", required=True)
    args = ap.parse_args(argv)
    table = arms()
    if args.cmd == "list":
        for name, arm in table.items():
            print(f"{name}\t{arm['patch']}\t{','.join(arm['bindings'])}\t{','.join(arm['lanes'])}")
        return 0
    if args.cmd == "apply-check":
        return apply_check(table)
    if args.arm not in table:
        raise SystemExit(f"no arm {args.arm}; `list` names them")
    if args.cmd == "plan":
        plan(table[args.arm])
        return 0
    return check(table[args.arm], args.clean, args.sabotaged)


if __name__ == "__main__":
    sys.exit(main())
