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
    python3 tools/coverage_sabotage.py check --arm NAME --clean C.json --sabotaged S.json --restored R.json

`check` reads three identity_break JSON records of the arm's lanes, from
clean, sabotaged and restored builds of the same commit and device, and
prints one line per lane and fixture, then `ARM <name> CAUGHT|MISSED`. It
requires complete matching fixtures, stable clean results, and restoration.
Every lane must move on at least one fixture. A refusal counts only with an
explicit --refusals JSON review: cell -> {error: exact full error, reason:
why this is caused by the injected fault}. Infrastructure failures never count.
This checks recorded metadata, not source-to-binary build attestation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

from verification_evidence import PARTS, same_cell_context
from verification_matrix import negative_control_moves, stable_digest

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


def _record(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _signature(cell):
    """What a cell answered: its verdict and every hash it carries."""
    sig = {}
    for key in ("hashes", *PARTS[1:]):
        if key in cell:
            verdict = "verdict" if key == "hashes" else key + "_verdict"
            sig[key] = (cell.get(verdict), cell[key])
    return cell.get("verdict"), sig


def _clean_cell(cell, repeats):
    if not stable_digest(cell, "train"):
        return False
    if len(cell["hashes"]) != repeats:
        return False
    for part in PARTS[1:]:
        if part not in cell:
            continue
        if not isinstance(cell[part], list) or len(cell[part]) != repeats:
            return False
        if stable_digest(cell, part):
            continue
        values = cell[part]
        if (cell.get(part + "_verdict") != "N/A" or not isinstance(values, list)
                or not values or not all(isinstance(v, str) and v.startswith("n/a:") for v in values)):
            return False
    return True


def _reviewed_refusal(cell, review):
    error = cell.get("error", "")
    # An unavailable execution environment is not detection of a value fault,
    # even if someone accidentally includes its message in the review file.
    infrastructure = r"ModuleNotFoundError|ImportError|no module named|missing.*(?:binding|binary)|cannot load|library not loaded|undefined symbol|out of memory|timed? out"
    return bool(isinstance(review, dict) and isinstance(error, str) and error
                and review.get("error") == error and str(review.get("reason", "")).strip()
                and not re.search(infrastructure, error, re.I))


def check(arm, clean_path, sab_path, restored_path=None, refusals=None):
    if restored_path is None:
        print("restored clean record required; detection alone is incomplete")
        return 1
    records = [_record(p) for p in (clean_path, sab_path, restored_path)]
    baseline = records[0]
    for label, record in zip(("clean", "sabotaged", "restored"), records):
        if (not isinstance(record, dict) or record.get("complete") is not True
                or not isinstance(record.get("cells"), dict)
                or not isinstance(record.get("fixtures"), dict) or not record["fixtures"]
                or not record.get("vendor")
                or not isinstance(record.get("repeats"), int) or record["repeats"] < 1
                or not re.fullmatch(r"[0-9a-f]{40}", str(record.get("commit", "")))
                or record.get("mode") != "identical"):
            print(f"{label}: incomplete record or missing source/device/input metadata")
            return 1
        for field in ("vendor", "platform", "repeats", "fixtures", "heldout",
                      "parts_collected", "parts_omitted", "batch_revisions"):
            if record.get(field) != baseline.get(field):
                print(f"{label}: {field} differs from clean")
                return 1
    clean, sab, restored = (r["cells"] for r in records)
    ok = True
    for lane in arm["lanes"]:
        keys = [f"{lane}/{fixture}" for fixture in baseline["fixtures"]]
        caught = 0
        for key in keys:
            if any(key not in cells for cells in (clean, sab, restored)):
                print(f"{key}: missing clean, sabotaged or restored cell")
                ok = False
                continue
            if not all(isinstance(cells[key], dict) for cells in (clean, sab, restored)):
                print(f"{key}: malformed cell")
                ok = False
                continue
            if (not _clean_cell(clean[key], baseline["repeats"])
                    or not _clean_cell(restored[key], baseline["repeats"])):
                print(f"{key}: clean/restored result is not a valid stable baseline")
                ok = False
                continue
            parts = [p for p in PARTS if ("hashes" if p == "train" else p) in clean[key]]
            if not all(same_cell_context(baseline, r, key, p) for r in records[1:] for p in parts):
                print(f"{key}: source, inputs, lane revision or protocol differ")
                ok = False
                continue
            if _signature(restored[key]) != _signature(clean[key]):
                print(f"{key}: restored output differs from clean")
                ok = False
                continue
            if sab[key].get("verdict") == "REFUSED":
                if not _reviewed_refusal(sab[key], (refusals or {}).get(key)):
                    print(f"{key}: refusal requires fault-specific review; not admitted")
                    ok = False
                    continue
                print(f"{key}: REFUSED (reviewed)")
                caught += 1
            else:
                if any(("hashes" if p == "train" else p) not in sab[key] for p in parts):
                    print(f"{key}: sabotaged result is missing collected parts")
                    ok = False
                    continue
                moved = [p for p in parts if negative_control_moves(sab[key], clean[key], p)]
                invalid = []
                for p in parts:
                    field = "hashes" if p == "train" else p
                    verdict = "verdict" if p == "train" else p + "_verdict"
                    values = sab[key][field]
                    if (not isinstance(values, list) or len(values) != baseline["repeats"]
                            or (stable_digest(clean[key], p) is not None
                                and not stable_digest(sab[key], p) and p not in moved)
                            or (stable_digest(clean[key], p) is None
                                and (values != clean[key][field]
                                     or sab[key].get(verdict) != clean[key].get(verdict)))):
                        invalid.append(p)
                if invalid:
                    print(f"{key}: invalid sabotaged parts: {', '.join(invalid)}")
                    ok = False
                    continue
                print(f"{key}: MOVED ({', '.join(moved)})" if moved else f"{key}: unmoved")
                caught += bool(moved)
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
    print("# 4. restore the clean source and binding set, then rerun the same command into restored.json")
    print(f"python3 tools/coverage_sabotage.py check --arm {arm['arm']} --clean clean.json --sabotaged sab.json --restored restored.json")


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
    c.add_argument("--restored", required=True)
    c.add_argument("--refusals", help="JSON of exact fault-caused refusal errors and review reasons")
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
    return check(table[args.arm], args.clean, args.sabotaged, args.restored,
                 _record(args.refusals) if args.refusals else None)


if __name__ == "__main__":
    sys.exit(main())
