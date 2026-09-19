#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PUSH THE BULK TO R2; THE EVIDENCE STAYS IN GIT.

    python3 tools/evidence_r2.py plan              # what would move, and what would not
    python3 tools/evidence_r2.py push <dir>...     # upload, verify, then print the rm
    python3 tools/evidence_r2.py ls [prefix]

MEASURED 2026-09-19, WHICH IS WHY THIS MOVES THE BULK AND NOT THE EVIDENCE:

    bench/results                               9.2 GB
      the identity evidence (column JSONs)       69 MB   1673 files
      bench/results/e1g    11521 .py + 11488 .pyc + 6410 logs   3.2 GB
      bench/results/wheels           895 .so, 1182 logs         1.7 GB
      bench/results/e1              5316 .card, 4013 logs       1.4 GB
      bench/results/resume     9814 .f32 + 3530 .i32 arrays     1.2 GB
      bench/results/releases          787 logs, 718 .pyc        988 MB

So 99% of the weight is copied virtualenvs, build logs, wheels and raw float
dumps, and 0% of it is the claim. `no-oversized-blobs-evidence-outside-repo`
was written for exactly this.

THE COLUMN JSONs DO NOT MOVE, and that is a decision, not an oversight.
`docs/VERIFY_EXTERNALLY.md` recipes 1 to 3 all end in a comparison against a
record THIS REPOSITORY SHIPS. Putting those records behind a bucket nobody
outside can read would make the claim less checkable, not more -- an outside
reader would have to trust a fetch instead of a file. 69 MB of hashes is
exactly what git is for: small, diffable, and signed by the same history as
the code that produced it.

WHAT R2 IS FOR HERE. Artifacts a rerun can regenerate and no reader needs to
audit: wheels, .so files, logs, raw f32/i32 dumps, copied environments. They
are reproducible outputs, not testimony.

THE RULE THIS ENCODES: if deleting it would weaken a claim in
docs/VERIFICATION_MATRIX.md or a recipe in docs/VERIFY_EXTERNALLY.md, it
stays in git. Otherwise it belongs in the bucket.

VERIFY BEFORE YOU DELETE. `push` re-reads every object back from R2 and
compares the sha256 before it prints a single `rm`, and it prints the `rm`
rather than running it. A pull that moved nothing followed by a delete has
already destroyed a run's results in this tree once (verify-the-pull-before-
the-reap); this is the same failure with the bytes going the other way.
"""
import argparse
import hashlib
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
PREFIX = "evidence/v1"

#: Kept in git whatever their size: these ARE the claim.
KEEP = ("identity_break", "verify_reference")
#: Regenerable outputs. Large, and no recipe reads them.
BULK_SUFFIX = (".so", ".pyc", ".log", ".f32", ".i32", ".whl", ".card", ".a", ".dylib")


def _creds():
    import bincache
    creds = bincache.creds_from_env(os.environ)
    if not all(creds.values()):
        rc = pathlib.Path.home() / ".mojolearn_r2"
        if rc.exists():
            env = dict(os.environ)
            for line in rc.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip().strip("'\"")
            creds = bincache.creds_from_env(env)
    missing = [k for k, v in creds.items() if not v]
    if missing:
        raise SystemExit(f"R2 credentials missing: {missing} (see ~/.mojolearn_r2)")
    return creds


def scan(base):
    keep = bulk = 0
    keep_n = bulk_n = 0
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        size = p.stat().st_size
        protected = any(k in p.parts for k in KEEP)
        if protected or p.suffix not in BULK_SUFFIX:
            keep += size; keep_n += 1
        else:
            bulk += size; bulk_n += 1
    return keep, keep_n, bulk, bulk_n


def cmd_plan(_):
    base = ROOT / "bench" / "results"
    keep, keep_n, bulk, bulk_n = scan(base)
    print(f"{'stays in git (the claim)':38s} {keep/1e9:6.2f} GB  {keep_n:6d} files")
    print(f"{'would move to R2 (regenerable)':38s} {bulk/1e9:6.2f} GB  {bulk_n:6d} files")
    print()
    print("per directory, largest first:")
    rows = []
    for d in sorted(base.iterdir()):
        if d.is_dir():
            k, kn, b, bn = scan(d)
            rows.append((b, k, d.name, bn, kn))
    for b, k, name, bn, kn in sorted(rows, reverse=True)[:12]:
        print(f"  {name:28s} move {b/1e9:5.2f} GB ({bn:5d})   keep {k/1e6:7.1f} MB ({kn:5d})")
    print("\nnothing was uploaded or deleted; `push <dir>` does that, and verifies first")
    return 0


def cmd_push(args):
    import bincache
    creds = _creds()
    moved = ok = 0
    removable = []
    for d in args.dirs:
        base = pathlib.Path(d)
        for p in sorted(base.rglob("*")):
            if not p.is_file() or any(k in p.parts for k in KEEP):
                continue
            if p.suffix not in BULK_SUFFIX:
                continue
            rel = p.relative_to(ROOT) if ROOT in p.parents else p
            key = f"{PREFIX}/{rel}"
            body = p.read_bytes()
            want = hashlib.sha256(body).hexdigest()
            st, _ = bincache.r2_request(creds, "PUT", key, data=body)
            moved += 1
            if st not in (200, 201):
                print(f"  FAILED {st} {key}")
                continue
            # READ IT BACK. An upload that reported 200 and stored nothing is
            # the same defect as a pull that moved zero files.
            st2, got = bincache.r2_request(creds, "GET", key)
            if st2 == 200 and hashlib.sha256(got).hexdigest() == want:
                ok += 1
                removable.append(p)
            else:
                print(f"  READBACK MISMATCH {key}")
    print(f"\nuploaded {moved}, verified by readback {ok}")
    if ok != moved:
        print("REFUSING to print any rm: some objects did not verify")
        return 1
    out = ROOT / "bench" / "results" / ".evidence_r2_removable.txt"
    out.write_text("".join(f"{p}\n" for p in removable))
    print(f"every object verified. The list of files now safely removable is at\n  {out}")
    print("Review it, then delete them yourself; this tool does not run rm.")
    return 0


def cmd_ls(args):
    import bincache
    creds = _creds()
    for key, when, size in bincache.r2_list(creds, args.prefix or PREFIX):
        print(f"  {size:12d}  {when}  {key}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("push"); p.add_argument("dirs", nargs="+")
    l = sub.add_parser("ls"); l.add_argument("prefix", nargs="?")
    args = ap.parse_args(argv)
    return dict(plan=cmd_plan, push=cmd_push, ls=cmd_ls)[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
