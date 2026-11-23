#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""Every sabotage patch applies to the commit, exactly (lane patch-recut, 2026-11-23).

WHY. A sabotage arm proves its verify lane can fail by applying its patch, building and seeing the lane move. On
2026-11-23 72 of the tree's 503 sabotage patches no longer applied and 17 more applied only with fuzz: those arms
proved nothing, and nothing said so before a build. This check runs before a build (`tools/prebuild_check.py --only
audits`), so a merge that moves the code a patch faults is caught at once and the patch is re-cut in the same wave.

WHAT IT CHECKS. Every `*/sabotage/*.patch` (and deeper `*/sabotage/**/*.patch`) of the commit, with
`git apply --cached --check` against the commit's own tree, read into a private index (no worktree, no checkout):
the patch must apply with no fuzz (git apply never fuzzes; an offset is allowed, as `git apply` would apply it).
And every such patch must be NAMED by something that runs it (lane patch-recut follow-up, 2026-11-23: 93 patches
applied but nothing ran them, so they proved nothing): the patch column of an arm in
`tools/identity_lanes/coverage_sabotage.tsv` (run by tools/coverage_sabotage.py) or a non-comment line of a
`tools/identity_lanes/*.checks` file (a seam check driver and its patch, run by tools/algos_lane_check.py). A patch
named by neither prints `SABPATCH ORPHAN <patch>` and fails the check.
Excluded: the PERMANENT RECORD patches under `experiments/removed/` (they restore deleted experiments and are
expected not to apply; AGENTS.md "Deleting an experiment"); they are counted and printed as excluded.

An arm whose faulted operation was removed from the product is retired by deleting its patch and its table row with a
dated one-line reason in that table; nothing here exempts a patch that is still in the tree.

PARALLEL PLAN. One unit per patch (one `git apply` process reading its own copy of the patch); the units are
independent and run on a thread pool sized to the allocated cores (`--jobs` overrides, e.g. `--jobs 2` on a loaded
laptop). The git processes share the one read-only private index. Seconds over the whole tree.

    python3 tools/sabotage_patches_apply_check.py [--commit SHA] [--root DIR] [--jobs N] [--verbose]

Prints `SABPATCH FAIL <patch>: <git's first line>` per patch that does not apply, `SABPATCH ORPHAN <patch>` per
patch nothing names, and one summary line `SABPATCH PASS|FAIL <applied> of <checked> apply, <orphans> orphan(s) at
<sha>; <excluded> permanent-record excluded`. Exit 0 when every checked patch applies and is named, 1 otherwise, 2 on a
usage or git error.
"""
import argparse
import fnmatch
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = ("*/sabotage/*.patch", "*/sabotage/**/*.patch")
PERMANENT_RECORD = "experiments/removed/"
ARM_TABLE = "tools/identity_lanes/coverage_sabotage.tsv"
CHECKS_GLOB = "tools/identity_lanes/*.checks"


def allocated_cores():
    """The cores this process may use (affinity or cgroup allocation where the OS gives it), never fewer than 1."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def git(root, *args, env=None, input_bytes=None):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, env=env, input=input_bytes)


def sabotage_patches(root, commit):
    """(checked, excluded): the commit's sabotage patch paths, sorted; the permanent-record ones apart."""
    p = git(root, "ls-tree", "-r", "--name-only", commit)
    if p.returncode != 0:
        raise SystemExit(f"SABPATCH ERROR git ls-tree {commit}: {p.stderr.decode(errors='replace').strip()[-300:]}")
    paths = [line for line in p.stdout.decode().splitlines()
             if line.endswith(".patch") and "/sabotage/" in f"/{line}"
             and any(fnmatch.fnmatch(line, pat) for pat in PATTERNS)]
    checked = sorted(x for x in paths if not x.startswith(PERMANENT_RECORD))
    excluded = sorted(x for x in paths if x.startswith(PERMANENT_RECORD))
    return checked, excluded


def named_patches(root, commit, paths):
    """The patch paths something runs: the arm table's patch column and every `.patch` token of a non-comment line of
    a `tools/identity_lanes/*.checks` file, read from the commit."""
    named = set()

    def text(path):
        r = git(root, "show", f"{commit}:{path}")
        return r.stdout.decode(errors="replace") if r.returncode == 0 else ""

    for line in text(ARM_TABLE).splitlines():
        cols = line.split("\t")
        if line.strip() and not line.lstrip().startswith("#") and len(cols) > 1:
            named.add(cols[1].strip())
    for path in paths:
        if fnmatch.fnmatch(path, CHECKS_GLOB):
            for line in text(path).splitlines():
                if line.strip() and not line.lstrip().startswith("#"):
                    named.update(tok for tok in line.split() if tok.endswith(".patch"))
    return named


def check_one(root, commit, env, patch):
    """None when PATCH applies to the commit's tree, else git's first line of complaint."""
    blob = git(root, "show", f"{commit}:{patch}")
    if blob.returncode != 0:
        return f"cannot read the patch: {blob.stderr.decode(errors='replace').strip()[:200]}"
    r = git(root, "apply", "--cached", "--check", "-", env=env, input_bytes=blob.stdout)
    if r.returncode == 0:
        return None
    lines = (r.stderr or r.stdout).decode(errors="replace").strip().splitlines()
    return lines[0] if lines else f"git apply rc={r.returncode}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--root", default=str(ROOT), help="the git repository (default: this file's repo)")
    ap.add_argument("--jobs", type=int, default=0, help="worker count (default: the allocated cores)")
    ap.add_argument("--verbose", action="store_true", help="also print every patch that applies")
    a = ap.parse_args(argv)

    rp = git(a.root, "rev-parse", "--verify", f"{a.commit}^{{commit}}")
    if rp.returncode != 0:
        print(f"SABPATCH ERROR unknown commit {a.commit}")
        return 2
    sha = rp.stdout.decode().strip()
    checked, excluded = sabotage_patches(a.root, sha)
    tree = git(a.root, "ls-tree", "-r", "--name-only", sha, "--", "tools/identity_lanes").stdout.decode().splitlines()
    named = named_patches(a.root, sha, tree)
    orphans = [p for p in checked if p not in named]

    with tempfile.TemporaryDirectory(prefix="sabpatch-") as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        rt = git(a.root, "read-tree", sha, env=env)
        if rt.returncode != 0:
            print(f"SABPATCH ERROR git read-tree {sha[:12]}: {rt.stderr.decode(errors='replace').strip()[-300:]}")
            return 2
        jobs = a.jobs if a.jobs > 0 else allocated_cores()
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(lambda p: (p, check_one(a.root, sha, env, p)), checked))

    bad = [(p, why) for p, why in results if why is not None]
    for p, why in bad:
        print(f"SABPATCH FAIL {p}: {why}")
    for p in orphans:
        print(f"SABPATCH ORPHAN {p}: named by no arm of {ARM_TABLE} and no {CHECKS_GLOB} line")
    if a.verbose:
        for p, why in results:
            if why is None:
                print(f"SABPATCH OK {p}")
        for p in excluded:
            print(f"SABPATCH EXCLUDED {p} (permanent record)")
    failed = bool(bad or orphans)
    print(f"SABPATCH {'FAIL' if failed else 'PASS'} {len(checked) - len(bad)} of {len(checked)} apply, "
          f"{len(orphans)} orphan(s) at {sha[:12]}; {len(excluded)} permanent-record excluded")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
