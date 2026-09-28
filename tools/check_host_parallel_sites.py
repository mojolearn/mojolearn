#!/usr/bin/env python3
"""Every host thread split goes through core/host_parallel.mojo (DEVIATION 5900).

    python3 tools/check_host_parallel_sites.py [repo-root]

Mojo's runtime worker threads run with FTZ+DAZ while the calling thread runs
IEEE, so a raw `sync_parallelize` makes a host loop's bits depend on the task
count. `core/host_parallel.mojo` is the one entry (`host_parallelize`, the
caller's environment; `host_parallelize_pool_env`, the worker's, for the GBDT
fit only). This check refuses, by file and line, any tracked `.mojo` outside
that module (and its seam check) that calls `sync_parallelize` or `parallelize`
directly, any `host_parallelize_pool_env` call outside the GBDT fit files, and
any second environment module (`core/host_fp_env.mojo`).

Recorded results under bench/ are history and are not read.
"""
import re
import subprocess
import sys
from pathlib import Path

ENTRY = "core/host_parallel.mojo"
EXEMPT = {ENTRY, "core/host_parallel_check.mojo"}
POOL_ENV_FILES = {
    "gbdt/train.mojo",
    "gbdt/resident_model.mojo",
    "gbdt/host/gbdt_oracle.mojo",
}
RAW = re.compile(r"(?<![\w.])(?:sync_parallelize|parallelize)\s*[\[(]")
POOL = re.compile(r"(?<![\w.])host_parallelize_pool_env\s*[\[(]")
SECOND_MODULES = ("core/host_fp_env.mojo",)


def code_part(line):
    """The line up to a `#` comment (strings holding `#` are rare enough in
    these files that a false cut only hides text, never adds a finding)."""
    return line.split("#", 1)[0]


def scan(files):
    """files: {path: text}. Returns a list of 'path:line: reason' strings."""
    bad = []
    for path, text in sorted(files.items()):
        if path in SECOND_MODULES:
            bad.append(f"{path}:1: a second host FP-environment module; "
                       f"use {ENTRY}")
            continue
        if path in EXEMPT:
            continue
        in_doc = False
        for n, line in enumerate(text.splitlines(), 1):
            quotes = line.count('"""')
            if in_doc:
                if quotes % 2 == 1:
                    in_doc = False
                continue
            if quotes % 2 == 1:
                in_doc = True
                continue
            code = code_part(line)
            if quotes:
                code = re.sub(r'""".*?"""', "", code)
            if RAW.search(code):
                bad.append(f"{path}:{n}: raw thread split; use "
                           f"host_parallelize from {ENTRY}")
            if POOL.search(code) and path not in POOL_ENV_FILES:
                bad.append(f"{path}:{n}: host_parallelize_pool_env outside "
                           f"the GBDT fit files; use host_parallelize")
    return bad


def tracked_mojo(root):
    out = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--cached", "--others",
         "--exclude-standard", "*.mojo"], text=True)
    files = {}
    for p in out.split():
        if p.startswith("bench/"):
            continue
        f = Path(root) / p
        if f.is_file():
            files[p] = f.read_text(errors="replace")
    return files


def main(argv):
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1]
    files = tracked_mojo(root)
    if ENTRY not in files:
        print(f"check_host_parallel_sites: FAIL: {ENTRY} is missing")
        return 1
    bad = scan(files)
    if bad:
        print("\n".join(bad))
        print(f"check_host_parallel_sites: FAIL: {len(bad)} site(s) bypass {ENTRY}")
        return 1
    print(f"check_host_parallel_sites: PASS ({len(files)} .mojo files, every "
          f"host thread split goes through {ENTRY})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
